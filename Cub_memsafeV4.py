#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 09:24:00 2025

@author: lukesnow
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cubature vs MC vs QMC vs MLMC convergence with *moment-correct* recombination.

Changes vs earlier version:
 - Adds Chebyshev Vandermonde helper `chebV`.
 - Recombination now enforces *exact* moment constraints (up to tol) with
   nonnegative weights using a small LP (scipy.linprog, if available).
   If SciPy is unavailable, falls back to a deterministic active-set
   reduction with equality projection and positivity checks.
 - Keeps QMC-on-uniform (Brownian bridge) and additionally QMC-on-cubature-grids.
 - Legend placed at top-right.

Requires: numpy, matplotlib, torch, torchsde
Optional:  scipy (for exact LP-based recombination)
"""

import math, gc, random
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
import torchsde
from torch.quasirandom import SobolEngine
from numpy.linalg import svd

# Optional SciPy for robust nonnegative LP
try:
    from scipy.optimize import linprog
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

# ===============================================================
# Global dtype / device
# ===============================================================
torch.set_default_dtype(torch.float64)
DTYPE_NP = np.float64
DEVICE = torch.device('cpu')  # set to 'cuda' if available

# ===============================================================
# Helpers
# ===============================================================
def nullspace(A, atol=1e-13, rtol=0):
    A = np.atleast_2d(A)
    u, s, vh = svd(A)
    tol = max(atol, rtol * s[0]) if len(s) else atol
    nnz = (s >= tol).sum() if len(s) else 0
    ns = vh[nnz:].conj().T if nnz < vh.shape[0] else np.zeros((A.shape[1], 0))
    return ns

def normal_icdf(u: torch.Tensor) -> torch.Tensor:
    return math.sqrt(2.0) * torch.erfinv(2.0 * u - 1.0)

# ===============================================================
# Chebyshev Vandermonde helper for recombination
# ===============================================================
def chebV(z, rdeg):
    """
    Build Chebyshev Vandermonde matrix of degree rdeg at points z in [-1,1].
    Returns V with shape (rdeg+1, m), V[j,i] = T_j(z_i).
    """
    z = np.asarray(z, dtype=np.float64).ravel()
    m = z.shape[0]
    V = np.empty((rdeg + 1, m), dtype=np.float64)
    V[0, :] = 1.0
    if rdeg >= 1:
        V[1, :] = z
    for j in range(2, rdeg + 1):
        V[j, :] = 2.0 * z * V[j - 1, :] - V[j - 2, :]
    return V

# ===============================================================
# Recombination (endpoint-only, 1D) with exact moment constraints
# ===============================================================
def recombine_bin_endpoints_cheb(pts_bin, w_bin, rdeg, tol=1e-12, verbose=False):
    """
    Recombine a bin of atoms (endpoints of driver) by matching Chebyshev
    moments T_0..T_rdeg exactly with nonnegative weights.

    If SciPy is available, solve LP:
        min 1^T w  s.t.  V w = b,  w >= 0
    Otherwise, use a deterministic active-set fallback:
        - iteratively drop atoms along a nullspace direction,
        - project to equality on survivors, enforcing nonnegativity.
    Returns:
        survivors_idx (indices into the original bin),
        weights_new   (weights for survivors, sum equals original mass).
    """
    pts_bin = np.asarray(pts_bin, dtype=np.float64).ravel()
    w_bin   = np.asarray(w_bin,   dtype=np.float64).ravel()
    m = len(pts_bin)
    if m == 0:
        return np.array([], dtype=int), np.array([], dtype=np.float64)
    if m <= rdeg + 1:
        # Already minimal; just return original (ensuring nonnegativity/mass)
        w_pos = np.maximum(w_bin, 0.0)
        mass = w_pos.sum()
        if mass <= 0:
            # fallback: uniform positive split to keep mass
            w_pos = np.full_like(w_bin, w_bin.sum() / m)
        return np.arange(m), w_pos

    a, b = float(pts_bin.min()), float(pts_bin.max())
    if abs(b - a) < 1e-14:
        # All points equal; collapse to a single survivor
        return np.array([0]), np.array([w_bin.sum()], dtype=np.float64)

    # Map to z in [-1,1] and build moments
    z = (2.0 * (pts_bin - a) / (b - a)) - 1.0
    V = chebV(z, rdeg)                 # (r+1, m)
    b_mom = V @ w_bin                  # target moments (r+1,)

    # --- Preferred path: small LP with exact equality + nonnegativity ---
    if HAVE_SCIPY:
        bounds = [(0.0, None)] * m
        c = np.ones(m, dtype=np.float64)  # any feasible nonneg solution that matches moments is fine
        res = linprog(c, A_eq=V, b_eq=b_mom, bounds=bounds, method='highs')
        if res.success and np.all(res.x >= -1e-14):
            w_new = np.maximum(res.x, 0.0)
            # Clean tiny negatives and renormalize mass exactly:
            mass_old = w_bin.sum()
            if w_new.sum() > 0:
                w_new *= (mass_old / w_new.sum())
            # Choose survivors where weight is nonzero
            survivors = np.where(w_new > tol * max(1.0, mass_old))[0]
            if survivors.size == 0:
                # keep the largest weight at least
                survivors = np.array([int(np.argmax(w_new))])
            return survivors, w_new[survivors]

    # --- Fallback: deterministic active-set elimination + equality projection ---
    S = np.arange(m)                   # active set indices
    w_work = w_bin.copy()

    # Drop until at most rdeg+1 survivors using nullspace directions
    while len(S) > (rdeg + 1):
        VS = V[:, S]
        K = nullspace(VS)
        if K.size == 0:
            break
        d = K[:, 0]                    # direction in weight space (len S)
        # step to hit a boundary (one weight to zero) while keeping w>=0
        with np.errstate(divide='ignore', invalid='ignore'):
            frac = np.where(d < -1e-16, w_work[S] / (-d), np.inf)
        j_local = int(np.argmin(frac))
        alpha = float(frac[j_local])
        if not np.isfinite(alpha):
            # degenerate; drop the smallest weight
            j_local = int(np.argmin(w_work[S]))
            S = np.delete(S, j_local)
            continue
        w_work[S] = w_work[S] + alpha * d
        # Remove zeros (or tiny)
        keep_mask = w_work[S] > tol * max(1.0, w_bin.sum())
        S = S[keep_mask]

    # Final equality projection on survivors
    for _ in range(8):  # small number of polishing rounds
        VS = V[:, S]
        # Solve VS wS = b_mom (least-squares, typically square or tall)
        #wS, *_ = np.linalg.lstsq(VS, b_mom, rcond=None)
        wS, *_ = np.linalg.lstsq(VS.T @ VS + 1e-12*np.eye(VS.shape[1]), VS.T @ b_mom, rcond=None)
        # Enforce nonnegativity by pruning negatives
        neg_idx = np.where(wS < -1e-12)[0]
        if neg_idx.size == 0:
            wS = np.maximum(wS, 0.0)
            break
        S = np.delete(S, neg_idx)
        if len(S) == 0:
            # fallback: keep original
            return np.arange(m), w_bin

    # Renormalize mass exactly
    mass_old = w_bin.sum()
    if wS.sum() > 0:
        wS *= (mass_old / wS.sum())

    # Residual check
    resid = np.linalg.norm(V[:, S] @ wS - b_mom, ord=np.inf)
   # if verbose and resid > 1e-6 * (1.0 + np.linalg.norm(b_mom, ord=np.inf)):
    if resid > 1e-6 * (1.0 + np.linalg.norm(b_mom, ord=np.inf)):
        #continue
        print(f"[recombine] large residual {resid:.2e} with |S|={len(S)}")

    return S, wS

# ===============================================================
# Degree-5 base paths (KLV 1D, weights [1/6, 2/3, 1/6])
# ===============================================================
def path5_1(t):
    if t <= 1/3:
        return -math.sqrt(3)*(0.5 * (4 - math.sqrt(22)) * t)
    elif 1/3 < t <= 2/3:
        return -math.sqrt(3)*(1/6 * (4 - math.sqrt(22)) + (-1 + math.sqrt(22)) * (t - 1/3))
    else:
        return -math.sqrt(3)*(1/6 * (2 + math.sqrt(22)) + 0.5 * (4 - math.sqrt(22)) * (t - 2/3))

def path5_2(t):
    if t <= 1/3:
        return  math.sqrt(3)*(0.5 * (4 - math.sqrt(22)) * t)
    elif 1/3 < t <= 2/3:
        return  math.sqrt(3)*(1/6 * (4 - math.sqrt(22)) + (-1 + math.sqrt(22)) * (t - 1/3))
    else:
        return  math.sqrt(3)*(1/6 * (2 + math.sqrt(22)) + 0.5 * (4 - math.sqrt(22)) * (t - 2/3))

def path5_3(_):
    return 0.0

# ---- Degree-3 linear base paths (endpoints ±√dt) ----
def path3_plus(u):   # u in [0,1]
    return +u
def path3_minus(u):
    return -u

# ===============================================================
# Model (Stratonovich SDE)
# ===============================================================
state_size, brownian_size = 1, 1

class TimeVaryingLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight_generator = nn.Linear(1, in_features * out_features)
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x, t):
        w = self.weight_generator(t.unsqueeze(0).unsqueeze(0))
        w = w.view(self.out_features, self.in_features)
        return torch.matmul(x, w.T)

class SDE(nn.Module):
    noise_type = "diagonal"
    sde_type   = "stratonovich"

    def __init__(self):
        super().__init__()
        self.tvl1 = TimeVaryingLinear(state_size, state_size)
        self.tvl2 = TimeVaryingLinear(state_size, state_size)
        self.tvl3 = TimeVaryingLinear(state_size, state_size)
        self.dfc1 = nn.Linear(state_size, state_size)
        self.dfc2 = nn.Linear(state_size, state_size)

    def f(self, t, x):
        m = nn.ELU()
        x = m(self.tvl1(x, t))
        x = m(self.tvl2(x, t))
        x = self.tvl3(x, t)
        return x

    def g(self, t, x):
        m = nn.ELU()
        x = m(self.dfc1(x))
        x = self.dfc2(x)
        return x

# ===============================================================
# Stratonovich midpoint simulator (vectorized) for MC/QMC/MLMC
# ===============================================================
@torch.no_grad()
def simulate_sde_strat_midpoint_batch(f, g, y0_scalar, ts, dW):
    N, S = dW.shape
    T = ts.shape[0]
    assert S == T - 1
    y = torch.full((N, 1), float(y0_scalar), device=ts.device, dtype=ts.dtype)
    Y = torch.empty((N, T, 1), device=ts.device, dtype=ts.dtype)
    Y[:, 0, :] = y
    for i in range(T - 1):
        t0, t1 = ts[i], ts[i+1]
        dt = t1 - t0
        dWi = dW[:, i:i+1]
        f0 = f(t0, y)
        g0 = g(t0, y)
        y_pred = y + 0.5 * f0 * dt + 0.5 * g0 * dWi
        tm = t0 + 0.5 * dt
        fm = f(tm, y_pred)
        gm = g(tm, y_pred)
        y = y + fm * dt + gm * dWi
        Y[:, i+1, :] = y
    return Y

@torch.no_grad()
def time_avg_mse_from_paths(Y, ts):
    fcn_t = torch.sin(2 * math.pi * ts)
    diff2 = (Y.squeeze(-1) - fcn_t) ** 2  # [N,T]
    integ = torch.trapz(diff2, ts, dim=1)
    return integ / (ts[-1] - ts[0])

# ===============================================================
# Controlled ODE solver (RK4) for piecewise-linear drive (Cubature)
# ===============================================================
@torch.no_grad()
def cde_time_avg_mse_batch(v1, v2, W, ts, y0_scalar, target_vals):
    device, dtype = ts.device, ts.dtype
    P, T = W.shape
    y  = torch.full((P, 1), float(y0_scalar), device=device, dtype=dtype)
    err_accum = torch.zeros(P, device=device, dtype=dtype)
    for i in range(T - 1):
        t0, t1 = ts[i], ts[i+1]
        dt = t1 - t0
        wdot = (W[:, i+1] - W[:, i]) / dt  # [P]
        def F(t, y_):
            return (v1(t, y_) + v2(t, y_) * wdot.view(-1, 1))
        k1 = F(t0,             y)
        k2 = F(t0 + 0.5*dt,    y + 0.5*dt*k1)
        k3 = F(t0 + 0.5*dt,    y + 0.5*dt*k2)
        k4 = F(t1,             y + dt*k3)
        y_next = y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        e0 = (target_vals[i]   - y.squeeze(1))**2
        e1 = (target_vals[i+1] - y_next.squeeze(1))**2
        err_accum += 0.5 * (e0 + e1) * dt
        y = y_next
    return err_accum / (ts[-1] - ts[0])

# ===============================================================
# Time partition and local grid (Cubature)
# ===============================================================
def make_time_partition(k, gamma):
    t_d = np.zeros(k + 1, dtype=DTYPE_NP)
    for i in range(k + 1):
        t_d[i] = 1.0 - (1.0 - i / k) ** gamma
    return t_d

def per_step_grid(dt, base_density, total_order, growth=1.0, dtn_min=6):
    dtn = max(dtn_min, int(math.ceil(dt * base_density * (total_order ** growth))))
    return np.linspace(0.0, dt, dtn + 1, endpoint=True, dtype=DTYPE_NP)[1:]

# ===============================================================
# Build child increments by degree (Cubature)
# ===============================================================
def build_child_increments(degree, dt, x_loc):
    if degree == 3:
        root = math.sqrt(dt)
        inc_plus  = np.array([ root * (z / dt) for z in x_loc], dtype=DTYPE_NP)
        inc_minus = np.array([-root * (z / dt) for z in x_loc], dtype=DTYPE_NP)
        return [inc_minus, inc_plus], np.array([0.5, 0.5], dtype=DTYPE_NP)
    elif degree == 5:
        inc0 = np.array([math.sqrt(dt) * path5_1(float(z / dt)) for z in x_loc], dtype=DTYPE_NP)
        inc1 = np.array([path5_3(float(z)) for z in x_loc], dtype=DTYPE_NP)
        inc2 = np.array([math.sqrt(dt) * path5_2(float(z / dt)) for z in x_loc], dtype=DTYPE_NP)
        return [inc0, inc1, inc2], np.array([1/6, 2/3, 1/6], dtype=DTYPE_NP)
    else:
        raise ValueError("degree must be 3 or 5")

def init_first_step_paths_times(t_d, base_density, total_order, growth, degree):
    dt0  = float(t_d[1] - t_d[0])
    xloc = per_step_grid(dt0, base_density, total_order, growth=growth, dtn_min=6)
    incs, w = build_child_increments(degree, dt0, xloc)
    p0 = np.stack(incs, axis=0)
    T = np.concatenate([np.array([0.0], dtype=DTYPE_NP), xloc])
    p0 = np.concatenate([np.zeros((p0.shape[0], 1), dtype=DTYPE_NP), p0], axis=1)
    return p0, w, T

# ===============================================================
# Oracles (MC baselines via torchsde)
# ===============================================================
@torch.no_grad()
def monte_carlo_oracle_loss(sde, batch, t_steps, y0_scalar, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    sde = sde.to(DEVICE).eval()
    for p in sde.parameters(): p.requires_grad_(False)
    ts = torch.linspace(0, 1, t_steps, device=DEVICE, dtype=torch.float64)
    y0 = torch.full((batch, state_size), y0_scalar, device=DEVICE, dtype=torch.float64)
    with torch.inference_mode():
        Y = torchsde.sdeint(sde, y0, ts, method='midpoint')
    fcn = torch.sin(2 * math.pi * ts).unsqueeze(1)
    return ((Y.squeeze(-1) - fcn)**2).mean(dim=0).mean().item()

@torch.no_grad()
def monte_carlo_oracle_on_custom_grid(sde, t_points, y0_scalar, batch, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    sde = sde.to(DEVICE).eval()
    for p in sde.parameters(): p.requires_grad_(False)
    ts = torch.tensor(t_points, device=DEVICE, dtype=torch.float64)
    if not bool(torch.all(ts[1:] > ts[:-1])):
        raise ValueError("Custom oracle grid must be strictly increasing.")
    y0 = torch.full((batch, state_size), y0_scalar, device=DEVICE, dtype=torch.float64)
    with torch.inference_mode():
        Y = torchsde.sdeint(sde, y0, ts, method='midpoint')
    fcn = torch.sin(2 * math.pi * ts).unsqueeze(1)
    err2 = (Y.squeeze(-1) - fcn)**2
    L = torch.trapz(err2.mean(dim=1), ts) / (ts[-1] - ts[0])
    return L.item()

# ===============================================================
# QMC (Sobol) supporting uniform or custom nonuniform grids
# ===============================================================
def brownian_bridge_increments(Z, T_steps):
    N, D = Z.shape
    assert D == T_steps - 1
    dt = 1.0 / (T_steps - 1)
    W = torch.zeros((N, T_steps), dtype=Z.dtype, device=Z.device)
    idx = 0
    W[:, -1] = Z[:, idx] * math.sqrt(1.0); idx += 1
    stack = [(0, T_steps - 1)]
    while stack:
        l, r = stack.pop()
        if r - l <= 1:
            continue
        m = (l + r) // 2
        t_l, t_r, t_m = l * dt, r * dt, ((l + r) // 2) * dt
        mean = ((t_m - t_l) / (t_r - t_l)) * W[:, r] + ((t_r - t_m) / (t_r - t_l)) * W[:, l]
        var  = ((t_m - t_l) * (t_r - t_m)) / (t_r - t_l)
        W[:, m] = mean + math.sqrt(var) * Z[:, idx]; idx += 1
        stack.append((l, m)); stack.append((m, r))
    dW = W[:, 1:] - W[:, :-1]
    return dW

@torch.no_grad()
def qmc_oracle_loss(sde, N, t_steps=None, y0_scalar=0.1, seed=0,
                    use_bb=True, ts_custom: torch.Tensor=None):
    torch.manual_seed(seed); np.random.seed(seed)
    if ts_custom is not None:
        ts = ts_custom.to(device=DEVICE, dtype=torch.float64)
        assert bool(torch.all(ts[1:] > ts[:-1])), "ts_custom must be strictly increasing."
        dims = ts.numel() - 1
        eng = SobolEngine(dimension=dims, scramble=True, seed=seed)
        U = eng.draw(N).to(dtype=torch.float64, device=DEVICE)
        Z = normal_icdf(torch.clamp(U, 1e-12, 1-1e-12))
        dts = ts[1:] - ts[:-1]
        dW  = Z * dts.sqrt().unsqueeze(0)
        Y   = simulate_sde_strat_midpoint_batch(sde.f, sde.g, y0_scalar, ts, dW)
        loss_per_path = time_avg_mse_from_paths(Y, ts)
        return loss_per_path.mean().item()
    assert t_steps is not None and t_steps >= 2
    ts = torch.linspace(0.0, 1.0, t_steps, device=DEVICE, dtype=torch.float64)
    eng = SobolEngine(dimension=t_steps-1, scramble=True, seed=seed)
    U = eng.draw(N).to(dtype=torch.float64, device=DEVICE)
    Z = normal_icdf(torch.clamp(U, 1e-12, 1-1e-12))
    if use_bb:
        dW = brownian_bridge_increments(Z, t_steps)
    else:
        dt = 1.0 / (t_steps - 1)
        dW = math.sqrt(dt) * Z
    Y = simulate_sde_strat_midpoint_batch(sde.f, sde.g, y0_scalar, ts, dW)
    loss_per_path = time_avg_mse_from_paths(Y, ts)
    return loss_per_path.mean().item()

# ===============================================================
# MLMC (coupled midpoint, coarse = sum of two fine increments)
# ===============================================================
@torch.no_grad()
def mlmc_level_pair_estimator(sde, Nl, M0, level, y0_scalar, seed=0):
    torch.manual_seed(seed + level + 7919); np.random.seed(seed + level + 7919)
    assert level >= 0
    Nc = int(M0 * (2 ** level))
    tc = torch.linspace(0.0, 1.0, Nc + 1, device=DEVICE, dtype=torch.float64)
    if level == 0:
        Zc = torch.randn((Nl, Nc), device=DEVICE, dtype=torch.float64)
        dWc = math.sqrt(1.0 / Nc) * Zc
        Yc = simulate_sde_strat_midpoint_batch(sde.f, sde.g, y0_scalar, tc, dWc)
        Pc = time_avg_mse_from_paths(Yc, tc)
        return Pc.mean().item()
    else:
        Nf = 2 * Nc
        tf = torch.linspace(0.0, 1.0, Nf + 1, device=DEVICE, dtype=torch.float64)
        Zf = torch.randn((Nl, Nf), device=DEVICE, dtype=torch.float64)
        dWf = math.sqrt(1.0 / Nf) * Zf
        dWc = dWf.view(Nl, Nc, 2).sum(dim=2)
        Yf = simulate_sde_strat_midpoint_batch(sde.f, sde.g, y0_scalar, tf, dWf)
        Yc = simulate_sde_strat_midpoint_batch(sde.f, sde.g, y0_scalar, tc, dWc)
        Pf = time_avg_mse_from_paths(Yf, tf)
        Pc = time_avg_mse_from_paths(Yc, tc)
        return (Pf - Pc).mean().item()

@torch.no_grad()
def mlmc_estimator_loss(sde, Lmax, N_per_level, y0_scalar, M0=16, seed=0):
    assert len(N_per_level) == Lmax + 1
    parts, used = [], []
    for ell in range(Lmax + 1):
        val = mlmc_level_pair_estimator(sde, int(N_per_level[ell]), M0, ell, y0_scalar, seed=seed)
        parts.append(val); used.append(int(N_per_level[ell]))
    est = parts[0] + sum(parts[1:])
    return est, parts, sum(used), used

# ===============================================================
# Cubature builder + loss (RK4/CDE) — returns grids as well
# ===============================================================
def cubature_loss_vs_order(
    sde,
    disc_order,
    gamma=0.6,
    y0_scalar=0.1,
    DEGREE=5,
    RECOMBINE=False,
    THRES=None,
    BIN_SCALE=2.0,
    RECOMBINE_EVERY=2,
    BASE_DENSITY=300,
    GRID_GROWTH=1.2,
    ORACLE_SAMEGRID_BATCH=20_000,
    seed_for_samegrid=0,
    verbose=False
):
    if THRES is None:
        THRES = 3 if DEGREE == 3 else 6
    rdeg = THRES - 1

    losses, path_counts, widths, works, samegrid = [], [], [], [], []
    grids = []

    for k in disc_order:
        t_d = make_time_partition(k, gamma)
        p, w, T = init_first_step_paths_times(t_d, BASE_DENSITY, k, GRID_GROWTH, DEGREE)
        work = p.shape[0]

        for step in range(1, k):
            dt  = float(t_d[step + 1] - t_d[step])
            x_loc = per_step_grid(dt, BASE_DENSITY, k, growth=GRID_GROWTH, dtn_min=6)
            dtn = len(x_loc)

            rows_old = p.shape[0]
            child_incs, child_w = build_child_increments(DEGREE, dt, x_loc)
            num_children = len(child_incs)
            rows_new = rows_old * num_children

            p_rep = np.repeat(p, num_children, axis=0)
            last_vals = p[:, -1:]
            base_block = np.repeat(last_vals, num_children, 0)

            new_block = np.empty((rows_new, dtn), dtype=DTYPE_NP)
            for r in range(rows_new):
                kind = r % num_children
                new_block[r, :] = base_block[r, 0] + child_incs[kind]

            p = np.concatenate([p_rep, new_block], axis=1)
            T = np.concatenate([T, T[-1] + x_loc], axis=0)
            w = np.repeat(w, num_children) * np.tile(child_w, rows_old)

            if RECOMBINE and (step % RECOMBINE_EVERY == 0):
                pts = p[:, -1]
                #u = max(1e-12, BIN_SCALE * dt)
                P_STAR = 6
                u = max(1e-12, (dt ** (P_STAR / (2.0 * gamma))))
                
                mx, mn = np.max(pts), np.min(pts)
                num_edges = max(2, int(max(1, (mx - mn) / u)))
                prt = np.linspace(mn, mx, num_edges)
                for j in range(len(prt) - 1):
                    find = np.where((pts >= prt[j]) & (pts < prt[j + 1]))[0]
                    if len(find) > THRES:
                        pts_bin = pts[find].astype(np.float64)
                        w_sub   = w[find].astype(np.float64)
                        # exact moment recombination on Chebyshev basis
                        survivors_local, w_new_local = recombine_bin_endpoints_cheb(
                            pts_bin, w_sub, rdeg, tol=1e-12, verbose=verbose
                        )
                        # map back to global indices
                        survivors = find[survivors_local]
                        dropped   = np.setdiff1d(find, survivors, assume_unique=False)
                        # zero old, set new
                        w[find] = 0.0
                        w[survivors] = w_new_local.astype(DTYPE_NP)
                        if dropped.size > 0:
                            p = np.delete(p, dropped, axis=0)
                            w = np.delete(w, dropped, axis=0)
                            pts = p[:, -1]

            work += p.shape[0]
            del p_rep, new_block, last_vals, base_block, child_incs, x_loc
            gc.collect()

        ts    = torch.tensor(T, device=DEVICE, dtype=torch.float64)
        fcn_t = torch.sin(2 * math.pi * ts)
        W = torch.from_numpy(p).to(DEVICE)
        w_t = torch.from_numpy(w).to(DEVICE)
        per_path_mse = cde_time_avg_mse_batch(sde.f, sde.g, W, ts, y0_scalar, fcn_t)
        total_loss = (w_t * per_path_mse).sum().item()
        cub_loss = float(abs(total_loss))

        same_grid_oracle = monte_carlo_oracle_on_custom_grid(
            sde, T, y0_scalar, batch=ORACLE_SAMEGRID_BATCH, seed=seed_for_samegrid
        )

        losses.append(cub_loss)
        path_counts.append(int(p.shape[0]))
        widths.append(len(T))
        works.append(int(work))
        samegrid.append(same_grid_oracle)
        grids.append(T.copy())

        del p, w, ts, fcn_t
        gc.collect()

    return (np.array(losses, dtype=np.float64),
            np.array(path_counts, dtype=int),
            np.array(widths, dtype=int),
            np.array(works, dtype=int),
            np.array(samegrid, dtype=np.float64),
            grids)

# ===============================================================
# Full experiment
# ===============================================================
def run_experiment(
    disc_order,
    gamma,
    y0_scalar,
    DEGREE,
    RECOMBINE,
    THRES,
    BIN_SCALE,
    RECOMBINE_EVERY,
    ORACLE_T,
    ORACLE_BATCH_TRUE,
    MC_N_LIST,
    BASE_DENSITY,
    GRID_GROWTH,
    QMC_N_LIST,
    QMC_SAMEGRID_N=2048,
    QMC_BB=True,
    MLMC_L_LIST=(0,1,2,3,4),
    MLMC_M0=16,
    MLMC_N0=4096,
    MLMC_DECAY=1.0,
    seed=12345
):
    sde = SDE().to(DEVICE).eval()
    for p in sde.parameters(): p.requires_grad_(False)

    true_oracle = monte_carlo_oracle_loss(
        sde, batch=ORACLE_BATCH_TRUE, t_steps=ORACLE_T, y0_scalar=y0_scalar, seed=seed
    )

    cub_losses, path_num, widths, works, samegrid_oracle, cub_grids = cubature_loss_vs_order(
        sde, disc_order, gamma, y0_scalar,
        DEGREE=DEGREE, RECOMBINE=RECOMBINE, THRES=THRES,
        BIN_SCALE=BIN_SCALE, RECOMBINE_EVERY=RECOMBINE_EVERY,
        BASE_DENSITY=BASE_DENSITY, GRID_GROWTH=GRID_GROWTH,
        ORACLE_SAMEGRID_BATCH=20_000, seed_for_samegrid=seed, verbose=False
    )
    err_to_high    = np.abs(cub_losses - true_oracle)
    err_to_samegd  = np.abs(cub_losses - samegrid_oracle)

    # MC (uniform)
    mc_vals, mc_errors = [], []
    for N in MC_N_LIST:
        est = monte_carlo_oracle_loss(sde, batch=int(N), t_steps=ORACLE_T, y0_scalar=y0_scalar, seed=seed)
        mc_vals.append(est); mc_errors.append(abs(est - true_oracle))
    mc_counts = np.array(MC_N_LIST, dtype=float)
    mc_errors = np.array(mc_errors, dtype=float)

    # QMC (uniform, BB)
    qmc_vals, qmc_errors = [], []
    for N in QMC_N_LIST:
        est_q = qmc_oracle_loss(sde, N=int(N), t_steps=ORACLE_T, y0_scalar=y0_scalar, seed=seed, use_bb=QMC_BB)
        qmc_vals.append(est_q); qmc_errors.append(abs(est_q - true_oracle))
    qmc_counts = np.array(QMC_N_LIST, dtype=float)
    qmc_errors = np.array(qmc_errors, dtype=float)

    # QMC on cubature grids vs same-grid oracle
    qmc_same_vals = []
    for T_k in cub_grids:
        ts_k = torch.tensor(T_k, device=DEVICE, dtype=torch.float64)
        est_q_same = qmc_oracle_loss(sde, N=int(QMC_SAMEGRID_N), ts_custom=ts_k,
                                     y0_scalar=y0_scalar, seed=seed, use_bb=False)
        qmc_same_vals.append(est_q_same)
    qmc_same_vals = np.array(qmc_same_vals, dtype=np.float64)
    qmc_same_errs = np.abs(qmc_same_vals - samegrid_oracle)

    # MLMC (uniform dyadic)
    mlmc_counts, mlmc_errors, mlmc_vals = [], [], []
    for L in MLMC_L_LIST:
        N_per_level = [max(2, int(MLMC_N0 / (2 ** (MLMC_DECAY * ell)))) for ell in range(L + 1)]
        est_mlmc, parts, n_eff, used = mlmc_estimator_loss(
            sde, Lmax=L, N_per_level=N_per_level, y0_scalar=y0_scalar, M0=MLMC_M0, seed=seed
        )
        mlmc_vals.append(est_mlmc)
        mlmc_errors.append(abs(est_mlmc - true_oracle))
        mlmc_counts.append(sum(used))

    mlmc_counts = np.array(mlmc_counts, dtype=float)
    mlmc_errors = np.array(mlmc_errors, dtype=float)

    return dict(
        true_oracle=true_oracle,
        cub=dict(losses=cub_losses, err_high=err_to_high, err_same=err_to_samegd, n=path_num),
        mc =dict(vals=np.array(mc_vals),  err=mc_errors,  n=mc_counts),
        qmc=dict(vals=np.array(qmc_vals), err=qmc_errors, n=qmc_counts),
        qmc_same=dict(vals=qmc_same_vals, err=qmc_same_errs, n=path_num),
        mlmc=dict(vals=np.array(mlmc_vals), err=mlmc_errors, n=mlmc_counts)
    )

# ===============================================================
# Entry
# ===============================================================
if __name__ == "__main__":
    # ---- Cubature settings ----
    DEGREE = 5
    disc_order = np.arange(1, 10, dtype=int)
    BASE_DENSITY = 500 #300
    GRID_GROWTH  = 1.2 #1.2
    RECOMBINE = True         # enable recombination
    THRES = (3 if DEGREE==3 else 6)
    RECOMBINE_EVERY = 1       # recombine every step for stricter control
    gamma = 0.6
    y0_scalar = 0.1

    # ---- Baselines ----
    mc_n_list  = np.array([3, 9, 27, 81, 243, 729, 2187, 6561,12000])
    #qmc_n_list = np.array([16, 32, 64, 128, 256, 512, 1024, 2048, 6561,12000])
    qmc_n_list = np.array([3, 9, 27, 81, 243, 729, 2187, 6561,12000])
    QMC_SAMEGRID_N = 2048

    MLMC_L_LIST = [0,1,2,3,4]
    MLMC_M0     = 16
    MLMC_N0     = 4096
    MLMC_DECAY  = 1.0

    # ---- Oracle resolution ----
    ORACLE_T = 1000
    ORACLE_BATCH_TRUE = 100000

    # ---- Repetitions ----
    N_REPEATS = 5
    outs = []

    for rep in range(N_REPEATS):
        print(f"[run] rep {rep+1}/{N_REPEATS}  (SciPy LP available: {HAVE_SCIPY})")
        out = run_experiment(
            disc_order=disc_order,
            gamma=gamma,
            y0_scalar=y0_scalar,
            DEGREE=DEGREE,
            RECOMBINE=RECOMBINE,
            THRES=THRES,
            BIN_SCALE=2.0,
            RECOMBINE_EVERY=RECOMBINE_EVERY,
            ORACLE_T=ORACLE_T,
            ORACLE_BATCH_TRUE=ORACLE_BATCH_TRUE,
            MC_N_LIST=mc_n_list,
            BASE_DENSITY=BASE_DENSITY,
            GRID_GROWTH=GRID_GROWTH,
            QMC_N_LIST=qmc_n_list,
            QMC_SAMEGRID_N=QMC_SAMEGRID_N,
            QMC_BB=True,
            MLMC_L_LIST=MLMC_L_LIST,
            MLMC_M0=MLMC_M0,
            MLMC_N0=MLMC_N0,
            MLMC_DECAY=MLMC_DECAY,
            seed=12345 + 1000*rep*random.randint(1,100)
        )
        outs.append(out)

    # ---- Aggregate (median across reps) ----
    def agg(outs, key_outer, key_inner):
        arr = np.stack([o[key_outer][key_inner] for o in outs], axis=0)
        return np.median(arr, axis=0)

    cub_n   = outs[0]['cub']['n']
    cub_e_h = agg(outs, 'cub', 'err_high')
    cub_e_s = agg(outs, 'cub', 'err_same')

    mc_n    = outs[0]['mc']['n'];     mc_e    = agg(outs, 'mc', 'err')
    qmc_n   = outs[0]['qmc']['n'];    qmc_e   = agg(outs, 'qmc', 'err')
    mlmc_n  = outs[0]['mlmc']['n'];   mlmc_e  = agg(outs, 'mlmc', 'err')
    qmc_same_e = agg(outs, 'qmc_same', 'err')  # aligned with cubature grids

    # ---------------- Plot ----------------
    plt.figure(figsize=(8.8,5.6))
    plt.loglog(cub_n,   cub_e_h, 'o-', color='tab:red',    label='Cubature (high-res oracle)')
    plt.loglog(cub_n,   cub_e_s, 'd-', color='tab:purple', label='Cubature (same-grid oracle)')
    plt.loglog(mc_n,    mc_e,    's-', color='tab:blue',   label='MC (uniform)')
    plt.loglog(qmc_n,   qmc_e,   'p-', color='tab:pink',   label='QMC (uniform, BB)')
    plt.loglog(mlmc_n,  mlmc_e,  '^-', color='tab:green',  label='MLMC (uniform)')
    plt.loglog(cub_n,   qmc_same_e, 'x-', color='tab:orange', label='QMC (same-grid as Cubature)')

    xmin = min(cub_n.min(), mc_n.min(), qmc_n.min(), mlmc_n.min())
    xmax = max(cub_n.max(), mc_n.max(), qmc_n.max(), mlmc_n.max())
    xx = np.logspace(np.log10(xmin), np.log10(xmax), 200)
    plt.loglog(xx, xx**(-1.0),  'k--', linewidth=1.0, label=r'$n^{-1}$')
    plt.loglog(xx, xx**(-0.5),  'k:',  linewidth=1.0, label=r'$n^{-1/2}$')

    plt.xlabel('Number of paths / coupled samples (n)')
    plt.ylabel('Absolute error')
    plt.title('Convergence vs number of paths (MC, QMC, MLMC, Cubature)')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(loc='upper right', framealpha=0.9)
    plt.tight_layout()
    plt.show()
 