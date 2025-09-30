#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 14:39:51 2025

@author: lukesnow
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 15:38:44 2025

@author: lukesnow
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -----------------------------------------------------------------------------
# Cubature-on-Wiener-Space vs Monte Carlo:
# - Build degree-3 or degree-5 cubature path trees with optional recombination
# - Solve controlled ODEs (Stratonovich CDE) using RK4 driven by piecewise-linear ω
# - Compare time-averaged MSE against a high-resolution Monte-Carlo oracle
# - Plot error vs work/support size, with repeated runs for error bars
#
# Notes:
# • All tensors/arrays use float64 to keep RK4 / SDE solves stable & comparable.
# • Only comments/docstrings added below; no functional changes were made.
# -----------------------------------------------------------------------------

import math, gc, random
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
import torchsde
from numpy.linalg import svd

# ===============================================================
# Global dtype / device
# ===============================================================
torch.set_default_dtype(torch.float64)   # keep solver & tensors in fp64
DTYPE_NP = np.float64                    # path arrays in fp64 too
DEVICE = torch.device('cpu')             # set to 'cuda' if available

# ===============================================================
# Helpers
# ===============================================================
def nullspace(A, atol=1e-13, rtol=0):
    """
    Compute a (right) nullspace basis of matrix A via SVD.

    Parameters
    ----------
    A : (m,n) array-like
    atol : float
        Absolute tolerance for singular values considered nonzero.
    rtol : float
        Relative tolerance (times largest singular value).

    Returns
    -------
    ns : (n, k) np.ndarray
        Columns span the nullspace {x : A x = 0}.
    """
    A = np.atleast_2d(A)
    u, s, vh = svd(A)
    tol = max(atol, rtol * s[0]) if len(s) else atol
    nnz = (s >= tol).sum() if len(s) else 0
    ns = vh[nnz:].conj().T if nnz < vh.shape[0] else np.zeros((A.shape[1], 0))
    return ns

# ---- Degree-5 piecewise base paths (KLV 1D, weights [1/6, 2/3, 1/6]) ----
def path5_1(t):
    """Degree-5 base path segment (scaled later by sqrt(dt)); piecewise in t∈[0,1]."""
    if t <= 1/3:
        return -math.sqrt(3)*(0.5 * (4 - math.sqrt(22)) * t)
    elif 1/3 < t <= 2/3:
        return -math.sqrt(3)*(1/6 * (4 - math.sqrt(22)) + (-1 + math.sqrt(22)) * (t - 1/3))
    else:
        return -math.sqrt(3)*(1/6 * (2 + math.sqrt(22)) + 0.5 * (4 - math.sqrt(22)) * (t - 2/3))

def path5_2(t):
    """Degree-5 base path segment (symmetric branch)."""
    if t <= 1/3:
        return  math.sqrt(3)*(0.5 * (4 - math.sqrt(22)) * t)
    elif 1/3 < t <= 2/3:
        return  math.sqrt(3)*(1/6 * (4 - math.sqrt(22)) + (-1 + math.sqrt(22)) * (t - 1/3))
    else:
        return  math.sqrt(3)*(1/6 * (2 + math.sqrt(22)) + 0.5 * (4 - math.sqrt(22)) * (t - 2/3))

def path5_3(_):
    """Degree-5 middle atom (zero path)."""
    return 0.0

# ---- Degree-3 linear base paths (endpoints ±√dt) ----
def path3_plus(u):   # u in [0,1]
    """Linear ramp +u for degree-3 atom (scaled by sqrt(dt) when used)."""
    return +u
def path3_minus(u):
    """Linear ramp -u for degree-3 atom (scaled by sqrt(dt) when used)."""
    return -u

# ===============================================================
# Model (Stratonovich SDE)
# ===============================================================
state_size, brownian_size = 1, 1

class TimeVaryingLinear(nn.Module):
    """
    Linear layer whose weights are generated from time t via a small network.
    Produces a (out_features × in_features) matrix applied to input x.
    """
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
    """
    Simple 1D Stratonovich SDE with time-varying linear drift and
    state-dependent diffusion (two-layer ELU MLPs for both).
    """
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
        """Drift f(t,x) with ELU nonlinearities and time-modulated linear maps."""
        m = nn.ELU()
        x = m(self.tvl1(x, t))
        x = m(self.tvl2(x, t))
        x = self.tvl3(x, t)
        return x

    def g(self, t, x):
        """Diffusion g(t,x) (diagonal in 1D) via small ELU MLP."""
        m = nn.ELU()
        x = m(self.dfc1(x))
        x = self.dfc2(x)
        return x

# ===============================================================
# Controlled ODE solver (RK4) for piecewise-linear drive
# ===============================================================
@torch.no_grad()
def solve_cde_rk4(v1, v2, w_vals, t_points, y0_scalar):
    """
    Integrate a controlled ODE (Stratonovich CDE) using RK4 with piecewise-constant ω̇.

    v1, v2   : callables f(t,y), g(t,y) for drift/diffusion
    w_vals   : ω(t_i) sampled on t_points (shape [T])
    t_points : strictly increasing times (1-D tensor)
    y0_scalar: initial state value (scalar)

    Returns
    -------
    t_points : same as input
    y_hist   : solution at all t_points (1-D tensor)
    """
    assert w_vals.shape == t_points.shape
    if not bool(torch.all(t_points[1:] > t_points[:-1])):
        raise ValueError("Evaluation times `ts` must be strictly increasing.")

    y = torch.tensor(float(y0_scalar), device=t_points.device, dtype=t_points.dtype)
    y_hist = torch.empty_like(t_points)
    y_hist[0] = y
    for i in range(len(t_points) - 1):
        t0, t1 = t_points[i], t_points[i+1]
        dt  = t1 - t0
        dω  = w_vals[i+1] - w_vals[i]
        wdot = dω / dt

        def F(t, y_):
            yy = y_.view(1,1)
            return (v1(t, yy) + v2(t, yy) * wdot).squeeze()

        # Classic RK4 integration with constant control on [t0,t1]
        k1 = F(t0,           y)
        k2 = F(t0 + 0.5*dt,  y + 0.5*dt*k1)
        k3 = F(t0 + 0.5*dt,  y + 0.5*dt*k2)
        k4 = F(t1,           y + dt*k3)
        y  = y + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        y_hist[i+1] = y
    return t_points, y_hist

@torch.no_grad()
def cde_time_avg_mse_batch(v1, v2, W, ts, y0_scalar, target_vals):
    """
    Batched CDE RK4 with trapezoidal time-averaged MSE accumulation.

    Parameters
    ----------
    v1,v2       : drift/diffusion callables
    W           : [P, T] tensor of ω(t_i) per path
    ts          : [T] strictly increasing times (torch.float64/32)
    y0_scalar   : scalar initial condition
    target_vals : [T] target signal values (e.g., sin(2π t))

    Returns
    -------
    per-path time-averaged MSE : torch.Tensor [P]
    """
    device, dtype = ts.device, ts.dtype
    P, T = W.shape
    y  = torch.full((P, 1), float(y0_scalar), device=device, dtype=dtype)
    err_accum = torch.zeros(P, device=device, dtype=dtype)

    for i in range(T - 1):
        t0, t1 = ts[i], ts[i+1]
        dt = t1 - t0
        wdot = (W[:, i+1] - W[:, i]) / dt  # [P]

        def F(t, y_):  # y_ shape [P,1] -> returns [P,1]
            return (v1(t, y_) + v2(t, y_) * wdot.view(-1, 1))

        # RK4 step
        k1 = F(t0,             y)
        k2 = F(t0 + 0.5*dt,    y + 0.5*dt*k1)
        k3 = F(t0 + 0.5*dt,    y + 0.5*dt*k2)
        k4 = F(t1,             y + dt*k3)
        y_next = y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        # Trapezoidal time-average contribution on [t0, t1]
        e0 = (target_vals[i]   - y.squeeze(1))**2
        e1 = (target_vals[i+1] - y_next.squeeze(1))**2
        err_accum += 0.5 * (e0 + e1) * dt

        y = y_next

    return err_accum / (ts[-1] - ts[0])

# ===============================================================
# Time partition and local grid
# ===============================================================
def make_time_partition(k, gamma):
    """
    Power-law macro partition t_d[0..k] with exponent gamma (denser near 1).

    Returns
    -------
    t_d : np.ndarray (k+1,)
    """
    t_d = np.zeros(k + 1, dtype=DTYPE_NP)
    for i in range(k + 1):
        t_d[i] = 1.0 - (1.0 - i / k) ** gamma
    return t_d

def per_step_grid(dt, base_density, total_order, growth=1.0, dtn_min=6):
    """
    Build a local microgrid inside a macro step of width dt.

    Returns offsets in (0, dt], i.e., excluding 0 to avoid duplication
    across steps but including dt to hit the step endpoint.

    Parameters
    ----------
    dt : float
    base_density : float
    total_order : int
    growth : float
        Exponent to scale #points with total_order.
    dtn_min : int
        Minimum points per step.

    Returns
    -------
    xloc : np.ndarray (dtn,)
    """
    dtn = max(dtn_min, int(math.ceil(dt * base_density * (total_order ** growth))))
    return np.linspace(0.0, dt, dtn + 1, endpoint=True, dtype=DTYPE_NP)[1:]

# ===============================================================
# Build child increments by degree
# ===============================================================
def build_child_increments(degree, dt, x_loc):
    """
    Construct absolute ω child paths within a macro step for a given degree.

    Returns
    -------
    inc_list : list[np.ndarray]
        Each entry has shape [len(x_loc)], representing absolute ω values
        to be appended after the parent’s last value on this step.
    weights : np.ndarray
        Cubature weights for each child.
    """
    if degree == 3:
        # Linear ± paths to endpoints ±√dt with equal weights 1/2.
        root = math.sqrt(dt)
        inc_plus  = np.array([ root * (z / dt) for z in x_loc], dtype=DTYPE_NP)
        inc_minus = np.array([-root * (z / dt) for z in x_loc], dtype=DTYPE_NP)
        return [inc_minus, inc_plus], np.array([0.5, 0.5], dtype=DTYPE_NP)  # order: -, +
    elif degree == 5:
        # KLV degree-5: three atoms with weights 1/6, 2/3, 1/6.
        inc0 = np.array([math.sqrt(dt) * path5_1(float(z / dt)) for z in x_loc], dtype=DTYPE_NP)
        inc1 = np.array([path5_3(float(z)) for z in x_loc], dtype=DTYPE_NP)
        inc2 = np.array([math.sqrt(dt) * path5_2(float(z / dt)) for z in x_loc], dtype=DTYPE_NP)
        return [inc0, inc1, inc2], np.array([1/6, 2/3, 1/6], dtype=DTYPE_NP)
    else:
        raise ValueError("degree must be 3 or 5")

def init_first_step_paths_times(t_d, base_density, total_order, growth, degree):
    """
    Initialize tree at first macro step:
      • Build microgrid, degree-specific child increments, and weights
      • Return initial path matrix p (children × time_points), weights w, and time grid T

    Returns
    -------
    p0 : np.ndarray [children, time_points_incl_0]
    w  : np.ndarray [children]
    T  : np.ndarray [time_points_incl_0]
    """
    dt0  = float(t_d[1] - t_d[0])
    xloc = per_step_grid(dt0, base_density, total_order, growth=growth, dtn_min=6)
    incs, w = build_child_increments(degree, dt0, xloc)  # list of arrays
    p0 = np.stack(incs, axis=0)  # rows = #children
    # Absolute times: prepend 0
    T = np.concatenate([np.array([0.0], dtype=DTYPE_NP), xloc])  # start at t=0
    # Prepend ω(0)=0 column
    p0 = np.concatenate([np.zeros((p0.shape[0], 1), dtype=DTYPE_NP), p0], axis=1)
    return p0, w, T

# ===============================================================
# Optional: Chebyshev Vandermonde (for recombination)
# ===============================================================
def chebV(z, r):
    """
    Chebyshev Vandermonde matrix up to degree r for nodes z∈[-1,1].

    Returns
    -------
    V : np.ndarray (r+1, m)
        Rows correspond to T_0(z),...,T_r(z).
    """
    z = np.asarray(z, dtype=np.float64)
    m = z.shape[0]
    V = np.empty((r + 1, m), dtype=np.float64)
    V[0, :] = 1.0
    if r >= 1:
        V[1, :] = z
    for k in range(1, r):
        V[k + 1, :] = 2.0 * z * V[k, :] - V[k - 1, :]
    return V

# ===============================================================
# Oracles
# ===============================================================
@torch.no_grad()
def monte_carlo_oracle_loss(sde, batch, t_steps, y0_scalar, seed):
    """
    High-resolution MC estimate of time-averaged MSE on a uniform grid.

    Returns
    -------
    float : mean over batch of ∫(Y(t) - sin(2πt))^2 dt / T
    """
    torch.manual_seed(seed); np.random.seed(seed)
    sde = sde.to(DEVICE).eval()
    for p in sde.parameters(): p.requires_grad_(False)
    ts = torch.linspace(0, 1, t_steps, device=DEVICE, dtype=torch.float64)
    y0 = torch.full((batch, state_size), y0_scalar, device=DEVICE, dtype=torch.float64)
    with torch.inference_mode():
        Y = torchsde.sdeint(sde, y0, ts, method='midpoint')  # [T,B,1]
    fcn = torch.sin(2 * math.pi * ts).unsqueeze(1)
    return ((Y.squeeze(-1) - fcn)**2).mean(dim=0).mean().item()

@torch.no_grad()
def monte_carlo_oracle_on_custom_grid(sde, t_points, y0_scalar, batch, seed):
    """
    MC oracle on a custom (possibly non-uniform) grid t_points.

    Returns
    -------
    float : time-averaged MSE via trapezoidal rule on t_points.
    """
    torch.manual_seed(seed); np.random.seed(seed)
    sde = sde.to(DEVICE).eval()
    for p in sde.parameters(): p.requires_grad_(False)
    ts = torch.tensor(t_points, device=DEVICE, dtype=torch.float64)
    if not bool(torch.all(ts[1:] > ts[:-1])):
        raise ValueError("Custom oracle grid must be strictly increasing.")
    y0 = torch.full((batch, state_size), y0_scalar, device=DEVICE, dtype=torch.float64)
    with torch.inference_mode():
        Y = torchsde.sdeint(sde, y0, ts, method='midpoint')  # [T,B,1]
    fcn = torch.sin(2 * math.pi * ts).unsqueeze(1)
    err2 = (Y.squeeze(-1) - fcn)**2  # [T,B]
    L = torch.trapz(err2.mean(dim=1), ts) / (ts[-1] - ts[0])
    return L.item()

# ===============================================================
# Cubature builder + loss (RK4+CDE) with WORK counting
# ===============================================================
def cubature_loss_vs_order(
    sde,
    disc_order,
    gamma=0.6,
    y0_scalar=0.1,
    DEGREE=3,                # 3 => expect O(work^{-1}), 5 => can approach O(work^{-2})
    RECOMBINE=True,
    THRES=None,              # if None: set to 3 (deg3) or 6 (deg5)
    BIN_SCALE=2.0,           # wider bins -> more stable elimination
    RECOMBINE_EVERY=2,       # recombine every k steps (>=1). Set to large to disable effectively.
    BASE_DENSITY=400,
    GRID_GROWTH=1.25,
    ORACLE_SAMEGRID_BATCH=8192,
    seed_for_samegrid=0,
    verbose=True
):
    """
    Build a degree-m (m∈{3,5}) cubature path tree across k macro steps, optionally
    recombining path endpoints using moment matching (Chebyshev basis), and
    evaluate the time-averaged MSE via RK4+CDE on the same native grid.

    Returns
    -------
    losses          : np.ndarray | cubature losses per order k
    path_counts     : np.ndarray | number of active paths at final step
    widths          : np.ndarray | total number of time points per k
    works           : np.ndarray | cumulative 'work' (sum of nodes per step)
    samegrid        : np.ndarray | MC oracle evaluated on the same non-uniform grid
    """
    if THRES is None:
        THRES = 3 if DEGREE == 3 else 6
    rdeg = THRES - 1

    losses, path_counts, widths, samegrid = [], [], [], []
    works = []   # cumulative work per order (sum of nodes per step)

    for k in disc_order:
        # Partition + first step (time stamps included)
        t_d = make_time_partition(k, gamma)
        p, w, T = init_first_step_paths_times(t_d, BASE_DENSITY, k, GRID_GROWTH, DEGREE)
        work = p.shape[0]  # nodes used for integrating over step 1

        # Grow tree over remaining macro steps
        for step in range(1, k):
            dt  = float(t_d[step + 1] - t_d[step])
            x_loc = per_step_grid(dt, BASE_DENSITY, k, growth=GRID_GROWTH, dtn_min=6)
            dtn = len(x_loc)

            rows_old = p.shape[0]
            child_incs, child_w = build_child_increments(DEGREE, dt, x_loc)
            num_children = len(child_incs)
            rows_new = rows_old * num_children

            # Repeat parent history across children
            p_rep = np.repeat(p, num_children, axis=0)
            last_vals = p[:, -1:]
            base_block = np.repeat(last_vals, num_children, 0)

            # Append child segments to each parent's endpoint
            new_block = np.empty((rows_new, dtn), dtype=DTYPE_NP)
            for r in range(rows_new):
                kind = r % num_children
                new_block[r, :] = base_block[r, 0] + child_incs[kind]

            p = np.concatenate([p_rep, new_block], axis=1)
            T = np.concatenate([T, T[-1] + x_loc], axis=0)

            # Update weights under product rule
            w = np.repeat(w, num_children) * np.tile(child_w, rows_old)

            # Optional endpoint-only recombination (coarse binning, moment matching)
            if RECOMBINE and (step % RECOMBINE_EVERY == 0):
                pts = p[:, -1]
                u = max(1e-12, BIN_SCALE * dt)
                mx, mn = np.max(pts), np.min(pts)
                num_edges = max(2, int(max(1, (mx - mn) / u)))
                prt = np.linspace(mn, mx, num_edges)
                for j in range(len(prt) - 1):
                    find = np.where((pts >= prt[j]) & (pts < prt[j + 1]))[0]
                    if len(find) > THRES:
                        pts_bin = pts[find].astype(np.float64)
                        w_work  = w[find].astype(np.float64)
                        mass0   = float(w_work.sum())

                        a, b_ = float(pts_bin.min()), float(pts_bin.max())
                        if b_ - a < 1e-14: continue
                        z = (2.0 * (pts_bin - a) / (b_ - a)) - 1.0
                        V = chebV(z, rdeg)
                        b_mom = V @ w_work

                        local_to_global = find.copy()
                        eps_dir = 1e-18
                        while len(local_to_global) > (rdeg + 1):
                            K = nullspace(V)
                            if K.size == 0: break
                            d = K[:, 0]
                            with np.errstate(divide='ignore', invalid='ignore'):
                                frac = np.where(d < -eps_dir, w_work / (-d), np.inf)
                            if not np.isfinite(frac).any(): break
                            alpha = np.min(frac); drop_local = int(np.argmin(frac))
                            w_work = w_work + alpha * d
                            w_work[w_work < 0] = 0.0
                            keep = np.ones(len(w_work), dtype=bool); keep[drop_local] = False
                            w_work = w_work[keep]; z = z[keep]; V = V[:, keep]; local_to_global = local_to_global[keep]

                        # Nonnegative LS fit + mass renormalization to preserve total weight
                        x, *_ = np.linalg.lstsq(V, b_mom, rcond=None)
                        x = np.maximum(x, 0.0); ssum = x.sum()
                        x = x * (mass0 / ssum) if ssum > 0 else np.full_like(x, mass0 / len(x))

                        survivors = local_to_global
                        dropped   = np.setdiff1d(find, survivors, assume_unique=False)
                        w[find] = 0.0; w[survivors] = x.astype(DTYPE_NP)
                        if dropped.size > 0:
                            p = np.delete(p, dropped, axis=0)
                            w = np.delete(w, dropped, axis=0)
                            pts = p[:, -1]

            # Accumulate 'work' as sum of active nodes per step
            work += p.shape[0]

            # Housekeeping to reduce peak memory churn
            del p_rep, new_block, last_vals, base_block, child_incs, x_loc
            gc.collect()

        # ----- Loss with batched RK4+CDE on SAME grid (time-weighted) -----
        ts    = torch.tensor(T, device=DEVICE, dtype=torch.float64)
        fcn_t = torch.sin(2 * math.pi * ts)
        
        W = torch.from_numpy(p).to(DEVICE)              # [P,T]
        w_t = torch.from_numpy(w).to(DEVICE)            # [P]
        per_path_mse = cde_time_avg_mse_batch(sde.f, sde.g, W, ts, y0_scalar, fcn_t)  # [P]
        total_loss = (w_t * per_path_mse).sum().item()

        cub_loss = float(abs(total_loss))
        same_grid_oracle = monte_carlo_oracle_on_custom_grid(
            sde, T, y0_scalar, batch=ORACLE_SAMEGRID_BATCH, seed=seed_for_samegrid
        )

        # Collect per-order diagnostics
        losses.append(cub_loss)
        path_counts.append(int(p.shape[0]))
        widths.append(len(T))
        works.append(int(work))
        samegrid.append(same_grid_oracle)

        # Free per-order buffers
        del p, w, ts, fcn_t
        gc.collect()

    return (np.array(losses, dtype=np.float64),
            np.array(path_counts, dtype=int),
            np.array(widths, dtype=int),
            np.array(works, dtype=int),
            np.array(samegrid, dtype=np.float64))

# ===============================================================
# Full experiment
# ===============================================================
def run_experiment(
    disc_order,
    gamma,
    y0_scalar,
    DEGREE,                 # 3 or 5
    RECOMBINE,
    THRES,
    BIN_SCALE,
    RECOMBINE_EVERY,
    ORACLE_T,
    ORACLE_BATCH_TRUE,
    MC_N_LIST,
    BASE_DENSITY,
    GRID_GROWTH,
    MC 
):
    """
    Execute one end-to-end comparison:
      1) High-res MC oracle on uniform grid (reference)
      2) Cubature loss vs discretization order (non-uniform native grid)
      3) Monte Carlo estimates vs sample size list MC_N_LIST

    Returns
    -------
    Tuple of arrays for cubature losses, errors to oracle(s), path counts, widths,
    works, same-grid MC oracle, MC counts/values/errors, and true oracle value.
    """
    # Fixed SDE params for a fair comparison
    sde = SDE().to(DEVICE).eval()
    for p in sde.parameters(): p.requires_grad_(False)

    seed = 12345 + random.randint(0, 10_000)

    true_oracle = monte_carlo_oracle_loss(
        sde, batch=ORACLE_BATCH_TRUE, t_steps=ORACLE_T, y0_scalar=y0_scalar, seed=seed
    )
  #  print(f"[Oracle] high-res ≈ {true_oracle:.8f}")

    cub_losses, path_num, widths, works, samegrid_oracle = cubature_loss_vs_order(
        sde, disc_order, gamma, y0_scalar,
        DEGREE=DEGREE,
        RECOMBINE=RECOMBINE,
        THRES=THRES,
        BIN_SCALE=BIN_SCALE,
        RECOMBINE_EVERY=RECOMBINE_EVERY,
        BASE_DENSITY=BASE_DENSITY,
        GRID_GROWTH=GRID_GROWTH,
        ORACLE_SAMEGRID_BATCH=8192,
        seed_for_samegrid=seed,
        verbose=True
    )
    err_to_high   = np.abs(cub_losses - true_oracle)
    err_to_samegd = np.abs(cub_losses - samegrid_oracle)

    """
    for i, k in enumerate(disc_order):
        print(f"[Cub] order={k:2d} paths={path_num[i]:5d} width={widths[i]:5d} "
              f"work={works[i]:7d} loss={cub_losses[i]:.8f} "
              f"|err(high)|={err_to_high[i]:.3e} |err(samegrid)|={err_to_samegd[i]:.3e}")
    """
    
    # Monte-Carlo estimates vs N (uniform fine grid)
    mc_errors = np.zeros((MC,len(MC_N_LIST)))
    
    mc_errors, mc_vals = [], []
    for N in MC_N_LIST:
        est = monte_carlo_oracle_loss(
            sde, batch=N, t_steps=ORACLE_T, y0_scalar=y0_scalar, seed=seed
        )
        mc_vals.append(est); mc_errors.append(abs(est - true_oracle))
        #print(f"[MC] N={N:5d} | est={est:.8f} | |err|={abs(est-true_oracle):.3e}")
    mc_errors = np.array(mc_errors, dtype=float)
    mc_counts = np.array(MC_N_LIST, dtype=float)


  #  print("\n=== Summary (last points) ===")
  #  print(f"Cubature: paths={path_num[-1]}, width={widths[-1]}, work={works[-1]}, "
    #      f"loss={cub_losses[-1]:.8f}, |err(high)|={err_to_high[-1]:.3e}, "
    #      f"|err(samegrid)|={err_to_samegd[-1]:.3e}")
   # print(f"Monte-Carlo: N={mc_counts[-1]:.0f}, est={mc_vals_[-1]:.8f}, |err|={np.mean(mc_errors,0)[-1]:.3e}")

    return cub_losses,\
        err_to_high,\
        err_to_samegd,\
        path_num,\
        widths,\
        works,\
        samegrid_oracle,\
        mc_counts,\
        mc_vals,\
        mc_errors,\
        true_oracle

# ===============================================================
# Entry
# ===============================================================

 # Pick the degree:
DEGREE = 5   # set to 5 to try the degree-5 KLV scheme

# Discretization orders (tree depth), and Monte-Carlo sample sizes to test
disc_order = np.arange(1, 10, dtype=int)   # path tree depth (k)
# mc_n_list = np.array([16,32,64,128,256,512,1024,2048])
mc_n_list = np.array([  3,    9,   27,   81,  243,  729, 2187, 6561, 20000])
#mc_n_list = np.array([2,4,8,16,32,64,128,256,512,1024, 2048, 6500])

# Number of outer repeats for error bars
MC = 20
mc_errors_ = np.zeros((MC,len(mc_n_list)))
err_to_high_ = np.zeros((MC,len(disc_order)))
 
for mc in range(MC):
    print(f'mc: {mc+1}/{MC}')
    cub_losses, err_to_high, err_to_samegd, path_num, widths,\
        works, samegrid_oracle, mc_counts, mc_vals, mc_errors, \
        true_oracle = run_experiment(
        disc_order=disc_order,
        gamma=0.6,
        y0_scalar=0.1,
        DEGREE=DEGREE,
        RECOMBINE=False,
        THRES=(3 if DEGREE==3 else 6),   # moments matched per bin
        BIN_SCALE=2.0,
        RECOMBINE_EVERY=2,               # recombine every 2 steps
        ORACLE_T=4000,
        ORACLE_BATCH_TRUE=100000,
        MC_N_LIST= mc_n_list,
        BASE_DENSITY=300,                # dense microgrid per unit time at k=1
        GRID_GROWTH=1.2,              # grow microgrid with order to keep RK4 error below cubature error
        MC = 10
        )
    mc_errors_[mc,:] = mc_errors
    err_to_high_[mc,:] = err_to_high
     
# ----- Means and Standard Errors across MC repeats -----
mean_cub = err_to_high_.mean(axis=0)                           # [len(disc_order)]
se_cub   = err_to_high_.std(axis=0, ddof=1) / np.sqrt(MC)      # SE = std/sqrt(M)
 
mean_mc  = mc_errors_.mean(axis=0)                             # [len(mc_n_list)]
se_mc    = mc_errors_.std(axis=0, ddof=1) / np.sqrt(MC)

# For log-y shading, avoid zeros (clip very small)
eps = 1e-18
lower_cub = np.clip(mean_cub - se_cub, eps, None)
upper_cub = np.clip(mean_cub + se_cub, eps, None)
lower_mc  = np.clip(mean_mc  - se_mc,  eps, None)
upper_mc  = np.clip(mean_mc  + se_mc,  eps, None)

     
# ---------------- Plots ----------------
# 1) Error vs WORK (theoretical axis)
plt.figure(figsize=(8,5.2))
plt.semilogy(path_num, np.mean(err_to_high_,0), 'o-r', label='Cubature error vs work')
plt.semilogy(mc_counts, np.mean(mc_errors_,0), 's-b', label='Monte-Carlo error vs N')

# Reference slopes (heuristic guides)
xx = np.linspace(min(works.min(), mc_counts.min()), max(works.max(), mc_counts.max()), 300)
Cref = err_to_high[0] * (xx[0]**1)
plt.plot(xx, Cref * xx**(-1), 'g--', label='O(n^{-1}) reference')
Cref2 = err_to_high[0] * (xx[0]**1)
plt.plot(xx, Cref2 * xx**(-1/2), 'k--', label='O(n^{-1/2}) reference')

plt.xlabel('Total work (sum of active nodes per step)')
plt.ylabel('Absolute error to high-res oracle')
plt.title(f'Cubature (deg={DEGREE}) vs MC: error vs work')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend(); plt.tight_layout(); plt.show()
 
# 2) Error vs final #paths (for comparison)
plt.figure(figsize=(8,5.2))
plt.loglog(path_num, np.mean(err_to_high_,0), 'o-r', label='Cubature error')
plt.loglog(mc_counts, np.mean(mc_errors_,0), 's-b', label='Monte-Carlo error')

# (Optional) reference slopes
xx = np.linspace(min(works.min(), mc_counts.min()), max(works.max(), mc_counts.max()), 300)
Cref = 1
#plt.loglog(xx, Cref * xx**(-1), 'g--', label='O(n^{-1}) reference')
Cref2 = 1
#plt.loglog(xx, Cref2 * xx**(-1/2), 'k--', label='O(n^{-1/2}) reference')

plt.xlabel('Number (n) of sample paths / ODE solves')
plt.ylabel('Absolute error to oracle')
plt.title('Error vs support size')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend(); plt.tight_layout(); plt.show()
 
# 3) Error vs support size with mean ± 1 Std shading (log y)
plt.figure(figsize=(8,5.2))
plt.loglog(path_num, mean_cub, 'o-', color='crimson', label='Cubature mean error')
plt.fill_between(path_num, np.clip(mean_cub - se_cub, eps, None),
                               np.clip(mean_cub + se_cub, eps, None),
                 color='crimson', alpha=0.18, linewidth=0, label='Cubature ±1 Std')

plt.loglog(mc_counts, mean_mc, 's-', color='royalblue', label='Monte-Carlo mean error')
plt.fill_between(mc_counts, np.clip(mean_mc - se_mc, eps, None),
                               np.clip(mean_mc + se_mc, eps, None),
                 color='royalblue', alpha=0.18, linewidth=0, label='MC ±1 Std')

plt.xlabel('Number (n) of sample paths / ODE solves', fontsize = 15)
plt.ylabel('Absolute error to oracle',fontsize = 15)
plt.title('Error vs path support size',fontsize = 15)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout(); plt.show()
