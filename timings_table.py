#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 14:27:39 2025

@author: lukesnow
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Benchmark script: MC (SDE solver) vs Cubature (ODE solver) for Neural SDEs
#
# What this does
# --------------
# • Generates synthetic 1-D observations from a known SDE (time-inhomogeneous).
# • Trains a latent SDE model in two ways:
#     (A) "MC branch": Monte Carlo simulation via torchsde.sdeint (Euler).
#     (B) "Cubature branch": Degree-3 Wiener-space cubature -> deterministic ODEs
#         driven by piecewise-constant ω̇ on the solver grid, then torchdiffeq.odeint.
# • Compares median per-epoch wall-clock time and Python peak memory across branches.
# • Optionally records GPU peak allocation (if CUDA available).
# • Saves a CSV table and simple plots (time, memory, overhead) vs latent dimension D.
#
# Notes
# -----
# • We only add comments for clarity; the executable statements remain unchanged.
# • The cubature construction here is a streamed degree-3 scheme with a macro-grid
#   and round-robin assignment of axis/sign choices per path.
# • SDE types:
#     - Data SDE (Stratonovich) produces observations.
#     - Latent SDE model (posterior drift f_φ, prior drift f_θ, diagonal diffusion g_θ).
# • Loss components:
#     - KL-like control term u = (f_φ - f_θ) / diag(g_θ).
#     - MSE data fit via a simple decoder from latent to observation.
#     - Small penalty to align initial prediction with data at t=0.
# -----------------------------------------------------------------------------


import math, time, gc, csv, os
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.optim as optim
import torchsde
from torchdiffeq import odeint  # ODE solver for cubature branch

# ----------------------------
# Repro & device
# ----------------------------
SEED = 1234
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.benchmark = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float32)

# ----------------------------
# Utility: epoch wall time + Python peak memory (MB)
# ----------------------------
import tracemalloc
def _epoch_wrapper(fn):
    """
    Measure wall time and Python heap (via tracemalloc) for a single epoch step.

    Returns
    -------
    dt : float
        Seconds elapsed.
    py_peak_mb : float
        Peak Python memory (MB) during the wrapped call, as seen by tracemalloc.
    """
    tracemalloc.start()
    tracemalloc.reset_peak()
    t0 = time.time()
    fn()
    dt = time.time() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    py_peak_mb = peak / (1024 ** 2)
    return dt, py_peak_mb

# ----------------------------
# Synthetic data SDE (1-D observations)
# ----------------------------
class SDE_Data(nn.Module):
    """
    Simple time-inhomogeneous 1-D Stratonovich SDE:
        dx = f(t, x) dt + g(t, x) ◦ dW_t,
    with diagonal noise (scalar here). Drift/diffusion only depend on time.
    """
    noise_type = "diagonal"
    sde_type   = "stratonovich"
    def __init__(self):
        super().__init__()
    def f(self, t, x):
        # Shared drift for all batch elements at time t
        B = x.size(0)
        drift = 1.0 + 0.5 * torch.sin(2 * torch.pi * t)
        return drift.expand(B, 1)
    def g(self, t, x):
        # Shared diffusion magnitude (diagonal entry) at time t
        B = x.size(0)
        diff = 0.1 + 0.05 * torch.cos(4 * torch.pi * t)
        return diff.expand(B, 1)  # diagonal entries

# ----------------------------
# Latent SDE model used for both branches
# ----------------------------
class MLP(nn.Module):
    """
    Small MLP taking concatenated [t, y] and returning a vector field value.
    Accepts batched inputs in either [B, D] or [T, B, D] layout.
    """
    def __init__(self, in_dim, out_dim, hidden=48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, t, y):
        # Handle both [B, D] and [T, B, D] forms while injecting time as a feature
        if y.dim() == 2:  # [B,D]
            B, D = y.shape
            tt = (t.expand(B) if t.dim() == 0 else t).unsqueeze(-1)
            return self.net(torch.cat([tt, y], dim=-1))
        elif y.dim() == 3:  # [T,B,D]
            T, B, D = y.shape
            if t.dim() == 1:
                tt = t[:, None].expand(-1, B).unsqueeze(-1)  # [T,B,1]
            else:
                tt = t.expand(T)[:, None].expand(-1, B).unsqueeze(-1)
            out = self.net(torch.cat([tt, y], dim=-1).reshape(-1, D + 1))
            return out.view(T, B, -1)
        else:
            raise ValueError("Unsupported y.dim()")

class DiagDiff(nn.Module):
    """Learned diagonal diffusion: returns per-dimension std as a vector [*, D]."""
    def __init__(self, D):
        super().__init__()
        self.log_std = nn.Parameter(torch.zeros(D))
    def forward(self, t, y):
        std = torch.exp(self.log_std)
        if y.dim() == 2:   # [B,D]
            B, D = y.shape
            return std.unsqueeze(0).expand(B, D)
        elif y.dim() == 3: # [T,B,D]
            T, B, D = y.shape
            return std.unsqueeze(0).unsqueeze(0).expand(T, B, D)
        else:
            raise ValueError("Unsupported y.dim()")

class PosteriorSDE(torchsde.SDEIto):
    """
    Posterior SDE in Itô form with diagonal noise, used by torchsde.sdeint.
    Wraps learned drift f_φ and diagonal diffusion g_θ (vector of stds).
    """
    def __init__(self, f_phi, g_theta):
        super().__init__(noise_type="diagonal")
        self._f, self._g = f_phi, g_theta
    def f(self, t, y): return self._f(t, y)
    def g(self, t, y): return self._g(t, y)   # diag vector [*,D]

class LatentSDE(nn.Module):
    """
    Latent SDE parameterization:
      - f_theta : prior drift
      - f_phi   : posterior drift
      - g_theta : diagonal diffusion (time/state-independent magnitude per dim)
      - decoder : maps latent to 1-D observation
    """
    def __init__(self, D, hidden=48):
        super().__init__()
        self.f_theta = MLP(D + 1, D, hidden=hidden)  # prior drift
        self.f_phi   = MLP(D + 1, D, hidden=hidden)  # posterior drift
        self.g_theta = DiagDiff(D)                   # diagonal diffusion (vector)
        self.decoder = nn.Sequential(nn.Linear(D, hidden), nn.ReLU(),
                                     nn.Linear(hidden, 1))
    def forward(self, z0, ts):
        """
        Simulate posterior SDE path using Euler scheme (no adjoint).
        This is used in the MC branch to keep fairness with ODE branch.
        """
        sde = PosteriorSDE(self.f_phi, self.g_theta)
        z = torchsde.sdeint(sde, z0, ts, method='euler')
        return z

# ------------------------------------------------------------
# ORDER-3 CUBATURE: plan -> omega_dot on solver grid -> ODEINT
# ------------------------------------------------------------
def build_cubature_deg3_plan(
    ts: torch.Tensor,   # [T]
    d: int,             # dimension
    k: int,             # macro steps
    gamma: float,
    P: int,             # number of cubature paths
    seed: int = 0,
):
    """
    Construct per-epoch plan for degree-3 cubature with nonuniform macro-grid.
    The solver grid ts is partitioned into k macro-steps using a power schedule.
    For each macro-step, we assign to each path P an axis/sign pair in round-robin.
    Returns a dict with everything needed to build ω̇(t) on the solver grid.
    """
    torch.manual_seed(seed)
    T = ts.numel()

    # Macro grid (denser near t=1 for gamma in (0,1])
    t_macro = torch.zeros(k + 1, dtype=ts.dtype, device=ts.device)
    for i in range(k + 1):
        t_macro[i] = 1.0 - (1.0 - i / k) ** gamma

    # Map each solver interval [ts[i], ts[i+1]) to macro-step index s
    step_of = torch.searchsorted(t_macro[1:], ts[:-1], right=False).clamp_(0, k - 1)  # [T-1]

    # Precompute axis and sign pools for children of degree-3 tree
    axes   = torch.arange(d, device=ts.device)
    signs2 = torch.tensor([-1.0, +1.0], device=ts.device)
    child_axes  = axes.repeat_interleave(2)  # [2d]
    child_signs = signs2.repeat(d)           # [2d]

    # Round-robin base pointer across P paths to stagger axis/sign selections
    offset  = int(torch.randint(0, 2 * d, (1,), device=ts.device).item())
    baseptr = (torch.arange(P, device=ts.device) + offset) % (2 * d)  # [P]

    return {
        "d": d, "k": k, "gamma": gamma, "P": P,
        "t_macro": t_macro,
        "step_of": step_of,
        "child_axes": child_axes,
        "child_signs": child_signs,
        "baseptr": baseptr,
    }

@torch.no_grad()
def omega_dot_from_plan(ts: torch.Tensor, plan: dict):
    """
    Build ω̇(t) sampled on solver grid times ts (shape [T]).
    For degree-3 cubature, on each solver subinterval inside a macro-step s:
        ω̇ = sign * sqrt(d * Δt_macro) / Δt_macro * e_axis
    Returns tensor of shape [P, T, D] on same device/dtype as ts.
    """
    T = ts.numel()
    d, k, P = plan["d"], plan["k"], plan["P"]
    t_macro   = plan["t_macro"]
    step_of   = plan["step_of"]
    child_axes  = plan["child_axes"]
    child_signs = plan["child_signs"]
    baseptr     = plan["baseptr"]

    od = torch.zeros(P, T, d, device=ts.device, dtype=ts.dtype)

    for s in range(k):
        # Indices of solver intervals belonging to macro step s
        idxs = torch.nonzero(step_of == s, as_tuple=False).flatten()
        if idxs.numel() == 0:
            continue
        t_s, t_sp1 = t_macro[s], t_macro[s + 1]
        dt_macro = (t_sp1 - t_s).clamp_min(1e-15)
        amp_over_dt = torch.sqrt(torch.tensor(float(d), device=ts.device, dtype=ts.dtype) * dt_macro) / dt_macro

        # Round-robin choose axis & sign per path for this macro step
        ptr_s = (baseptr + s * baseptr.numel()) % (2 * d)  # [P]
        j_s   = child_axes[ptr_s]                           # [P]
        s_s   = child_signs[ptr_s]                          # [P]

        # Fill ω̇ for all subinterval starts i in this macro step s (constant on each)
        val = (s_s * amp_over_dt).view(P)                   # [P]
        for i in idxs.tolist():
            od[torch.arange(P), i, j_s] = val

    # Copy the last defined ω̇ value to the terminal stamp (keeps shape consistent)
    if T >= 2:
        od[:, T-1, :] = od[:, T-2, :]
    return od  # [P,T,D]

class AugmentedDrift(nn.Module):
    """
    ODE drift for cubature branch:
        v̇ = f_φ(t, v) + diag(g_θ(t, v)) * ω̇(t),
    where ω̇ is pre-built on the solver grid and indexed by current time t.
    """
    def __init__(self, model: LatentSDE, omega_dot: torch.Tensor, ts: torch.Tensor):
        super().__init__()
        self.model = model
        self.register_buffer("omega_dot", omega_dot)  # [P,T,D]
        self.register_buffer("ts", ts)               # [T]
    def forward(self, t, y):  # y: [P,D]
        # Locate the nearest grid index k for the current scalar time t
        T = self.ts.numel()
        k = torch.searchsorted(self.ts, t.expand(1)).clamp(0, T-1)
        od = self.omega_dot.index_select(1, k).squeeze(1)   # [P,D]
        fphi  = self.model.f_phi(t, y)                      # [P,D]
        gdiag = self.model.g_theta(t, y)                    # [P,D]
        return fphi + gdiag * od

# ----------------------------
# One experiment: median per-epoch time/mem for MC vs Cubature
# ----------------------------
def run_experiment(
    T=500,
    M_MATCH=32,          # number of paths (both branches)
    D=1,                 # latent/state & Brownian dimension
    train_itr=30,
    batch_size=32,
    k_cubature=4,
    gamma=0.6,
    sigma_obs=0.1,
    lam_init=10.0,
    seed=SEED,
    hidden=48,
):
    """
    Run one full training experiment for given (T, D, ...), returning summary stats.

    Parameters
    ----------
    T : int
        Number of solver grid points in [0,1] (inclusive).
    M_MATCH : int
        Number of posterior paths in both branches (apples-to-apples).
    D : int
        Latent (and Brownian) dimension of the model.
    train_itr : int
        Number of optimization epochs.
    batch_size : int
        Batch size for MSE term (random matching of predicted vs true series).
    k_cubature : int
        Number of macro steps for degree-3 cubature plan.
    gamma : float
        Exponent for macro grid density (0<gamma<=1 gives denser near t=1).
    sigma_obs : float
        Observation noise std used in NLL scaling.
    lam_init : float
        Penalty weight aligning initial prediction with data mean at t=0.
    seed : int
        Base seed for reproducibility; cubature plan uses seed+epoch.
    hidden : int
        Hidden width for MLPs.

    Returns
    -------
    stats : dict
        Timing/memory medians and (if CUDA) gpu_max_alloc_mb.
    """
    ts = torch.linspace(0.0, 1.0, T, device=device)

    # Data once: simulate ground-truth trajectories y_true ~ SDE_Data
    B_data = max(M_MATCH, 64)
    y0_data = torch.full((B_data, 1), 0.1, device=device)
    sde_data = SDE_Data().to(device)
    with torch.no_grad():
        y_true = torchsde.sdeint(sde_data, y0_data, ts, method='midpoint')  # [T,B,1]

    # ----- MC branch: SDEINT (Euler) -----
    model_mc = LatentSDE(D, hidden=hidden).to(device)
    opt_mc = optim.Adam(model_mc.parameters(), lr=1e-3)
    y0_sim = torch.full((M_MATCH, D), 0.1, device=device)

    mc_times, mc_py_peaks = [], []
    for _ in range(train_itr):
        def step_mc():
            # Forward simulate posterior latent paths
            opt_mc.zero_grad()
            z = model_mc(y0_sim, ts)                   # torchsde.sdeint(..., method='euler')

            # KL-like control cost u = (f_φ - f_θ) / diag(g_θ)
            fφ = model_mc.f_phi(ts, z)
            fθ = model_mc.f_theta(ts, z)
            gV = model_mc.g_theta(ts, z)               # diag vector [T,M,D]
            u  = (fφ - fθ) / (gV + 1e-12)
            kl = 0.5 * (1.0 / T) * (u.pow(2).sum(dim=-1).mean(dim=1)).sum() / M_MATCH

            # Mini-batch MSE between decoded predictions and true data
            with torch.no_grad():
                pred_idx = torch.randint(0, M_MATCH, (batch_size,), device=device)
                true_idx = torch.randint(0, B_data,  (batch_size,), device=device)
                uniq_pred, inv = torch.unique(pred_idx, return_inverse=True)

            z_sel = z[:, uniq_pred, :]
            y_pred_sel = model_mc.decoder(z_sel)
            y_pred_batch = y_pred_sel[:, inv, :]
            y_true_batch = y_true[:, true_idx, :]
            nll = 0.5 * ((y_pred_batch - y_true_batch)**2).sum() / (T * batch_size * sigma_obs**2)

            # Encourage correct initial mean
            y0_pred = y_pred_sel[0].mean()
            y0_true = y_true[0].mean()
            init_penalty = lam_init * (y0_pred - y0_true).pow(2)

            loss = kl + nll + init_penalty
            loss.backward()
            opt_mc.step()
        dt, py_mb = _epoch_wrapper(step_mc)
        mc_times.append(dt); mc_py_peaks.append(py_mb)

    # ----- Cubature branch: ODEINT (Euler) with ω̇(t) -----
    model_cu = LatentSDE(D, hidden=hidden).to(device)
    opt_cu = optim.Adam(model_cu.parameters(), lr=1e-3)
    y0_batched = torch.full((M_MATCH, D), 0.1, device=device)
    w_tensor = torch.full((M_MATCH,), 1.0 / M_MATCH, device=device)  # uniform weights

    cu_times, cu_py_peaks = [], []
    cu_overheads = []  # per-epoch plan+omega_dot construction overhead (seconds)

    for epoch in range(train_itr):
        cu_overheads_epoch = []

        def step_cu():
            # Precompute degree-3 plan & ω̇(t) on the solver grid (overhead measured)
            opt_cu.zero_grad()
            t0_over = time.perf_counter()
            plan = build_cubature_deg3_plan(ts, d=D, k=k_cubature, gamma=gamma, P=M_MATCH, seed=seed + epoch)
            od   = omega_dot_from_plan(ts, plan)                    # [P,T,D]
            t1_over = time.perf_counter()
            cu_overheads_epoch.append(t1_over - t0_over)

            # ODE dynamics with augmented drift (posterior drift + diag(diff)*ω̇)
            drift = AugmentedDrift(model_cu, od, ts)
            v = odeint(drift, y0_batched, ts, method='euler')       # [T,P,D]

            # KL-like control term (importance weights w_tensor)
            fφ = model_cu.f_phi(ts, v)
            fθ = model_cu.f_theta(ts, v)
            gV = model_cu.g_theta(ts, v)                            # diag vector [T,P,D]
            u  = (fφ - fθ) / (gV + 1e-12)
            u2 = u.pow(2).sum(dim=-1)                               # [T,P]
            kl = 0.5 * (u2 * w_tensor.unsqueeze(0)).sum(dim=1).mean()

            # Mini-batch MSE using weighted sampling of predicted paths
            with torch.no_grad():
                pred_idx = torch.multinomial(w_tensor, batch_size, replacement=True)
                true_idx = torch.randint(0, y_true.shape[1], (batch_size,), device=device)
                uniq_pred, inv = torch.unique(pred_idx, return_inverse=True)

            v_sel = v[:, uniq_pred, :]
            y_pred_sel = model_cu.decoder(v_sel)
            y_pred_batch = y_pred_sel[:, inv, :]
            y_true_batch = y_true[:, true_idx, :]
            nll = 0.5 * ((y_pred_batch - y_true_batch)**2).sum() / (T * batch_size * sigma_obs**2)

            # Weighted initial alignment
            y0_pred = (y_pred_sel[0] * w_tensor[uniq_pred].unsqueeze(-1)).sum() / (w_tensor[uniq_pred].sum() + 1e-12)
            y0_true = y_true[0].mean()
            init_penalty = lam_init * (y0_pred - y0_true).pow(2)

            loss = kl + nll + init_penalty
            loss.backward()
            opt_cu.step()

        dt, py_mb = _epoch_wrapper(step_cu)
        cu_times.append(dt); cu_py_peaks.append(py_mb)
        if cu_overheads_epoch:
            cu_overheads.append(cu_overheads_epoch[-1])

    # Collect summary statistics (medians across epochs)
    stats = {
        "nsde_time_per_epoch": float(np.median(mc_times)),
        "cub_time_per_epoch":  float(np.median(cu_times)),
        "nsde_py_peak_mb":     float(np.median(mc_py_peaks)),
        "cub_py_peak_mb":      float(np.median(cu_py_peaks)),
        "cub_overhead_construct_s": float(np.median(cu_overheads)) if cu_overheads else 0.0,
    }
    if torch.cuda.is_available():
        stats["gpu_max_alloc_mb"] = torch.cuda.max_memory_allocated() / (1024 ** 2)
        torch.cuda.reset_peak_memory_stats()
    return stats

# ----------------------------
# Sweep over D and T & make CSV + plots
# ----------------------------
if __name__ == "__main__":
    # Default sweep hyperparameters
    M_MATCH     = 50
    train_itr   = 20
    batch_size  = 50
    k_cubature  = 4
    gamma       = 0.6
    hidden      = 48

    # Latent dimension grid & solver grid sizes to test
    D_grid = [4,8,16,32,64,128,256]
    T_grid = [1000]
    
    rows = []
    header = [
        "D","T","M",
        "MC_time_s","Cub_time_s","Speedup_MC_over_Cub",
        "MC_Python_peak_MB","Cub_Python_peak_MB","Ratio_MC_over_Cub",
        "Cub_overhead_construct_s"
    ]
    if torch.cuda.is_available():
        header += ["GPU_max_alloc_MB (last run)"]

    print(",".join(header))

    # Run experiments across grid and accumulate results in memory and stdout
    for T_fixed in T_grid:
        for D in D_grid:
            stats = run_experiment(
                T=T_fixed, M_MATCH=M_MATCH, D=D,
                train_itr=train_itr, batch_size=batch_size,
                k_cubature=k_cubature, gamma=gamma, hidden=hidden
            )
            mc_t   = stats["nsde_time_per_epoch"]
            cub_t  = stats["cub_time_per_epoch"]
            mc_mb  = stats["nsde_py_peak_mb"]
            cub_mb = stats["cub_py_peak_mb"]
            overhead = stats["cub_overhead_construct_s"]
            speed  = mc_t / max(cub_t, 1e-12)
            ratio  = mc_mb / max(cub_mb, 1e-12)

            row = [D, T_fixed, M_MATCH,
                   f"{mc_t:.6f}", f"{cub_t:.6f}", f"{speed:.2f}",
                   f"{mc_mb:.2f}", f"{cub_mb:.2f}", f"{ratio:.2f}",
                   f"{overhead:.4f}"]
            if torch.cuda.is_available():
                row.append(f"{stats['gpu_max_alloc_mb']:.2f}")

            print(",".join(map(str, row)))
            rows.append(row)

            # Be nice to the allocator between runs
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Save CSV with all rows
    csv_path = "timings_dim_table.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)
    print(f"\nSaved {csv_path}")

    # Make simple plots per T showing time, memory, and cubature overhead vs D
    for T_fixed in T_grid:
        # filter rows for this T
        sub = [r for r in rows if int(r[1]) == T_fixed]
        if not sub:
            continue
        D_arr = np.array([int(r[0]) for r in sub], dtype=int)
        mc_t  = np.array([float(r[3]) for r in sub])
        cu_t  = np.array([float(r[4]) for r in sub])
        mc_mb = np.array([float(r[6]) for r in sub])
        cu_mb = np.array([float(r[7]) for r in sub])
        overhead = np.array([float(r[9]) for r in sub])

        fig, axs = plt.subplots(1, 3, figsize=(15, 5.0))

        # Median wall time per epoch
        axs[0].plot(D_arr, mc_t, 'o-', label='MC (sdeint) per-epoch time')
        axs[0].plot(D_arr, cu_t, 's-', label='Cubature (odeint) per-epoch time')
        axs[0].set_xlabel('Dimension (D)')
        axs[0].set_ylabel('Median wall time per epoch (s)')
        axs[0].set_title(f"Time vs D (T={T_fixed})")
        axs[0].grid(True, ls='--', alpha=0.6)
        axs[0].legend()

        # Python heap peak MB
        axs[1].plot(D_arr, mc_mb, 'o-', label='MC Python peak MB')
        axs[1].plot(D_arr, cu_mb, 's-', label='Cubature Python peak MB')
        axs[1].set_xlabel('Dimension (D)')
        axs[1].set_ylabel('Median Python peak memory (MB)')
        axs[1].set_title(f"Python heap vs D (T={T_fixed})")
        axs[1].grid(True, ls='--', alpha=0.6)
        axs[1].legend()

        # Overhead to construct cubature plan + ω̇
        axs[2].plot(D_arr, overhead, 'd-', label='Cubature overhead (s)')
        axs[2].set_xlabel('Dimension (D)')
        axs[2].set_ylabel('Median overhead (s)')
        axs[2].set_title(f"Cubature plan+ω̇ overhead (T={T_fixed})")
        axs[2].grid(True, ls='--', alpha=0.6)
        axs[2].legend()

        fig.suptitle('NSDE training: MC (sdeint) vs Cubature (odeint, Euler)')
        plt.tight_layout()
        png_path = f"complexity_vs_dimension_T{T_fixed}.png"
        plt.savefig(png_path, dpi=150)
        plt.close(fig)
        print(f"Saved {png_path}")
