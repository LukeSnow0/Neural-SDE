#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 14:34:23 2025

@author: lukesnow
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# NSDE Training: MC (stochastic) vs Cubature (deterministic ω̇) — Repeats & Plots
#
# What this script does
# ---------------------
# • Builds a power-law ("gamma") macro-grid for order-3 cubature and a native
#   uniform grid on [0,1]; samples degree-3 cubature paths on the fly (no tree).
# • Trains the same latent SDE model in two modes:
#     (A) MC branch: simulate via torchsde.sdeint (Euler) with diagonal noise.
#     (B) Cubature branch: sample ω paths p(t) -> piecewise-constant ω̇(t),
#         then integrate an ODE with drift f_φ + diag(g_θ)·ω̇ using odeint.
# • Repeats runs to gather mean ± 1σ bands for loss, time, and NN-call counts.
# • Plots:
#     - Variational loss vs cumulative NN forward calls
#     - Variational loss vs iteration
#     - Variational loss vs cumulative wall-clock (log-x)
# • Also produces one-off snapshot grids comparing trajectories before/mid/after
#   training for MC and cubature (first marginal only).
#
# Notes
# -----
# • No code behavior was changed — only comments/docstrings were added.
# • NFETracker/NNCounter track solver calls and NN forward calls for diagnostics.
# • The "data SDE" produces synthetic trajectories with time-varying drift/diff.
# • The latent model has learned drifts f_θ, f_φ and learned diagonal diffusion g_θ.
# -----------------------------------------------------------------------------


import math, time, gc, argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.optim as optim
import torchsde
from torchdiffeq import odeint  # non-adjoint

torch.set_default_dtype(torch.float32)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =======================
# Simple counters
# =======================
class NFETracker:
    """Tracks number of function evaluations (ODE and SDE f/g calls)."""
    def __init__(self): self.reset()
    def reset(self): self.ode = 0; self.sde_f = 0; self.sde_g = 0

class NNCounter:
    """Tracks forward passes through f_φ, f_θ, and g_θ networks."""
    def __init__(self): self.reset()
    def reset(self): self.f_phi = 0; self.f_theta = 0; self.g_theta = 0
    @property
    def total(self): return self.f_phi + self.f_theta + self.g_theta

# ============================================================
# Cubature grid (order-3) + on-the-fly sampled paths (no blowup)
# ============================================================
def build_cubature_grid_o3(k: int, base_steps_per_unit: int = 200, gamma: float = 0.6):
    """
    Construct a power-law macro grid with k steps, then expand to a native
    uniform grid in [0,1] with ~base_steps_per_unit points per unit time.

    Returns
    -------
      ts         : torch.Tensor [Tc] uniform grid on [0,1] (DEVICE)
      dt_list    : list[float] of macro step sizes (len k)
      dtn_list   : list[int] native sub-steps per macro step (len k)
      Tc         : int total native grid length (= sum dtn_list)
    """
    assert k >= 1
    t_d = np.array([1 - (1 - i/k)**gamma for i in range(k+1)], dtype=np.float64)
    dt_list, dtn_list = [], []
    for i in range(k):
        dt  = float(t_d[i+1] - t_d[i])
        dtn = max(2, int(round(dt * base_steps_per_unit)))
        dt_list.append(dt)
        dtn_list.append(dtn)
    Tc = int(np.sum(dtn_list))
    ts = torch.linspace(0.0, 1.0, Tc, device=DEVICE)
    return ts, dt_list, dtn_list, Tc

@torch.no_grad()
def sample_cubature_paths_o3_J(J: int, D: int, dt_list, dtn_list, Tc: int, device=DEVICE, dtype=torch.float32):
    """
    Sample J degree-3 cubature paths in D dimensions on the native grid.

    Idea
    ----
    • Order-3 1D atoms are ±1; lift them across dims with independent signs per
      macro step and scale by sqrt(Δt) to create piecewise-constant increments.

    Returns
    -------
    p : torch.Tensor [J, Tc, D]
        Piecewise-constant path values on the native grid.
    """
    p = torch.zeros((J, Tc, D), device=device, dtype=dtype)
    cur = 0
    for dt, dtn in zip(dt_list, dtn_list):
        signs = (torch.randint(0, 2, (J, D), device=device, dtype=torch.int64) * 2 - 1).to(dtype)
        jump  = signs * math.sqrt(dt)                                # [J,D]
        prev  = p[:, cur-1, :] if cur > 0 else torch.zeros_like(jump)
        seg   = (prev + jump).unsqueeze(1).expand(J, dtn, D)         # [J,dtn,D]
        p[:, cur:cur+dtn, :] = seg
        cur += dtn
    return p

def omega_dot_from_paths(p: torch.Tensor, ts: torch.Tensor):
    """
    Differentiate piecewise-constant paths p on the uniform native grid to get ω̇.

    Parameters
    ----------
    p  : [J, Tc, D] cubature paths (piecewise-constant)
    ts : [Tc]       uniform native grid

    Returns
    -------
    od : [J, Tc, D] piecewise-constant ω̇ aligned with ts (padded at front)
    """
    dts = (ts[1:] - ts[:-1]).view(1, -1, 1)              # [1,Tc-1,1]
    od  = (p[:, 1:, :] - p[:, :-1, :]) / dts             # [J,Tc-1,D]
    od  = torch.cat([od[:, :1, :], od], dim=1)           # pad front -> [J,Tc,D]
    return od

# ============================================================
# Data SDE (D-dim, diagonal noise)
# ============================================================
class SDE_Data(nn.Module):
    """
    Simple synthetic Stratonovich SDE with diagonal noise in D dims:
        dX = f(t) dt + g(t) ◦ dW_t,
    used to generate ground-truth training data (no state dependence).
    """
    noise_type = "diagonal"
    sde_type   = "stratonovich"
    def __init__(self, D: int):
        super().__init__(); self.D = D
    def f(self, t, x):
        B = x.size(0)
        drift = 1.0 + 0.5 * torch.sin(2 * torch.pi * t)
        return drift.expand(B, self.D)
    def g(self, t, x):
        B = x.size(0)
        diff = 0.1 + 0.05 * torch.cos(4 * torch.pi * t)
        return diff.expand(B, self.D)  # diagonal entries

# ============================================================
# Latent model + diagonal diffusion
# ============================================================
class MLP(nn.Module):
    """Small MLP taking [t, y] and outputting a vector field value."""
    def __init__(self, in_dim, out_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, t, y):
        # Supports [B,D] or [T,B,D], injects time as a feature
        if y.dim() == 2:
            B, D = y.shape
            tt = t.expand(B,1) if t.dim()==0 else t.unsqueeze(-1)
            return self.net(torch.cat([tt, y], dim=-1))
        elif y.dim() == 3:
            T, B, D = y.shape
            tt = t[:,None].expand(T,B) if t.dim()==1 else t.expand(T,B)
            tt = tt.unsqueeze(-1)
            out = self.net(torch.cat([tt, y], dim=-1).reshape(-1, D+1))
            return out.view(T, B, -1)
        else: raise ValueError

class DiagDiff(nn.Module):
    """
    Learned diagonal diffusion:
      - diag_vals returns per-dimension std vector [*, D] for KL term etc.
      - forward returns full diagonal matrices [*, D, D] for general use.
    """
    def __init__(self, D: int):
        super().__init__(); assert D>=1
        self.log_std = nn.Parameter(torch.zeros(D))
    def diag_vals(self, t, y):
        std = self.log_std.exp()
        if y.dim()==2:
            B, D = y.shape; assert D==std.numel()
            return std.unsqueeze(0).expand(B, D)
        elif y.dim()==3:
            T, B, D = y.shape; assert D==std.numel()
            return std.unsqueeze(0).unsqueeze(0).expand(T,B,D)
        else: raise ValueError
    def forward(self, t, y):
        std = self.log_std.exp(); D = std.numel()
        mat = torch.diag(std)
        if y.dim()==2:
            B = y.shape[0]; return mat.unsqueeze(0).expand(B,D,D)
        elif y.dim()==3:
            T,B = y.shape[:2]; return mat.unsqueeze(0).unsqueeze(0).expand(T,B,D,D)
        else: raise ValueError

class PosteriorSDE(torchsde.SDEIto):
    """
    Wrapper for posterior SDE used by torchsde.sdeint.
    Increments NFE/NN counters for profiling.
    """
    def __init__(self, f_phi, g_theta, nfe=None, nnc=None):
        super().__init__(noise_type="diagonal")
        self._f, self._g = f_phi, g_theta
        self._nfe, self._nnc = nfe, nnc
    def f(self, t, y):
        if self._nfe is not None: self._nfe.sde_f += 1
        if self._nnc is not None: self._nnc.f_phi += 1
        return self._f(t, y)
    def g(self, t, y):
        if self._nfe is not None: self._nfe.sde_g += 1
        if self._nnc is not None: self._nnc.g_theta += 1
        return self._g.diag_vals(t, y)  # [*,D]

class LatentSDE(nn.Module):
    """
    Latent SDE with learned prior drift f_θ, posterior drift f_φ, and diagonal g_θ.
    Decoder maps latent state to D-dim output (same dimensionality).
    """
    def __init__(self, D: int):
        super().__init__(); self.D = D
        self.f_theta = MLP(D+1, D)
        self.f_phi   = MLP(D+1, D)
        self.g_theta = DiagDiff(D)
        self.decoder = nn.Sequential(nn.Linear(D,64), nn.ReLU(), nn.Linear(64,D))
    def forward(self, z0, ts, nfe=None, nnc=None):
        sde = PosteriorSDE(self.f_phi, self.g_theta, nfe=nfe, nnc=nnc)
        return torchsde.sdeint(sde, z0, ts, method='euler')

# ============================================================
# Augmented cubature drift using sampled ω̇
# ============================================================
class AugmentedDrift(nn.Module):
    """
    ODE drift for cubature branch:
        v̇ = f_φ(t, v) + diag(g_θ(t, v)) * ω̇(t),
    with ω̇(t) pre-sampled on the native grid and indexed at runtime.
    """
    def __init__(self, model: LatentSDE, omega_dot: torch.Tensor, ts_native: torch.Tensor,
                 nfe=None, nnc=None):
        super().__init__()
        self.model = model
        self.register_buffer("omega_dot", omega_dot)  # [J,Tc,D]
        self.register_buffer("ts", ts_native)         # [Tc]
        self._nfe, self._nnc = nfe, nnc
    def forward(self, t, y):  # y:[J,D]
        if self._nfe is not None: self._nfe.ode += 1
        Tc = self.ts.numel()
        k = torch.searchsorted(self.ts, t.expand(1)).clamp(0, Tc-1)
        od = self.omega_dot.index_select(1, k).squeeze(1)      # [J,D]
        if self._nnc is not None: self._nnc.f_phi += 1; self._nnc.g_theta += 1
        fphi  = self.model.f_phi(t, y)                         # [J,D]
        gdiag = self.model.g_theta.diag_vals(t, y)             # [J,D]
        return fphi + gdiag * od

# ============================================================
# Losses
# ============================================================
def kl_mc(ts, fphi, ftheta, gdiag, B_sim):
    """Discrete-time KL-like control cost for MC branch."""
    dt = ts[1] - ts[0]
    u = (fphi - ftheta) / (gdiag + 1e-12)     # [T,B,D]
    return 0.5 * dt * (u.pow(2).sum(dim=-1).sum() / B_sim)

def nll_mc(y_pred_all, y_true, sigma_obs=0.1, batch_size=50):
    """
    Simple MSE-based NLL surrogate: randomly match predicted paths with data
    paths to avoid O(B_true × B_sim) pairing each step.
    """
    T, B_sim, D = y_pred_all.shape
    nll = 0.0
    for _ in range(batch_size):
        i = torch.randint(0, y_true.shape[1], (1,), device=y_true.device).item()
        j = torch.randint(0, B_sim, (1,), device=y_pred_all.device).item()
        nll += 0.5 * ((y_pred_all[:, j:j+1, :] - y_true[:, i:i+1, :])**2).sum() / (T * D * sigma_obs**2)
    return nll / batch_size

def kl_cub_sampled(ts, fphi_batch, ftheta_batch, gdiag_batch):
    """KL-like cost for cubature branch averaged over sampled paths J."""
    dt = ts[1] - ts[0]
    u2 = ((fphi_batch - ftheta_batch) / (gdiag_batch + 1e-12)).pow(2).sum(dim=-1)  # [Tc,J]
    return 0.5 * dt * u2.mean(dim=1).sum()

# ===============================================================
# Plot Helpers
# ===============================================================

@torch.no_grad()
def _sample_paths(model, ts, B_show=12):
    """Utility to render a few decoded trajectories from the model."""
    z0 = torch.full((B_show, model.D), 0.1, device=DEVICE)
    z  = model(z0, ts)                 # [T,B,D]
    y  = model.decoder(z)              # [T,B,D]
    return y

def _plot_snapshot_grid(ts, y_true, mc_snaps, cu_snaps, snap_epochs, B_plot=10):
    """
    Plot a 2×3 panel:
      (a/b/c) MC before/mid/after; (d/e/f) Cubature before/mid/after.
    Uses only the first output dimension for clarity.
    """
    import matplotlib.pyplot as plt
    ts = ts.detach().cpu().view(-1)
    y_true = y_true.detach().cpu()     # [T,B,D]

    labels = [
        (mc_snaps, "MC pre-training",          f"(a) Before MC training"),
        (mc_snaps, "MC mid training",          f"(b) After epoch {snap_epochs[1]} of MC training"),
        (mc_snaps, "MC trained",               f"(c) After epoch {snap_epochs[2]} of MC training"),
        (cu_snaps, "Cubature pre-training",    f"(d) Before Cubature training"),
        (cu_snaps, "Cubature mid training",    f"(e) After epoch {snap_epochs[1]} of Cubature training"),
        (cu_snaps, "Cubature trained",         f"(f) After epoch {snap_epochs[2]} of Cubature training"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True, sharey=True)
    for idx, ax in enumerate(axes.flat):
        snaps_dict, _, title = labels[idx]
        ep = snap_epochs[0 if idx%3==0 else (1 if idx%3==1 else 2)]
        y_hat = snaps_dict[ep].detach().cpu()   # [T,B,D]
        B = min(B_plot, y_true.shape[1], y_hat.shape[1])
        for i in range(B):
            ax.plot(ts, y_true[:, i, 0], alpha=0.45, lw=1.4, color="#1f77b4")      # observed data, blue
            ax.plot(ts, y_hat[:,  i, 0], "--", alpha=0.9, lw=1.2, color="#ff7f0e") # NSDE approx, orange dashed
        ax.set_title(title, fontsize=10)
        ax.grid(True, ls="--", lw=0.5)
    axes[1,0].set_xlabel("t"); axes[1,1].set_xlabel("t"); axes[1,2].set_xlabel("t")
    axes[0,0].set_ylabel("dim 0"); axes[1,0].set_ylabel("dim 0")
    fig.tight_layout()
    plt.show()
    
def train_mc_snaps(base_model, y_true, ts, train_itr=100, lr=1e-3, batch_size=50, B_sim=27, snap_epochs=(0,50,99)):
    """
    Train MC branch and collect decoded snapshots at selected epochs.
    Returns trained model and dict[epoch] -> decoded paths [T,B,D].
    """
    D = base_model.D
    model = LatentSDE(D=D).to(DEVICE); model.load_state_dict(base_model.state_dict())
    opt = optim.Adam(model.parameters(), lr=lr)
    z0 = torch.full((B_sim, D), 0.1, device=DEVICE)

    snaps = {}
    for ep in range(train_itr):
        opt.zero_grad()
        z = model(z0, ts)
        fphi   = model.f_phi(ts, z)
        ftheta = model.f_theta(ts, z)
        gdiag  = model.g_theta.diag_vals(ts, z)
        kl = kl_mc(ts, fphi, ftheta, gdiag, B_sim)
        y_pred_all = model.decoder(z)
        nll = nll_mc(y_pred_all, y_true, 0.1, batch_size)
        ic  = 10.0 * ((y_pred_all[0] - y_true[0,:B_sim])**2).mean()
        (kl+nll+ic).backward(); opt.step()
        if ep in snap_epochs:
            snaps[ep] = _sample_paths(model, ts, B_show=27)
    return model, snaps

def train_cubature_sampled_snaps(base_model, y_true, ts, dt_list, dtn_list, Tc,
                                 train_itr=100, lr=1e-3, batch_size=50, B_sim=27, snap_epochs=(0,50,99)):
    """
    Train cubature branch with J=B_sim sampled paths per epoch; save snapshots.
    Returns trained model and dict[epoch] -> decoded paths [T,B,D].
    """
    D = base_model.D
    model = LatentSDE(D=D).to(DEVICE); model.load_state_dict(base_model.state_dict())
    opt = optim.Adam(model.parameters(), lr=lr)

    Bn, J = y_true.shape[1], B_sim
    snaps = {}

    for ep in range(train_itr):
        opt.zero_grad()
        p_J  = sample_cubature_paths_o3_J(J, D, dt_list, dtn_list, Tc, device=DEVICE)
        od_J = omega_dot_from_paths(p_J, ts)
        y0_J = torch.full((J, D), 0.1, device=DEVICE)
        v = odeint(AugmentedDrift(model, od_J, ts), y0_J, ts, method='euler')
        fphi_J   = model.f_phi(ts, v)
        ftheta_J = model.f_theta(ts, v)
        gdiag_J  = model.g_theta.diag_vals(ts, v)
        kl = kl_cub_sampled(ts, fphi_J, ftheta_J, gdiag_J)
        y_pred_J  = model.decoder(v)
        idx_b     = torch.randint(0, Bn, (J,), device=DEVICE)
        y_true_J  = y_true[:, idx_b, :]
        nll = 0.5 * ((y_pred_J - y_true_J)**2).sum() / (ts.numel() * J * D * 0.1**2)
        ic  = 10.0 * ((y_pred_J[0] - y_true_J[0])**2).mean()
        (kl+nll+ic).backward(); opt.step()
        if ep in snap_epochs:
            snaps[ep] = _sample_paths(model, ts, B_show=27)
    return model, snaps

    
# ============================================================
# Training
# ============================================================
def init_base_model(seed: int, D: int):
    """Initialize a LatentSDE with Xavier init and fixed seed."""
    torch.manual_seed(seed); np.random.seed(seed)
    m = LatentSDE(D=D).to(DEVICE)
    for mod in m.modules():
        if isinstance(mod, nn.Linear):
            nn.init.xavier_uniform_(mod.weight)
            if mod.bias is not None: nn.init.zeros_(mod.bias)
    return m

def train_mc(base_model, y_true, ts, train_itr=100, lr=1e-3, batch_size=50, B_sim=27):
    """
    MC branch training loop. Tracks loss, wall time, and NN forward-call counts.
    Returns trained model and histories as numpy arrays.
    """
    D = base_model.D
    model = LatentSDE(D=D).to(DEVICE); model.load_state_dict(base_model.state_dict())
    opt = optim.Adam(model.parameters(), lr=lr)
    z0 = torch.full((B_sim, D), 0.1, device=DEVICE)

    nfe, nnc = NFETracker(), NNCounter()
    loss_hist, time_hist, nn_hist = [], [], []

    for _ in range(train_itr):
        nfe.reset(); nnc.reset()
        t0 = time.time(); opt.zero_grad()

        z = model(z0, ts, nfe=nfe, nnc=nnc)                     # [Tc,B_sim,D]
        nnc.f_phi   += 1; fphi   = model.f_phi(ts, z)
        nnc.f_theta += 1; ftheta = model.f_theta(ts, z)
        nnc.g_theta += 1; gdiag  = model.g_theta.diag_vals(ts, z)
        kl = kl_mc(ts, fphi, ftheta, gdiag, B_sim)

        y_pred_all = model.decoder(z)                           # [Tc,B_sim,D]
        nll = nll_mc(y_pred_all, y_true, 0.1, batch_size)

        lam = 10.0
        ic = lam * ((y_pred_all[0] - y_true[0,:B_sim])**2).mean()

        (kl+nll+ic).backward(); opt.step()

        time_hist.append(time.time() - t0)
        loss_hist.append(float((kl+nll+ic).detach().cpu()))
        nn_hist.append(nnc.total)

    return model, np.array(loss_hist), np.array(time_hist), np.array(nn_hist)

def train_cubature_sampled(base_model, y_true, ts, dt_list, dtn_list, Tc,
                           train_itr=100, lr=1e-3, batch_size=50, B_sim=27):
    """
    Cubature branch training loop. Each epoch samples J=B_sim cubature paths
    and integrates the augmented ODE. Returns model and histories.
    """
    D = base_model.D
    model = LatentSDE(D=D).to(DEVICE); model.load_state_dict(base_model.state_dict())
    opt = optim.Adam(model.parameters(), lr=lr)

    nfe, nnc = NFETracker(), NNCounter()
    loss_hist, time_hist, nn_hist = [], [], []

    Bn = y_true.shape[1]
    J  = B_sim

    for _ in range(train_itr):
        nfe.reset(); nnc.reset()
        t0 = time.time(); opt.zero_grad()

        # sample J paths and compute ω̇ on the native grid
        p_J  = sample_cubature_paths_o3_J(J, D, dt_list, dtn_list, Tc, device=DEVICE)
        od_J = omega_dot_from_paths(p_J, ts)                     # [J,Tc,D]

        y0_J = torch.full((J, D), 0.1, device=DEVICE)
        drift_J = AugmentedDrift(model, od_J, ts, nfe=nfe, nnc=nnc)

        v = odeint(drift_J, y0_J, ts, method='euler')            # [Tc,J,D]

        nnc.f_phi   += 1; fphi_J   = model.f_phi(ts, v)
        nnc.f_theta += 1; ftheta_J = model.f_theta(ts, v)
        nnc.g_theta += 1; gdiag_J  = model.g_theta.diag_vals(ts, v)
        kl = kl_cub_sampled(ts, fphi_J, ftheta_J, gdiag_J)

        y_pred_J = model.decoder(v)                               # [Tc,J,D]
        idx_b = torch.randint(0, Bn, (J,), device=DEVICE)
        y_true_J = y_true[:, idx_b, :]
        sigma_obs = 0.1
        nll = 0.5 * ((y_pred_J - y_true_J)**2).sum() / (Tc * J * D * sigma_obs**2)

        lam = 10.0
        ic  = lam * ((y_pred_J[0] - y_true_J[0])**2).mean()

        (kl+nll+ic).backward(); opt.step()

        time_hist.append(time.time() - t0)
        loss_hist.append(float((kl+nll+ic).detach().cpu()))
        nn_hist.append(nnc.total)

    return model, np.array(loss_hist), np.array(time_hist), np.array(nn_hist)

# ============================================================
# One run for a given D
# ============================================================
def run_once(seed=1234, D=2, k_cub=6, train_itr=120, base_steps_per_unit=200, gamma=0.6,
             B_sim=27, lr=1e-3, batch_size=50):
    """
    Single end-to-end experiment at fixed D:
      • Build grid, simulate synthetic data.
      • Train MC and cubature branches for 'train_itr' iterations.
      • Return histories for loss, time, and NN-call counts.
    """
    torch.manual_seed(seed); np.random.seed(seed)

    ts, dt_list, dtn_list, Tc = build_cubature_grid_o3(k=k_cub, base_steps_per_unit=base_steps_per_unit, gamma=gamma)

    # Generate synthetic data on the SAME grid
    y0 = torch.full((B_sim, D), 0.1, device=DEVICE)
    with torch.no_grad():
        y_true = torchsde.sdeint(SDE_Data(D).to(DEVICE), y0, ts, method='midpoint')  # [Tc,B_sim,D]

    base_model = init_base_model(seed, D=D)

    mc_model, mc_loss, mc_time, mc_nn = train_mc(
        base_model, y_true, ts, train_itr=train_itr, lr=lr, batch_size=batch_size, B_sim=B_sim
    )
    cu_model, cu_loss, cu_time, cu_nn = train_cubature_sampled(
        base_model, y_true, ts, dt_list, dtn_list, Tc,
        train_itr=train_itr, lr=lr, batch_size=batch_size, B_sim=B_sim
    )

    return {"mc_loss": mc_loss, "mc_time": mc_time, "mc_nn": mc_nn,
            "cu_loss": cu_loss, "cu_time": cu_time, "cu_nn": cu_nn}

# ============================================================
# Repeats: average & std bands
# ============================================================
def run_repeats(D, repeats, seed0, **kwargs):
    """
    Repeat 'run_once' with different seeds and aggregate mean/std curves.
    Returns a dict of numpy arrays for MC/Cubature loss/time/NN-call stats.
    """
    runs = []
    for r in range(repeats):
        s = seed0 + r
        out = run_once(seed=s, D=D, **kwargs)
        runs.append(out)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # stack per metric: shape [R, T]
    mc_loss = np.stack([o["mc_loss"] for o in runs], axis=0)
    cu_loss = np.stack([o["cu_loss"] for o in runs], axis=0)
    mc_time = np.stack([o["mc_time"] for o in runs], axis=0)
    cu_time = np.stack([o["cu_time"] for o in runs], axis=0)
    mc_nn   = np.stack([o["mc_nn"]   for o in runs], axis=0)
    cu_nn   = np.stack([o["cu_nn"]   for o in runs], axis=0)

    stats = {
        "mc_loss_mean": mc_loss.mean(0), "mc_loss_std": mc_loss.std(0),
        "cu_loss_mean": cu_loss.mean(0), "cu_loss_std": cu_loss.std(0),
        "mc_time_mean": mc_time.mean(0), "mc_time_std": mc_time.std(0),
        "cu_time_mean": cu_time.mean(0), "cu_time_std": cu_time.std(0),
        "mc_nn_mean":   mc_nn.mean(0),   "mc_nn_std":   mc_nn.std(0),
        "cu_nn_mean":   cu_nn.mean(0),   "cu_nn_std":   cu_nn.std(0),
    }
    return stats

# ============================================================
# Main: loop over D and plot smoothed curves with ±1σ shading
# ============================================================
ap = argparse.ArgumentParser()
ap.add_argument("--iters", type=int, default=200)
ap.add_argument("--k_cub", type=int, default=6)
ap.add_argument("--gamma", type=float, default=0.6)
ap.add_argument("--bspu",  type=int, default=200)
ap.add_argument("--Bsim",  type=int, default=27)
ap.add_argument("--lr",    type=float, default=1e-3)
ap.add_argument("--batch", type=int, default=50)
ap.add_argument("--dims",  type=str,  default="8,32,64,128")
ap.add_argument("--repeats", type=int, default=10, help="MC repeats per dimension")
args = ap.parse_args()

# Parse comma-separated list of dimensions to test
dims = [int(x) for x in args.dims.split(",") if x.strip()]
TRAIN_ITR = args.iters

# Run repeated experiments for each dimension and collect stats
results = {}
for D in dims:
    print(f"\n=== Running D={D} with {args.repeats} repeats ===")
    stats = run_repeats(
        D=D, repeats=args.repeats, seed0=1234,
        k_cub=args.k_cub, train_itr=TRAIN_ITR,
        base_steps_per_unit=args.bspu, gamma=args.gamma,
        B_sim=args.Bsim, lr=args.lr, batch_size=args.batch
    )
    results[D] = stats

# --------- Plot: loss vs cumulative NN-evals (mean ± 1σ) ---------
colors = plt.cm.tab10(np.linspace(0, 1, len(dims)))
# Override last two colors for distinction
colors[-2] = (0.0, 0.6, 0.0, 1.0)   # green (RGBA)
colors[-1] = (1.0, 0.5, 0.0, 1.0)   # orange (RGBA)

plt.figure(figsize=(9,5))
for color, D in zip(colors, dims):
    r = results[D]
    # Means
    x_cu = np.cumsum(r["cu_nn_mean"])
    y_cu = r["cu_loss_mean"]
    x_mc = np.cumsum(r["mc_nn_mean"])
    y_mc = r["mc_loss_mean"]
    # Curves
    plt.plot(x_cu, y_cu, '-',  color=color, label=f'Cub D={D}')
    plt.plot(x_mc, y_mc, '--', color=color, label=f'MC  D={D}')
    # Shaded ±1σ (project variance on Y only)
    plt.fill_between(x_cu, y_cu - r["cu_loss_std"], y_cu + r["cu_loss_std"], color=color, alpha=0.12)
    plt.fill_between(x_mc, y_mc - r["mc_loss_std"], y_mc + r["mc_loss_std"], color=color, alpha=0.12)
plt.xlabel("Cumulative NN forward calls")
plt.ylabel("Variational loss")
plt.title(f"Convergence vs NN-evals (mean ± 1σ over {args.repeats} runs)")
plt.grid(True, ls='--', lw=0.5)
plt.legend(ncols=2, fontsize=9)
plt.tight_layout(); plt.show()

# --------- Plot: loss vs epoch (mean ± 1σ) ---------
it = np.arange(TRAIN_ITR)
plt.figure(figsize=(9,5))
for color, D in zip(colors, dims):
    r = results[D]
    plt.plot(it, r["cu_loss_mean"], '-',  color=color, label=f'Cub D={D}')
    plt.plot(it, r["mc_loss_mean"], '--', color=color, label=f'MC  D={D}')
    plt.fill_between(it, r["cu_loss_mean"] - r["cu_loss_std"], r["cu_loss_mean"] + r["cu_loss_std"],
                     color=color, alpha=0.12)
    plt.fill_between(it, r["mc_loss_mean"] - r["mc_loss_std"], r["mc_loss_mean"] + r["mc_loss_std"],
                     color=color, alpha=0.12)
plt.xlabel("Training iteration"); plt.ylabel("Variational loss")
plt.title(f"Convergence vs iteration (mean ± 1σ over {args.repeats} runs)")
plt.grid(True, ls='--', lw=0.5)
plt.legend(ncols=2, fontsize=9)
plt.tight_layout(); plt.show()

# --------- Plot: loss vs wall-clock (mean ± 1σ on Y) ---------
plt.figure(figsize=(9,5))
for color, D in zip(colors, dims):
    r = results[D]
    t_cu = np.cumsum(r["cu_time_mean"])
    t_mc = np.cumsum(r["mc_time_mean"])
    plt.plot(t_cu, r["cu_loss_mean"], '-',  color=color, label=f'Cub D={D}')
    plt.plot(t_mc, r["mc_loss_mean"], '--', color=color, label=f'MC  D={D}')
    plt.fill_between(t_cu, r["cu_loss_mean"] - r["cu_loss_std"], r["cu_loss_mean"] + r["cu_loss_std"],
                     color=color, alpha=0.12)
    plt.fill_between(t_mc, r["mc_loss_mean"] - r["mc_loss_std"], r["mc_loss_mean"] + r["mc_loss_std"],
                     color=color, alpha=0.12)
plt.xlabel("Cumulative wall-clock (sec)"); plt.ylabel("Variational loss")
plt.title(f"Convergence vs compute time (mean ± 1σ over {args.repeats} runs)")
plt.xscale('log')
plt.grid(True, ls='--', lw=0.5)
plt.legend(ncols=2, fontsize=9)
plt.tight_layout(); plt.show()


# -------- One-off snapshot grid (first marginal only) --------
# choose a dimension to visualize (e.g., the first in your --dims list)
D_plot = int(dims[0]) if isinstance(dims, (list, tuple)) and len(dims) else 8
SNAP_EPOCHS = (0, min(50, TRAIN_ITR-1), TRAIN_ITR-1)

# Build the SAME grid you use for training & generate data once
ts, dt_list, dtn_list, Tc = build_cubature_grid_o3(k=args.k_cub, base_steps_per_unit=args.bspu, gamma=args.gamma)
y0 = torch.full((args.Bsim, D_plot), 0.1, device=DEVICE)
with torch.no_grad():
    y_true = torchsde.sdeint(SDE_Data(D_plot).to(DEVICE), y0, ts, method='midpoint')  # [T,B,D]

# init and train once (MC + Cubature) while saving snapshots
base_model = init_base_model(seed=1234, D=D_plot)
_, mc_snaps = train_mc_snaps(base_model, y_true, ts,
                             train_itr=TRAIN_ITR, lr=args.lr, batch_size=args.batch,
                             B_sim=args.Bsim, snap_epochs=SNAP_EPOCHS)
_, cu_snaps = train_cubature_sampled_snaps(base_model, y_true, ts, dt_list, dtn_list, Tc,
                                           train_itr=TRAIN_ITR, lr=args.lr, batch_size=args.batch,
                                           B_sim=args.Bsim, snap_epochs=SNAP_EPOCHS)

_plot_snapshot_grid(ts, y_true, mc_snaps, cu_snaps, SNAP_EPOCHS, B_plot=10)
