"""
sampling.py — Collocation point generation for PINN training.

Seven point sets are produced, matching the keys expected by
equations.compute_all_residuals():

  interior  : Latin Hypercube in (r, x, t)     — covers both subdomains
  interface : Adaptive — r set to r̂_int(x,t)  — chases the moving interface
  wall      : r = r_w  (outer cold wall)
  inlet     : x = 0    (pipe inlet)
  outlet    : x = L    (pipe outlet)
  axis      : r ≈ 0    (symmetry axis, see note below)
  ic        : t = 0    (initial condition)

All tensors are returned flat [N] with requires_grad=True so that
equations._grad() can differentiate through them.

Axis note:
  r is set to 1e-4 (not exactly 0). This is well above the _R_MIN=1e-6
  clamp in _laplacian_cyl, keeps BC gradients finite, and accurately
  represents the symmetry condition for all practical purposes.

Design decisions logged in PROJECT_STATE.md:
  - Pure-PyTorch LHS (no scipy) for portability
  - sample_interface uses torch.no_grad() + .detach() to freeze
    interface coordinates — prevents stale graph references across iters
  - Per-function seeds derived from a single base seed for reproducibility
"""

import torch
import matplotlib
matplotlib.use("Agg")          # non-interactive backend safe for headless runs
import matplotlib.pyplot as plt
from pathlib import Path

from config import Config

# r coordinate used for all axis-BC points
_AXIS_R = 1e-4


# ═══════════════════════════════════════════════════════════════════════════
# 3.1  Latin Hypercube utility
# ═══════════════════════════════════════════════════════════════════════════

def _lhs(N: int, d: int, seed: int = 0) -> torch.Tensor:
    """Latin Hypercube Sample in [0, 1]^d, shape [N, d].

    Each of the d dimensions is divided into N equal strata of width 1/N.
    Exactly one sample is drawn uniformly within each stratum, and the
    strata are independently permuted across dimensions.  This guarantees
    full coverage of the marginal distribution of every input axis, which
    dramatically reduces clustering compared to plain uniform random sampling.

    No external dependencies — pure PyTorch.
    """
    gen = torch.Generator()
    gen.manual_seed(seed)

    # Stratum left edges: 0, 1/N, 2/N, …, (N-1)/N
    edges = torch.arange(N, dtype=torch.float32) / N   # [N]

    out = torch.empty(N, d)
    for i in range(d):
        perm    = torch.randperm(N, generator=gen)          # random stratum order
        offsets = torch.rand(N, generator=gen) / N          # position within stratum
        out[:, i] = edges[perm] + offsets

    return out  # [N, d], values in (0, 1)


def _pack(r: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> dict:
    """Return a point dict with requires_grad=True on all entries."""
    return {
        "r": r.detach().requires_grad_(True),
        "x": x.detach().requires_grad_(True),
        "t": t.detach().requires_grad_(True),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 3.1  Interior sampler
# ═══════════════════════════════════════════════════════════════════════════

def sample_interior(N: int, cfg: Config, seed: int = 0) -> dict:
    """Latin Hypercube points in the full space-time domain (r, x, t).

    Domain:  r ∈ [0, r_w],  x ∈ [0, L],  t ∈ [0, t_end].

    Points intentionally span BOTH fluid and deposit subdomains.
    The smooth Heaviside H_ε in equations.py selects which PDE residual
    is active at each point — no hard domain splitting is needed here.
    """
    s = _lhs(N, 3, seed)                    # [N, 3] in (0,1)
    r_min = max(0.0, float(cfg.sampling.r_min_interior))
    if r_min >= cfg.case.r_w:
        raise ValueError(
            f"cfg.sampling.r_min_interior must be smaller than r_w={cfg.case.r_w}"
        )
    r = r_min + s[:, 0] * (cfg.case.r_w - r_min)
    x = s[:, 1] * cfg.case.L
    t = s[:, 2] * cfg.case.t_end
    return _pack(r, x, t)


# ═══════════════════════════════════════════════════════════════════════════
# 3.2  Boundary samplers
# ═══════════════════════════════════════════════════════════════════════════

def sample_wall(N: int, cfg: Config, seed: int = 0) -> dict:
    """Outer wall points: r = r̂_w = 1, random (x, t) via LHS."""
    s = _lhs(N, 2, seed)
    r = torch.full((N,), cfg.case.r_w)
    x = s[:, 0] * cfg.case.L
    t = s[:, 1] * cfg.case.t_end
    return _pack(r, x, t)


def sample_inlet(N: int, cfg: Config, seed: int = 0) -> dict:
    """Inlet points: x = 0, random (r, t) via LHS.

    r spans the full cross-section [0, r_w] so the inlet Poiseuille
    profile and temperature BC are enforced at all radii.
    """
    s = _lhs(N, 2, seed)
    r = s[:, 0] * cfg.case.r_w
    x = torch.zeros(N)
    t = s[:, 1] * cfg.case.t_end
    return _pack(r, x, t)


def sample_outlet(N: int, cfg: Config, seed: int = 0) -> dict:
    """Outlet points: x = L, random (r, t) via LHS."""
    s = _lhs(N, 2, seed)
    r = s[:, 0] * cfg.case.r_w
    x = torch.full((N,), cfg.case.L)
    t = s[:, 1] * cfg.case.t_end
    return _pack(r, x, t)


def sample_axis(N: int, cfg: Config, seed: int = 0) -> dict:
    """Symmetry axis points: r = _AXIS_R ≈ 0, random (x, t) via LHS.

    r is set to 1e-4 rather than exactly 0 to avoid the 1/r singularity
    in _laplacian_cyl while still faithfully representing the axis BC.
    """
    s = _lhs(N, 2, seed)
    r = torch.full((N,), _AXIS_R)
    x = s[:, 0] * cfg.case.L
    t = s[:, 1] * cfg.case.t_end
    return _pack(r, x, t)


# ═══════════════════════════════════════════════════════════════════════════
# 3.3  Initial-condition sampler
# ═══════════════════════════════════════════════════════════════════════════

def sample_ic(N: int, cfg: Config, seed: int = 0) -> dict:
    """IC points: t = 0, random (r, x) via LHS."""
    s = _lhs(N, 2, seed)
    r = s[:, 0] * cfg.case.r_w
    x = s[:, 1] * cfg.case.L
    t = torch.zeros(N)
    return _pack(r, x, t)


# ═══════════════════════════════════════════════════════════════════════════
# 3.4  Adaptive interface sampler
# ═══════════════════════════════════════════════════════════════════════════

def sample_interface(model, N: int, cfg: Config, seed: int = 0) -> dict:
    """Adaptive interface points: r = r̂_int(x, t) from the current network.

    Algorithm
    ---------
    1. Draw N (x, t) pairs uniformly via LHS.
    2. Query the network's interface head under torch.no_grad() to obtain
       r_int values — these become the r-coordinate of each point.
    3. Detach and re-wrap with requires_grad=True so the point positions
       are frozen for this training step (preventing stale graph references).

    The interface head is masked to (x, t) only (network.py design), so
    passing r_dummy=0 does not affect the r_int prediction.

    Called once at startup and then every cfg.sampling.resample_every iters
    via resample_interface() to track the evolving interface shape.
    """
    device = next(model.parameters()).device
    s = _lhs(N, 2, seed).to(device)
    x_samp = s[:, 0] * cfg.case.L
    t_samp = s[:, 1] * cfg.case.t_end

    # TODO: Test a staged differentiable T_cont path later.  Current interface
    # sampling intentionally freezes r_int coordinates, so T_cont trains the
    # temperature fields at the sampled interface but does not directly update
    # the interface head in the same backward pass.
    with torch.no_grad():
        r_dummy = torch.zeros(N, device=device)
        out      = model(r_dummy, x_samp, t_samp)
        r_int    = out["r_int"].detach().clamp(0.0, cfg.case.r_w)

    return _pack(r_int.clone(), x_samp.clone(), t_samp.clone())


def resample_interface(model, N: int, cfg: Config, seed: int = 0) -> dict:
    """Convenience wrapper — identical to sample_interface.

    Kept as a separate function so training code clearly signals that a
    resample step is happening (improves readability in train.py).
    """
    return sample_interface(model, N, cfg, seed)


# ═══════════════════════════════════════════════════════════════════════════
# 3.5  Full batch builder
# ═══════════════════════════════════════════════════════════════════════════

def build_batch(model, cfg: Config, seed: int = 0) -> dict:
    """Assemble the complete batch dict for one training step.

    Keys match those expected by equations.compute_all_residuals().
    Each point set uses a distinct derived seed for statistical independence.

    Parameters
    ----------
    model : PINNSolidification (used only for interface head query)
    cfg   : full Config object
    seed  : base seed; sub-seeds are seed+0 … seed+6

    Returns
    -------
    dict with keys: interior, interface, wall, inlet, outlet, axis, ic
    """
    sc = cfg.sampling
    return {
        "interior":  sample_interior(sc.N_interior,  cfg, seed + 0),
        "interface": sample_interface(model, sc.N_interface, cfg, seed + 1),
        "wall":      sample_wall(sc.N_boundary,      cfg, seed + 2),
        "inlet":     sample_inlet(sc.N_boundary,     cfg, seed + 3),
        "outlet":    sample_outlet(sc.N_boundary,    cfg, seed + 4),
        "axis":      sample_axis(sc.N_boundary,      cfg, seed + 5),
        "ic":        sample_ic(sc.N_ic,              cfg, seed + 6),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 3.6  Visualisation helper  (Phase 3 milestone)
# ═══════════════════════════════════════════════════════════════════════════

_POINT_STYLES = {
    "interior":  dict(c="silver",   s=2,  alpha=0.25, label="Interior Ω",   zorder=1),
    "interface": dict(c="crimson",  s=12, alpha=0.80, label="Interface Γ",  zorder=5),
    "wall":      dict(c="navy",     s=8,  alpha=0.70, label="Wall",          zorder=4),
    "inlet":     dict(c="green",    s=8,  alpha=0.70, label="Inlet",         zorder=4),
    "outlet":    dict(c="darkorange", s=8, alpha=0.70, label="Outlet",       zorder=4),
    "axis":      dict(c="purple",   s=8,  alpha=0.70, label="Axis",          zorder=4),
    "ic":        dict(c="teal",     s=6,  alpha=0.55, label="IC (t̂=0)",     zorder=3),
}


def plot_batch(batch: dict, cfg: Config,
               save_path: str | Path | None = None) -> plt.Figure:
    """Scatter all collocation points in the (x̂, r̂) plane (time collapsed).

    Interior points are shown as a faint grey cloud; boundary/IC/interface
    points are drawn on top.  Red interface points should form a coherent
    curve tracing r̂_int(x̂) — if they appear scattered randomly across r̂,
    the interface head has not yet converged.

    Parameters
    ----------
    batch     : dict from build_batch()
    cfg       : Config (used for axis limits)
    save_path : if given, save the figure to this path; otherwise display

    Returns
    -------
    matplotlib Figure (caller can close or further annotate)
    """
    fig, ax = plt.subplots(figsize=(11, 4))

    for key, style in _POINT_STYLES.items():
        if key not in batch:
            continue
        pts = batch[key]
        x_np = pts["x"].detach().cpu().numpy()
        r_np = pts["r"].detach().cpu().numpy()
        ax.scatter(x_np, r_np, **style)

    # Annotate domain boundaries
    ax.axhline(cfg.case.r_w, color="black", lw=1.2, ls="--",
               label=r"$\hat{r}_w = 1$", zorder=6)
    ax.axhline(0.0, color="black", lw=0.6, ls=":", zorder=6)

    ax.set_xlabel(r"$\hat{x}$",  fontsize=12)
    ax.set_ylabel(r"$\hat{r}$",  fontsize=12)
    ax.set_xlim(-0.1, cfg.case.L + 0.1)
    ax.set_ylim(-0.05, cfg.case.r_w * 1.12)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.set_title("Collocation point distribution  (all $\\hat{t}$ overlaid)",
                 fontsize=11)

    # Point-count annotation
    counts = {k: len(v["r"]) for k, v in batch.items() if k in _POINT_STYLES}
    info   = "  ".join(f"{k}: {n}" for k, n in counts.items())
    fig.text(0.01, 0.01, info, fontsize=7, color="gray",
             transform=fig.transFigure)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[sampling] figure saved → {save_path}")

    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Quick smoke-run
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    from config import cfg
    from network import PINNSolidification

    torch.manual_seed(0)
    model = PINNSolidification(cfg.network, cfg.case, seed=0)
    model.eval()

    print("Building batch …")
    batch = build_batch(model, cfg, seed=0)

    print(f"\n{'Point set':<12}  {'N':>7}  {'r range':>18}  {'x range':>18}  {'t range':>18}")
    print("─" * 80)
    for key, pts in batch.items():
        r = pts["r"].detach()
        x = pts["x"].detach()
        t = pts["t"].detach()
        print(f"  {key:<10}  {len(r):>7}  "
              f"[{r.min():.3f}, {r.max():.3f}]  "
              f"[{x.min():.3f}, {x.max():.3f}]  "
              f"[{t.min():.3f}, {t.max():.3f}]")

    # Milestone plot
    out_dir = Path(__file__).parent / "outputs"
    out_dir.mkdir(exist_ok=True)
    fig = plot_batch(batch, cfg, save_path=out_dir / "collocation_points.png")
    plt.close(fig)
    print("\n[PASS] Batch built and figure saved to outputs/collocation_points.png")
