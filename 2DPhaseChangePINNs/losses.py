"""
losses.py — Weighted composite loss function for PINN training.

Architecture
────────────
  compute_loss(model, batch, cfg, physics_on)
      │
      ├─ physics_on=False  (Phase 1 BC/IC pre-train)
      │       └─ _compute_bc_ic_residuals()   ← first derivatives only, fast
      │
      └─ physics_on=True   (Phase 2 + 3 full training)
              └─ equations.compute_all_residuals()   ← includes Laplacians

Both paths feed into compute_loss_terms() → 8 named scalar MSE terms,
then total_loss() weights and sums them.

The 8 loss groups and their default weights (from config.LossWeights):
──────────────────────────────────────────────────────────────────────
  L_mass         w = 1     R1: incompressible continuity
  L_mom_x        w = 1     R2: axial momentum
  L_mom_r        w = 1     R3: radial momentum
  L_energy_fluid w = 1     R4: fluid energy (advection-diffusion)
  L_energy_dep   w = 1     R5: deposit energy (pure diffusion)
  L_stefan       w = 10    R6: Stefan condition (latent heat balance)
  L_T_cont       w = 10    R7: temperature continuity at interface
  L_bc_ic        w = 100   All 15 BC + IC residuals averaged

Rationale for weights:
  Physics terms are soft constraints — large residuals across the whole
  domain are expected early in training.  Stefan and T_continuity are
  harder constraints that must hold on a lower-dimensional surface (Γ).
  BC/IC are the hardest: they anchor the solution to known values at
  specific locations and must be tight before the PDE residuals can
  drive the interior solution.

Phase 1 fast path:
  During BC/IC pre-training (phase=1) we skip all second-derivative
  Laplacian computations (~10× cheaper per step).  Only the 15 BC/IC
  residuals are evaluated.  physics_on=False signals this path.
"""

import torch
from pathlib import Path
from config import Config, LossWeights
from equations import (
    compute_all_residuals,
    res_wall_bc, res_inlet_bc, res_outlet_bc, res_axis_bc, res_ic,
)


# ═══════════════════════════════════════════════════════════════════════════
# 4.1  Primitive helper
# ═══════════════════════════════════════════════════════════════════════════

def mse(residual: torch.Tensor) -> torch.Tensor:
    """Mean squared residual — the basic building block of every loss term."""
    return residual.pow(2).mean()


# ═══════════════════════════════════════════════════════════════════════════
# 4.3  Phase 1 fast path — BC / IC residuals only
# ═══════════════════════════════════════════════════════════════════════════

def _compute_bc_ic_residuals(model, batch: dict, cfg: Config) -> dict:
    """Compute only the 15 BC + IC residuals (no Laplacians, no Stefan).

    Used in Phase 1 pre-training.  Avoids all second-order autograd,
    cutting cost roughly 10× vs compute_all_residuals().

    Returns a dict with the same BC/IC keys as compute_all_residuals()
    so that compute_loss_terms() can be called identically in both phases.
    Physics keys (mass, mom_x, …, stefan, T_cont_*) are absent — the
    caller must pass physics_on=False to avoid KeyErrors.
    """
    case = cfg.case
    eps  = cfg.network.interface_epsilon
    results: dict[str, torch.Tensor] = {}

    def _fwd(pts: dict):
        r, x, t = pts["r"], pts["x"], pts["t"]
        out = model(r, x, t)
        out["H"] = model.smooth_heaviside(r, out["r_int"].detach(), eps)
        return out, r, x, t

    # Wall BC
    out, r, x, t = _fwd(batch["wall"])
    R_wT, R_wur, R_wux = res_wall_bc(out, case)
    results["bc_wall_T"]    = R_wT
    results["bc_wall_ur"]   = R_wur
    results["bc_wall_ux"]   = R_wux

    # Inlet BC
    out, r, x, t = _fwd(batch["inlet"])
    R_iT, R_iux, R_iur = res_inlet_bc(out, r, case)
    results["bc_inlet_T"]   = R_iT
    results["bc_inlet_ux"]  = R_iux
    results["bc_inlet_ur"]  = R_iur

    # Outlet BC
    out, r, x, t = _fwd(batch["outlet"])
    R_oTf, R_oux = res_outlet_bc(out, x)
    results["bc_outlet_Tf"] = R_oTf
    results["bc_outlet_ux"] = R_oux

    # Axis BC
    out, r, x, t = _fwd(batch["axis"])
    R_aur, R_aTf, R_aux = res_axis_bc(out, r)
    results["bc_axis_ur"]   = R_aur
    results["bc_axis_Tf"]   = R_aTf
    results["bc_axis_ux"]   = R_aux

    # Initial condition
    out, r, x, t = _fwd(batch["ic"])
    R_rint, R_Tf, R_ux, R_ur = res_ic(out, r, case)
    results["ic_rint"] = R_rint
    results["ic_Tf"]   = R_Tf
    results["ic_ux"]   = R_ux
    results["ic_ur"]   = R_ur

    return results


# ═══════════════════════════════════════════════════════════════════════════
# 4.1  Loss-term computation — 23 residuals → 8 named scalars
# ═══════════════════════════════════════════════════════════════════════════

# BC/IC residual keys — averaged together into a single L_bc_ic term
_BC_IC_KEYS = [
    "bc_wall_T",   "bc_wall_ur",   "bc_wall_ux",
    "bc_inlet_T",  "bc_inlet_ux",  "bc_inlet_ur",
    "bc_outlet_Tf","bc_outlet_ux",
    "bc_axis_ur",  "bc_axis_Tf",   "bc_axis_ux",
    "ic_rint",     "ic_Tf",        "ic_ux",        "ic_ur",
]


def compute_loss_terms(residuals: dict,
                       weights: LossWeights,
                       physics_on: bool = True) -> dict[str, torch.Tensor]:
    """Map the residual dict to 8 named, weighted scalar loss terms.

    Parameters
    ----------
    residuals   : output of compute_all_residuals() or _compute_bc_ic_residuals()
    weights     : LossWeights config (default: mass=1, stefan=10, bc_ic=100, …)
    physics_on  : if False, physics keys are absent from residuals — skip them

    Returns
    -------
    dict of 8 scalar tensors (still in the computation graph for backward()).
    Keys: mass, mom_x, mom_r, energy_fluid, energy_dep, stefan, T_cont, bc_ic.
    """
    terms: dict[str, torch.Tensor] = {}

    if physics_on:
        # ── Interior PDE residuals R1–R5 ────────────────────────────────
        terms["mass"]          = weights.mass         * mse(residuals["mass"])
        terms["mom_x"]         = weights.mom_x        * mse(residuals["mom_x"])
        terms["mom_r"]         = weights.mom_r        * mse(residuals["mom_r"])
        terms["energy_fluid"]  = weights.energy_fluid * mse(residuals["energy_fluid"])
        terms["energy_dep"]    = weights.energy_dep   * mse(residuals["energy_dep"])

        # ── Interface residuals R6–R7 ────────────────────────────────────
        terms["stefan"]  = weights.stefan * mse(residuals["stefan"])
        # R7 averages fluid and deposit continuity conditions equally
        terms["T_cont"]  = weights.T_continuity * (
            mse(residuals["T_cont_fluid"]) + mse(residuals["T_cont_dep"])
        ) * 0.5
    else:
        # Physics terms get zero scalar placeholders for logging consistency
        zero = torch.tensor(0.0)
        for k in ("mass", "mom_x", "mom_r", "energy_fluid", "energy_dep",
                  "stefan", "T_cont"):
            terms[k] = zero

    # ── BC + IC residuals (averaged over all 15 sub-terms) ───────────────
    present = [k for k in _BC_IC_KEYS if k in residuals]
    bc_ic_mean = sum(mse(residuals[k]) for k in present) / len(present)
    terms["bc_ic"] = weights.bc_ic * bc_ic_mean

    return terms


# ═══════════════════════════════════════════════════════════════════════════
# 4.2  Total loss aggregator
# ═══════════════════════════════════════════════════════════════════════════

def total_loss(terms: dict[str, torch.Tensor],
               physics_on: bool = True) -> torch.Tensor:
    """Sum weighted loss terms into a single backward-able scalar.

    When physics_on=False (Phase 1), only L_bc_ic contributes — the
    physics placeholder zeros are skipped to avoid poisoning the graph.
    """
    if physics_on:
        return sum(terms.values())
    else:
        return terms["bc_ic"]


# ═══════════════════════════════════════════════════════════════════════════
# 4.4  Master pipeline function
# ═══════════════════════════════════════════════════════════════════════════

def compute_loss(model,
                 batch: dict,
                 cfg: Config,
                 physics_on: bool = True
                 ) -> tuple[torch.Tensor, dict[str, float]]:
    """Full pipeline: batch → residuals → weighted loss terms → total.

    Parameters
    ----------
    model       : PINNSolidification
    batch       : from sampling.build_batch()
    cfg         : full Config
    physics_on  : False during Phase 1 pre-train (skips Laplacians)

    Returns
    -------
    loss   : scalar tensor with grad_fn (call .backward() on this)
    log    : dict[str, float] — detached values of all 9 terms
             (8 named terms + "total") for CSV / console logging.
             Values are the weighted contributions, not raw MSEs.
    """
    # Step 1 — compute residuals
    if physics_on:
        residuals = compute_all_residuals(model, batch, cfg)
    else:
        residuals = _compute_bc_ic_residuals(model, batch, cfg)

    # Step 2 — weighted scalar terms
    terms = compute_loss_terms(residuals, cfg.weights, physics_on)

    # Step 3 — total scalar
    loss = total_loss(terms, physics_on)

    # Step 4 — detached log dict (safe to store / print without holding graph)
    log: dict[str, float] = {k: v.item() for k, v in terms.items()}
    log["total"] = loss.item()

    return loss, log


# ═══════════════════════════════════════════════════════════════════════════
# 4.5  Training monitor formatter
# ═══════════════════════════════════════════════════════════════════════════

def format_loss_line(log: dict[str, float], iteration: int,
                     physics_on: bool = True) -> str:
    """Format a compact one-line summary for console / CSV logging.

    Example output (physics_on=True):
      iter  2500 | total 3.4521e-02 | bc_ic 2.8e-02 | phys 4.1e-03
                 | stefan 1.2e-04 | T_cont 9.8e-05

    Example output (physics_on=False, Phase 1):
      iter   500 | total 1.2345e-01 | bc_ic 1.2345e-01  [BC/IC only]
    """
    total_str  = f"total {log['total']:.4e}"
    bc_ic_str  = f"bc_ic {log['bc_ic']:.2e}"

    if physics_on:
        phys = (log.get("mass", 0) + log.get("mom_x", 0)
                + log.get("mom_r", 0) + log.get("energy_fluid", 0)
                + log.get("energy_dep", 0))
        phys_str   = f"phys {phys:.2e}"
        stefan_str = f"stefan {log.get('stefan', 0):.2e}"
        Tcont_str  = f"T_cont {log.get('T_cont', 0):.2e}"
        return (f"iter {iteration:5d} | {total_str} | {bc_ic_str} | "
                f"{phys_str} | {stefan_str} | {Tcont_str}")
    else:
        return (f"iter {iteration:5d} | {total_str} | {bc_ic_str}"
                f"  [BC/IC only]")


def log_to_csv(log: dict[str, float], iteration: int,
               csv_path: Path, write_header: bool = False) -> None:
    """Append one row to the loss history CSV file.

    Parameters
    ----------
    log         : detached loss dict from compute_loss()
    iteration   : current training iteration number
    csv_path    : Path to the CSV file (created if absent)
    write_header: write column names first (call with True on iter 0)
    """
    import csv
    fields = ["iteration"] + sorted(log.keys())
    row    = {"iteration": iteration, **log}

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ═══════════════════════════════════════════════════════════════════════════
# Quick sanity run
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    from config import cfg
    from network import PINNSolidification
    from sampling import build_batch

    torch.manual_seed(0)
    model = PINNSolidification(cfg.network, cfg.case, seed=0)
    model.eval()

    print("Building batch …")
    batch = build_batch(model, cfg, seed=0)

    # ── Phase 1 (BC/IC only) ─────────────────────────────────────────────
    print("\n── Phase 1 (physics_on=False) ──")
    loss1, log1 = compute_loss(model, batch, cfg, physics_on=False)
    print(format_loss_line(log1, 0, physics_on=False))
    assert loss1.requires_grad, "loss must be differentiable"
    loss1.backward()
    print("[PASS] Phase 1 backward() completed.")

    # ── Phase 2 (all terms) ──────────────────────────────────────────────
    print("\n── Phase 2 (physics_on=True) ──")
    # Reset gradients
    for p in model.parameters():
        p.grad = None
    batch = build_batch(model, cfg, seed=1)   # fresh requires_grad tensors
    loss2, log2 = compute_loss(model, batch, cfg, physics_on=True)
    print(format_loss_line(log2, 0, physics_on=True))
    assert loss2.requires_grad
    loss2.backward()

    # Check that at least one parameter received a gradient
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0, "No gradients flowed to model parameters"
    print("[PASS] Phase 2 backward() completed — gradients flow to all params.")

    # ── Summary table ────────────────────────────────────────────────────
    print("\nDetailed loss terms (Phase 2):")
    print(f"  {'Term':<16}  {'Value':>14}")
    print("  " + "─" * 32)
    for k, v in sorted(log2.items()):
        print(f"  {k:<16}  {v:>14.6e}")
