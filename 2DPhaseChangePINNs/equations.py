"""
equations.py — All PDE residuals for 2D axisymmetric pipe solidification.

Each residual function receives:
  - out  : dict returned by model.forward() + an "H" key (smooth Heaviside)
  - r, x, t : input tensors with requires_grad=True
  - cfg  : CaseConfig (physics parameters)

Returns a residual tensor of shape [N].  The caller (losses.py) squares and
means these to form scalar loss terms.

Non-dimensional variables
─────────────────────────
  r̂ = r / r_w        x̂ = x / r_w        t̂ = t · α_f / r_w²
  û = u / U_in        p̂ = p / (ρ_f U_in²)
  Θ = (T − T_w) / (T_in − T_w)  ∈ [0, 1]
  r̂_int ∈ (0, 1]    (interface position, non-dim)

Non-dimensional groups used
────────────────────────────
  Pe  = ρ_f Cp_f U_in r_w / k_f        (Péclet)
  Re  = U_in r_w / ν                    (Reynolds, radius-based)
  Ste = Cp_f (T_in − T_w) / L_f        (Stefan)
  k_ratio = k_dep / k_f                 (conductivity ratio)
"""

import torch
from config import CaseConfig, Config

# Minimum r value — prevents 1/r singularity at the cylinder axis
_R_MIN = 1e-6


# ═══════════════════════════════════════════════════════════════════════════
# 2.1  Derivative utilities
# ═══════════════════════════════════════════════════════════════════════════

def _grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Element-wise ∂y_i/∂x_i for a batch via reverse-mode autograd.

    Uses grad_outputs=ones so the VJP gives per-element derivatives when each
    y_i depends only on x_i (true for a point-wise MLP with no batch-norm).
    create_graph=True keeps the graph alive for computing higher-order derivatives.
    allow_unused=True returns zeros instead of raising when x does not appear
    in the computation graph (e.g. interface head masked to (x,t) so ∂/∂r = 0).
    """
    g = torch.autograd.grad(
        y, x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        allow_unused=True,
    )[0]
    return g if g is not None else torch.zeros_like(x)


def _laplacian_cyl(f: torch.Tensor,
                   r: torch.Tensor,
                   x: torch.Tensor) -> torch.Tensor:
    """Cylindrical Laplacian  ∇²_cyl f = ∂²f/∂r² + (1/r)∂f/∂r + ∂²f/∂x².

    The 1/r term is singular at r = 0; r is clamped to _R_MIN.
    Both first derivatives use create_graph=True so second derivatives work.
    """
    f_r  = _grad(f, r)                        # ∂f/∂r    [N]
    f_rr = _grad(f_r, r)                       # ∂²f/∂r²  [N]
    f_x  = _grad(f, x)                         # ∂f/∂x    [N]
    f_xx = _grad(f_x, x)                       # ∂²f/∂x²  [N]
    r_safe = torch.clamp(r, min=_R_MIN)
    return f_rr + f_r / r_safe + f_xx


def _poiseuille(r: torch.Tensor, r_int: torch.Tensor) -> torch.Tensor:
    """Fully-developed laminar (Poiseuille) axial velocity profile.

    û_x(r) = 2 · (1 − r² / r_int²)   with mean velocity normalised to 1.
    Clamps r_int to avoid division by zero if the interface collapses.
    """
    ri_safe = torch.clamp(r_int, min=_R_MIN)
    return 2.0 * (1.0 - (r / ri_safe) ** 2)


# ═══════════════════════════════════════════════════════════════════════════
# 2.2  Interior residuals  R1 – R5   (evaluated on 𝒫_Ω)
# ═══════════════════════════════════════════════════════════════════════════

def res_mass(out: dict,
             r: torch.Tensor,
             x: torch.Tensor,
             t: torch.Tensor) -> torch.Tensor:
    """R1 — Incompressible continuity in cylindrical coordinates.

    Equation:
        ∂û_r/∂r̂ + û_r / r̂ + ∂û_x/∂x̂ = 0

    Weighted by (1 − H_ε):  active only in the fluid subdomain.
    """
    u_r = out["u_r"]
    u_x = out["u_x"]
    H   = out["H"]
    r_safe = torch.clamp(r, min=_R_MIN)

    du_r_dr = _grad(u_r, r)
    du_x_dx = _grad(u_x, x)

    residual = du_r_dr + u_r / r_safe + du_x_dx
    return (1.0 - H) * residual


def res_mom_x(out: dict,
              r: torch.Tensor,
              x: torch.Tensor,
              t: torch.Tensor,
              cfg: CaseConfig) -> torch.Tensor:
    """R2 — Axial (x) momentum equation.

    Equation (non-dim, thermal time scale t̂ = t α_f / r_w²):
        (1/Pe) ∂û_x/∂t̂  +  û_r ∂û_x/∂r̂  +  û_x ∂û_x/∂x̂
            = −∂p̂/∂x̂  +  (1/Re) ∇²_cyl û_x

    The 1/Pe prefactor on the unsteady term arises because time is scaled by
    the thermal diffusion time, not the convective time (Pe = U_in r_w / α_f).
    For Pe = 1 the two time scales coincide and the factor vanishes.

    Weighted by (1 − H_ε):  fluid subdomain only.
    """
    u_r = out["u_r"]
    u_x = out["u_x"]
    p   = out["p"]
    H   = out["H"]

    du_x_dt = _grad(u_x, t)
    du_x_dr = _grad(u_x, r)
    du_x_dx = _grad(u_x, x)
    dp_dx   = _grad(p,   x)
    lap_ux  = _laplacian_cyl(u_x, r, x)

    residual = (
        (1.0 / cfg.Pe) * du_x_dt
        + u_r * du_x_dr
        + u_x * du_x_dx
        + dp_dx
        - (1.0 / cfg.Re) * lap_ux
    )
    return (1.0 - H) * residual


def res_mom_r(out: dict,
              r: torch.Tensor,
              x: torch.Tensor,
              t: torch.Tensor,
              cfg: CaseConfig) -> torch.Tensor:
    """R3 — Radial (r) momentum equation.

    Equation:
        (1/Pe) ∂û_r/∂t̂  +  û_r ∂û_r/∂r̂  +  û_x ∂û_r/∂x̂
            = −∂p̂/∂r̂  +  (1/Re) (∇²_cyl û_r − û_r / r̂²)

    The geometric source term −û_r/r̂² has no Cartesian analogue; it arises
    from the axisymmetric form of the viscous stress tensor.

    Weighted by (1 − H_ε):  fluid subdomain only.
    """
    u_r = out["u_r"]
    u_x = out["u_x"]
    p   = out["p"]
    H   = out["H"]
    r_safe = torch.clamp(r, min=_R_MIN)

    du_r_dt = _grad(u_r, t)
    du_r_dr = _grad(u_r, r)
    du_r_dx = _grad(u_r, x)
    dp_dr   = _grad(p,   r)
    lap_ur  = _laplacian_cyl(u_r, r, x)

    residual = (
        (1.0 / cfg.Pe) * du_r_dt
        + u_r * du_r_dr
        + u_x * du_r_dx
        + dp_dr
        - (1.0 / cfg.Re) * (lap_ur - u_r / r_safe ** 2)
    )
    return (1.0 - H) * residual


def res_energy_fluid(out: dict,
                     r: torch.Tensor,
                     x: torch.Tensor,
                     t: torch.Tensor,
                     cfg: CaseConfig) -> torch.Tensor:
    """R4 — Fluid energy equation (advection–diffusion).

    Equation:
        ∂Θ_f/∂t̂  +  Pe (û_r ∂Θ_f/∂r̂  +  û_x ∂Θ_f/∂x̂)  =  ∇²_cyl Θ_f

    Weighted by (1 − H_ε):  fluid subdomain only.
    """
    u_r     = out["u_r"]
    u_x     = out["u_x"]
    Theta_f = out["Theta_f"]
    H       = out["H"]

    dTf_dt = _grad(Theta_f, t)
    dTf_dr = _grad(Theta_f, r)
    dTf_dx = _grad(Theta_f, x)
    lap_Tf = _laplacian_cyl(Theta_f, r, x)

    residual = dTf_dt + cfg.Pe * (u_r * dTf_dr + u_x * dTf_dx) - lap_Tf
    return (1.0 - H) * residual


def res_energy_dep(out: dict,
                   r: torch.Tensor,
                   x: torch.Tensor,
                   t: torch.Tensor,
                   cfg: CaseConfig) -> torch.Tensor:
    """R5 — Deposit (solid) energy equation (pure diffusion, no advection).

    Equation:
        ∂Θ_dep/∂t̂  =  k_ratio · ∇²_cyl Θ_dep

    k_ratio = k_dep / k_f serves as the effective diffusivity ratio when
    ρ·Cp is equal in both phases (simple test case).

    Weighted by H_ε:  deposit subdomain only.
    """
    Theta_dep = out["Theta_dep"]
    H         = out["H"]

    dTd_dt = _grad(Theta_dep, t)
    lap_Td = _laplacian_cyl(Theta_dep, r, x)

    residual = dTd_dt - cfg.k_ratio * lap_Td
    return H * residual


# ═══════════════════════════════════════════════════════════════════════════
# 2.3  Interface residuals  R6 – R7   (evaluated on 𝒫_Γ, r ≈ r_int)
# ═══════════════════════════════════════════════════════════════════════════

def res_stefan(out: dict,
               r: torch.Tensor,
               x: torch.Tensor,
               t: torch.Tensor,
               cfg: CaseConfig) -> torch.Tensor:
    """R6 — Stefan condition (latent-heat balance at the moving interface).

    Equation  (non-dim, derived from ρ_dep L_f ṙ_int = flux jump):
        Ste · ∂r̂_int/∂t̂  =  k_ratio · ∂Θ_dep/∂r̂|_Γ  −  ∂Θ_f/∂r̂|_Γ

    ∂r̂_int/∂t̂ is the autograd gradient of the network's interface-head output
    w.r.t. the input t̂.  Since the interface head is masked to (x̂, t̂), this
    gives ṙ_int correctly without any r dependency.

    Evaluated at interface collocation points where input r = r_int(x,t).
    """
    r_int     = out["r_int"]
    Theta_f   = out["Theta_f"]
    Theta_dep = out["Theta_dep"]

    drint_dt  = _grad(r_int,     t)   # ∂r_int/∂t̂
    dTf_dr    = _grad(Theta_f,   r)   # ∂Θ_f/∂r̂   at the interface
    dTdep_dr  = _grad(Theta_dep, r)   # ∂Θ_dep/∂r̂ at the interface

    # Correct nondimensional scaling: smaller Ste slows the front.
    flux_jump = cfg.k_ratio * dTdep_dr - dTf_dr
    residual = drint_dt - cfg.Ste * flux_jump
    return residual


def res_T_continuity(out: dict,
                     cfg: CaseConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """R7 — Temperature continuity and solidus condition at the interface.

    Both phases must equal the solidification temperature at Γ:
        Θ_f  (r_int, x, t) = Θ_solidus
        Θ_dep(r_int, x, t) = Θ_solidus

    Returns (R7a, R7b) as separate tensors so each can be weighted individually
    in the loss function if needed.
    """
    R7a = out["Theta_f"]   - cfg.Theta_solidus
    R7b = out["Theta_dep"] - cfg.Theta_solidus
    return R7a, R7b


def res_rint_monotonic(out: dict, t: torch.Tensor) -> torch.Tensor:
    """Penalize interface radius growth in time.

    Solidification should move the interface inward, so dr_int/dt <= 0.
    Positive values indicate local melting or an optimizer shortcut.
    """
    drint_dt = _grad(out["r_int"], t)
    return torch.relu(drint_dt)


def res_rint_smoothness(out: dict, x: torch.Tensor) -> torch.Tensor:
    """Penalize sharp axial curvature in the learned interface."""
    drint_dx = _grad(out["r_int"], x)
    return _grad(drint_dx, x)


def res_rint_x_monotonic(out: dict, x: torch.Tensor) -> torch.Tensor:
    """Penalize interface radius increases downstream.

    For the simple cold-wall/hot-inlet case, deposit thickness should not
    decrease along the pipe, so r_int should be non-increasing with x.
    """
    drint_dx = _grad(out["r_int"], x)
    return torch.relu(drint_dx)


# ═══════════════════════════════════════════════════════════════════════════
# 2.4  Boundary-condition and initial-condition residuals
# ═══════════════════════════════════════════════════════════════════════════

def res_wall_bc(out: dict,
                cfg: CaseConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Wall BC at r̂ = r̂_w = 1  (cold, no-slip).

    Θ_dep = Θ_wall  (Dirichlet cold wall)
    û_r   = 0       (no-slip, radial)
    û_x   = 0       (no-slip, axial)
    """
    R_T  = out["Theta_dep"] - cfg.Theta_wall
    R_ur = out["u_r"]
    R_ux = out["u_x"]
    return R_T, R_ur, R_ux


def res_inlet_bc(out: dict,
                 r: torch.Tensor,
                 cfg: CaseConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Inlet BC at x̂ = 0  (hot fluid, Poiseuille profile).

    Θ_f = Θ_in               (hot inlet temperature)
    û_x = Poiseuille(r, r_int) (fully-developed laminar profile)
    û_r = 0                   (no radial inflow)

    The Poiseuille profile uses the predicted r_int at x=0.  Since the
    inlet is in the non-deposition region, r_int ≈ r_w throughout training.
    """
    r_int    = out["r_int"]
    u_x_pois = _poiseuille(r, r_int)

    R_T  = out["Theta_f"] - cfg.Theta_in
    R_ux = out["u_x"]     - u_x_pois
    R_ur = out["u_r"]
    return R_T, R_ux, R_ur


def res_inlet_rint_bc(out: dict, cfg: CaseConfig) -> torch.Tensor:
    """Keep the inlet interface clean: r_int(x=0,t) = r_w."""
    return out["r_int"] - cfg.r_w


def res_outlet_bc(out: dict,
                  x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Outlet BC at x̂ = L/r_w  (zero axial gradient — outflow condition).

    ∂Θ_f/∂x̂ = 0
    ∂û_x/∂x̂ = 0
    """
    dTf_dx = _grad(out["Theta_f"], x)
    dux_dx = _grad(out["u_x"],     x)
    return dTf_dx, dux_dx


def res_axis_bc(out: dict,
                r: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Symmetry (axis) BC at r̂ = 0  (axisymmetry implies zero radial gradient).

    û_r        = 0    (no flow through axis)
    ∂Θ_f/∂r̂   = 0    (symmetry)
    ∂û_x/∂r̂   = 0    (symmetry)
    """
    dTf_dr = _grad(out["Theta_f"], r)
    dux_dr = _grad(out["u_x"],     r)
    R_ur   = out["u_r"]
    return R_ur, dTf_dr, dux_dr


def res_ic(out: dict,
           r: torch.Tensor,
           cfg: CaseConfig) -> tuple[torch.Tensor, torch.Tensor,
                                     torch.Tensor, torch.Tensor]:
    """Initial condition at t̂ = 0  (clean pipe, hot uniform fluid).

    r̂_int = r̂_w   (no deposit at t = 0)
    Θ_f   = Θ_in   (fluid at inlet temperature throughout)
    û_x   = Poiseuille(r, r_w)
    û_r   = 0
    """
    u_x_pois_ic = _poiseuille(r, torch.full_like(r, cfg.r_w))

    R_rint = out["r_int"]   - cfg.r_int_ic
    R_Tf   = out["Theta_f"] - cfg.Theta_in
    R_ux   = out["u_x"]     - u_x_pois_ic
    R_ur   = out["u_r"]
    return R_rint, R_Tf, R_ux, R_ur


# ═══════════════════════════════════════════════════════════════════════════
# 2.5  Master function
# ═══════════════════════════════════════════════════════════════════════════

def compute_all_residuals(model,
                          batch: dict,
                          cfg: "Config") -> dict[str, torch.Tensor]:
    """Compute every residual for one training step.

    Parameters
    ----------
    model : PINNSolidification
    batch : dict with keys:
        "interior"  — P_Ω  (r, x, t), all interior collocation points
        "interface" — P_Γ  (r, x, t), points on the predicted interface
        "wall"      — r = r_w  boundary points
        "inlet"     — x = 0   boundary points
        "outlet"    — x = L   boundary points
        "axis"      — r = 0   boundary points
        "ic"        — t = 0   initial-condition points
      Each sub-dict must have keys "r", "x", "t" as 1-D tensors
      with requires_grad=True.
    cfg : top-level Config (uses cfg.case for physics, cfg.network.interface_epsilon)

    Returns
    -------
    dict mapping residual name → flat tensor [N].
    losses.py will call .pow(2).mean() on each to produce scalar loss terms.
    """
    case = cfg.case
    eps  = cfg.network.interface_epsilon

    def _fwd(pts: dict):
        """Forward pass + attach smooth Heaviside indicator H."""
        r, x, t = pts["r"], pts["x"], pts["t"]
        out = model(r, x, t)
        out["H"] = model.smooth_heaviside(r, out["r_int"].detach(), eps)
        return out, r, x, t

    results: dict[str, torch.Tensor] = {}

    # ── Interior residuals ─────────────────────────────────────────────────
    out, r, x, t = _fwd(batch["interior"])
    results["mass"]         = res_mass(out, r, x, t)
    results["mom_x"]        = res_mom_x(out, r, x, t, case)
    results["mom_r"]        = res_mom_r(out, r, x, t, case)
    results["energy_fluid"] = res_energy_fluid(out, r, x, t, case)
    results["energy_dep"]   = res_energy_dep(out, r, x, t, case)

    # ── Interface residuals ────────────────────────────────────────────────
    out, r, x, t = _fwd(batch["interface"])
    results["stefan"]       = res_stefan(out, r, x, t, case)
    R7a, R7b = res_T_continuity(out, case)
    results["T_cont_fluid"] = R7a
    results["T_cont_dep"]   = R7b
    results["rint_mono"]    = res_rint_monotonic(out, t)
    results["rint_x_mono"]  = res_rint_x_monotonic(out, x)
    results["rint_smooth"]  = res_rint_smoothness(out, x)

    # ── Wall BC ───────────────────────────────────────────────────────────
    out, r, x, t = _fwd(batch["wall"])
    R_wT, R_wur, R_wux = res_wall_bc(out, case)
    results["bc_wall_T"]    = R_wT
    results["bc_wall_ur"]   = R_wur
    results["bc_wall_ux"]   = R_wux

    # ── Inlet BC ──────────────────────────────────────────────────────────
    out, r, x, t = _fwd(batch["inlet"])
    R_iT, R_iux, R_iur = res_inlet_bc(out, r, case)
    results["bc_inlet_T"]   = R_iT
    results["bc_inlet_ux"]  = R_iux
    results["bc_inlet_ur"]  = R_iur
    results["bc_inlet_rint"] = res_inlet_rint_bc(out, case)

    # ── Outlet BC ─────────────────────────────────────────────────────────
    out, r, x, t = _fwd(batch["outlet"])
    R_oTf, R_oux = res_outlet_bc(out, x)
    results["bc_outlet_Tf"] = R_oTf
    results["bc_outlet_ux"] = R_oux

    # ── Axis BC ───────────────────────────────────────────────────────────
    out, r, x, t = _fwd(batch["axis"])
    R_aur, R_aTf, R_aux = res_axis_bc(out, r)
    results["bc_axis_ur"]   = R_aur
    results["bc_axis_Tf"]   = R_aTf
    results["bc_axis_ux"]   = R_aux

    # ── Initial condition ─────────────────────────────────────────────────
    out, r, x, t = _fwd(batch["ic"])
    R_ic_rint, R_ic_Tf, R_ic_ux, R_ic_ur = res_ic(out, r, case)
    results["ic_rint"] = R_ic_rint
    results["ic_Tf"]   = R_ic_Tf
    results["ic_ux"]   = R_ic_ux
    results["ic_ur"]   = R_ic_ur

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Quick sanity check
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))

    from config import cfg
    from network import PINNSolidification

    torch.manual_seed(0)
    model = PINNSolidification(cfg.network, cfg.case, seed=0)
    model.eval()

    def _make_pts(N: int, r_val=None, x_val=None, t_val=None):
        r = (torch.full((N,), r_val) if r_val is not None
             else torch.rand(N)).requires_grad_(True)
        x = (torch.full((N,), x_val) if x_val is not None
             else torch.rand(N) * cfg.case.L).requires_grad_(True)
        t = (torch.full((N,), t_val) if t_val is not None
             else torch.rand(N) * cfg.case.t_end).requires_grad_(True)
        return {"r": r, "x": x, "t": t}

    batch = {
        "interior":  _make_pts(64),
        "interface": _make_pts(32),
        "wall":      _make_pts(16, r_val=cfg.case.r_w),
        "inlet":     _make_pts(16, x_val=0.0),
        "outlet":    _make_pts(16, x_val=cfg.case.L),
        "axis":      _make_pts(16, r_val=0.01),   # small r, not exactly 0
        "ic":        _make_pts(32, t_val=0.0),
    }

    res = compute_all_residuals(model, batch, cfg)

    print(f"{'Residual':<22}  {'shape':>10}  {'mean|R|':>12}  {'has NaN':>8}")
    print("─" * 60)
    for name, R in res.items():
        has_nan = "YES ⚠" if torch.isnan(R).any() else "no"
        print(f"  {name:<20}  {str(tuple(R.shape)):>10}  "
              f"{R.abs().mean().item():>12.4e}  {has_nan:>8}")
    print("\n[PASS] All residuals computed without error.")
