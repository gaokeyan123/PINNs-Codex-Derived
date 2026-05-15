"""
PDE residuals for the 2D axisymmetric sharp-interface pipe-solidification PINN.

The active Case B convention matches the Excel/Fluent setup:

    r = r_dim / r_w
    x = x_dim / r_w
    tau = t_dim * U / L_total
    u = u_dim / U
    p = p_dim / (rho_f * U**2)
    Theta = (T - T_wall) / (T_int - T_wall)

The dimensionless pipe length is A = L_total/r_w = cfg.L. Re_D and Pe_D are
diameter-based groups:

    Re_D = U * (2 r_w) / nu
    Pe_D = U * (2 r_w) / alpha_f

Network input normalization happens inside network.py only for conditioning.
Autograd residuals here are differentiated with respect to the external
physics coordinates r, x, and tau.
"""

import torch
from config import CaseConfig, Config


_R_MIN = 1e-6


def _grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Elementwise dy_i/dx_i for pointwise network outputs."""
    if not y.requires_grad:
        return torch.zeros_like(x)
    g = torch.autograd.grad(
        y,
        x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        allow_unused=True,
    )[0]
    return g if g is not None else torch.zeros_like(x)


def _laplacian_cyl(f: torch.Tensor,
                   r: torch.Tensor,
                   x: torch.Tensor) -> torch.Tensor:
    """Axisymmetric scalar/axial cylindrical Laplacian."""
    f_r = _grad(f, r)
    f_rr = _grad(f_r, r)
    f_x = _grad(f, x)
    f_xx = _grad(f_x, x)
    r_safe = torch.clamp(r, min=_R_MIN)
    return f_rr + f_r / r_safe + f_xx


def _poiseuille(r: torch.Tensor, r_int: torch.Tensor) -> torch.Tensor:
    """Fully developed laminar axial velocity with cross-sectional mean 1."""
    ri_safe = torch.clamp(r_int, min=_R_MIN)
    return 2.0 * (1.0 - (r / ri_safe) ** 2)


def _wall_temperature(x: torch.Tensor, cfg: CaseConfig) -> torch.Tensor:
    """Piecewise wall temperature for the upstream non-deposition section."""
    x_hot = torch.as_tensor(cfg.hot_wall_length, dtype=x.dtype, device=x.device)
    return torch.where(
        x <= x_hot,
        torch.full_like(x, cfg.Theta_in),
        torch.full_like(x, cfg.Theta_wall),
    )


def res_mass(out: dict,
             r: torch.Tensor,
             x: torch.Tensor,
             t: torch.Tensor) -> torch.Tensor:
    """Continuity: du_r/dr + u_r/r + du_x/dx = 0."""
    u_r = out["u_r"]
    u_x = out["u_x"]
    H = out["H"]
    r_safe = torch.clamp(r, min=_R_MIN)

    residual = _grad(u_r, r) + u_r / r_safe + _grad(u_x, x)
    return (1.0 - H) * residual


def res_mom_x(out: dict,
              r: torch.Tensor,
              x: torch.Tensor,
              t: torch.Tensor,
              cfg: CaseConfig) -> torch.Tensor:
    """
    Axial momentum:

        (1/A) du_x/dtau + u_r du_x/dr + u_x du_x/dx
        + dp/dx - (2/Re_D) lap_cyl(u_x) = 0.
    """
    u_r = out["u_r"]
    u_x = out["u_x"]
    p = out["p"]
    H = out["H"]
    A = cfg.L

    residual = (
        (1.0 / A) * _grad(u_x, t)
        + u_r * _grad(u_x, r)
        + u_x * _grad(u_x, x)
        + _grad(p, x)
        - (2.0 / cfg.Re_D) * _laplacian_cyl(u_x, r, x)
    )
    return (1.0 - H) * residual


def res_mom_r(out: dict,
              r: torch.Tensor,
              x: torch.Tensor,
              t: torch.Tensor,
              cfg: CaseConfig) -> torch.Tensor:
    """
    Radial momentum:

        (1/A) du_r/dtau + u_r du_r/dr + u_x du_r/dx
        + dp/dr - (2/Re_D) [lap_cyl(u_r) - u_r/r^2] = 0.
    """
    u_r = out["u_r"]
    u_x = out["u_x"]
    p = out["p"]
    H = out["H"]
    A = cfg.L
    r_safe = torch.clamp(r, min=_R_MIN)

    residual = (
        (1.0 / A) * _grad(u_r, t)
        + u_r * _grad(u_r, r)
        + u_x * _grad(u_r, x)
        + _grad(p, r)
        - (2.0 / cfg.Re_D) * (_laplacian_cyl(u_r, r, x) - u_r / r_safe ** 2)
    )
    return (1.0 - H) * residual


def res_energy_fluid(out: dict,
                     r: torch.Tensor,
                     x: torch.Tensor,
                     t: torch.Tensor,
                     cfg: CaseConfig) -> torch.Tensor:
    """
    Fluid energy:

        (1/A) dTheta_f/dtau + u_r dTheta_f/dr + u_x dTheta_f/dx
        - (2/Pe_D) lap_cyl(Theta_f) = 0.
    """
    u_r = out["u_r"]
    u_x = out["u_x"]
    Theta_f = out["Theta_f"]
    H = out["H"]
    A = cfg.L

    residual = (
        (1.0 / A) * _grad(Theta_f, t)
        + u_r * _grad(Theta_f, r)
        + u_x * _grad(Theta_f, x)
        - (2.0 / cfg.Pe_D) * _laplacian_cyl(Theta_f, r, x)
    )
    return (1.0 - H) * residual


def res_energy_dep(out: dict,
                   r: torch.Tensor,
                   x: torch.Tensor,
                   t: torch.Tensor,
                   cfg: CaseConfig) -> torch.Tensor:
    """
    Deposit energy:

        (1/A) dTheta_dep/dtau - (2*k_ratio/Pe_D) lap_cyl(Theta_dep) = 0.
    """
    Theta_dep = out["Theta_dep"]
    H = out["H"]
    A = cfg.L

    residual = (
        (1.0 / A) * _grad(Theta_dep, t)
        - (2.0 * cfg.k_ratio / cfg.Pe_D) * _laplacian_cyl(Theta_dep, r, x)
    )
    return H * residual


def res_stefan(out: dict,
               r: torch.Tensor,
               x: torch.Tensor,
               t: torch.Tensor,
               cfg: CaseConfig) -> torch.Tensor:
    """
    Stefan condition at r = r_int(x, tau):

        dr_int/dtau
        - (2*A/Pe_D)*Ste*(k_ratio*dTheta_dep/dr - dTheta_f/dr) = 0.
    """
    A = cfg.L
    flux_jump = cfg.k_ratio * _grad(out["Theta_dep"], r) - _grad(out["Theta_f"], r)
    return _grad(out["r_int"], t) - (2.0 * A / cfg.Pe_D) * cfg.Ste * flux_jump


def res_T_continuity(out: dict,
                     cfg: CaseConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Both phase temperatures equal the solidus temperature at the interface."""
    R7a = out["Theta_f"] - cfg.Theta_solidus
    R7b = out["Theta_dep"] - cfg.Theta_solidus
    return R7a, R7b


def res_interface_velocity_bc(out: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """No-slip velocity condition at the stationary deposit interface."""
    return out["u_x"], out["u_r"]


def res_rint_monotonic(out: dict, t: torch.Tensor) -> torch.Tensor:
    """Penalize interface radius growth in tau."""
    return torch.relu(_grad(out["r_int"], t))


def res_rint_x_monotonic(out: dict, x: torch.Tensor) -> torch.Tensor:
    """Penalize interface radius increases downstream in x."""
    return torch.relu(_grad(out["r_int"], x))


def res_wall_bc(out: dict,
                x: torch.Tensor,
                cfg: CaseConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Wall BC at r = r_w: wall temperature and no-slip velocity."""
    R_T = out["Theta_dep"] - _wall_temperature(x, cfg)
    R_ur = out["u_r"]
    R_ux = out["u_x"]
    return R_T, R_ur, R_ux


def res_inlet_bc(out: dict,
                 r: torch.Tensor,
                 cfg: CaseConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Inlet BC at x = 0: hot fluid, Poiseuille profile, zero radial flow."""
    r_int = out["r_int"]
    u_x_pois = _poiseuille(r, r_int)

    R_T = out["Theta_f"] - cfg.Theta_in
    R_ux = out["u_x"] - u_x_pois
    R_ur = out["u_r"]
    return R_T, R_ux, R_ur


def res_inlet_rint_bc(out: dict, cfg: CaseConfig) -> torch.Tensor:
    """Keep the inlet interface clean: r_int(x=0, tau) = r_w."""
    return out["r_int"] - cfg.r_w


def res_outlet_bc(out: dict,
                  x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Outlet BC at x = L: zero axial gradients for Theta_f and u_x."""
    dTf_dx = _grad(out["Theta_f"], x)
    dux_dx = _grad(out["u_x"], x)
    return dTf_dx, dux_dx


def res_axis_bc(out: dict,
                r: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Axis BC near r = 0: symmetry and zero radial velocity."""
    dTf_dr = _grad(out["Theta_f"], r)
    dux_dr = _grad(out["u_x"], r)
    R_ur = out["u_r"]
    return R_ur, dTf_dr, dux_dr


def res_ic(out: dict,
           r: torch.Tensor,
           cfg: CaseConfig) -> tuple[torch.Tensor, torch.Tensor,
                                     torch.Tensor, torch.Tensor]:
    """Initial condition at tau = 0: clean pipe, hot uniform fluid."""
    u_x_pois_ic = _poiseuille(r, torch.full_like(r, cfg.r_w))

    R_rint = out["r_int"] - cfg.r_int_ic
    R_Tf = out["Theta_f"] - cfg.Theta_in
    R_ux = out["u_x"] - u_x_pois_ic
    R_ur = out["u_r"]
    return R_rint, R_Tf, R_ux, R_ur


def compute_all_residuals(model,
                          batch: dict,
                          cfg: "Config") -> dict[str, torch.Tensor]:
    """Compute every residual tensor for one training step."""
    case = cfg.case
    eps = cfg.network.interface_epsilon

    def _fwd(pts: dict):
        r, x, t = pts["r"], pts["x"], pts["t"]
        out = model(r, x, t)
        out["H"] = model.smooth_heaviside(r, out["r_int"].detach(), eps)
        return out, r, x, t

    results: dict[str, torch.Tensor] = {}

    out, r, x, t = _fwd(batch["interior"])
    results["mass"] = res_mass(out, r, x, t)
    results["mom_x"] = res_mom_x(out, r, x, t, case)
    results["mom_r"] = res_mom_r(out, r, x, t, case)
    results["energy_fluid"] = res_energy_fluid(out, r, x, t, case)
    results["energy_dep"] = res_energy_dep(out, r, x, t, case)

    out, r, x, t = _fwd(batch["interface"])
    results["stefan"] = res_stefan(out, r, x, t, case)
    R7a, R7b = res_T_continuity(out, case)
    results["T_cont_fluid"] = R7a
    results["T_cont_dep"] = R7b
    R_ivx, R_ivr = res_interface_velocity_bc(out)
    results["interface_u_x"] = R_ivx
    results["interface_u_r"] = R_ivr
    results["rint_mono"] = res_rint_monotonic(out, t)
    results["rint_x_mono"] = res_rint_x_monotonic(out, x)

    out, r, x, t = _fwd(batch["wall"])
    R_wT, R_wur, R_wux = res_wall_bc(out, x, case)
    results["bc_wall_T"] = R_wT
    results["bc_wall_ur"] = R_wur
    results["bc_wall_ux"] = R_wux

    out, r, x, t = _fwd(batch["inlet"])
    R_iT, R_iux, R_iur = res_inlet_bc(out, r, case)
    results["bc_inlet_T"] = R_iT
    results["bc_inlet_ux"] = R_iux
    results["bc_inlet_ur"] = R_iur
    results["bc_inlet_rint"] = res_inlet_rint_bc(out, case)

    out, r, x, t = _fwd(batch["outlet"])
    R_oTf, R_oux = res_outlet_bc(out, x)
    results["bc_outlet_Tf"] = R_oTf
    results["bc_outlet_ux"] = R_oux

    out, r, x, t = _fwd(batch["axis"])
    R_aur, R_aTf, R_aux = res_axis_bc(out, r)
    results["bc_axis_ur"] = R_aur
    results["bc_axis_Tf"] = R_aTf
    results["bc_axis_ux"] = R_aux

    out, r, x, t = _fwd(batch["ic"])
    R_ic_rint, R_ic_Tf, R_ic_ux, R_ic_ur = res_ic(out, r, case)
    results["ic_rint"] = R_ic_rint
    results["ic_Tf"] = R_ic_Tf
    results["ic_ux"] = R_ic_ux
    results["ic_ur"] = R_ic_ur

    return results


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
             else torch.rand(N) * cfg.case.tau_end).requires_grad_(True)
        return {"r": r, "x": x, "t": t}

    batch = {
        "interior": _make_pts(64),
        "interface": _make_pts(32),
        "wall": _make_pts(16, r_val=cfg.case.r_w),
        "inlet": _make_pts(16, x_val=0.0),
        "outlet": _make_pts(16, x_val=cfg.case.L),
        "axis": _make_pts(16, r_val=0.01),
        "ic": _make_pts(32, t_val=0.0),
    }

    residuals = compute_all_residuals(model, batch, cfg)

    print(f"{'Residual':<22}  {'shape':>10}  {'mean|R|':>12}  {'has NaN':>8}")
    print("-" * 60)
    for name, residual in residuals.items():
        has_nan = "YES" if torch.isnan(residual).any() else "no"
        print(
            f"{name:<22}  {str(tuple(residual.shape)):>10}  "
            f"{residual.abs().mean().item():>12.4e}  {has_nan:>8}"
        )
    print("\n[PASS] All residuals computed without error.")
