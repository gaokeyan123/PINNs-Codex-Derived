"""
PDE residuals for the active sharp-interface Case B PINN.

Autograd differentiates with respect to the external nondimensional
coordinates ``r``, ``x``, and ``tau``. Network input scaling is handled only
inside ``network.py`` for conditioning.
"""

import torch

from config import CaseConfig, Config

_R_MIN = 1e-6


def _grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Elementwise dy_i/dx_i for pointwise network outputs."""
    if not y.requires_grad:
        return torch.zeros_like(x)
    grad = torch.autograd.grad(
        y,
        x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        allow_unused=True,
    )[0]
    return grad if grad is not None else torch.zeros_like(x)


def _laplacian_cyl(f: torch.Tensor, r: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    f_r = _grad(f, r)
    f_rr = _grad(f_r, r)
    f_x = _grad(f, x)
    f_xx = _grad(f_x, x)
    r_safe = torch.clamp(r, min=_R_MIN)
    return f_rr + f_r / r_safe + f_xx


def _poiseuille(r: torch.Tensor, r_int: torch.Tensor) -> torch.Tensor:
    ri_safe = torch.clamp(r_int, min=_R_MIN)
    return 2.0 * (1.0 - (r / ri_safe) ** 2)


def _wall_temperature(x: torch.Tensor, cfg: CaseConfig) -> torch.Tensor:
    x_hot = torch.as_tensor(cfg.hot_wall_length, dtype=x.dtype, device=x.device)
    return torch.where(
        x <= x_hot,
        torch.full_like(x, cfg.Theta_in),
        torch.full_like(x, cfg.Theta_wall),
    )


def res_mass(
    out: dict[str, torch.Tensor],
    r: torch.Tensor,
    x: torch.Tensor,
    tau: torch.Tensor,
) -> torch.Tensor:
    del tau
    u_r = out["u_r"]
    u_x = out["u_x"]
    H = out["H"]
    r_safe = torch.clamp(r, min=_R_MIN)
    return (1.0 - H) * (_grad(u_r, r) + u_r / r_safe + _grad(u_x, x))


def res_mom_x(
    out: dict[str, torch.Tensor],
    r: torch.Tensor,
    x: torch.Tensor,
    tau: torch.Tensor,
    cfg: CaseConfig,
) -> torch.Tensor:
    u_r = out["u_r"]
    u_x = out["u_x"]
    p = out["p"]
    H = out["H"]
    A = cfg.L
    residual = (
        (1.0 / A) * _grad(u_x, tau)
        + u_r * _grad(u_x, r)
        + u_x * _grad(u_x, x)
        + _grad(p, x)
        - (2.0 / cfg.Re_D) * _laplacian_cyl(u_x, r, x)
    )
    return (1.0 - H) * residual


def res_mom_r(
    out: dict[str, torch.Tensor],
    r: torch.Tensor,
    x: torch.Tensor,
    tau: torch.Tensor,
    cfg: CaseConfig,
) -> torch.Tensor:
    u_r = out["u_r"]
    u_x = out["u_x"]
    p = out["p"]
    H = out["H"]
    A = cfg.L
    r_safe = torch.clamp(r, min=_R_MIN)
    residual = (
        (1.0 / A) * _grad(u_r, tau)
        + u_r * _grad(u_r, r)
        + u_x * _grad(u_r, x)
        + _grad(p, r)
        - (2.0 / cfg.Re_D) * (_laplacian_cyl(u_r, r, x) - u_r / r_safe**2)
    )
    return (1.0 - H) * residual


def res_energy_fluid(
    out: dict[str, torch.Tensor],
    r: torch.Tensor,
    x: torch.Tensor,
    tau: torch.Tensor,
    cfg: CaseConfig,
) -> torch.Tensor:
    u_r = out["u_r"]
    u_x = out["u_x"]
    theta_f = out["Theta_f"]
    H = out["H"]
    A = cfg.L
    residual = (
        (1.0 / A) * _grad(theta_f, tau)
        + u_r * _grad(theta_f, r)
        + u_x * _grad(theta_f, x)
        - (2.0 / cfg.Pe_D) * _laplacian_cyl(theta_f, r, x)
    )
    return (1.0 - H) * residual


def res_energy_dep(
    out: dict[str, torch.Tensor],
    r: torch.Tensor,
    x: torch.Tensor,
    tau: torch.Tensor,
    cfg: CaseConfig,
) -> torch.Tensor:
    theta_dep = out["Theta_dep"]
    H = out["H"]
    A = cfg.L
    residual = (
        (1.0 / A) * _grad(theta_dep, tau)
        - (2.0 * cfg.k_ratio / cfg.Pe_D) * _laplacian_cyl(theta_dep, r, x)
    )
    return H * residual


def res_stefan(
    out: dict[str, torch.Tensor],
    r: torch.Tensor,
    x: torch.Tensor,
    tau: torch.Tensor,
    cfg: CaseConfig,
) -> torch.Tensor:
    del x
    flux_jump = cfg.k_ratio * _grad(out["Theta_dep"], r) - _grad(out["Theta_f"], r)
    return _grad(out["r_int"], tau) - (2.0 * cfg.L / cfg.Pe_D) * cfg.Ste * flux_jump


def res_T_continuity(
    out: dict[str, torch.Tensor],
    cfg: CaseConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    return out["Theta_f"] - cfg.Theta_solidus, out["Theta_dep"] - cfg.Theta_solidus


def res_interface_velocity_bc(out: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    return out["u_x"], out["u_r"]


def res_rint_monotonic(out: dict[str, torch.Tensor], tau: torch.Tensor) -> torch.Tensor:
    return torch.relu(_grad(out["r_int"], tau))


def res_rint_x_monotonic(out: dict[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    return torch.relu(_grad(out["r_int"], x))


def res_wall_bc(
    out: dict[str, torch.Tensor],
    x: torch.Tensor,
    cfg: CaseConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return out["Theta_dep"] - _wall_temperature(x, cfg), out["u_r"], out["u_x"]


def res_inlet_bc(
    out: dict[str, torch.Tensor],
    r: torch.Tensor,
    cfg: CaseConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        out["Theta_f"] - cfg.Theta_in,
        out["u_x"] - _poiseuille(r, out["r_int"]),
        out["u_r"],
    )


def res_inlet_rint_bc(out: dict[str, torch.Tensor], cfg: CaseConfig) -> torch.Tensor:
    return out["r_int"] - cfg.r_w


def res_outlet_bc(
    out: dict[str, torch.Tensor],
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _grad(out["Theta_f"], x), _grad(out["u_x"], x)


def res_axis_bc(
    out: dict[str, torch.Tensor],
    r: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return out["u_r"], _grad(out["Theta_f"], r), _grad(out["u_x"], r)


def res_ic(
    out: dict[str, torch.Tensor],
    r: torch.Tensor,
    cfg: CaseConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        out["r_int"] - cfg.r_int_ic,
        out["Theta_f"] - cfg.Theta_in,
        out["u_x"] - _poiseuille(r, torch.full_like(r, cfg.r_w)),
        out["u_r"],
    )


def compute_all_residuals(
    model,
    batch: dict[str, dict[str, torch.Tensor]],
    cfg: Config,
) -> dict[str, torch.Tensor]:
    case = cfg.case
    eps = cfg.network.interface_epsilon

    def forward_points(points: dict[str, torch.Tensor]):
        r, x, tau = points["r"], points["x"], points["tau"]
        out = model(r, x, tau)
        out["H"] = model.smooth_heaviside(r, out["r_int"].detach(), eps)
        return out, r, x, tau

    results: dict[str, torch.Tensor] = {}

    out, r, x, tau = forward_points(batch["interior"])
    results["mass"] = res_mass(out, r, x, tau)
    results["mom_x"] = res_mom_x(out, r, x, tau, case)
    results["mom_r"] = res_mom_r(out, r, x, tau, case)
    results["energy_fluid"] = res_energy_fluid(out, r, x, tau, case)
    results["energy_dep"] = res_energy_dep(out, r, x, tau, case)

    out, r, x, tau = forward_points(batch["interface"])
    results["stefan"] = res_stefan(out, r, x, tau, case)
    results["T_cont_fluid"], results["T_cont_dep"] = res_T_continuity(out, case)
    results["interface_u_x"], results["interface_u_r"] = res_interface_velocity_bc(out)
    results["rint_mono"] = res_rint_monotonic(out, tau)
    results["rint_x_mono"] = res_rint_x_monotonic(out, x)

    out, r, x, tau = forward_points(batch["wall"])
    results["bc_wall_T"], results["bc_wall_ur"], results["bc_wall_ux"] = res_wall_bc(out, x, case)

    out, r, x, tau = forward_points(batch["inlet"])
    results["bc_inlet_T"], results["bc_inlet_ux"], results["bc_inlet_ur"] = res_inlet_bc(out, r, case)
    results["bc_inlet_rint"] = res_inlet_rint_bc(out, case)

    out, r, x, tau = forward_points(batch["outlet"])
    results["bc_outlet_Tf"], results["bc_outlet_ux"] = res_outlet_bc(out, x)

    out, r, x, tau = forward_points(batch["axis"])
    results["bc_axis_ur"], results["bc_axis_Tf"], results["bc_axis_ux"] = res_axis_bc(out, r)

    out, r, x, tau = forward_points(batch["ic"])
    results["ic_rint"], results["ic_Tf"], results["ic_ux"], results["ic_ur"] = res_ic(out, r, case)

    return results
