"""Weighted composite loss assembly for PINN training."""

from pathlib import Path

import torch

from config import Config, LossWeights
from equations import (
    compute_all_residuals,
    res_axis_bc,
    res_ic,
    res_inlet_bc,
    res_inlet_rint_bc,
    res_outlet_bc,
    res_wall_bc,
)

BC_IC_KEYS = [
    "bc_wall_T",
    "bc_wall_ur",
    "bc_wall_ux",
    "bc_inlet_T",
    "bc_inlet_ux",
    "bc_inlet_ur",
    "bc_inlet_rint",
    "bc_outlet_Tf",
    "bc_outlet_ux",
    "bc_axis_ur",
    "bc_axis_Tf",
    "bc_axis_ux",
    "ic_rint",
    "ic_Tf",
    "ic_ux",
    "ic_ur",
]

PHYSICS_TERMS = (
    "mass",
    "mom_x",
    "mom_r",
    "energy_fluid",
    "energy_dep",
    "stefan",
    "T_cont",
    "interface_vel",
    "rint_mono",
    "rint_x_mono",
)


def mse(residual: torch.Tensor) -> torch.Tensor:
    return residual.pow(2).mean()


def _compute_bc_ic_residuals(model, batch: dict, cfg: Config) -> dict[str, torch.Tensor]:
    case = cfg.case
    eps = cfg.network.interface_epsilon

    def forward_points(points: dict[str, torch.Tensor]):
        r, x, tau = points["r"], points["x"], points["tau"]
        out = model(r, x, tau)
        out["H"] = model.smooth_heaviside(r, out["r_int"].detach(), eps)
        return out, r, x

    results: dict[str, torch.Tensor] = {}

    out, _, x = forward_points(batch["wall"])
    results["bc_wall_T"], results["bc_wall_ur"], results["bc_wall_ux"] = res_wall_bc(out, x, case)

    out, r, _ = forward_points(batch["inlet"])
    results["bc_inlet_T"], results["bc_inlet_ux"], results["bc_inlet_ur"] = res_inlet_bc(out, r, case)
    results["bc_inlet_rint"] = res_inlet_rint_bc(out, case)

    out, _, x = forward_points(batch["outlet"])
    results["bc_outlet_Tf"], results["bc_outlet_ux"] = res_outlet_bc(out, x)

    out, r, _ = forward_points(batch["axis"])
    results["bc_axis_ur"], results["bc_axis_Tf"], results["bc_axis_ux"] = res_axis_bc(out, r)

    out, r, _ = forward_points(batch["ic"])
    results["ic_rint"], results["ic_Tf"], results["ic_ux"], results["ic_ur"] = res_ic(out, r, case)

    return results


def compute_loss_terms(
    residuals: dict[str, torch.Tensor],
    weights: LossWeights,
    physics_on: bool = True,
) -> dict[str, torch.Tensor]:
    terms: dict[str, torch.Tensor] = {}

    if physics_on:
        terms["mass"] = weights.mass * mse(residuals["mass"])
        terms["mom_x"] = weights.mom_x * mse(residuals["mom_x"])
        terms["mom_r"] = weights.mom_r * mse(residuals["mom_r"])
        terms["energy_fluid"] = weights.energy_fluid * mse(residuals["energy_fluid"])
        terms["energy_dep"] = weights.energy_dep * mse(residuals["energy_dep"])
        terms["stefan"] = weights.stefan * mse(residuals["stefan"])
        terms["T_cont"] = weights.T_continuity * (
            mse(residuals["T_cont_fluid"]) + mse(residuals["T_cont_dep"])
        ) * 0.5
        terms["interface_vel"] = weights.interface_velocity * (
            mse(residuals["interface_u_x"]) + mse(residuals["interface_u_r"])
        ) * 0.5
        terms["rint_mono"] = weights.rint_mono * mse(residuals["rint_mono"])
        terms["rint_x_mono"] = weights.rint_x_mono * mse(residuals["rint_x_mono"])
    else:
        zero = torch.tensor(0.0)
        for name in PHYSICS_TERMS:
            terms[name] = zero

    present = [key for key in BC_IC_KEYS if key in residuals]
    terms["bc_ic"] = weights.bc_ic * sum(mse(residuals[key]) for key in present) / len(present)
    return terms


def total_loss(terms: dict[str, torch.Tensor], physics_on: bool = True) -> torch.Tensor:
    if physics_on:
        return sum(terms.values())
    return terms["bc_ic"]


def compute_loss(
    model,
    batch: dict,
    cfg: Config,
    physics_on: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    if physics_on:
        residuals = compute_all_residuals(model, batch, cfg)
    else:
        residuals = _compute_bc_ic_residuals(model, batch, cfg)

    terms = compute_loss_terms(residuals, cfg.weights, physics_on)
    loss = total_loss(terms, physics_on)
    log = {key: value.item() for key, value in terms.items()}
    log["total"] = loss.item()
    return loss, log


def format_loss_line(log: dict[str, float], iteration: int, physics_on: bool = True) -> str:
    total_str = f"total {log['total']:.4e}"
    bc_ic_str = f"bc_ic {log['bc_ic']:.2e}"

    if not physics_on:
        return f"iter {iteration:5d} | {total_str} | {bc_ic_str}  [BC/IC only]"

    physics_total = (
        log.get("mass", 0.0)
        + log.get("mom_x", 0.0)
        + log.get("mom_r", 0.0)
        + log.get("energy_fluid", 0.0)
        + log.get("energy_dep", 0.0)
    )
    rint_total = log.get("rint_mono", 0.0) + log.get("rint_x_mono", 0.0)
    return (
        f"iter {iteration:5d} | {total_str} | {bc_ic_str} | "
        f"phys {physics_total:.2e} | stefan {log.get('stefan', 0.0):.2e} | "
        f"T_cont {log.get('T_cont', 0.0):.2e} | "
        f"int_vel {log.get('interface_vel', 0.0):.2e} | rint {rint_total:.2e}"
    )


def log_to_csv(
    log: dict[str, float],
    iteration: int,
    csv_path: Path,
    write_header: bool = False,
) -> None:
    import csv

    fields = ["iteration"] + sorted(log.keys())
    row = {"iteration": iteration, **log}
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
