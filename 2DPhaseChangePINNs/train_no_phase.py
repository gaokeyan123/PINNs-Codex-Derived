"""Clean-pipe no-phase-change diagnostic trainer.

This script intentionally disables the moving-interface physics so momentum
can be tested without phase-change coupling:

  - full domain is treated as fluid, H = 0
  - no Stefan loss
  - no interface temperature-continuity loss
  - no deposit energy loss
  - wall, inlet, and IC temperatures are equal and above solidus

The target thermal solution is constant, so the energy residual should become
small quickly.  If the remaining dominant residual is momentum, the phase-change
coupling is not the cause.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import torch

from config import Config
from equations import (
    _grad,
    _poiseuille,
    res_energy_fluid,
    res_mass,
    res_mom_r,
    res_mom_x,
)
from network import PINNSolidification, count_parameters
from sampling import (
    sample_axis,
    sample_ic,
    sample_inlet,
    sample_interior,
    sample_outlet,
    sample_wall,
)
from train import choose_device, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train clean-pipe no-phase diagnostic case.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/no_phase"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/no_phase/latest.pth"))
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--n-interior", type=int, default=512)
    parser.add_argument("--n-boundary", type=int, default=128)
    parser.add_argument("--n-ic", type=int, default=256)
    parser.add_argument("--r-min-interior", type=float, default=0.0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--resample-every", type=int, default=100)
    parser.add_argument("--target-rms", type=float, default=None)
    parser.add_argument("--target-check-every", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--pressure-weight", type=float, default=1.0)
    parser.add_argument("--bc-weight", type=float, default=100.0)
    parser.add_argument("--temp-weight", type=float, default=100.0)
    return parser.parse_args()


def make_no_phase_config(args: argparse.Namespace) -> Config:
    run_cfg = Config()
    run_cfg.seed = args.seed
    run_cfg.output_dir = args.output_dir
    run_cfg.case.Theta_wall = args.temperature
    run_cfg.case.Theta_in = args.temperature
    run_cfg.case.Theta_solidus = 0.5
    run_cfg.case.r_int_ic = run_cfg.case.r_w
    run_cfg.sampling.N_interior = args.n_interior
    run_cfg.sampling.N_boundary = args.n_boundary
    run_cfg.sampling.N_ic = args.n_ic
    return run_cfg


def move_points(points: dict[str, torch.Tensor],
                device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().to(device).requires_grad_(True)
        for key, value in points.items()
    }


def sample_interior_away_axis(N: int,
                              run_cfg: Config,
                              r_min: float,
                              seed: int) -> dict[str, torch.Tensor]:
    """Interior points with r remapped from [0, r_w] to [r_min, r_w].

    Axis physics is still enforced by the separate axis boundary-condition
    point set.  This only prevents singular cylindrical PDE terms from being
    evaluated at tiny nonzero radii where a soft PINN residual is badly
    conditioned.
    """
    pts = sample_interior(N, run_cfg, seed)
    r_w = run_cfg.case.r_w
    if r_min <= 0.0:
        return pts
    if r_min >= r_w:
        raise ValueError(f"--r-min-interior must be smaller than r_w={r_w}")
    r_unit = pts["r"].detach() / r_w
    r = r_min + r_unit * (r_w - r_min)
    return {
        "r": r.detach().requires_grad_(True),
        "x": pts["x"].detach().requires_grad_(True),
        "t": pts["t"].detach().requires_grad_(True),
    }


def build_clean_batch(run_cfg: Config,
                      seed: int,
                      device: torch.device,
                      r_min_interior: float = 0.0) -> dict[str, dict[str, torch.Tensor]]:
    sc = run_cfg.sampling
    batch = {
        "interior": sample_interior_away_axis(
            sc.N_interior, run_cfg, r_min_interior, seed + 0
        ),
        "wall": sample_wall(sc.N_boundary, run_cfg, seed + 1),
        "inlet": sample_inlet(sc.N_boundary, run_cfg, seed + 2),
        "outlet": sample_outlet(sc.N_boundary, run_cfg, seed + 3),
        "axis": sample_axis(sc.N_boundary, run_cfg, seed + 4),
        "ic": sample_ic(sc.N_ic, run_cfg, seed + 5),
    }
    return {key: move_points(value, device) for key, value in batch.items()}


def mse(value: torch.Tensor) -> torch.Tensor:
    return value.pow(2).mean()


def fwd_fluid(model: PINNSolidification,
              pts: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    r, x, t = pts["r"], pts["x"], pts["t"]
    out = model(r, x, t)
    out["H"] = torch.zeros_like(r)
    return out, r, x, t


def poiseuille_clean(r: torch.Tensor, run_cfg: Config) -> torch.Tensor:
    return _poiseuille(r, torch.full_like(r, run_cfg.case.r_w))


def compute_no_phase_residuals(model: PINNSolidification,
                               batch: dict[str, dict[str, torch.Tensor]],
                               run_cfg: Config) -> dict[str, torch.Tensor]:
    case = run_cfg.case
    residuals: dict[str, torch.Tensor] = {}

    out, r, x, t = fwd_fluid(model, batch["interior"])
    residuals["mass"] = res_mass(out, r, x, t)
    residuals["mom_x"] = res_mom_x(out, r, x, t, case)
    residuals["mom_r"] = res_mom_r(out, r, x, t, case)
    residuals["energy_fluid"] = res_energy_fluid(out, r, x, t, case)
    residuals["temp_interior"] = out["Theta_f"] - case.Theta_wall

    out, r, x, t = fwd_fluid(model, batch["wall"])
    residuals["bc_wall_Tf"] = out["Theta_f"] - case.Theta_wall
    residuals["bc_wall_ur"] = out["u_r"]
    residuals["bc_wall_ux"] = out["u_x"]

    out, r, x, t = fwd_fluid(model, batch["inlet"])
    residuals["bc_inlet_Tf"] = out["Theta_f"] - case.Theta_in
    residuals["bc_inlet_ux"] = out["u_x"] - poiseuille_clean(r, run_cfg)
    residuals["bc_inlet_ur"] = out["u_r"]

    out, r, x, t = fwd_fluid(model, batch["outlet"])
    residuals["bc_outlet_Tf_x"] = _grad(out["Theta_f"], x)
    residuals["bc_outlet_ux_x"] = _grad(out["u_x"], x)
    residuals["pressure_gauge"] = out["p"]

    out, r, x, t = fwd_fluid(model, batch["axis"])
    residuals["bc_axis_ur"] = out["u_r"]
    residuals["bc_axis_Tf_r"] = _grad(out["Theta_f"], r)
    residuals["bc_axis_ux_r"] = _grad(out["u_x"], r)

    out, r, x, t = fwd_fluid(model, batch["ic"])
    residuals["ic_Tf"] = out["Theta_f"] - case.Theta_in
    residuals["ic_ux"] = out["u_x"] - poiseuille_clean(r, run_cfg)
    residuals["ic_ur"] = out["u_r"]

    return residuals


def compute_terms(residuals: dict[str, torch.Tensor],
                  args: argparse.Namespace) -> dict[str, torch.Tensor]:
    flow_bc_keys = [
        "bc_wall_ur",
        "bc_wall_ux",
        "bc_inlet_ux",
        "bc_inlet_ur",
        "bc_outlet_ux_x",
        "bc_axis_ur",
        "bc_axis_ux_r",
        "ic_ux",
        "ic_ur",
    ]
    temp_keys = [
        "temp_interior",
        "bc_wall_Tf",
        "bc_inlet_Tf",
        "bc_outlet_Tf_x",
        "bc_axis_Tf_r",
        "ic_Tf",
    ]
    terms = {
        "mass": mse(residuals["mass"]),
        "mom_x": mse(residuals["mom_x"]),
        "mom_r": mse(residuals["mom_r"]),
        "energy_fluid": mse(residuals["energy_fluid"]),
        "flow_bc": args.bc_weight * sum(mse(residuals[k]) for k in flow_bc_keys) / len(flow_bc_keys),
        "temp_const": args.temp_weight * sum(mse(residuals[k]) for k in temp_keys) / len(temp_keys),
        "pressure_gauge": args.pressure_weight * mse(residuals["pressure_gauge"]),
    }
    terms["total"] = sum(terms.values())
    return terms


def tensor_rms(value: torch.Tensor) -> float:
    return math.sqrt(float(value.detach().pow(2).mean().cpu().item()))


def tensor_mean_abs(value: torch.Tensor) -> float:
    return float(value.detach().abs().mean().cpu().item())


def tensor_max_abs(value: torch.Tensor) -> float:
    return float(value.detach().abs().max().cpu().item())


def print_terms(iteration: int, terms: dict[str, torch.Tensor]) -> None:
    pieces = [
        f"iter {iteration:5d}",
        f"total {terms['total'].detach().item():.4e}",
        f"mom_r {terms['mom_r'].detach().item():.2e}",
        f"mom_x {terms['mom_x'].detach().item():.2e}",
        f"mass {terms['mass'].detach().item():.2e}",
        f"energy {terms['energy_fluid'].detach().item():.2e}",
        f"flow_bc {terms['flow_bc'].detach().item():.2e}",
        f"T_const {terms['temp_const'].detach().item():.2e}",
    ]
    print(" | ".join(pieces), flush=True)


def append_csv(path: Path,
               iteration: int,
               terms: dict[str, torch.Tensor],
               write_header: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"iteration": iteration}
    row.update({key: float(value.detach().cpu().item()) for key, value in terms.items()})
    fields = ["iteration"] + sorted(key for key in row if key != "iteration")
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def save_checkpoint(path: Path,
                    model: PINNSolidification,
                    run_cfg: Config,
                    iteration: int,
                    terms: dict[str, torch.Tensor],
                    optimizer: torch.optim.Optimizer | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "case": "no_phase_clean_pipe",
        "iteration": iteration,
        "model_state": model.state_dict(),
        "config": {
            "Pe": run_cfg.case.Pe,
            "Re": run_cfg.case.Re,
            "Theta_wall": run_cfg.case.Theta_wall,
            "Theta_in": run_cfg.case.Theta_in,
            "Theta_solidus": run_cfg.case.Theta_solidus,
        },
        "last_log": {key: float(value.detach().cpu().item()) for key, value in terms.items()},
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
    }, path)


def diagnostic_rows(residuals: dict[str, torch.Tensor]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, value in residuals.items():
        rows.append({
            "name": name,
            "rms": tensor_rms(value),
            "mean_abs": tensor_mean_abs(value),
            "max_abs": tensor_max_abs(value),
            "n": int(value.numel()),
        })
    return sorted(rows, key=lambda item: item["rms"], reverse=True)


def write_residual_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "rms", "mean_abs", "max_abs", "n"])
        writer.writeheader()
        writer.writerows(rows)


def print_residual_table(rows: list[dict[str, Any]], top: int = 16) -> None:
    print("\nRaw residual RMS after no-phase training")
    print("----------------------------------------")
    print(f"{'rank':>4}  {'name':<18}  {'rms':>12}  {'mean_abs':>12}  {'max_abs':>12}  {'n':>8}")
    for idx, row in enumerate(rows[:top], start=1):
        print(
            f"{idx:>4}  {row['name']:<18}  {row['rms']:>12.5e}  "
            f"{row['mean_abs']:>12.5e}  {row['max_abs']:>12.5e}  {row['n']:>8}"
        )


def evaluate_core_rms(model: PINNSolidification,
                      run_cfg: Config,
                      args: argparse.Namespace,
                      device: torch.device,
                      seed: int) -> dict[str, float]:
    batch = build_clean_batch(run_cfg, seed, device, args.r_min_interior)
    residuals = compute_no_phase_residuals(model, batch, run_cfg)
    core = ("mass", "mom_x", "mom_r", "energy_fluid")
    return {name: tensor_rms(residuals[name]) for name in core}


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    seed_everything(args.seed)
    run_cfg = make_no_phase_config(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = PINNSolidification(run_cfg.network, run_cfg.case, seed=args.seed).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    start_iter = 0
    if args.resume is not None:
        payload = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(payload["model_state"])
        optimizer_state = payload.get("optimizer_state")
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
        start_iter = int(payload.get("iteration", 0))
        print(f"[no-phase] resumed {args.resume} at iter {start_iter}", flush=True)
    print(
        f"[no-phase] device={device} seed={args.seed} params={count_parameters(model):,} "
        f"Theta={args.temperature:g} Theta_solidus={run_cfg.case.Theta_solidus:g} "
        f"r_min_interior={args.r_min_interior:g}",
        flush=True,
    )

    batch = build_clean_batch(run_cfg, args.seed, device, args.r_min_interior)
    csv_path = args.output_dir / "loss_history.csv"
    write_header = True
    last_terms: dict[str, torch.Tensor] = {}

    final_iter = start_iter
    for local_step in range(1, args.iters + 1):
        iteration = start_iter + local_step
        if iteration == 1 or (
            args.resample_every > 0
            and iteration % args.resample_every == 0
        ):
            batch = build_clean_batch(
                run_cfg, args.seed + iteration, device, args.r_min_interior
            )

        model.train()
        optimizer.zero_grad(set_to_none=True)
        residuals = compute_no_phase_residuals(model, batch, run_cfg)
        terms = compute_terms(residuals, args)
        loss = terms["total"]
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at iter {iteration}")
        loss.backward()
        optimizer.step()

        last_terms = terms
        append_csv(csv_path, iteration, terms, write_header)
        write_header = False
        if iteration == 1 or iteration % args.log_every == 0:
            print_terms(iteration, terms)
        if (
            args.target_rms is not None
            and args.target_check_every > 0
            and (local_step == 1 or local_step % args.target_check_every == 0)
        ):
            model.eval()
            core_rms = evaluate_core_rms(
                model, run_cfg, args, device, args.seed + 200_000 + iteration
            )
            worst_name = max(core_rms, key=core_rms.get)
            worst = core_rms[worst_name]
            print(
                "[target] "
                + " ".join(f"{name}={value:.3e}" for name, value in core_rms.items())
                + f" worst={worst_name}:{worst:.3e} target={args.target_rms:.3e}",
                flush=True,
            )
            model.train()
            if worst < args.target_rms:
                print(f"[stop] target fresh core RMS reached at iter {iteration}", flush=True)
                final_iter = iteration
                break
        final_iter = iteration

    model.eval()
    if not last_terms:
        residuals = compute_no_phase_residuals(model, batch, run_cfg)
        last_terms = compute_terms(residuals, args)
    save_checkpoint(args.checkpoint, model, run_cfg, final_iter, last_terms, optimizer)
    fresh_batch = build_clean_batch(
        run_cfg, args.seed + 100_000, device, args.r_min_interior
    )
    fresh_residuals = compute_no_phase_residuals(model, fresh_batch, run_cfg)
    rows = diagnostic_rows(fresh_residuals)
    residual_csv = args.output_dir / "raw_residuals.csv"
    write_residual_csv(residual_csv, rows)
    print_residual_table(rows)
    print(f"\n[no-phase] wrote {csv_path}")
    print(f"[no-phase] wrote {residual_csv}")
    print(f"[no-phase] wrote {args.checkpoint}")


if __name__ == "__main__":
    main()
