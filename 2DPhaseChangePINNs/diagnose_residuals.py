"""Raw residual diagnostics for trained PINN checkpoints.

The training CSV stores weighted loss groups.  This script evaluates the
underlying residual tensors directly, reports raw RMS values, and also reports
the same weighted grouped losses used during training.  It is intended for
debugging convergence: the largest raw residual is often not the same item as
the largest weighted contribution.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import torch

from config import Config, config_from_dict
from equations import compute_all_residuals
from losses import compute_loss_terms
from network import PINNSolidification
from sampling import build_batch
from train import choose_device, move_batch


GROUP_RESIDUALS = {
    "mass": ["mass"],
    "mom_x": ["mom_x"],
    "mom_r": ["mom_r"],
    "energy_fluid": ["energy_fluid"],
    "energy_dep": ["energy_dep"],
    "stefan": ["stefan"],
    "T_cont": ["T_cont_fluid", "T_cont_dep"],
    "interface_vel": ["interface_u_x", "interface_u_r"],
    "rint_mono": ["rint_mono"],
    "rint_x_mono": ["rint_x_mono"],
    "bc_ic": [
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
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose raw PINN residual errors.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--n-interior", type=int, default=1024)
    parser.add_argument("--n-interface", type=int, default=1024)
    parser.add_argument("--n-boundary", type=int, default=256)
    parser.add_argument("--n-ic", type=int, default=512)
    parser.add_argument("--top", type=int, default=12)
    return parser.parse_args()


def apply_sampling_overrides(run_cfg: Config, args: argparse.Namespace) -> None:
    run_cfg.sampling.N_interior = args.n_interior
    run_cfg.sampling.N_interface = args.n_interface
    run_cfg.sampling.N_boundary = args.n_boundary
    run_cfg.sampling.N_ic = args.n_ic


def rms(tensor: torch.Tensor) -> float:
    return math.sqrt(float(tensor.detach().pow(2).mean().cpu().item()))


def mean_abs(tensor: torch.Tensor) -> float:
    return float(tensor.detach().abs().mean().cpu().item())


def max_abs(tensor: torch.Tensor) -> float:
    return float(tensor.detach().abs().max().cpu().item())


def build_residual_rows(residuals: dict[str, torch.Tensor]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, values in residuals.items():
        detached = values.detach()
        rows.append({
            "kind": "raw_residual",
            "name": name,
            "rms": rms(detached),
            "mean_abs": mean_abs(detached),
            "max_abs": max_abs(detached),
            "n": int(detached.numel()),
            "weighted_loss": "",
        })
    return sorted(rows, key=lambda row: row["rms"], reverse=True)


def build_group_rows(residuals: dict[str, torch.Tensor],
                     terms: dict[str, torch.Tensor]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group, names in GROUP_RESIDUALS.items():
        present = [residuals[name].detach() for name in names if name in residuals]
        if not present:
            continue
        group_mse = sum(float(t.pow(2).mean().cpu().item()) for t in present) / len(present)
        rows.append({
            "kind": "loss_group",
            "name": group,
            "rms": math.sqrt(group_mse),
            "mean_abs": "",
            "max_abs": "",
            "n": sum(int(t.numel()) for t in present),
            "weighted_loss": float(terms[group].detach().cpu().item()),
        })
    return sorted(rows, key=lambda row: row["weighted_loss"], reverse=True)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["kind", "name", "rms", "mean_abs", "max_abs", "n", "weighted_loss"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def print_table(title: str,
                rows: list[dict[str, Any]],
                top: int,
                weighted: bool = False) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    if weighted:
        print(f"{'rank':>4}  {'name':<18}  {'raw_rms':>12}  {'weighted_loss':>14}  {'n':>8}")
        for idx, row in enumerate(rows[:top], start=1):
            print(
                f"{idx:>4}  {row['name']:<18}  {row['rms']:>12.5e}  "
                f"{row['weighted_loss']:>14.5e}  {row['n']:>8}"
            )
    else:
        print(f"{'rank':>4}  {'name':<18}  {'raw_rms':>12}  {'mean_abs':>12}  {'max_abs':>12}  {'n':>8}")
        for idx, row in enumerate(rows[:top], start=1):
            print(
                f"{idx:>4}  {row['name']:<18}  {row['rms']:>12.5e}  "
                f"{row['mean_abs']:>12.5e}  {row['max_abs']:>12.5e}  {row['n']:>8}"
            )


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if isinstance(payload, dict) and isinstance(payload.get("config"), dict):
        run_cfg = config_from_dict(payload["config"], fallback=Config())
    else:
        print("[diagnose] checkpoint has no saved config; using current config.py")
        run_cfg = Config()
    apply_sampling_overrides(run_cfg, args)

    model = PINNSolidification(run_cfg.network, run_cfg.case, seed=run_cfg.seed).to(device)
    state = payload.get("model_state", payload.get("model_state_dict", payload))
    model.load_state_dict(state)
    model.eval()

    batch = move_batch(build_batch(model, run_cfg, seed=args.seed), device)
    residuals = compute_all_residuals(model, batch, run_cfg)
    terms = compute_loss_terms(residuals, run_cfg.weights, physics_on=True)

    raw_rows = build_residual_rows(residuals)
    group_rows = build_group_rows(residuals, terms)
    total = sum(float(value.detach().cpu().item()) for value in terms.values())

    print(f"[diagnose] checkpoint={args.checkpoint}")
    print(f"[diagnose] checkpoint_iter={payload.get('iteration', 'unknown')} phase={payload.get('phase', 'unknown')}")
    print(
        f"[diagnose] points interior/interface/boundary/ic="
        f"{run_cfg.sampling.N_interior}/{run_cfg.sampling.N_interface}/"
        f"{run_cfg.sampling.N_boundary}/{run_cfg.sampling.N_ic}"
    )
    print(f"[diagnose] weighted_total={total:.6e}")
    print_table("Weighted loss groups", group_rows, args.top, weighted=True)
    print_table("Raw residual RMS by residual", raw_rows, args.top, weighted=False)

    if args.output is not None:
        rows = group_rows + raw_rows
        write_csv(args.output, rows)
        print(f"\n[diagnose] wrote {args.output}")


if __name__ == "__main__":
    main()
