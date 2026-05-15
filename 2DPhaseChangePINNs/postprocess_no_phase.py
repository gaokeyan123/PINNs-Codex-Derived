"""Postprocess clean-pipe no-phase diagnostic checkpoints.

Produces field plots and raw residual maps for the no-phase case where the
whole pipe is fluid and the target temperature is constant.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from config import Config
from network import PINNSolidification
from train import choose_device
from train_no_phase import compute_no_phase_residuals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot clean-pipe no-phase PINN results.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/no_phase_plots"))
    parser.add_argument("--time", type=float, default=None,
                        help="Tau value to plot; defaults to checkpoint/config tau_end.")
    parser.add_argument("--grid-nx", type=int, default=140)
    parser.add_argument("--grid-nr", type=int, default=80)
    parser.add_argument("--r-min-residual", type=float, default=0.01)
    return parser.parse_args()


def load_model(path: Path, device: torch.device) -> tuple[PINNSolidification, Config, dict]:
    payload = torch.load(path, map_location=device, weights_only=False)
    run_cfg = Config()
    saved_cfg = payload.get("config", {})
    if "nondim_scheme" in saved_cfg:
        run_cfg.case.nondim_scheme = str(saved_cfg["nondim_scheme"])
    if "Theta_wall" in saved_cfg:
        run_cfg.case.Theta_wall = float(saved_cfg["Theta_wall"])
    if "Theta_in" in saved_cfg:
        run_cfg.case.Theta_in = float(saved_cfg["Theta_in"])
    if "Theta_solidus" in saved_cfg:
        run_cfg.case.Theta_solidus = float(saved_cfg["Theta_solidus"])
    if "Pe_D" in saved_cfg:
        run_cfg.case.Pe_D = float(saved_cfg["Pe_D"])
    elif "Pe" in saved_cfg:
        run_cfg.case.Pe_D = float(saved_cfg["Pe"])
    if "Re_D" in saved_cfg:
        run_cfg.case.Re_D = float(saved_cfg["Re_D"])
    elif "Re" in saved_cfg:
        run_cfg.case.Re_D = float(saved_cfg["Re"])
    if "L" in saved_cfg:
        run_cfg.case.L = float(saved_cfg["L"])
    if "tau_end" in saved_cfg:
        run_cfg.case.tau_end = float(saved_cfg["tau_end"])
    elif "t_end" in saved_cfg:
        run_cfg.case.tau_end = float(saved_cfg["t_end"])

    model = PINNSolidification(run_cfg.network, run_cfg.case, seed=run_cfg.seed).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, run_cfg, payload


def make_grid(run_cfg: Config,
              device: torch.device,
              nx: int,
              nr: int,
              t_value: float,
              requires_grad: bool = False) -> dict[str, torch.Tensor]:
    x = torch.linspace(0.0, run_cfg.case.L, nx, device=device)
    r = torch.linspace(0.0, run_cfg.case.r_w, nr, device=device)
    rr, xx = torch.meshgrid(r, x, indexing="ij")
    tt = torch.full_like(rr, t_value)
    return {
        "r": rr.reshape(-1).detach().requires_grad_(requires_grad),
        "x": xx.reshape(-1).detach().requires_grad_(requires_grad),
        "t": tt.reshape(-1).detach().requires_grad_(requires_grad),
        "rr": rr.detach().cpu().numpy(),
        "xx": xx.detach().cpu().numpy(),
    }


def plot_fields(model: PINNSolidification,
                run_cfg: Config,
                args: argparse.Namespace,
                device: torch.device) -> Path:
    grid = make_grid(run_cfg, device, args.grid_nx, args.grid_nr, args.time)
    with torch.no_grad():
        out = model(grid["r"], grid["x"], grid["t"])

    shape = (args.grid_nr, args.grid_nx)
    fields = [
        ("Theta_f", out["Theta_f"].reshape(shape).detach().cpu().numpy(), "inferno"),
        ("u_x", out["u_x"].reshape(shape).detach().cpu().numpy(), "viridis"),
        ("u_r", out["u_r"].reshape(shape).detach().cpu().numpy(), "coolwarm"),
        ("p", out["p"].reshape(shape).detach().cpu().numpy(), "coolwarm"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for ax, (name, values, cmap) in zip(axes.flat, fields):
        im = ax.contourf(grid["xx"], grid["rr"], values, levels=40, cmap=cmap)
        ax.set_title(name)
        ax.set_xlabel("x")
        ax.set_ylabel("r")
        fig.colorbar(im, ax=ax)
    fig.suptitle(f"No-phase clean-pipe fields at tau={args.time:g}")
    path = args.output_dir / f"fields_tau{args.time:g}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_residuals(model: PINNSolidification,
                   run_cfg: Config,
                   args: argparse.Namespace,
                   device: torch.device) -> Path:
    grid = make_grid(
        run_cfg,
        device,
        args.grid_nx,
        args.grid_nr,
        args.time,
        requires_grad=True,
    )
    pts = {"r": grid["r"], "x": grid["x"], "t": grid["t"]}
    batch = {
        "interior": pts,
        "wall": pts,
        "inlet": pts,
        "outlet": pts,
        "axis": pts,
        "ic": pts,
    }
    residuals = compute_no_phase_residuals(model, batch, run_cfg)

    shape = (args.grid_nr, args.grid_nx)
    r_grid = grid["rr"]
    mask = r_grid < args.r_min_residual
    names = ["mass", "mom_x", "mom_r", "energy_fluid"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for ax, name in zip(axes.flat, names):
        values = residuals[name].reshape(shape).detach().cpu().numpy()
        values = np.where(mask, np.nan, values)
        vmax = np.nanpercentile(np.abs(values), 98)
        if not np.isfinite(vmax) or vmax <= 0.0:
            vmax = 1.0
        im = ax.contourf(
            grid["xx"],
            grid["rr"],
            values,
            levels=40,
            cmap="coolwarm",
            vmin=-vmax,
            vmax=vmax,
        )
        ax.axhline(args.r_min_residual, color="black", lw=0.8, ls="--")
        ax.set_title(name)
        ax.set_xlabel("x")
        ax.set_ylabel("r")
        fig.colorbar(im, ax=ax)
    fig.suptitle(f"No-phase residual maps at tau={args.time:g}; r<{args.r_min_residual:g} masked")
    path = args.output_dir / f"residuals_tau{args.time:g}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def print_summary(model: PINNSolidification,
                  run_cfg: Config,
                  args: argparse.Namespace,
                  device: torch.device) -> None:
    grid = make_grid(run_cfg, device, args.grid_nx, args.grid_nr, args.time)
    with torch.no_grad():
        out = model(grid["r"], grid["x"], grid["t"])
    theta = out["Theta_f"].detach()
    ux = out["u_x"].detach()
    ur = out["u_r"].detach()
    p = out["p"].detach()
    print(
        "[post-no-phase] "
        f"Theta min/mean/max={theta.min().item():.4f}/{theta.mean().item():.4f}/{theta.max().item():.4f}"
    )
    print(
        "[post-no-phase] "
        f"u_x min/mean/max={ux.min().item():.4f}/{ux.mean().item():.4f}/{ux.max().item():.4f}"
    )
    print(
        "[post-no-phase] "
        f"u_r min/mean/max={ur.min().item():.4f}/{ur.mean().item():.4f}/{ur.max().item():.4f}"
    )
    print(
        "[post-no-phase] "
        f"p min/mean/max={p.min().item():.4f}/{p.mean().item():.4f}/{p.max().item():.4f}"
    )


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model, run_cfg, payload = load_model(args.checkpoint, device)
    if args.time is None:
        args.time = run_cfg.case.tau_end
    print(
        f"[post-no-phase] checkpoint={args.checkpoint} "
        f"iter={payload.get('iteration', 'unknown')}"
    )
    print_summary(model, run_cfg, args, device)
    field_path = plot_fields(model, run_cfg, args, device)
    residual_path = plot_residuals(model, run_cfg, args, device)
    print(f"[post-no-phase] wrote {field_path}")
    print(f"[post-no-phase] wrote {residual_path}")


if __name__ == "__main__":
    main()
