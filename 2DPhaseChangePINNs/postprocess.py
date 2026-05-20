"""
Post-processing and validation plots for trained pipe-solidification PINNs.

The script always produces PINN-only plots. If a MATLAB reference file is
available, interface overlays are added when recognizable arrays are found.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import cfg, Config, POSTPROCESS_ROOT, config_from_dict
from equations import (
    res_energy_dep,
    res_energy_fluid,
    res_mass,
    res_mom_r,
    res_mom_x,
)
from network import PINNSolidification


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot trained PINN fields and residual maps.")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Checkpoint to postprocess. Defaults to the lowest-loss checkpoint, "
                             "falling back to latest.pth.")
    parser.add_argument("--last-checkpoint", type=Path, default=None,
                        help="Final-iteration checkpoint. Defaults to last.pth beside --checkpoint.")
    parser.add_argument("--matlab-ref", type=Path, default=cfg.output_dir / "matlab_ref.mat")
    parser.add_argument(
        "--taus",
        type=str,
        default="0.1951219512195122,0.3902439024390244,0.5853658536585366,0.7804878048780488",
        help="Comma-separated tau values for snapshots.",
    )
    parser.add_argument("--grid-nx", type=int, default=200)
    parser.add_argument("--grid-nr", type=int, default=200)
    parser.add_argument("--output-dir", type=Path, default=POSTPROCESS_ROOT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--skip-residual-maps", action="store_true",
                        help="Skip expensive PDE residual contour maps.")
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    if requested == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def parse_taus(text: str) -> list[float]:
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("--taus must contain at least one comma-separated value.")
    return values


def _loss_from_best_name(path: Path) -> float:
    try:
        return float(path.stem.rsplit("_loss", 1)[1])
    except (IndexError, ValueError):
        return math.inf


def find_best_checkpoint(checkpoint_dir: Path) -> Path | None:
    """Return the lowest-loss checkpoint in the current naming convention."""
    candidates = sorted(checkpoint_dir.glob("pinns_v*_ep*_loss*.pth"))
    if not candidates:
        return None
    return min(candidates, key=lambda path: (_loss_from_best_name(path), path.name))


def resolve_primary_checkpoint(checkpoint: Path | None,
                               checkpoint_dir: Path = cfg.training.checkpoint_dir) -> Path:
    if checkpoint is not None:
        return checkpoint
    best = find_best_checkpoint(checkpoint_dir)
    if best is not None:
        print(f"[postprocess] best checkpoint={best}")
        return best
    latest = checkpoint_dir / cfg.training.latest_checkpoint
    print(f"[postprocess] no best checkpoint found; using {latest}")
    return latest


def resolve_last_checkpoint(primary_checkpoint: Path,
                            last_checkpoint: Path | None = None) -> Path | None:
    path = last_checkpoint if last_checkpoint is not None else primary_checkpoint.parent / cfg.training.last_checkpoint
    if path.exists():
        print(f"[postprocess] last checkpoint={path}")
        return path
    print(f"[postprocess] last checkpoint missing; skipping last.pth plots and loss marker: {path}")
    return None


def checkpoint_iteration(checkpoint: Path | None) -> int | None:
    if checkpoint is None:
        return None
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "iteration" not in payload:
        print(f"[postprocess] no 'iteration' metadata in {checkpoint}; skipping loss marker")
        return None
    return int(payload["iteration"])


def find_loss_history(checkpoint: Path, run_cfg: Config) -> Path | None:
    candidates = [
        run_cfg.output_dir / "loss_history.csv",
        checkpoint.parent.parent / "outputs" / "loss_history.csv",
        checkpoint.parent / "loss_history.csv",
        cfg.output_dir / "loss_history.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def load_model(checkpoint: Path,
               device: torch.device) -> tuple[PINNSolidification, Config]:
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint must be a dict payload: {checkpoint}")
    if "config" not in payload:
        raise ValueError(f"checkpoint has no config: {checkpoint}")
    if "model_state" not in payload:
        raise ValueError(f"checkpoint has no model_state: {checkpoint}")
    run_cfg = config_from_dict(payload["config"])
    state = payload["model_state"]
    model = PINNSolidification(run_cfg.network, run_cfg.case, seed=run_cfg.seed).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, run_cfg


def make_grid(run_cfg: Config,
              grid_nr: int,
              grid_nx: int,
              tau_value: float,
              device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r = torch.linspace(0.0, run_cfg.case.r_w, grid_nr, device=device)
    x = torch.linspace(0.0, run_cfg.case.L, grid_nx, device=device)
    rr, xx = torch.meshgrid(r, x, indexing="ij")
    tt = torch.full_like(rr, tau_value)
    return rr, xx, tt


def evaluate_fields(model: PINNSolidification,
                    run_cfg: Config,
                    grid_nr: int,
                    grid_nx: int,
                    tau_value: float,
                    device: torch.device) -> dict[str, np.ndarray]:
    rr, xx, tt = make_grid(run_cfg, grid_nr, grid_nx, tau_value, device)
    with torch.no_grad():
        out = model(rr.reshape(-1), xx.reshape(-1), tt.reshape(-1))
        x_line = torch.linspace(0.0, run_cfg.case.L, grid_nx, device=device)
        r_int = model(
            torch.zeros_like(x_line),
            x_line,
            torch.full_like(x_line, tau_value),
        )["r_int"]

    fields = {
        "r": rr.detach().cpu().numpy(),
        "x": xx.detach().cpu().numpy(),
        "u_x": out["u_x"].reshape(grid_nr, grid_nx).detach().cpu().numpy(),
        "p": out["p"].reshape(grid_nr, grid_nx).detach().cpu().numpy(),
        "Theta_f": out["Theta_f"].reshape(grid_nr, grid_nx).detach().cpu().numpy(),
        "Theta_dep": out["Theta_dep"].reshape(grid_nr, grid_nx).detach().cpu().numpy(),
        "x_line": x_line.detach().cpu().numpy(),
        "r_int": r_int.detach().cpu().numpy(),
    }
    return fields


def load_matlab_reference(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        print(f"[postprocess] MATLAB reference not found: {path}")
        return None
    try:
        from scipy.io import loadmat
    except Exception as exc:
        print(f"[postprocess] scipy unavailable; skipping MATLAB overlay: {exc}")
        return None
    try:
        raw = loadmat(path)
    except Exception as exc:
        print(f"[postprocess] Could not read MATLAB reference {path}: {exc}")
        return None

    def pick(candidates: tuple[str, ...]) -> np.ndarray | None:
        lowered = {key.lower(): key for key in raw if not key.startswith("__")}
        for candidate in candidates:
            key = lowered.get(candidate.lower())
            if key is not None:
                arr = np.asarray(raw[key]).squeeze()
                if arr.size > 0 and np.issubdtype(arr.dtype, np.number):
                    return arr
        return None

    ref = {
        "x": pick(("x", "x_hat", "x_ref", "xgrid", "x_grid")),
        "tau": pick(("tau", "tau_hat", "tau_values")),
        "r_int": pick(("r_int", "rint", "r_int_ref", "interface", "r_interface")),
    }
    if ref["x"] is None or ref["r_int"] is None:
        print(
            "[postprocess] MATLAB file found, but no recognizable interface arrays "
            "(need x and r_int-like variables)."
        )
        return None
    print(f"[postprocess] MATLAB reference loaded from {path}")
    return ref


def select_reference_interface(ref: dict[str, Any] | None,
                               tau_value: float) -> tuple[np.ndarray, np.ndarray] | None:
    if ref is None:
        return None
    x = np.asarray(ref["x"]).squeeze()
    r_int = np.asarray(ref["r_int"]).squeeze()
    tau_ref = None if ref.get("tau") is None else np.asarray(ref["tau"]).squeeze()

    if x.ndim != 1:
        x = x.reshape(-1)
    if r_int.ndim == 1:
        return x, r_int.reshape(-1)
    if tau_ref is None or tau_ref.ndim == 0:
        idx = 0
        if r_int.shape[0] == x.size:
            return x, r_int[:, idx]
        if r_int.shape[-1] == x.size:
            return x, r_int[idx, :]
        return None

    idx = int(np.argmin(np.abs(tau_ref.reshape(-1) - tau_value)))
    if r_int.shape[0] == tau_ref.size and r_int.shape[1] == x.size:
        return x, r_int[idx, :]
    if r_int.shape[1] == tau_ref.size and r_int.shape[0] == x.size:
        return x, r_int[:, idx]
    if r_int.shape[-1] == x.size:
        return x, r_int[min(idx, r_int.shape[0] - 1), :]
    if r_int.shape[0] == x.size:
        return x, r_int[:, min(idx, r_int.shape[1] - 1)]
    return None


def plot_interface_profiles(field_by_time: dict[float, dict[str, np.ndarray]],
                            matlab_ref: dict[str, Any] | None,
                            output_dir: Path,
                            filename: str = "interface_profiles.png") -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for tau_value, fields in field_by_time.items():
        ax.plot(fields["x_line"], fields["r_int"], lw=2, label=f"PINN tau={tau_value:g}")
        ref_line = select_reference_interface(matlab_ref, tau_value)
        if ref_line is not None:
            ax.plot(ref_line[0], ref_line[1], "--", lw=1.5, label=f"MATLAB tau={tau_value:g}")
    ax.set_xlabel("x_hat")
    ax.set_ylabel("r_int_hat")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Interface position")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=180)
    plt.close(fig)


def plot_thickness_profiles(field_by_time: dict[float, dict[str, np.ndarray]],
                            run_cfg: Config,
                            matlab_ref: dict[str, Any] | None,
                            output_dir: Path,
                            filename: str = "thickness_profiles.png") -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for tau_value, fields in field_by_time.items():
        thickness = run_cfg.case.r_w - fields["r_int"]
        ax.plot(fields["x_line"], thickness, lw=2, label=f"PINN tau={tau_value:g}")
        ref_line = select_reference_interface(matlab_ref, tau_value)
        if ref_line is not None:
            ax.plot(
                ref_line[0],
                run_cfg.case.r_w - ref_line[1],
                "--",
                lw=1.5,
                label=f"MATLAB tau={tau_value:g}",
            )
    ax.set_xlabel("x_hat")
    ax.set_ylabel("r_w_hat - r_int_hat")
    ax.set_xlim(0.0, run_cfg.case.L)
    ax.set_ylim(0.0, 0.5 * run_cfg.case.r_w)
    ax.set_title("Nondimensional deposit thickness")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=180)
    plt.close(fig)


def plot_field_snapshots(tau_value: float,
                         fields: dict[str, np.ndarray],
                         output_dir: Path,
                         filename_prefix: str = "fields") -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharex=True, sharey=True)
    specs = (
        ("Theta_f", "Theta_f", "inferno"),
        ("Theta_dep", "Theta_dep", "viridis"),
        ("u_x", "u_x", "coolwarm"),
    )
    for ax, (key, title, cmap) in zip(axes, specs):
        im = ax.contourf(fields["x"], fields["r"], fields[key], levels=40, cmap=cmap)
        ax.plot(fields["x_line"], fields["r_int"], color="cyan", lw=1.4)
        ax.set_title(title)
        ax.set_xlabel("x_hat")
        fig.colorbar(im, ax=ax, shrink=0.82)
    axes[0].set_ylabel("r_hat")
    fig.suptitle(f"PINN fields at tau={tau_value:g}")
    fig.tight_layout()
    fig.savefig(output_dir / f"{filename_prefix}_tau{tau_value:.3f}.png", dpi=180)
    plt.close(fig)


def plot_pressure_profiles(field_by_time: dict[float, dict[str, np.ndarray]],
                           output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for tau_value, fields in field_by_time.items():
        centerline = fields["p"][0, :]
        ax.plot(fields["x_line"], centerline, lw=2, label=f"tau={tau_value:g}")
    ax.set_xlabel("x_hat")
    ax.set_ylabel("p_hat(r=0, x)")
    ax.set_title("Centerline pressure profile")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "pressure_profiles.png", dpi=180)
    plt.close(fig)


def _numeric_column(rows: list[dict[str, str]], key: str) -> np.ndarray:
    values: list[float] = []
    for row in rows:
        raw = row.get(key, "")
        try:
            values.append(float(raw))
        except (TypeError, ValueError):
            values.append(np.nan)
    return np.array(values, dtype=float)


def plot_loss_curve(loss_history: Path | None,
                    output_dir: Path,
                    last_iteration: int | None = None) -> None:
    if loss_history is None:
        print("[postprocess] loss_history.csv not found; skipping loss_curve.png")
        return
    with open(loss_history, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"[postprocess] empty loss history; skipping loss_curve.png: {loss_history}")
        return

    iterations = _numeric_column(rows, "iteration")
    total = _numeric_column(rows, "total")
    finite = np.isfinite(iterations) & np.isfinite(total)
    if not finite.any():
        print(f"[postprocess] no plottable loss rows in {loss_history}")
        return

    term_names = [
        name for name in (rows[0].keys() if rows else [])
        if name not in {"iteration", "phase", "total"} and np.isfinite(_numeric_column(rows, name)).any()
    ]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax1.semilogy(iterations[finite], np.maximum(total[finite], 1e-30), lw=2, label="total")
    ax1.set_ylabel("total loss")
    ax1.set_title("Loss trajectory")
    ax1.grid(True, alpha=0.4, which="both")
    if last_iteration is not None:
        ax1.axvline(last_iteration, color="k", linestyle="--", linewidth=1.2,
                    label=f"last.pth iter {last_iteration}")
    ax1.legend(fontsize=8, loc="best")

    for name in term_names:
        values = _numeric_column(rows, name)
        mask = np.isfinite(iterations) & np.isfinite(values) & (values > 0.0)
        if mask.any():
            ax2.semilogy(iterations[mask], np.maximum(values[mask], 1e-30), label=name)
    if last_iteration is not None:
        ax2.axvline(last_iteration, color="k", linestyle="--", linewidth=1.2)
    ax2.set_xlabel("iteration")
    ax2.set_ylabel("loss term")
    ax2.grid(True, alpha=0.4, which="both")
    if term_names:
        ax2.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "loss_curve.png", dpi=180)
    plt.close(fig)


def compute_residual_map(model: PINNSolidification,
                         run_cfg: Config,
                         grid_nr: int,
                         grid_nx: int,
                         tau_value: float,
                         device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r_min = max(0.0, min(float(run_cfg.sampling.r_min_interior), run_cfg.case.r_w))
    r = torch.linspace(r_min, run_cfg.case.r_w, grid_nr, device=device)
    x = torch.linspace(0.0, run_cfg.case.L, grid_nx, device=device)
    rr, xx = torch.meshgrid(r, x, indexing="ij")
    tt = torch.full_like(rr, tau_value)
    r = rr.reshape(-1).detach().requires_grad_(True)
    x = xx.reshape(-1).detach().requires_grad_(True)
    tau = tt.reshape(-1).detach().requires_grad_(True)

    out = model(r, x, tau)
    out["H"] = model.smooth_heaviside(
        r,
        out["r_int"].detach(),
        run_cfg.network.interface_epsilon,
    )
    residuals = [
        res_mass(out, r, x, tau),
        res_mom_x(out, r, x, tau, run_cfg.case),
        res_mom_r(out, r, x, tau, run_cfg.case),
        res_energy_fluid(out, r, x, tau, run_cfg.case),
        res_energy_dep(out, r, x, tau, run_cfg.case),
    ]
    stacked = torch.stack([res.pow(2) for res in residuals], dim=0)
    residual_norm = torch.sqrt(stacked.mean(dim=0) + 1e-30)
    return (
        rr.detach().cpu().numpy(),
        xx.detach().cpu().numpy(),
        residual_norm.reshape(grid_nr, grid_nx).detach().cpu().numpy(),
    )


def plot_residual_map(tau_value: float,
                      rr: np.ndarray,
                      xx: np.ndarray,
                      residual_norm: np.ndarray,
                      fields: dict[str, np.ndarray],
                      output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    safe = np.log10(np.maximum(residual_norm, 1e-12))
    im = ax.contourf(xx, rr, safe, levels=40, cmap="magma")
    ax.plot(fields["x_line"], fields["r_int"], color="cyan", lw=1.4)
    ax.set_xlabel("x_hat")
    ax.set_ylabel("r_hat")
    ax.set_title(f"log10 PDE residual norm at tau={tau_value:g}")
    fig.colorbar(im, ax=ax, shrink=0.86)
    fig.tight_layout()
    fig.savefig(output_dir / f"residual_map_tau{tau_value:.3f}.png", dpi=180)
    plt.close(fig)


def summarize_matlab_error(field_by_time: dict[float, dict[str, np.ndarray]],
                           matlab_ref: dict[str, Any] | None,
                           output_dir: Path) -> None:
    rows = ["tau,pinn_vs_matlab_rint_l2"]
    if matlab_ref is None:
        rows.append("no_reference,nan")
    else:
        for tau_value, fields in field_by_time.items():
            ref_line = select_reference_interface(matlab_ref, tau_value)
            if ref_line is None:
                rows.append(f"{tau_value},nan")
                continue
            x_ref, r_ref = ref_line
            r_interp = np.interp(x_ref, fields["x_line"], fields["r_int"])
            denom = np.linalg.norm(r_ref)
            err = np.nan if denom == 0 else np.linalg.norm(r_interp - r_ref) / denom
            rows.append(f"{tau_value},{err:.8e}" if math.isfinite(err) else f"{tau_value},nan")
    (output_dir / "validation_summary.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    taus = parse_taus(args.taus)
    device = choose_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    primary_checkpoint = resolve_primary_checkpoint(args.checkpoint)
    last_checkpoint = resolve_last_checkpoint(primary_checkpoint, args.last_checkpoint)
    last_iteration = checkpoint_iteration(last_checkpoint)

    model, run_cfg = load_model(primary_checkpoint, device)
    matlab_ref = load_matlab_reference(args.matlab_ref)
    print(
        f"[postprocess] checkpoint={primary_checkpoint} device={device} "
        f"grid={args.grid_nr}x{args.grid_nx} case={run_cfg.case.name}",
        flush=True,
    )

    field_by_time: dict[float, dict[str, np.ndarray]] = {}
    for tau_value in taus:
        fields = evaluate_fields(model, run_cfg, args.grid_nr, args.grid_nx, tau_value, device)
        field_by_time[tau_value] = fields
        plot_field_snapshots(tau_value, fields, args.output_dir)
        if not args.skip_residual_maps:
            rr, xx, residual_norm = compute_residual_map(
                model, run_cfg, args.grid_nr, args.grid_nx, tau_value, device
            )
            plot_residual_map(tau_value, rr, xx, residual_norm, fields, args.output_dir)

    plot_interface_profiles(field_by_time, matlab_ref, args.output_dir)
    plot_thickness_profiles(field_by_time, run_cfg, matlab_ref, args.output_dir)
    plot_pressure_profiles(field_by_time, args.output_dir)
    plot_loss_curve(find_loss_history(primary_checkpoint, run_cfg), args.output_dir, last_iteration)
    summarize_matlab_error(field_by_time, matlab_ref, args.output_dir)

    if last_checkpoint is not None:
        last_model, last_run_cfg = load_model(last_checkpoint, device)
        last_field_by_time: dict[float, dict[str, np.ndarray]] = {}
        for tau_value in taus:
            fields = evaluate_fields(last_model, last_run_cfg, args.grid_nr, args.grid_nx, tau_value, device)
            last_field_by_time[tau_value] = fields
            plot_field_snapshots(tau_value, fields, args.output_dir, filename_prefix="fields_last")
        plot_interface_profiles(
            last_field_by_time,
            matlab_ref,
            args.output_dir,
            filename="interface_profiles_last.png",
        )
        plot_thickness_profiles(
            last_field_by_time,
            last_run_cfg,
            matlab_ref,
            args.output_dir,
            filename="thickness_profiles_last.png",
        )

    print(f"[postprocess] plots saved to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
