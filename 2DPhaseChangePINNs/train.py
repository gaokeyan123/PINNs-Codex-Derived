"""
Three-phase training driver for the 2D pipe-solidification PINN.

Default settings follow config.py:
  Phase 1: BC/IC pre-training with Adam
  Phase 2: full physics with Adam + cosine annealing
  Phase 3: full physics with L-BFGS refinement

Use --smoke for a tiny CPU-safe run that verifies the complete pipeline.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import Config, cfg
from losses import compute_loss, format_loss_line
from network import PINNSolidification, count_parameters
from sampling import build_batch, resample_interface


MODEL_VERSION = "1.0"
LOSS_FIELDS = [
    "bc_ic",
    "energy_dep",
    "energy_fluid",
    "mass",
    "mom_r",
    "mom_x",
    "stefan",
    "T_cont",
    "total",
]


@dataclass
class TargetStop:
    thickness: float
    mode: str
    t_value: float
    grid_n: int
    check_every: int
    min_iter: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the pipe-solidification PINN.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--phase1-iters", type=int, default=None)
    parser.add_argument("--phase2-iters", type=int, default=None)
    parser.add_argument("--phase3-iters", type=int, default=None)
    parser.add_argument("--plot-every", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=None)
    parser.add_argument("--n-interior", type=int, default=None)
    parser.add_argument("--n-interface", type=int, default=None)
    parser.add_argument("--n-boundary", type=int, default=None)
    parser.add_argument("--n-ic", type=int, default=None)
    parser.add_argument("--target-solid-thickness", type=float, default=None)
    parser.add_argument("--target-thickness-mode", choices=("max", "mean", "min"), default="mean")
    parser.add_argument("--target-time", type=float, default=None)
    parser.add_argument("--target-grid-n", type=int, default=128)
    parser.add_argument("--target-check-every", type=int, default=None)
    parser.add_argument("--target-min-iter", type=int, default=0)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true", help="Run a tiny end-to-end verification job.")
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    if requested == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_plain(obj: Any) -> Any:
    """Convert dataclasses and Paths to checkpoint-safe plain Python objects."""
    if is_dataclass(obj):
        return to_plain(asdict(obj))
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain(v) for v in obj]
    return obj


def apply_cli_overrides(base: Config, args: argparse.Namespace) -> Config:
    run_cfg = copy.deepcopy(base)
    if args.seed is not None:
        run_cfg.seed = args.seed
    if args.output_dir is not None:
        run_cfg.output_dir = args.output_dir
    if args.checkpoint_dir is not None:
        run_cfg.training.checkpoint_dir = args.checkpoint_dir
    if args.phase1_iters is not None:
        run_cfg.training.phase1_iters = args.phase1_iters
    if args.phase2_iters is not None:
        run_cfg.training.phase2_iters = args.phase2_iters
    if args.phase3_iters is not None:
        run_cfg.training.phase3_iters = args.phase3_iters
    if args.plot_every is not None:
        run_cfg.training.plot_every = args.plot_every
    if args.checkpoint_every is not None:
        run_cfg.training.checkpoint_every = args.checkpoint_every
    if args.n_interior is not None:
        run_cfg.sampling.N_interior = args.n_interior
    if args.n_interface is not None:
        run_cfg.sampling.N_interface = args.n_interface
    if args.n_boundary is not None:
        run_cfg.sampling.N_boundary = args.n_boundary
    if args.n_ic is not None:
        run_cfg.sampling.N_ic = args.n_ic

    if args.smoke:
        run_cfg.sampling.N_interior = 32
        run_cfg.sampling.N_interface = 16
        run_cfg.sampling.N_boundary = 12
        run_cfg.sampling.N_ic = 16
        run_cfg.sampling.resample_every = 2
        run_cfg.training.phase1_iters = 2 if args.phase1_iters is None else args.phase1_iters
        run_cfg.training.phase2_iters = 4 if args.phase2_iters is None else args.phase2_iters
        run_cfg.training.phase3_iters = 1 if args.phase3_iters is None else args.phase3_iters
        run_cfg.training.log_every = 1
        run_cfg.training.plot_every = 2 if args.plot_every is None else args.plot_every
        run_cfg.training.checkpoint_every = 10 if args.checkpoint_every is None else args.checkpoint_every
    return run_cfg


def move_points(points: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().to(device).requires_grad_(True)
        for key, value in points.items()
    }


def move_batch(batch: dict[str, dict[str, torch.Tensor]],
               device: torch.device) -> dict[str, dict[str, torch.Tensor]]:
    return {key: move_points(points, device) for key, points in batch.items()}


def build_training_batch(model: PINNSolidification,
                         run_cfg: Config,
                         seed: int,
                         device: torch.device) -> dict[str, dict[str, torch.Tensor]]:
    return move_batch(build_batch(model, run_cfg, seed=seed), device)


def replace_interface_points(batch: dict[str, dict[str, torch.Tensor]],
                             model: PINNSolidification,
                             run_cfg: Config,
                             seed: int,
                             device: torch.device) -> None:
    batch["interface"] = move_points(
        resample_interface(model, run_cfg.sampling.N_interface, run_cfg, seed=seed),
        device,
    )


def ensure_dirs(run_cfg: Config) -> None:
    run_cfg.output_dir.mkdir(parents=True, exist_ok=True)
    run_cfg.training.checkpoint_dir.mkdir(parents=True, exist_ok=True)


def checkpoint_path(run_cfg: Config, name: str) -> Path:
    return run_cfg.training.checkpoint_dir / name


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path: Path,
                    model: PINNSolidification,
                    run_cfg: Config,
                    phase: str,
                    iteration: int,
                    best_loss: float,
                    log: dict[str, float],
                    optimizer: torch.optim.Optimizer | None = None,
                    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
                    optimizer_name: str | None = None) -> Path:
    payload = {
        "model_version": MODEL_VERSION,
        "phase": phase,
        "iteration": iteration,
        "best_loss": best_loss,
        "last_log": log,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "optimizer_name": optimizer_name,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "config": to_plain(run_cfg),
        "rng_state": capture_rng_state(),
        "saved_at_unix": time.time(),
    }
    target = unique_checkpoint_path(path, iteration) if path.exists() else path
    if target != path:
        print(
            f"[checkpoint] target exists, wrote fallback {target.name}",
            flush=True,
        )
    torch.save(payload, target)
    return target


def unique_checkpoint_path(path: Path, iteration: int) -> Path:
    stamp = int(time.time())
    candidate = path.with_name(f"{path.stem}_iter{iteration:05d}_{stamp}{path.suffix}")
    idx = 1
    while candidate.exists():
        candidate = path.with_name(
            f"{path.stem}_iter{iteration:05d}_{stamp}_{idx}{path.suffix}"
        )
        idx += 1
    return candidate


def cleanup_old_best_checkpoints(run_cfg: Config, keep_name: str) -> None:
    pattern = f"pinns_v{MODEL_VERSION}_ep*_loss*.pth"
    for path in run_cfg.training.checkpoint_dir.glob(pattern):
        if path.name == keep_name:
            continue
        try:
            path.unlink()
        except OSError:
            pass


def load_checkpoint(path: Path,
                    model: PINNSolidification,
                    device: torch.device) -> dict[str, Any]:
    payload = torch.load(path, map_location=device, weights_only=False)
    state = payload.get("model_state", payload.get("model_state_dict", payload))
    model.load_state_dict(state)
    restore_rng_state(payload.get("rng_state"))
    return payload


def update_best_models_md(run_cfg: Config,
                          filename: str,
                          loss_value: float,
                          iteration: int,
                          phase: str) -> None:
    path = run_cfg.training.checkpoint_dir / "BEST_MODELS.md"
    text = (
        "# Best Model Checkpoints\n\n"
        "This file is tracked by git. The `.pth` weight files are **not** tracked "
        "(see root `.gitignore`).\n\n"
        "## Naming convention\n\n"
        "```\n"
        "pinns_v{MAJOR}.{MINOR}_ep{EPOCH:05d}_loss{VAL:.4e}.pth\n"
        "```\n\n"
        "| Filename | Val Loss | Epoch | Notes |\n"
        "|---|---|---|---|\n"
        f"| `{filename}` | {loss_value:.6e} | {iteration} | Best {phase} checkpoint from training run |\n"
    )
    path.write_text(text, encoding="utf-8")


def save_best_and_latest(model: PINNSolidification,
                         run_cfg: Config,
                         phase: str,
                         iteration: int,
                         best_loss: float,
                         log: dict[str, float],
                         optimizer: torch.optim.Optimizer | None,
                         scheduler: torch.optim.lr_scheduler.LRScheduler | None,
                         optimizer_name: str | None,
                         save_latest: bool = False) -> float:
    total = log["total"]
    if math.isfinite(total) and total < best_loss:
        best_loss = total
        best_name = f"pinns_v{MODEL_VERSION}_ep{iteration:05d}_loss{total:.4e}.pth"
        best_path = save_checkpoint(
            checkpoint_path(run_cfg, best_name),
            model,
            run_cfg,
            phase,
            iteration,
            best_loss,
            log,
            optimizer=optimizer,
            scheduler=scheduler,
            optimizer_name=optimizer_name,
        )
        cleanup_old_best_checkpoints(run_cfg, best_path.name)
        update_best_models_md(run_cfg, best_path.name, total, iteration, phase)

    if save_latest:
        latest = checkpoint_path(run_cfg, run_cfg.training.latest_checkpoint)
        save_checkpoint(
            latest, model, run_cfg, phase, iteration, best_loss, log,
            optimizer=optimizer, scheduler=scheduler, optimizer_name=optimizer_name,
        )
    return best_loss


def append_phase_csv(csv_path: Path,
                     iteration: int,
                     phase: str,
                     log: dict[str, float],
                     write_header: bool = False) -> None:
    fields = ["iteration", "phase"] + sorted(log.keys())
    row = {"iteration": iteration, "phase": phase, **log}
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def plot_monitor_snapshot(model: PINNSolidification,
                          run_cfg: Config,
                          iteration: int,
                          device: torch.device) -> None:
    model.eval()
    nx, nr = 120, 80
    x = torch.linspace(0.0, run_cfg.case.L, nx, device=device)
    r = torch.linspace(0.0, run_cfg.case.r_w, nr, device=device)
    rr, xx = torch.meshgrid(r, x, indexing="ij")
    tt = torch.full_like(rr, 0.5 * run_cfg.case.t_end)

    with torch.no_grad():
        out = model(rr.reshape(-1), xx.reshape(-1), tt.reshape(-1))
        theta = out["Theta_f"].reshape(nr, nx).detach().cpu().numpy()
        r_int = model(
            torch.zeros_like(x),
            x,
            torch.full_like(x, 0.5 * run_cfg.case.t_end),
        )["r_int"].detach().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(x.detach().cpu().numpy(), r_int, lw=2)
    axes[0].set_xlabel("x_hat")
    axes[0].set_ylabel("r_int_hat")
    axes[0].set_ylim(0.0, run_cfg.case.r_w * 1.05)
    axes[0].set_title("Interface at t_mid")

    im = axes[1].contourf(
        xx.detach().cpu().numpy(),
        rr.detach().cpu().numpy(),
        theta,
        levels=32,
        cmap="inferno",
    )
    axes[1].plot(x.detach().cpu().numpy(), r_int, color="cyan", lw=1.5)
    axes[1].set_xlabel("x_hat")
    axes[1].set_ylabel("r_hat")
    axes[1].set_title("Theta_f at t_mid")
    fig.colorbar(im, ax=axes[1], shrink=0.85)
    fig.suptitle(f"Training monitor, iter {iteration}")
    fig.tight_layout()
    fig.savefig(run_cfg.output_dir / f"monitor_iter{iteration:05d}.png", dpi=150)
    plt.close(fig)
    model.train()


def should_log(iteration: int, run_cfg: Config) -> bool:
    return iteration == 1 or iteration % run_cfg.training.log_every == 0


def should_plot(iteration: int, run_cfg: Config) -> bool:
    return run_cfg.training.plot_every > 0 and iteration % run_cfg.training.plot_every == 0


def should_checkpoint(iteration: int, run_cfg: Config) -> bool:
    return iteration == 1 or (
        run_cfg.training.checkpoint_every > 0
        and iteration % run_cfg.training.checkpoint_every == 0
    )


def check_log_is_finite(log: dict[str, float], phase: str, iteration: int) -> None:
    bad = {k: v for k, v in log.items() if not math.isfinite(v)}
    if bad:
        raise FloatingPointError(f"Non-finite loss in {phase} at iter {iteration}: {bad}")


def build_target_stop(args: argparse.Namespace, run_cfg: Config) -> TargetStop | None:
    if args.target_solid_thickness is None:
        return None
    check_every = (
        args.target_check_every
        if args.target_check_every is not None
        else max(1, run_cfg.training.log_every)
    )
    return TargetStop(
        thickness=args.target_solid_thickness,
        mode=args.target_thickness_mode,
        t_value=run_cfg.case.t_end if args.target_time is None else args.target_time,
        grid_n=args.target_grid_n,
        check_every=check_every,
        min_iter=args.target_min_iter,
    )


def interface_thickness_stats(model: PINNSolidification,
                              run_cfg: Config,
                              device: torch.device,
                              target: TargetStop) -> dict[str, float]:
    was_training = model.training
    model.eval()
    x = torch.linspace(0.0, run_cfg.case.L, target.grid_n, device=device)
    t = torch.full_like(x, target.t_value)
    r = torch.zeros_like(x)
    with torch.no_grad():
        r_int = model(r, x, t)["r_int"]
        thickness = run_cfg.case.r_w - r_int
    if was_training:
        model.train()
    return {
        "min": float(thickness.min().item()),
        "mean": float(thickness.mean().item()),
        "max": float(thickness.max().item()),
        "r_int_min": float(r_int.min().item()),
        "r_int_mean": float(r_int.mean().item()),
        "r_int_max": float(r_int.max().item()),
    }


def check_target_stop(model: PINNSolidification,
                      run_cfg: Config,
                      device: torch.device,
                      target: TargetStop | None,
                      phase: str,
                      iteration: int) -> bool:
    if target is None:
        return False
    if iteration < target.min_iter:
        return False
    if iteration == 1 or iteration % target.check_every != 0:
        return False
    stats = interface_thickness_stats(model, run_cfg, device, target)
    value = stats[target.mode]
    print(
        f"[target] iter {iteration} {phase} thickness "
        f"min/mean/max={stats['min']:.4f}/{stats['mean']:.4f}/{stats['max']:.4f} "
        f"at t={target.t_value:g} ({target.mode} target {target.thickness:.4f})",
        flush=True,
    )
    return value >= target.thickness


def run_phase1(model: PINNSolidification,
               run_cfg: Config,
               device: torch.device,
               csv_path: Path,
               start_iter: int,
               best_loss: float,
               resume_payload: dict[str, Any] | None,
               target: TargetStop | None) -> tuple[int, float, bool]:
    end_iter = run_cfg.training.phase1_iters
    if start_iter >= end_iter:
        return start_iter, best_loss, False

    optimizer = torch.optim.Adam(model.parameters(), lr=run_cfg.training.phase1_lr)
    if resume_payload and resume_payload.get("phase") == "phase1" and resume_payload.get("optimizer_state"):
        optimizer.load_state_dict(resume_payload["optimizer_state"])

    batch = build_training_batch(model, run_cfg, run_cfg.seed + start_iter, device)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0

    for iteration in range(start_iter + 1, end_iter + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, log = compute_loss(model, batch, run_cfg, physics_on=False)
        check_log_is_finite(log, "phase1", iteration)
        loss.backward()
        optimizer.step()

        append_phase_csv(csv_path, iteration, "phase1", log, write_header=write_header)
        write_header = False

        if should_log(iteration, run_cfg):
            print(format_loss_line(log, iteration, physics_on=False), flush=True)
        if should_plot(iteration, run_cfg):
            plot_monitor_snapshot(model, run_cfg, iteration, device)

        best_loss = save_best_and_latest(
            model, run_cfg, "phase1", iteration, best_loss, log,
            optimizer=optimizer, scheduler=None, optimizer_name="Adam",
            save_latest=(
                should_checkpoint(iteration, run_cfg)
                or iteration == end_iter
                or log["total"] < run_cfg.training.loss_target
            ),
        )
        if check_target_stop(model, run_cfg, device, target, "phase1", iteration):
            print(f"[stop] target solid thickness reached in phase1 at iter {iteration}", flush=True)
            return iteration, best_loss, True
        if log["total"] < run_cfg.training.loss_target:
            print(f"[stop] target loss reached in phase1 at iter {iteration}", flush=True)
            return iteration, best_loss, True
    return end_iter, best_loss, False


def run_phase2(model: PINNSolidification,
               run_cfg: Config,
               device: torch.device,
               csv_path: Path,
               start_iter: int,
               best_loss: float,
               resume_payload: dict[str, Any] | None,
               target: TargetStop | None) -> tuple[int, float, bool]:
    phase1_end = run_cfg.training.phase1_iters
    phase2_end = run_cfg.training.phase2_iters
    if start_iter >= phase2_end:
        return start_iter, best_loss, False

    first_iter = max(start_iter + 1, phase1_end + 1)
    if start_iter <= phase1_end:
        best_loss = float("inf")
    phase2_steps = max(phase2_end - phase1_end, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=run_cfg.training.phase2_lr_start)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=phase2_steps,
        eta_min=run_cfg.training.phase2_lr_end,
    )
    if resume_payload and resume_payload.get("phase") == "phase2":
        if resume_payload.get("optimizer_state"):
            optimizer.load_state_dict(resume_payload["optimizer_state"])
        if resume_payload.get("scheduler_state"):
            scheduler.load_state_dict(resume_payload["scheduler_state"])

    batch = build_training_batch(model, run_cfg, run_cfg.seed + first_iter, device)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0

    for iteration in range(first_iter, phase2_end + 1):
        if (iteration - phase1_end) == 1 or iteration % run_cfg.sampling.resample_every == 0:
            replace_interface_points(batch, model, run_cfg, run_cfg.seed + iteration, device)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, log = compute_loss(model, batch, run_cfg, physics_on=True)
        check_log_is_finite(log, "phase2", iteration)
        loss.backward()
        optimizer.step()
        scheduler.step()

        append_phase_csv(csv_path, iteration, "phase2", log, write_header=write_header)
        write_header = False

        if should_log(iteration, run_cfg):
            print(format_loss_line(log, iteration, physics_on=True), flush=True)
        if should_plot(iteration, run_cfg):
            plot_monitor_snapshot(model, run_cfg, iteration, device)

        best_loss = save_best_and_latest(
            model, run_cfg, "phase2", iteration, best_loss, log,
            optimizer=optimizer, scheduler=scheduler, optimizer_name="Adam",
            save_latest=(
                should_checkpoint(iteration, run_cfg)
                or iteration == phase2_end
                or log["total"] < run_cfg.training.loss_target
            ),
        )
        if check_target_stop(model, run_cfg, device, target, "phase2", iteration):
            print(f"[stop] target solid thickness reached in phase2 at iter {iteration}", flush=True)
            return iteration, best_loss, True
        if log["total"] < run_cfg.training.loss_target:
            print(f"[stop] target loss reached in phase2 at iter {iteration}", flush=True)
            return iteration, best_loss, True
    return phase2_end, best_loss, False


def run_phase3(model: PINNSolidification,
               run_cfg: Config,
               device: torch.device,
               csv_path: Path,
               start_iter: int,
               best_loss: float,
               resume_payload: dict[str, Any] | None,
               target: TargetStop | None) -> tuple[int, float, bool]:
    phase2_end = run_cfg.training.phase2_iters
    phase3_end = phase2_end + run_cfg.training.phase3_iters
    if start_iter >= phase3_end:
        return start_iter, best_loss, False

    first_iter = max(start_iter + 1, phase2_end + 1)
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=1,
        max_eval=5,
        tolerance_grad=1e-9,
        tolerance_change=1e-11,
        history_size=50,
        line_search_fn="strong_wolfe",
    )
    if resume_payload and resume_payload.get("phase") == "phase3" and resume_payload.get("optimizer_state"):
        optimizer.load_state_dict(resume_payload["optimizer_state"])

    batch = build_training_batch(model, run_cfg, run_cfg.seed + first_iter, device)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0

    for iteration in range(first_iter, phase3_end + 1):
        if (iteration - phase2_end) == 1 or iteration % run_cfg.sampling.resample_every == 0:
            replace_interface_points(batch, model, run_cfg, run_cfg.seed + iteration, device)

        closure_log: dict[str, float] = {}

        def closure() -> torch.Tensor:
            optimizer.zero_grad(set_to_none=True)
            loss, log = compute_loss(model, batch, run_cfg, physics_on=True)
            check_log_is_finite(log, "phase3", iteration)
            loss.backward()
            closure_log.clear()
            closure_log.update(log)
            return loss

        model.train()
        optimizer.step(closure)
        log = dict(closure_log)
        if not log:
            _, log = compute_loss(model, batch, run_cfg, physics_on=True)
        check_log_is_finite(log, "phase3", iteration)

        append_phase_csv(csv_path, iteration, "phase3", log, write_header=write_header)
        write_header = False

        if should_log(iteration, run_cfg):
            print(format_loss_line(log, iteration, physics_on=True), flush=True)
        if should_plot(iteration, run_cfg):
            plot_monitor_snapshot(model, run_cfg, iteration, device)

        best_loss = save_best_and_latest(
            model, run_cfg, "phase3", iteration, best_loss, log,
            optimizer=optimizer, scheduler=None, optimizer_name="LBFGS",
            save_latest=(
                should_checkpoint(iteration, run_cfg)
                or iteration == phase3_end
                or log["total"] < run_cfg.training.loss_target
            ),
        )
        if check_target_stop(model, run_cfg, device, target, "phase3", iteration):
            print(f"[stop] target solid thickness reached in phase3 at iter {iteration}", flush=True)
            return iteration, best_loss, True
        if log["total"] < run_cfg.training.loss_target:
            print(f"[stop] target loss reached in phase3 at iter {iteration}", flush=True)
            return iteration, best_loss, True
    return phase3_end, best_loss, False


def save_run_config(run_cfg: Config) -> None:
    path = run_cfg.output_dir / "run_config.json"
    path.write_text(json.dumps(to_plain(run_cfg), indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_cfg = apply_cli_overrides(cfg, args)
    device = choose_device(args.device)
    ensure_dirs(run_cfg)
    seed_everything(run_cfg.seed)
    save_run_config(run_cfg)
    target = build_target_stop(args, run_cfg)

    model = PINNSolidification(run_cfg.network, run_cfg.case, seed=run_cfg.seed).to(device)
    print(f"[train] device={device} seed={run_cfg.seed} params={count_parameters(model):,}", flush=True)
    print(
        f"[train] phase ends: p1={run_cfg.training.phase1_iters}, "
        f"p2={run_cfg.training.phase2_iters}, "
        f"p3={run_cfg.training.phase2_iters + run_cfg.training.phase3_iters}",
        flush=True,
    )
    print(
        f"[train] case: Ste={run_cfg.case.Ste:g} Pe={run_cfg.case.Pe:g} "
        f"Re={run_cfg.case.Re:g} k_ratio={run_cfg.case.k_ratio:g} "
        f"t_end={run_cfg.case.t_end:g}",
        flush=True,
    )
    if target is not None:
        print(
            f"[train] target: {target.mode} solid thickness >= {target.thickness:g} "
            f"at t={target.t_value:g}, min_iter={target.min_iter}",
            flush=True,
        )

    resume_payload: dict[str, Any] | None = None
    start_iter = 0
    best_loss = float("inf")
    if args.resume is not None:
        resume_payload = load_checkpoint(args.resume, model, device)
        start_iter = int(resume_payload.get("iteration", 0))
        best_loss = float(resume_payload.get("best_loss", best_loss))
        print(f"[resume] loaded {args.resume} at iter {start_iter}", flush=True)

    csv_path = run_cfg.output_dir / "loss_history.csv"
    t0 = time.time()
    start_iter, best_loss, stopped = run_phase1(
        model, run_cfg, device, csv_path, start_iter, best_loss, resume_payload, target
    )
    if not stopped:
        start_iter, best_loss, stopped = run_phase2(
            model, run_cfg, device, csv_path, start_iter, best_loss, resume_payload, target
        )
    if not stopped:
        start_iter, best_loss, stopped = run_phase3(
            model, run_cfg, device, csv_path, start_iter, best_loss, resume_payload, target
        )
    elapsed = time.time() - t0
    print(
        f"[done] stopped at iter {start_iter}, best_loss={best_loss:.6e}, "
        f"elapsed={elapsed:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
