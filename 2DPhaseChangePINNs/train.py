"""
Three-phase training driver for the 2D pipe-solidification PINN.

Default settings follow config.py:
  Phase 1: BC/IC pre-training with Adam
  Phase 2: full physics with Adam + cosine annealing
  Phase 3: full physics with L-BFGS refinement

Use --smoke for a tiny CPU-safe run that verifies the complete pipeline.
"""

from __future__ import annotations  # Keep modern type-annotation behavior available.

import argparse  # Import the CLI argument parser.
import copy  # Import deepcopy support for config cloning.
import csv  # Import CSV writing support for loss logs.
import json  # Import JSON support for saved run config files.
import math  # Import math helpers for numerical checks.
import random  # Import Python random seeding support.
import time  # Import wall-clock timing utilities.
from dataclasses import asdict, dataclass, is_dataclass  # Import dataclass helpers used by configs and target-stop state.
from pathlib import Path  # Import Path for filesystem-safe paths.
from typing import Any  # Import flexible type hints.

import numpy as np  # Import NumPy for RNG seeding and serialized state.
import torch  # Import PyTorch for model training and checkpoints.
import matplotlib  # Import Matplotlib backend control.

matplotlib.use("Agg")  # Use a non-interactive plotting backend for saved images.
import matplotlib.pyplot as plt  # Import plotting functions for monitor snapshots.

from config import Config, cfg  # Import the project configuration object and type.
from losses import compute_loss, format_loss_line  # Import loss assembly and display helpers.
from network import PINNSolidification, count_parameters  # Import the PINN model and parameter counter.
from sampling import build_batch, resample_interface  # Import collocation batch sampling helpers.


MODEL_VERSION = "1.0"  # Set the model/checkpoint version label.
LOSS_FIELDS = [  # Start the ordered list of loss fields used by some logs.
    "bc_ic",  # String literal entry in a list or dictionary.
    "energy_dep",  # String literal entry in a list or dictionary.
    "energy_fluid",  # String literal entry in a list or dictionary.
    "mass",  # String literal entry in a list or dictionary.
    "mom_r",  # String literal entry in a list or dictionary.
    "mom_x",  # String literal entry in a list or dictionary.
    "stefan",  # String literal entry in a list or dictionary.
    "T_cont",  # String literal entry in a list or dictionary.
    "total",  # String literal entry in a list or dictionary.
]  # Close a multi-line list.


@dataclass  # Mark the next class as a lightweight data container.
class TargetStop:  # Define the TargetStop data structure.
    thickness: float  # Training-driver logic line.
    mode: str  # Training-driver logic line.
    t_value: float  # Training-driver logic line.
    grid_n: int  # Training-driver logic line.
    check_every: int  # Training-driver logic line.
    min_iter: int  # Training-driver logic line.


def parse_args() -> argparse.Namespace:  # Define the parse_args helper function.
    parser = argparse.ArgumentParser(description="Train the pipe-solidification PINN.")  # Create the command-line parser.
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")  # Register the --device command-line option.
    parser.add_argument("--seed", type=int, default=None)  # Register the --seed command-line option.
    parser.add_argument("--output-dir", type=Path, default=None)  # Register the --output-dir command-line option.
    parser.add_argument("--checkpoint-dir", type=Path, default=None)  # Register the --checkpoint-dir command-line option.
    parser.add_argument("--phase1-iters", type=int, default=None)  # Register the --phase1-iters command-line option.
    parser.add_argument("--phase2-iters", type=int, default=None)  # Register the --phase2-iters command-line option.
    parser.add_argument("--phase3-iters", type=int, default=None)  # Register the --phase3-iters command-line option.
    parser.add_argument("--phase1-lr", type=float, default=None)  # Register the --phase1-lr command-line option.
    parser.add_argument("--phase2-lr-start", type=float, default=None)  # Register the --phase2-lr-start command-line option.
    parser.add_argument("--phase2-lr-end", type=float, default=None)  # Register the --phase2-lr-end command-line option.
    parser.add_argument("--plot-every", type=int, default=None)  # Register the --plot-every command-line option.
    parser.add_argument("--checkpoint-every", type=int, default=None)  # Register the --checkpoint-every command-line option.
    parser.add_argument("--n-interior", type=int, default=None)  # Register the --n-interior command-line option.
    parser.add_argument("--n-interface", type=int, default=None)  # Register the --n-interface command-line option.
    parser.add_argument("--n-boundary", type=int, default=None)  # Register the --n-boundary command-line option.
    parser.add_argument("--n-ic", type=int, default=None)  # Register the --n-ic command-line option.
    parser.add_argument("--r-min-interior", type=float, default=None)  # Register the --r-min-interior command-line option.
    parser.add_argument("--resample-all-every", type=int, default=None)  # Register the --resample-all-every command-line option.
    parser.add_argument("--target-solid-thickness", type=float, default=None)  # Register the --target-solid-thickness command-line option.
    parser.add_argument("--target-thickness-mode", choices=("max", "mean", "min"), default="mean")  # Register the --target-thickness-mode command-line option.
    parser.add_argument("--target-time", type=float, default=None)  # Register the --target-time command-line option.
    parser.add_argument("--target-grid-n", type=int, default=128)  # Register the --target-grid-n command-line option.
    parser.add_argument("--target-check-every", type=int, default=None)  # Register the --target-check-every command-line option.
    parser.add_argument("--target-min-iter", type=int, default=0)  # Register the --target-min-iter command-line option.
    parser.add_argument("--resume", type=Path, default=None)  # Register the --resume command-line option.
    parser.add_argument("--resume-weights-only", action="store_true")  # Register the --resume-weights-only command-line option.
    parser.add_argument("--smoke", action="store_true", help="Run a tiny end-to-end verification job.")  # Register the --smoke command-line option.
    return parser.parse_args()  # Parse CLI arguments and return them.


def choose_device(requested: str) -> torch.device:  # Define the choose_device helper function.
    if requested == "cuda":  # Handle explicit CUDA selection.
        if not torch.cuda.is_available():  # Branch only when this condition is true.
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")  # Stop execution with an explicit error.
        return torch.device("cuda")  # Return the selected PyTorch device.
    if requested == "auto" and torch.cuda.is_available():  # Prefer CUDA automatically when it is available.
        return torch.device("cuda")  # Return the selected PyTorch device.
    return torch.device("cpu")  # Return the selected PyTorch device.


def seed_everything(seed: int) -> None:  # Define the seed_everything helper function.
    random.seed(seed)  # Seed Python random number generation.
    np.random.seed(seed)  # Seed NumPy random number generation.
    torch.manual_seed(seed)  # Seed PyTorch CPU random number generation.
    if torch.cuda.is_available():  # Run CUDA-specific logic only when a GPU is available.
        torch.cuda.manual_seed_all(seed)  # Seed all CUDA random number generators.


def to_plain(obj: Any) -> Any:  # Define the to_plain helper function.
    """Convert dataclasses and Paths to checkpoint-safe plain Python objects."""
    if is_dataclass(obj):  # Branch only when this condition is true.
        return to_plain(asdict(obj))  # Return this value to the caller.
    if isinstance(obj, Path):  # Branch only when this condition is true.
        return str(obj)  # Return this value to the caller.
    if isinstance(obj, dict):  # Branch only when this condition is true.
        return {k: to_plain(v) for k, v in obj.items()}  # Return this value to the caller.
    if isinstance(obj, (list, tuple)):  # Branch only when this condition is true.
        return [to_plain(v) for v in obj]  # Return this value to the caller.
    return obj  # Return this value to the caller.


def apply_cli_overrides(base: Config, args: argparse.Namespace) -> Config:  # Define the apply_cli_overrides helper function.
    run_cfg = copy.deepcopy(base)  # Clone the default config so CLI overrides do not mutate globals.
    if args.seed is not None:  # Check whether CLI option seed was supplied.
        run_cfg.seed = args.seed  # Update the run-specific configuration.
    if args.output_dir is not None:  # Check whether CLI option output_dir was supplied.
        run_cfg.output_dir = args.output_dir  # Update the run-specific configuration.
    if args.checkpoint_dir is not None:  # Check whether CLI option checkpoint_dir was supplied.
        run_cfg.training.checkpoint_dir = args.checkpoint_dir  # Update the run-specific configuration.
    if args.phase1_iters is not None:  # Check whether CLI option phase1_iters was supplied.
        run_cfg.training.phase1_iters = args.phase1_iters  # Update the run-specific configuration.
    if args.phase2_iters is not None:  # Check whether CLI option phase2_iters was supplied.
        run_cfg.training.phase2_iters = args.phase2_iters  # Update the run-specific configuration.
    if args.phase3_iters is not None:  # Check whether CLI option phase3_iters was supplied.
        run_cfg.training.phase3_iters = args.phase3_iters  # Update the run-specific configuration.
    if args.phase1_lr is not None:  # Check whether CLI option phase1_lr was supplied.
        run_cfg.training.phase1_lr = args.phase1_lr  # Update the run-specific configuration.
    if args.phase2_lr_start is not None:  # Check whether CLI option phase2_lr_start was supplied.
        run_cfg.training.phase2_lr_start = args.phase2_lr_start  # Update the run-specific configuration.
    if args.phase2_lr_end is not None:  # Check whether CLI option phase2_lr_end was supplied.
        run_cfg.training.phase2_lr_end = args.phase2_lr_end  # Update the run-specific configuration.
    if args.plot_every is not None:  # Check whether CLI option plot_every was supplied.
        run_cfg.training.plot_every = args.plot_every  # Update the run-specific configuration.
    if args.checkpoint_every is not None:  # Check whether CLI option checkpoint_every was supplied.
        run_cfg.training.checkpoint_every = args.checkpoint_every  # Update the run-specific configuration.
    if args.n_interior is not None:  # Check whether CLI option n_interior was supplied.
        run_cfg.sampling.N_interior = args.n_interior  # Update the run-specific configuration.
    if args.n_interface is not None:  # Check whether CLI option n_interface was supplied.
        run_cfg.sampling.N_interface = args.n_interface  # Update the run-specific configuration.
    if args.n_boundary is not None:  # Check whether CLI option n_boundary was supplied.
        run_cfg.sampling.N_boundary = args.n_boundary  # Update the run-specific configuration.
    if args.n_ic is not None:  # Check whether CLI option n_ic was supplied.
        run_cfg.sampling.N_ic = args.n_ic  # Update the run-specific configuration.
    if args.r_min_interior is not None:  # Check whether CLI option r_min_interior was supplied.
        run_cfg.sampling.r_min_interior = args.r_min_interior  # Update the run-specific configuration.
    if args.resample_all_every is not None:  # Check whether CLI option resample_all_every was supplied.
        run_cfg.sampling.resample_all_every = args.resample_all_every  # Update the run-specific configuration.

    if args.smoke:  # Switch to tiny settings for a fast code-path smoke test.
        run_cfg.sampling.N_interior = 32  # Update the run-specific configuration.
        run_cfg.sampling.N_interface = 16  # Update the run-specific configuration.
        run_cfg.sampling.N_boundary = 12  # Update the run-specific configuration.
        run_cfg.sampling.N_ic = 16  # Update the run-specific configuration.
        run_cfg.sampling.resample_every = 2  # Update the run-specific configuration.
        run_cfg.sampling.resample_all_every = 2  # Update the run-specific configuration.
        run_cfg.training.phase1_iters = 2 if args.phase1_iters is None else args.phase1_iters  # Update the run-specific configuration.
        run_cfg.training.phase2_iters = 4 if args.phase2_iters is None else args.phase2_iters  # Update the run-specific configuration.
        run_cfg.training.phase3_iters = 1 if args.phase3_iters is None else args.phase3_iters  # Update the run-specific configuration.
        run_cfg.training.log_every = 1  # Update the run-specific configuration.
        run_cfg.training.plot_every = 2 if args.plot_every is None else args.plot_every  # Update the run-specific configuration.
        run_cfg.training.checkpoint_every = 10 if args.checkpoint_every is None else args.checkpoint_every  # Update the run-specific configuration.
    return run_cfg  # Return the finalized runtime configuration.


def move_points(points: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:  # Define the move_points helper function.
    return {  # Return this value to the caller.
        key: value.detach().to(device).requires_grad_(True)  # Move one point tensor to the selected device and re-enable gradients.
        for key, value in points.items()  # Training-driver logic line.
    }  # Close a multi-line dictionary.


def move_batch(batch: dict[str, dict[str, torch.Tensor]],  # Define the move_batch helper function.
               device: torch.device) -> dict[str, dict[str, torch.Tensor]]:  # Training-driver logic line.
    return {key: move_points(points, device) for key, points in batch.items()}  # Move every point group in the batch to the selected device.


def build_training_batch(model: PINNSolidification,  # Define the build_training_batch helper function.
                         run_cfg: Config,  # Training-driver logic line.
                         seed: int,  # Training-driver logic line.
                         device: torch.device) -> dict[str, dict[str, torch.Tensor]]:  # Training-driver logic line.
    return move_batch(build_batch(model, run_cfg, seed=seed), device)  # Build a fresh collocation batch and place it on the device.


def replace_interface_points(batch: dict[str, dict[str, torch.Tensor]],  # Define the replace_interface_points helper function.
                             model: PINNSolidification,  # Training-driver logic line.
                             run_cfg: Config,  # Training-driver logic line.
                             seed: int,  # Training-driver logic line.
                             device: torch.device) -> None:  # Training-driver logic line.
    batch["interface"] = move_points(  # Replace only the adaptive interface collocation points.
        resample_interface(model, run_cfg.sampling.N_interface, run_cfg, seed=seed),  # Training-driver logic line.
        device,  # Training-driver logic line.
    )  # Close a multi-line function call or expression.


def ensure_dirs(run_cfg: Config) -> None:  # Define the ensure_dirs helper function.
    run_cfg.output_dir.mkdir(parents=True, exist_ok=True)  # Update the run-specific configuration.
    run_cfg.training.checkpoint_dir.mkdir(parents=True, exist_ok=True)  # Update the run-specific configuration.


def checkpoint_path(run_cfg: Config, name: str) -> Path:  # Define the checkpoint_path helper function.
    return run_cfg.training.checkpoint_dir / name  # Return the finalized runtime configuration.


def capture_rng_state() -> dict[str, Any]:  # Define the capture_rng_state helper function.
    state: dict[str, Any] = {  # Start a dictionary that stores random generator states.
        "python": random.getstate(),  # Store Python RNG state.
        "numpy": np.random.get_state(),  # Store NumPy RNG state.
        "torch": torch.get_rng_state(),  # Store PyTorch RNG state.
    }  # Close a multi-line dictionary.
    if torch.cuda.is_available():  # Run CUDA-specific logic only when a GPU is available.
        state["cuda"] = torch.cuda.get_rng_state_all()  # Store CUDA RNG state when running with GPU.
    return state  # Return the captured RNG states.


def restore_rng_state(state: dict[str, Any] | None) -> None:  # Define the restore_rng_state helper function.
    if not state:  # Branch only when this condition is true.
        return  # Training-driver logic line.

    def _cpu_byte_tensor(value: Any) -> torch.Tensor:  # Define the _cpu_byte_tensor helper function.
        if isinstance(value, torch.Tensor):  # Branch only when this condition is true.
            return value.detach().cpu().to(torch.uint8)  # Return this value to the caller.
        return torch.tensor(value, dtype=torch.uint8)  # Return this value to the caller.

    if "python" in state:  # Branch only when this condition is true.
        random.setstate(state["python"])  # Restore Python RNG state.
    if "numpy" in state:  # Branch only when this condition is true.
        np.random.set_state(state["numpy"])  # Restore NumPy RNG state.
    if "torch" in state:  # Branch only when this condition is true.
        torch.set_rng_state(_cpu_byte_tensor(state["torch"]))  # Restore PyTorch CPU RNG state.
    if "cuda" in state and torch.cuda.is_available():  # Branch only when this condition is true.
        torch.cuda.set_rng_state_all([_cpu_byte_tensor(s) for s in state["cuda"]])  # Restore CUDA RNG state.


def save_checkpoint(path: Path,  # Define the save_checkpoint helper function.
                    model: PINNSolidification,  # Training-driver logic line.
                    run_cfg: Config,  # Training-driver logic line.
                    phase: str,  # Training-driver logic line.
                    iteration: int,  # Training-driver logic line.
                    best_loss: float,  # Training-driver logic line.
                    log: dict[str, float],  # Training-driver logic line.
                    optimizer: torch.optim.Optimizer | None = None,  # Training-driver logic line.
                    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,  # Training-driver logic line.
                    optimizer_name: str | None = None,  # Training-driver logic line.
                    overwrite: bool = False) -> Path | None:  # Training-driver logic line.
    payload = {  # Create or load a checkpoint payload.
        "model_version": MODEL_VERSION,  # Training-driver logic line.
        "phase": phase,  # Training-driver logic line.
        "iteration": iteration,  # Training-driver logic line.
        "best_loss": best_loss,  # Training-driver logic line.
        "last_log": log,  # Training-driver logic line.
        "model_state": model.state_dict(),  # Training-driver logic line.
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,  # Training-driver logic line.
        "optimizer_name": optimizer_name,  # Training-driver logic line.
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,  # Training-driver logic line.
        "config": to_plain(run_cfg),  # Training-driver logic line.
        "rng_state": capture_rng_state(),  # Training-driver logic line.
        "saved_at_unix": time.time(),  # Training-driver logic line.
    }  # Close a multi-line dictionary.
    target = path if overwrite or not path.exists() else unique_checkpoint_path(path, iteration)  # Assign a local variable used by the training workflow.
    if target != path:  # Branch only when this condition is true.
        print(  # Print a progress/status message.
            f"[checkpoint] target exists, wrote fallback {target.name}",  # Training-driver logic line.
            flush=True,  # Assign a local variable used by the training workflow.
        )  # Close a multi-line function call or expression.
    try:  # Start protected block for recoverable errors.
        if overwrite:  # Branch only when this condition is true.
            # Some OneDrive-backed folders deny rename/delete operations while
            # still allowing normal writes. Direct overwrite keeps resume
            # checkpoints usable in that environment.
            torch.save(payload, path)  # Serialize checkpoint data to disk.
        else:  # Fallback branch when earlier conditions were false.
            torch.save(payload, target)  # Serialize checkpoint data to disk.
        return target  # Return this value to the caller.
    except (OSError, RuntimeError) as exc:  # Handle an expected exception path.
        print(f"[checkpoint] skipped {target.name}: {exc}", flush=True)  # Print a progress/status message.
        return None  # Return this value to the caller.


def unique_checkpoint_path(path: Path, iteration: int) -> Path:  # Define the unique_checkpoint_path helper function.
    stamp = int(time.time())  # Assign a local variable used by the training workflow.
    candidate = path.with_name(f"{path.stem}_iter{iteration:05d}_{stamp}{path.suffix}")  # Assign a local variable used by the training workflow.
    idx = 1  # Assign a local variable used by the training workflow.
    while candidate.exists():  # Training-driver logic line.
        candidate = path.with_name(  # Assign a local variable used by the training workflow.
            f"{path.stem}_iter{iteration:05d}_{stamp}_{idx}{path.suffix}"  # Training-driver logic line.
        )  # Close a multi-line function call or expression.
        idx += 1  # Training-driver logic line.
    return candidate  # Return this value to the caller.


def cleanup_old_best_checkpoints(run_cfg: Config, keep_name: str) -> None:  # Define the cleanup_old_best_checkpoints helper function.
    pattern = f"pinns_v{MODEL_VERSION}_ep*_loss*.pth"  # Assign a local variable used by the training workflow.
    for path in run_cfg.training.checkpoint_dir.glob(pattern):  # Loop over items in this collection.
        if path.name == keep_name:  # Branch only when this condition is true.
            continue  # Training-driver logic line.
        try:  # Start protected block for recoverable errors.
            path.unlink()  # Call a method on an object.
        except OSError:  # Handle an expected exception path.
            pass  # Training-driver logic line.


def load_checkpoint(path: Path,  # Define the load_checkpoint helper function.
                    model: PINNSolidification,  # Training-driver logic line.
                    device: torch.device) -> dict[str, Any]:  # Training-driver logic line.
    payload = torch.load(path, map_location=device, weights_only=False)  # Create or load a checkpoint payload.
    state = payload.get("model_state", payload.get("model_state_dict", payload))  # Assign a local variable used by the training workflow.
    model.load_state_dict(state)  # Load saved neural-network weights into the model.
    restore_rng_state(payload.get("rng_state"))  # Training-driver logic line.
    return payload  # Return this value to the caller.


def update_best_models_md(run_cfg: Config,  # Define the update_best_models_md helper function.
                          filename: str,  # Training-driver logic line.
                          loss_value: float,  # Training-driver logic line.
                          iteration: int,  # Training-driver logic line.
                          phase: str) -> None:  # Training-driver logic line.
    path = run_cfg.training.checkpoint_dir / "BEST_MODELS.md"  # Assign a local variable used by the training workflow.
    text = (  # Assign a local variable used by the training workflow.
        "# Best Model Checkpoints\n\n"
        "This file is tracked by git. The `.pth` weight files are **not** tracked "  # String literal entry in a list or dictionary.
        "(see root `.gitignore`).\n\n"  # String literal entry in a list or dictionary.
        "## Naming convention\n\n"
        "```\n"  # String literal entry in a list or dictionary.
        "pinns_v{MAJOR}.{MINOR}_ep{EPOCH:05d}_loss{VAL:.4e}.pth\n"  # String literal entry in a list or dictionary.
        "```\n\n"  # String literal entry in a list or dictionary.
        "| Filename | Val Loss | Epoch | Notes |\n"  # String literal entry in a list or dictionary.
        "|---|---|---|---|\n"  # String literal entry in a list or dictionary.
        f"| `{filename}` | {loss_value:.6e} | {iteration} | Best {phase} checkpoint from training run |\n"  # Training-driver logic line.
    )  # Close a multi-line function call or expression.
    path.write_text(text, encoding="utf-8")  # Call a method on an object.


def save_best_and_latest(model: PINNSolidification,  # Define the save_best_and_latest helper function.
                         run_cfg: Config,  # Training-driver logic line.
                         phase: str,  # Training-driver logic line.
                         iteration: int,  # Training-driver logic line.
                         best_loss: float,  # Training-driver logic line.
                         log: dict[str, float],  # Training-driver logic line.
                         optimizer: torch.optim.Optimizer | None,  # Training-driver logic line.
                         scheduler: torch.optim.lr_scheduler.LRScheduler | None,  # Training-driver logic line.
                         optimizer_name: str | None,  # Training-driver logic line.
                         save_latest: bool = False) -> float:  # Training-driver logic line.
    total = log["total"]  # Assign a local variable used by the training workflow.
    if math.isfinite(total) and total < best_loss:  # Branch only when this condition is true.
        best_name = f"pinns_v{MODEL_VERSION}_ep{iteration:05d}_loss{total:.4e}.pth"  # Assign a local variable used by the training workflow.
        best_path = save_checkpoint(  # Assign a local variable used by the training workflow.
            checkpoint_path(run_cfg, best_name),  # Training-driver logic line.
            model,  # Training-driver logic line.
            run_cfg,  # Training-driver logic line.
            phase,  # Training-driver logic line.
            iteration,  # Training-driver logic line.
            total,  # Store the new best value inside the best-checkpoint payload.
            log,  # Training-driver logic line.
            optimizer=optimizer,  # Assign a local variable used by the training workflow.
            scheduler=scheduler,  # Assign a local variable used by the training workflow.
            optimizer_name=optimizer_name,  # Assign a local variable used by the training workflow.
        )  # Close a multi-line function call or expression.
        if best_path is not None:  # Branch only when this condition is true.
            best_loss = total  # Assign a local variable used by the training workflow.
            cleanup_old_best_checkpoints(run_cfg, best_path.name)  # Training-driver logic line.
            update_best_models_md(run_cfg, best_path.name, total, iteration, phase)  # Training-driver logic line.

    if save_latest:  # Branch only when this condition is true.
        latest = checkpoint_path(run_cfg, run_cfg.training.latest_checkpoint)  # Assign a local variable used by the training workflow.
        save_checkpoint(  # Training-driver logic line.
            latest, model, run_cfg, phase, iteration, best_loss, log,  # Training-driver logic line.
            optimizer=optimizer, scheduler=scheduler, optimizer_name=optimizer_name,  # Assign a local variable used by the training workflow.
            overwrite=True,  # Assign a local variable used by the training workflow.
        )  # Close a multi-line function call or expression.
    return best_loss  # Return this value to the caller.


def append_phase_csv(csv_path: Path,  # Define the append_phase_csv helper function.
                     iteration: int,  # Training-driver logic line.
                     phase: str,  # Training-driver logic line.
                     log: dict[str, float],  # Training-driver logic line.
                     write_header: bool = False) -> None:  # Training-driver logic line.
    fields = ["iteration", "phase"] + sorted(log.keys())  # Assign a local variable used by the training workflow.
    row = {"iteration": iteration, "phase": phase, **log}  # Assign a local variable used by the training workflow.
    with open(csv_path, "a", newline="") as f:  # Open a file for reading or writing.
        writer = csv.DictWriter(f, fieldnames=fields)  # Assign a local variable used by the training workflow.
        if write_header:  # Branch only when this condition is true.
            writer.writeheader()  # Call a method on an object.
        writer.writerow(row)  # Call a method on an object.


def plot_monitor_snapshot(model: PINNSolidification,  # Define the plot_monitor_snapshot helper function.
                          run_cfg: Config,  # Training-driver logic line.
                          iteration: int,  # Training-driver logic line.
                          device: torch.device) -> None:  # Training-driver logic line.
    model.eval()  # Put the model in evaluation mode.
    nx, nr = 120, 80  # Training-driver logic line.
    x = torch.linspace(0.0, run_cfg.case.L, nx, device=device)  # Assign a local variable used by the training workflow.
    r = torch.linspace(0.0, run_cfg.case.r_w, nr, device=device)  # Assign a local variable used by the training workflow.
    rr, xx = torch.meshgrid(r, x, indexing="ij")  # Training-driver logic line.
    tt = torch.full_like(rr, 0.5 * run_cfg.case.t_end)  # Assign a local variable used by the training workflow.

    with torch.no_grad():  # Disable gradient tracking for evaluation-only work.
        out = model(rr.reshape(-1), xx.reshape(-1), tt.reshape(-1))  # Assign a local variable used by the training workflow.
        theta = out["Theta_f"].reshape(nr, nx).detach().cpu().numpy()  # Assign a local variable used by the training workflow.
        r_int = model(  # Assign a local variable used by the training workflow.
            torch.zeros_like(x),  # Call a method on an object.
            x,  # Training-driver logic line.
            torch.full_like(x, 0.5 * run_cfg.case.t_end),  # Call a method on an object.
        )["r_int"].detach().cpu().numpy()  # Close a multi-line function call or expression.

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))  # Create a Matplotlib figure and axes.
    axes[0].plot(x.detach().cpu().numpy(), r_int, lw=2)  # Training-driver logic line.
    axes[0].set_xlabel("x_hat")  # Training-driver logic line.
    axes[0].set_ylabel("r_int_hat")  # Training-driver logic line.
    axes[0].set_ylim(0.0, run_cfg.case.r_w * 1.05)  # Training-driver logic line.
    axes[0].set_title("Interface at t_mid")  # Training-driver logic line.

    im = axes[1].contourf(  # Assign a local variable used by the training workflow.
        xx.detach().cpu().numpy(),  # Call a method on an object.
        rr.detach().cpu().numpy(),  # Call a method on an object.
        theta,  # Training-driver logic line.
        levels=32,  # Assign a local variable used by the training workflow.
        cmap="inferno",  # Assign a local variable used by the training workflow.
    )  # Close a multi-line function call or expression.
    axes[1].plot(x.detach().cpu().numpy(), r_int, color="cyan", lw=1.5)  # Training-driver logic line.
    axes[1].set_xlabel("x_hat")  # Training-driver logic line.
    axes[1].set_ylabel("r_hat")  # Training-driver logic line.
    axes[1].set_title("Theta_f at t_mid")  # Training-driver logic line.
    fig.colorbar(im, ax=axes[1], shrink=0.85)  # Call a method on an object.
    fig.suptitle(f"Training monitor, iter {iteration}")  # Call a method on an object.
    fig.tight_layout()  # Call a method on an object.
    fig.savefig(run_cfg.output_dir / f"monitor_iter{iteration:05d}.png", dpi=150)  # Write the figure image to disk.
    plt.close(fig)  # Close the figure to release memory.
    model.train()  # Put the model in training mode.


def should_log(iteration: int, run_cfg: Config) -> bool:  # Define the should_log helper function.
    return iteration == 1 or iteration % run_cfg.training.log_every == 0  # Return this value to the caller.


def should_plot(iteration: int, run_cfg: Config) -> bool:  # Define the should_plot helper function.
    return run_cfg.training.plot_every > 0 and iteration % run_cfg.training.plot_every == 0  # Return the finalized runtime configuration.


def should_checkpoint(iteration: int, run_cfg: Config) -> bool:  # Define the should_checkpoint helper function.
    return iteration == 1 or (  # Return this value to the caller.
        run_cfg.training.checkpoint_every > 0  # Update the run-specific configuration.
        and iteration % run_cfg.training.checkpoint_every == 0  # Training-driver logic line.
    )  # Close a multi-line function call or expression.


def check_log_is_finite(log: dict[str, float], phase: str, iteration: int) -> None:  # Define the check_log_is_finite helper function.
    bad = {k: v for k, v in log.items() if not math.isfinite(v)}  # Assign a local variable used by the training workflow.
    if bad:  # Branch only when this condition is true.
        raise FloatingPointError(f"Non-finite loss in {phase} at iter {iteration}: {bad}")  # Stop execution with an explicit error.


def build_target_stop(args: argparse.Namespace, run_cfg: Config) -> TargetStop | None:  # Define the build_target_stop helper function.
    if args.target_solid_thickness is None:  # Branch only when this condition is true.
        return None  # Return this value to the caller.
    check_every = (  # Assign a local variable used by the training workflow.
        args.target_check_every  # Call a method on an object.
        if args.target_check_every is not None  # Check whether CLI option target_check_every was supplied.
        else max(1, run_cfg.training.log_every)  # Training-driver logic line.
    )  # Close a multi-line function call or expression.
    return TargetStop(  # Return this value to the caller.
        thickness=args.target_solid_thickness,  # Assign a local variable used by the training workflow.
        mode=args.target_thickness_mode,  # Assign a local variable used by the training workflow.
        t_value=run_cfg.case.t_end if args.target_time is None else args.target_time,  # Assign a local variable used by the training workflow.
        grid_n=args.target_grid_n,  # Assign a local variable used by the training workflow.
        check_every=check_every,  # Assign a local variable used by the training workflow.
        min_iter=args.target_min_iter,  # Assign a local variable used by the training workflow.
    )  # Close a multi-line function call or expression.


def interface_thickness_stats(model: PINNSolidification,  # Define the interface_thickness_stats helper function.
                              run_cfg: Config,  # Training-driver logic line.
                              device: torch.device,  # Training-driver logic line.
                              target: TargetStop) -> dict[str, float]:  # Training-driver logic line.
    was_training = model.training  # Assign a local variable used by the training workflow.
    model.eval()  # Put the model in evaluation mode.
    x = torch.linspace(0.0, run_cfg.case.L, target.grid_n, device=device)  # Assign a local variable used by the training workflow.
    t = torch.full_like(x, target.t_value)  # Assign a local variable used by the training workflow.
    r = torch.zeros_like(x)  # Assign a local variable used by the training workflow.
    with torch.no_grad():  # Disable gradient tracking for evaluation-only work.
        r_int = model(r, x, t)["r_int"]  # Assign a local variable used by the training workflow.
        thickness = run_cfg.case.r_w - r_int  # Assign a local variable used by the training workflow.
    if was_training:  # Branch only when this condition is true.
        model.train()  # Put the model in training mode.
    return {  # Return this value to the caller.
        "min": float(thickness.min().item()),  # Training-driver logic line.
        "mean": float(thickness.mean().item()),  # Training-driver logic line.
        "max": float(thickness.max().item()),  # Training-driver logic line.
        "r_int_min": float(r_int.min().item()),  # Training-driver logic line.
        "r_int_mean": float(r_int.mean().item()),  # Training-driver logic line.
        "r_int_max": float(r_int.max().item()),  # Training-driver logic line.
    }  # Close a multi-line dictionary.


def check_target_stop(model: PINNSolidification,  # Define the check_target_stop helper function.
                      run_cfg: Config,  # Training-driver logic line.
                      device: torch.device,  # Training-driver logic line.
                      target: TargetStop | None,  # Training-driver logic line.
                      phase: str,  # Training-driver logic line.
                      iteration: int) -> bool:  # Training-driver logic line.
    if target is None:  # Branch only when this condition is true.
        return False  # Return this value to the caller.
    if iteration < target.min_iter:  # Branch only when this condition is true.
        return False  # Return this value to the caller.
    if iteration == 1 or iteration % target.check_every != 0:  # Branch only when this condition is true.
        return False  # Return this value to the caller.
    stats = interface_thickness_stats(model, run_cfg, device, target)  # Assign a local variable used by the training workflow.
    value = stats[target.mode]  # Assign a local variable used by the training workflow.
    print(  # Print a progress/status message.
        f"[target] iter {iteration} {phase} thickness "  # Training-driver logic line.
        f"min/mean/max={stats['min']:.4f}/{stats['mean']:.4f}/{stats['max']:.4f} "  # Training-driver logic line.
        f"at t={target.t_value:g} ({target.mode} target {target.thickness:.4f})",  # Training-driver logic line.
        flush=True,  # Assign a local variable used by the training workflow.
    )  # Close a multi-line function call or expression.
    return value >= target.thickness  # Return this value to the caller.


def run_phase1(model: PINNSolidification,  # Define the run_phase1 helper function.
               run_cfg: Config,  # Training-driver logic line.
               device: torch.device,  # Training-driver logic line.
               csv_path: Path,  # Training-driver logic line.
               start_iter: int,  # Training-driver logic line.
               best_loss: float,  # Training-driver logic line.
               resume_payload: dict[str, Any] | None,  # Training-driver logic line.
               target: TargetStop | None) -> tuple[int, float, bool]:  # Training-driver logic line.
    end_iter = run_cfg.training.phase1_iters  # Assign a local variable used by the training workflow.
    if start_iter >= end_iter:  # Branch only when this condition is true.
        return start_iter, best_loss, False  # Return this value to the caller.

    optimizer = torch.optim.Adam(model.parameters(), lr=run_cfg.training.phase1_lr)  # Assign a local variable used by the training workflow.
    if resume_payload and resume_payload.get("phase") == "phase1" and resume_payload.get("optimizer_state"):  # Branch only when this condition is true.
        optimizer.load_state_dict(resume_payload["optimizer_state"])  # Restore optimizer state for exact resume.

    batch = build_training_batch(model, run_cfg, run_cfg.seed + start_iter, device)  # Assign a local variable used by the training workflow.
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0  # Assign a local variable used by the training workflow.

    for iteration in range(start_iter + 1, end_iter + 1):  # Loop through training iterations for this phase.
        if (  # Branch only when this condition is true.
            run_cfg.sampling.resample_all_every > 0  # Update the run-specific configuration.
            and iteration > start_iter + 1  # Training-driver logic line.
            and iteration % run_cfg.sampling.resample_all_every == 0  # Training-driver logic line.
        ):  # Close a multi-line function call or expression.
            batch = build_training_batch(model, run_cfg, run_cfg.seed + iteration, device)  # Assign a local variable used by the training workflow.

        model.train()  # Put the model in training mode.
        optimizer.zero_grad(set_to_none=True)  # Clear old gradients before the new backward pass.
        loss, log = compute_loss(model, batch, run_cfg, physics_on=False)  # Compute total loss and named loss terms.
        check_log_is_finite(log, "phase1", iteration)  # Training-driver logic line.
        loss.backward()  # Backpropagate derivatives through the network.
        optimizer.step()  # Update trainable parameters using the optimizer.

        append_phase_csv(csv_path, iteration, "phase1", log, write_header=write_header)  # Write this iteration loss record to CSV.
        write_header = False  # Assign a local variable used by the training workflow.

        if should_log(iteration, run_cfg):  # Check whether this iteration should be printed.
            print(format_loss_line(log, iteration, physics_on=False), flush=True)  # Print a progress/status message.
        if should_plot(iteration, run_cfg):  # Check whether this iteration should generate plots.
            plot_monitor_snapshot(model, run_cfg, iteration, device)  # Save monitor plots for the current model state.

        best_loss = save_best_and_latest(  # Assign a local variable used by the training workflow.
            model, run_cfg, "phase1", iteration, best_loss, log,  # Training-driver logic line.
            optimizer=optimizer, scheduler=None, optimizer_name="Adam",  # Assign a local variable used by the training workflow.
            save_latest=(  # Assign a local variable used by the training workflow.
                should_checkpoint(iteration, run_cfg)  # Training-driver logic line.
                or iteration == end_iter  # Training-driver logic line.
                or log["total"] < run_cfg.training.loss_target  # Training-driver logic line.
            ),  # Close a multi-line function call or expression.
        )  # Close a multi-line function call or expression.
        if check_target_stop(model, run_cfg, device, target, "phase1", iteration):  # Check whether the requested target thickness has been reached.
            print(f"[stop] target solid thickness reached in phase1 at iter {iteration}", flush=True)  # Print a progress/status message.
            return iteration, best_loss, True  # Return this value to the caller.
        if log["total"] < run_cfg.training.loss_target:  # Check early stopping by total loss target.
            print(f"[stop] target loss reached in phase1 at iter {iteration}", flush=True)  # Print a progress/status message.
            return iteration, best_loss, True  # Return this value to the caller.
    return end_iter, best_loss, False  # Return this value to the caller.


def run_phase2(model: PINNSolidification,  # Define the run_phase2 helper function.
               run_cfg: Config,  # Training-driver logic line.
               device: torch.device,  # Training-driver logic line.
               csv_path: Path,  # Training-driver logic line.
               start_iter: int,  # Training-driver logic line.
               best_loss: float,  # Training-driver logic line.
               resume_payload: dict[str, Any] | None,  # Training-driver logic line.
               target: TargetStop | None) -> tuple[int, float, bool]:  # Training-driver logic line.
    phase1_end = run_cfg.training.phase1_iters  # Assign a local variable used by the training workflow.
    phase2_end = run_cfg.training.phase2_iters  # Assign a local variable used by the training workflow.
    if start_iter >= phase2_end:  # Branch only when this condition is true.
        return start_iter, best_loss, False  # Return this value to the caller.

    first_iter = max(start_iter + 1, phase1_end + 1)  # Assign a local variable used by the training workflow.
    if start_iter <= phase1_end:  # Branch only when this condition is true.
        best_loss = float("inf")  # Assign a local variable used by the training workflow.
    phase2_steps = max(phase2_end - first_iter + 1, 1)  # Assign a local variable used by the training workflow.
    optimizer = torch.optim.Adam(model.parameters(), lr=run_cfg.training.phase2_lr_start)  # Assign a local variable used by the training workflow.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(  # Assign a local variable used by the training workflow.
        optimizer,  # Training-driver logic line.
        T_max=phase2_steps,  # Assign a local variable used by the training workflow.
        eta_min=run_cfg.training.phase2_lr_end,  # Assign a local variable used by the training workflow.
    )  # Close a multi-line function call or expression.
    if resume_payload and resume_payload.get("phase") == "phase2":  # Branch only when this condition is true.
        if resume_payload.get("optimizer_state"):  # Branch only when this condition is true.
            optimizer.load_state_dict(resume_payload["optimizer_state"])  # Restore optimizer state for exact resume.
        if resume_payload.get("scheduler_state"):  # Branch only when this condition is true.
            scheduler.load_state_dict(resume_payload["scheduler_state"])  # Restore scheduler state for exact resume.

    batch = build_training_batch(model, run_cfg, run_cfg.seed + first_iter, device)  # Assign a local variable used by the training workflow.
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0  # Assign a local variable used by the training workflow.

    for iteration in range(first_iter, phase2_end + 1):  # Loop through training iterations for this phase.
        if (  # Branch only when this condition is true.
            run_cfg.sampling.resample_all_every > 0  # Update the run-specific configuration.
            and iteration > first_iter  # Training-driver logic line.
            and iteration % run_cfg.sampling.resample_all_every == 0  # Training-driver logic line.
        ):  # Close a multi-line function call or expression.
            batch = build_training_batch(model, run_cfg, run_cfg.seed + iteration, device)  # Assign a local variable used by the training workflow.
        elif (iteration - phase1_end) == 1 or iteration % run_cfg.sampling.resample_every == 0:  # Alternative branch for the previous condition.
            replace_interface_points(batch, model, run_cfg, run_cfg.seed + iteration, device)  # Training-driver logic line.

        model.train()  # Put the model in training mode.
        optimizer.zero_grad(set_to_none=True)  # Clear old gradients before the new backward pass.
        loss, log = compute_loss(model, batch, run_cfg, physics_on=True)  # Compute total loss and named loss terms.
        check_log_is_finite(log, "phase2", iteration)  # Training-driver logic line.
        loss.backward()  # Backpropagate derivatives through the network.
        optimizer.step()  # Update trainable parameters using the optimizer.
        scheduler.step()  # Advance the learning-rate schedule.

        append_phase_csv(csv_path, iteration, "phase2", log, write_header=write_header)  # Write this iteration loss record to CSV.
        write_header = False  # Assign a local variable used by the training workflow.

        if should_log(iteration, run_cfg):  # Check whether this iteration should be printed.
            print(format_loss_line(log, iteration, physics_on=True), flush=True)  # Print a progress/status message.
        if should_plot(iteration, run_cfg):  # Check whether this iteration should generate plots.
            plot_monitor_snapshot(model, run_cfg, iteration, device)  # Save monitor plots for the current model state.

        best_loss = save_best_and_latest(  # Assign a local variable used by the training workflow.
            model, run_cfg, "phase2", iteration, best_loss, log,  # Training-driver logic line.
            optimizer=optimizer, scheduler=scheduler, optimizer_name="Adam",  # Assign a local variable used by the training workflow.
            save_latest=(  # Assign a local variable used by the training workflow.
                should_checkpoint(iteration, run_cfg)  # Training-driver logic line.
                or iteration == phase2_end  # Training-driver logic line.
                or log["total"] < run_cfg.training.loss_target  # Training-driver logic line.
            ),  # Close a multi-line function call or expression.
        )  # Close a multi-line function call or expression.
        if check_target_stop(model, run_cfg, device, target, "phase2", iteration):  # Check whether the requested target thickness has been reached.
            print(f"[stop] target solid thickness reached in phase2 at iter {iteration}", flush=True)  # Print a progress/status message.
            return iteration, best_loss, True  # Return this value to the caller.
        if log["total"] < run_cfg.training.loss_target:  # Check early stopping by total loss target.
            print(f"[stop] target loss reached in phase2 at iter {iteration}", flush=True)  # Print a progress/status message.
            return iteration, best_loss, True  # Return this value to the caller.
    return phase2_end, best_loss, False  # Return this value to the caller.


def run_phase3(model: PINNSolidification,  # Define the run_phase3 helper function.
               run_cfg: Config,  # Training-driver logic line.
               device: torch.device,  # Training-driver logic line.
               csv_path: Path,  # Training-driver logic line.
               start_iter: int,  # Training-driver logic line.
               best_loss: float,  # Training-driver logic line.
               resume_payload: dict[str, Any] | None,  # Training-driver logic line.
               target: TargetStop | None) -> tuple[int, float, bool]:  # Training-driver logic line.
    phase2_end = run_cfg.training.phase2_iters  # Assign a local variable used by the training workflow.
    phase3_end = phase2_end + run_cfg.training.phase3_iters  # Assign a local variable used by the training workflow.
    if start_iter >= phase3_end:  # Branch only when this condition is true.
        return start_iter, best_loss, False  # Return this value to the caller.

    first_iter = max(start_iter + 1, phase2_end + 1)  # Assign a local variable used by the training workflow.
    optimizer = torch.optim.LBFGS(  # Assign a local variable used by the training workflow.
        model.parameters(),  # Call a method on an object.
        lr=1.0,  # Assign a local variable used by the training workflow.
        max_iter=1,  # Assign a local variable used by the training workflow.
        max_eval=5,  # Assign a local variable used by the training workflow.
        tolerance_grad=1e-9,  # Assign a local variable used by the training workflow.
        tolerance_change=1e-11,  # Assign a local variable used by the training workflow.
        history_size=50,  # Assign a local variable used by the training workflow.
        line_search_fn="strong_wolfe",  # Assign a local variable used by the training workflow.
    )  # Close a multi-line function call or expression.
    if resume_payload and resume_payload.get("phase") == "phase3" and resume_payload.get("optimizer_state"):  # Branch only when this condition is true.
        optimizer.load_state_dict(resume_payload["optimizer_state"])  # Restore optimizer state for exact resume.

    batch = build_training_batch(model, run_cfg, run_cfg.seed + first_iter, device)  # Assign a local variable used by the training workflow.
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0  # Assign a local variable used by the training workflow.

    for iteration in range(first_iter, phase3_end + 1):  # Loop through training iterations for this phase.
        if (  # Branch only when this condition is true.
            run_cfg.sampling.resample_all_every > 0  # Update the run-specific configuration.
            and iteration > first_iter  # Training-driver logic line.
            and iteration % run_cfg.sampling.resample_all_every == 0  # Training-driver logic line.
        ):  # Close a multi-line function call or expression.
            batch = build_training_batch(model, run_cfg, run_cfg.seed + iteration, device)  # Assign a local variable used by the training workflow.
        elif (iteration - phase2_end) == 1 or iteration % run_cfg.sampling.resample_every == 0:  # Alternative branch for the previous condition.
            replace_interface_points(batch, model, run_cfg, run_cfg.seed + iteration, device)  # Training-driver logic line.

        closure_log: dict[str, float] = {}  # Training-driver logic line.

        def closure() -> torch.Tensor:  # Define the closure helper function.
            optimizer.zero_grad(set_to_none=True)  # Clear old gradients before the new backward pass.
            loss, log = compute_loss(model, batch, run_cfg, physics_on=True)  # Compute total loss and named loss terms.
            check_log_is_finite(log, "phase3", iteration)  # Training-driver logic line.
            loss.backward()  # Backpropagate derivatives through the network.
            closure_log.clear()  # Call a method on an object.
            closure_log.update(log)  # Call a method on an object.
            return loss  # Return this value to the caller.

        model.train()  # Put the model in training mode.
        optimizer.step(closure)  # Update trainable parameters using the optimizer.
        log = dict(closure_log)  # Assign a local variable used by the training workflow.
        if not log:  # Branch only when this condition is true.
            _, log = compute_loss(model, batch, run_cfg, physics_on=True)  # Training-driver logic line.
        check_log_is_finite(log, "phase3", iteration)  # Training-driver logic line.

        append_phase_csv(csv_path, iteration, "phase3", log, write_header=write_header)  # Write this iteration loss record to CSV.
        write_header = False  # Assign a local variable used by the training workflow.

        if should_log(iteration, run_cfg):  # Check whether this iteration should be printed.
            print(format_loss_line(log, iteration, physics_on=True), flush=True)  # Print a progress/status message.
        if should_plot(iteration, run_cfg):  # Check whether this iteration should generate plots.
            plot_monitor_snapshot(model, run_cfg, iteration, device)  # Save monitor plots for the current model state.

        best_loss = save_best_and_latest(  # Assign a local variable used by the training workflow.
            model, run_cfg, "phase3", iteration, best_loss, log,  # Training-driver logic line.
            optimizer=optimizer, scheduler=None, optimizer_name="LBFGS",  # Assign a local variable used by the training workflow.
            save_latest=(  # Assign a local variable used by the training workflow.
                should_checkpoint(iteration, run_cfg)  # Training-driver logic line.
                or iteration == phase3_end  # Training-driver logic line.
                or log["total"] < run_cfg.training.loss_target  # Training-driver logic line.
            ),  # Close a multi-line function call or expression.
        )  # Close a multi-line function call or expression.
        if check_target_stop(model, run_cfg, device, target, "phase3", iteration):  # Check whether the requested target thickness has been reached.
            print(f"[stop] target solid thickness reached in phase3 at iter {iteration}", flush=True)  # Print a progress/status message.
            return iteration, best_loss, True  # Return this value to the caller.
        if log["total"] < run_cfg.training.loss_target:  # Check early stopping by total loss target.
            print(f"[stop] target loss reached in phase3 at iter {iteration}", flush=True)  # Print a progress/status message.
            return iteration, best_loss, True  # Return this value to the caller.
    return phase3_end, best_loss, False  # Return this value to the caller.


def save_run_config(run_cfg: Config) -> None:  # Define the save_run_config helper function.
    path = run_cfg.output_dir / "run_config.json"  # Assign a local variable used by the training workflow.
    path.write_text(json.dumps(to_plain(run_cfg), indent=2), encoding="utf-8")  # Call a method on an object.


def main() -> None:  # Define the main helper function.
    args = parse_args()  # Assign a local variable used by the training workflow.
    run_cfg = apply_cli_overrides(cfg, args)  # Assign a local variable used by the training workflow.
    device = choose_device(args.device)  # Assign a local variable used by the training workflow.
    ensure_dirs(run_cfg)  # Training-driver logic line.
    seed_everything(run_cfg.seed)  # Training-driver logic line.
    save_run_config(run_cfg)  # Training-driver logic line.
    target = build_target_stop(args, run_cfg)  # Assign a local variable used by the training workflow.

    model = PINNSolidification(run_cfg.network, run_cfg.case, seed=run_cfg.seed).to(device)  # Assign a local variable used by the training workflow.
    print(f"[train] device={device} seed={run_cfg.seed} params={count_parameters(model):,}", flush=True)  # Print a progress/status message.
    print(  # Print a progress/status message.
        f"[train] phase ends: p1={run_cfg.training.phase1_iters}, "  # Training-driver logic line.
        f"p2={run_cfg.training.phase2_iters}, "  # Training-driver logic line.
        f"p3={run_cfg.training.phase2_iters + run_cfg.training.phase3_iters}",  # Training-driver logic line.
        flush=True,  # Assign a local variable used by the training workflow.
    )  # Close a multi-line function call or expression.
    print(  # Print a progress/status message.
        f"[train] case={run_cfg.case.name} Ste={run_cfg.case.Ste:g} Pe={run_cfg.case.Pe:g} "  # Training-driver logic line.
        f"Re={run_cfg.case.Re:g} k_ratio={run_cfg.case.k_ratio:g} "  # Training-driver logic line.
        f"L={run_cfg.case.L:g} r_w={run_cfg.case.r_w:g} t_end={run_cfg.case.t_end:g}",  # Training-driver logic line.
        flush=True,  # Assign a local variable used by the training workflow.
    )  # Close a multi-line function call or expression.
    if target is not None:  # Branch only when this condition is true.
        print(  # Print a progress/status message.
            f"[train] target: {target.mode} solid thickness >= {target.thickness:g} "  # Training-driver logic line.
            f"at t={target.t_value:g}, min_iter={target.min_iter}",  # Training-driver logic line.
            flush=True,  # Assign a local variable used by the training workflow.
        )  # Close a multi-line function call or expression.

    resume_payload: dict[str, Any] | None = None  # Training-driver logic line.
    start_iter = 0  # Assign a local variable used by the training workflow.
    best_loss = float("inf")  # Assign a local variable used by the training workflow.
    if args.resume is not None:  # Check whether CLI option resume was supplied.
        resume_payload = load_checkpoint(args.resume, model, device)  # Assign a local variable used by the training workflow.
        start_iter = int(resume_payload.get("iteration", 0))  # Assign a local variable used by the training workflow.
        best_loss = float(resume_payload.get("best_loss", best_loss))  # Assign a local variable used by the training workflow.
        print(f"[resume] loaded {args.resume} at iter {start_iter}", flush=True)  # Print a progress/status message.
        if args.resume_weights_only:  # Branch only when this condition is true.
            resume_payload["phase"] = None  # Training-driver logic line.
            resume_payload["optimizer_state"] = None  # Training-driver logic line.
            resume_payload["scheduler_state"] = None  # Training-driver logic line.
            print("[resume] using checkpoint weights only; optimizer/scheduler reset", flush=True)  # Print a progress/status message.

    csv_path = run_cfg.output_dir / "loss_history.csv"  # Assign a local variable used by the training workflow.
    t0 = time.time()  # Assign a local variable used by the training workflow.
    start_iter, best_loss, stopped = run_phase1(  # Training-driver logic line.
        model, run_cfg, device, csv_path, start_iter, best_loss, resume_payload, target  # Training-driver logic line.
    )  # Close a multi-line function call or expression.
    if not stopped:  # Branch only when this condition is true.
        start_iter, best_loss, stopped = run_phase2(  # Training-driver logic line.
            model, run_cfg, device, csv_path, start_iter, best_loss, resume_payload, target  # Training-driver logic line.
        )  # Close a multi-line function call or expression.
    if not stopped:  # Branch only when this condition is true.
        start_iter, best_loss, stopped = run_phase3(  # Training-driver logic line.
            model, run_cfg, device, csv_path, start_iter, best_loss, resume_payload, target  # Training-driver logic line.
        )  # Close a multi-line function call or expression.
    elapsed = time.time() - t0  # Assign a local variable used by the training workflow.
    print(  # Print a progress/status message.
        f"[done] stopped at iter {start_iter}, best_loss={best_loss:.6e}, "  # Training-driver logic line.
        f"elapsed={elapsed:.1f}s",  # Training-driver logic line.
        flush=True,  # Assign a local variable used by the training workflow.
    )  # Close a multi-line function call or expression.


if __name__ == "__main__":  # Run main only when this file is executed directly.
    main()  # Start the training CLI entry point.
