"""
Single source of truth for the active Case B PINN configuration.
"""

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent
OUTPUT_ROOT = ROOT / "outputs"
POSTPROCESS_ROOT = OUTPUT_ROOT / "postprocess"
HPC_RESULTS_ROOT = OUTPUT_ROOT / "hpc_results"


@dataclass
class CaseConfig:
    """Excel-consistent nondimensional physical parameters for Case B."""

    name: str = "20260410_nonDimm_goodmatchCaseB"

    Pe_D: float = 14.329521
    Ste: float = 0.06275
    Re_D: float = 1.53
    k_ratio: float = 1.0

    r_a: float = 0.0
    r_w: float = 1.0
    L: float = 10.25
    tau_end: float = 8.0 / 10.25

    Theta_in: float = 2.0
    Theta_wall: float = 0.0
    Theta_solidus: float = 1.0
    hot_wall_length: float = 0.25

    r_int_ic: float = 1.0


@dataclass
class NetworkConfig:
    """Architecture hyperparameters."""

    n_hidden_layers: int = 8
    n_hidden_units: int = 128
    fourier_features: int = 256
    fourier_sigma: float = 1.0
    interface_epsilon: float = 0.02


@dataclass
class SamplingConfig:
    """Collocation point counts and sampling controls."""

    N_interior: int = 20_000
    N_interface: int = 5_000
    N_boundary: int = 2_000
    N_ic: int = 5_000
    r_min_interior: float = 0.01
    resample_every: int = 1
    resample_all_every: int = 5_000


@dataclass
class LossWeights:
    """Weights for the composite PINN loss."""

    mass: float = 1.0
    mom_x: float = 1.0
    mom_r: float = 1.0
    energy_fluid: float = 1.0
    energy_dep: float = 1.0
    stefan: float = 10.0
    T_continuity: float = 100.0
    interface_velocity: float = 10.0
    rint_mono: float = 10.0
    rint_x_mono: float = 10.0
    bc_ic: float = 100.0


@dataclass
class TrainingConfig:
    """Phase 1 + Phase 2 training schedule.

    Phase 3 is disabled by default because the L-BFGS refinement degraded the
    latest HPC continuation run. Enable it only with an explicit
    --phase3-iters override for a deliberate experiment.
    """

    phase1_iters: int = 2_000
    phase1_lr: float = 1e-3

    phase2_iters: int = 40_000
    phase2_lr_start: float = 1e-3
    phase2_lr_end: float = 1e-5

    phase3_iters: int = 0
    loss_target: float = 1e-4

    checkpoint_dir: Path = ROOT / "checkpoints"
    log_every: int = 500
    plot_every: int = 1_000
    checkpoint_every: int = 500
    latest_checkpoint: str = "latest.pth"
    last_checkpoint: str = "last.pth"


@dataclass
class Config:
    """Top-level config bundling all sub-configs."""

    case: CaseConfig = field(default_factory=CaseConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    weights: LossWeights = field(default_factory=LossWeights)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output_dir: Path = OUTPUT_ROOT
    seed: int = 42


def config_from_dict(data: dict[str, Any]) -> Config:
    """Rebuild a Config from a current-format checkpoint config dictionary."""
    if not isinstance(data, dict):
        raise ValueError("checkpoint config must be a dictionary")

    run_cfg = Config()

    def apply_values(obj: Any, values: dict[str, Any], path: str) -> None:
        if not is_dataclass(obj):
            raise TypeError(f"{path} is not a dataclass config object")
        valid = {item.name for item in fields(obj)}
        unknown = set(values) - valid
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"unsupported config field(s) in {path}: {names}")
        for key, value in values.items():
            current = getattr(obj, key)
            if is_dataclass(current):
                if not isinstance(value, dict):
                    raise ValueError(f"{path}.{key} must be a dictionary")
                apply_values(current, value, f"{path}.{key}")
            elif isinstance(current, Path):
                setattr(obj, key, Path(value))
            else:
                setattr(obj, key, value)

    apply_values(run_cfg, data, "config")
    return run_cfg


cfg = Config()
