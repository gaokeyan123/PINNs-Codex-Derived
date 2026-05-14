"""
Single source of truth for all hyperparameters.
Edit this file only — everything else imports from here.
"""
import copy
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent


@dataclass
class CaseConfig:
    """Non-dimensional physical parameters for the simple test case."""
    name: str = "20260410_nonDimm_goodmatchCaseC"

    Pe: float = 1.43e1     # Peclet number:  rho_f Cp_f U r_w / k_f
    Ste: float = 0.06275   # Stefan number:  Cp_f dT / L_f
    Re: float = 1.53       # Reynolds number (laminar)
    k_ratio: float = 1.0   # k_dep / k_f  (equal conductivities, simplest case)

    # Geometry (non-dimensional, r_w = 1 is the reference length)
    r_a: float = 0.0       # inner axis radius (0 = solid cylinder)
    r_w: float = 1.0       # outer wall radius (= 1 by definition of scaling)
    L: float = 10.25       # pipe length, including the hot-wall inlet section

    # Time
    t_end: float = 0.8     # final non-dimensional time

    # Boundary / initial conditions (non-dimensional temperatures)
    # Reference dimensional values for this verification case:
    # T_in = 5 C, T_interface = 0 C, T_wall = -5 C.
    # Theta = (T - T_wall) / (T_interface - T_wall).
    Theta_in: float = 2.0       # hot inlet fluid temperature
    Theta_wall: float = 0.0     # cold outer wall temperature
    Theta_solidus: float = 1.0  # solidification temperature
    hot_wall_length: float = 0.25  # set wall temperature to Theta_in for 0 <= x <= this length

    # Initial condition: clean pipe, no deposit
    r_int_ic: float = 1.0       # r_int(x, 0) = r_w everywhere


@dataclass
class NetworkConfig:
    """Architecture hyperparameters (network v1.0)."""
    n_hidden_layers: int = 8
    n_hidden_units: int = 128
    fourier_features: int = 256    # number of Fourier feature pairs (= output dim / 2)
    fourier_sigma: float = 1.0     # std of random Fourier frequencies B
    interface_epsilon: float = 0.02  # smooth Heaviside width (fraction of r_w)


@dataclass
class SamplingConfig:
    """Collocation point counts."""
    N_interior: int = 20_000
    N_interface: int = 5_000
    N_boundary: int = 2_000    # per boundary (wall, inlet, outlet)
    N_ic: int = 5_000
    r_min_interior: float = 0.01  # exclude singular near-axis PDE points; axis BC remains active
    resample_every: int = 500  # iters between interface point resampling
    resample_all_every: int = 100  # iters between full collocation resampling; 0 disables


@dataclass
class LossWeights:
    """Loss term weights  w_k in L_total = sum w_k L_k."""
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
    """Three-phase training schedule."""
    # Phase 1 — BC/IC pre-train
    phase1_iters: int = 2_000
    phase1_lr: float = 1e-3

    # Phase 2 — full physics (Adam + cosine annealing)
    phase2_iters: int = 40_000
    phase2_lr_start: float = 1e-3
    phase2_lr_end: float = 1e-5

    # Phase 3 — L-BFGS refinement
    phase3_iters: int = 10_000   # max function evaluations

    # Stopping criterion
    loss_target: float = 1e-4

    # Checkpointing
    checkpoint_dir: Path = ROOT / "checkpoints"
    log_every: int = 500         # print + csv log interval
    plot_every: int = 1_000      # monitoring plot interval
    checkpoint_every: int = 500  # latest checkpoint interval
    latest_checkpoint: str = "latest.pth"


@dataclass
class Config:
    """Top-level config bundling all sub-configs."""
    case: CaseConfig = field(default_factory=CaseConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    weights: LossWeights = field(default_factory=LossWeights)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output_dir: Path = ROOT / "outputs"
    seed: int = 42


def config_from_dict(data: dict[str, Any] | None,
                     fallback: Config | None = None) -> Config:
    """Rebuild a Config object from a checkpoint/run_config dictionary."""
    run_cfg = Config() if fallback is None else copy.deepcopy(fallback)

    def _apply(obj: Any, values: dict[str, Any]) -> None:
        valid = {item.name for item in fields(obj)} if is_dataclass(obj) else set()
        for key, value in values.items():
            if key not in valid:
                continue
            current = getattr(obj, key)
            if is_dataclass(current) and isinstance(value, dict):
                _apply(current, value)
            elif isinstance(current, Path):
                setattr(obj, key, Path(value))
            else:
                setattr(obj, key, value)

    if isinstance(data, dict):
        _apply(run_cfg, data)
    return run_cfg


# Default config used by all scripts
cfg = Config()
