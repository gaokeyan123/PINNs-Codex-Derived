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
    """Excel-consistent non-dimensional physical parameters for Case B."""
    name: str = "20260410_nonDimm_goodmatchCaseB"

    nondim_scheme: str = "excel_radius_x_convective_tau_diameter_groups_v1"

    Pe_D: float = 14.329521  # Pe_D = U * (2 r_w) / alpha_f
    Ste: float = 0.06275     # Ste = Cp_f * (T_int - T_wall) / L_f
    Re_D: float = 1.53       # Re_D = U * (2 r_w) / nu
    k_ratio: float = 1.0     # k_dep / k_f

    # Geometry: r = r_dim/r_w and x = x_dim/r_w.
    r_a: float = 0.0       # inner axis radius (0 = solid cylinder)
    r_w: float = 1.0       # wall radius after scaling by r_w
    L: float = 10.25       # A = L_total/r_w, including the hot-wall inlet section

    # Time: tau = t_dim * U / L_total.
    tau_end: float = 8.0 / 10.25

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

    @property
    def Pe(self) -> float:
        """Backward-compatible alias for old checkpoints/scripts."""
        return self.Pe_D

    @Pe.setter
    def Pe(self, value: float) -> None:
        self.Pe_D = float(value)

    @property
    def Re(self) -> float:
        """Backward-compatible alias for old checkpoints/scripts."""
        return self.Re_D

    @Re.setter
    def Re(self, value: float) -> None:
        self.Re_D = float(value)

    @property
    def t_end(self) -> float:
        """Backward-compatible alias; current time variable is tau."""
        return self.tau_end

    @t_end.setter
    def t_end(self, value: float) -> None:
        self.tau_end = float(value)


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
        if isinstance(obj, CaseConfig):
            values = dict(values)
            if "Pe" in values and "Pe_D" not in values:
                values["Pe_D"] = values["Pe"]
            if "Re" in values and "Re_D" not in values:
                values["Re_D"] = values["Re"]
            if "t_end" in values and "tau_end" not in values:
                values["tau_end"] = values["t_end"]
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
        values = dict(data)
        case_keys = {item.name for item in fields(CaseConfig)} | {"Pe", "Re", "t_end"}
        flat_case_values = {key: value for key, value in values.items() if key in case_keys}
        if flat_case_values:
            nested_case = values.get("case", {})
            merged_case = dict(flat_case_values)
            if isinstance(nested_case, dict):
                merged_case.update(nested_case)
            values["case"] = merged_case
        _apply(run_cfg, values)
    return run_cfg


# Default config used by all scripts
cfg = Config()
