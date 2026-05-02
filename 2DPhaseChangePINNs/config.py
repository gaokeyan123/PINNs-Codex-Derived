"""
Single source of truth for all hyperparameters.
Edit this file only — everything else imports from here.
"""
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).parent


@dataclass
class CaseConfig:
    """Non-dimensional physical parameters for the simple test case."""
    Pe: float = 1.0        # Peclet number:  rho_f Cp_f U r_w / k_f
    Ste: float = 0.1       # Stefan number:  Cp_f dT / L_f
    Re: float = 10.0       # Reynolds number (laminar)
    k_ratio: float = 1.0   # k_dep / k_f  (equal conductivities, simplest case)

    # Geometry (non-dimensional, r_w = 1 is the reference length)
    r_a: float = 0.0       # inner axis radius (0 = solid cylinder)
    r_w: float = 1.0       # outer wall radius (= 1 by definition of scaling)
    L: float = 5.0         # pipe length

    # Time
    t_end: float = 1.0     # one thermal diffusion time = r_w^2 / alpha_dep

    # Boundary / initial conditions (non-dimensional temperatures)
    Theta_in: float = 1.0       # hot inlet fluid temperature
    Theta_wall: float = 0.0     # cold outer wall temperature
    Theta_solidus: float = 0.5  # solidification temperature (midpoint)

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
    rint_mono: float = 10.0
    rint_x_mono: float = 10.0
    rint_smooth: float = 1.0
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


# Default config used by all scripts
cfg = Config()
