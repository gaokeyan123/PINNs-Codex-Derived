# ROADMAP - 2D Axisymmetric PINNs for Pipe Solidification

Checklist tracking. Cross-reference `PROJECT_STATE.md` for architecture and
decision history.

## Phase 0 - Problem Setup

- [x] Derive 2D axisymmetric mass, momentum, energy, and Stefan equations.
- [x] Define sharp-interface verification case and repository structure.
- [x] Track checkpoints outside git while keeping model metadata in docs.

## Phase 1 - Network Architecture And Nondimensionalization

Goal: a forward pass that takes `(r, x, tau)` and predicts fields plus the
moving interface.

- [x] `config.py`: single source of truth for `Pe_D`, `Re_D`, `Ste`,
  `tau_end`, geometry, temperatures, sampling, training, and loss weights.
- [x] `network.py`: Fourier-encoded MLP with field head
  `(u_r, u_x, p, Theta_f, Theta_dep)` and interface head `r_int(x,tau)`.
- [x] Network input scaling: `r_n=r/r_w`, `x_n=x/L`, `tau_n=tau/tau_end`.
  This is feature scaling only; equations differentiate with respect to
  `(r, x, tau)`.
- [x] Unit tests check output shapes, bounded temperatures/interface, and
  zero `dr_int/dr`.

## Phase 2 - PDE Residuals Via Autograd

Goal: all PDE, interface, BC, and IC residuals are computable from one forward
pass.

- [x] `equations.py`: `_grad`, `_laplacian_cyl`, `_poiseuille`, and all
  residual functions.
- [x] Continuity with cylindrical `u_r/r` term and fluid-region weighting.
- [x] Momentum uses Excel-consistent coefficients: `(1/A)` on unsteady terms
  and `(2/Re_D)` on viscous terms.
- [x] Energy uses Excel-consistent coefficients: `(1/A)` on unsteady terms and
  `(2/Pe_D)` on thermal diffusion.
- [x] Stefan condition uses
  `(2A/Pe_D) Ste (k_ratio dTheta_dep/dr - dTheta_f/dr)`.
- [x] Interface terms include temperature continuity, no-slip velocity,
  time monotonicity, and downstream monotonicity.
- [x] BC/IC residuals cover wall, inlet, outlet, axis, and clean-pipe initial
  condition at `tau=0`.
- [x] Tests include analytic coefficient checks for momentum, energy, and
  Stefan residuals.

## Phase 3 - Sampling Strategy

- [x] Pure-PyTorch Latin Hypercube sampler.
- [x] Interior points in `r in [r_min, r_w]`, `x in [0,L]`,
  `tau in [0,tau_end]`.
- [x] Boundary samplers for wall, inlet, outlet, axis, and IC.
- [x] Adaptive interface sampler queries the current `r_int(x,tau)` and
  detaches sampled coordinates for the training step.
- [x] Full-batch resampling is available to avoid fixed-collocation overfit.

## Phase 4 - Loss Assembly

- [x] `losses.py`: weighted MSE terms and Phase 1 BC/IC-only fast path.
- [x] Current weights: PDE terms `1`, Stefan `10`, interface temperature `100`,
  interface velocity `10`, interface monotonic penalties `10`, BC/IC `100`.
- [x] CSV logging and compact terminal formatting.

## Phase 5 - Training Loop

- [x] `train.py`: three-phase driver with smoke mode, checkpoint/resume, CSV
  logs, monitor plots, and target-thickness stopping.
- [x] Smoke training exercises Phase 1, Phase 2, optional Phase 3, checkpoint
  writing, and postprocess compatibility.
- [ ] Run a fresh Case B Excel-nondim training job; do not resume old
  thermal-time checkpoints into this new physics setup.

## Phase 6 - Postprocessing And Validation

- [x] `postprocess.py`: field snapshots, interface profiles, thickness
  profiles, pressure profiles, and residual maps.
- [x] Default Case B comparison snapshots map Fluent `2,4,6,8 s` to
  `tau = 2/10.25, 4/10.25, 6/10.25, 8/10.25`.
- [ ] Compare fresh Excel-nondim PINN results against Case B Fluent reference.
- [ ] Log verification metrics in `Verification_Log.md`.

## Phase 7 - Parametric Extension

- [ ] Extend input space to include governing groups after the single-case
  nondimensionalization is validated.
- [ ] Train over ranges of `Pe_D`, `Re_D`, and `Ste`.
- [ ] Compare surrogate inference time and error against CFD.
