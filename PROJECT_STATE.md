# Project State

## Scope

This repository now tracks only the active sharp-interface Case B PINN pipeline:

- configuration, network, sampling, equations, losses, training, postprocess,
  and residual diagnostics
- pytest coverage for the active implementation
- current project notes and equation references

Generated artifacts are intentionally not part of the tracked project. Training
will recreate `2DPhaseChangePINNs/checkpoints/` and
`2DPhaseChangePINNs/outputs/` as needed, and those folders remain ignored.
Within `outputs/`, postprocess products live under `postprocess/` and pulled
HPC job folders live under `hpc_results/`.

Training is Phase 1 + Phase 2 by default. Phase 3 L-BFGS refinement is
intentionally disabled with `phase3_iters = 0`; do not re-enable it unless a
future run explicitly asks for a Phase 3 experiment.

## Active Entry Points

Run commands from `2DPhaseChangePINNs/`.

| Command | Purpose |
|---|---|
| `python train.py --smoke` | Fast training smoke run |
| `python train.py --help` | Full training CLI |
| `python postprocess.py --help` | Plot fields, interface, thickness, pressure, and residual maps |
| `python diagnose_residuals.py --help` | Compute raw residual diagnostics from a checkpoint |
| `python -m pytest -q` | Run the verification test suite |

## Active Case

The current case uses only the Excel-consistent nondimensional convention:

| Quantity | Value |
|---|---:|
| `Pe_D` | `14.329521` |
| `Re_D` | `1.53` |
| `Ste` | `0.06275` |
| `k_ratio` | `1.0` |
| `L` | `10.25` |
| `tau_end` | `0.780487804878` |
| `Theta_in` | `2.0` |
| `Theta_solidus` | `1.0` |
| `Theta_wall` | `0.0` |
| `hot_wall_length` | `0.25` |

Coordinates and temperature are

```text
r = r_dim / r_w
x = x_dim / r_w
tau = t_dim U / L_total
Theta = (T - T_w) / (T_int - T_w)
```

The equations and loss definitions are documented in
`2DPhaseChangePINNs/SHARP_INTERFACE_EQUATIONS.md`.

## Code Map

| File | Responsibility |
|---|---|
| `config.py` | Dataclass configuration and checkpoint config rebuild |
| `network.py` | Fourier-feature PINN with field and interface heads |
| `sampling.py` | Interior, interface, boundary, and IC collocation batches |
| `equations.py` | PDE, interface, BC, and IC residuals |
| `losses.py` | Weighted loss assembly and logging terms |
| `train.py` | Phase 1 + Phase 2 training loop, checkpoints, CSV logs, monitor plots |
| `postprocess.py` | Plots and validation summaries into `outputs/postprocess/` by default |
| `diagnose_residuals.py` | Raw residual RMS diagnostics |
| `tests/` | Unit tests for network, sampling, equations, and losses |

## Cleanup Policy

The tracked tree should stay small and readable. Keep source, tests, and current
docs in Git. Leave generated checkpoints, plots, CSV logs, temporary caches, and
machine-specific helper files ignored.

The Python code is intentionally single-path: current-format checkpoints,
`tau` time coordinates, and the active Excel-consistent Case B convention only.
