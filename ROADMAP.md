# Roadmap

## Done

- Derived the sharp-interface 2D axisymmetric PINN formulation for Case B.
- Implemented the active Excel-consistent nondimensionalization.
- Built the core Python pipeline:
  `config.py`, `network.py`, `sampling.py`, `equations.py`, `losses.py`,
  `train.py`, `postprocess.py`, and `diagnose_residuals.py`.
- Added tests for network outputs, sampling, residual coefficients, boundary
  conditions, and loss assembly.
- Cleaned the project tree so generated artifacts and obsolete run helpers are
  outside the tracked source set.
- Removed compatibility code for superseded nondimensionalization conventions.
- Standardized the public training and postprocess interfaces on `tau`.

## Current Pipeline

- `train.py` runs staged training with smoke mode, checkpoint/resume, CSV logs,
  monitor plots, and target-thickness stopping.
- `postprocess.py` generates field snapshots, interface profiles, thickness
  profiles, pressure profiles, residual maps, and validation summaries.
- `diagnose_residuals.py` reports raw residual RMS values from a checkpoint.
- Tests are kept as the safety net for the cleaned single-convention code.

## Next Phase

- Re-run the available verification checks after each cleanup pass.
- Keep tightening comments and names where they still obscure the active math.
- Preserve the current equations while preparing for fresh Case B training.

## Later Work

- Run a fresh long Case B training job using the cleaned code.
- Compare the resulting interface and thickness profiles against the Fluent
  reference data.
- Extend the single-case model into a parametric surrogate after the active
  nondimensionalization is stable.
