# Verification Log

This file records verification of the current cleaned project. Historical run
artifacts are not tracked here; regenerate plots, checkpoints, and CSV files
from the active commands when needed.

## Required Checks

Run from `2DPhaseChangePINNs/` unless noted otherwise.

| Check | Command |
|---|---|
| Unit tests | `python -m pytest -q` |
| Training CLI | `python train.py --help` |
| Postprocess CLI | `python postprocess.py --help` |
| Diagnostics CLI | `python diagnose_residuals.py --help` |
| Repository status | `git status --short` from repo root |

## Current Notes

- The active case is documented in `PROJECT_STATE.md`.
- The governing equations are documented in
  `2DPhaseChangePINNs/SHARP_INTERFACE_EQUATIONS.md`.
- Generated `checkpoints/`, `outputs/`, cache folders, and local helper scripts
  are intentionally ignored. Postprocess artifacts belong in
  `2DPhaseChangePINNs/outputs/postprocess/`; pulled HPC results belong in
  `2DPhaseChangePINNs/outputs/hpc_results/`.
- Phase 3 is intentionally disabled by default (`phase3_iters = 0`). The
  standard workflow is Phase 1 + Phase 2 only; re-enable Phase 3 only for an
  explicit L-BFGS experiment.

## Latest Verification

| Date | Change | Result |
|---|---|---|
| 2026-05-15 | Phase 1 project cleanup | Completed before Phase 2 code cleanup |
| 2026-05-15 | Phase 2 single-nondim code cleanup | `python -m pytest -q -p no:cacheprovider --basetemp=.pytest_tmp tests` passed: 92 tests |
| 2026-05-15 | Phase 2 CLI smoke checks | `train.py --help`, `postprocess.py --help`, and `diagnose_residuals.py --help` passed |
| 2026-05-16 | Disabled Phase 3 by default | `tests/test_config.py` passed; `train.py --smoke --device cpu --plot-every 0` stopped after Phase 2 with `phase3 disabled` log |
