# ROADMAP — 2D Axisymmetric PINNs for Pipe Solidification
> Checklist tracking. Check off `[x]` as each task is completed.  
> Cross-reference `PROJECT_STATE.md` for architecture decisions at each step.

---

## PHASE 0 — Problem Setup ✅

- [x] Derive full 2D axisymmetric governing equations (mass, momentum, energy, Stefan)
- [x] Non-dimensionalise all equations (Pe, Ste, Re, Fo)
- [x] Define simple test case: Pe=1, Ste=0.1, Re=10, constant wall-T, laminar
- [x] Initialise git repository in `Claude Derived/`
- [x] Create project folder `2DPhaseChangePINNs/`
- [x] Write `PROJECT_STATE.md`, `ROADMAP.md`, `Verification_Log.md`
- [x] Write `.gitignore` with `.pth` weight exclusion and naming convention

---

## PHASE 1 — Network Architecture & Non-Dimensionalisation 🟡

> **Goal:** A working forward pass that takes $(r,x,t)$ and produces all fields.

- [ ] **`config.py`** — Single source of truth for all hyperparameters
  - Case parameters: Pe, Ste, Re, geometry, $t_{end}$
  - Network hyperparameters: layers, width, Fourier $\sigma$, $\epsilon$ (interface smoothing)
  - Training hyperparameters: lr, batch sizes, phase boundaries, loss weights
  - Output paths and checkpoint naming

- [ ] **`network.py`** — Fourier-encoded MLP with two heads
  - Fourier feature layer: $\gamma(v) = [\sin(2\pi Bv), \cos(2\pi Bv)]$, $B \sim \mathcal{N}(0,\sigma^2)$
  - Shared trunk: 8 × 128, tanh activation
  - Field head: outputs $(\hat{u}_r, \hat{u}_x, \hat{p}, \Theta_f, \Theta_{dep})$
    - $\Theta_f, \Theta_{dep}$: sigmoid activation
    - $\hat{u}_r, \hat{u}_x, \hat{p}$: linear
  - Interface head: masked input $(x,t)$ → $\hat{r}_{int}$ via sigmoid × $r_w$
  - Unit test: check output shapes and that $\hat{r}_{int}$ gradient w.r.t. $r$ is exactly 0

- [ ] **Milestone:** Forward pass runs, output shapes correct, Fourier encoding verified.  
  → Update `PROJECT_STATE.md` Section 2 checkboxes.

---

## PHASE 2 — PDE Residuals via Autograd ✅

> **Goal:** All five PDE residuals + Stefan condition computable from a single forward pass.

- [x] **`equations.py`** — All residuals using `torch.autograd.grad`
  - [x] Derivative utilities: `_grad`, `_laplacian_cyl`, `_poiseuille`
  - [x] $\mathcal{R}_1$: mass conservation — incompressible continuity, $(1-H_\epsilon)$ weighted
  - [x] $\mathcal{R}_2$: axial momentum — includes $(1/\text{Pe})$ unsteady, $(1/\text{Re})$ viscous
  - [x] $\mathcal{R}_3$: radial momentum — geometric source $-u_r/r^2$ included
  - [x] $\mathcal{R}_4$: fluid energy — advection–diffusion with Pe coefficient
  - [x] $\mathcal{R}_5$: deposit energy — pure diffusion, $k_{ratio}$ coefficient, $H_\epsilon$ weighted
  - [x] $\mathcal{R}_6$: Stefan condition — Ste·$\dot{r}_{int}$ = flux jump (autograd $\partial r_{int}/\partial t$)
  - [x] $\mathcal{R}_7$: temperature continuity $\Theta_f = \Theta_{dep} = \Theta_{solidus}$ at interface
  - [x] BC residuals: wall (Dirichlet T, no-slip), inlet (Poiseuille + hot T), outlet (zero-grad), axis (symmetry)
  - [x] IC residuals: $r_{int}=r_w$, $\Theta_f=1$, Poiseuille $u_x$, $u_r=0$ at $t=0$
  - [x] `compute_all_residuals(model, batch, cfg)` — master function returning all 23 residual tensors
- [x] `tests/test_equations.py` — 22 tests: derivative accuracy, Heaviside weighting, shape/NaN checks

- [ ] **Milestone:** Each residual verified $O(1)$ on random inputs. Log in `Verification_Log.md`.

---

## PHASE 3 — Sampling Strategy ✅

> **Goal:** Collocation points that cover the domain well and chase the moving interface.

- [x] **`sampling.py`** — Point generators (no scipy dependency)
  - [x] `_lhs(N, d, seed)`: pure-PyTorch Latin Hypercube; verified by stratum bincount test
  - [x] `sample_interior(N, cfg)`: LHS in $(\hat{r}, \hat{x}, \hat{t})$, spans both subdomains
  - [x] `sample_wall / inlet / outlet / axis`: fixed boundary coordinate + LHS for free coords
  - [x] `sample_axis`: r = 1e-4 (not 0) to avoid 1/r singularity
  - [x] `sample_ic(N, cfg)`: LHS in $(\hat{r}, \hat{x})$ at $\hat{t}=0$
  - [x] `sample_interface(model, N, cfg)`: no_grad query → r_int values become r coords; detach + re-wrap
  - [x] `resample_interface(model, N, cfg, seed)`: thin wrapper called every 500 iters in training
  - [x] `build_batch(model, cfg, seed)`: assembles all 7 point sets with per-set sub-seeds
  - [x] `plot_batch(batch, cfg, save_path)`: scatter in $(x,r)$ plane — milestone visualisation
- [x] `tests/test_sampling.py` — 27 tests: LHS stratification, bounds, requires_grad, interface accuracy, reproducibility, plot

- [ ] **Milestone:** Run `python sampling.py` and verify `outputs/collocation_points.png` — interface points should form a coherent curve.

---

## PHASE 4 — Loss Function Assembly

> **Goal:** Single scalar loss with all weighted residual terms.

- [ ] **`losses.py`** — Composite loss
  - `physics_loss(model, P_Ω)` → $\mathcal{L}_1 + \mathcal{L}_2 + \mathcal{L}_3 + \mathcal{L}_4 + \mathcal{L}_5$
  - `interface_loss(model, P_Γ)` → $\mathcal{L}_6 + \mathcal{L}_7$
  - `bc_ic_loss(model, P_wall, P_in, P_out, P_0)` → $\mathcal{L}_8$
  - `total_loss(model, all_points, weights)` → weighted sum
  - Individual term logging (each $\mathcal{L}_k$ returned separately for monitoring)

- [ ] Initial weights: $w_{1-5}=1$, $w_{6,7}=10$, $w_8=100$
- [ ] **Milestone:** Total loss computes without NaN on random network. Log initial loss values.

---

## PHASE 5 — Training Loop

> **Goal:** Full three-phase training with monitoring and checkpointing.

- [ ] **`train.py`** — Training driver
  - Phase 1 (0–2k iters): Adam lr=$10^{-3}$, BC/IC loss only
  - Phase 2 (2k–40k iters): Adam with cosine annealing lr=$10^{-3}\to10^{-5}$, all losses
    - Resample $\mathcal{P}_\Gamma$ every 500 iters
    - Log each $\mathcal{L}_k$ to terminal and to `outputs/loss_history.csv`
    - Save checkpoint if total loss improves: `pinns_v1.0_ep{ep:05d}_loss{loss:.4e}.pth`
    - Update `checkpoints/BEST_MODELS.md` on new best
  - Phase 3 (40k–50k iters): L-BFGS full-batch refinement
  - Early stop if total $\mathcal{L} < 10^{-4}$

- [ ] **Milestone:** Training completes Phase 1 without divergence. Loss curve plotted and saved. Update `PROJECT_STATE.md`.

---

## PHASE 6 — Post-Processing & Validation

> **Goal:** Produce publication-quality figures and validate against MATLAB CFD output.

- [ ] **`postprocess.py`** — Result extraction and plotting
  - Dense grid evaluation: $200 \times 200 \times 5$ time snapshots
  - Plots at $\hat{t} = 0.25, 0.5, 0.75, 1.0$:
    - $\Theta(r,x)$ filled contour (fluid and deposit with interface marked)
    - $\hat{u}_x(r,x)$ showing velocity acceleration in narrowing channel
    - $\hat{p}(x)$ axial pressure profile
    - $\hat{r}_{int}(x)$ interface position overlaid on MATLAB reference
  - Pointwise PDE residual heatmap (where network violates equations most)
  - $L^2$ relative error vs MATLAB CFD for $T_f$, $r_{int}$

- [ ] Run MATLAB code at Pe=1, Ste=0.1 to generate reference data, export to `outputs/matlab_ref.mat`
- [ ] **Milestone:** $L^2$ error < 5% for $T_f$ and $r_{int}$. Log in `Verification_Log.md`.

---

## PHASE 7 — Parametric Extension (Future)

> **Goal:** Train across a range of (Pe, Ste) and use network as a true surrogate.

- [ ] Extend inputs to $(r, x, t, \text{Pe}, \text{Ste})$ — 5D input
- [ ] Sample training data across Pe $\in [0.1, 10]$, Ste $\in [0.05, 0.5]$
- [ ] Evaluate generalisation error on held-out (Pe, Ste) pairs
- [ ] Compare inference time: PINN forward pass vs MATLAB simulation
- [ ] Write results section for thesis Chapter X

---

## Appendix: File Map

```
2DPhaseChangePINNs/
├── config.py           ← Phase 1
├── network.py          ← Phase 1
├── equations.py        ← Phase 2
├── sampling.py         ← Phase 3
├── losses.py           ← Phase 4
├── train.py            ← Phase 5
├── postprocess.py      ← Phase 6
├── checkpoints/
│   ├── BEST_MODELS.md  ← tracked by git
│   └── *.pth           ← gitignored
├── outputs/
│   ├── loss_history.csv
│   ├── matlab_ref.mat
│   └── *.png / *.pdf   ← gitignored
└── tests/
    └── test_network.py ← smoke tests for forward pass
```
