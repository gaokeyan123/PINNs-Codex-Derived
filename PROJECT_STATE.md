# PROJECT STATE
> **Auto-maintained by Claude.** Updated at every milestone or model logic change.  
> Last updated: 2026-04-29 | Phase: **3 — Sampling Strategy** ✅

---

## 1. Project Summary

**Title:** Physics-Informed Neural Networks (PINNs) as a CFD Surrogate for 2D Axisymmetric Pipe Solidification with Phase Change

**PhD Context:**  
This project develops a PINN-based surrogate model to replace conventional CFD (currently implemented in MATLAB) for a pipe solidification problem. Liquid flowing through a pipe solidifies against a cold outer wall, forming a growing deposit layer. The PINN takes the space-time coordinates $(r, x, t)$ as inputs and predicts all field variables as outputs, learning the governing physics through a composite PDE residual loss rather than labelled data.

**Governing Physics (2D axisymmetric conservative form):**
- Mass conservation (fluid)
- Axial + radial momentum (Navier–Stokes, laminar)
- Fluid energy equation (advection–diffusion)
- Deposit (solid) energy equation (pure diffusion)
- Stefan condition at the moving solid–liquid interface $r_\text{int}(x,t)$

**Benchmark / simple test case:**
| Parameter | Value |
|---|---|
| Péclet number Pe | 1.0 |
| Stefan number Ste | 0.1 |
| Reynolds number Re | 10 (laminar) |
| Geometry | $r_a=0$, $r_w=1$, $L=5$ (non-dim) |
| Wall BC | $\Theta_w = 0$ (cold, Dirichlet) |
| Inlet BC | $\Theta_{in} = 1$, Poiseuille $u_x$ |
| IC | Clean pipe, $r_{int}=1$, $\Theta_f=1$ |
| End time | $\hat{t}=1.0$ (one thermal diffusion time) |

---

## 2. Current Phase

**Phase 1 — Network Architecture & Non-Dimensionalisation**  
Status: 🟡 In Progress

Completed sub-tasks:
- [x] Governing equations derived in 2D axisymmetric conservative form
- [x] Non-dimensional groups defined (Pe, Ste, Re, Fo)
- [x] Roadmap and project files initialised
- [x] `config.py` — case parameters and non-dim scales
- [x] `network.py` — Fourier-encoded MLP with field + interface heads
- [x] `equations.py` — autograd PDE residuals (R1–R7 + all BCs/ICs)
- [x] `sampling.py` — Latin Hypercube interior, adaptive interface, boundary/IC samplers, plot_batch
- [ ] `losses.py` — weighted composite loss
- [ ] `train.py` — three-phase training loop
- [ ] `postprocess.py` — plots and MATLAB comparison

---

## 3. Architecture Memory

### 3.1 Network Design (v1.0)

| Property | Value | Rationale |
|---|---|---|
| Input | $(r, x, t)$ — 3 scalars | Full space-time coordinates |
| Fourier encoding | $\sigma=1.0$, 256 features | Overcomes spectral bias near interface |
| Shared trunk | 8 × 128, tanh | Tanh preferred over ReLU for smooth second derivatives needed by PDE residuals |
| Field head | → $(u_r, u_x, p, \Theta_f, \Theta_{dep})$ | Linear outputs except $\Theta$ use sigmoid |
| Interface head | Input masked to $(x, t)$ → $r_{int}$ | Sigmoid output scaled to $(0,1)$ |
| Total params | ~130k | Small enough for CPU training, fast on GPU |

**Decision log:**
- Single-network with two heads chosen over two separate networks (fluid + solid) to avoid discontinuous gradients at the interface during backpropagation.
- Smooth Heaviside $H_\epsilon(r, r_{int})$ with $\epsilon=0.02$ used to blend fluid and deposit PDE residuals, avoiding hard domain splitting.

### 3.2 Loss Function Configuration (v1.0)

| Term | Equation | Weight | Evaluated on |
|---|---|---|---|
| $\mathcal{L}_1$ | Mass conservation | 1 | Interior collocation $\mathcal{P}_\Omega$ |
| $\mathcal{L}_2$ | Axial momentum | 1 | Interior collocation |
| $\mathcal{L}_3$ | Radial momentum | 1 | Interior collocation |
| $\mathcal{L}_4$ | Fluid energy | 1 | Interior collocation |
| $\mathcal{L}_5$ | Deposit energy | 1 | Interior collocation |
| $\mathcal{L}_6$ | Stefan condition | 10 | Interface points $\mathcal{P}_\Gamma$ |
| $\mathcal{L}_7$ | $T$ continuity at interface | 10 | Interface points |
| $\mathcal{L}_8$ | All BCs + IC | 100 | Boundary/initial points |

**Collocation counts:** 20k interior, 5k interface (resampled every 500 iters), 2k per boundary, 5k IC.

### 3.3 Training Protocol (v1.0)

| Phase | Iters | Optimizer | LR | Loss active |
|---|---|---|---|---|
| 1 — BC/IC pre-train | 0–2k | Adam | $10^{-3}$ | $\mathcal{L}_8$ only |
| 2 — Full physics | 2k–40k | Adam + cosine anneal | $10^{-3} \to 10^{-5}$ | All terms |
| 3 — Refinement | 40k–50k | L-BFGS | 1.0 | All terms |

---

## 4. Key Technical Decisions Log

| Date | Decision | Reason |
|---|---|---|
| 2026-04-29 | Chosen single MLP + smooth Heaviside over sharp domain split | Avoids discontinuous interface gradient in backprop |
| 2026-04-29 | Fourier feature encoding with $\sigma=1.0$ | PINNs have spectral bias; Fourier encoding enables high-freq spatial modes |
| 2026-04-29 | Interface head masked to $(x,t)$ only | $r_{int}$ is physically 1D (no $r$-dependence) — encoding this as an inductive bias |
| 2026-04-29 | Stefan and BC losses weighted 10× and 100× | These are hard constraints; PDE residuals are softer |
| 2026-04-29 | `.pth` files gitignored; `BEST_MODELS.md` tracked | Prevents repo bloat; maintains version record for thesis |
| 2026-04-29 | `_grad` uses `grad_outputs=ones` + `allow_unused=True` | Standard batch-PINN gradient; allow_unused avoids crash when interface head receives r=0 gradient |
| 2026-04-29 | Cylindrical Laplacian clamps r to `_R_MIN=1e-6` | Prevents NaN at axis r=0; affects only axis-adjacent points |
| 2026-04-29 | `(1/Pe)` factor on unsteady momentum term | Time is scaled by thermal diffusion time τ=r_w²/α_f; convective and thermal scales coincide only at Pe=1 |
| 2026-04-29 | H_ε uses `.detach()` on r_int when computing Heaviside | Prevents gradient from flowing through H into interface head during interior PDE residuals |
| 2026-04-29 | Pure-PyTorch LHS (no scipy) in sampling.py | Portability: avoids scipy dependency; stratification verified by bincount test |
| 2026-04-29 | Axis points use r=1e-4, not r=0 | Avoids 1/r singularity in _laplacian_cyl; safely above _R_MIN=1e-6 |
| 2026-04-29 | sample_interface detaches r_int and re-wraps with requires_grad | Prevents stale computation graph references across training iterations |
| 2026-04-29 | resample_interface uses advancing seed each call | Ensures fresh (x,t) pairs every resample step; same seed → reproducible |

---

## 5. Autonomous Update Instructions

> **For Claude:** The following rules are mandatory and must be followed in every future session working on this project.

1. **At every milestone** (new file implemented, training phase completed, architecture changed, verification result obtained): update this file immediately before ending the session.
2. **When the network architecture changes**: update Section 3.1 with a new version number (e.g. v1.1) and record the old config inline as a strikethrough or sub-entry.
3. **When loss weights or training protocol change**: update Section 3.2/3.3 with the new values and log the reason in Section 4.
4. **When a verification run completes**: add a summary row to `Verification_Log.md` and update Section 2 checkboxes.
5. **Always update the `Last updated` date and `Phase` tag** at the top of this file.
6. **Never** leave the project in a state where `PROJECT_STATE.md` is stale relative to the code — if the code and this file disagree, fix this file first.
