# Verification Log — 2D Axisymmetric PINNs for Pipe Solidification
> Append a new row to each table whenever a verification run completes.  
> Reference: cross-check every entry against `PROJECT_STATE.md` network version.

---

## 1. Residual Magnitude Checks (unit tests)

Verifies that each PDE residual is $O(1)$ on a known analytical IC (Poiseuille flow, uniform temperature).

| Date | Net version | $\mathcal{R}_1$ mass | $\mathcal{R}_2$ mom-x | $\mathcal{R}_3$ mom-r | $\mathcal{R}_4$ E-fluid | $\mathcal{R}_5$ E-dep | $\mathcal{R}_6$ Stefan | Notes |
|---|---|---|---|---|---|---|---|---|
| _(pending)_ | — | — | — | — | — | — | — | — |

---

## 2. Training Convergence Records

| Date | Net version | Phase reached | Final $\mathcal{L}_{total}$ | $\mathcal{L}_{BC}$ | $\mathcal{L}_{Stefan}$ | Iters | Wall time | Notes |
|---|---|---|---|---|---|---|---|---|
| _(pending)_ | — | — | — | — | — | — | — | — |

---

## 3. Accuracy vs MATLAB CFD Reference

**Reference case:** Pe=1, Ste=0.1, Re=10, $\hat{t}_{end}=1.0$

$L^2$ relative error defined as:
$$e_{L^2}(\phi) = \frac{\|\phi_\text{PINN} - \phi_\text{CFD}\|_2}{\|\phi_\text{CFD}\|_2}$$

| Date | Net version | $e_{L^2}(T_f)$ | $e_{L^2}(r_{int})$ | $e_{L^2}(u_x)$ | $e_{L^2}(p)$ | Snapshot times | Notes |
|---|---|---|---|---|---|---|---|
| _(pending)_ | — | — | — | — | — | — | — |

**Target:** $e_{L^2} < 5\%$ for $T_f$ and $r_{int}$ to pass Phase 6 milestone.

---

## 4. Analytical Solution Checks (where available)

For limiting cases where analytical solutions exist.

### 4a. Pure conduction in annulus (no flow, no phase change)
Steady-state: $T(r) = T_w + (T_{in}-T_w) \ln(r/r_w)/\ln(r_a/r_w)$

| Date | Net version | Max pointwise error | $L^2$ error | Notes |
|---|---|---|---|---|
| _(pending)_ | — | — | — | — |

### 4b. Planar Stefan problem (1D, semi-infinite)
Interface position: $r_{int}(t) = 1 - 2\lambda\sqrt{t}$ where $\lambda$ satisfies the transcendental equation.

| Date | Net version | $e_{L^2}(r_{int})$ at $\hat{t}=1$ | Notes |
|---|---|---|---|
| _(pending)_ | — | — | — |

---

## 5. Sensitivity / Ablation Studies

| Date | Change tested | Baseline loss | Modified loss | Verdict |
|---|---|---|---|---|
| _(pending)_ | — | — | — | — |
