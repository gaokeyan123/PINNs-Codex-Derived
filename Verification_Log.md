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
| 2026-04-29 | v1.0 | Phase 3 short CPU smoke | $9.4286\times10^1$ | $2.66\times10^1$ | $1.42$ | 48 | 6.7 s | Best checkpoint at iter 30 with $L=5.4630$; full 50k training not run locally |
| 2026-04-29 | v1.0 | Phase 2 reduced CPU case | $1.0638\times10^1$ | $4.58$ | $0.539$ | 1600 | 441.9 s total | Pe=1, Ste=0.1, Re=10; mean thickness at $t=1$ was 0.355, max thickness 1.000; target mean 0.5 not reached |
| 2026-04-29 | v1.0 + interface stabilization | Phase 3 smoke | $2.8572\times10^2$ | $6.67\times10^1$ | $7.84$ | 5 | 4.0 s | Code-path smoke only: new `rint_mono` and `rint_smooth` terms logged; full reduced case still pending after interruption |
| 2026-04-29 | v1.1 + corrected Stefan | Phase 2 reduced CPU case | $7.4526$ | $2.316$ | $0.064$ | 1600 | 410.8 s | Pe=1, Ste=0.1, Re=10, $\hat t_{end}=1$; no local collapse, max thickness at $t=1$ was 0.078, so 0.5 thickness is not physically reached in this time window |

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
| 2026-04-29 | Add inlet $r_{int}$ anchor, monotone-in-time/downstream interface penalties, stronger axial smoothness, and raise $T$ continuity weight to 100 | $1.0638\times10^1$ at iter 1600; nonphysical local collapse | Removed local collapse but did not fix Stefan speed by itself | Superseded |
| 2026-04-29 | Correct Stefan scaling and normalize network coordinates | $1.0638\times10^1$ with local collapse | $7.4526$ at iter 1600; max thickness 0.078 at $t=1$ | Accepted as physics fix; target 0.5 requires longer-time case or MATLAB reference confirmation |
| _(pending)_ | — | — | — | — |
