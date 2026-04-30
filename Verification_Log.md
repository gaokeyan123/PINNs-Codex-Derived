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
| 2026-04-30 | Add raw residual RMS diagnostics and run on `plot_case_corrected` iter 800 checkpoint | Weighted training groups only | Fresh-batch weighted total $9.6072\times10^3$; dominant group is radial momentum ($RMS=97.0$, weighted $9.4092\times10^3$), caused mainly by near-axis singular spikes; next large residual is fluid energy ($RMS=13.05$) | Diagnostic accepted; next fix should handle axis/radial-velocity conditioning before retuning physics weights |
| 2026-04-30 | Clean-pipe no-phase diagnostic: set $\Theta_w=\Theta_{in}=\Theta_{IC}=1>\Theta_s$, force $H=0$, disable Stefan/interface/deposit losses | Phase-change-coupled checkpoint had mixed energy/momentum errors | Conservative fixed-batch run reached training loss $47.17$ at 500 iters; fresh residuals were dominated by near-axis spikes: $RMS(R_{mom,r})=1.01\times10^4$, but excluding only $r<0.005$ reduced it to $11.9$ and excluding $r<0.02$ reduced it to $2.86$ | Confirms phase change is not the first blocker; axis regularity/sampling must be fixed before judging momentum convergence |
| 2026-04-30 | No-phase clean-pipe run with interior PDE sampling restricted to $r\ge0.01$ while keeping axis BC points | Previous no-phase fresh diagnostic had $RMS(R_{mom,r})=1.01\times10^4$ due to tiny nonzero $r$ points | 1000 Adam iters, lr $3\times10^{-4}$, resample every 100: final training loss $2.240$; fresh residuals $RMS(R_{mom,x})=1.34$, $RMS(R_{mom,r})=0.634$, $RMS(R_E)=0.263$, $RMS(R_{mass})=0.213$ | Accepted as evidence that excluding near-axis interior PDE points is a viable mild fix before hard-coding $u_r=r\tilde u_r$ |
| 2026-04-30 | Longer no-phase clean-pipe run with $r\ge0.01$ and field/residual plots | 1000-iter run had fresh $RMS(R_{mom,x})=1.34$ and $RMS(R_{mom,r})=0.634$ | 2000 Adam iters: fresh $RMS(R_{mom,x})=0.820$, $RMS(R_{mom,r})=0.359$, $RMS(R_E)=0.118$, $RMS(R_{mass})=0.155$; $\Theta_f$ at $t=1$ was $0.9857/0.9939/0.9970$ min/mean/max; short fixed-batch L-BFGS lowered training loss but did not improve fresh momentum residuals | No-phase thermal field is effectively steady; flow is much improved but not yet accurate enough to declare full momentum convergence |
| 2026-04-30 | Continued no-phase clean-pipe Adam run from iter 2000 to iter 10000 with target fresh RMS $<10^{-3}$ | At iter 2000, fresh $RMS(R_{mom,x})=0.820$ | Target was not reached. At iter 10000, fresh residuals were $RMS(R_{mom,x})=0.278$, $RMS(R_{mom,r})=0.0804$, $RMS(R_{mass})=0.0659$, $RMS(R_E)=0.00192$; $\Theta_f$ at $t=1$ was $0.9998/1.0000/1.0000$ min/mean/max | Plain longer Adam training is not sufficient to reach $10^{-3}$ momentum accuracy; axial momentum/pressure conditioning needs a method change |
| 2026-04-30 | Switched execution environment to CUDA PyTorch on Python 3.12 and continued no-phase run to iter 10500 with unchanged equations/code | Iter 10000 fresh residuals were $RMS(R_{mom,x})=0.278$, $RMS(R_{mom,r})=0.0804$, $RMS(R_{mass})=0.0659$, $RMS(R_E)=0.00192$ | CUDA sanity passed on RTX 4070 Laptop GPU. After 500 GPU Adam steps from a model-only iter-10000 checkpoint: fresh $RMS(R_{mom,x})=0.248$, $RMS(R_{mom,r})=0.192$, $RMS(R_{mass})=0.0785$, $RMS(R_E)=0.00127$ | GPU path works, but target $10^{-3}$ is still not reached; current GPU memory was heavily occupied, so larger batches may require freeing VRAM |
| 2026-04-30 | Long CUDA no-phase run with larger collocation counts, target fresh RMS $<10^{-3}$ | Iter 10500 GPU checkpoint had fresh $RMS(R_{mom,x})=0.248$, $RMS(R_{mom,r})=0.192$, $RMS(R_{mass})=0.0785$, $RMS(R_E)=0.00127$ | Ran to iter 30500 with $N_\Omega=1024$, $N_{BC}=256$, $N_{IC}=512$, lr $10^{-5}$, $r\ge0.01$. Target was not reached; final fresh residuals were $RMS(R_{mass})=0.0538$, $RMS(R_{mom,x})=0.0495$, $RMS(R_{mom,r})=0.0318$, $RMS(R_E)=3.67\times10^{-5}$ | GPU and larger batches strongly improved momentum but plain Adam still plateaus above $10^{-3}$ |
| 2026-04-30 | CUDA no-phase continuation from iter 30500 to iter 80500 with unchanged equations/code | Iter 30500 fresh residuals were $RMS(R_{mass})=0.0538$, $RMS(R_{mom,x})=0.0495$, $RMS(R_{mom,r})=0.0318$, $RMS(R_E)=3.67\times10^{-5}$ | Target was not reached. Final fresh residuals: $RMS(R_{mom,x})=1.78\times10^{-2}$, $RMS(R_{mom,r})=1.32\times10^{-2}$, $RMS(R_{mass})=4.53\times10^{-3}$, $RMS(R_E)=8.70\times10^{-6}$; plots from `postprocess_no_phase.py` show $\Theta_f=1.0000/1.0000/1.0000$, $u_x=-0.0031/1.3034/1.9996$, $u_r=-0.0020/0.0006/0.0041$, and $p=-0.0032/1.6898/3.2080$ min/mean/max. Active training time for the full no-phase sequence from scratch to iter 80500 was about 2 h 9 min by CSV timestamps | Temperature is converged and flow shape is physically reasonable, but momentum residuals plateau around $10^{-2}$ under plain Adam |
| _(pending)_ | — | — | — | — |
