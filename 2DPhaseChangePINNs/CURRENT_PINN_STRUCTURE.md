# Current PINN Structure - Sharp Interface Case B

This note documents the active sharp-interface verification model. The current
nondimensionalization is the Excel-consistent Case B convention.

## Active Case

| Quantity | Value |
|---|---:|
| `Pe_D` | `14.329521` |
| `Ste` | `0.06275` |
| `Re_D` | `1.53` |
| `k_ratio = k_dep/k_f` | `1.0` |
| `r_a` | `0.0` |
| `r_w` | `1.0` |
| `L = L_total/r_w` | `10.25` |
| `tau_end = 8/L` | `0.780487804878` |
| `Theta_in` | `2.0` |
| `Theta_solidus` | `1.0` |
| `Theta_wall` | `0.0` |
| `hot_wall_length` | `0.25` |

The physics coordinates are

$$
r=\frac{r_{dim}}{r_w},\qquad
x=\frac{x_{dim}}{r_w},\qquad
\tau=\frac{t_{dim}U}{L_{total}}.
$$

The diameter-based groups are

$$
Re_D=\frac{U(2r_w)}{\nu},\qquad
Pe_D=\frac{U(2r_w)}{\alpha_f}.
$$

Temperature uses

$$
\Theta = \frac{T - T_w}{T_{int} - T_w}.
$$

For `Tin = 5 C`, `Tint = 0 C`, and `Tw = -5 C`:

$$
\Theta_{in}=2,\qquad
\Theta_{int}=\Theta_{solidus}=1,\qquad
\Theta_w=0.
$$

The current deposit energy residual assumes equal fluid/deposit volumetric
heat capacity. The only solid/fluid thermal-material ratio in the code is
`k_ratio = k_dep/k_f`.

## Network

The field branch receives `(r, x, tau)` and predicts

$$
(u_r,\ u_x,\ p,\ \Theta_f,\ \Theta_{dep}).
$$

The interface branch receives only `(x, tau)` and predicts

$$
r_{int}(x,\tau).
$$

Inputs are normalized only for neural-network conditioning:

$$
r_n=r/r_w,\qquad x_n=x/L,\qquad \tau_n=\tau/\tau_{end}.
$$

This normalization is inside `network.forward()`. Autograd still computes PDE
derivatives with respect to the external physics coordinates `(r, x, tau)`.

Important implementation details:

- `Theta_f` and `Theta_dep` are bounded with a sigmoid to `[0, theta_max]`,
  where `theta_max = 2` for Case B.
- `r_int` is bounded to `[0, r_w]` with a sigmoid.
- `u_r`, `u_x`, and `p` are unbounded network outputs.
- The smooth region mask is `H=0.5*(1+tanh((r-r_int_detached)/eps))`, with
  `H approx 0` in fluid and `H approx 1` in deposit.
- `T_cont` trains temperatures at sampled interface points. The mask uses a
  detached interface for interior residual weighting.
- The current interface regularizers are time monotonicity and downstream
  monotonicity of `r_int`.

## Sampling

Training batches contain:

| Set | Coordinates |
|---|---|
| `interior` | LHS in `r in [r_min, r_w]`, `x in [0,L]`, `tau in [0,tau_end]` |
| `interface` | LHS in `(x,tau)`, with `r = current r_int(x,tau)` |
| `wall` | `r = r_w` |
| `inlet` | `x = 0` |
| `outlet` | `x = L` |
| `axis` | `r = 1e-4` |
| `ic` | `tau = 0` |

## Governing Residuals

Define

$$
\nabla^2_{cyl} f
=
\frac{\partial^2 f}{\partial r^2}
+
\frac{1}{r}\frac{\partial f}{\partial r}
+
\frac{\partial^2 f}{\partial x^2}.
$$

The implemented residuals are:

$$
R_{mass}=(1-H)\left(
\frac{\partial u_r}{\partial r}
+
\frac{u_r}{r}
+
\frac{\partial u_x}{\partial x}
\right).
$$

$$
R_{mom,x}=(1-H)\left(
\frac{1}{A}\frac{\partial u_x}{\partial \tau}
+
u_r\frac{\partial u_x}{\partial r}
+
u_x\frac{\partial u_x}{\partial x}
+
\frac{\partial p}{\partial x}
-
\frac{2}{Re_D}\nabla^2_{cyl}u_x
\right).
$$

$$
R_{mom,r}=(1-H)\left(
\frac{1}{A}\frac{\partial u_r}{\partial \tau}
+
u_r\frac{\partial u_r}{\partial r}
+
u_x\frac{\partial u_r}{\partial x}
+
\frac{\partial p}{\partial r}
-
\frac{2}{Re_D}
\left(\nabla^2_{cyl}u_r-\frac{u_r}{r^2}\right)
\right).
$$

$$
R_{E,f}=(1-H)\left(
\frac{1}{A}\frac{\partial \Theta_f}{\partial \tau}
+
u_r\frac{\partial \Theta_f}{\partial r}
+
u_x\frac{\partial \Theta_f}{\partial x}
-
\frac{2}{Pe_D}\nabla^2_{cyl}\Theta_f
\right).
$$

$$
R_{E,dep}=H\left(
\frac{1}{A}\frac{\partial \Theta_{dep}}{\partial \tau}
-
\frac{2k_{ratio}}{Pe_D}\nabla^2_{cyl}\Theta_{dep}
\right).
$$

$$
R_{Stefan}=
\frac{\partial r_{int}}{\partial \tau}
-
\frac{2A}{Pe_D}Ste
\left(
k_{ratio}\frac{\partial \Theta_{dep}}{\partial r}
-
\frac{\partial \Theta_f}{\partial r}
\right).
$$

Interface temperature, no-slip, and monotonic residuals:

$$
R_{T,f}=\Theta_f(r_{int},x,\tau)-\Theta_{solidus},
\qquad
R_{T,dep}=\Theta_{dep}(r_{int},x,\tau)-\Theta_{solidus},
$$

$$
R_{u,\Gamma,x}=u_x(r_{int},x,\tau),\qquad
R_{u,\Gamma,r}=u_r(r_{int},x,\tau),
$$

$$
R_{rint,\tau}=\max\left(0,\frac{\partial r_{int}}{\partial \tau}\right),
\qquad
R_{rint,x}=\max\left(0,\frac{\partial r_{int}}{\partial x}\right).
$$

## Boundary And Initial Conditions

Wall:

$$
\Theta_{dep}(r_w,x,\tau)=\Theta_w(x),\qquad
u_r(r_w,x,\tau)=0,\qquad
u_x(r_w,x,\tau)=0.
$$

Inlet:

$$
\Theta_f(r,0,\tau)=\Theta_{in},\qquad
u_r(r,0,\tau)=0,
$$

$$
u_x(r,0,\tau)=2\left[1-\left(\frac{r}{r_{int}(0,\tau)}\right)^2\right],
\qquad
r_{int}(0,\tau)=r_w.
$$

Outlet:

$$
\frac{\partial \Theta_f}{\partial x}(r,L,\tau)=0,\qquad
\frac{\partial u_x}{\partial x}(r,L,\tau)=0.
$$

Axis:

$$
u_r(0,x,\tau)=0,\qquad
\frac{\partial \Theta_f}{\partial r}(0,x,\tau)=0,\qquad
\frac{\partial u_x}{\partial r}(0,x,\tau)=0.
$$

Initial condition:

$$
r_{int}(x,0)=r_w,\qquad
\Theta_f(r,x,0)=\Theta_{in},\qquad
u_x(r,x,0)=2(1-r^2),\qquad
u_r(r,x,0)=0.
$$

## Loss Assembly

The weighted terms are:

| Term | Weight |
|---|---:|
| `mass` | `1` |
| `mom_x`, `mom_r` | `1` |
| `energy_fluid`, `energy_dep` | `1` |
| `stefan` | `10` |
| `T_continuity` | `100` |
| `interface_velocity` | `10` |
| `rint_mono`, `rint_x_mono` | `10` |
| `bc_ic` | `100` |

`bc_ic` is the average MSE over the 16 wall, inlet, outlet, axis, and initial
condition residual arrays. Phase 1 trains only `bc_ic`; full physics training
sums all terms.
