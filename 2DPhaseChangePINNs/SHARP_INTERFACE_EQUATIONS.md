# Sharp-Interface PINN Equations

This file is the source of truth for the nondimensional equations currently
implemented in `equations.py` for case `20260410_nonDimm_goodmatchCaseB`.

## Nondimensional Variables

The active convention matches the Excel/Fluent setup:

$$
r=\frac{r_{dim}}{r_w},\qquad
x=\frac{x_{dim}}{r_w},\qquad
\tau=\frac{t_{dim}U}{L_{total}}.
$$

Velocity, pressure, and temperature are

$$
u=\frac{u_{dim}}{U},\qquad
p=\frac{p_{dim}}{\rho_f U^2},\qquad
\Theta=\frac{T-T_w}{T_{int}-T_w}.
$$

The pipe aspect ratio used in the equations is

$$
A=\frac{L_{total}}{r_w}=10.25.
$$

The Reynolds and Peclet numbers are diameter-based, as in the Excel workbook:

$$
Re_D=\frac{U(2r_w)}{\nu},\qquad
Pe_D=\frac{U(2r_w)}{\alpha_f}.
$$

The Stefan number is

$$
Ste=\frac{C_{p,f}(T_{int}-T_w)}{L_f}.
$$

For Case B:

$$
Pe_D=14.329521,\qquad
Re_D=1.53,\qquad
Ste=0.06275,\qquad
k_{ratio}=1.
$$

The implemented deposit energy equation assumes equal fluid/deposit
volumetric heat capacity. If the verification case later uses
$(\rho C_p)_{dep} \ne (\rho C_p)_f$, a separate `rhoCp_ratio` factor must be
added to the deposit transient term.

The final nondimensional time for the 8 s Fluent comparison is

$$
\tau_{end}=\frac{8U}{L_{total}}=\frac{8}{10.25}=0.780487804878.
$$

For `T_in=5 C`, `T_int=0 C`, and `T_wall=-5 C`:

$$
\Theta_{in}=2,\qquad
\Theta_{solidus}=1,\qquad
\Theta_w=0.
$$

## Network Scaling

The model receives collocation coordinates `(r, x, tau)`. Inside
`network.forward()`, these inputs are scaled only for numerical conditioning:

$$
r_n=\frac{r}{r_w},\qquad
x_n=\frac{x}{L},\qquad
\tau_n=\frac{\tau}{\tau_{end}}.
$$

This is not a second nondimensionalization. Autograd differentiates through the
scaling, so residuals are still derivatives with respect to `(r, x, tau)`.

## Operators

For scalar fields and axial velocity:

$$
\nabla^2_{cyl} f
=
\frac{\partial^2 f}{\partial r^2}
+
\frac{1}{r}\frac{\partial f}{\partial r}
+
\frac{\partial^2 f}{\partial x^2}.
$$

For radial velocity, the viscous vector term is

$$
\nabla^2_{cyl}u_r-\frac{u_r}{r^2}.
$$

The implementation clamps the singular `1/r` factors internally for numerical
stability. Interior PDE sampling excludes the near-axis region, while the axis
boundary condition remains active.

## Region Weighting

The smooth deposit indicator is

$$
H_\epsilon(r,r_{int})
=
\frac{1}{2}\left[1+\tanh\left(\frac{r-r_{int}}{\epsilon}\right)\right].
$$

`H` is approximately 0 in fluid and 1 in deposit. Fluid residuals are
multiplied by `1-H`; deposit residuals are multiplied by `H`.

## Interior Residuals

Continuity:

$$
R_{mass}
=
\frac{\partial u_r}{\partial r}
+
\frac{u_r}{r}
+
\frac{\partial u_x}{\partial x}.
$$

Axial momentum:

$$
R_{mom,x}
=
\frac{1}{A}\frac{\partial u_x}{\partial \tau}
+
u_r\frac{\partial u_x}{\partial r}
+
u_x\frac{\partial u_x}{\partial x}
+
\frac{\partial p}{\partial x}
-
\frac{2}{Re_D}\nabla^2_{cyl}u_x.
$$

Radial momentum:

$$
R_{mom,r}
=
\frac{1}{A}\frac{\partial u_r}{\partial \tau}
+
u_r\frac{\partial u_r}{\partial r}
+
u_x\frac{\partial u_r}{\partial x}
+
\frac{\partial p}{\partial r}
-
\frac{2}{Re_D}
\left(
\nabla^2_{cyl}u_r-\frac{u_r}{r^2}
\right).
$$

Fluid energy:

$$
R_{E,f}
=
\frac{1}{A}\frac{\partial \Theta_f}{\partial \tau}
+
u_r\frac{\partial \Theta_f}{\partial r}
+
u_x\frac{\partial \Theta_f}{\partial x}
-
\frac{2}{Pe_D}\nabla^2_{cyl}\Theta_f.
$$

Deposit energy:

$$
R_{E,dep}
=
\frac{1}{A}\frac{\partial \Theta_{dep}}{\partial \tau}
-
\frac{2k_{ratio}}{Pe_D}\nabla^2_{cyl}\Theta_{dep}.
$$

## Interface Residuals

The learned interface is `r = r_int(x, tau)`.

Stefan condition:

$$
R_{Stefan}
=
\frac{\partial r_{int}}{\partial \tau}
-
\frac{2A}{Pe_D}Ste
\left(
k_{ratio}\left.\frac{\partial \Theta_{dep}}{\partial r}\right|_\Gamma
-
\left.\frac{\partial \Theta_f}{\partial r}\right|_\Gamma
\right).
$$

Interface temperature:

$$
R_{T,f}=\Theta_f(r_{int},x,\tau)-\Theta_{solidus},
\qquad
R_{T,dep}=\Theta_{dep}(r_{int},x,\tau)-\Theta_{solidus}.
$$

Interface no-slip velocity:

$$
R_{u,\Gamma,x}=u_x(r_{int},x,\tau),\qquad
R_{u,\Gamma,r}=u_r(r_{int},x,\tau).
$$

Interface monotonic penalties:

$$
R_{rint,\tau}=\max\left(0,\frac{\partial r_{int}}{\partial \tau}\right),
\qquad
R_{rint,x}=\max\left(0,\frac{\partial r_{int}}{\partial x}\right).
$$

No current loss term penalizes `d2r_int/dx2`.

## Boundary And Initial Conditions

Wall, `r = r_w`:

$$
\Theta_{dep}(r_w,x,\tau)=\Theta_w(x),\qquad
u_r(r_w,x,\tau)=0,\qquad
u_x(r_w,x,\tau)=0.
$$

The wall temperature is

$$
\Theta_w(x)=
\begin{cases}
\Theta_{in}, & 0 \le x \le 0.25,\\
\Theta_{wall}, & x > 0.25.
\end{cases}
$$

Inlet, `x = 0`:

$$
\Theta_f(r,0,\tau)=\Theta_{in},\qquad
u_r(r,0,\tau)=0,
$$

$$
u_x(r,0,\tau)=2\left[1-\left(\frac{r}{r_{int}(0,\tau)}\right)^2\right],
\qquad
r_{int}(0,\tau)=r_w.
$$

Outlet, `x = L`:

$$
\frac{\partial \Theta_f}{\partial x}(r,L,\tau)=0,\qquad
\frac{\partial u_x}{\partial x}(r,L,\tau)=0.
$$

Axis, implemented at `r = 1e-4`:

$$
u_r(0,x,\tau)=0,\qquad
\frac{\partial \Theta_f}{\partial r}(0,x,\tau)=0,\qquad
\frac{\partial u_x}{\partial r}(0,x,\tau)=0.
$$

Initial condition, `tau = 0`:

$$
r_{int}(x,0)=r_w,\qquad
\Theta_f(r,x,0)=\Theta_{in},\qquad
u_r(r,x,0)=0,
$$

$$
u_x(r,x,0)=2(1-r^2).
$$

There is no explicit `Theta_dep` initial-condition residual.

## Loss Assembly

Every residual is converted to MSE. The current weighted loss terms are:

$$
L_{mass}=1\,MSE((1-H)R_{mass}),
$$

$$
L_{mom,x}=1\,MSE((1-H)R_{mom,x}),\qquad
L_{mom,r}=1\,MSE((1-H)R_{mom,r}),
$$

$$
L_{E,f}=1\,MSE((1-H)R_{E,f}),\qquad
L_{E,dep}=1\,MSE(HR_{E,dep}),
$$

$$
L_{Stefan}=10\,MSE(R_{Stefan}),
$$

$$
L_{Tcont}
=100\cdot0.5\left[MSE(R_{T,f})+MSE(R_{T,dep})\right],
$$

$$
L_{interface\_vel}
=10\cdot0.5\left[MSE(R_{u,\Gamma,x})+MSE(R_{u,\Gamma,r})\right],
$$

$$
L_{rint}=10\,MSE(R_{rint,\tau})+10\,MSE(R_{rint,x}).
$$

The BC/IC term averages the 16 implemented BC/IC residual arrays and applies
weight 100:

$$
L_{BC/IC}=100\,\frac{1}{16}\sum_j MSE(R_{BC/IC,j}).
$$

Full physics training sums all terms. Phase 1 uses only `L_BC/IC`.
