# Sharp-Interface PINN Equations

This file records the nondimensional governing equations currently implemented
in the sharp-interface PINN branch for case `20260410_nonDimm_goodmatchCaseC`.

## Nondimensional Variables

The temperature scale is

$$
\Theta = \frac{T - T_w}{T_{int} - T_w}.
$$

For the active verification case,

$$
T_{in}=5^\circ C,\qquad T_{int}=0^\circ C,\qquad T_w=-5^\circ C,
$$

so

$$
\Theta_{in}=2,\qquad \Theta_{solidus}=1,\qquad \Theta_w=0.
$$

The active nondimensional case parameters are

$$
Pe=14.3,\qquad Ste=0.06275,\qquad Re=1.53,\qquad k_{ratio}=1,
$$

$$
r_a=0,\qquad r_w=1,\qquad L=10.25,\qquad \hat{t}_{end}=0.8.
$$

The network predicts

$$
\left(u_r,\ u_x,\ p,\ \Theta_f,\ \Theta_{dep},\ r_{int}(x,t)\right).
$$

## Cylindrical Operators

For any scalar or axial component \(f(r,x,t)\), the implemented cylindrical
Laplacian is

$$
\nabla^2_{cyl} f
=
\frac{\partial^2 f}{\partial r^2}
+
\frac{1}{r}\frac{\partial f}{\partial r}
+
\frac{\partial^2 f}{\partial x^2}.
$$

For radial velocity, the viscous cylindrical vector term is

$$
\nabla^2_{cyl} u_r - \frac{u_r}{r^2}.
$$

The implementation clamps \(r\) internally in the singular \(1/r\) terms for
numerical stability, while interior PDE sampling excludes only the very near
axis region and keeps the axis boundary condition active.

## Smooth Region Weighting

The code uses a smooth Heaviside indicator \(H_\epsilon(r,r_{int})\), where

$$
H_\epsilon \approx 0 \quad \text{in fluid},
\qquad
H_\epsilon \approx 1 \quad \text{in deposit}.
$$

Fluid residuals are multiplied by

$$
1-H_\epsilon,
$$

and deposit residuals are multiplied by

$$
H_\epsilon.
$$

## Fluid Equations

### Continuity

$$
R_{mass}
=
\frac{\partial u_r}{\partial r}
+
\frac{u_r}{r}
+
\frac{\partial u_x}{\partial x}.
$$

The loss uses

$$
(1-H_\epsilon)R_{mass}.
$$

### Axial Momentum

$$
R_{mom,x}
=
\frac{1}{Pe}\frac{\partial u_x}{\partial t}
+
u_r\frac{\partial u_x}{\partial r}
+
u_x\frac{\partial u_x}{\partial x}
+
\frac{\partial p}{\partial x}
-
\frac{1}{Re}\nabla^2_{cyl}u_x.
$$

The loss uses

$$
(1-H_\epsilon)R_{mom,x}.
$$

### Radial Momentum

$$
R_{mom,r}
=
\frac{1}{Pe}\frac{\partial u_r}{\partial t}
+
u_r\frac{\partial u_r}{\partial r}
+
u_x\frac{\partial u_r}{\partial x}
+
\frac{\partial p}{\partial r}
-
\frac{1}{Re}
\left(
\nabla^2_{cyl}u_r-\frac{u_r}{r^2}
\right).
$$

The loss uses

$$
(1-H_\epsilon)R_{mom,r}.
$$

### Fluid Energy

$$
R_{E,f}
=
\frac{\partial \Theta_f}{\partial t}
+
Pe
\left(
u_r\frac{\partial \Theta_f}{\partial r}
+
u_x\frac{\partial \Theta_f}{\partial x}
\right)
-
\nabla^2_{cyl}\Theta_f.
$$

The loss uses

$$
(1-H_\epsilon)R_{E,f}.
$$

## Deposit Equation

### Deposit Energy

$$
R_{E,dep}
=
\frac{\partial \Theta_{dep}}{\partial t}
-
k_{ratio}\nabla^2_{cyl}\Theta_{dep}.
$$

The loss uses

$$
H_\epsilon R_{E,dep}.
$$

For the active case,

$$
k_{ratio}=1.
$$

## Interface Conditions

The learned interface is

$$
r=r_{int}(x,t).
$$

### Stefan Condition

The Stefan residual implemented in code is

$$
R_{Stefan}
=
\frac{\partial r_{int}}{\partial t}
-
Ste
\left(
k_{ratio}
\left.\frac{\partial \Theta_{dep}}{\partial r}\right|_{\Gamma}
-
\left.\frac{\partial \Theta_f}{\partial r}\right|_{\Gamma}
\right).
$$

The target condition is

$$
R_{Stefan}=0.
$$

For the active case,

$$
Ste=0.06275,\qquad k_{ratio}=1.
$$

### Interface Temperature

At the interface,

$$
\Theta_f(r_{int},x,t)=\Theta_{solidus},
$$

$$
\Theta_{dep}(r_{int},x,t)=\Theta_{solidus}.
$$

The implemented residuals are

$$
R_{T,f}
=
\Theta_f(r_{int},x,t)-\Theta_{solidus},
$$

$$
R_{T,dep}
=
\Theta_{dep}(r_{int},x,t)-\Theta_{solidus}.
$$

For the active case,

$$
\Theta_{solidus}=1.
$$

### Interface Velocity

The deposit is treated as stationary and attached to the wall.  The current
sharp-interface loss therefore enforces no slip at the fluid/deposit interface:

$$
R_{u,\Gamma,x}=u_x(r_{int},x,t),
$$

$$
R_{u,\Gamma,r}=u_r(r_{int},x,t).
$$

The target condition is

$$
R_{u,\Gamma,x}=0,\qquad R_{u,\Gamma,r}=0.
$$

## Interface Stabilization Terms

These are auxiliary training penalties in the current sharp-interface loss.
They are not separate conservation laws.

Solidification should move the interface inward, so the model penalizes
positive interface-radius growth:

$$
R_{r,t}
=
\max\left(\frac{\partial r_{int}}{\partial t},0\right).
$$

For the current cold-wall/hot-inlet verification setup, deposit thickness is
expected not to decrease downstream, so the code penalizes downstream increases
of \(r_{int}\):

$$
R_{r,x}
=
\max\left(\frac{\partial r_{int}}{\partial x},0\right).
$$

## Boundary Conditions

### Wall Boundary, \(r=r_w\)

At the wall,

$$
\Theta_{dep}(r_w,x,t)=
\begin{cases}
\Theta_{in}, & 0 \le x \le 0.25,\\
\Theta_w, & x > 0.25,
\end{cases}
$$

$$
u_r(r_w,x,t)=0,
\qquad
u_x(r_w,x,t)=0.
$$

For the active case,

$$
\Theta_{in}=2,\qquad \Theta_w=0.
$$

### Inlet Boundary, \(x=0\)

At the inlet,

$$
\Theta_f(r,0,t)=\Theta_{in},
$$

$$
u_r(r,0,t)=0,
$$

$$
u_x(r,0,t)
=
2\left(1-\frac{r^2}{r_{int}(0,t)^2}\right),
$$

so the nondimensional cross-sectional average inlet velocity is 1 and the
centerline maximum is 2.

$$
r_{int}(0,t)=r_w.
$$

For the active case,

$$
\Theta_{in}=2,\qquad r_w=1.
$$

### Outlet Boundary, \(x=L\)

At the outlet,

$$
\frac{\partial \Theta_f}{\partial x}(r,L,t)=0,
$$

$$
\frac{\partial u_x}{\partial x}(r,L,t)=0.
$$

### Axis Boundary, \(r=0\)

At the axis,

$$
u_r(0,x,t)=0,
$$

$$
\frac{\partial \Theta_f}{\partial r}(0,x,t)=0,
$$

$$
\frac{\partial u_x}{\partial r}(0,x,t)=0.
$$

## Initial Conditions

At \(t=0\), the pipe is initialized as clean and hot:

$$
r_{int}(x,0)=r_w,
$$

$$
\Theta_f(r,x,0)=\Theta_{in},
$$

$$
u_x(r,x,0)
=
2\left(1-\frac{r^2}{r_w^2}\right),
$$

$$
u_r(r,x,0)=0.
$$

For the active case,

$$
r_w=1,\qquad \Theta_{in}=2.
$$

## Composite Loss

Each residual group is converted to a mean-squared error and then weighted.
For the two paired interface constraints, the code averages the two sides:

$$
\mathcal{L}_T
=
\frac{1}{2}
\left(
\operatorname{MSE}(R_{T,f})
+
\operatorname{MSE}(R_{T,dep})
\right),
$$

$$
\mathcal{L}_{u,\Gamma}
=
\frac{1}{2}
\left(
\operatorname{MSE}(R_{u,\Gamma,x})
+
\operatorname{MSE}(R_{u,\Gamma,r})
\right).
$$

The implemented total loss is

$$
\mathcal{L}
=
w_{mass}\mathcal{L}_{mass}
+
w_{mom,x}\mathcal{L}_{mom,x}
+
w_{mom,r}\mathcal{L}_{mom,r}
+
w_{E,f}\mathcal{L}_{E,f}
+
w_{E,dep}\mathcal{L}_{E,dep}
+
w_{Stefan}\mathcal{L}_{Stefan}
+
w_T\mathcal{L}_T
+
w_{u,\Gamma}\mathcal{L}_{u,\Gamma}
+
w_{r,t}\mathcal{L}_{r,t}
+
w_{r,x}\mathcal{L}_{r,x}
+
w_{BC/IC}\mathcal{L}_{BC/IC}.
$$

The active default weights are

$$
w_{mass}=w_{mom,x}=w_{mom,r}=w_{E,f}=w_{E,dep}=1,
$$

$$
w_{Stefan}=10,\qquad
w_T=100,\qquad
w_{u,\Gamma}=10,\qquad
w_{r,t}=10,\qquad
w_{r,x}=10,\qquad
w_{BC/IC}=100.
$$
