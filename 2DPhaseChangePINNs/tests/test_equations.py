"""
Smoke tests for equations.py — run with:  python -m pytest tests/ -v
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
from config import cfg
from network import PINNSolidification
from equations import (
    _grad, _laplacian_cyl, _poiseuille,
    res_mass, res_mom_x, res_mom_r,
    res_energy_fluid, res_energy_dep,
    res_stefan, res_T_continuity,
    res_interface_velocity_bc,
    res_rint_monotonic, res_rint_x_monotonic,
    res_wall_bc, res_inlet_bc, res_outlet_bc, res_axis_bc, res_ic,
    compute_all_residuals,
)

torch.manual_seed(0)

N_SMALL = 32






@pytest.fixture(scope="module")
def model():
    m = PINNSolidification(cfg.network, cfg.case, seed=0)
    m.eval()
    return m


def _pts(N=N_SMALL, r_val=None, x_val=None, tau_val=None):
    """Helper: make a point batch with requires_grad=True."""
    r = (torch.full((N,), r_val, dtype=torch.float32) if r_val is not None
         else torch.rand(N)).requires_grad_(True)
    x = (torch.full((N,), x_val, dtype=torch.float32) if x_val is not None
         else (torch.rand(N) * cfg.case.L)).requires_grad_(True)
    tau = (torch.full((N,), tau_val, dtype=torch.float32) if tau_val is not None
         else (torch.rand(N) * cfg.case.tau_end)).requires_grad_(True)
    return {"r": r, "x": x, "tau": tau}


def _full_batch():
    return {
        "interior":  _pts(N_SMALL),
        "interface": _pts(N_SMALL),
        "wall":      _pts(N_SMALL, r_val=cfg.case.r_w),
        "inlet":     _pts(N_SMALL, x_val=0.0),
        "outlet":    _pts(N_SMALL, x_val=cfg.case.L),
        "axis":      _pts(N_SMALL, r_val=0.01),
        "ic":        _pts(N_SMALL, tau_val=0.0),
    }


def _fwd_with_H(model, pts):
    """Forward pass + Heaviside, returns (out, r, x, tau)."""
    r, x, tau = pts["r"], pts["x"], pts["tau"]
    out = model(r, x, tau)
    out["H"] = model.smooth_heaviside(r, out["r_int"].detach(),
                                       cfg.network.interface_epsilon)
    return out, r, x, tau






def test_grad_linear():
    """_grad on f = 3r should return 3 everywhere."""
    r = torch.rand(20).requires_grad_(True)
    f = 3.0 * r
    df_dr = _grad(f, r)
    assert df_dr is not None
    torch.testing.assert_close(df_dr, torch.full_like(r, 3.0))


def test_grad_quadratic():
    """_grad on f = r² should return 2r."""
    r = torch.rand(20).requires_grad_(True)
    f = r ** 2
    df_dr = _grad(f, r)
    torch.testing.assert_close(df_dr, 2.0 * r, atol=1e-5, rtol=1e-4)


def test_laplacian_cyl_constant():
    """∇²_cyl(c) = 0 for any constant c."""
    r = (torch.rand(20) * 0.9 + 0.05).requires_grad_(True)
    x = torch.rand(20).requires_grad_(True)
    f = torch.ones_like(r) * 5.0

    f = f + 0.0 * r + 0.0 * x
    lap = _laplacian_cyl(f, r, x)
    torch.testing.assert_close(lap, torch.zeros_like(lap), atol=1e-5, rtol=0)


def test_laplacian_cyl_r_squared():
    """∇²_cyl(r²) = 4  (∂²r²/∂r² + (1/r)∂r²/∂r = 2 + 2 = 4)."""
    r = (torch.rand(50) * 0.8 + 0.1).requires_grad_(True)
    x = (torch.rand(50)).requires_grad_(True)
    f = r ** 2 + 0.0 * x
    lap = _laplacian_cyl(f, r, x)
    torch.testing.assert_close(lap, torch.full_like(lap, 4.0), atol=1e-4, rtol=1e-4)


def test_poiseuille_mean_velocity():
    """Mean of Poiseuille profile over [0, r_int] should be 1."""
    r_int = torch.tensor(0.8)
    r = torch.linspace(0.0, 0.8, 1000)
    u = _poiseuille(r, r_int.expand_as(r))

    dr = r[1] - r[0]
    mean_u = 2.0 / (r_int ** 2) * (u * r * dr).sum()
    assert abs(mean_u.item() - 1.0) < 0.01, f"Poiseuille mean = {mean_u:.4f}, expected 1.0"


def test_poiseuille_zero_at_wall():
    """Poiseuille velocity = 0 at r = r_int (no-slip at wall)."""
    r_int = torch.tensor([0.7])
    u_wall = _poiseuille(r_int, r_int)
    torch.testing.assert_close(u_wall, torch.zeros_like(u_wall), atol=1e-6, rtol=0)






def test_res_mass_shape_no_nan(model):
    out, r, x, tau = _fwd_with_H(model, _pts())
    R = res_mass(out, r, x, tau)
    assert R.shape == (N_SMALL,)
    assert not torch.isnan(R).any(), "NaN in res_mass"
    assert not torch.isinf(R).any(), "Inf in res_mass"


def test_res_mom_x_shape_no_nan(model):
    out, r, x, tau = _fwd_with_H(model, _pts())
    R = res_mom_x(out, r, x, tau, cfg.case)
    assert R.shape == (N_SMALL,)
    assert not torch.isnan(R).any()


def test_res_mom_r_shape_no_nan(model):
    pts = _pts()
    pts["r"] = (torch.rand(N_SMALL) * 0.8 + 0.05).requires_grad_(True)
    out, r, x, tau = _fwd_with_H(model, pts)
    R = res_mom_r(out, r, x, tau, cfg.case)
    assert R.shape == (N_SMALL,)
    assert not torch.isnan(R).any()


def test_res_energy_fluid_shape_no_nan(model):
    out, r, x, tau = _fwd_with_H(model, _pts())
    R = res_energy_fluid(out, r, x, tau, cfg.case)
    assert R.shape == (N_SMALL,)
    assert not torch.isnan(R).any()


def test_res_energy_dep_shape_no_nan(model):
    out, r, x, tau = _fwd_with_H(model, _pts())
    R = res_energy_dep(out, r, x, tau, cfg.case)
    assert R.shape == (N_SMALL,)
    assert not torch.isnan(R).any()


def _analytic_points():
    r = torch.linspace(0.2, 0.8, N_SMALL).requires_grad_(True)
    x = torch.linspace(0.1, 0.9, N_SMALL).requires_grad_(True)
    tau = torch.linspace(0.05, 0.75, N_SMALL).requires_grad_(True)
    return r, x, tau


def _quadratic_field(r, x, tau):
    return 3.0 * tau + 5.0 * r + 7.0 * x + 11.0 * r ** 2 + 13.0 * x ** 2


def _quadratic_laplacian(r):
    return 22.0 + (5.0 + 22.0 * r) / r + 26.0


def test_momentum_coefficients_use_excel_nondim():
    r, x, tau = _analytic_points()
    u_x = _quadratic_field(r, x, tau)
    u_r = 0.25 + 2.0 * r + 3.0 * x
    p = 17.0 * x + 19.0 * r
    out = {
        "u_r": u_r,
        "u_x": u_x,
        "p": p,
        "H": torch.zeros_like(r),
    }

    R = res_mom_x(out, r, x, tau, cfg.case)
    expected = (
        (1.0 / cfg.case.L) * 3.0
        + u_r * (5.0 + 22.0 * r)
        + u_x * (7.0 + 26.0 * x)
        + 17.0
        - (2.0 / cfg.case.Re_D) * _quadratic_laplacian(r)
    )
    torch.testing.assert_close(R, expected, atol=1e-4, rtol=1e-4)


def test_radial_momentum_coefficients_use_excel_nondim():
    r, x, tau = _analytic_points()
    u_r = _quadratic_field(r, x, tau)
    u_x = 0.25 + 2.0 * r + 3.0 * x
    p = 17.0 * r + 19.0 * x
    out = {
        "u_r": u_r,
        "u_x": u_x,
        "p": p,
        "H": torch.zeros_like(r),
    }

    R = res_mom_r(out, r, x, tau, cfg.case)
    expected = (
        (1.0 / cfg.case.L) * 3.0
        + u_r * (5.0 + 22.0 * r)
        + u_x * (7.0 + 26.0 * x)
        + 17.0
        - (2.0 / cfg.case.Re_D) * (_quadratic_laplacian(r) - u_r / r ** 2)
    )
    torch.testing.assert_close(R, expected, atol=1e-4, rtol=1e-4)


def test_energy_coefficients_use_excel_nondim():
    r, x, tau = _analytic_points()
    theta = _quadratic_field(r, x, tau)
    u_r = 0.25 + 2.0 * r
    u_x = 0.5 + 3.0 * x
    out = {
        "u_r": u_r,
        "u_x": u_x,
        "Theta_f": theta,
        "Theta_dep": theta,
        "H": torch.zeros_like(r),
    }

    R = res_energy_fluid(out, r, x, tau, cfg.case)
    expected = (
        (1.0 / cfg.case.L) * 3.0
        + u_r * (5.0 + 22.0 * r)
        + u_x * (7.0 + 26.0 * x)
        - (2.0 / cfg.case.Pe_D) * _quadratic_laplacian(r)
    )
    torch.testing.assert_close(R, expected, atol=1e-4, rtol=1e-4)


def test_deposit_energy_coefficients_use_excel_nondim():
    r, x, tau = _analytic_points()
    theta = _quadratic_field(r, x, tau)
    out = {
        "Theta_dep": theta,
        "H": torch.ones_like(r),
    }

    R = res_energy_dep(out, r, x, tau, cfg.case)
    expected = (
        (1.0 / cfg.case.L) * 3.0
        - (2.0 * cfg.case.k_ratio / cfg.case.Pe_D) * _quadratic_laplacian(r)
    )
    torch.testing.assert_close(R, expected, atol=1e-4, rtol=1e-4)


def test_stefan_coefficient_uses_excel_nondim():
    r, x, tau = _analytic_points()
    out = {
        "r_int": 0.8 + 3.0 * tau + 5.0 * x,
        "Theta_f": 2.0 * r + 0.0 * x + 0.0 * tau,
        "Theta_dep": 7.0 * r + 0.0 * x + 0.0 * tau,
    }

    R = res_stefan(out, r, x, tau, cfg.case)
    expected = 3.0 - (2.0 * cfg.case.L / cfg.case.Pe_D) * cfg.case.Ste * (
        cfg.case.k_ratio * 7.0 - 2.0
    )
    torch.testing.assert_close(R, torch.full_like(r, expected), atol=1e-6, rtol=1e-6)






def test_fluid_residual_zero_deep_in_deposit(model):
    """res_mass weighted by (1-H) should be ≈ 0 when r >> r_int (H ≈ 1)."""

    pts = _pts(N_SMALL, r_val=cfg.case.r_w)
    out, r, x, tau = _fwd_with_H(model, pts)

    out["H"] = torch.ones(N_SMALL)
    R = res_mass(out, r, x, tau)
    torch.testing.assert_close(R, torch.zeros_like(R), atol=1e-6, rtol=0)


def test_deposit_residual_zero_deep_in_fluid(model):
    """res_energy_dep weighted by H should be ≈ 0 when r << r_int (H ≈ 0)."""
    pts = _pts(N_SMALL)
    out, r, x, tau = _fwd_with_H(model, pts)
    out["H"] = torch.zeros(N_SMALL)
    R = res_energy_dep(out, r, x, tau, cfg.case)
    torch.testing.assert_close(R, torch.zeros_like(R), atol=1e-6, rtol=0)






def test_stefan_shape_no_nan(model):
    out, r, x, tau = _fwd_with_H(model, _pts())
    R = res_stefan(out, r, x, tau, cfg.case)
    assert R.shape == (N_SMALL,)
    assert not torch.isnan(R).any()


def test_T_continuity_shape_no_nan(model):
    out, r, x, tau = _fwd_with_H(model, _pts())
    R7a, R7b = res_T_continuity(out, cfg.case)
    assert R7a.shape == (N_SMALL,) and R7b.shape == (N_SMALL,)
    assert not torch.isnan(R7a).any() and not torch.isnan(R7b).any()


def test_interface_velocity_bc_shape_no_nan(model):
    out, r, x, tau = _fwd_with_H(model, _pts())
    R_ux, R_ur = res_interface_velocity_bc(out)
    assert R_ux.shape == (N_SMALL,) and R_ur.shape == (N_SMALL,)
    assert not torch.isnan(R_ux).any() and not torch.isnan(R_ur).any()


def test_rint_regularizers_shape_no_nan(model):
    out, r, x, tau = _fwd_with_H(model, _pts())
    R_mono = res_rint_monotonic(out, tau)
    R_x_mono = res_rint_x_monotonic(out, x)
    assert R_mono.shape == (N_SMALL,)
    assert R_x_mono.shape == (N_SMALL,)
    assert not torch.isnan(R_mono).any()
    assert not torch.isnan(R_x_mono).any()






def test_wall_bc_Theta_dep_at_cold_wall(model):
    """If network predicts cold-wall temperature downstream, R_wall_T should be 0."""
    pts = _pts(N_SMALL, r_val=cfg.case.r_w, x_val=cfg.case.hot_wall_length + 1.0)
    out, r, x, tau = _fwd_with_H(model, pts)
    out["Theta_dep"] = torch.full((N_SMALL,), cfg.case.Theta_wall)
    R_T, _, _ = res_wall_bc(out, x, cfg.case)
    torch.testing.assert_close(R_T, torch.zeros_like(R_T), atol=1e-6, rtol=0)


def test_wall_bc_Theta_dep_at_hot_upstream_wall(model):
    """If network predicts inlet temperature on the hot upstream wall, R_wall_T should be 0."""
    pts = _pts(N_SMALL, r_val=cfg.case.r_w, x_val=0.5 * cfg.case.hot_wall_length)
    out, r, x, tau = _fwd_with_H(model, pts)
    out["Theta_dep"] = torch.full((N_SMALL,), cfg.case.Theta_in)
    R_T, _, _ = res_wall_bc(out, x, cfg.case)
    torch.testing.assert_close(R_T, torch.zeros_like(R_T), atol=1e-6, rtol=0)


def test_inlet_bc_T_residual(model):
    pts = _pts(N_SMALL, x_val=0.0)
    out, r, x, tau = _fwd_with_H(model, pts)
    R_T, _, _ = res_inlet_bc(out, r, cfg.case)
    assert R_T.shape == (N_SMALL,)
    assert not torch.isnan(R_T).any()


def test_outlet_bc_shape_no_nan(model):
    pts = _pts(N_SMALL, x_val=cfg.case.L)
    out, r, x, tau = _fwd_with_H(model, pts)
    dTf, dux = res_outlet_bc(out, x)
    assert dTf.shape == (N_SMALL,) and dux.shape == (N_SMALL,)
    assert not torch.isnan(dTf).any()


def test_axis_bc_shape_no_nan(model):
    pts = _pts(N_SMALL, r_val=0.01)
    out, r, x, tau = _fwd_with_H(model, pts)
    R_ur, R_Tf, R_ux = res_axis_bc(out, r)
    assert R_ur.shape == (N_SMALL,)
    assert not torch.isnan(R_ur).any()


def test_ic_rint_residual(model):
    """At tau=0 the r_int residual should be r_int_predicted - r_w."""
    pts = _pts(N_SMALL, tau_val=0.0)
    out, r, x, tau = _fwd_with_H(model, pts)

    out["r_int"] = torch.full((N_SMALL,), cfg.case.r_int_ic)
    R_rint, _, _, _ = res_ic(out, r, cfg.case)
    torch.testing.assert_close(R_rint, torch.zeros_like(R_rint), atol=1e-6, rtol=0)






def test_compute_all_residuals_keys(model):
    """All expected residual keys are present in the output dict."""
    expected = {
        "mass", "mom_x", "mom_r", "energy_fluid", "energy_dep",
        "stefan", "T_cont_fluid", "T_cont_dep",
        "interface_u_x", "interface_u_r",
        "rint_mono", "rint_x_mono",
        "bc_wall_T", "bc_wall_ur", "bc_wall_ux",
        "bc_inlet_T", "bc_inlet_ux", "bc_inlet_ur", "bc_inlet_rint",
        "bc_outlet_Tf", "bc_outlet_ux",
        "bc_axis_ur", "bc_axis_Tf", "bc_axis_ux",
        "ic_rint", "ic_Tf", "ic_ux", "ic_ur",
    }
    res = compute_all_residuals(model, _full_batch(), cfg)
    assert expected == set(res.keys()),\
        f"Missing keys: {expected - set(res.keys())}"


def test_compute_all_residuals_no_nan(model):
    """No residual should contain NaN or Inf."""
    res = compute_all_residuals(model, _full_batch(), cfg)
    for name, R in res.items():
        assert not torch.isnan(R).any(), f"NaN in {name}"
        assert not torch.isinf(R).any(), f"Inf in {name}"


def test_compute_all_residuals_shapes(model):
    """Every residual tensor should have shape [N_SMALL]."""
    res = compute_all_residuals(model, _full_batch(), cfg)
    for name, R in res.items():
        assert R.shape == (N_SMALL,), f"{name}: expected ({N_SMALL},), got {R.shape}"
