"""
Smoke tests for sampling.py — run with:  python -m pytest tests/ -v
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
from config import cfg
from network import PINNSolidification
from sampling import (
    _lhs,
    sample_interior, sample_wall, sample_inlet,
    sample_outlet, sample_axis, sample_ic,
    sample_interface, resample_interface,
    build_batch, _AXIS_R,
)

torch.manual_seed(0)
N = 64


# ─────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def model():
    m = PINNSolidification(cfg.network, cfg.case, seed=0)
    m.eval()
    return m


# ─────────────────────────────────────────────────────────────────────────
# 3.1  LHS utility
# ─────────────────────────────────────────────────────────────────────────

def test_lhs_shape():
    s = _lhs(N, 3, seed=0)
    assert s.shape == (N, 3)


def test_lhs_range():
    """All LHS samples must lie in (0, 1)."""
    s = _lhs(200, 4, seed=7)
    assert s.min().item() >= 0.0
    assert s.max().item() <= 1.0


def test_lhs_stratification_1d():
    """Each stratum [k/N, (k+1)/N) must contain exactly one sample."""
    N_test = 50
    s = _lhs(N_test, 1, seed=42)[:, 0]
    strata = (s * N_test).long()
    # All stratum indices 0…N-1 should appear exactly once
    counts = torch.bincount(strata, minlength=N_test)
    assert (counts == 1).all(), "LHS stratification violated"


def test_lhs_different_seeds_differ():
    s1 = _lhs(N, 2, seed=0)
    s2 = _lhs(N, 2, seed=1)
    assert not torch.allclose(s1, s2)


def test_lhs_same_seed_reproducible():
    s1 = _lhs(N, 2, seed=99)
    s2 = _lhs(N, 2, seed=99)
    torch.testing.assert_close(s1, s2)


# ─────────────────────────────────────────────────────────────────────────
# 3.1  sample_interior
# ─────────────────────────────────────────────────────────────────────────

def test_interior_shape():
    pts = sample_interior(N, cfg)
    for k in ("r", "x", "t"):
        assert pts[k].shape == (N,)


def test_interior_requires_grad():
    pts = sample_interior(N, cfg)
    for k in ("r", "x", "t"):
        assert pts[k].requires_grad, f"{k} missing requires_grad"


def test_interior_bounds():
    pts = sample_interior(500, cfg, seed=3)
    assert pts["r"].min() >= 0.0 and pts["r"].max() <= cfg.case.r_w
    assert pts["x"].min() >= 0.0 and pts["x"].max() <= cfg.case.L
    assert pts["t"].min() >= 0.0 and pts["t"].max() <= cfg.case.t_end


# ─────────────────────────────────────────────────────────────────────────
# 3.2  Boundary samplers
# ─────────────────────────────────────────────────────────────────────────

def test_wall_r_fixed():
    pts = sample_wall(N, cfg)
    torch.testing.assert_close(
        pts["r"].detach(), torch.full((N,), cfg.case.r_w))


def test_inlet_x_zero():
    pts = sample_inlet(N, cfg)
    torch.testing.assert_close(pts["x"].detach(), torch.zeros(N))


def test_outlet_x_L():
    pts = sample_outlet(N, cfg)
    torch.testing.assert_close(
        pts["x"].detach(), torch.full((N,), cfg.case.L))


def test_axis_r_small():
    pts = sample_axis(N, cfg)
    torch.testing.assert_close(
        pts["r"].detach(), torch.full((N,), _AXIS_R))


def test_boundary_requires_grad():
    for fn in (sample_wall, sample_inlet, sample_outlet, sample_axis):
        pts = fn(N, cfg)
        for k in ("r", "x", "t"):
            assert pts[k].requires_grad, f"{fn.__name__}: {k} missing requires_grad"


def test_boundary_t_in_range():
    for fn in (sample_wall, sample_inlet, sample_outlet, sample_axis):
        pts = fn(200, cfg, seed=5)
        assert pts["t"].min() >= 0.0 and pts["t"].max() <= cfg.case.t_end


# ─────────────────────────────────────────────────────────────────────────
# 3.3  IC sampler
# ─────────────────────────────────────────────────────────────────────────

def test_ic_t_zero():
    pts = sample_ic(N, cfg)
    torch.testing.assert_close(pts["t"].detach(), torch.zeros(N))


def test_ic_bounds():
    pts = sample_ic(200, cfg, seed=2)
    assert pts["r"].min() >= 0.0 and pts["r"].max() <= cfg.case.r_w
    assert pts["x"].min() >= 0.0 and pts["x"].max() <= cfg.case.L


def test_ic_requires_grad():
    pts = sample_ic(N, cfg)
    for k in ("r", "x", "t"):
        assert pts[k].requires_grad


# ─────────────────────────────────────────────────────────────────────────
# 3.4  Adaptive interface sampler
# ─────────────────────────────────────────────────────────────────────────

def test_interface_shape(model):
    pts = sample_interface(model, N, cfg)
    for k in ("r", "x", "t"):
        assert pts[k].shape == (N,)


def test_interface_r_within_pipe(model):
    """Interface r must be within [0, r_w] at all times."""
    pts = sample_interface(model, 200, cfg, seed=10)
    assert pts["r"].detach().min() >= 0.0
    assert pts["r"].detach().max() <= cfg.case.r_w + 1e-5


def test_interface_r_equals_network_rint(model):
    """r coords from sampler should match a fresh network r_int prediction."""
    pts = sample_interface(model, N, cfg, seed=7)
    x = pts["x"].detach()
    t = pts["t"].detach()
    with torch.no_grad():
        out = model(torch.zeros(N), x, t)
        r_int_fresh = out["r_int"].detach()
    torch.testing.assert_close(
        pts["r"].detach(), r_int_fresh,
        atol=1e-5, rtol=0,
        msg="Interface sampler r should match network r_int prediction"
    )


def test_interface_requires_grad(model):
    pts = sample_interface(model, N, cfg)
    for k in ("r", "x", "t"):
        assert pts[k].requires_grad


def test_resample_changes_with_seed(model):
    """Different seeds must produce different point sets."""
    p1 = resample_interface(model, N, cfg, seed=0)
    p2 = resample_interface(model, N, cfg, seed=1)
    assert not torch.allclose(p1["x"].detach(), p2["x"].detach())


# ─────────────────────────────────────────────────────────────────────────
# 3.5  build_batch
# ─────────────────────────────────────────────────────────────────────────

def test_build_batch_keys(model):
    expected = {"interior", "interface", "wall", "inlet", "outlet", "axis", "ic"}
    batch = build_batch(model, cfg, seed=0)
    assert set(batch.keys()) == expected


def test_build_batch_all_requires_grad(model):
    batch = build_batch(model, cfg, seed=0)
    for key, pts in batch.items():
        for coord in ("r", "x", "t"):
            assert pts[coord].requires_grad, \
                f"batch['{key}']['{coord}'] missing requires_grad"


def test_build_batch_sizes(model):
    batch = build_batch(model, cfg, seed=0)
    sc = cfg.sampling
    assert len(batch["interior"]["r"])  == sc.N_interior
    assert len(batch["interface"]["r"]) == sc.N_interface
    assert len(batch["ic"]["r"])        == sc.N_ic
    for key in ("wall", "inlet", "outlet", "axis"):
        assert len(batch[key]["r"]) == sc.N_boundary, \
            f"{key}: expected {sc.N_boundary}, got {len(batch[key]['r'])}"


def test_build_batch_reproducible(model):
    b1 = build_batch(model, cfg, seed=42)
    b2 = build_batch(model, cfg, seed=42)
    torch.testing.assert_close(
        b1["interior"]["r"].detach(),
        b2["interior"]["r"].detach(),
    )


# ─────────────────────────────────────────────────────────────────────────
# 3.6  plot_batch (non-crashing check)
# ─────────────────────────────────────────────────────────────────────────

def test_plot_batch_runs_without_error(model, tmp_path):
    """plot_batch should produce a figure and save without raising."""
    import matplotlib.pyplot as plt
    batch = build_batch(model, cfg, seed=0)
    from sampling import plot_batch
    fig = plot_batch(batch, cfg, save_path=tmp_path / "test_scatter.png")
    assert (tmp_path / "test_scatter.png").exists()
    plt.close(fig)
