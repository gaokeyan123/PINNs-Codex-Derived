"""Smoke tests for network.py — run with: python -m pytest tests/"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import torch
import pytest
from config import cfg
from network import PINNSolidification, count_parameters


@pytest.fixture
def model():
    torch.manual_seed(0)
    return PINNSolidification(cfg.network, cfg.case, seed=0)


def make_inputs(N=64):
    r = torch.rand(N, requires_grad=True)
    x = (torch.rand(N) * cfg.case.L).requires_grad_(True)
    tau = (torch.rand(N) * cfg.case.tau_end).requires_grad_(True)
    return r, x, tau


def test_output_shapes(model):
    r, x, tau = make_inputs()
    out = model(r, x, tau)
    for key in ("u_r", "u_x", "p", "Theta_f", "Theta_dep", "r_int"):
        assert out[key].shape == (64,), f"{key} shape mismatch"


def test_temperature_bounds(model):
    r, x, tau = make_inputs(N=1000)
    out = model(r, x, tau)
    theta_max = max(cfg.case.Theta_in, cfg.case.Theta_solidus, cfg.case.Theta_wall, 1.0)
    assert out["Theta_f"].min() >= 0.0 and out["Theta_f"].max() <= theta_max
    assert out["Theta_dep"].min() >= 0.0 and out["Theta_dep"].max() <= theta_max


def test_r_int_bounds(model):
    r, x, tau = make_inputs(N=1000)
    out = model(r, x, tau)
    assert out["r_int"].min() >= 0.0 and out["r_int"].max() <= cfg.case.r_w + 1e-6


def test_r_int_independent_of_r(model):
    """r_int must not depend on r; interface head is masked to (x, tau) only."""
    r, x, tau = make_inputs(N=16)
    out = model(r, x, tau)
    out["r_int"].sum().backward()
    assert r.grad is None or r.grad.abs().max().item() < 1e-10


def test_parameter_count(model):
    n = count_parameters(model)
    print(f"\nTotal parameters: {n:,}")
    assert 50_000 < n < 500_000, f"Unexpected parameter count: {n}"


def test_smooth_heaviside(model):
    r = torch.tensor([0.3, 0.5, 0.7, 0.9])
    r_int = torch.tensor([0.6, 0.6, 0.6, 0.6])
    H = model.smooth_heaviside(r, r_int, epsilon=0.02)
    assert H[0].item() < 0.01
    assert H[3].item() > 0.99
