"""
Smoke tests for losses.py — run with:  python -m pytest tests/ -v
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
from config import cfg, LossWeights
from network import PINNSolidification
from sampling import build_batch
from losses import (
    mse,
    compute_loss_terms,
    total_loss,
    compute_loss,
    format_loss_line,
    log_to_csv,
    _BC_IC_KEYS,
)

torch.manual_seed(0)


# ─────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def model():
    m = PINNSolidification(cfg.network, cfg.case, seed=0)
    m.eval()
    return m


@pytest.fixture(scope="module")
def batch(model):
    return build_batch(model, cfg, seed=0)


@pytest.fixture(scope="module")
def loss_phase1(model, batch):
    """Cached Phase 1 loss + log."""
    loss, log = compute_loss(model, batch, cfg, physics_on=False)
    return loss, log


@pytest.fixture(scope="module")
def loss_phase2(model):
    """Cached Phase 2 loss + log (fresh batch to get clean requires_grad)."""
    b = build_batch(model, cfg, seed=10)
    loss, log = compute_loss(model, b, cfg, physics_on=True)
    return loss, log


# ─────────────────────────────────────────────────────────────────────────
# 4.1  mse primitive
# ─────────────────────────────────────────────────────────────────────────

def test_mse_zeros():
    r = torch.zeros(50)
    assert mse(r).item() == pytest.approx(0.0)


def test_mse_ones():
    r = torch.ones(50)
    assert mse(r).item() == pytest.approx(1.0)


def test_mse_known_value():
    r = torch.tensor([1.0, -1.0, 2.0, -2.0])
    # mean([1, 1, 4, 4]) = 2.5
    assert mse(r).item() == pytest.approx(2.5)


def test_mse_is_scalar():
    r = torch.randn(100)
    assert mse(r).shape == ()


# ─────────────────────────────────────────────────────────────────────────
# 4.1  compute_loss_terms
# ─────────────────────────────────────────────────────────────────────────

def test_loss_terms_keys_physics_on(model, batch):
    from equations import compute_all_residuals
    residuals = compute_all_residuals(model, batch, cfg)
    terms = compute_loss_terms(residuals, cfg.weights, physics_on=True)
    expected = {"mass", "mom_x", "mom_r", "energy_fluid", "energy_dep",
                "stefan", "T_cont", "bc_ic"}
    assert set(terms.keys()) == expected


def test_loss_terms_keys_physics_off(model, batch):
    from losses import _compute_bc_ic_residuals
    residuals = _compute_bc_ic_residuals(model, batch, cfg)
    terms = compute_loss_terms(residuals, cfg.weights, physics_on=False)
    expected = {"mass", "mom_x", "mom_r", "energy_fluid", "energy_dep",
                "stefan", "T_cont", "bc_ic"}
    assert set(terms.keys()) == expected


def test_physics_off_placeholder_zeros(model, batch):
    from losses import _compute_bc_ic_residuals
    residuals = _compute_bc_ic_residuals(model, batch, cfg)
    terms = compute_loss_terms(residuals, cfg.weights, physics_on=False)
    for k in ("mass", "mom_x", "mom_r", "energy_fluid", "energy_dep",
              "stefan", "T_cont"):
        assert terms[k].item() == pytest.approx(0.0), \
            f"{k} should be zero placeholder when physics_on=False"


def test_loss_terms_all_non_negative(model, batch):
    from equations import compute_all_residuals
    residuals = compute_all_residuals(model, batch, cfg)
    terms = compute_loss_terms(residuals, cfg.weights, physics_on=True)
    for k, v in terms.items():
        assert v.item() >= 0.0, f"{k} loss is negative"


def test_loss_terms_no_nan(model, batch):
    from equations import compute_all_residuals
    residuals = compute_all_residuals(model, batch, cfg)
    terms = compute_loss_terms(residuals, cfg.weights, physics_on=True)
    for k, v in terms.items():
        assert not torch.isnan(v), f"NaN in {k}"
        assert not torch.isinf(v), f"Inf in {k}"


def test_stefan_weight_scales_term(model, batch):
    """Doubling stefan weight should double the stefan term exactly."""
    from equations import compute_all_residuals
    residuals = compute_all_residuals(model, batch, cfg)

    w1 = LossWeights(stefan=10.0)
    w2 = LossWeights(stefan=20.0)
    t1 = compute_loss_terms(residuals, w1, physics_on=True)
    t2 = compute_loss_terms(residuals, w2, physics_on=True)

    assert t2["stefan"].item() == pytest.approx(2.0 * t1["stefan"].item(),
                                                  rel=1e-5)


def test_bc_ic_weight_scales_term(model, batch):
    """Doubling bc_ic weight should double the bc_ic term exactly."""
    from equations import compute_all_residuals
    residuals = compute_all_residuals(model, batch, cfg)

    w1 = LossWeights(bc_ic=100.0)
    w2 = LossWeights(bc_ic=200.0)
    t1 = compute_loss_terms(residuals, w1, physics_on=True)
    t2 = compute_loss_terms(residuals, w2, physics_on=True)

    assert t2["bc_ic"].item() == pytest.approx(2.0 * t1["bc_ic"].item(),
                                                 rel=1e-5)


# ─────────────────────────────────────────────────────────────────────────
# 4.2  total_loss
# ─────────────────────────────────────────────────────────────────────────

def test_total_loss_physics_off_equals_bc_ic(model, batch):
    from losses import _compute_bc_ic_residuals
    residuals = _compute_bc_ic_residuals(model, batch, cfg)
    terms = compute_loss_terms(residuals, cfg.weights, physics_on=False)
    tot = total_loss(terms, physics_on=False)
    assert tot.item() == pytest.approx(terms["bc_ic"].item(), rel=1e-6)


def test_total_loss_physics_on_geq_bc_ic(model, batch):
    """Total loss with all physics should be >= bc_ic alone."""
    from equations import compute_all_residuals
    residuals = compute_all_residuals(model, batch, cfg)
    terms = compute_loss_terms(residuals, cfg.weights, physics_on=True)
    tot = total_loss(terms, physics_on=True)
    assert tot.item() >= terms["bc_ic"].item() - 1e-8


def test_total_loss_is_scalar(loss_phase2):
    loss, _ = loss_phase2
    assert loss.shape == ()


# ─────────────────────────────────────────────────────────────────────────
# 4.4  compute_loss — backward-ability and gradient flow
# ─────────────────────────────────────────────────────────────────────────

def test_phase1_loss_requires_grad(loss_phase1):
    loss, _ = loss_phase1
    assert loss.requires_grad, "Phase 1 loss must be differentiable"


def test_phase2_loss_requires_grad(loss_phase2):
    loss, _ = loss_phase2
    assert loss.requires_grad, "Phase 2 loss must be differentiable"


def test_phase1_backward_no_error(model):
    """Phase 1 backward() must not raise."""
    b = build_batch(model, cfg, seed=20)
    loss, _ = compute_loss(model, b, cfg, physics_on=False)
    loss.backward()   # should not raise


def test_phase2_backward_gradients_flow_to_params(model):
    """After Phase 2 backward(), every parameter must have a gradient."""
    for p in model.parameters():
        p.grad = None
    b = build_batch(model, cfg, seed=30)
    loss, _ = compute_loss(model, b, cfg, physics_on=True)
    loss.backward()
    missing = [name for name, p in model.named_parameters()
               if p.requires_grad and p.grad is None]
    assert len(missing) == 0, \
        f"Parameters with no gradient after backward: {missing}"


def test_compute_loss_log_keys(loss_phase2):
    _, log = loss_phase2
    expected = {"mass", "mom_x", "mom_r", "energy_fluid", "energy_dep",
                "stefan", "T_cont", "bc_ic", "total"}
    assert set(log.keys()) == expected


def test_compute_loss_log_values_are_float(loss_phase2):
    _, log = loss_phase2
    for k, v in log.items():
        assert isinstance(v, float), f"{k} log value is not float"


def test_compute_loss_log_total_consistent(loss_phase2):
    """log['total'] must equal sum of the 8 individual terms."""
    _, log = loss_phase2
    terms_sum = sum(v for k, v in log.items() if k != "total")
    assert log["total"] == pytest.approx(terms_sum, rel=1e-4)


def test_phase1_loss_less_than_phase2(model):
    """Phase 1 loss (bc_ic only) should be strictly less than Phase 2 total."""
    b = build_batch(model, cfg, seed=40)
    loss1, log1 = compute_loss(model, b, cfg, physics_on=False)
    b2 = build_batch(model, cfg, seed=40)   # same seed → same points
    loss2, log2 = compute_loss(model, b2, cfg, physics_on=True)
    # Phase 2 adds physics terms — total must be >= Phase 1
    assert log2["total"] >= log1["total"] - 1e-6


# ─────────────────────────────────────────────────────────────────────────
# 4.5  format_loss_line
# ─────────────────────────────────────────────────────────────────────────

def test_format_loss_line_contains_iter(loss_phase2):
    _, log = loss_phase2
    line = format_loss_line(log, 1234, physics_on=True)
    assert "1234" in line


def test_format_loss_line_phase1_tag(loss_phase1):
    _, log = loss_phase1
    line = format_loss_line(log, 0, physics_on=False)
    assert "BC/IC only" in line


def test_format_loss_line_is_string(loss_phase2):
    _, log = loss_phase2
    assert isinstance(format_loss_line(log, 0, physics_on=True), str)


# ─────────────────────────────────────────────────────────────────────────
# 4.5  log_to_csv
# ─────────────────────────────────────────────────────────────────────────

def test_log_to_csv_creates_file(tmp_path, loss_phase2):
    _, log = loss_phase2
    csv_path = tmp_path / "loss_history.csv"
    log_to_csv(log, iteration=0, csv_path=csv_path, write_header=True)
    assert csv_path.exists()


def test_log_to_csv_row_count(tmp_path, loss_phase2):
    _, log = loss_phase2
    csv_path = tmp_path / "loss_hist2.csv"
    log_to_csv(log, 0, csv_path, write_header=True)
    for i in range(1, 4):
        log_to_csv(log, i, csv_path, write_header=False)
    lines = csv_path.read_text().strip().splitlines()
    assert len(lines) == 5, f"Expected header + 4 rows, got {len(lines)}"
