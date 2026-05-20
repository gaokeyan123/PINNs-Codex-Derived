"""Tests for training checkpoint save policy."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from network import PINNSolidification
from train import (
    cleanup_previous_best_checkpoints,
    clear_last_checkpoint,
    final_training_iteration,
    save_last_checkpoint,
)


def small_config(tmp_path: Path) -> Config:
    run_cfg = Config()
    run_cfg.output_dir = tmp_path
    run_cfg.training.checkpoint_dir = tmp_path
    run_cfg.network.n_hidden_layers = 2
    run_cfg.network.n_hidden_units = 4
    run_cfg.network.fourier_features = 4
    return run_cfg


def test_save_last_checkpoint_payload(tmp_path):
    run_cfg = small_config(tmp_path)
    model = PINNSolidification(run_cfg.network, run_cfg.case, seed=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    log = {"total": 1.5, "bc_ic": 1.5}

    target = save_last_checkpoint(
        model,
        run_cfg,
        "phase2",
        3000,
        1.25,
        log,
        optimizer=optimizer,
        scheduler=None,
        optimizer_name="Adam",
    )

    assert target == tmp_path / "last.pth"
    payload = torch.load(target, map_location="cpu", weights_only=False)
    assert payload["phase"] == "phase2"
    assert payload["iteration"] == 3000
    assert payload["best_loss"] == 1.25
    assert payload["last_log"] == log
    assert payload["model_state"]
    assert payload["optimizer_state"] is not None
    assert payload["optimizer_name"] == "Adam"
    assert payload["config"]["training"]["last_checkpoint"] == "last.pth"
    assert "rng_state" in payload


def test_cleanup_previous_best_checkpoints_keeps_last(tmp_path):
    run_cfg = small_config(tmp_path)
    last_path = tmp_path / "last.pth"
    old_best = tmp_path / "pinns_v1.0_ep00001_loss1.0000e+00.pth"
    keep_best = tmp_path / "pinns_v1.0_ep00002_loss9.0000e-01.pth"
    last_path.write_bytes(b"last")
    old_best.write_bytes(b"old")
    keep_best.write_bytes(b"keep")

    cleanup_previous_best_checkpoints(run_cfg, keep_best.name)

    assert last_path.exists()
    assert keep_best.exists()
    assert not old_best.exists()


def test_clear_last_checkpoint_removes_stale_file(tmp_path):
    run_cfg = small_config(tmp_path)
    last_path = tmp_path / "last.pth"
    last_path.write_bytes(b"stale")

    clear_last_checkpoint(run_cfg)

    assert not last_path.exists()
    clear_last_checkpoint(run_cfg)


def test_final_training_iteration_uses_last_active_phase(tmp_path):
    run_cfg = small_config(tmp_path)
    run_cfg.training.phase1_iters = 10
    run_cfg.training.phase2_iters = 10
    run_cfg.training.phase3_iters = 0
    assert final_training_iteration(run_cfg) == 10

    run_cfg.training.phase2_iters = 30
    assert final_training_iteration(run_cfg) == 30

    run_cfg.training.phase3_iters = 5
    assert final_training_iteration(run_cfg) == 35

    run_cfg.training.phase2_iters = 8
    run_cfg.training.phase3_iters = 0
    assert final_training_iteration(run_cfg) == 10
