"""Tests for postprocess checkpoint selection helpers."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from postprocess import (
    checkpoint_iteration,
    find_best_checkpoint,
    resolve_last_checkpoint,
    resolve_primary_checkpoint,
)


def test_find_best_checkpoint_chooses_lowest_loss(tmp_path):
    high = tmp_path / "pinns_v1.0_ep00010_loss5.0000e-01.pth"
    low = tmp_path / "pinns_v1.0_ep00020_loss2.5000e-01.pth"
    high.write_bytes(b"high")
    low.write_bytes(b"low")

    assert find_best_checkpoint(tmp_path) == low


def test_resolve_primary_checkpoint_falls_back_to_latest(tmp_path):
    latest = tmp_path / "latest.pth"
    latest.write_bytes(b"latest")

    assert resolve_primary_checkpoint(None, checkpoint_dir=tmp_path) == latest


def test_checkpoint_iteration_reads_iteration(tmp_path):
    ckpt = tmp_path / "last.pth"
    torch.save({"iteration": 1234}, ckpt)

    assert checkpoint_iteration(ckpt) == 1234


def test_missing_last_checkpoint_returns_none(tmp_path):
    primary = tmp_path / "latest.pth"
    primary.write_bytes(b"latest")

    assert resolve_last_checkpoint(primary) is None
