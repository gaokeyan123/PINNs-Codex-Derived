"""Tests for current-format configuration rebuilding."""

import sys
from dataclasses import asdict
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import cfg, config_from_dict


def test_config_from_dict_accepts_current_format():
    data = asdict(cfg)
    data["training"]["checkpoint_dir"] = str(data["training"]["checkpoint_dir"])
    data["output_dir"] = str(data["output_dir"])

    restored = config_from_dict(data)

    assert restored.case.Pe_D == cfg.case.Pe_D
    assert restored.case.Re_D == cfg.case.Re_D
    assert restored.case.tau_end == cfg.case.tau_end
    assert restored.training.last_checkpoint == cfg.training.last_checkpoint


def test_phase3_disabled_by_default():
    assert cfg.training.phase3_iters == 0


def test_last_checkpoint_default_name():
    assert cfg.training.last_checkpoint == "last.pth"


def test_config_from_dict_rejects_legacy_case_fields():
    data = asdict(cfg)
    data["case"]["nondim_scheme"] = "legacy"

    with pytest.raises(ValueError, match="unsupported config field"):
        config_from_dict(data)
