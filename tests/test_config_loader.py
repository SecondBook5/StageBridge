"""Mission 2 config-loading tests for the rebuilt repo layout."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from stagebridge.notebook_api import compose_config
from stagebridge.utils.config_loader import load_yaml_config


def test_compose_config_uses_normalized_profiles() -> None:
    cfg = compose_config(
        "default",
        overrides=["data=local", "train=smoke", "evaluation=baseline"],
    )

    assert cfg.run_name == "stagebridge_v1"
    assert cfg.data.dataset == "luad_evo"
    assert cfg.spatial_mapping.method == "tangram"
    assert cfg.context_model.mode == "set_only"
    assert cfg.train.profile == "smoke"
    assert cfg.evaluation.profile == "baseline"
    assert cfg.transition_model.wes_regularizer.enabled is True


def test_load_yaml_config_reads_scoped_train_profile() -> None:
    cfg = load_yaml_config(Path("configs/train/smoke.yaml"), expand_env=False)

    assert cfg["train"]["profile"] == "smoke"
    assert cfg["train"]["device"] == "cpu"
    assert cfg["train"]["max_epochs"] == 2


def test_env_expansion(tmp_path: Path) -> None:
    yaml_content = "data_root: '${TEST_SB_ROOT}/data'\n"
    config_file = tmp_path / "test.yaml"
    config_file.write_text(yaml_content, encoding="utf-8")

    os.environ["TEST_SB_ROOT"] = "/tmp/test_data"
    try:
        cfg = load_yaml_config(config_file, expand_env=True)
    finally:
        del os.environ["TEST_SB_ROOT"]

    assert cfg["data_root"] == "/tmp/test_data/data"


def test_env_expansion_missing_var(tmp_path: Path) -> None:
    yaml_content = "path: '${NONEXISTENT_VAR_XYZ}/foo'\n"
    config_file = tmp_path / "test.yaml"
    config_file.write_text(yaml_content, encoding="utf-8")

    with pytest.raises(EnvironmentError, match="NONEXISTENT_VAR_XYZ"):
        load_yaml_config(config_file, expand_env=True)
