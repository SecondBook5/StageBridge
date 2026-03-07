"""Tests for the plain YAML config loader."""
from __future__ import annotations

import os
from pathlib import Path

import pytest


def test_load_smoke_config():
    from stagebridge.utils.config_loader import load_yaml_config

    cfg = load_yaml_config(
        Path("configs/topoflow/smoke.yaml"),
        expand_env=False,
    )
    assert cfg["model"]["input_dim"] == 16
    assert cfg["model"]["hidden_dim"] == 32
    assert cfg["training"]["device"] == "cpu"
    assert cfg["stages"]["order"] == ["AAH", "AIS"]
    assert cfg["data"]["synthetic"] is True


def test_load_default_config_no_env():
    """Default config has env vars; loading without expansion should work."""
    from stagebridge.utils.config_loader import load_yaml_config

    cfg = load_yaml_config(
        Path("configs/topoflow/default.yaml"),
        expand_env=False,
    )
    assert "model" in cfg
    assert "training" in cfg
    assert cfg["model"]["input_dim"] == 64
    # Env vars should remain unexpanded
    assert "${STAGEBRIDGE_DATA_ROOT}" in str(cfg["data"]["data_root"])


def test_env_expansion(tmp_path):
    from stagebridge.utils.config_loader import load_yaml_config

    yaml_content = "data_root: '${TEST_SB_ROOT}/data'\n"
    config_file = tmp_path / "test.yaml"
    config_file.write_text(yaml_content)

    os.environ["TEST_SB_ROOT"] = "/tmp/test_data"
    try:
        cfg = load_yaml_config(config_file, expand_env=True)
        assert cfg["data_root"] == "/tmp/test_data/data"
    finally:
        del os.environ["TEST_SB_ROOT"]


def test_env_expansion_missing_var(tmp_path):
    from stagebridge.utils.config_loader import load_yaml_config

    yaml_content = "path: '${NONEXISTENT_VAR_XYZ}/foo'\n"
    config_file = tmp_path / "test.yaml"
    config_file.write_text(yaml_content)

    with pytest.raises(EnvironmentError, match="NONEXISTENT_VAR_XYZ"):
        load_yaml_config(config_file, expand_env=True)


def test_artifacts_roundtrip(tmp_path):
    from stagebridge.utils.artifacts import RunArtifacts

    arts = RunArtifacts(
        run_id="test_run",
        config={"model": {"input_dim": 16}},
        metrics={"val_loss": 0.5},
    )
    arts.save(tmp_path / "artifacts")

    loaded = RunArtifacts.load(tmp_path / "artifacts")
    assert loaded.run_id == "test_run"
    assert loaded.config["model"]["input_dim"] == 16
    assert loaded.metrics["val_loss"] == 0.5
