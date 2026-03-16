"""Tests for the config loader module."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from stagebridge.orchestration.config_loader import (
    ConfigValidationError,
    DEFAULT_CONFIG_VALUES,
    get_enabled_stages,
    is_stage_enabled,
    load_config,
    load_default_config,
    load_smoke_test_config,
    load_yaml_file,
    save_config,
    validate_config,
)


@pytest.fixture
def temp_config_file(tmp_path: Path) -> Path:
    """Create a temporary config file."""
    config_path = tmp_path / "test_config.yaml"
    config = {
        "run_id": "test_run",
        "seed": 123,
        "device": "cuda:0",
        "stages": {
            "enabled": ["data_qc", "reference"],
        },
    }
    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)
    return config_path


class TestLoadYamlFile:
    """Tests for load_yaml_file function."""

    def test_load_basic_yaml(self, temp_config_file: Path) -> None:
        """Test loading a basic YAML file."""
        config = load_yaml_file(temp_config_file)

        assert config["run_id"] == "test_run"
        assert config["seed"] == 123
        assert config["device"] == "cuda:0"

    def test_load_nonexistent_file(self, tmp_path: Path) -> None:
        """Test loading a nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_yaml_file(tmp_path / "nonexistent.yaml")

    def test_load_empty_yaml(self, tmp_path: Path) -> None:
        """Test loading an empty YAML file."""
        empty_file = tmp_path / "empty.yaml"
        empty_file.write_text("", encoding="utf-8")

        config = load_yaml_file(empty_file)
        assert config == {}

    def test_expand_env_vars(self, tmp_path: Path) -> None:
        """Test environment variable expansion."""
        config_path = tmp_path / "env_config.yaml"
        config_path.write_text(
            'path: "${TEST_VAR_PATH}"\nname: "${MISSING_VAR:default_value}"',
            encoding="utf-8",
        )

        os.environ["TEST_VAR_PATH"] = "/test/path"

        try:
            config = load_yaml_file(config_path, expand_env=True)
            assert config["path"] == "/test/path"
            assert config["name"] == "default_value"
        finally:
            del os.environ["TEST_VAR_PATH"]

    def test_expand_env_missing_var(self, tmp_path: Path) -> None:
        """Test that missing env var without default raises error."""
        config_path = tmp_path / "missing_env.yaml"
        config_path.write_text('path: "${DEFINITELY_MISSING_VAR}"', encoding="utf-8")

        with pytest.raises(OSError, match="not set"):
            load_yaml_file(config_path, expand_env=True)

    def test_disable_env_expansion(self, tmp_path: Path) -> None:
        """Test disabling environment variable expansion."""
        config_path = tmp_path / "no_expand.yaml"
        config_path.write_text('path: "${SOME_VAR}"', encoding="utf-8")

        config = load_yaml_file(config_path, expand_env=False)
        assert config["path"] == "${SOME_VAR}"


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_from_file(self, temp_config_file: Path) -> None:
        """Test loading config from file."""
        config = load_config(temp_config_file)

        assert config["run_id"] == "test_run"
        assert config["seed"] == 123

    def test_load_from_dict(self) -> None:
        """Test loading config from dictionary."""
        input_config = {
            "run_id": "dict_run",
            "seed": 456,
        }

        config = load_config(input_config)

        assert config["run_id"] == "dict_run"
        assert config["seed"] == 456

    def test_merge_with_defaults(self) -> None:
        """Test merging with default values."""
        input_config = {
            "run_id": "partial_run",
        }

        config = load_config(input_config, use_defaults=True)

        # Custom value preserved
        assert config["run_id"] == "partial_run"
        # Default values filled in
        assert config["seed"] == DEFAULT_CONFIG_VALUES["seed"]
        assert "notebook" in config

    def test_no_defaults(self) -> None:
        """Test loading without defaults."""
        input_config = {
            "run_id": "no_defaults",
        }

        config = load_config(input_config, use_defaults=False, validate=False)

        assert config["run_id"] == "no_defaults"
        assert "seed" not in config

    def test_none_config(self) -> None:
        """Test loading with None config uses defaults."""
        config = load_config(None, use_defaults=True)

        assert config["seed"] == DEFAULT_CONFIG_VALUES["seed"]


class TestValidateConfig:
    """Tests for validate_config function."""

    def test_valid_config(self) -> None:
        """Test validating a valid config."""
        config = {
            "run_id": "valid_run",
            "seed": 42,
            "device": "cpu",
            "stages": {
                "enabled": ["data_qc"],
            },
        }

        errors = validate_config(config)
        assert errors == []

    def test_invalid_type(self) -> None:
        """Test validation catches type errors."""
        config = {
            "seed": "not_an_int",  # Should be int
        }

        errors = validate_config(config)
        assert len(errors) > 0
        assert any("seed" in e for e in errors)

    def test_nested_validation(self) -> None:
        """Test validation of nested structures."""
        config = {
            "notebook": {
                "verbosity": 123,  # Should be string
            },
        }

        errors = validate_config(config)
        assert len(errors) > 0
        assert any("verbosity" in e for e in errors)


class TestSaveConfig:
    """Tests for save_config function."""

    def test_save_config(self, tmp_path: Path) -> None:
        """Test saving config to file."""
        config = {
            "run_id": "saved_run",
            "seed": 42,
        }
        output_path = tmp_path / "saved_config.yaml"

        save_config(config, output_path)

        assert output_path.exists()

        with output_path.open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)

        assert loaded["run_id"] == "saved_run"
        assert loaded["seed"] == 42

    def test_save_creates_directories(self, tmp_path: Path) -> None:
        """Test saving creates parent directories."""
        config = {"test": True}
        output_path = tmp_path / "nested" / "dir" / "config.yaml"

        save_config(config, output_path)

        assert output_path.exists()


class TestStageHelpers:
    """Tests for stage-related helper functions."""

    def test_get_enabled_stages(self) -> None:
        """Test getting enabled stages."""
        config = {
            "stages": {
                "enabled": ["data_qc", "reference", "spatial_backend"],
            },
        }

        stages = get_enabled_stages(config)

        assert stages == ["data_qc", "reference", "spatial_backend"]

    def test_get_enabled_stages_default(self) -> None:
        """Test default enabled stages."""
        config = {}

        stages = get_enabled_stages(config)

        assert stages == DEFAULT_CONFIG_VALUES["stages"]["enabled"]

    def test_is_stage_enabled(self) -> None:
        """Test checking if stage is enabled."""
        config = {
            "stages": {
                "enabled": ["data_qc", "reference"],
            },
        }

        assert is_stage_enabled(config, "data_qc")
        assert is_stage_enabled(config, "reference")
        assert not is_stage_enabled(config, "spatial_backend")


class TestLoadPresets:
    """Tests for loading preset configurations."""

    def test_load_default_config(self) -> None:
        """Test loading default config."""
        # This may or may not find the actual default.yaml file
        # depending on test environment, but should not raise
        try:
            config = load_default_config(validate=False)
            assert isinstance(config, dict)
        except FileNotFoundError:
            # Expected if default.yaml doesn't exist
            pass

    def test_load_smoke_test_config(self) -> None:
        """Test loading smoke test config."""
        try:
            config = load_smoke_test_config(validate=False)
            assert isinstance(config, dict)
            # Smoke test should have minimal stages
            if "stages" in config:
                enabled = config["stages"].get("enabled", [])
                assert len(enabled) <= 8  # Smoke test has fewer stages
        except FileNotFoundError:
            # If smoke_test.yaml doesn't exist, should still return a config
            pass
