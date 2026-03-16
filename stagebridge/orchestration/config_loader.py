"""Configuration loading and validation for StageBridge orchestration.

This module provides config loading, merging, schema validation,
and support for default and smoke test configurations.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml


# Environment variable expansion pattern
_ENV_RE = re.compile(r"\$\{(\w+)(?::([^}]*))?\}")

# Default config paths
_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"
_DEFAULT_CONFIG = _CONFIG_DIR / "default.yaml"
_SMOKE_TEST_CONFIG = _CONFIG_DIR / "smoke_test.yaml"


# Config schema for validation
CONFIG_SCHEMA = {
    "run_id": {"type": "string", "required": False},
    "seed": {"type": "int", "required": False, "default": 42},
    "device": {"type": "string", "required": False, "default": "cpu"},
    "dataset": {
        "type": "dict",
        "required": False,
        "properties": {
            "name": {"type": "string", "required": False},
            "path": {"type": "string", "required": False},
        },
    },
    "stages": {
        "type": "dict",
        "required": False,
        "properties": {
            "enabled": {"type": "list", "required": False},
        },
    },
    "spatial_backends": {"type": "list", "required": False},
    "baselines": {"type": "list", "required": False},
    "ablations": {"type": "list", "required": False},
    "resume_if_possible": {"type": "bool", "required": False, "default": True},
    "force_rerun": {"type": "bool", "required": False, "default": False},
    "notebook": {
        "type": "dict",
        "required": False,
        "properties": {
            "verbosity": {"type": "string", "required": False, "default": "normal"},
            "show_figures": {"type": "bool", "required": False, "default": True},
            "figure_dpi": {"type": "int", "required": False, "default": 100},
        },
    },
}


# Default configuration values
DEFAULT_CONFIG_VALUES: dict[str, Any] = {
    "seed": 42,
    "device": "cpu",
    "stages": {
        "enabled": [
            "data_qc",
            "reference",
            "spatial_backend",
            "baselines",
            "full_model",
            "ablations",
            "biology",
            "figures",
        ],
    },
    "spatial_backends": ["tangram"],
    "baselines": ["mlp", "gcn"],
    "ablations": ["no_spatial", "no_attention"],
    "resume_if_possible": True,
    "force_rerun": False,
    "notebook": {
        "verbosity": "normal",
        "show_figures": True,
        "figure_dpi": 100,
    },
}


def _expand_env(value: str) -> str:
    """Expand environment variables in a string.

    Supports ${VAR} and ${VAR:default} syntax.
    """

    def _sub(m: re.Match) -> str:
        name = m.group(1)
        default = m.group(2)
        val = os.environ.get(name)
        if val is None:
            if default is not None:
                return default
            raise OSError(
                f"Environment variable '{name}' is not set. "
                f"Export it or use ${{VAR:default}} syntax."
            )
        return val

    return _ENV_RE.sub(_sub, value)


def _expand_recursive(obj: Any) -> Any:
    """Recursively expand environment variables in a nested structure."""
    if isinstance(obj, str):
        return _expand_env(obj)
    if isinstance(obj, dict):
        return {k: _expand_recursive(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_recursive(v) for v in obj]
    return obj


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dictionaries, with override taking precedence."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


class ConfigValidationError(Exception):
    """Raised when config validation fails."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("Config validation failed:\n" + "\n".join(f"  - {e}" for e in errors))


def _validate_type(value: Any, expected_type: str, path: str) -> list[str]:
    """Validate a value against an expected type."""
    errors: list[str] = []

    type_checks = {
        "string": lambda v: isinstance(v, str),
        "int": lambda v: isinstance(v, int) and not isinstance(v, bool),
        "float": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
        "bool": lambda v: isinstance(v, bool),
        "list": lambda v: isinstance(v, list),
        "dict": lambda v: isinstance(v, dict),
    }

    if expected_type in type_checks:
        if not type_checks[expected_type](value):
            errors.append(f"{path}: expected {expected_type}, got {type(value).__name__}")

    return errors


def _validate_config_recursive(
    config: dict[str, Any],
    schema: dict[str, Any],
    path: str = "",
) -> list[str]:
    """Recursively validate config against schema."""
    errors: list[str] = []

    for key, spec in schema.items():
        full_path = f"{path}.{key}" if path else key

        if key not in config:
            if spec.get("required", False):
                errors.append(f"{full_path}: required field is missing")
            continue

        value = config[key]
        expected_type = spec.get("type", "string")

        # Type check
        errors.extend(_validate_type(value, expected_type, full_path))

        # Nested dict validation
        if expected_type == "dict" and "properties" in spec and isinstance(value, dict):
            errors.extend(_validate_config_recursive(value, spec["properties"], full_path))

    return errors


def validate_config(config: dict[str, Any]) -> list[str]:
    """Validate a configuration against the schema.

    Parameters
    ----------
    config : dict
        The configuration to validate

    Returns
    -------
    list of str
        List of validation errors (empty if valid)
    """
    return _validate_config_recursive(config, CONFIG_SCHEMA)


def load_yaml_file(
    path: str | Path,
    *,
    expand_env: bool = True,
) -> dict[str, Any]:
    """Load a YAML file with optional environment variable expansion.

    Parameters
    ----------
    path : str or Path
        Path to YAML file
    expand_env : bool
        Whether to expand ${VAR} references (default: True)

    Returns
    -------
    dict
        Loaded configuration
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        return {}

    if expand_env:
        return _expand_recursive(raw)

    return raw


def load_config(
    config: dict[str, Any] | str | Path | None = None,
    *,
    use_defaults: bool = True,
    validate: bool = True,
    expand_env: bool = True,
) -> dict[str, Any]:
    """Load and merge configuration from various sources.

    Parameters
    ----------
    config : dict, str, Path, or None
        Configuration source:
        - dict: use directly
        - str/Path: load from YAML file
        - None: use defaults only
    use_defaults : bool
        Whether to merge with default values (default: True)
    validate : bool
        Whether to validate the result (default: True)
    expand_env : bool
        Whether to expand environment variables (default: True)

    Returns
    -------
    dict
        The resolved configuration

    Raises
    ------
    ConfigValidationError
        If validation is enabled and fails
    """
    # Start with defaults if requested
    if use_defaults:
        result = dict(DEFAULT_CONFIG_VALUES)
    else:
        result = {}

    # Load from file if path provided
    if isinstance(config, (str, Path)):
        file_config = load_yaml_file(config, expand_env=expand_env)
        result = _deep_merge(result, file_config)
    elif isinstance(config, dict):
        if expand_env:
            config = _expand_recursive(config)
        result = _deep_merge(result, config)

    # Validate if requested
    if validate:
        errors = validate_config(result)
        if errors:
            raise ConfigValidationError(errors)

    return result


def load_default_config(*, validate: bool = True) -> dict[str, Any]:
    """Load the default configuration.

    Parameters
    ----------
    validate : bool
        Whether to validate the result (default: True)

    Returns
    -------
    dict
        The default configuration
    """
    if _DEFAULT_CONFIG.exists():
        return load_config(_DEFAULT_CONFIG, validate=validate)
    return load_config(None, validate=validate)


def load_smoke_test_config(*, validate: bool = True) -> dict[str, Any]:
    """Load the smoke test configuration.

    Parameters
    ----------
    validate : bool
        Whether to validate the result (default: True)

    Returns
    -------
    dict
        The smoke test configuration
    """
    if _SMOKE_TEST_CONFIG.exists():
        return load_config(_SMOKE_TEST_CONFIG, validate=validate)

    # Create minimal smoke test config
    smoke_config = {
        "run_id": "smoke_test",
        "seed": 42,
        "device": "cpu",
        "stages": {
            "enabled": ["data_qc", "reference"],
        },
        "spatial_backends": ["tangram"],
        "baselines": ["mlp"],
        "ablations": [],
        "notebook": {
            "verbosity": "minimal",
            "show_figures": False,
            "figure_dpi": 72,
        },
    }

    return load_config(smoke_config, validate=validate)


def save_config(config: dict[str, Any], path: str | Path) -> None:
    """Save a configuration to a YAML file.

    Parameters
    ----------
    config : dict
        The configuration to save
    path : str or Path
        Output file path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def get_enabled_stages(config: dict[str, Any]) -> list[str]:
    """Get the list of enabled stages from config.

    Parameters
    ----------
    config : dict
        The configuration

    Returns
    -------
    list of str
        List of enabled stage names
    """
    stages = config.get("stages", {})
    if isinstance(stages, dict):
        return stages.get("enabled", DEFAULT_CONFIG_VALUES["stages"]["enabled"])
    return DEFAULT_CONFIG_VALUES["stages"]["enabled"]


def is_stage_enabled(config: dict[str, Any], stage_name: str) -> bool:
    """Check if a stage is enabled in the config.

    Parameters
    ----------
    config : dict
        The configuration
    stage_name : str
        Name of the stage to check

    Returns
    -------
    bool
        True if stage is enabled
    """
    return stage_name in get_enabled_stages(config)
