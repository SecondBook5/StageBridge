"""Plain YAML config loader (no Hydra dependency).

Loads YAML files and expands ``${ENV_VAR}`` references in string values.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml


_ENV_RE = re.compile(r"\$\{(\w+)\}")


def _expand_env(value: str) -> str:
    """Replace ``${VAR}`` with the environment variable value."""

    def _sub(m: re.Match) -> str:
        name = m.group(1)
        val = os.environ.get(name)
        if val is None:
            raise OSError(
                f"Environment variable '{name}' is not set. "
                f"Export it or remove the ${{...}} reference from the config."
            )
        return val

    return _ENV_RE.sub(_sub, value)


def _expand_recursive(obj: Any) -> Any:
    """Walk a nested dict/list and expand env vars in string values."""
    if isinstance(obj, str):
        return _expand_env(obj)
    if isinstance(obj, dict):
        return {k: _expand_recursive(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_recursive(v) for v in obj]
    return obj


def load_yaml_config(
    path: str | Path,
    *,
    expand_env: bool = True,
) -> dict[str, Any]:
    """Load a YAML config file, optionally expanding env vars.

    Parameters
    ----------
    path : path to YAML file
    expand_env : if True (default), expand ``${VAR}`` references

    Returns
    -------
    dict with config values
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if raw is None:
        return {}

    if expand_env:
        return _expand_recursive(raw)
    return raw
