"""Notebook-facing orchestration API for StageBridge workflows."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from stagebridge.settings import workflow_request_from_cfg
from stagebridge.workflows import (
    build_snrna,
    build_spatial,
    map_hlca,
    run_evaluation,
    run_tangram,
    run_training,
)

_CONFIG_DIR = (Path(__file__).resolve().parent.parent / "configs").resolve()

StepFn = Callable[[DictConfig], dict[str, Any]]

_STEP_REGISTRY: dict[str, StepFn] = {
    "build_snrna": build_snrna,
    "build_spatial": build_spatial,
    "map_hlca": map_hlca,
    "run_tangram": run_tangram,
    "train": run_training,
    "evaluate": run_evaluation,
    # compatibility aliases
    "run_snrna_pipeline": build_snrna,
    "run_spatial_pipeline": build_spatial,
    "run_hlca_mapping": map_hlca,
    "run_tangram_mapping": run_tangram,
    "train_stagebridge": run_training,
    "eval_stagebridge": run_evaluation,
}


def compose_config(entrypoint: str = "config", overrides: list[str] | None = None) -> DictConfig:
    """Compose a Hydra config object for notebook/CLI use.

    Parameters
    ----------
    entrypoint:
        Config root filename without extension (e.g. ``config``, ``train``, ``eval``).
    overrides:
        Optional Hydra override strings (e.g. ``["data=local", "experiment=smoke"]``).
    """
    overrides = overrides or []
    with initialize_config_dir(version_base=None, config_dir=str(_CONFIG_DIR)):
        cfg = compose(config_name=entrypoint, overrides=overrides)
    return cfg


def run_step(step: str, cfg: DictConfig) -> dict[str, Any]:
    """Run a single workflow step using a composed config."""
    fn = _STEP_REGISTRY.get(step)
    if fn is None:
        valid = ", ".join(sorted(_STEP_REGISTRY))
        raise ValueError(f"Unknown step '{step}'. Valid steps: {valid}")

    # Build typed settings once to fail fast on malformed config.
    workflow_request_from_cfg(step=step, cfg=cfg)
    return fn(cfg)


def run_pipeline(cfg: DictConfig, steps: list[str] | None = None) -> dict[str, dict[str, Any]]:
    """Run a sequence of workflow steps and return per-step summaries."""
    ordered_steps = steps or [
        "build_snrna",
        "build_spatial",
        "map_hlca",
        "run_tangram",
        "train",
    ]

    out: dict[str, dict[str, Any]] = {}
    for step in ordered_steps:
        out[step] = run_step(step, cfg)
    return out


def load_run(run_dir: str | Path) -> dict[str, Any]:
    """Load common run artifacts (`metrics`, manifests, eval) from a run directory."""
    run_dir = Path(run_dir)
    tables_dir = run_dir / "tables"
    payload: dict[str, Any] = {
        "run_dir": str(run_dir),
        "tables_dir": str(tables_dir),
        "files": {},
    }

    candidates = [
        *tables_dir.glob("metrics_*.json"),
        *tables_dir.glob("run_manifest*.json"),
        *tables_dir.glob("eval_*.json"),
    ]
    for path in sorted(candidates):
        try:
            payload["files"][path.name] = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            payload["files"][path.name] = {"_error": "failed_to_parse_json"}

    return payload


def available_steps() -> list[str]:
    """Return sorted list of accepted step identifiers."""
    return sorted(_STEP_REGISTRY)


__all__ = [
    "available_steps",
    "compose_config",
    "load_run",
    "run_pipeline",
    "run_step",
]
