"""Notebook-facing orchestration API for the rebuilt StageBridge layout."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from omegaconf import DictConfig, OmegaConf

from stagebridge.pipelines.run_context_model import run_context_model
from stagebridge.pipelines.run_evaluation import run_evaluation
from stagebridge.pipelines.run_full import run_full
from stagebridge.pipelines.run_reference import run_reference
from stagebridge.pipelines.run_spatial_mapping import run_spatial_mapping
from stagebridge.pipelines.run_transition_model import run_transition_model
from stagebridge.utils.config_loader import load_yaml_config

_CONFIG_DIR = (Path(__file__).resolve().parent.parent / "configs").resolve()
_DEFAULT_CONFIG = _CONFIG_DIR / "default.yaml"
_COMPONENT_DIRS = {
    "data": _CONFIG_DIR / "data",
    "spatial_mapping": _CONFIG_DIR / "spatial_mapping",
    "context_model": _CONFIG_DIR / "context_model",
    "splits": _CONFIG_DIR / "splits",
    "train": _CONFIG_DIR / "train",
    "evaluation": _CONFIG_DIR / "evaluation",
}
_TRANSITION_COMPONENTS = [
    _CONFIG_DIR / "transition_model" / "disease_edges.yaml",
    _CONFIG_DIR / "transition_model" / "gaussian_init.yaml",
    _CONFIG_DIR / "transition_model" / "stochastic_dynamics.yaml",
    _CONFIG_DIR / "transition_model" / "schrodinger_bridge.yaml",
    _CONFIG_DIR / "transition_model" / "wes_regularizer.yaml",
]
_ENTRYPOINT_ALIASES = {
    "config": "default",
    "train": "default",
    "eval": "default",
}

StepFn = Callable[[DictConfig], dict[str, Any]]


def _load_component(path: Path) -> DictConfig:
    if not path.exists():
        raise FileNotFoundError(f"Missing config component: {path}")
    return OmegaConf.create(load_yaml_config(path))


def _selected_profiles(base_cfg: DictConfig) -> dict[str, str]:
    defaults = {
        "data": "luad_evo",
        "spatial_mapping": "tangram",
        "context_model": "set_only",
        "splits": "donor_holdout",
        "train": "full_v1",
        "evaluation": "baseline",
    }
    profiles = OmegaConf.to_container(base_cfg.get("profiles", {}), resolve=False) or {}
    return {key: str(profiles.get(key, value)) for key, value in defaults.items()}


def compose_config(entrypoint: str = "default", overrides: list[str] | None = None) -> DictConfig:
    """Compose a config tree from the normalized repo layout."""
    entrypoint = _ENTRYPOINT_ALIASES.get(entrypoint, entrypoint)
    if entrypoint != "default":
        raise ValueError(f"Unsupported config entrypoint '{entrypoint}'. Use 'default'.")

    overrides = list(overrides or [])
    cfg = _load_component(_DEFAULT_CONFIG)
    selected = _selected_profiles(cfg)

    if entrypoint == "default":
        pass

    if "train=smoke" not in overrides and "evaluation=baseline" not in overrides:
        if entrypoint == "train":
            selected["train"] = "full_v1"
        if entrypoint == "eval":
            selected["evaluation"] = "baseline"

    dotlist_overrides: list[str] = []
    for override in overrides:
        if "=" not in override:
            dotlist_overrides.append(override)
            continue
        key, value = override.split("=", 1)
        if "." not in key and key in _COMPONENT_DIRS:
            selected[key] = value
            continue
        dotlist_overrides.append(override)

    for group, profile in selected.items():
        cfg = OmegaConf.merge(cfg, _load_component(_COMPONENT_DIRS[group] / f"{profile}.yaml"))
    for component_path in _TRANSITION_COMPONENTS:
        cfg = OmegaConf.merge(cfg, _load_component(component_path))
    if dotlist_overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(dotlist_overrides))
    return cfg


_STEP_REGISTRY: dict[str, StepFn] = {
    "reference": run_reference,
    "spatial_mapping": run_spatial_mapping,
    "context_model": run_context_model,
    "transition_model": run_transition_model,
    "evaluation": run_evaluation,
    "full": run_full,
    # compatibility aliases retained at the API boundary only
    "build_snrna": run_reference,
    "build_spatial": run_reference,
    "map_hlca": run_reference,
    "run_tangram": run_spatial_mapping,
    "train": run_transition_model,
    "evaluate": run_evaluation,
}


def run_step(step: str, cfg: DictConfig) -> dict[str, Any]:
    """Run one pipeline step from the rebuilt pipeline namespace."""
    fn = _STEP_REGISTRY.get(step)
    if fn is None:
        valid = ", ".join(sorted(_STEP_REGISTRY))
        raise ValueError(f"Unknown step '{step}'. Valid steps: {valid}")
    return fn(cfg)


def run_pipeline(cfg: DictConfig, steps: list[str] | None = None) -> dict[str, dict[str, Any]]:
    """Run the default StageBridge v1 orchestration order."""
    ordered_steps = steps or [
        "reference",
        "spatial_mapping",
        "context_model",
        "transition_model",
        "evaluation",
    ]
    return {step: run_step(step, cfg) for step in ordered_steps}


def load_run(run_dir: str | Path) -> dict[str, Any]:
    """Load JSON/YAML artifacts from a run-like directory for notebook inspection."""
    run_dir = Path(run_dir)
    payload: dict[str, Any] = {"run_dir": str(run_dir), "files": {}}
    if not run_dir.exists():
        payload["error"] = "missing_run_dir"
        return payload

    for path in sorted(run_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix not in {".json", ".yaml", ".yml", ".md"}:
            continue
        rel = path.relative_to(run_dir).as_posix()
        if path.suffix == ".json":
            try:
                payload["files"][rel] = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                payload["files"][rel] = {"_error": "failed_to_parse_json"}
        else:
            payload["files"][rel] = path.read_text(encoding="utf-8")
    return payload


def available_steps() -> list[str]:
    return sorted(_STEP_REGISTRY)


__all__ = ["available_steps", "compose_config", "load_run", "run_pipeline", "run_step"]
