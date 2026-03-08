"""Transition-model pipeline entrypoint."""
from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, OmegaConf


def run_transition_model(cfg: DictConfig) -> dict[str, Any]:
    return {
        "ok": True,
        "pipeline": "transition_model",
        "status": "structural_stub",
        "transition_model": OmegaConf.to_container(cfg.get("transition_model", {}), resolve=True),
        "train": OmegaConf.to_container(cfg.get("train", {}), resolve=True),
    }
