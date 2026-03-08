"""Evaluation pipeline entrypoint."""
from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, OmegaConf


def run_evaluation(cfg: DictConfig) -> dict[str, Any]:
    return {
        "ok": True,
        "pipeline": "evaluation",
        "status": "structural_stub",
        "evaluation": OmegaConf.to_container(cfg.get("evaluation", {}), resolve=True),
    }
