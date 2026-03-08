"""Spatial-mapping pipeline entrypoint."""
from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, OmegaConf


def run_spatial_mapping(cfg: DictConfig) -> dict[str, Any]:
    return {
        "ok": True,
        "pipeline": "spatial_mapping",
        "status": "structural_stub",
        "spatial_mapping": OmegaConf.to_container(cfg.get("spatial_mapping", {}), resolve=True),
    }
