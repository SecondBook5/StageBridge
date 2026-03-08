"""Spatial-mapping pipeline entrypoint."""
from __future__ import annotations

from typing import Any

from omegaconf import DictConfig

from stagebridge.spatial_mapping.destvi_mapper import run_destvi
from stagebridge.spatial_mapping.tacco_mapper import run_tacco
from stagebridge.spatial_mapping.tangram_mapper import run_tangram


def run_spatial_mapping(
    cfg: DictConfig,
    *,
    reference_output: dict[str, Any] | None = None,
) -> dict[str, Any]:
    method = str(cfg.get("spatial_mapping", {}).get("method", "tangram"))
    stages = list(cfg.get("data", {}).get("stages", [])) or None
    max_spots_per_stage = int(cfg.get("spatial_mapping", {}).get("max_spots_per_stage", 128))
    if method == "tangram":
        result = run_tangram(
            cfg,
            donors=None,
            stages=stages,
            max_spots_per_stage=max_spots_per_stage,
            seed=int(cfg.get("seed", 42)),
        )
    elif method == "tacco":
        result = run_tacco(
            cfg,
            donors=None,
            stages=stages,
            max_spots_per_stage=max_spots_per_stage,
            seed=int(cfg.get("seed", 42)),
        )
    elif method == "destvi":
        result = run_destvi(
            cfg,
            donors=None,
            stages=stages,
            max_spots_per_stage=max_spots_per_stage,
            seed=int(cfg.get("seed", 42)),
        )
    else:
        raise ValueError(f"Unsupported spatial_mapping.method '{method}'.")
    return {
        "ok": True,
        "pipeline": "spatial_mapping",
        "status": result.status,
        "spatial_mapping": result.summary(),
        "mapping_result": result,
        "reference_summary": None if reference_output is None else reference_output.get("reference"),
    }
