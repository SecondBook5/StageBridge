"""Reference-layer pipeline entrypoint."""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig

from stagebridge.reference.hlca_mapper import run_active_reference_latent


def run_reference(cfg: DictConfig) -> dict[str, Any]:
    """Run the active HLCA reference-latent branch for LUAD evolution."""
    max_cells_per_stage = int(cfg.get("reference", {}).get("max_cells_per_stage", 256))
    stages = list(cfg.get("data", {}).get("stages", [])) or None
    result = run_active_reference_latent(
        cfg,
        stages=stages,
        max_cells_per_stage=max_cells_per_stage,
        seed=int(cfg.get("seed", 42)),
    )
    return {
        "ok": True,
        "pipeline": "reference",
        "status": "complete",
        "reference": result.summary(),
        "cohort": result.cohort,
    }
