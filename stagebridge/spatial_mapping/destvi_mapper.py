"""DestVI provider surface for continuous-state spatial mapping."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from stagebridge.data.luad_evo.visium import load_luad_evo_spatial_mapping
from stagebridge.spatial_mapping.base import SpatialMappingResult
from stagebridge.spatial_mapping.qc import summarize_mapping_qc
from stagebridge.spatial_mapping.tangram_mapper import _provider_version


def run_destvi(
    cfg: Any,
    *,
    stages: list[str] | None = None,
    donors: list[str] | None = None,
    max_spots_per_stage: int | None = None,
    seed: int = 42,
) -> SpatialMappingResult:
    provider_cfg = dict(cfg.get("spatial_mapping", {})) if hasattr(cfg, "get") else dict(cfg["spatial_mapping"])
    execution_mode = str(provider_cfg.get("execution_mode", "load_precomputed"))
    precomputed_path = provider_cfg.get("precomputed_h5ad")
    provider_version = _provider_version("scvi-tools")

    if precomputed_path and Path(str(precomputed_path)).exists():
        cohort = load_luad_evo_spatial_mapping(
            cfg,
            mapping_h5ad_path=Path(str(precomputed_path)),
            composition_key="X_destvi_ct",
            columns_key="destvi_ct_columns",
            stages=stages,
            donors=donors,
            max_spots_per_stage=max_spots_per_stage,
            seed=seed,
        )
        return SpatialMappingResult(
            method="destvi",
            status="complete",
            provider_version=provider_version,
            execution_mode=execution_mode,
            compositions=cohort.compositions,
            coords=cohort.coords,
            obs=cohort.obs,
            feature_names=cohort.feature_names,
            source_path=cohort.source_path,
            qc=summarize_mapping_qc(cohort.compositions),
            provenance={"mode": "loaded", "precomputed_h5ad": str(precomputed_path)},
            notes="Loaded precomputed DestVI spatial mapping output.",
        )

    return SpatialMappingResult(
        method="destvi",
        status="not_configured",
        provider_version=provider_version,
        execution_mode=execution_mode,
        provenance={"mode": "unavailable", "precomputed_h5ad": precomputed_path},
        notes=(
            "DestVI support is present in the provider surface, but no precomputed output is configured "
            "and a raw rebuild path is not implemented yet."
        ),
    )
