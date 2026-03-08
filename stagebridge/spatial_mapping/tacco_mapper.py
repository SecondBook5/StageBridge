"""TACCO provider wrapper for raw snRNA -> spatial mapping."""
from __future__ import annotations

from importlib import metadata
from pathlib import Path
from typing import Any

import anndata
import numpy as np
import pandas as pd

from stagebridge.data.luad_evo.metadata import resolve_luad_evo_paths
from stagebridge.data.luad_evo.visium import load_luad_evo_spatial_mapping
from stagebridge.spatial_mapping.base import SpatialMappingResult
from stagebridge.spatial_mapping.qc import summarize_mapping_qc
from stagebridge.spatial_mapping.tangram_mapper import (
    _aligned_label_series_from_sources,
    _coerce_csr_float32,
    _read_h5ad_csr_rows,
    _read_h5ad_obs_frame,
    _read_h5ad_var_index,
    _mapping_cache_root,
    _normalize_obs_fields,
    _provider_version,
    _select_stage_rows,
    _stable_hash,
    _write_spatial_subset_h5ad,
)


def _write_reference_subset_h5ad(
    *,
    snrna_h5ad_path: Path,
    subset_h5ad_path: Path,
    label_col: str,
    stages: list[str] | None,
    max_cells_per_label: int,
    seed: int,
    fallback_labels_parquet_path: Path | None = None,
    fallback_latent_h5ad_path: Path | None = None,
) -> dict[str, Any]:
    all_obs = _normalize_obs_fields(
        _read_h5ad_obs_frame(
            snrna_h5ad_path,
            columns=["donor_id", "patient_id", "sample_id", "stage"],
        )
    )
    selected_rows = _select_stage_rows(
        all_obs,
        stages=stages,
        donors=None,
        max_rows_per_stage=None,
        seed=seed,
    )
    obs = all_obs.iloc[selected_rows].copy()
    row_lookup = pd.Series(selected_rows, index=obs.index)
    labels, label_meta = _aligned_label_series_from_sources(
        obs=obs,
        obs_index=obs.index,
        label_col=label_col,
        fallback_labels_parquet_path=fallback_labels_parquet_path,
        fallback_latent_h5ad_path=fallback_latent_h5ad_path,
    )
    obs[label_col] = labels
    obs = obs.loc[obs[label_col].notna()].copy()
    rows = row_lookup.reindex(obs.index).to_numpy(dtype=np.int64)
    if max_cells_per_label > 0:
        rng = np.random.default_rng(int(seed))
        keep_labels: list[str] = []
        for _, frame in obs.groupby(label_col, sort=True):
            names = frame.index.to_numpy()
            if frame.shape[0] <= max_cells_per_label:
                keep_labels.extend(names.tolist())
                continue
            keep = rng.choice(names, size=int(max_cells_per_label), replace=False)
            keep_labels.extend(keep.tolist())
        obs = obs.loc[keep_labels].copy()
        rows = row_lookup.reindex(obs.index).to_numpy(dtype=np.int64)

    subset_h5ad_path.parent.mkdir(parents=True, exist_ok=True)
    subset = anndata.AnnData(
        X=_read_h5ad_csr_rows(snrna_h5ad_path, rows, group_name="X"),
        obs=obs.copy(),
        var=pd.DataFrame(index=_read_h5ad_var_index(snrna_h5ad_path)),
    )
    try:
        subset.layers["counts"] = _read_h5ad_csr_rows(snrna_h5ad_path, rows, group_name="layers/counts")
    except Exception:
        pass
    subset.write_h5ad(subset_h5ad_path, compression="lzf")
    return {
        "n_cells": int(subset.n_obs),
        "n_genes": int(subset.n_vars),
        "n_labels": int(subset.obs[label_col].astype(str).nunique()),
        "label_col": label_col,
        "label_source": label_meta,
    }


def _tacco_cache_bundle(
    cfg: Any,
    *,
    stages: list[str] | None,
    donors: list[str] | None,
    max_spots_per_stage: int | None,
    seed: int,
) -> dict[str, Path]:
    paths = resolve_luad_evo_paths(cfg)
    provider_cfg = dict(cfg.get("spatial_mapping", {})) if hasattr(cfg, "get") else dict(cfg["spatial_mapping"])
    cache_key = _stable_hash(
        {
            "method": "tacco",
            "snrna_h5ad": str(paths.snrna_h5ad),
            "snrna_latent_h5ad": str(paths.snrna_latent_h5ad),
            "hlca_labels_parquet": str(paths.hlca_labels_parquet),
            "spatial_h5ad": str(paths.spatial_h5ad),
            "stages": stages,
            "donors": donors,
            "max_spots_per_stage": max_spots_per_stage,
            "seed": seed,
            "provider_cfg": provider_cfg,
        }
    )
    cache_dir = _mapping_cache_root("tacco") / cache_key
    cache_dir.mkdir(parents=True, exist_ok=True)
    return {
        "cache_dir": cache_dir,
        "reference_subset_h5ad": cache_dir / "snrna_reference_subset.h5ad",
        "spatial_subset_h5ad": cache_dir / "spatial_subset.h5ad",
        "spatial_h5ad": cache_dir / "spatial_projected.h5ad",
    }


def _load_tacco_mapping(
    cfg: Any,
    *,
    mapping_h5ad_path: Path,
    stages: list[str] | None,
    donors: list[str] | None,
    max_spots_per_stage: int | None,
    seed: int,
) -> SpatialMappingResult:
    cohort = load_luad_evo_spatial_mapping(
        cfg,
        mapping_h5ad_path=mapping_h5ad_path,
        composition_key="X_tacco_ct",
        columns_key="tacco_ct_columns",
        stages=stages,
        donors=donors,
        max_spots_per_stage=max_spots_per_stage,
        seed=seed,
    )
    return SpatialMappingResult(
        method="tacco",
        status="complete",
        provider_version=_provider_version("tacco"),
        execution_mode="load_precomputed",
        compositions=cohort.compositions,
        coords=cohort.coords,
        obs=cohort.obs,
        feature_names=cohort.feature_names,
        source_path=cohort.source_path,
        qc=summarize_mapping_qc(cohort.compositions),
        provenance={"mode": "loaded"},
        notes="Loaded TACCO spatial mapping output.",
    )


def run_tacco(
    cfg: Any,
    *,
    stages: list[str] | None = None,
    donors: list[str] | None = None,
    max_spots_per_stage: int | None = None,
    seed: int = 42,
) -> SpatialMappingResult:
    provider_cfg = dict(cfg.get("spatial_mapping", {})) if hasattr(cfg, "get") else dict(cfg["spatial_mapping"])
    execution_mode = str(provider_cfg.get("execution_mode", "rebuild_cached"))
    provider_version = _provider_version("tacco")

    precomputed_path = provider_cfg.get("precomputed_h5ad")
    if precomputed_path and Path(str(precomputed_path)).exists() and execution_mode == "load_precomputed":
        return _load_tacco_mapping(
            cfg,
            mapping_h5ad_path=Path(str(precomputed_path)),
            stages=stages,
            donors=donors,
            max_spots_per_stage=max_spots_per_stage,
            seed=seed,
        )

    if execution_mode not in {"rebuild_cached", "force_rebuild", "load_precomputed"}:
        raise ValueError(
            f"Unsupported TACCO execution_mode '{execution_mode}'. "
            "Use 'load_precomputed', 'rebuild_cached', or 'force_rebuild'."
        )

    try:
        paths = resolve_luad_evo_paths(cfg)
    except Exception as exc:
        return SpatialMappingResult(
            method="tacco",
            status="missing_inputs",
            provider_version=provider_version,
            execution_mode=execution_mode,
            provenance={"mode": "unavailable", "error": str(exc)},
            notes="TACCO inputs are not configured well enough to run this provider.",
        )
    if not paths.snrna_h5ad.exists() or not paths.spatial_h5ad.exists():
        return SpatialMappingResult(
            method="tacco",
            status="missing_inputs",
            provider_version=provider_version,
            execution_mode=execution_mode,
            provenance={
                "mode": "unavailable",
                "snrna_h5ad": str(paths.snrna_h5ad),
                "spatial_h5ad": str(paths.spatial_h5ad),
            },
            notes="TACCO requires both raw snRNA and spatial h5ad inputs.",
        )

    import tacco as tc

    cache = _tacco_cache_bundle(
        cfg,
        stages=stages,
        donors=donors,
        max_spots_per_stage=max_spots_per_stage,
        seed=seed,
    )
    needs_rebuild = execution_mode == "force_rebuild" or not cache["spatial_h5ad"].exists()
    reference_meta: dict[str, Any] | None = None
    spatial_meta: dict[str, Any] | None = None
    method_used = str(provider_cfg.get("annotation_method", "OT"))
    if needs_rebuild:
        label_col = str(provider_cfg.get("label_col", "hlca_label"))
        reference_meta = _write_reference_subset_h5ad(
            snrna_h5ad_path=paths.snrna_h5ad,
            subset_h5ad_path=cache["reference_subset_h5ad"],
            label_col=label_col,
            stages=stages,
            max_cells_per_label=int(provider_cfg.get("max_reference_cells_per_label", 1000)),
            seed=seed,
            fallback_labels_parquet_path=paths.hlca_labels_parquet,
            fallback_latent_h5ad_path=paths.snrna_latent_h5ad,
        )
        spatial_meta = _write_spatial_subset_h5ad(
            spatial_h5ad_path=paths.spatial_h5ad,
            subset_h5ad_path=cache["spatial_subset_h5ad"],
            stages=stages,
            donors=donors,
            max_spots_per_stage=max_spots_per_stage,
            seed=seed,
        )
        adata_ref = anndata.read_h5ad(cache["reference_subset_h5ad"])
        adata_sp = anndata.read_h5ad(cache["spatial_subset_h5ad"])
        try:
            annotation = tc.tl.annotate(
                adata_sp,
                adata_ref,
                annotation_key=label_col,
                result_key=None,
                method=method_used,
                verbose=int(provider_cfg.get("verbose", 0)),
            )
        except Exception:
            fallback_method = provider_cfg.get("fallback_annotation_method", "nnls")
            method_used = str(fallback_method)
            annotation = tc.tl.annotate(
                adata_sp,
                adata_ref,
                annotation_key=label_col,
                result_key=None,
                method=method_used,
                verbose=int(provider_cfg.get("verbose", 0)),
            )
        if isinstance(annotation, pd.DataFrame):
            compositions = annotation
        else:
            compositions = pd.DataFrame(annotation, index=adata_sp.obs_names.astype(str))
        adata_sp.obsm["X_tacco_ct"] = compositions.to_numpy(dtype=np.float32, copy=False)
        adata_sp.uns["tacco_ct_columns"] = [str(col) for col in compositions.columns]
        adata_sp.write_h5ad(cache["spatial_h5ad"], compression="lzf")

    result = _load_tacco_mapping(
        cfg,
        mapping_h5ad_path=cache["spatial_h5ad"],
        stages=stages,
        donors=donors,
        max_spots_per_stage=max_spots_per_stage,
        seed=seed,
    )
    return SpatialMappingResult(
        method="tacco",
        status=result.status,
        provider_version=provider_version,
        execution_mode=execution_mode,
        compositions=result.compositions,
        coords=result.coords,
        obs=result.obs,
        feature_names=result.feature_names,
        source_path=result.source_path,
        qc=result.qc,
        provenance={
            "mode": "rebuilt" if needs_rebuild else "cached",
            "cache_dir": str(cache["cache_dir"]),
            "annotation_method_used": method_used,
            "reference_subset_metadata": reference_meta,
            "spatial_subset_metadata": spatial_meta,
        },
        notes=(
            "TACCO mapping rebuilt from raw snRNA and spatial assets."
            if needs_rebuild
            else "TACCO mapping loaded from the reusable rebuild cache."
        ),
    )
