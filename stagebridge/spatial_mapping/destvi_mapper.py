"""DestVI provider wrapper for raw snRNA -> spatial mapping."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import anndata
import numpy as np
import pandas as pd
import scvi
import scipy.sparse as sp
from scvi.model import CondSCVI, DestVI

from stagebridge.data.luad_evo.metadata import resolve_luad_evo_paths
from stagebridge.data.luad_evo.visium import load_luad_evo_spatial_mapping
from stagebridge.spatial_mapping.base import SpatialMappingResult
from stagebridge.spatial_mapping.qc import summarize_mapping_qc
from stagebridge.spatial_mapping.tacco_mapper import _write_reference_subset_h5ad
from stagebridge.spatial_mapping.tangram_mapper import (
    _mapping_cache_root,
    _provider_version,
    _stable_hash,
    _subset_training_genes,
    _sorted_shared_genes,
    _write_spatial_subset_h5ad,
)


def _destvi_cache_bundle(
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
            "method": "destvi",
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
    cache_dir = _mapping_cache_root("destvi") / cache_key
    cache_dir.mkdir(parents=True, exist_ok=True)
    return {
        "cache_dir": cache_dir,
        "reference_subset_h5ad": cache_dir / "snrna_reference_subset.h5ad",
        "spatial_subset_h5ad": cache_dir / "spatial_subset.h5ad",
        "spatial_h5ad": cache_dir / "spatial_projected.h5ad",
        "report_json": cache_dir / "report.json",
    }


def _load_destvi_mapping(
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
        provider_version=_provider_version("scvi-tools"),
        execution_mode="load_precomputed",
        compositions=cohort.compositions,
        coords=cohort.coords,
        obs=cohort.obs,
        feature_names=cohort.feature_names,
        source_path=cohort.source_path,
        qc=summarize_mapping_qc(cohort.compositions),
        provenance={"mode": "loaded"},
        notes="Loaded DestVI spatial mapping output.",
    )


def _prepare_destvi_inputs(
    *,
    reference_subset_h5ad: Path,
    spatial_subset_h5ad: Path,
    label_col: str,
    max_training_genes: int,
    min_shared_genes: int,
) -> tuple[anndata.AnnData, anndata.AnnData, list[str]]:
    adata_sc = anndata.read_h5ad(reference_subset_h5ad)
    adata_sp = anndata.read_h5ad(spatial_subset_h5ad)
    adata_sc.var_names_make_unique()
    adata_sp.var_names_make_unique()

    shared_genes = _sorted_shared_genes(adata_sc.var_names, adata_sp.var_names)
    if len(shared_genes) < min_shared_genes:
        raise RuntimeError(
            f"Only {len(shared_genes)} shared genes between snRNA and spatial, "
            f"below min_shared_genes={min_shared_genes}."
        )
    training_genes = _subset_training_genes(shared_genes, max_training_genes=max_training_genes)

    adata_sc = adata_sc[:, training_genes].copy()
    adata_sp = adata_sp[:, training_genes].copy()
    if "counts" not in adata_sc.layers:
        adata_sc.layers["counts"] = (
            adata_sc.X.astype(np.float32)
            if sp.issparse(adata_sc.X)
            else np.asarray(adata_sc.X, dtype=np.float32)
        )
    if "counts" not in adata_sp.layers:
        adata_sp.layers["counts"] = (
            adata_sp.X.astype(np.float32)
            if sp.issparse(adata_sp.X)
            else np.asarray(adata_sp.X, dtype=np.float32)
        )

    adata_sc.obs[label_col] = adata_sc.obs[label_col].astype(str).astype("category")
    return adata_sc, adata_sp, training_genes


def _run_destvi_training(
    *,
    reference_subset_h5ad: Path,
    spatial_subset_h5ad: Path,
    output_h5ad_path: Path,
    report_json_path: Path,
    provider_cfg: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    scvi.settings.seed = int(seed)
    scvi.settings.dl_num_workers = 0

    label_col = str(provider_cfg.get("label_col", "hlca_label"))
    max_training_genes = int(provider_cfg.get("max_training_genes", 1000))
    min_shared_genes = int(provider_cfg.get("min_shared_genes", 200))
    batch_size = int(provider_cfg.get("batch_size", 128))
    device = str(provider_cfg.get("device", "cpu"))
    accelerator = "gpu" if device == "cuda" else "cpu"
    devices = 1

    condscvi_cfg = dict(provider_cfg.get("condscvi", {}))
    destvi_cfg = dict(provider_cfg.get("destvi", {}))
    condscvi_epochs = int(condscvi_cfg.get("max_epochs", 2))
    destvi_epochs = int(destvi_cfg.get("max_epochs", 2))

    adata_sc, adata_sp, training_genes = _prepare_destvi_inputs(
        reference_subset_h5ad=reference_subset_h5ad,
        spatial_subset_h5ad=spatial_subset_h5ad,
        label_col=label_col,
        max_training_genes=max_training_genes,
        min_shared_genes=min_shared_genes,
    )

    CondSCVI.setup_anndata(adata_sc, layer="counts", labels_key=label_col)
    rna_model = CondSCVI(
        adata_sc,
        n_hidden=int(condscvi_cfg.get("n_hidden", 64)),
        n_latent=int(condscvi_cfg.get("n_latent", 5)),
        n_layers=int(condscvi_cfg.get("n_layers", 1)),
        weight_obs=bool(condscvi_cfg.get("weight_obs", False)),
        dropout_rate=float(condscvi_cfg.get("dropout_rate", 0.05)),
        prior=str(condscvi_cfg.get("prior", "mog")),
        num_classes_mog=int(condscvi_cfg.get("num_classes_mog", 8)),
    )
    rna_model.train(
        max_epochs=condscvi_epochs,
        accelerator=accelerator,
        devices=devices,
        batch_size=batch_size,
        plan_kwargs={"lr": float(condscvi_cfg.get("lr", 1e-3))},
        enable_progress_bar=bool(provider_cfg.get("show_progress", False)),
    )

    destvi_model = DestVI.from_rna_model(
        adata_sp,
        rna_model,
        vamp_prior_p=int(destvi_cfg.get("vamp_prior_p", 8)),
    )
    destvi_model.train(
        max_epochs=destvi_epochs,
        accelerator=accelerator,
        devices=devices,
        batch_size=batch_size,
        plan_kwargs={"lr": float(destvi_cfg.get("lr", 3e-3))},
        enable_progress_bar=bool(provider_cfg.get("show_progress", False)),
    )
    proportions = destvi_model.get_proportions(keep_additional=False, normalize=True)
    if isinstance(proportions, pd.DataFrame):
        prop_df = proportions.copy()
    else:
        prop_df = pd.DataFrame(proportions, index=adata_sp.obs_names.astype(str))
    adata_sp.obsm["X_destvi_ct"] = prop_df.to_numpy(dtype=np.float32, copy=False)
    adata_sp.uns["destvi_ct_columns"] = [str(col) for col in prop_df.columns]
    adata_sp.write_h5ad(output_h5ad_path, compression="lzf")

    report = {
        "label_col": label_col,
        "n_spots": int(adata_sp.n_obs),
        "n_reference_cells": int(adata_sc.n_obs),
        "n_features": int(adata_sp.n_vars),
        "n_labels": int(adata_sc.obs[label_col].nunique()),
        "training_genes": len(training_genes),
        "condscvi_epochs": condscvi_epochs,
        "destvi_epochs": destvi_epochs,
        "batch_size": batch_size,
        "device": device,
    }
    report_json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def run_destvi(
    cfg: Any,
    *,
    stages: list[str] | None = None,
    donors: list[str] | None = None,
    max_spots_per_stage: int | None = None,
    seed: int = 42,
) -> SpatialMappingResult:
    provider_cfg = dict(cfg.get("spatial_mapping", {})) if hasattr(cfg, "get") else dict(cfg["spatial_mapping"])
    execution_mode = str(provider_cfg.get("execution_mode", "rebuild_cached"))
    provider_version = _provider_version("scvi-tools")
    precomputed_path = provider_cfg.get("precomputed_h5ad")

    if precomputed_path and Path(str(precomputed_path)).exists() and execution_mode == "load_precomputed":
        return _load_destvi_mapping(
            cfg,
            mapping_h5ad_path=Path(str(precomputed_path)),
            stages=stages,
            donors=donors,
            max_spots_per_stage=max_spots_per_stage,
            seed=seed,
        )

    if execution_mode not in {"rebuild_cached", "force_rebuild", "load_precomputed"}:
        raise ValueError(
            f"Unsupported DestVI execution_mode '{execution_mode}'. "
            "Use 'load_precomputed', 'rebuild_cached', or 'force_rebuild'."
        )

    try:
        paths = resolve_luad_evo_paths(cfg)
    except Exception as exc:
        return SpatialMappingResult(
            method="destvi",
            status="missing_inputs",
            provider_version=provider_version,
            execution_mode=execution_mode,
            provenance={"mode": "unavailable", "error": str(exc)},
            notes="DestVI inputs are not configured well enough to run this provider.",
        )

    if not paths.snrna_h5ad.exists() or not paths.spatial_h5ad.exists():
        return SpatialMappingResult(
            method="destvi",
            status="missing_inputs",
            provider_version=provider_version,
            execution_mode=execution_mode,
            provenance={
                "mode": "unavailable",
                "snrna_h5ad": str(paths.snrna_h5ad),
                "spatial_h5ad": str(paths.spatial_h5ad),
            },
            notes="DestVI requires both raw snRNA and spatial h5ad inputs.",
        )

    cache = _destvi_cache_bundle(
        cfg,
        stages=stages,
        donors=donors,
        max_spots_per_stage=max_spots_per_stage,
        seed=seed,
    )
    needs_rebuild = execution_mode == "force_rebuild" or not cache["spatial_h5ad"].exists()
    reference_meta: dict[str, Any] | None = None
    spatial_meta: dict[str, Any] | None = None
    report: dict[str, Any] | None = None
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
        report = _run_destvi_training(
            reference_subset_h5ad=cache["reference_subset_h5ad"],
            spatial_subset_h5ad=cache["spatial_subset_h5ad"],
            output_h5ad_path=cache["spatial_h5ad"],
            report_json_path=cache["report_json"],
            provider_cfg=provider_cfg,
            seed=seed,
        )
    elif cache["report_json"].exists():
        report = json.loads(cache["report_json"].read_text(encoding="utf-8"))

    result = _load_destvi_mapping(
        cfg,
        mapping_h5ad_path=cache["spatial_h5ad"],
        stages=stages,
        donors=donors,
        max_spots_per_stage=max_spots_per_stage,
        seed=seed,
    )
    return SpatialMappingResult(
        method="destvi",
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
            "reference_subset_metadata": reference_meta,
            "spatial_subset_metadata": spatial_meta,
            "training_report": report,
        },
        notes=(
            "DestVI mapping rebuilt from raw snRNA and spatial assets."
            if needs_rebuild
            else "DestVI mapping loaded from the reusable rebuild cache."
        ),
    )
