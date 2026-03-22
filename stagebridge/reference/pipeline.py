"""Main reference geometry pipeline for dual-reference embedding construction.

This module provides the high-level pipeline interface for running the
complete reference geometry workflow, integrating all components.

Supports both full runs and smoke mode for fast validation.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from stagebridge.logging_utils import get_logger
from stagebridge.geometry import EuclideanBackend, get_geometry_backend

log = get_logger(__name__)


@dataclass
class ReferenceGeometryConfig:
    """Configuration for reference geometry pipeline."""

    # Reference paths
    hlca_reference_path: str | None = None
    luca_reference_path: str | None = None

    # Query data path
    query_data_path: str | None = None

    # Mapping parameters
    mapping_method: Literal["knn_projection", "pca_projection", "scvi_query"] = "knn_projection"
    k_neighbors: int = 50
    hlca_latent_key: str = "X_scanvi_emb"
    luca_latent_key: str = "X_scVI"

    # Fusion parameters
    fusion_method: Literal["concat", "average", "weighted"] = "concat"
    normalize_fused: bool = True

    # Geometry backend
    geometry_backend: str = "euclidean"

    # Metadata columns
    cell_id_col: str | None = None
    donor_col: str = "donor_id"
    sample_col: str = "sample_id"
    stage_col: str = "stage"

    # Smoke mode
    smoke_mode: bool = False
    smoke_n_cells: int = 1000

    # Validation
    min_feature_overlap: float = 0.3
    held_out_donors: set[str] | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ReferenceGeometryConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ReferenceGeometryResult:
    """Result of reference geometry pipeline."""

    run_id: str
    success: bool
    output_dir: Path
    n_cells: int
    hlca_dim: int
    luca_dim: int
    fused_dim: int
    wall_time_seconds: float
    validation_status: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


def run_reference_pipeline(
    config: ReferenceGeometryConfig | dict[str, Any],
    query_data: Any | None = None,
    run_dir: str | Path | None = None,
    *,
    run_id: str | None = None,
    progress_callback: Any = None,
) -> ReferenceGeometryResult:
    """Run the complete reference geometry pipeline.

    This is the main entry point for reference geometry processing.

    Parameters
    ----------
    config : ReferenceGeometryConfig or dict
        Pipeline configuration
    query_data : AnnData, optional
        Query data. If None, loaded from config.query_data_path
    run_dir : str or Path, optional
        Output directory. If None, uses artifacts/runs/<run_id>/references/
    run_id : str, optional
        Run identifier. If None, generated from timestamp.
    progress_callback : callable, optional
        Callback for progress updates (receives step name and progress 0-1)

    Returns
    -------
    ReferenceGeometryResult
        Pipeline result with outputs and metrics
    """
    import anndata

    wall_t0 = time.perf_counter()

    # Parse config
    if isinstance(config, dict):
        config = ReferenceGeometryConfig.from_dict(config)

    # Generate run ID
    if run_id is None:
        run_id = f"ref_geo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Setup output directory
    if run_dir is None:
        run_dir = Path("artifacts/runs") / run_id / "references"
    else:
        run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    log.info("Starting reference geometry pipeline: run_id=%s", run_id)
    log.info("Output directory: %s", run_dir)

    errors = []
    warnings = []
    metrics = {}

    def _progress(step: str, pct: float) -> None:
        if progress_callback:
            progress_callback(step, pct)
        log.info("Progress: %s (%.0f%%)", step, pct * 100)

    try:
        # Step 1: Load query data
        _progress("load_query", 0.0)
        if query_data is None:
            if config.query_data_path is None:
                raise ValueError("No query data provided and query_data_path not set")
            query_data = anndata.read_h5ad(config.query_data_path)
            log.info("Loaded query data: %d cells, %d genes", query_data.n_obs, query_data.n_vars)

        # Apply smoke mode subsampling
        if config.smoke_mode:
            query_data = _subsample_for_smoke(query_data, config.smoke_n_cells)
            log.info("Smoke mode: subsampled to %d cells", query_data.n_obs)

        n_cells = query_data.n_obs
        _progress("load_query", 0.1)

        # Step 2: Load references
        _progress("load_references", 0.1)
        from stagebridge.reference.loaders import (
            load_hlca_reference,
            load_luca_reference,
            compute_feature_overlap,
        )

        hlca_ref = None
        luca_ref = None

        if config.hlca_reference_path:
            try:
                hlca_ref = load_hlca_reference(
                    config.hlca_reference_path,
                    latent_key=config.hlca_latent_key,
                )
                if not hlca_ref.is_valid:
                    warnings.extend(hlca_ref.validation_errors)
            except FileNotFoundError as e:
                log.warning("HLCA reference not found: %s", e)
                warnings.append(f"HLCA reference not found: {e}")

        if config.luca_reference_path:
            try:
                luca_ref = load_luca_reference(
                    config.luca_reference_path,
                    latent_key=config.luca_latent_key,
                )
                if not luca_ref.is_valid:
                    warnings.extend(luca_ref.validation_errors)
            except FileNotFoundError as e:
                log.warning("LuCa reference not found: %s", e)
                warnings.append(f"LuCa reference not found: {e}")

        if hlca_ref is None and luca_ref is None:
            raise ValueError("At least one reference (HLCA or LuCa) must be available")

        _progress("load_references", 0.2)

        # Step 3: Compute feature overlap
        _progress("feature_overlap", 0.2)
        feature_overlap = {}
        if hlca_ref:
            hlca_overlap = compute_feature_overlap(
                query_data,
                hlca_ref,
                min_overlap_threshold=config.min_feature_overlap,
            )
            feature_overlap["hlca"] = hlca_overlap.to_dict()
            if hlca_overlap.overlap_fraction < config.min_feature_overlap:
                warnings.append(f"Low HLCA feature overlap: {hlca_overlap.overlap_fraction:.1%}")

        if luca_ref:
            luca_overlap = compute_feature_overlap(
                query_data,
                luca_ref,
                min_overlap_threshold=config.min_feature_overlap,
            )
            feature_overlap["luca"] = luca_overlap.to_dict()
            if luca_overlap.overlap_fraction < config.min_feature_overlap:
                warnings.append(f"Low LuCa feature overlap: {luca_overlap.overlap_fraction:.1%}")

        _progress("feature_overlap", 0.3)

        # Step 4: Map to references
        _progress("map_query", 0.3)
        from stagebridge.reference.map_query import map_to_hlca, map_to_luca

        geometry = get_geometry_backend(config.geometry_backend)
        metadata_cols = {
            "cell_id": config.cell_id_col,
            "donor_id": config.donor_col,
            "sample_id": config.sample_col,
            "stage_id": config.stage_col,
        }

        hlca_result = None
        luca_result = None

        if hlca_ref:
            hlca_result = map_to_hlca(
                query_data,
                hlca_ref,
                method=config.mapping_method,
                latent_key=config.hlca_latent_key,
                k_neighbors=config.k_neighbors,
                held_out_donors=config.held_out_donors,
                geometry=geometry,
                metadata_cols=metadata_cols,
            )
            log.info(
                "HLCA mapping: %d cells -> %d dims", hlca_result.n_cells, hlca_result.latent_dim
            )

        _progress("map_query", 0.5)

        if luca_ref:
            luca_result = map_to_luca(
                query_data,
                luca_ref,
                method=config.mapping_method,
                latent_key=config.luca_latent_key,
                k_neighbors=config.k_neighbors,
                held_out_donors=config.held_out_donors,
                geometry=geometry,
                metadata_cols=metadata_cols,
            )
            log.info(
                "LuCa mapping: %d cells -> %d dims", luca_result.n_cells, luca_result.latent_dim
            )

        _progress("map_query", 0.6)

        # Step 5: Compute confidence
        _progress("confidence", 0.6)
        from stagebridge.reference.confidence import (
            compute_dual_confidence,
            compute_hlca_confidence,
            detect_mapping_collapse,
            detect_nan_embeddings,
        )

        # Check for mapping issues
        if hlca_result:
            collapse_check = detect_mapping_collapse(hlca_result)
            if collapse_check["is_collapsed"]:
                errors.append("HLCA mapping collapsed - all cells at same point")
            nan_check = detect_nan_embeddings(hlca_result)
            if nan_check["has_nan"]:
                warnings.append(f"HLCA has {nan_check['total_nan_count']} NaN values")

        if luca_result:
            collapse_check = detect_mapping_collapse(luca_result)
            if collapse_check["is_collapsed"]:
                errors.append("LuCa mapping collapsed - all cells at same point")
            nan_check = detect_nan_embeddings(luca_result)
            if nan_check["has_nan"]:
                warnings.append(f"LuCa has {nan_check['total_nan_count']} NaN values")

        # Compute confidence scores
        if hlca_result and luca_result:
            confidence = compute_dual_confidence(hlca_result, luca_result)
        elif hlca_result:
            hlca_conf = compute_hlca_confidence(hlca_result)
            confidence = type(
                "Conf",
                (),
                {
                    "hlca_confidence": hlca_conf,
                    "luca_confidence": np.zeros_like(hlca_conf),
                    "cell_ids": hlca_result.cell_ids,
                    "to_dataframe": lambda: pd.DataFrame(
                        {
                            "cell_id": hlca_result.cell_ids,
                            "hlca_confidence": hlca_conf,
                            "luca_confidence": np.zeros_like(hlca_conf),
                        }
                    ),
                },
            )()
        else:
            luca_conf = compute_hlca_confidence(luca_result)  # Same method
            confidence = type(
                "Conf",
                (),
                {
                    "hlca_confidence": np.zeros_like(luca_conf),
                    "luca_confidence": luca_conf,
                    "cell_ids": luca_result.cell_ids,
                    "to_dataframe": lambda: pd.DataFrame(
                        {
                            "cell_id": luca_result.cell_ids,
                            "hlca_confidence": np.zeros_like(luca_conf),
                            "luca_confidence": luca_conf,
                        }
                    ),
                },
            )()

        _progress("confidence", 0.7)

        # Step 6: Fuse embeddings
        _progress("fuse", 0.7)
        from stagebridge.reference.fuse import fuse_dual_reference, fuse_single_reference

        if hlca_result and luca_result:
            fused = fuse_dual_reference(
                hlca_result,
                luca_result,
                method=config.fusion_method,
                hlca_confidence=confidence.hlca_confidence,
                luca_confidence=confidence.luca_confidence,
                normalize=config.normalize_fused,
            )
        elif hlca_result:
            fused = fuse_single_reference(hlca_result, "hlca", normalize=config.normalize_fused)
        else:
            fused = fuse_single_reference(luca_result, "luca", normalize=config.normalize_fused)

        log.info("Fused embedding: %d cells, %d dims", fused.n_cells, fused.fused_dim)
        _progress("fuse", 0.8)

        # Step 7: Export outputs
        _progress("export", 0.8)
        from stagebridge.reference.schema import (
            export_reference_outputs,
            create_manifest,
            validate_output_integrity,
        )

        # Create DataFrames
        if hlca_result:
            hlca_df = hlca_result.to_dataframe(prefix="hlca_")
        else:
            hlca_df = _create_dummy_embedding_df(fused.cell_ids, 0, "hlca_", fused)

        if luca_result:
            luca_df = luca_result.to_dataframe(prefix="luca_")
        else:
            luca_df = _create_dummy_embedding_df(fused.cell_ids, 0, "luca_", fused)

        fused_df = fused.to_dataframe()
        confidence_df = confidence.to_dataframe()

        # Create manifest
        manifest = create_manifest(
            run_id=run_id,
            hlca_dim=hlca_result.latent_dim if hlca_result else 0,
            luca_dim=luca_result.latent_dim if luca_result else 0,
            fused_dim=fused.fused_dim,
            n_cells=n_cells,
            fusion_method=config.fusion_method,
            mapping_method=config.mapping_method,
            hlca_path=str(config.hlca_reference_path or ""),
            luca_path=str(config.luca_reference_path) if config.luca_reference_path else None,
            query_path=str(config.query_data_path or "in_memory"),
            geometry=config.geometry_backend,
            parameters={
                "k_neighbors": config.k_neighbors,
                "smoke_mode": config.smoke_mode,
                "normalize_fused": config.normalize_fused,
            },
        )

        # Export
        export_reference_outputs(
            hlca_df=hlca_df,
            luca_df=luca_df,
            fused_df=fused_df,
            confidence_df=confidence_df,
            manifest=manifest,
            feature_overlap=feature_overlap,
            output_dir=run_dir,
        )

        _progress("export", 0.9)

        # Step 8: Validate outputs
        _progress("validate", 0.9)
        validation = validate_output_integrity(run_dir)
        validation_status = "pass" if validation["valid"] else "fail"

        if not validation["valid"]:
            errors.extend(validation["errors"])
        warnings.extend(validation.get("warnings", []))

        metrics = {
            "n_cells": n_cells,
            "hlca_dim": hlca_result.latent_dim if hlca_result else 0,
            "luca_dim": luca_result.latent_dim if luca_result else 0,
            "fused_dim": fused.fused_dim,
            "hlca_mean_confidence": float(np.mean(confidence.hlca_confidence)),
            "luca_mean_confidence": float(np.mean(confidence.luca_confidence)),
            "feature_overlap": feature_overlap,
        }

        _progress("validate", 1.0)

        wall_time = time.perf_counter() - wall_t0
        log.info(
            "Pipeline complete: %d cells, wall_time=%.1fs, status=%s",
            n_cells,
            wall_time,
            validation_status,
        )

        return ReferenceGeometryResult(
            run_id=run_id,
            success=len(errors) == 0,
            output_dir=run_dir,
            n_cells=n_cells,
            hlca_dim=hlca_result.latent_dim if hlca_result else 0,
            luca_dim=luca_result.latent_dim if luca_result else 0,
            fused_dim=fused.fused_dim,
            wall_time_seconds=wall_time,
            validation_status=validation_status,
            errors=errors,
            warnings=warnings,
            metrics=metrics,
        )

    except Exception as e:
        wall_time = time.perf_counter() - wall_t0
        log.exception("Pipeline failed: %s", e)
        return ReferenceGeometryResult(
            run_id=run_id,
            success=False,
            output_dir=run_dir,
            n_cells=0,
            hlca_dim=0,
            luca_dim=0,
            fused_dim=0,
            wall_time_seconds=wall_time,
            validation_status="error",
            errors=[str(e)],
            warnings=warnings,
            metrics=metrics,
        )


def _subsample_for_smoke(
    adata: Any,
    n_cells: int,
    seed: int = 42,
) -> Any:
    """Subsample AnnData for smoke mode."""
    if adata.n_obs <= n_cells:
        return adata

    rng = np.random.default_rng(seed)
    idx = rng.choice(adata.n_obs, size=n_cells, replace=False)
    idx = np.sort(idx)

    return adata[idx].copy()


def _create_dummy_embedding_df(
    cell_ids: np.ndarray,
    latent_dim: int,
    prefix: str,
    fused: Any,
) -> pd.DataFrame:
    """Create dummy embedding DataFrame when reference not available."""
    df = pd.DataFrame(
        {
            "cell_id": cell_ids,
            "donor_id": fused.donor_ids,
            "sample_id": fused.sample_ids,
            "stage_id": fused.stage_ids,
        }
    )

    # Add zero-filled latent columns (at least one)
    dim = max(latent_dim, 1)
    for i in range(dim):
        df[f"{prefix}latent_{i}"] = 0.0

    return df


def run_smoke_test(
    config: ReferenceGeometryConfig | dict[str, Any],
    query_data: Any | None = None,
) -> ReferenceGeometryResult:
    """Run a fast smoke test of the reference pipeline.

    Parameters
    ----------
    config : ReferenceGeometryConfig or dict
        Pipeline configuration (smoke_mode will be forced True)
    query_data : AnnData, optional
        Query data

    Returns
    -------
    ReferenceGeometryResult
        Pipeline result
    """
    if isinstance(config, dict):
        config = ReferenceGeometryConfig.from_dict(config)

    # Force smoke mode
    config.smoke_mode = True
    config.smoke_n_cells = min(config.smoke_n_cells, 1000)

    log.info("Running smoke test with max %d cells", config.smoke_n_cells)

    return run_reference_pipeline(
        config,
        query_data=query_data,
        run_id=f"smoke_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
