"""
Main data pipeline orchestration for StageBridge.

This module provides the entry point for running the complete data pipeline:
1. Raw data ingestion
2. Metadata harmonization
3. Quality control
4. Normalization and feature preparation
5. Canonical export

Integrates with stagebridge/orchestration/ for progress tracking.

Usage:
    from stagebridge.data.pipeline import run_data_pipeline, DataPipelineConfig

    config = DataPipelineConfig(
        dataset_name="luad_evo",
        data_root="/path/to/data",
        output_dir="/path/to/output",
    )
    result = run_data_pipeline(config)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from stagebridge.logging_utils import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class DataPipelineConfig:
    """Configuration for the data pipeline."""

    # Required
    dataset_name: str
    data_root: Path | str
    output_dir: Path | str

    # Optional paths
    run_dir: Path | str | None = None  # For run-specific artifacts

    # Processing options
    modality: Literal["snrna", "spatial", "both"] = "both"
    normalize_method: str = "log1p"
    target_sum: float = 1e4
    n_hvg: int = 2000
    hvg_flavor: str = "seurat_v3"

    # QC thresholds (None = use dataset defaults)
    min_counts: int | None = None
    max_counts: int | None = None
    min_genes: int | None = None
    max_genes: int | None = None
    max_mito_pct: float | None = None

    # Smoke test mode
    smoke_mode: bool = False
    smoke_n_donors: int = 2
    smoke_n_cells: int = 1000

    # Execution options
    skip_qc: bool = False
    skip_normalization: bool = False
    skip_hvg: bool = False
    skip_export: bool = False
    generate_figures: bool = True
    force_rerun: bool = False

    # Metadata columns
    donor_column: str = "donor_id"
    sample_column: str = "sample_id"
    stage_column: str = "stage"

    def __post_init__(self) -> None:
        """Convert paths to Path objects."""
        self.data_root = Path(self.data_root)
        self.output_dir = Path(self.output_dir)
        if self.run_dir is not None:
            self.run_dir = Path(self.run_dir)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "dataset_name": self.dataset_name,
            "data_root": str(self.data_root),
            "output_dir": str(self.output_dir),
            "run_dir": str(self.run_dir) if self.run_dir else None,
            "modality": self.modality,
            "normalize_method": self.normalize_method,
            "target_sum": self.target_sum,
            "n_hvg": self.n_hvg,
            "hvg_flavor": self.hvg_flavor,
            "min_counts": self.min_counts,
            "max_counts": self.max_counts,
            "min_genes": self.min_genes,
            "max_genes": self.max_genes,
            "max_mito_pct": self.max_mito_pct,
            "smoke_mode": self.smoke_mode,
            "smoke_n_donors": self.smoke_n_donors,
            "smoke_n_cells": self.smoke_n_cells,
            "skip_qc": self.skip_qc,
            "skip_normalization": self.skip_normalization,
            "skip_hvg": self.skip_hvg,
            "skip_export": self.skip_export,
            "generate_figures": self.generate_figures,
            "force_rerun": self.force_rerun,
        }


@dataclass
class DataPipelineResult:
    """Result of data pipeline execution."""

    config: DataPipelineConfig
    status: Literal["success", "partial", "failed"]
    started_at: str
    completed_at: str | None = None
    duration_seconds: float | None = None

    # Output paths
    cells_h5ad: Path | None = None
    spatial_h5ad: Path | None = None
    export_result_path: Path | None = None
    qc_result_path: Path | None = None

    # Statistics
    n_cells_input: int = 0
    n_cells_output: int = 0
    n_spots_input: int = 0
    n_spots_output: int = 0
    n_genes: int = 0
    n_hvgs: int = 0
    n_donors: int = 0
    n_stages: int = 0

    # Errors and warnings
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stage_results: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Whether pipeline completed successfully."""
        return self.status == "success"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "config": self.config.to_dict(),
            "status": self.status,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_seconds": self.duration_seconds,
            "cells_h5ad": str(self.cells_h5ad) if self.cells_h5ad else None,
            "spatial_h5ad": str(self.spatial_h5ad) if self.spatial_h5ad else None,
            "export_result_path": str(self.export_result_path)
            if self.export_result_path
            else None,
            "qc_result_path": str(self.qc_result_path) if self.qc_result_path else None,
            "n_cells_input": self.n_cells_input,
            "n_cells_output": self.n_cells_output,
            "n_spots_input": self.n_spots_input,
            "n_spots_output": self.n_spots_output,
            "n_genes": self.n_genes,
            "n_hvgs": self.n_hvgs,
            "n_donors": self.n_donors,
            "n_stages": self.n_stages,
            "errors": self.errors,
            "warnings": self.warnings,
            "stage_results": self.stage_results,
        }

    def save(self, path: Path | str) -> None:
        """Save result to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------


def _run_ingest_stage(
    config: DataPipelineConfig,
    result: DataPipelineResult,
) -> tuple[Any, Any]:  # (cells_adata, spatial_adata)
    """Run data ingestion stage.

    Returns raw AnnData objects for cells and spatial.
    """
    import anndata

    cells_adata = None
    spatial_adata = None

    log.info("=== Stage 1: Data Ingestion ===")

    # Try to load from canonical paths
    cells_path = config.data_root / "processed" / "anndata" / "snrna_merged.h5ad"
    spatial_path = config.data_root / "processed" / "anndata" / "spatial_merged.h5ad"

    if config.modality in ("snrna", "both"):
        if cells_path.exists():
            log.info("Loading cells from %s", cells_path)
            cells_adata = anndata.read_h5ad(cells_path)
            result.n_cells_input = cells_adata.n_obs
            log.info("Loaded %d cells, %d genes", cells_adata.n_obs, cells_adata.n_vars)
        else:
            result.warnings.append(f"Cells file not found: {cells_path}")

    if config.modality in ("spatial", "both"):
        if spatial_path.exists():
            log.info("Loading spatial from %s", spatial_path)
            spatial_adata = anndata.read_h5ad(spatial_path)
            result.n_spots_input = spatial_adata.n_obs
            log.info("Loaded %d spots, %d genes", spatial_adata.n_obs, spatial_adata.n_vars)
        else:
            result.warnings.append(f"Spatial file not found: {spatial_path}")

    # Apply smoke mode subset
    if config.smoke_mode:
        log.info(
            "Smoke mode: subsetting to %d donors, %d cells max",
            config.smoke_n_donors,
            config.smoke_n_cells,
        )

        if cells_adata is not None:
            cells_adata = _apply_smoke_subset(
                cells_adata,
                config.donor_column,
                config.smoke_n_donors,
                config.smoke_n_cells,
            )
            result.n_cells_input = cells_adata.n_obs

        if spatial_adata is not None:
            spatial_adata = _apply_smoke_subset(
                spatial_adata,
                config.donor_column,
                config.smoke_n_donors,
                config.smoke_n_cells,
            )
            result.n_spots_input = spatial_adata.n_obs

    result.stage_results["ingest"] = {
        "status": "completed",
        "n_cells": result.n_cells_input,
        "n_spots": result.n_spots_input,
    }

    return cells_adata, spatial_adata


def _apply_smoke_subset(
    adata: Any,  # AnnData
    donor_column: str,
    n_donors: int,
    n_cells: int,
) -> Any:  # AnnData
    """Apply smoke test subset to AnnData."""
    import numpy as np

    if donor_column not in adata.obs.columns:
        # Just take first n_cells
        return adata[:n_cells, :].copy()

    # Select first n_donors
    donors = sorted(adata.obs[donor_column].astype(str).unique())[:n_donors]
    donor_mask = adata.obs[donor_column].astype(str).isin(donors)
    adata = adata[donor_mask, :].copy()

    # Limit cells per donor
    if adata.n_obs > n_cells:
        cells_per_donor = n_cells // len(donors)
        indices = []
        for donor in donors:
            donor_mask = adata.obs[donor_column].astype(str) == donor
            donor_indices = np.where(donor_mask)[0]
            indices.extend(donor_indices[:cells_per_donor].tolist())
        adata = adata[sorted(indices), :].copy()

    log.info("Smoke subset: %d cells, %d donors", adata.n_obs, len(donors))
    return adata


def _run_harmonize_stage(
    cells_adata: Any | None,  # AnnData
    spatial_adata: Any | None,  # AnnData
    config: DataPipelineConfig,
    result: DataPipelineResult,
) -> tuple[Any, Any]:  # (cells_adata, spatial_adata)
    """Run metadata harmonization stage."""
    from stagebridge.data.common.harmonize import (
        canonicalize_gene_symbols,
        ensure_required_obs_fields,
    )

    log.info("=== Stage 2: Metadata Harmonization ===")

    if cells_adata is not None:
        canonicalize_gene_symbols(cells_adata)
        ensure_required_obs_fields(cells_adata)
        cells_adata.obs["modality"] = "snrna"
        log.info("Harmonized cells metadata")

    if spatial_adata is not None:
        canonicalize_gene_symbols(spatial_adata)
        ensure_required_obs_fields(spatial_adata)
        spatial_adata.obs["modality"] = "spatial"
        log.info("Harmonized spatial metadata")

    result.stage_results["harmonize"] = {"status": "completed"}
    return cells_adata, spatial_adata


def _run_qc_stage(
    cells_adata: Any | None,  # AnnData
    spatial_adata: Any | None,  # AnnData
    config: DataPipelineConfig,
    result: DataPipelineResult,
) -> tuple[Any, Any]:  # (cells_adata, spatial_adata)
    """Run QC filtering stage."""
    from stagebridge.data.qc import QCConfig, run_qc, generate_qc_figures

    log.info("=== Stage 3: Quality Control ===")

    qc_dir = config.output_dir / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)

    cells_qc_result = None
    spatial_qc_result = None

    if cells_adata is not None:
        # Build QC config
        qc_config = QCConfig.default_snrna()
        if config.min_counts is not None:
            qc_config.min_counts = config.min_counts
        if config.max_counts is not None:
            qc_config.max_counts = config.max_counts
        if config.min_genes is not None:
            qc_config.min_genes = config.min_genes
        if config.max_genes is not None:
            qc_config.max_genes = config.max_genes
        if config.max_mito_pct is not None:
            qc_config.max_mito_pct = config.max_mito_pct

        cells_adata, cells_qc_result = run_qc(
            cells_adata,
            qc_config,
            donor_column=config.donor_column,
            stage_column=config.stage_column,
        )

        result.n_cells_output = cells_adata.n_obs
        result.n_genes = cells_adata.n_vars

        # Save QC result
        cells_qc_result.save(qc_dir / "cells_qc_result.json")
        result.qc_result_path = qc_dir / "cells_qc_result.json"

        # Generate figures
        if config.generate_figures:
            generate_qc_figures(
                cells_adata,
                cells_qc_result,
                qc_dir,
                donor_column=config.donor_column,
                stage_column=config.stage_column,
            )

    if spatial_adata is not None:
        qc_config = QCConfig.default_spatial()
        if config.min_counts is not None:
            qc_config.min_counts = config.min_counts
        if config.max_mito_pct is not None:
            qc_config.max_mito_pct = config.max_mito_pct

        spatial_adata, spatial_qc_result = run_qc(
            spatial_adata,
            qc_config,
            donor_column=config.donor_column,
            stage_column=config.stage_column,
        )

        result.n_spots_output = spatial_adata.n_obs

        # Save QC result
        spatial_qc_result.save(qc_dir / "spatial_qc_result.json")

        # Generate figures
        if config.generate_figures:
            generate_qc_figures(
                spatial_adata,
                spatial_qc_result,
                qc_dir / "spatial",
                donor_column=config.donor_column,
                stage_column=config.stage_column,
            )

    result.stage_results["qc"] = {
        "status": "completed",
        "n_cells_post": result.n_cells_output,
        "n_spots_post": result.n_spots_output,
        "cells_qc": cells_qc_result.to_dict() if cells_qc_result else None,
        "spatial_qc": spatial_qc_result.to_dict() if spatial_qc_result else None,
    }

    return cells_adata, spatial_adata


def _run_normalize_stage(
    cells_adata: Any | None,  # AnnData
    spatial_adata: Any | None,  # AnnData
    config: DataPipelineConfig,
    result: DataPipelineResult,
) -> tuple[Any, Any]:  # (cells_adata, spatial_adata)
    """Run normalization and feature preparation stage."""
    from stagebridge.data.normalize import normalize_counts, compute_hvgs

    log.info("=== Stage 4: Normalization ===")

    hvgs = []

    if cells_adata is not None:
        # Normalize
        normalize_counts(
            cells_adata,
            method=config.normalize_method,
            target_sum=config.target_sum,
            preserve_raw=True,
        )
        log.info("Normalized cells data")

        # HVGs
        if not config.skip_hvg:
            hvgs = compute_hvgs(
                cells_adata,
                n_hvg=config.n_hvg,
                flavor=config.hvg_flavor,
                layer="counts",
            )
            result.n_hvgs = len(hvgs)
            log.info("Selected %d HVGs", len(hvgs))

    if spatial_adata is not None:
        normalize_counts(
            spatial_adata,
            method=config.normalize_method,
            target_sum=config.target_sum,
            preserve_raw=True,
        )
        log.info("Normalized spatial data")

    result.stage_results["normalize"] = {
        "status": "completed",
        "method": config.normalize_method,
        "n_hvgs": result.n_hvgs,
    }

    return cells_adata, spatial_adata


def _run_export_stage(
    cells_adata: Any | None,  # AnnData
    spatial_adata: Any | None,  # AnnData
    config: DataPipelineConfig,
    result: DataPipelineResult,
) -> None:
    """Run canonical export stage."""
    from stagebridge.data.export import export_canonical_dataset
    from stagebridge.data.normalize import generate_feature_spec

    log.info("=== Stage 5: Export ===")

    # Generate feature spec
    feature_spec = None
    if cells_adata is not None:
        feature_spec = generate_feature_spec(cells_adata)

    # Export
    export_result = export_canonical_dataset(
        adata=cells_adata,
        spatial_adata=spatial_adata,
        output_dir=config.output_dir,
        dataset_name=config.dataset_name,
        feature_spec=feature_spec,
        donor_column=config.donor_column,
        sample_column=config.sample_column,
        stage_column=config.stage_column,
    )

    result.export_result_path = config.output_dir / "export_result.json"
    result.cells_h5ad = config.output_dir / "cells.h5ad" if cells_adata is not None else None
    result.spatial_h5ad = config.output_dir / "spatial.h5ad" if spatial_adata is not None else None

    # Update counts
    if cells_adata is not None:
        result.n_donors = cells_adata.obs[config.donor_column].nunique()
        result.n_stages = cells_adata.obs[config.stage_column].nunique()

    result.stage_results["export"] = {
        "status": "completed",
        "files_written": [str(p) for p in export_result.files_written],
    }


# ---------------------------------------------------------------------------
# Main pipeline function
# ---------------------------------------------------------------------------


def run_data_pipeline(
    config: DataPipelineConfig | dict[str, Any],
    *,
    run_dir: Path | str | None = None,
    artifact_registry: Any = None,  # ArtifactRegistry
    progress_callback: Any = None,  # Callable
) -> DataPipelineResult:
    """Run the complete data pipeline.

    Stages:
    1. Ingest: Load raw data
    2. Harmonize: Normalize metadata
    3. QC: Filter low-quality cells/spots
    4. Normalize: Normalize expression, select HVGs
    5. Export: Write canonical outputs

    Parameters
    ----------
    config : DataPipelineConfig or dict
        Pipeline configuration.
    run_dir : Path, optional
        Run-specific output directory.
    artifact_registry : ArtifactRegistry, optional
        Artifact registry for tracking outputs.
    progress_callback : callable, optional
        Callback for progress updates.

    Returns
    -------
    DataPipelineResult
        Pipeline execution result.
    """
    if isinstance(config, dict):
        config = DataPipelineConfig(**config)

    # Initialize result
    result = DataPipelineResult(
        config=config,
        status="failed",
        started_at=datetime.now(timezone.utc).isoformat(),
    )

    log.info("Starting data pipeline for dataset '%s'", config.dataset_name)
    log.info("Data root: %s", config.data_root)
    log.info("Output dir: %s", config.output_dir)

    if config.smoke_mode:
        log.info(
            "SMOKE MODE: %d donors, %d cells max", config.smoke_n_donors, config.smoke_n_cells
        )

    # Create output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_path = config.output_dir / "pipeline_config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2)

    try:
        # Stage 1: Ingest
        if progress_callback:
            progress_callback("ingest", "running")

        cells_adata, spatial_adata = _run_ingest_stage(config, result)

        if cells_adata is None and spatial_adata is None:
            raise ValueError("No data loaded. Check data paths.")

        # Stage 2: Harmonize
        if progress_callback:
            progress_callback("harmonize", "running")

        cells_adata, spatial_adata = _run_harmonize_stage(
            cells_adata, spatial_adata, config, result
        )

        # Stage 3: QC
        if not config.skip_qc:
            if progress_callback:
                progress_callback("qc", "running")

            cells_adata, spatial_adata = _run_qc_stage(cells_adata, spatial_adata, config, result)

        # Stage 4: Normalize
        if not config.skip_normalization:
            if progress_callback:
                progress_callback("normalize", "running")

            cells_adata, spatial_adata = _run_normalize_stage(
                cells_adata, spatial_adata, config, result
            )

        # Stage 5: Export
        if not config.skip_export:
            if progress_callback:
                progress_callback("export", "running")

            _run_export_stage(cells_adata, spatial_adata, config, result)

        result.status = "success"
        log.info("Data pipeline completed successfully")

    except Exception as e:
        result.status = "failed"
        result.errors.append(str(e))
        log.error("Data pipeline failed: %s", e)
        raise

    finally:
        result.completed_at = datetime.now(timezone.utc).isoformat()
        if result.started_at and result.completed_at:
            start = datetime.fromisoformat(result.started_at)
            end = datetime.fromisoformat(result.completed_at)
            result.duration_seconds = (end - start).total_seconds()

        # Save result
        result_path = config.output_dir / "pipeline_result.json"
        result.save(result_path)

        # Register artifacts
        if artifact_registry is not None:
            try:
                artifact_registry.register_artifacts_from_dir(
                    config.output_dir,
                    stage="data_pipeline",
                )
            except Exception as e:
                log.warning("Failed to register artifacts: %s", e)

    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Command-line entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run StageBridge data pipeline")
    parser.add_argument("--data-root", required=True, help="Data root directory")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--dataset-name", default="dataset", help="Dataset name")
    parser.add_argument("--modality", default="both", choices=["snrna", "spatial", "both"])
    parser.add_argument("--smoke", action="store_true", help="Enable smoke test mode")
    parser.add_argument(
        "--smoke-donors", type=int, default=2, help="Number of donors for smoke test"
    )
    parser.add_argument("--smoke-cells", type=int, default=1000, help="Max cells for smoke test")
    parser.add_argument("--skip-qc", action="store_true", help="Skip QC filtering")
    parser.add_argument("--skip-normalize", action="store_true", help="Skip normalization")
    parser.add_argument("--skip-export", action="store_true", help="Skip export")
    parser.add_argument("--no-figures", action="store_true", help="Skip figure generation")

    args = parser.parse_args()

    config = DataPipelineConfig(
        dataset_name=args.dataset_name,
        data_root=args.data_root,
        output_dir=args.output_dir,
        modality=args.modality,
        smoke_mode=args.smoke,
        smoke_n_donors=args.smoke_donors,
        smoke_n_cells=args.smoke_cells,
        skip_qc=args.skip_qc,
        skip_normalization=args.skip_normalize,
        skip_export=args.skip_export,
        generate_figures=not args.no_figures,
    )

    result = run_data_pipeline(config)

    if result.success:
        print(f"Pipeline completed successfully in {result.duration_seconds:.1f}s")
        print(f"Output: {config.output_dir}")
    else:
        print(f"Pipeline failed: {result.errors}")
        exit(1)


if __name__ == "__main__":
    main()
