"""
Canonical output writing and validation for StageBridge.

This module handles:
- Writing processed data in canonical format
- Generating manifests (donor, sample, stage)
- Output validation
- Atomic file writing

Canonical output structure:
    data/processed/<dataset>/
    ├── cells.h5ad              # Single-cell/nucleus AnnData
    ├── spatial.h5ad            # Spatial transcriptomics AnnData
    ├── cells.parquet           # Cell metadata table
    ├── spatial.parquet         # Spot metadata table
    ├── feature_spec.yaml       # Feature sets, HVGs, gene lists
    ├── sample_manifest.csv     # Sample-level metadata
    ├── donor_manifest.csv      # Donor-level metadata
    └── stage_manifest.csv      # Stage-level metadata

Usage:
    from stagebridge.data.export import export_canonical_dataset, validate_canonical_output

    result = export_canonical_dataset(adata, output_dir, dataset_name="luad_evo")
    valid, issues = validate_canonical_output(output_dir)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from stagebridge.logging_utils import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ExportResult:
    """Result of canonical export operation."""

    dataset_name: str
    output_dir: Path
    files_written: list[Path] = field(default_factory=list)
    n_cells: int = 0
    n_spots: int = 0
    n_genes: int = 0
    n_donors: int = 0
    n_samples: int = 0
    n_stages: int = 0
    exported_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """Whether export completed without errors."""
        return len(self.errors) == 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "dataset_name": self.dataset_name,
            "output_dir": str(self.output_dir),
            "files_written": [str(p) for p in self.files_written],
            "n_cells": self.n_cells,
            "n_spots": self.n_spots,
            "n_genes": self.n_genes,
            "n_donors": self.n_donors,
            "n_samples": self.n_samples,
            "n_stages": self.n_stages,
            "exported_at": self.exported_at,
            "errors": self.errors,
            "warnings": self.warnings,
            "success": self.success,
        }

    def save(self, path: Path | str) -> None:
        """Save export result to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass
class ExportValidationResult:
    """Result of canonical output validation."""

    output_dir: Path
    is_valid: bool
    files_found: list[str] = field(default_factory=list)
    files_missing: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "output_dir": str(self.output_dir),
            "is_valid": self.is_valid,
            "files_found": self.files_found,
            "files_missing": self.files_missing,
            "issues": self.issues,
            "warnings": self.warnings,
        }


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Required columns for manifests
REQUIRED_DONOR_COLUMNS = {"donor_id"}
REQUIRED_SAMPLE_COLUMNS = {"sample_id", "donor_id"}
REQUIRED_STAGE_COLUMNS = {"stage"}

# Required columns in cell/spot metadata
REQUIRED_OBS_COLUMNS = {"donor_id", "sample_id", "stage"}

# Canonical file names
CANONICAL_FILES = {
    "cells_h5ad": "cells.h5ad",
    "spatial_h5ad": "spatial.h5ad",
    "cells_parquet": "cells.parquet",
    "spatial_parquet": "spatial.parquet",
    "feature_spec": "feature_spec.yaml",
    "sample_manifest": "sample_manifest.csv",
    "donor_manifest": "donor_manifest.csv",
    "stage_manifest": "stage_manifest.csv",
    "export_result": "export_result.json",
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _require_anndata():
    """Import anndata lazily."""
    try:
        import anndata
    except ImportError as e:
        raise ImportError("anndata is required for export operations") from e
    return anndata


def _write_h5ad_atomic(adata: Any, path: Path, compression: str = "lzf") -> None:
    """Write h5ad file atomically."""
    from stagebridge.data.common.h5ad_atomic import write_h5ad_atomic

    write_h5ad_atomic(adata, path, compression=compression)


def _ensure_required_columns(
    obs: pd.DataFrame,
    required: set[str],
    fill_value: str = "unknown",
) -> pd.DataFrame:
    """Ensure required columns exist in DataFrame."""
    obs = obs.copy()
    for col in required:
        if col not in obs.columns:
            log.warning("Required column '%s' not found, filling with '%s'", col, fill_value)
            obs[col] = fill_value
    return obs


# ---------------------------------------------------------------------------
# Manifest generation
# ---------------------------------------------------------------------------


def generate_donor_manifest(
    adata: Any,  # AnnData
    *,
    donor_column: str = "donor_id",
    extra_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Generate donor-level manifest from AnnData.

    Parameters
    ----------
    adata : AnnData
        AnnData object.
    donor_column : str
        Column name for donor IDs.
    extra_columns : list[str], optional
        Additional columns to include (must be donor-level).

    Returns
    -------
    pd.DataFrame
        Donor manifest with columns: donor_id, n_cells, stages, samples, etc.
    """
    if donor_column not in adata.obs.columns:
        raise KeyError(f"Donor column '{donor_column}' not found in adata.obs")

    obs = adata.obs.copy()
    obs["donor_id"] = obs[donor_column].astype(str)

    # Aggregate by donor
    donor_data = []
    for donor_id, group in obs.groupby("donor_id"):
        row = {
            "donor_id": donor_id,
            "n_cells": len(group),
        }

        # Stages
        if "stage" in group.columns:
            stages = sorted(group["stage"].astype(str).unique())
            row["stages"] = ",".join(stages)
            row["n_stages"] = len(stages)

        # Samples
        if "sample_id" in group.columns:
            samples = sorted(group["sample_id"].astype(str).unique())
            row["samples"] = ",".join(samples)
            row["n_samples"] = len(samples)

        # Extra columns (take first value if consistent)
        if extra_columns:
            for col in extra_columns:
                if col in group.columns:
                    values = group[col].unique()
                    if len(values) == 1:
                        row[col] = values[0]
                    else:
                        row[col] = str(values[0]) + " (varies)"

        donor_data.append(row)

    manifest = pd.DataFrame(donor_data)
    manifest = manifest.sort_values("donor_id").reset_index(drop=True)

    log.info("Generated donor manifest: %d donors", len(manifest))
    return manifest


def generate_sample_manifest(
    adata: Any,  # AnnData
    *,
    sample_column: str = "sample_id",
    donor_column: str = "donor_id",
    extra_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Generate sample-level manifest from AnnData.

    Parameters
    ----------
    adata : AnnData
        AnnData object.
    sample_column : str
        Column name for sample IDs.
    donor_column : str
        Column name for donor IDs.
    extra_columns : list[str], optional
        Additional columns to include.

    Returns
    -------
    pd.DataFrame
        Sample manifest.
    """
    if sample_column not in adata.obs.columns:
        raise KeyError(f"Sample column '{sample_column}' not found in adata.obs")

    obs = adata.obs.copy()
    obs["sample_id"] = obs[sample_column].astype(str)

    if donor_column in obs.columns:
        obs["donor_id"] = obs[donor_column].astype(str)

    # Aggregate by sample
    sample_data = []
    for sample_id, group in obs.groupby("sample_id"):
        row = {
            "sample_id": sample_id,
            "n_cells": len(group),
        }

        if "donor_id" in group.columns:
            donors = group["donor_id"].unique()
            row["donor_id"] = donors[0] if len(donors) == 1 else ",".join(sorted(donors))

        if "stage" in group.columns:
            stages = group["stage"].unique()
            row["stage"] = (
                stages[0] if len(stages) == 1 else ",".join(sorted(str(s) for s in stages))
            )

        if "modality" in group.columns:
            modalities = group["modality"].unique()
            row["modality"] = (
                modalities[0]
                if len(modalities) == 1
                else ",".join(sorted(str(m) for m in modalities))
            )

        # Extra columns
        if extra_columns:
            for col in extra_columns:
                if col in group.columns:
                    values = group[col].unique()
                    row[col] = values[0] if len(values) == 1 else str(values[0])

        sample_data.append(row)

    manifest = pd.DataFrame(sample_data)
    manifest = manifest.sort_values("sample_id").reset_index(drop=True)

    log.info("Generated sample manifest: %d samples", len(manifest))
    return manifest


def generate_stage_manifest(
    adata: Any,  # AnnData
    *,
    stage_column: str = "stage",
    donor_column: str = "donor_id",
) -> pd.DataFrame:
    """Generate stage-level manifest from AnnData.

    Parameters
    ----------
    adata : AnnData
        AnnData object.
    stage_column : str
        Column name for stage labels.
    donor_column : str
        Column name for donor IDs.

    Returns
    -------
    pd.DataFrame
        Stage manifest.
    """
    if stage_column not in adata.obs.columns:
        raise KeyError(f"Stage column '{stage_column}' not found in adata.obs")

    obs = adata.obs.copy()
    obs["stage"] = obs[stage_column].astype(str)

    if donor_column in obs.columns:
        obs["donor_id"] = obs[donor_column].astype(str)

    # Aggregate by stage
    stage_data = []
    for stage, group in obs.groupby("stage"):
        row = {
            "stage": stage,
            "n_cells": len(group),
        }

        if "donor_id" in group.columns:
            donors = sorted(group["donor_id"].unique())
            row["n_donors"] = len(donors)
            row["donors"] = ",".join(donors)

        if "sample_id" in group.columns:
            samples = sorted(group["sample_id"].astype(str).unique())
            row["n_samples"] = len(samples)

        stage_data.append(row)

    manifest = pd.DataFrame(stage_data)

    # Sort stages in biological order if known
    stage_order = ["Normal", "AAH", "AIS", "MIA", "LUAD"]
    if all(s in stage_order for s in manifest["stage"].values):
        manifest["_order"] = manifest["stage"].map({s: i for i, s in enumerate(stage_order)})
        manifest = manifest.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)
    else:
        manifest = manifest.sort_values("stage").reset_index(drop=True)

    log.info("Generated stage manifest: %d stages", len(manifest))
    return manifest


# ---------------------------------------------------------------------------
# Export functions
# ---------------------------------------------------------------------------


def export_canonical_dataset(
    adata: Any | None = None,  # AnnData for cells
    spatial_adata: Any | None = None,  # AnnData for spatial
    output_dir: str | Path = ".",
    dataset_name: str = "dataset",
    *,
    feature_spec: Any | None = None,  # FeatureSpec
    donor_column: str = "donor_id",
    sample_column: str = "sample_id",
    stage_column: str = "stage",
    compression: str = "lzf",
    write_parquet: bool = True,
    write_manifests: bool = True,
) -> ExportResult:
    """Export processed data in canonical format.

    Writes:
    - cells.h5ad / spatial.h5ad (AnnData files)
    - cells.parquet / spatial.parquet (metadata tables)
    - feature_spec.yaml (feature specification)
    - donor_manifest.csv, sample_manifest.csv, stage_manifest.csv
    - export_result.json (export metadata)

    Parameters
    ----------
    adata : AnnData, optional
        Single-cell/nucleus AnnData object.
    spatial_adata : AnnData, optional
        Spatial transcriptomics AnnData object.
    output_dir : Path
        Output directory.
    dataset_name : str
        Dataset name for logging and metadata.
    feature_spec : FeatureSpec, optional
        Feature specification object.
    donor_column, sample_column, stage_column : str
        Column names for metadata.
    compression : str
        H5AD compression method.
    write_parquet : bool
        Whether to write parquet metadata tables.
    write_manifests : bool
        Whether to write manifest CSVs.

    Returns
    -------
    ExportResult
        Export result with file paths and statistics.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = ExportResult(
        dataset_name=dataset_name,
        output_dir=output_dir,
    )

    log.info("Exporting dataset '%s' to %s ...", dataset_name, output_dir)

    # Export cells.h5ad
    if adata is not None:
        try:
            cells_path = output_dir / CANONICAL_FILES["cells_h5ad"]

            # Ensure required columns
            adata.obs = _ensure_required_columns(adata.obs, REQUIRED_OBS_COLUMNS)

            _write_h5ad_atomic(adata, cells_path, compression=compression)
            result.files_written.append(cells_path)
            result.n_cells = adata.n_obs
            result.n_genes = adata.n_vars

            log.info("Wrote cells.h5ad: %d cells, %d genes", adata.n_obs, adata.n_vars)

            # Export cells.parquet
            if write_parquet:
                cells_parquet = output_dir / CANONICAL_FILES["cells_parquet"]
                adata.obs.to_parquet(cells_parquet)
                result.files_written.append(cells_parquet)
                log.info("Wrote cells.parquet: %d rows", len(adata.obs))

            # Count unique values
            if donor_column in adata.obs.columns:
                result.n_donors = adata.obs[donor_column].nunique()
            if sample_column in adata.obs.columns:
                result.n_samples = adata.obs[sample_column].nunique()
            if stage_column in adata.obs.columns:
                result.n_stages = adata.obs[stage_column].nunique()

        except Exception as e:
            result.errors.append(f"Failed to export cells data: {e}")
            log.error("Failed to export cells data: %s", e)

    # Export spatial.h5ad
    if spatial_adata is not None:
        try:
            spatial_path = output_dir / CANONICAL_FILES["spatial_h5ad"]

            # Ensure required columns
            spatial_adata.obs = _ensure_required_columns(spatial_adata.obs, REQUIRED_OBS_COLUMNS)

            _write_h5ad_atomic(spatial_adata, spatial_path, compression=compression)
            result.files_written.append(spatial_path)
            result.n_spots = spatial_adata.n_obs

            log.info("Wrote spatial.h5ad: %d spots", spatial_adata.n_obs)

            # Export spatial.parquet
            if write_parquet:
                spatial_parquet = output_dir / CANONICAL_FILES["spatial_parquet"]
                spatial_adata.obs.to_parquet(spatial_parquet)
                result.files_written.append(spatial_parquet)
                log.info("Wrote spatial.parquet: %d rows", len(spatial_adata.obs))

        except Exception as e:
            result.errors.append(f"Failed to export spatial data: {e}")
            log.error("Failed to export spatial data: %s", e)

    # Export feature_spec.yaml
    if feature_spec is not None:
        try:
            feature_path = output_dir / CANONICAL_FILES["feature_spec"]
            feature_spec.save(feature_path)
            result.files_written.append(feature_path)
            log.info("Wrote feature_spec.yaml")
        except Exception as e:
            result.errors.append(f"Failed to export feature spec: {e}")
            log.error("Failed to export feature spec: %s", e)

    # Generate and export manifests
    if write_manifests:
        primary_adata = adata if adata is not None else spatial_adata

        if primary_adata is not None:
            # Donor manifest
            try:
                donor_manifest = generate_donor_manifest(primary_adata, donor_column=donor_column)
                donor_path = output_dir / CANONICAL_FILES["donor_manifest"]
                donor_manifest.to_csv(donor_path, index=False)
                result.files_written.append(donor_path)
            except Exception as e:
                result.warnings.append(f"Could not generate donor manifest: {e}")

            # Sample manifest
            try:
                sample_manifest = generate_sample_manifest(
                    primary_adata,
                    sample_column=sample_column,
                    donor_column=donor_column,
                )
                sample_path = output_dir / CANONICAL_FILES["sample_manifest"]
                sample_manifest.to_csv(sample_path, index=False)
                result.files_written.append(sample_path)
            except Exception as e:
                result.warnings.append(f"Could not generate sample manifest: {e}")

            # Stage manifest
            try:
                stage_manifest = generate_stage_manifest(
                    primary_adata,
                    stage_column=stage_column,
                    donor_column=donor_column,
                )
                stage_path = output_dir / CANONICAL_FILES["stage_manifest"]
                stage_manifest.to_csv(stage_path, index=False)
                result.files_written.append(stage_path)
            except Exception as e:
                result.warnings.append(f"Could not generate stage manifest: {e}")

    # Write export result
    result_path = output_dir / CANONICAL_FILES["export_result"]
    result.save(result_path)
    result.files_written.append(result_path)

    if result.success:
        log.info(
            "Export complete: %d files written, %d cells, %d spots",
            len(result.files_written),
            result.n_cells,
            result.n_spots,
        )
    else:
        log.error("Export completed with %d errors", len(result.errors))

    return result


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_canonical_output(
    output_dir: str | Path,
    *,
    require_cells: bool = True,
    require_spatial: bool = False,
    require_manifests: bool = True,
) -> tuple[bool, list[str]]:
    """Validate canonical output directory.

    Checks:
    - Required files exist
    - H5AD files are readable
    - Manifests have required columns
    - Data is non-empty

    Parameters
    ----------
    output_dir : Path
        Output directory to validate.
    require_cells : bool
        Whether cells.h5ad is required.
    require_spatial : bool
        Whether spatial.h5ad is required.
    require_manifests : bool
        Whether manifest CSVs are required.

    Returns
    -------
    tuple[bool, list[str]]
        (is_valid, list of issues)
    """
    output_dir = Path(output_dir)
    issues = []

    if not output_dir.exists():
        return False, [f"Output directory does not exist: {output_dir}"]

    if not output_dir.is_dir():
        return False, [f"Output path is not a directory: {output_dir}"]

    # Check cells.h5ad
    if require_cells:
        cells_path = output_dir / CANONICAL_FILES["cells_h5ad"]
        if not cells_path.exists():
            issues.append(f"Missing required file: {CANONICAL_FILES['cells_h5ad']}")
        else:
            try:
                import anndata

                adata = anndata.read_h5ad(cells_path, backed="r")
                if adata.n_obs == 0:
                    issues.append("cells.h5ad is empty (0 cells)")
                if adata.n_vars == 0:
                    issues.append("cells.h5ad has 0 genes")

                # Check required columns
                for col in REQUIRED_OBS_COLUMNS:
                    if col not in adata.obs.columns:
                        issues.append(f"cells.h5ad missing required obs column: {col}")

                adata.file.close()
            except Exception as e:
                issues.append(f"cells.h5ad is not readable: {e}")

    # Check spatial.h5ad
    if require_spatial:
        spatial_path = output_dir / CANONICAL_FILES["spatial_h5ad"]
        if not spatial_path.exists():
            issues.append(f"Missing required file: {CANONICAL_FILES['spatial_h5ad']}")
        else:
            try:
                import anndata

                adata = anndata.read_h5ad(spatial_path, backed="r")
                if adata.n_obs == 0:
                    issues.append("spatial.h5ad is empty (0 spots)")

                # Check for spatial coordinates
                if "spatial" not in adata.obsm:
                    issues.append("spatial.h5ad missing obsm['spatial'] coordinates")

                adata.file.close()
            except Exception as e:
                issues.append(f"spatial.h5ad is not readable: {e}")

    # Check manifests
    if require_manifests:
        for manifest_key, required_cols in [
            ("donor_manifest", REQUIRED_DONOR_COLUMNS),
            ("sample_manifest", REQUIRED_SAMPLE_COLUMNS),
            ("stage_manifest", REQUIRED_STAGE_COLUMNS),
        ]:
            manifest_path = output_dir / CANONICAL_FILES[manifest_key]
            if not manifest_path.exists():
                issues.append(f"Missing manifest: {CANONICAL_FILES[manifest_key]}")
            else:
                try:
                    df = pd.read_csv(manifest_path)
                    if len(df) == 0:
                        issues.append(f"{CANONICAL_FILES[manifest_key]} is empty")
                    for col in required_cols:
                        if col not in df.columns:
                            issues.append(f"{CANONICAL_FILES[manifest_key]} missing column: {col}")
                except Exception as e:
                    issues.append(f"Cannot read {CANONICAL_FILES[manifest_key]}: {e}")

    # Check feature spec
    feature_path = output_dir / CANONICAL_FILES["feature_spec"]
    if feature_path.exists():
        try:
            import yaml

            with feature_path.open("r") as f:
                spec = yaml.safe_load(f)
            if not spec.get("all_genes"):
                issues.append("feature_spec.yaml has no genes listed")
        except Exception as e:
            issues.append(f"Cannot read feature_spec.yaml: {e}")

    is_valid = len(issues) == 0

    if is_valid:
        log.info("Canonical output validation passed: %s", output_dir)
    else:
        log.warning("Canonical output validation failed with %d issues", len(issues))
        for issue in issues:
            log.warning("  - %s", issue)

    return is_valid, issues


def load_canonical_dataset(
    output_dir: str | Path,
    *,
    load_cells: bool = True,
    load_spatial: bool = False,
    backed: bool = False,
) -> dict[str, Any]:
    """Load canonical dataset from output directory.

    Parameters
    ----------
    output_dir : Path
        Output directory.
    load_cells : bool
        Whether to load cells.h5ad.
    load_spatial : bool
        Whether to load spatial.h5ad.
    backed : bool
        Whether to load h5ad in backed mode.

    Returns
    -------
    dict
        Dictionary with loaded data:
        - cells: AnnData or None
        - spatial: AnnData or None
        - donor_manifest: DataFrame or None
        - sample_manifest: DataFrame or None
        - stage_manifest: DataFrame or None
        - feature_spec: dict or None
    """
    import anndata

    output_dir = Path(output_dir)
    result = {
        "cells": None,
        "spatial": None,
        "donor_manifest": None,
        "sample_manifest": None,
        "stage_manifest": None,
        "feature_spec": None,
    }

    # Load cells
    if load_cells:
        cells_path = output_dir / CANONICAL_FILES["cells_h5ad"]
        if cells_path.exists():
            result["cells"] = anndata.read_h5ad(cells_path, backed="r" if backed else None)
            log.info("Loaded cells: %d cells", result["cells"].n_obs)

    # Load spatial
    if load_spatial:
        spatial_path = output_dir / CANONICAL_FILES["spatial_h5ad"]
        if spatial_path.exists():
            result["spatial"] = anndata.read_h5ad(spatial_path, backed="r" if backed else None)
            log.info("Loaded spatial: %d spots", result["spatial"].n_obs)

    # Load manifests
    for key in ["donor_manifest", "sample_manifest", "stage_manifest"]:
        path = output_dir / CANONICAL_FILES[key]
        if path.exists():
            result[key] = pd.read_csv(path)

    # Load feature spec
    feature_path = output_dir / CANONICAL_FILES["feature_spec"]
    if feature_path.exists():
        try:
            import yaml

            with feature_path.open("r") as f:
                result["feature_spec"] = yaml.safe_load(f)
        except ImportError:
            with feature_path.with_suffix(".json").open("r") as f:
                result["feature_spec"] = json.load(f)

    return result
