"""Standardized output schemas for reference geometry outputs.

This module defines the canonical schema for reference embeddings and
provides utilities for exporting and loading outputs in a standardized format.

All outputs are consumable by downstream models through standardized schemas.
No custom per-backend hacks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stagebridge.logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class ReferenceEmbeddingSchema:
    """Schema definition for reference embedding outputs.

    This defines the required columns for each output file type.
    """

    # Required metadata columns
    METADATA_COLS: list[str] = field(
        default_factory=lambda: [
            "cell_id",
            "donor_id",
            "sample_id",
            "stage_id",
        ]
    )

    # HLCA embedding columns pattern
    HLCA_LATENT_PREFIX: str = "hlca_latent_"

    # LuCa embedding columns pattern
    LUCA_LATENT_PREFIX: str = "luca_latent_"

    # Fused embedding columns pattern
    FUSED_LATENT_PREFIX: str = "fused_latent_"

    # Confidence columns
    CONFIDENCE_COLS: list[str] = field(
        default_factory=lambda: [
            "hlca_confidence",
            "luca_confidence",
        ]
    )

    # Reference mode column
    MODE_COL: str = "reference_mode_used"


# Global schema instance
SCHEMA = ReferenceEmbeddingSchema()


@dataclass
class ReferenceManifest:
    """Manifest describing a reference geometry run.

    Saved as reference_manifest.json for provenance tracking.
    """

    run_id: str
    created_at: str
    hlca_latent_dim: int
    luca_latent_dim: int
    fused_latent_dim: int
    n_cells: int
    fusion_method: str
    mapping_method: str
    hlca_reference_path: str
    luca_reference_path: str | None
    query_data_path: str
    geometry_backend: str
    parameters: dict[str, Any] = field(default_factory=dict)
    validation_status: str = "pending"
    validation_errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReferenceManifest":
        """Create from dictionary."""
        return cls(**data)


def export_reference_outputs(
    hlca_df: pd.DataFrame,
    luca_df: pd.DataFrame,
    fused_df: pd.DataFrame,
    confidence_df: pd.DataFrame,
    manifest: ReferenceManifest,
    feature_overlap: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Export all reference outputs to standardized format.

    Creates the following files in output_dir:
    - hlca_embedding.parquet
    - luca_embedding.parquet
    - fused_embedding.parquet
    - reference_confidence.parquet
    - reference_manifest.json
    - feature_overlap_report.json

    Parameters
    ----------
    hlca_df : pd.DataFrame
        HLCA embedding DataFrame
    luca_df : pd.DataFrame
        LuCa embedding DataFrame
    fused_df : pd.DataFrame
        Fused embedding DataFrame
    confidence_df : pd.DataFrame
        Confidence scores DataFrame
    manifest : ReferenceManifest
        Run manifest
    feature_overlap : dict
        Feature overlap report
    output_dir : str or Path
        Output directory

    Returns
    -------
    dict[str, Path]
        Mapping of output names to file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {}

    # Validate schema compliance
    _validate_dataframe_schema(hlca_df, "hlca")
    _validate_dataframe_schema(luca_df, "luca")
    _validate_dataframe_schema(fused_df, "fused")
    _validate_confidence_schema(confidence_df)

    # Export parquet files
    hlca_path = output_dir / "hlca_embedding.parquet"
    hlca_df.to_parquet(hlca_path, index=False)
    paths["hlca_embedding"] = hlca_path

    luca_path = output_dir / "luca_embedding.parquet"
    luca_df.to_parquet(luca_path, index=False)
    paths["luca_embedding"] = luca_path

    fused_path = output_dir / "fused_embedding.parquet"
    fused_df.to_parquet(fused_path, index=False)
    paths["fused_embedding"] = fused_path

    confidence_path = output_dir / "reference_confidence.parquet"
    confidence_df.to_parquet(confidence_path, index=False)
    paths["reference_confidence"] = confidence_path

    # Export JSON files
    manifest_path = output_dir / "reference_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, indent=2)
    paths["reference_manifest"] = manifest_path

    overlap_path = output_dir / "feature_overlap_report.json"
    with open(overlap_path, "w", encoding="utf-8") as f:
        json.dump(feature_overlap, f, indent=2)
    paths["feature_overlap_report"] = overlap_path

    # Create plots directory
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    paths["plots_dir"] = plots_dir

    log.info(
        "Exported reference outputs to %s: %d files",
        output_dir,
        len(paths),
    )

    return paths


def load_reference_outputs(
    output_dir: str | Path,
) -> dict[str, Any]:
    """Load reference outputs from standardized format.

    Parameters
    ----------
    output_dir : str or Path
        Directory containing reference outputs

    Returns
    -------
    dict
        Dictionary with loaded outputs:
        - hlca_df: HLCA embedding DataFrame
        - luca_df: LuCa embedding DataFrame
        - fused_df: Fused embedding DataFrame
        - confidence_df: Confidence DataFrame
        - manifest: ReferenceManifest
        - feature_overlap: Feature overlap report

    Raises
    ------
    FileNotFoundError
        If required files are missing
    """
    output_dir = Path(output_dir)

    result = {}

    # Load parquet files
    hlca_path = output_dir / "hlca_embedding.parquet"
    if hlca_path.exists():
        result["hlca_df"] = pd.read_parquet(hlca_path)
    else:
        raise FileNotFoundError(f"Missing HLCA embedding: {hlca_path}")

    luca_path = output_dir / "luca_embedding.parquet"
    if luca_path.exists():
        result["luca_df"] = pd.read_parquet(luca_path)
    else:
        raise FileNotFoundError(f"Missing LuCa embedding: {luca_path}")

    fused_path = output_dir / "fused_embedding.parquet"
    if fused_path.exists():
        result["fused_df"] = pd.read_parquet(fused_path)
    else:
        raise FileNotFoundError(f"Missing fused embedding: {fused_path}")

    confidence_path = output_dir / "reference_confidence.parquet"
    if confidence_path.exists():
        result["confidence_df"] = pd.read_parquet(confidence_path)
    else:
        raise FileNotFoundError(f"Missing confidence scores: {confidence_path}")

    # Load JSON files
    manifest_path = output_dir / "reference_manifest.json"
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            result["manifest"] = ReferenceManifest.from_dict(json.load(f))
    else:
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    overlap_path = output_dir / "feature_overlap_report.json"
    if overlap_path.exists():
        with open(overlap_path, encoding="utf-8") as f:
            result["feature_overlap"] = json.load(f)
    else:
        result["feature_overlap"] = {}

    log.info("Loaded reference outputs from %s", output_dir)

    return result


def _validate_dataframe_schema(
    df: pd.DataFrame,
    embedding_type: str,
) -> None:
    """Validate DataFrame has required schema columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate
    embedding_type : str
        One of "hlca", "luca", or "fused"

    Raises
    ------
    ValueError
        If required columns are missing
    """
    required_cols = SCHEMA.METADATA_COLS.copy()

    # Add latent columns based on type
    if embedding_type == "hlca":
        prefix = SCHEMA.HLCA_LATENT_PREFIX
    elif embedding_type == "luca":
        prefix = SCHEMA.LUCA_LATENT_PREFIX
    elif embedding_type == "fused":
        prefix = SCHEMA.FUSED_LATENT_PREFIX
        # Fused should have all three prefixes
        required_cols.append(SCHEMA.MODE_COL)
    else:
        raise ValueError(f"Unknown embedding type: {embedding_type}")

    # Check metadata columns
    missing_metadata = [col for col in SCHEMA.METADATA_COLS if col not in df.columns]
    if missing_metadata:
        raise ValueError(
            f"{embedding_type} embedding missing metadata columns: {missing_metadata}"
        )

    # Check for at least one latent column
    latent_cols = [c for c in df.columns if c.startswith(prefix)]
    if not latent_cols:
        raise ValueError(
            f"{embedding_type} embedding has no latent columns with prefix '{prefix}'"
        )

    # Check for duplicated cell IDs
    if df["cell_id"].duplicated().any():
        n_dups = int(df["cell_id"].duplicated().sum())
        raise ValueError(f"{embedding_type} embedding has {n_dups} duplicated cell IDs")


def _validate_confidence_schema(df: pd.DataFrame) -> None:
    """Validate confidence DataFrame schema.

    Parameters
    ----------
    df : pd.DataFrame
        Confidence DataFrame

    Raises
    ------
    ValueError
        If required columns are missing
    """
    required = ["cell_id"] + SCHEMA.CONFIDENCE_COLS
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Confidence DataFrame missing columns: {missing}")


def validate_output_integrity(output_dir: str | Path) -> dict[str, Any]:
    """Validate integrity of saved reference outputs.

    Checks:
    - All required files exist
    - DataFrames can be loaded with correct dtypes
    - Cell IDs are consistent across files
    - No NaN values in cell IDs
    - Embedding dimensions are consistent

    Parameters
    ----------
    output_dir : str or Path
        Directory containing reference outputs

    Returns
    -------
    dict
        Validation report
    """
    output_dir = Path(output_dir)
    report = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "checks": {},
    }

    # Check file existence
    required_files = [
        "hlca_embedding.parquet",
        "luca_embedding.parquet",
        "fused_embedding.parquet",
        "reference_confidence.parquet",
        "reference_manifest.json",
    ]

    for filename in required_files:
        path = output_dir / filename
        report["checks"][filename] = path.exists()
        if not path.exists():
            report["errors"].append(f"Missing required file: {filename}")
            report["valid"] = False

    if not report["valid"]:
        return report

    # Load and validate DataFrames
    try:
        outputs = load_reference_outputs(output_dir)
    except Exception as e:
        report["errors"].append(f"Failed to load outputs: {e}")
        report["valid"] = False
        return report

    # Check cell ID consistency
    hlca_cells = set(outputs["hlca_df"]["cell_id"])
    luca_cells = set(outputs["luca_df"]["cell_id"])
    fused_cells = set(outputs["fused_df"]["cell_id"])
    conf_cells = set(outputs["confidence_df"]["cell_id"])

    if hlca_cells != luca_cells:
        report["errors"].append("Cell IDs mismatch between HLCA and LuCa")
        report["valid"] = False
    if hlca_cells != fused_cells:
        report["errors"].append("Cell IDs mismatch between HLCA and fused")
        report["valid"] = False
    if hlca_cells != conf_cells:
        report["errors"].append("Cell IDs mismatch between HLCA and confidence")
        report["valid"] = False

    # Check for NaN in cell IDs
    for name, df in [
        ("hlca", outputs["hlca_df"]),
        ("luca", outputs["luca_df"]),
        ("fused", outputs["fused_df"]),
        ("confidence", outputs["confidence_df"]),
    ]:
        if df["cell_id"].isna().any():
            report["errors"].append(f"NaN values in {name} cell_id column")
            report["valid"] = False

    # Check embedding dimensions match manifest
    manifest = outputs["manifest"]
    hlca_dim = len([c for c in outputs["hlca_df"].columns if c.startswith("hlca_latent_")])
    luca_dim = len([c for c in outputs["luca_df"].columns if c.startswith("luca_latent_")])
    fused_dim = len([c for c in outputs["fused_df"].columns if c.startswith("fused_latent_")])

    if hlca_dim != manifest.hlca_latent_dim:
        report["warnings"].append(
            f"HLCA dim mismatch: manifest={manifest.hlca_latent_dim}, actual={hlca_dim}"
        )
    if luca_dim != manifest.luca_latent_dim:
        report["warnings"].append(
            f"LuCa dim mismatch: manifest={manifest.luca_latent_dim}, actual={luca_dim}"
        )
    if fused_dim != manifest.fused_latent_dim:
        report["warnings"].append(
            f"Fused dim mismatch: manifest={manifest.fused_latent_dim}, actual={fused_dim}"
        )

    # Record statistics
    report["stats"] = {
        "n_cells": len(hlca_cells),
        "hlca_dim": hlca_dim,
        "luca_dim": luca_dim,
        "fused_dim": fused_dim,
    }

    log.info(
        "Output validation: valid=%s, errors=%d, warnings=%d",
        report["valid"],
        len(report["errors"]),
        len(report["warnings"]),
    )

    return report


def create_manifest(
    run_id: str,
    hlca_dim: int,
    luca_dim: int,
    fused_dim: int,
    n_cells: int,
    fusion_method: str,
    mapping_method: str,
    hlca_path: str,
    luca_path: str | None,
    query_path: str,
    geometry: str = "euclidean",
    parameters: dict[str, Any] | None = None,
) -> ReferenceManifest:
    """Create a reference manifest for a run.

    Parameters
    ----------
    run_id : str
        Unique run identifier
    hlca_dim : int
        HLCA latent dimension
    luca_dim : int
        LuCa latent dimension
    fused_dim : int
        Fused latent dimension
    n_cells : int
        Number of cells processed
    fusion_method : str
        Fusion method used
    mapping_method : str
        Mapping method used
    hlca_path : str
        Path to HLCA reference
    luca_path : str, optional
        Path to LuCa reference
    query_path : str
        Path to query data
    geometry : str
        Geometry backend name
    parameters : dict, optional
        Additional parameters

    Returns
    -------
    ReferenceManifest
        Created manifest
    """
    return ReferenceManifest(
        run_id=run_id,
        created_at=datetime.now().isoformat(),
        hlca_latent_dim=hlca_dim,
        luca_latent_dim=luca_dim,
        fused_latent_dim=fused_dim,
        n_cells=n_cells,
        fusion_method=fusion_method,
        mapping_method=mapping_method,
        hlca_reference_path=hlca_path,
        luca_reference_path=luca_path,
        query_data_path=query_path,
        geometry_backend=geometry,
        parameters=parameters or {},
    )
