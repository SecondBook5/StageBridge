"""Reference loading and validation for HLCA and LuCa references.

This module provides unified loading interfaces for reference atlases with
comprehensive validation, feature overlap analysis, and metadata checking.

Supported references:
- HLCA (Human Lung Cell Atlas): Healthy lung reference
- LuCa (Lung Cancer Atlas): Disease-aware reference (placeholder for future)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stagebridge.logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class ReferenceInfo:
    """Metadata container for a loaded reference atlas."""

    name: str
    source_path: Path
    n_cells: int
    n_genes: int
    latent_key: str
    latent_dim: int
    available_labels: list[str]
    metadata_columns: list[str]
    load_mode: str  # "full" or "backed"


@dataclass
class FeatureOverlapReport:
    """Report on feature overlap between query and reference."""

    query_gene_count: int
    reference_gene_count: int
    shared_gene_count: int
    overlap_fraction: float
    missing_in_query: list[str] = field(default_factory=list)
    missing_in_reference: list[str] = field(default_factory=list)
    status: str = "complete"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "query_gene_count": self.query_gene_count,
            "reference_gene_count": self.reference_gene_count,
            "shared_gene_count": self.shared_gene_count,
            "overlap_fraction": self.overlap_fraction,
            "missing_in_query_count": len(self.missing_in_query),
            "missing_in_reference_count": len(self.missing_in_reference),
            "missing_in_query_sample": self.missing_in_query[:20],
            "missing_in_reference_sample": self.missing_in_reference[:20],
            "status": self.status,
        }


@dataclass
class LoadedReference:
    """Container for a loaded reference atlas with validation metadata."""

    adata: Any  # AnnData
    info: ReferenceInfo
    validation_errors: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Check if reference passed validation."""
        return len(self.validation_errors) == 0


def _validate_reference_common(
    adata: Any,
    reference_type: str,
    required_obs_cols: list[str],
    latent_key: str,
) -> list[str]:
    """Validate common reference requirements."""
    errors = []

    # Check obs columns
    missing_cols = [col for col in required_obs_cols if col not in adata.obs.columns]
    if missing_cols:
        errors.append(f"Missing required obs columns for {reference_type}: {missing_cols}")

    # Check latent embedding exists
    if latent_key not in adata.obsm:
        errors.append(
            f"Missing latent embedding '{latent_key}' in obsm for {reference_type}. "
            f"Available keys: {list(adata.obsm.keys())}"
        )
    else:
        latent = adata.obsm[latent_key]
        if latent.ndim != 2:
            errors.append(f"Latent embedding must be 2D, got shape {latent.shape}")
        if np.any(np.isnan(latent)):
            nan_count = int(np.sum(np.isnan(latent)))
            errors.append(f"Latent embedding contains {nan_count} NaN values")
        if np.any(np.isinf(latent)):
            inf_count = int(np.sum(np.isinf(latent)))
            errors.append(f"Latent embedding contains {inf_count} Inf values")

    # Check for empty reference
    if adata.n_obs == 0:
        errors.append(f"{reference_type} reference has 0 cells")

    return errors


def load_hlca_reference(
    path: str | Path,
    *,
    backed: str | None = None,
    latent_key: str = "X_scanvi_emb",
    validate: bool = True,
) -> LoadedReference:
    """Load HLCA (Human Lung Cell Atlas) reference with validation.

    Parameters
    ----------
    path : str or Path
        Path to HLCA h5ad file
    backed : str, optional
        AnnData backed mode ("r" for read-only). None loads fully into memory.
    latent_key : str
        Key in obsm containing the reference latent embedding
    validate : bool
        Whether to run validation checks

    Returns
    -------
    LoadedReference
        Container with loaded AnnData and validation info

    Raises
    ------
    FileNotFoundError
        If reference file does not exist
    """
    try:
        import anndata
    except ImportError as exc:
        raise ImportError(
            "anndata is required for reference loading. Install with: pip install anndata"
        ) from exc

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"HLCA reference not found at {path}")

    log.info("Loading HLCA reference from %s (backed=%s)", path, backed)
    adata = anndata.read_h5ad(path, backed=backed)
    log.info("Loaded HLCA reference: %d cells, %d genes", adata.n_obs, adata.n_vars)

    # Expected HLCA columns
    required_obs = ["ann_level_1", "ann_level_2", "ann_level_3"]
    validation_errors = []

    if validate:
        validation_errors = _validate_reference_common(adata, "HLCA", required_obs, latent_key)

    # Determine latent dimension
    latent_dim = 0
    if latent_key in adata.obsm:
        latent_dim = adata.obsm[latent_key].shape[1]

    # Collect available label columns
    label_cols = [
        col for col in adata.obs.columns if col.startswith("ann_") or col.endswith("_label")
    ]

    info = ReferenceInfo(
        name="HLCA",
        source_path=path,
        n_cells=adata.n_obs,
        n_genes=adata.n_vars,
        latent_key=latent_key,
        latent_dim=latent_dim,
        available_labels=label_cols,
        metadata_columns=list(adata.obs.columns),
        load_mode="backed" if backed else "full",
    )

    return LoadedReference(
        adata=adata,
        info=info,
        validation_errors=validation_errors,
    )


def load_luca_reference(
    path: str | Path,
    *,
    backed: str | None = None,
    latent_key: str = "X_scVI",
    validate: bool = True,
) -> LoadedReference:
    """Load LuCa (Lung Cancer Atlas) reference with validation.

    Note: LuCa is a placeholder for disease-aware reference. The actual
    implementation may need adjustment based on the final LuCa data format.

    Parameters
    ----------
    path : str or Path
        Path to LuCa h5ad file
    backed : str, optional
        AnnData backed mode ("r" for read-only). None loads fully into memory.
    latent_key : str
        Key in obsm containing the reference latent embedding
    validate : bool
        Whether to run validation checks

    Returns
    -------
    LoadedReference
        Container with loaded AnnData and validation info

    Raises
    ------
    FileNotFoundError
        If reference file does not exist
    """
    try:
        import anndata
    except ImportError as exc:
        raise ImportError(
            "anndata is required for reference loading. Install with: pip install anndata"
        ) from exc

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"LuCa reference not found at {path}")

    log.info("Loading LuCa reference from %s (backed=%s)", path, backed)
    adata = anndata.read_h5ad(path, backed=backed)
    log.info("Loaded LuCa reference: %d cells, %d genes", adata.n_obs, adata.n_vars)

    # Expected LuCa columns (may need adjustment for actual data)
    required_obs = ["cell_type"]  # Minimal requirement
    validation_errors = []

    if validate:
        validation_errors = _validate_reference_common(adata, "LuCa", required_obs, latent_key)

    # Determine latent dimension
    latent_dim = 0
    if latent_key in adata.obsm:
        latent_dim = adata.obsm[latent_key].shape[1]

    # Collect available label columns
    label_cols = [
        col for col in adata.obs.columns if "type" in col.lower() or col.endswith("_label")
    ]

    info = ReferenceInfo(
        name="LuCa",
        source_path=path,
        n_cells=adata.n_obs,
        n_genes=adata.n_vars,
        latent_key=latent_key,
        latent_dim=latent_dim,
        available_labels=label_cols,
        metadata_columns=list(adata.obs.columns),
        load_mode="backed" if backed else "full",
    )

    return LoadedReference(
        adata=adata,
        info=info,
        validation_errors=validation_errors,
    )


def validate_reference(
    adata: Any,
    reference_type: str,
    latent_key: str = "X_scanvi_emb",
) -> list[str]:
    """Validate a reference AnnData object.

    Parameters
    ----------
    adata : AnnData
        Reference AnnData object
    reference_type : str
        Type of reference ("HLCA" or "LuCa")
    latent_key : str
        Expected latent embedding key

    Returns
    -------
    list[str]
        List of validation error messages (empty if valid)
    """
    if reference_type.upper() == "HLCA":
        required_obs = ["ann_level_1", "ann_level_2", "ann_level_3"]
    elif reference_type.upper() == "LUCA":
        required_obs = ["cell_type"]
    else:
        required_obs = []

    return _validate_reference_common(adata, reference_type, required_obs, latent_key)


def compute_feature_overlap(
    query: Any,
    reference: Any,
    *,
    min_overlap_threshold: float = 0.3,
    max_missing_to_report: int = 100,
) -> FeatureOverlapReport:
    """Compute feature (gene) overlap between query and reference data.

    Parameters
    ----------
    query : AnnData
        Query AnnData object
    reference : AnnData or LoadedReference
        Reference AnnData object or LoadedReference container
    min_overlap_threshold : float
        Minimum acceptable overlap fraction (for status)
    max_missing_to_report : int
        Maximum number of missing genes to include in report

    Returns
    -------
    FeatureOverlapReport
        Detailed overlap report
    """
    # Handle LoadedReference wrapper
    if hasattr(reference, "adata"):
        reference = reference.adata

    query_genes = set(query.var_names.astype(str))
    ref_genes = set(reference.var_names.astype(str))

    shared = query_genes & ref_genes
    missing_in_query = sorted(ref_genes - query_genes)[:max_missing_to_report]
    missing_in_reference = sorted(query_genes - ref_genes)[:max_missing_to_report]

    # Overlap fraction relative to reference (what fraction of reference genes are in query)
    overlap_fraction = len(shared) / max(len(ref_genes), 1)

    status = "complete"
    if overlap_fraction < min_overlap_threshold:
        status = f"low_overlap_warning (< {min_overlap_threshold:.0%})"

    report = FeatureOverlapReport(
        query_gene_count=len(query_genes),
        reference_gene_count=len(ref_genes),
        shared_gene_count=len(shared),
        overlap_fraction=overlap_fraction,
        missing_in_query=missing_in_query,
        missing_in_reference=missing_in_reference,
        status=status,
    )

    log.info(
        "Feature overlap: %d/%d query genes, %d/%d ref genes, %d shared (%.1f%%)",
        len(query_genes),
        len(query_genes),
        len(ref_genes),
        len(ref_genes),
        len(shared),
        overlap_fraction * 100,
    )

    return report


def compute_feature_overlap_from_paths(
    query_path: str | Path,
    reference_path: str | Path,
    *,
    min_overlap_threshold: float = 0.3,
) -> FeatureOverlapReport:
    """Compute feature overlap from file paths (memory-efficient).

    Uses backed mode to avoid loading full datasets.

    Parameters
    ----------
    query_path : str or Path
        Path to query h5ad file
    reference_path : str or Path
        Path to reference h5ad file
    min_overlap_threshold : float
        Minimum acceptable overlap fraction

    Returns
    -------
    FeatureOverlapReport
        Detailed overlap report
    """
    try:
        import anndata
    except ImportError as exc:
        raise ImportError("anndata required for feature overlap computation") from exc

    query_path = Path(query_path)
    reference_path = Path(reference_path)

    if not query_path.exists():
        return FeatureOverlapReport(
            query_gene_count=0,
            reference_gene_count=0,
            shared_gene_count=0,
            overlap_fraction=0.0,
            status=f"query_not_found: {query_path}",
        )

    if not reference_path.exists():
        return FeatureOverlapReport(
            query_gene_count=0,
            reference_gene_count=0,
            shared_gene_count=0,
            overlap_fraction=0.0,
            status=f"reference_not_found: {reference_path}",
        )

    # Load in backed mode for memory efficiency
    query = anndata.read_h5ad(query_path, backed="r")
    reference = anndata.read_h5ad(reference_path, backed="r")

    try:
        return compute_feature_overlap(
            query, reference, min_overlap_threshold=min_overlap_threshold
        )
    finally:
        # Clean up backed files
        try:
            query.file.close()
        except Exception:
            pass
        try:
            reference.file.close()
        except Exception:
            pass
