"""
Base classes for spatial mapping backends.

Defines standardized interface and output format for all spatial mapping methods.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import anndata as ad
import numpy as np
from ..utils.data_cache import get_data_cache


@dataclass
class BackendMappingResult:
    """
    Standardized output from spatial mapping backends.

    All backends must produce this format for downstream compatibility.
    """

    # Cell type proportions per spot
    cell_type_proportions: pd.DataFrame  # (n_spots, n_celltypes)

    # Mapping confidence scores
    confidence: pd.Series  # (n_spots,) - per-spot confidence

    # Upstream quality metrics
    upstream_metrics: dict[str, float]

    # Backend-specific metadata
    metadata: dict[str, Any]

    # Optional: Cell-level assignments (if backend supports)
    cell_assignments: pd.DataFrame | None = None  # (n_cells, n_spots) or None

    # Optional: Gene expression reconstruction
    reconstructed_expression: pd.DataFrame | None = None  # (n_spots, n_genes)

    def save(self, output_dir: Path):
        """Save results to standardized format."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save main outputs
        self.cell_type_proportions.to_parquet(output_dir / "cell_type_proportions.parquet")
        self.confidence.to_frame("confidence").to_parquet(
            output_dir / "mapping_confidence.parquet"
        )

        # Save metrics as JSON
        import json

        with open(output_dir / "upstream_metrics.json", "w") as f:
            json.dump(self.upstream_metrics, f, indent=2)

        with open(output_dir / "backend_metadata.json", "w") as f:
            json.dump(self.metadata, f, indent=2)

        # Save optional outputs
        if self.cell_assignments is not None:
            self.cell_assignments.to_parquet(output_dir / "cell_assignments.parquet")

        if self.reconstructed_expression is not None:
            self.reconstructed_expression.to_parquet(
                output_dir / "reconstructed_expression.parquet"
            )

    @classmethod
    def load(cls, output_dir: Path, use_cache: bool = True) -> "BackendMappingResult":
        """Load results from standardized format (with optional caching)."""
        output_dir = Path(output_dir)
        cache = get_data_cache() if use_cache else None

        # Load main outputs (OPTIMIZED: Use cache to avoid redundant reads)
        if cache:
            cell_type_proportions = cache.read_parquet(
                output_dir / "cell_type_proportions.parquet"
            )
            confidence = cache.read_parquet(output_dir / "mapping_confidence.parquet")[
                "confidence"
            ]
        else:
            cell_type_proportions = pd.read_parquet(output_dir / "cell_type_proportions.parquet")
            confidence = pd.read_parquet(output_dir / "mapping_confidence.parquet")["confidence"]

        # Load metrics
        import json

        with open(output_dir / "upstream_metrics.json") as f:
            upstream_metrics = json.load(f)

        with open(output_dir / "backend_metadata.json") as f:
            metadata = json.load(f)

        # Load optional outputs
        cell_assignments = None
        if (output_dir / "cell_assignments.parquet").exists():
            if cache:
                cell_assignments = cache.read_parquet(output_dir / "cell_assignments.parquet")
            else:
                cell_assignments = pd.read_parquet(output_dir / "cell_assignments.parquet")

        reconstructed_expression = None
        if (output_dir / "reconstructed_expression.parquet").exists():
            if cache:
                reconstructed_expression = cache.read_parquet(
                    output_dir / "reconstructed_expression.parquet"
                )
            else:
                reconstructed_expression = pd.read_parquet(
                    output_dir / "reconstructed_expression.parquet"
                )

        return cls(
            cell_type_proportions=cell_type_proportions,
            confidence=confidence,
            upstream_metrics=upstream_metrics,
            metadata=metadata,
            cell_assignments=cell_assignments,
            reconstructed_expression=reconstructed_expression,
        )


class SpatialBackend(ABC):
    """
    Abstract base class for spatial mapping backends.

    All backends must implement:
    - map(): Run spatial mapping
    - compute_upstream_metrics(): Compute quality metrics
    - estimate_confidence(): Estimate per-spot confidence

    Backends should be stateless - all configuration in __init__,
    all outputs returned from map().
    """

    def __init__(self, **kwargs):
        """Initialize backend with configuration."""
        self.config = kwargs

    @abstractmethod
    def map(
        self,
        snrna: ad.AnnData,
        spatial: ad.AnnData,
        output_dir: Path | None = None,
    ) -> BackendMappingResult:
        """
        Run spatial mapping.

        Args:
            snrna: Single-cell reference (anndata with .X, .obs['cell_type'])
            spatial: Spatial data (anndata with .X, .obsm['spatial'])
            output_dir: Optional directory to save intermediate results

        Returns:
            BackendMappingResult with standardized outputs
        """
        pass

    @abstractmethod
    def compute_upstream_metrics(
        self,
        snrna: ad.AnnData,
        spatial: ad.AnnData,
        result: BackendMappingResult,
    ) -> dict[str, float]:
        """
        Compute upstream quality metrics.

        Metrics to include:
        - Gene reconstruction error (if applicable)
        - Cell type entropy (diversity)
        - Coverage (fraction of spots with confident mapping)
        - Sparsity (fraction of zero proportions)

        Args:
            snrna: Single-cell reference
            spatial: Spatial data
            result: Mapping result

        Returns:
            Dictionary of metric name → value
        """
        pass

    @abstractmethod
    def estimate_confidence(
        self,
        snrna: ad.AnnData,
        spatial: ad.AnnData,
        result: BackendMappingResult,
    ) -> pd.Series:
        """
        Estimate per-spot mapping confidence.

        Confidence should be in [0, 1] where:
        - 1.0 = highly confident mapping
        - 0.0 = low confidence / uncertain

        Args:
            snrna: Single-cell reference
            spatial: Spatial data
            result: Mapping result (before confidence is set)

        Returns:
            Series of confidence scores indexed by spot ID
        """
        pass

    def validate_inputs(
        self,
        snrna: ad.AnnData,
        spatial: ad.AnnData,
    ):
        """
        Validate input data format.

        Checks:
        - snrna has .obs['cell_type']
        - spatial has .obsm['spatial']
        - Genes overlap exists
        """
        # Check cell types
        if "cell_type" not in snrna.obs.columns:
            raise ValueError("snrna must have .obs['cell_type']")

        # Check spatial coordinates
        if "spatial" not in spatial.obsm.keys():
            raise ValueError("spatial must have .obsm['spatial']")

        # Check gene overlap
        common_genes = snrna.var_names.intersection(spatial.var_names)
        if len(common_genes) == 0:
            raise ValueError("No overlapping genes between snrna and spatial")

        overlap_frac = len(common_genes) / len(snrna.var_names)
        if overlap_frac < 0.1:
            import warnings

            warnings.warn(
                f"Low gene overlap: {overlap_frac:.1%} "
                f"({len(common_genes)}/{len(snrna.var_names)} genes)",
                stacklevel=2,
            )

    def preprocess(
        self,
        snrna: ad.AnnData,
        spatial: ad.AnnData,
    ) -> tuple[ad.AnnData, ad.AnnData]:
        """
        Preprocess data for mapping.

        - Subset to common genes
        - Ensure correct format
        - Normalize if needed

        Returns:
            Preprocessed (snrna, spatial) tuple
        """
        # Subset to common genes
        common_genes = snrna.var_names.intersection(spatial.var_names)
        snrna = snrna[:, common_genes].copy()
        spatial = spatial[:, common_genes].copy()

        return snrna, spatial


def compute_cell_type_entropy(proportions: pd.DataFrame) -> pd.Series:
    """
    Compute Shannon entropy of cell type proportions per spot.

    High entropy = diverse mixture
    Low entropy = dominated by one cell type

    Args:
        proportions: (n_spots, n_celltypes) with values in [0, 1]

    Returns:
        Series of entropy values per spot
    """
    # Avoid log(0)
    p = proportions.values + 1e-10
    p = p / p.sum(axis=1, keepdims=True)

    entropy = -np.sum(p * np.log(p), axis=1) / np.log(proportions.shape[1])
    return pd.Series(entropy, index=proportions.index, name="entropy")


def compute_sparsity(proportions: pd.DataFrame) -> float:
    """
    Compute sparsity (fraction of zeros) in proportion matrix.

    Args:
        proportions: (n_spots, n_celltypes)

    Returns:
        Sparsity fraction in [0, 1]
    """
    return (proportions.values == 0).mean()
