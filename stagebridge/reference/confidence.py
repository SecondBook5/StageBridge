"""Confidence scoring for reference mappings.

This module provides confidence metrics for evaluating the quality of
query-to-reference mappings. Confidence scores enable downstream systems
to weight or filter cells based on mapping reliability.

All mappings produce explicit uncertainty - never embeddings without quality metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from stagebridge.logging_utils import get_logger
from stagebridge.reference.schema import MappingResult, ReferenceNeighborhood

log = get_logger(__name__)


@dataclass
class ConfidenceScores:
    """Confidence scores for reference mappings.

    Contains per-cell confidence metrics for HLCA and LuCa mappings,
    along with aggregate statistics.
    """

    # Per-cell scores (0-1 scale, higher = more confident)
    hlca_confidence: np.ndarray  # Shape: (n_cells,)
    luca_confidence: np.ndarray  # Shape: (n_cells,)

    # Cell IDs for alignment
    cell_ids: np.ndarray

    # Aggregate statistics
    hlca_stats: dict[str, float] = field(default_factory=dict)
    luca_stats: dict[str, float] = field(default_factory=dict)

    # Quality flags
    hlca_low_confidence_count: int = 0
    luca_low_confidence_count: int = 0
    nan_count: int = 0

    @property
    def n_cells(self) -> int:
        """Number of scored cells."""
        return len(self.cell_ids)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame for export."""
        return pd.DataFrame(
            {
                "cell_id": self.cell_ids,
                "hlca_confidence": self.hlca_confidence,
                "luca_confidence": self.luca_confidence,
            }
        )

    def get_high_confidence_mask(
        self,
        hlca_threshold: float = 0.5,
        luca_threshold: float = 0.5,
        require_both: bool = False,
    ) -> np.ndarray:
        """Get boolean mask for high-confidence cells.

        Parameters
        ----------
        hlca_threshold : float
            Minimum HLCA confidence
        luca_threshold : float
            Minimum LuCa confidence
        require_both : bool
            If True, require both references above threshold.
            If False, require at least one.

        Returns
        -------
        np.ndarray
            Boolean mask of shape (n_cells,)
        """
        hlca_ok = self.hlca_confidence >= hlca_threshold
        luca_ok = self.luca_confidence >= luca_threshold

        if require_both:
            return hlca_ok & luca_ok
        return hlca_ok | luca_ok


def compute_hlca_confidence(
    mapping_result: MappingResult,
    *,
    neighborhood: ReferenceNeighborhood | None = None,
    distance_scale: float | None = None,
) -> np.ndarray:
    """Compute confidence scores for HLCA mapping.

    Confidence is based on:
    1. Distance to nearest reference neighbors (closer = more confident)
    2. Neighbor label consistency (if available)
    3. Reconstruction quality (if available)

    Parameters
    ----------
    mapping_result : MappingResult
        Result from map_to_hlca
    neighborhood : ReferenceNeighborhood, optional
        Pre-computed neighborhood for more detailed scoring
    distance_scale : float, optional
        Scale parameter for distance-to-confidence transform.
        If None, automatically determined from data.

    Returns
    -------
    np.ndarray
        Confidence scores in [0, 1], shape (n_cells,)
    """
    n_cells = mapping_result.n_cells
    confidence = np.ones(n_cells, dtype=np.float32)

    # Use neighbor distances if available
    distances = None
    if neighborhood is not None:
        distances = neighborhood.neighbor_distances.mean(axis=1)
    elif mapping_result.neighbor_distances is not None:
        distances = mapping_result.neighbor_distances

    if distances is not None:
        # Transform distance to confidence using exponential decay
        if distance_scale is None:
            # Use median distance as scale
            distance_scale = float(np.median(distances)) + 1e-6

        # Confidence = exp(-distance / scale)
        # Closer cells (small distance) get higher confidence
        confidence = np.exp(-distances / distance_scale)
        confidence = np.clip(confidence, 0.0, 1.0)

        log.debug(
            "HLCA confidence from distances: median=%.3f, scale=%.3f",
            float(np.median(distances)),
            distance_scale,
        )

    # Boost confidence for consistent neighbor labels
    if neighborhood is not None and neighborhood.neighbor_labels is not None:
        labels = neighborhood.neighbor_labels
        # Compute mode frequency (what fraction of neighbors have same label)
        label_consistency = np.zeros(n_cells, dtype=np.float32)
        for i in range(n_cells):
            cell_labels = labels[i]
            unique, counts = np.unique(cell_labels, return_counts=True)
            label_consistency[i] = counts.max() / len(cell_labels)

        # Combine: average of distance-based and label-based confidence
        confidence = 0.7 * confidence + 0.3 * label_consistency

    # Handle NaN values
    nan_mask = np.isnan(confidence)
    if nan_mask.any():
        log.warning(
            "HLCA confidence: %d NaN values replaced with 0.0",
            int(nan_mask.sum()),
        )
        confidence[nan_mask] = 0.0

    return confidence.astype(np.float32)


def compute_luca_confidence(
    mapping_result: MappingResult,
    *,
    neighborhood: ReferenceNeighborhood | None = None,
    distance_scale: float | None = None,
) -> np.ndarray:
    """Compute confidence scores for LuCa mapping.

    Same methodology as HLCA confidence, adapted for disease reference.

    Parameters
    ----------
    mapping_result : MappingResult
        Result from map_to_luca
    neighborhood : ReferenceNeighborhood, optional
        Pre-computed neighborhood for more detailed scoring
    distance_scale : float, optional
        Scale parameter for distance-to-confidence transform

    Returns
    -------
    np.ndarray
        Confidence scores in [0, 1], shape (n_cells,)
    """
    # Use same methodology as HLCA
    return compute_hlca_confidence(
        mapping_result,
        neighborhood=neighborhood,
        distance_scale=distance_scale,
    )


def compute_dual_confidence(
    hlca_result: MappingResult,
    luca_result: MappingResult,
    *,
    hlca_neighborhood: ReferenceNeighborhood | None = None,
    luca_neighborhood: ReferenceNeighborhood | None = None,
    low_confidence_threshold: float = 0.3,
) -> ConfidenceScores:
    """Compute confidence scores for both references.

    Parameters
    ----------
    hlca_result : MappingResult
        HLCA mapping result
    luca_result : MappingResult
        LuCa mapping result
    hlca_neighborhood : ReferenceNeighborhood, optional
        HLCA neighborhood for detailed scoring
    luca_neighborhood : ReferenceNeighborhood, optional
        LuCa neighborhood for detailed scoring
    low_confidence_threshold : float
        Threshold below which cells are flagged as low confidence

    Returns
    -------
    ConfidenceScores
        Combined confidence scores
    """
    hlca_conf = compute_hlca_confidence(hlca_result, neighborhood=hlca_neighborhood)
    luca_conf = compute_luca_confidence(luca_result, neighborhood=luca_neighborhood)

    # Compute statistics
    hlca_stats = _compute_confidence_stats(hlca_conf)
    luca_stats = _compute_confidence_stats(luca_conf)

    # Count low confidence and NaN
    hlca_low = int((hlca_conf < low_confidence_threshold).sum())
    luca_low = int((luca_conf < low_confidence_threshold).sum())
    nan_count = int(np.isnan(hlca_conf).sum() + np.isnan(luca_conf).sum())

    log.info(
        "Confidence scores: HLCA mean=%.3f (low=%d), LuCa mean=%.3f (low=%d)",
        hlca_stats["mean"],
        hlca_low,
        luca_stats["mean"],
        luca_low,
    )

    return ConfidenceScores(
        hlca_confidence=hlca_conf,
        luca_confidence=luca_conf,
        cell_ids=hlca_result.cell_ids,
        hlca_stats=hlca_stats,
        luca_stats=luca_stats,
        hlca_low_confidence_count=hlca_low,
        luca_low_confidence_count=luca_low,
        nan_count=nan_count,
    )


def _compute_confidence_stats(confidence: np.ndarray) -> dict[str, float]:
    """Compute summary statistics for confidence array."""
    valid = confidence[~np.isnan(confidence)]
    if len(valid) == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "q25": float("nan"),
            "q75": float("nan"),
        }
    return {
        "mean": float(np.mean(valid)),
        "std": float(np.std(valid)),
        "median": float(np.median(valid)),
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
        "q25": float(np.percentile(valid, 25)),
        "q75": float(np.percentile(valid, 75)),
    }


def detect_mapping_collapse(
    mapping_result: MappingResult,
    *,
    collapse_threshold: float = 0.01,
) -> dict[str, Any]:
    """Detect if mapping has collapsed to a small region.

    Mapping collapse occurs when all query cells map to nearly the same
    point in reference space, indicating a failure in the mapping process.

    Parameters
    ----------
    mapping_result : MappingResult
        Mapping result to check
    collapse_threshold : float
        Threshold for collapse detection (fraction of expected variance)

    Returns
    -------
    dict
        Collapse detection report
    """
    embeddings = mapping_result.embeddings

    # Compute variance per dimension
    var_per_dim = np.var(embeddings, axis=0)
    mean_var = float(np.mean(var_per_dim))
    max_var = float(np.max(var_per_dim))

    # Compute pairwise distances for sample
    n_sample = min(1000, embeddings.shape[0])
    if n_sample < embeddings.shape[0]:
        idx = np.random.choice(embeddings.shape[0], n_sample, replace=False)
        sample = embeddings[idx]
    else:
        sample = embeddings

    # Mean pairwise distance
    from scipy.spatial.distance import pdist

    pairwise_dists = pdist(sample)
    mean_pairwise_dist = float(np.mean(pairwise_dists))

    # Check for collapse
    is_collapsed = mean_var < collapse_threshold or mean_pairwise_dist < 0.1

    report = {
        "is_collapsed": is_collapsed,
        "mean_variance": mean_var,
        "max_variance": max_var,
        "mean_pairwise_distance": mean_pairwise_dist,
        "collapse_threshold": collapse_threshold,
        "n_cells": mapping_result.n_cells,
        "latent_dim": mapping_result.latent_dim,
    }

    if is_collapsed:
        log.error(
            "MAPPING COLLAPSE DETECTED for %s: mean_var=%.6f, mean_dist=%.6f. "
            "All cells mapped to nearly same point!",
            mapping_result.reference_name,
            mean_var,
            mean_pairwise_dist,
        )

    return report


def detect_nan_embeddings(
    mapping_result: MappingResult,
) -> dict[str, Any]:
    """Detect and report NaN values in embeddings.

    Parameters
    ----------
    mapping_result : MappingResult
        Mapping result to check

    Returns
    -------
    dict
        NaN detection report
    """
    embeddings = mapping_result.embeddings

    nan_mask = np.isnan(embeddings)
    nan_per_cell = nan_mask.sum(axis=1)
    nan_per_dim = nan_mask.sum(axis=0)

    cells_with_nan = int((nan_per_cell > 0).sum())
    dims_with_nan = int((nan_per_dim > 0).sum())
    total_nan = int(nan_mask.sum())

    report = {
        "has_nan": total_nan > 0,
        "total_nan_count": total_nan,
        "cells_with_nan": cells_with_nan,
        "dims_with_nan": dims_with_nan,
        "nan_fraction": total_nan / embeddings.size if embeddings.size > 0 else 0.0,
        "n_cells": mapping_result.n_cells,
        "latent_dim": mapping_result.latent_dim,
    }

    if total_nan > 0:
        log.error(
            "NaN VALUES DETECTED in %s embeddings: %d total (%d cells, %d dims)",
            mapping_result.reference_name,
            total_nan,
            cells_with_nan,
            dims_with_nan,
        )

    return report
