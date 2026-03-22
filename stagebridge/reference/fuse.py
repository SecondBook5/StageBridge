"""Dual-reference fusion for combining HLCA and LuCa embeddings.

This module provides methods to fuse embeddings from multiple reference
mappings into a unified representation that captures both healthy structure
(from HLCA) and disease-aware structure (from LuCa).

Fusion methods:
- concat: Simple concatenation
- average: Element-wise average (requires same dimensions)
- weighted: Confidence-weighted combination
- learned: Placeholder for learned fusion (future)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from stagebridge.logging_utils import get_logger
from stagebridge.reference.schema import MappingResult

log = get_logger(__name__)


@dataclass
class FusedEmbeddingResult:
    """Result of fusing dual-reference embeddings.

    Contains the fused latent representation along with metadata
    and per-reference embeddings for downstream analysis.
    """

    # Fused embedding
    fused_embeddings: np.ndarray  # Shape: (n_cells, fused_dim)
    fused_dim: int

    # Per-reference embeddings (for inspection/debugging)
    hlca_embeddings: np.ndarray
    luca_embeddings: np.ndarray
    hlca_dim: int
    luca_dim: int

    # Cell metadata
    cell_ids: np.ndarray
    donor_ids: np.ndarray
    sample_ids: np.ndarray
    stage_ids: np.ndarray

    # Fusion info
    fusion_method: str
    fusion_params: dict[str, Any] = field(default_factory=dict)

    # Mode selection (which reference was primary)
    reference_mode_used: np.ndarray | None = None  # "hlca", "luca", or "both"

    @property
    def n_cells(self) -> int:
        """Number of fused cells."""
        return self.fused_embeddings.shape[0]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame with standardized schema.

        Returns DataFrame with columns:
        - cell_id, donor_id, sample_id, stage_id
        - hlca_latent_0, ..., hlca_latent_{k-1}
        - luca_latent_0, ..., luca_latent_{k-1}
        - fused_latent_0, ..., fused_latent_{k-1}
        - reference_mode_used
        """
        df = pd.DataFrame(
            {
                "cell_id": self.cell_ids,
                "donor_id": self.donor_ids,
                "sample_id": self.sample_ids,
                "stage_id": self.stage_ids,
            }
        )

        # HLCA latent columns
        for i in range(self.hlca_dim):
            df[f"hlca_latent_{i}"] = self.hlca_embeddings[:, i]

        # LuCa latent columns
        for i in range(self.luca_dim):
            df[f"luca_latent_{i}"] = self.luca_embeddings[:, i]

        # Fused latent columns
        for i in range(self.fused_dim):
            df[f"fused_latent_{i}"] = self.fused_embeddings[:, i]

        # Reference mode
        if self.reference_mode_used is not None:
            df["reference_mode_used"] = self.reference_mode_used
        else:
            df["reference_mode_used"] = "both"

        return df


def fuse_dual_reference(
    hlca_result: MappingResult,
    luca_result: MappingResult,
    *,
    method: Literal["concat", "average", "weighted", "learned"] = "concat",
    hlca_confidence: np.ndarray | None = None,
    luca_confidence: np.ndarray | None = None,
    normalize: bool = True,
    learned_hlca_weight: float = 0.5,
    learned_output_dim: int | None = None,
) -> FusedEmbeddingResult:
    """Fuse HLCA and LuCa embeddings into unified representation.

    Parameters
    ----------
    hlca_result : MappingResult
        Mapping result from HLCA reference
    luca_result : MappingResult
        Mapping result from LuCa reference
    method : str
        Fusion method:
        - "concat": Concatenate embeddings [hlca | luca]
        - "average": Element-wise average (requires same dimensions)
        - "weighted": Confidence-weighted average
        - "learned": Placeholder for learned fusion
    hlca_confidence : np.ndarray, optional
        Per-cell confidence scores for HLCA mapping (for weighted fusion)
    luca_confidence : np.ndarray, optional
        Per-cell confidence scores for LuCa mapping (for weighted fusion)
    normalize : bool
        Whether to normalize fused embeddings (z-score per dimension)

    Returns
    -------
    FusedEmbeddingResult
        Fused embedding result with metadata

    Raises
    ------
    ValueError
        If cell IDs don't match between results
        If dimensions don't match for average/weighted methods
    """
    # Validate cell alignment
    if not np.array_equal(hlca_result.cell_ids, luca_result.cell_ids):
        raise ValueError(
            "Cell IDs must match between HLCA and LuCa mapping results. "
            "Ensure both mappings use the same query data."
        )

    n_cells = hlca_result.n_cells
    hlca_emb = hlca_result.embeddings.astype(np.float32)
    luca_emb = luca_result.embeddings.astype(np.float32)

    # Check for NaN values
    hlca_nan_count = int(np.sum(np.isnan(hlca_emb)))
    luca_nan_count = int(np.sum(np.isnan(luca_emb)))
    if hlca_nan_count > 0 or luca_nan_count > 0:
        log.warning(
            "NaN values detected: HLCA=%d, LuCa=%d. These will propagate to fused embeddings.",
            hlca_nan_count,
            luca_nan_count,
        )

    if method == "concat":
        fused, ref_mode = _fuse_concat(hlca_emb, luca_emb)
    elif method == "average":
        fused, ref_mode = _fuse_average(hlca_emb, luca_emb)
    elif method == "weighted":
        fused, ref_mode = _fuse_weighted(hlca_emb, luca_emb, hlca_confidence, luca_confidence)
    elif method == "learned":
        fused, ref_mode = _fuse_learned(
            hlca_emb, luca_emb,
            hlca_weight=learned_hlca_weight,
            output_dim=learned_output_dim,
        )
    else:
        raise ValueError(f"Unknown fusion method: {method}")

    if normalize:
        fused = _normalize_embeddings(fused)

    return FusedEmbeddingResult(
        fused_embeddings=fused,
        fused_dim=fused.shape[1],
        hlca_embeddings=hlca_emb,
        luca_embeddings=luca_emb,
        hlca_dim=hlca_emb.shape[1],
        luca_dim=luca_emb.shape[1],
        cell_ids=hlca_result.cell_ids,
        donor_ids=hlca_result.donor_ids,
        sample_ids=hlca_result.sample_ids,
        stage_ids=hlca_result.stage_ids,
        fusion_method=method,
        fusion_params={"normalize": normalize},
        reference_mode_used=ref_mode,
    )


def _fuse_concat(
    hlca_emb: np.ndarray,
    luca_emb: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate embeddings: [hlca | luca]."""
    fused = np.concatenate([hlca_emb, luca_emb], axis=1)
    ref_mode = np.full(hlca_emb.shape[0], "both", dtype=object)
    log.info(
        "Concatenation fusion: HLCA(%d) + LuCa(%d) = %d dims",
        hlca_emb.shape[1],
        luca_emb.shape[1],
        fused.shape[1],
    )
    return fused, ref_mode


def _fuse_average(
    hlca_emb: np.ndarray,
    luca_emb: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Element-wise average of embeddings."""
    if hlca_emb.shape[1] != luca_emb.shape[1]:
        raise ValueError(
            f"Average fusion requires same dimensions. "
            f"Got HLCA={hlca_emb.shape[1]}, LuCa={luca_emb.shape[1]}"
        )
    fused = (hlca_emb + luca_emb) / 2.0
    ref_mode = np.full(hlca_emb.shape[0], "both", dtype=object)
    log.info("Average fusion: %d dims", fused.shape[1])
    return fused, ref_mode


def _fuse_weighted(
    hlca_emb: np.ndarray,
    luca_emb: np.ndarray,
    hlca_conf: np.ndarray | None,
    luca_conf: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Confidence-weighted fusion of embeddings."""
    if hlca_emb.shape[1] != luca_emb.shape[1]:
        raise ValueError(
            f"Weighted fusion requires same dimensions. "
            f"Got HLCA={hlca_emb.shape[1]}, LuCa={luca_emb.shape[1]}"
        )

    n_cells = hlca_emb.shape[0]

    # Default to equal weights if confidence not provided
    if hlca_conf is None:
        hlca_conf = np.ones(n_cells, dtype=np.float32)
    if luca_conf is None:
        luca_conf = np.ones(n_cells, dtype=np.float32)

    # Normalize weights
    total = hlca_conf + luca_conf + 1e-8
    w_hlca = hlca_conf / total
    w_luca = luca_conf / total

    # Weighted average
    fused = w_hlca[:, np.newaxis] * hlca_emb + w_luca[:, np.newaxis] * luca_emb

    # Determine primary reference per cell
    ref_mode = np.where(w_hlca > w_luca, "hlca", "luca")
    ref_mode = np.where(np.abs(w_hlca - w_luca) < 0.1, "both", ref_mode)

    log.info(
        "Weighted fusion: %d dims, HLCA-dominant=%d, LuCa-dominant=%d, balanced=%d",
        fused.shape[1],
        int((ref_mode == "hlca").sum()),
        int((ref_mode == "luca").sum()),
        int((ref_mode == "both").sum()),
    )

    return fused, ref_mode


def _fuse_learned(
    hlca_emb: np.ndarray,
    luca_emb: np.ndarray,
    *,
    hlca_weight: float = 0.5,
    output_dim: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Learned weighted fusion of embeddings.

    Uses a simple learned weight approach:
    - Projects both embeddings to same dimension if needed
    - Applies learned weight: fused = w * hlca_proj + (1-w) * luca_proj

    For full learned fusion with gradients, use LearnedFusionModule (PyTorch).
    This NumPy version uses fixed weights for inference.

    Parameters
    ----------
    hlca_emb : np.ndarray
        HLCA embeddings (n_cells, hlca_dim)
    luca_emb : np.ndarray
        LuCA embeddings (n_cells, luca_dim)
    hlca_weight : float
        Weight for HLCA component (0-1). LuCA weight = 1 - hlca_weight.
    output_dim : int, optional
        Output dimension. If None, uses max of input dims.

    Returns
    -------
    fused : np.ndarray
        Fused embeddings
    ref_mode : np.ndarray
        Reference mode per cell
    """
    n_cells = hlca_emb.shape[0]
    hlca_dim = hlca_emb.shape[1]
    luca_dim = luca_emb.shape[1]

    if output_dim is None:
        output_dim = max(hlca_dim, luca_dim)

    # L2 normalize inputs
    hlca_norm = hlca_emb / (np.linalg.norm(hlca_emb, axis=1, keepdims=True) + 1e-8)
    luca_norm = luca_emb / (np.linalg.norm(luca_emb, axis=1, keepdims=True) + 1e-8)

    # Project to common dimension via zero-padding or truncation
    if hlca_dim < output_dim:
        hlca_proj = np.zeros((n_cells, output_dim), dtype=np.float32)
        hlca_proj[:, :hlca_dim] = hlca_norm
    else:
        hlca_proj = hlca_norm[:, :output_dim]

    if luca_dim < output_dim:
        luca_proj = np.zeros((n_cells, output_dim), dtype=np.float32)
        luca_proj[:, :luca_dim] = luca_norm
    else:
        luca_proj = luca_norm[:, :output_dim]

    # Weighted combination
    w = np.clip(hlca_weight, 0.0, 1.0)
    fused = w * hlca_proj + (1.0 - w) * luca_proj

    # Determine reference mode
    ref_mode = np.where(w > 0.6, "hlca", np.where(w < 0.4, "luca", "both"))
    ref_mode = np.full(n_cells, ref_mode if isinstance(ref_mode, str) else "both", dtype=object)

    log.info(
        "Learned fusion: HLCA(%d) + LuCA(%d) -> %d dims, weight=%.2f",
        hlca_dim, luca_dim, output_dim, w
    )

    return fused.astype(np.float32), ref_mode


def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Z-score normalize embeddings per dimension."""
    mu = np.nanmean(embeddings, axis=0, keepdims=True)
    std = np.nanstd(embeddings, axis=0, keepdims=True) + 1e-8
    return ((embeddings - mu) / std).astype(np.float32)


def fuse_single_reference(
    mapping_result: MappingResult,
    reference_name: Literal["hlca", "luca"],
    *,
    target_dim: int | None = None,
    normalize: bool = True,
) -> FusedEmbeddingResult:
    """Create fused result from single reference (for fallback scenarios).

    Parameters
    ----------
    mapping_result : MappingResult
        Mapping result from single reference
    reference_name : str
        Which reference was used ("hlca" or "luca")
    target_dim : int, optional
        Target dimension for output. If provided, will pad with zeros.
    normalize : bool
        Whether to normalize embeddings

    Returns
    -------
    FusedEmbeddingResult
        Fused result with single reference
    """
    emb = mapping_result.embeddings.astype(np.float32)
    n_cells = emb.shape[0]
    dim = emb.shape[1]

    if target_dim and target_dim > dim:
        padded = np.zeros((n_cells, target_dim), dtype=np.float32)
        padded[:, :dim] = emb
        emb = padded
        dim = target_dim

    # Create dummy embedding for missing reference
    dummy = np.zeros_like(emb)

    if reference_name == "hlca":
        hlca_emb = emb
        luca_emb = dummy
    else:
        hlca_emb = dummy
        luca_emb = emb

    fused = emb.copy()
    if normalize:
        fused = _normalize_embeddings(fused)

    return FusedEmbeddingResult(
        fused_embeddings=fused,
        fused_dim=fused.shape[1],
        hlca_embeddings=hlca_emb,
        luca_embeddings=luca_emb,
        hlca_dim=hlca_emb.shape[1],
        luca_dim=luca_emb.shape[1],
        cell_ids=mapping_result.cell_ids,
        donor_ids=mapping_result.donor_ids,
        sample_ids=mapping_result.sample_ids,
        stage_ids=mapping_result.stage_ids,
        fusion_method=f"single_{reference_name}",
        fusion_params={"normalize": normalize},
        reference_mode_used=np.full(n_cells, reference_name, dtype=object),
    )
