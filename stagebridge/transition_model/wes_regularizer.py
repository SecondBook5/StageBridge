"""Unified genomic niche encoder for cross-dataset conditioning.

Maps heterogeneous genomic features from different assays into a shared
niche embedding space:

  WES (Peng/LUAD dataset): 8 features — TMB, driver mutation binary flags
  lpWGS (Rossi/brain mets): 13 features — FGA, ploidy, CNA locus flags

The encoder handles missing modalities gracefully: if a patient only has
WES data (no lpWGS), the lpWGS features are zeroed and vice versa.
A modality indicator is concatenated so the model knows which assay
produced the features.

Architecture:
    wes_features (8) ──→ Linear(8, niche_dim) ──→┐
                                                   ├─→ fused (niche_dim)
    lpwgs_features (13) → Linear(13, niche_dim) ─→┘
                                                   + modality_emb
                                                   → LayerNorm → GELU
                                                   → Linear(niche_dim, niche_dim)

The fused niche embedding conditions the Schrödinger bridge drift via
FiLM or concatenation, just like the existing WES feature path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas

import torch
from torch import Tensor, nn


@dataclass
class GenomicNicheConfig:
    """Configuration for the unified genomic niche encoder."""

    wes_dim: int = 8  # WES feature dimensionality
    lpwgs_dim: int = 13  # lpWGS feature dimensionality
    niche_dim: int = 32  # shared niche embedding dimension
    dropout: float = 0.1
    num_modalities: int = 3  # 0=none, 1=wes, 2=lpwgs


@dataclass(slots=True, frozen=True)
class WESRegularizerConfig:
    """Configuration for WES-based transport regularization."""

    penalty_scale: float = 0.5
    normalize: bool = True


class GenomicNicheEncoder(nn.Module):
    """Encode heterogeneous genomic features into a unified niche vector.

    Handles two genomic modalities (WES somatic variants, lpWGS copy-number)
    from different datasets, projecting each into a shared space and fusing
    them with modality-aware gating.
    """

    def __init__(self, config: GenomicNicheConfig | None = None) -> None:
        super().__init__()
        if config is None:
            config = GenomicNicheConfig()
        self.config = config

        # Per-modality projection
        self.wes_proj = nn.Sequential(
            nn.Linear(config.wes_dim, config.niche_dim),
            nn.GELU(),
        )
        self.lpwgs_proj = nn.Sequential(
            nn.Linear(config.lpwgs_dim, config.niche_dim),
            nn.GELU(),
        )

        # Modality embedding: 0=none, 1=wes, 2=lpwgs
        self.modality_emb = nn.Embedding(config.num_modalities, config.niche_dim)

        # Fusion layer
        self.fuse = nn.Sequential(
            nn.LayerNorm(config.niche_dim),
            nn.Linear(config.niche_dim, config.niche_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

    def forward(
        self,
        wes_features: Tensor | None = None,
        lpwgs_features: Tensor | None = None,
        modality_id: Tensor | None = None,
    ) -> Tensor:
        """Encode genomic features into unified niche embedding.

        Parameters
        ----------
        wes_features : (batch, wes_dim) or None
        lpwgs_features : (batch, lpwgs_dim) or None
        modality_id : (batch,) long tensor — 0=none, 1=wes, 2=lpwgs
            If None, inferred from which features are provided.

        Returns
        -------
        (batch, niche_dim) niche embedding
        """
        if wes_features is not None:
            batch_size = wes_features.shape[0]
            device = wes_features.device
        elif lpwgs_features is not None:
            batch_size = lpwgs_features.shape[0]
            device = lpwgs_features.device
        else:
            raise ValueError("At least one of wes_features or lpwgs_features must be provided")

        # Project available modalities
        h = torch.zeros(batch_size, self.config.niche_dim, device=device)

        if wes_features is not None:
            h = h + self.wes_proj(wes_features)

        if lpwgs_features is not None:
            h = h + self.lpwgs_proj(lpwgs_features)

        # Add modality embedding
        if modality_id is None:
            if wes_features is not None and lpwgs_features is None:
                modality_id = torch.ones(batch_size, dtype=torch.long, device=device)
            elif lpwgs_features is not None and wes_features is None:
                modality_id = torch.full((batch_size,), 2, dtype=torch.long, device=device)
            else:
                modality_id = torch.zeros(batch_size, dtype=torch.long, device=device)

        h = h + self.modality_emb(modality_id)

        return self.fuse(h)


def build_niche_lookup(
    wes_df: "pandas.DataFrame | None" = None,
    lpwgs_df: "pandas.DataFrame | None" = None,
    wes_feature_cols: list[str] | None = None,
    lpwgs_feature_cols: list[str] | None = None,
) -> dict[str, dict[str, object]]:
    """Build a patient_id → niche feature dict for training-time lookup.

    Returns
    -------
    dict mapping patient_id → {
        "wes": np.ndarray (wes_dim,) or None,
        "lpwgs": np.ndarray (lpwgs_dim,) or None,
        "modality": int  — 0=none, 1=wes, 2=lpwgs
    }
    """
    import numpy as np

    lookup: dict[str, dict[str, object]] = {}

    if wes_df is not None:
        if wes_feature_cols is None:
            wes_feature_cols = [c for c in wes_df.columns if c not in ("patient_id", "stage")]
        for _, row in wes_df.iterrows():
            pid = row["patient_id"]
            if pid not in lookup:
                lookup[pid] = {"wes": None, "lpwgs": None, "modality": 0}
            lookup[pid]["wes"] = row[wes_feature_cols].values.astype(np.float32)
            lookup[pid]["modality"] = 1

    if lpwgs_df is not None:
        if lpwgs_feature_cols is None:
            lpwgs_feature_cols = [c for c in lpwgs_df.columns if c != "patient_id"]
        for _, row in lpwgs_df.iterrows():
            pid = row["patient_id"]
            if pid not in lookup:
                lookup[pid] = {"wes": None, "lpwgs": None, "modality": 0}
            lookup[pid]["lpwgs"] = row[lpwgs_feature_cols].values.astype(np.float32)
            # If already has WES, keep modality=1; otherwise set to 2
            if lookup[pid]["modality"] == 0:
                lookup[pid]["modality"] = 2

    return lookup


def lookup_wes_vectors(
    obs: Any,
    wes_lookup: dict[tuple[str, str], Any],
    *,
    donor_col: str = "donor_id",
    patient_col: str = "patient_id",
    stage_col: str = "stage",
) -> Tensor:
    """Align `(donor, stage)` WES vectors to observation rows, zero-filling missing pairs."""
    if hasattr(obs, "to_dict"):
        records = obs.to_dict(orient="records")
    else:
        records = list(obs)

    feature_dim = 0
    if wes_lookup:
        first_value = next(iter(wes_lookup.values()))
        feature_dim = int(len(first_value))

    vectors: list[Tensor] = []
    for row in records:
        donor_id = str(row.get(donor_col) or row.get(patient_col) or "")
        stage = str(row.get(stage_col) or "")
        value = wes_lookup.get((donor_id, stage))
        if value is None:
            vectors.append(torch.zeros(feature_dim, dtype=torch.float32))
        else:
            vectors.append(torch.as_tensor(value, dtype=torch.float32))

    if not vectors:
        return torch.zeros((0, feature_dim), dtype=torch.float32)
    return torch.stack(vectors, dim=0)


def pairwise_wes_penalty(
    src_wes: Tensor | Any,
    tgt_wes: Tensor | Any,
    *,
    penalty_scale: float = 0.5,
    normalize: bool = True,
) -> Tensor:
    """Compute a pairwise WES compatibility penalty for transport regularization."""
    src = src_wes if isinstance(src_wes, Tensor) else torch.as_tensor(src_wes, dtype=torch.float32)
    tgt = tgt_wes if isinstance(tgt_wes, Tensor) else torch.as_tensor(tgt_wes, dtype=torch.float32)
    if src.ndim != 2 or tgt.ndim != 2:
        raise ValueError(
            f"src_wes and tgt_wes must be 2D, got {tuple(src.shape)} and {tuple(tgt.shape)}."
        )
    if src.shape[1] != tgt.shape[1]:
        raise ValueError(
            "src_wes and tgt_wes must share the same feature dimension: "
            f"{src.shape[1]} != {tgt.shape[1]}."
        )

    penalty = torch.cdist(src, tgt, p=1)
    if normalize and src.shape[1] > 0:
        penalty = penalty / float(src.shape[1])
    return float(penalty_scale) * penalty
