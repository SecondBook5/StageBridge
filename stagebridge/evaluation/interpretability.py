"""Interpretability outputs for StageBridge model.

Extracts and structures attention-based interpretability signals:
- Niche influence (attention weights aggregated by cell type)
- Transition attributions
- Reference contributions (HLCA vs LuCA)

These are FIRST-CLASS model outputs, not post-hoc reconstructions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor


@dataclass
class NicheInfluence:
    """Niche influence attribution from attention weights.

    Aggregates attention weights by neighbor cell type to show
    which cell types the model attends to when making predictions.
    """
    cell_id: str
    stage: int | str
    neighbor_type: str
    attention_weight: float
    ring_idx: int | None = None  # Spatial ring if hierarchical


@dataclass
class CellInterpretability:
    """Complete interpretability output for a single cell."""
    cell_id: str
    stage: int | str

    # Attention weights per neighbor
    attention_weights: np.ndarray  # [K] - raw attention to each neighbor
    neighbor_cell_types: list[str]  # [K] - cell type of each neighbor
    neighbor_distances: np.ndarray  # [K] - distance to each neighbor

    # Aggregated by cell type
    type_influence: dict[str, float]  # cell_type -> mean attention

    # Model outputs
    transition_prob: float | None = None
    predicted_stage: int | None = None
    uncertainty: float | None = None

    # Reference contributions (if dual-reference)
    hlca_contribution: float | None = None
    luca_contribution: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "cell_id": self.cell_id,
            "stage": self.stage,
            "attention_weights": self.attention_weights.tolist(),
            "neighbor_cell_types": self.neighbor_cell_types,
            "neighbor_distances": self.neighbor_distances.tolist(),
            "type_influence": self.type_influence,
            "transition_prob": self.transition_prob,
            "predicted_stage": self.predicted_stage,
            "uncertainty": self.uncertainty,
            "hlca_contribution": self.hlca_contribution,
            "luca_contribution": self.luca_contribution,
        }


@dataclass
class BatchInterpretability:
    """Interpretability outputs for a batch of cells."""
    cells: list[CellInterpretability]

    # Batch-level summaries
    mean_attention_by_type: dict[str, float] = field(default_factory=dict)
    attention_entropy_mean: float = 0.0
    attention_entropy_std: float = 0.0

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame for analysis."""
        rows = []
        for cell in self.cells:
            for i, (ct, attn) in enumerate(zip(cell.neighbor_cell_types, cell.attention_weights)):
                rows.append({
                    "cell_id": cell.cell_id,
                    "stage": cell.stage,
                    "neighbor_idx": i,
                    "neighbor_type": ct,
                    "attention": attn,
                    "distance": cell.neighbor_distances[i] if i < len(cell.neighbor_distances) else None,
                    "transition_prob": cell.transition_prob,
                })
        return pd.DataFrame(rows)

    def get_type_influence_df(self) -> pd.DataFrame:
        """Get aggregated influence by cell type."""
        rows = []
        for cell in self.cells:
            for ct, influence in cell.type_influence.items():
                rows.append({
                    "cell_id": cell.cell_id,
                    "stage": cell.stage,
                    "neighbor_type": ct,
                    "influence": influence,
                    "transition_prob": cell.transition_prob,
                })
        return pd.DataFrame(rows)


def extract_attention_interpretability(
    attention_weights: Tensor,
    neighbor_cell_types: list[list[str]],
    neighbor_distances: Tensor,
    cell_ids: list[str],
    stages: list[int | str],
    transition_probs: Tensor | None = None,
) -> BatchInterpretability:
    """Extract interpretability from model attention weights.

    Args:
        attention_weights: [B, K] attention weights from niche encoder
        neighbor_cell_types: List of lists of cell types for each cell's neighbors
        neighbor_distances: [B, K] distances to neighbors
        cell_ids: List of cell IDs
        stages: List of stages
        transition_probs: [B] optional transition probabilities

    Returns:
        BatchInterpretability with per-cell and aggregated interpretability
    """
    B = attention_weights.shape[0]

    attn_np = attention_weights.detach().cpu().numpy()
    dist_np = neighbor_distances.detach().cpu().numpy()

    if transition_probs is not None:
        trans_np = transition_probs.detach().cpu().numpy()
    else:
        trans_np = [None] * B

    cells = []
    all_type_influence = {}
    entropies = []

    for i in range(B):
        attn = attn_np[i]
        cell_types = neighbor_cell_types[i] if i < len(neighbor_cell_types) else []
        dists = dist_np[i]

        # Aggregate by cell type
        type_influence = {}
        for j, ct in enumerate(cell_types):
            if j >= len(attn):
                break
            if ct not in type_influence:
                type_influence[ct] = []
            type_influence[ct].append(attn[j])

        # Mean per type
        type_influence = {ct: float(np.mean(vals)) for ct, vals in type_influence.items()}

        # Track global
        for ct, val in type_influence.items():
            if ct not in all_type_influence:
                all_type_influence[ct] = []
            all_type_influence[ct].append(val)

        # Compute entropy
        attn_valid = attn[attn > 0]
        if len(attn_valid) > 1:
            entropy = -np.sum(attn_valid * np.log(attn_valid + 1e-10))
            entropies.append(entropy)

        cell = CellInterpretability(
            cell_id=cell_ids[i] if i < len(cell_ids) else f"cell_{i}",
            stage=stages[i] if i < len(stages) else 0,
            attention_weights=attn,
            neighbor_cell_types=cell_types,
            neighbor_distances=dists,
            type_influence=type_influence,
            transition_prob=float(trans_np[i]) if trans_np[i] is not None else None,
        )
        cells.append(cell)

    # Batch-level summaries
    mean_by_type = {ct: float(np.mean(vals)) for ct, vals in all_type_influence.items()}

    return BatchInterpretability(
        cells=cells,
        mean_attention_by_type=mean_by_type,
        attention_entropy_mean=float(np.mean(entropies)) if entropies else 0.0,
        attention_entropy_std=float(np.std(entropies)) if entropies else 0.0,
    )


def compute_reference_contribution(
    hlca_embedding: Tensor,
    luca_embedding: Tensor,
    fused_embedding: Tensor,
) -> dict[str, float]:
    """Compute relative contribution of HLCA vs LuCA to fused embedding.

    Uses variance explained approach.

    Args:
        hlca_embedding: [B, D_hlca] HLCA embeddings
        luca_embedding: [B, D_luca] LuCA embeddings
        fused_embedding: [B, D_fused] fused embeddings

    Returns:
        Dictionary with hlca_contribution and luca_contribution (sum to 1)
    """
    hlca_var = hlca_embedding.var(dim=0).sum().item()
    luca_var = luca_embedding.var(dim=0).sum().item()
    total_var = hlca_var + luca_var

    if total_var < 1e-10:
        return {"hlca_contribution": 0.5, "luca_contribution": 0.5}

    return {
        "hlca_contribution": hlca_var / total_var,
        "luca_contribution": luca_var / total_var,
    }


def compute_ablation_importance(
    model,
    receiver: Tensor,
    neighbors: Tensor,
    distances: Tensor,
    baseline_output: Tensor,
) -> Tensor:
    """Compute importance of each neighbor via leave-one-out ablation.

    Args:
        model: Niche encoder model with ablate_neighbor method
        receiver: [B, D] receiver embeddings
        neighbors: [B, K, D] neighbor embeddings
        distances: [B, K] distances
        baseline_output: [B, D] output with all neighbors

    Returns:
        [B, K] importance scores (higher = more important)
    """
    B, K, _ = neighbors.shape
    importance = torch.zeros(B, K, device=neighbors.device)

    for k in range(K):
        # Ablate neighbor k
        mask = torch.ones(B, K, dtype=torch.bool, device=neighbors.device)
        mask[:, k] = False

        with torch.no_grad():
            ablated_output = model(receiver, neighbors, distances, neighbor_mask=mask)

        # Importance = change in output
        if hasattr(ablated_output, 'context'):
            delta = (baseline_output - ablated_output.context).norm(dim=-1)
        else:
            delta = (baseline_output - ablated_output).norm(dim=-1)

        importance[:, k] = delta

    # Normalize
    max_imp = importance.max(dim=-1, keepdim=True).values.clamp(min=1e-8)
    importance = importance / max_imp

    return importance
