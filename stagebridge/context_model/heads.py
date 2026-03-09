"""Prediction heads for lesion-level EA-MIST models."""
from __future__ import annotations

import torch
from torch import Tensor, nn


class EdgeSpecificBinaryHeads(nn.Module):
    """Separate binary heads for `AAH->AIS` and `AIS->MIA` lesion prediction."""

    def __init__(self, model_dim: int, *, dropout: float = 0.1) -> None:
        super().__init__()
        self.aah_to_ais = nn.Sequential(
            nn.Linear(int(model_dim), int(model_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(model_dim), 1),
        )
        self.ais_to_mia = nn.Sequential(
            nn.Linear(int(model_dim), int(model_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(model_dim), 1),
        )

    def forward(self, lesion_embedding: Tensor) -> Tensor:
        """Return stacked logits with shape ``(B, 2)``."""
        return torch.cat(
            [
                self.aah_to_ais(lesion_embedding),
                self.ais_to_mia(lesion_embedding),
            ],
            dim=-1,
        )


def select_edge_logits(logits: Tensor, edge_ids: Tensor, *, aah_edge_id: int, ais_edge_id: int) -> Tensor:
    """Select one binary logit per lesion from a stacked edge-head output."""
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError(f"Expected logits with shape (B, 2), got {tuple(logits.shape)}.")
    out = torch.empty((logits.shape[0],), dtype=logits.dtype, device=logits.device)
    aah_mask = edge_ids == int(aah_edge_id)
    ais_mask = edge_ids == int(ais_edge_id)
    if (~(aah_mask | ais_mask)).any():
        unknown = torch.unique(edge_ids[~(aah_mask | ais_mask)]).tolist()
        raise ValueError(f"Encountered unsupported edge ids in select_edge_logits: {unknown}")
    out[aah_mask] = logits[aah_mask, 0]
    out[ais_mask] = logits[ais_mask, 1]
    return out
