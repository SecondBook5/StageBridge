"""Lesion-level baselines for EA-MIST."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from stagebridge.context_model.heads import EdgeSpecificBinaryHeads, select_edge_logits
from stagebridge.context_model.set_encoder import PMA, SAB
from stagebridge.transition_model.disease_edges import edge_id_map


@dataclass(slots=True, frozen=True)
class LesionModelOutput:
    """Common output contract for lesion-level models."""

    lesion_embedding: Tensor
    logits: Tensor
    selected_logits: Tensor
    attention_weights: Tensor | None = None


def _masked_mean(x: Tensor, mask: Tensor) -> Tensor:
    masked = x * mask.unsqueeze(-1).to(x.dtype)
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1).to(x.dtype)
    return masked.sum(dim=1) / denom


def _masked_max(x: Tensor, mask: Tensor) -> Tensor:
    neg_inf = torch.full_like(x, -1e9)
    masked = torch.where(mask.unsqueeze(-1), x, neg_inf)
    return masked.max(dim=1).values


class PooledLesionBaseline(nn.Module):
    """Pooled lesion summary baseline over local niche embeddings."""

    def __init__(self, input_dim: int, *, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(int(input_dim) * 2, int(hidden_dim)),
            nn.GELU(),
            nn.LayerNorm(int(hidden_dim)),
            nn.Dropout(float(dropout)),
        )
        self.heads = EdgeSpecificBinaryHeads(int(hidden_dim), dropout=dropout)
        self.edge_lookup = edge_id_map()

    def forward(self, embeddings: Tensor, mask: Tensor, edge_ids: Tensor) -> LesionModelOutput:
        mean = _masked_mean(embeddings, mask)
        maxv = _masked_max(embeddings, mask)
        lesion = self.input_proj(torch.cat([mean, maxv], dim=-1))
        logits = self.heads(lesion)
        selected = select_edge_logits(
            logits,
            edge_ids,
            aah_edge_id=self.edge_lookup["AAH->AIS"],
            ais_edge_id=self.edge_lookup["AIS->MIA"],
        )
        return LesionModelOutput(lesion_embedding=lesion, logits=logits, selected_logits=selected)


class DeepSetsLesionBaseline(nn.Module):
    """Deep Sets lesion baseline over local niche embeddings."""

    def __init__(self, input_dim: int, *, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.GELU(),
            nn.LayerNorm(int(hidden_dim)),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.LayerNorm(int(hidden_dim)),
        )
        self.rho = nn.Sequential(
            nn.Linear(int(hidden_dim) * 2, int(hidden_dim)),
            nn.GELU(),
            nn.LayerNorm(int(hidden_dim)),
            nn.Dropout(float(dropout)),
        )
        self.heads = EdgeSpecificBinaryHeads(int(hidden_dim), dropout=dropout)
        self.edge_lookup = edge_id_map()

    def forward(self, embeddings: Tensor, mask: Tensor, edge_ids: Tensor) -> LesionModelOutput:
        encoded = self.phi(embeddings)
        lesion = self.rho(torch.cat([_masked_mean(encoded, mask), _masked_max(encoded, mask)], dim=-1))
        logits = self.heads(lesion)
        selected = select_edge_logits(
            logits,
            edge_ids,
            aah_edge_id=self.edge_lookup["AAH->AIS"],
            ais_edge_id=self.edge_lookup["AIS->MIA"],
        )
        return LesionModelOutput(lesion_embedding=lesion, logits=logits, selected_logits=selected)


class LesionSetTransformerBaseline(nn.Module):
    """Lesion-level Set Transformer baseline without prototype bottleneck."""

    def __init__(self, input_dim: int, *, hidden_dim: int = 128, num_heads: int = 4, num_layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.input_proj = nn.Linear(int(input_dim), int(hidden_dim))
        self.blocks = nn.ModuleList([SAB(dim=int(hidden_dim), num_heads=int(num_heads), dropout=float(dropout)) for _ in range(int(num_layers))])
        self.pool = PMA(dim=int(hidden_dim), num_heads=int(num_heads), num_seed_vectors=1, dropout=float(dropout))
        self.norm = nn.LayerNorm(int(hidden_dim))
        self.heads = EdgeSpecificBinaryHeads(int(hidden_dim), dropout=dropout)
        self.edge_lookup = edge_id_map()

    def forward(self, embeddings: Tensor, mask: Tensor, edge_ids: Tensor, *, return_attention: bool = False) -> LesionModelOutput:
        hidden = self.input_proj(embeddings)
        attention = None
        for layer_idx, block in enumerate(self.blocks):
            if return_attention and layer_idx == len(self.blocks) - 1:
                hidden, attention = block(hidden, mask=mask, return_attention=True)
            else:
                hidden = block(hidden, mask=mask)
        pooled = self.pool(hidden, mask=mask)
        lesion = self.norm(pooled[:, 0, :])
        logits = self.heads(lesion)
        selected = select_edge_logits(
            logits,
            edge_ids,
            aah_edge_id=self.edge_lookup["AAH->AIS"],
            ais_edge_id=self.edge_lookup["AIS->MIA"],
        )
        return LesionModelOutput(
            lesion_embedding=lesion,
            logits=logits,
            selected_logits=selected,
            attention_weights=attention,
        )
