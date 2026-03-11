"""Lesion-level baselines for EA-MIST."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from stagebridge.context_model.heads import LesionMultitaskHeads
from stagebridge.context_model.set_encoder import PMA, SAB


@dataclass(slots=True, frozen=True)
class LesionModelOutput:
    """Common output contract for lesion-level models."""

    lesion_embedding: Tensor
    stage_logits: Tensor
    displacement: Tensor
    edge_logits: Tensor | None = None
    attention_weights: Tensor | None = None
    niche_transition_scores: Tensor | None = None


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

    def __init__(self, input_dim: int, *, hidden_dim: int = 128, num_stage_classes: int = 5, num_edge_heads: int = 0, dropout: float = 0.1) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(int(input_dim) * 2, int(hidden_dim)),
            nn.GELU(),
            nn.LayerNorm(int(hidden_dim)),
            nn.Dropout(float(dropout)),
        )
        self.heads = LesionMultitaskHeads(int(hidden_dim), num_stage_classes=num_stage_classes, num_edge_heads=num_edge_heads, dropout=dropout)

    def forward(self, embeddings: Tensor, mask: Tensor) -> LesionModelOutput:
        mean = _masked_mean(embeddings, mask)
        maxv = _masked_max(embeddings, mask)
        lesion = self.input_proj(torch.cat([mean, maxv], dim=-1))
        task_output = self.heads(lesion)
        return LesionModelOutput(
            lesion_embedding=lesion,
            stage_logits=task_output.stage_logits,
            displacement=task_output.displacement,
            edge_logits=task_output.edge_logits,
        )


class DeepSetsLesionBaseline(nn.Module):
    """Deep Sets lesion baseline over local niche embeddings."""

    def __init__(self, input_dim: int, *, hidden_dim: int = 128, num_stage_classes: int = 5, num_edge_heads: int = 0, dropout: float = 0.1) -> None:
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
        self.heads = LesionMultitaskHeads(int(hidden_dim), num_stage_classes=num_stage_classes, num_edge_heads=num_edge_heads, dropout=dropout)

    def forward(self, embeddings: Tensor, mask: Tensor) -> LesionModelOutput:
        encoded = self.phi(embeddings)
        lesion = self.rho(torch.cat([_masked_mean(encoded, mask), _masked_max(encoded, mask)], dim=-1))
        task_output = self.heads(lesion)
        return LesionModelOutput(
            lesion_embedding=lesion,
            stage_logits=task_output.stage_logits,
            displacement=task_output.displacement,
            edge_logits=task_output.edge_logits,
        )


class LesionSetTransformerBaseline(nn.Module):
    """Lesion-level Set Transformer baseline without prototype bottleneck."""

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        num_stage_classes: int = 5,
        num_edge_heads: int = 0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(int(input_dim), int(hidden_dim))
        self.blocks = nn.ModuleList([SAB(dim=int(hidden_dim), num_heads=int(num_heads), dropout=float(dropout)) for _ in range(int(num_layers))])
        self.pool = PMA(dim=int(hidden_dim), num_heads=int(num_heads), num_seed_vectors=1, dropout=float(dropout))
        self.norm = nn.LayerNorm(int(hidden_dim))
        self.heads = LesionMultitaskHeads(int(hidden_dim), num_stage_classes=num_stage_classes, num_edge_heads=num_edge_heads, dropout=dropout)

    def forward(self, embeddings: Tensor, mask: Tensor, *, return_attention: bool = False) -> LesionModelOutput:
        hidden = self.input_proj(embeddings)
        attention = None
        for layer_idx, block in enumerate(self.blocks):
            if return_attention and layer_idx == len(self.blocks) - 1:
                hidden, attention = block(hidden, mask=mask, return_attention=True)
            else:
                hidden = block(hidden, mask=mask)
        pooled = self.pool(hidden, mask=mask)
        lesion = self.norm(pooled[:, 0, :])
        task_output = self.heads(lesion)
        return LesionModelOutput(
            lesion_embedding=lesion,
            stage_logits=task_output.stage_logits,
            displacement=task_output.displacement,
            edge_logits=task_output.edge_logits,
            attention_weights=attention,
        )
