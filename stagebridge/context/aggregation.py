"""Hierarchical aggregation for sample-level representations.

Aggregates multiple niche embeddings per sample into a single sample-level
embedding using ISAB-based set transformers and optional prototype bottleneck.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from stagebridge.context.layers import ISAB, PMA


@dataclass(slots=True, frozen=True)
class PrototypeBottleneckOutput:
    """Output from prototype bottleneck."""
    aligned_embeddings: Tensor
    assignment_weights: Tensor
    prototype_composition: Tensor
    prototype_bank: Tensor


class PrototypeBottleneck(nn.Module):
    """Learned motif bottleneck over local niche embeddings.

    Routes niche embeddings through learned prototypes for interpretability.

    Args:
        model_dim: Niche embedding dimension
        num_prototypes: Number of learned prototypes
        sparse_assignment: Use sparse (top-1) assignment
        temperature: Softmax temperature
    """

    def __init__(
        self,
        model_dim: int,
        *,
        num_prototypes: int = 16,
        sparse_assignment: bool = False,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if model_dim <= 0 or num_prototypes <= 1:
            raise ValueError("Requires positive model_dim and num_prototypes > 1")

        self.model_dim = int(model_dim)
        self.num_prototypes = int(num_prototypes)
        self.sparse_assignment = bool(sparse_assignment)
        self.temperature = float(temperature)
        self.prototypes = nn.Parameter(torch.randn(self.num_prototypes, self.model_dim) * 0.02)

    def _normalize(self, x: Tensor) -> Tensor:
        return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    def get_assignment_weights(self, embeddings: Tensor) -> Tensor:
        """Return soft assignment weights with shape (..., K)."""
        normalized_embeddings = self._normalize(embeddings)
        normalized_prototypes = self._normalize(self.prototypes)
        logits = torch.einsum("...d,kd->...k", normalized_embeddings, normalized_prototypes)
        logits = logits / max(self.temperature, 1e-6)
        weights = logits.softmax(dim=-1)

        if not self.sparse_assignment:
            return weights

        top_idx = weights.argmax(dim=-1, keepdim=True)
        sparse = torch.zeros_like(weights).scatter_(-1, top_idx, 1.0)
        return sparse + (weights - weights.detach())

    def forward(self, embeddings: Tensor, *, mask: Tensor | None = None) -> PrototypeBottleneckOutput:
        """Align embeddings to learned prototypes."""
        if embeddings.ndim not in {2, 3}:
            raise ValueError(f"Expected 2D or 3D embeddings, got shape={tuple(embeddings.shape)}")

        weights = self.get_assignment_weights(embeddings)
        aligned = torch.einsum("...k,kd->...d", weights, self.prototypes)

        if mask is not None:
            aligned = aligned * mask.unsqueeze(-1).to(aligned.dtype)

        if embeddings.ndim == 3:
            if mask is not None:
                denom = mask.sum(dim=1, keepdim=True).clamp_min(1).to(weights.dtype)
                weights_masked = weights * mask.unsqueeze(-1).to(weights.dtype)
                composition = weights_masked.sum(dim=1) / denom
            else:
                composition = weights.mean(dim=1)
        else:
            composition = weights.mean(dim=0, keepdim=True)

        return PrototypeBottleneckOutput(
            aligned_embeddings=aligned,
            assignment_weights=weights,
            prototype_composition=composition,
            prototype_bank=self.prototypes,
        )


def prototype_diversity_loss(prototypes: Tensor) -> Tensor:
    """Encourage prototype diversity by penalizing off-diagonal similarity."""
    normalized = prototypes / prototypes.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    similarity = normalized @ normalized.transpose(0, 1)
    eye = torch.eye(similarity.shape[0], device=similarity.device, dtype=similarity.dtype)
    off_diag = similarity * (1.0 - eye)
    return off_diag.pow(2).mean()


class HierarchicalAggregator(nn.Module):
    """Aggregate multiple niche embeddings into sample-level representation.

    Uses ISAB-based hierarchical set transformer to aggregate N niche
    embeddings per sample into a single sample embedding.

    Args:
        hidden_dim: Niche embedding dimension
        num_heads: Number of attention heads
        num_layers: Number of ISAB layers
        num_inducing_points: ISAB inducing points
        dropout: Dropout rate
        use_prototypes: Route through prototype bottleneck
        num_prototypes: Number of prototypes
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        num_inducing_points: int = 16,
        dropout: float = 0.1,
        use_prototypes: bool = False,
        num_prototypes: int = 16,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_prototypes = use_prototypes

        if use_prototypes:
            self.prototype_bottleneck = PrototypeBottleneck(
                hidden_dim,
                num_prototypes=num_prototypes,
                sparse_assignment=False,
            )
        else:
            self.prototype_bottleneck = None

        self.isab_layers = nn.ModuleList([
            ISAB(
                dim=hidden_dim,
                num_heads=num_heads,
                num_inducing_points=num_inducing_points,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.pma = PMA(
            dim=hidden_dim,
            num_heads=num_heads,
            num_seed_vectors=1,
            dropout=dropout,
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        niche_embeddings: Tensor,
        mask: Tensor | None = None,
        return_attention: bool = False,
    ) -> dict:
        """Aggregate niche embeddings to sample-level.

        Args:
            niche_embeddings: [B, N, D] batch of N niches per sample
            mask: [B, N] boolean mask (True = valid niche)
            return_attention: Return attention weights

        Returns:
            dict with sample_embedding, prototype_output, attention_weights
        """
        B, N, D = niche_embeddings.shape
        prototype_output = None
        attention_weights = {}

        if self.prototype_bottleneck is not None:
            prototype_output = self.prototype_bottleneck(niche_embeddings, mask=mask)
            h = prototype_output.aligned_embeddings
        else:
            h = niche_embeddings

        for i, isab in enumerate(self.isab_layers):
            if return_attention and i == len(self.isab_layers) - 1:
                h, attn = isab(h, mask=mask, return_attention=True)
                attention_weights[f"isab_{i}"] = attn
            else:
                h = isab(h, mask=mask)

        if return_attention:
            pooled, pma_attn = self.pma(h, mask=mask, return_attention=True)
            attention_weights["pma"] = pma_attn
        else:
            pooled = self.pma(h, mask=mask)

        sample_embedding = self.norm(pooled[:, 0, :])

        return {
            "sample_embedding": sample_embedding,
            "prototype_output": prototype_output,
            "attention_weights": attention_weights if return_attention else None,
        }


class SampleLevelHeads(nn.Module):
    """Sample-level prediction heads.

    Predicts stage classification and displacement vector from
    aggregated sample embedding.

    Args:
        input_dim: Sample embedding dimension
        num_stage_classes: Number of stage classes
        dropout: Dropout rate
    """

    def __init__(
        self,
        input_dim: int = 128,
        num_stage_classes: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.stage_head = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, num_stage_classes),
        )

        self.displacement_head = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, input_dim),
        )

    def forward(self, sample_embedding: Tensor) -> dict:
        """Predict stage and displacement."""
        return {
            "stage_logits": self.stage_head(sample_embedding),
            "displacement": self.displacement_head(sample_embedding),
        }
