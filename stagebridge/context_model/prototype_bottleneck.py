"""Prototype bottleneck for lesion-level niche motif compression."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(slots=True, frozen=True)
class PrototypeBottleneckOutput:
    """Structured output from the prototype bottleneck."""

    aligned_embeddings: Tensor
    assignment_weights: Tensor
    prototype_composition: Tensor
    prototype_bank: Tensor


class PrototypeBottleneck(nn.Module):
    """Learned motif bottleneck over local niche embeddings.

    Args:
        model_dim: Local niche embedding dimension.
        num_prototypes: Number of learned prototypes.
        sparse_assignment: When True, uses a straight-through top-k style sparse
            assignment mask. In v1 the default remains dense softmax.
        temperature: Softmax temperature for assignment logits.
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
            raise ValueError("PrototypeBottleneck requires positive model_dim and num_prototypes > 1.")
        self.model_dim = int(model_dim)
        self.num_prototypes = int(num_prototypes)
        self.sparse_assignment = bool(sparse_assignment)
        self.temperature = float(temperature)
        self.prototypes = nn.Parameter(torch.randn(self.num_prototypes, self.model_dim) * 0.02)

    def _normalize(self, x: Tensor) -> Tensor:
        """Apply L2 normalization for stable similarity scores."""
        return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    def get_assignment_weights(self, embeddings: Tensor) -> Tensor:
        """Return soft assignment weights with shape ``(..., K)``."""
        normalized_embeddings = self._normalize(embeddings)
        normalized_prototypes = self._normalize(self.prototypes)
        logits = torch.einsum("...d,kd->...k", normalized_embeddings, normalized_prototypes) / max(self.temperature, 1e-6)
        weights = logits.softmax(dim=-1)
        if not self.sparse_assignment:
            return weights

        # Apply a sparse straight-through mask while keeping backward stability.
        top_idx = weights.argmax(dim=-1, keepdim=True)
        sparse = torch.zeros_like(weights).scatter_(-1, top_idx, 1.0)
        return sparse + (weights - weights.detach())

    def get_prototype_occupancy(self, assignment_weights: Tensor, mask: Tensor | None = None) -> Tensor:
        """Return prototype occupancy counts or masses."""
        weights = assignment_weights
        if mask is not None:
            if mask.ndim != weights.ndim - 1:
                raise ValueError("mask must match assignment weights except for the prototype axis.")
            weights = weights * mask.unsqueeze(-1).to(weights.dtype)
        reduce_dims = tuple(range(weights.ndim - 1))
        return weights.sum(dim=reduce_dims)

    def export_lesion_prototype_composition(
        self,
        assignment_weights: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        """Return per-lesion mean prototype composition with shape ``(B, K)``."""
        if assignment_weights.ndim != 3:
            raise ValueError("assignment_weights must have shape (B, N, K) for lesion composition export.")
        weights = assignment_weights
        if mask is not None:
            weights = weights * mask.unsqueeze(-1).to(weights.dtype)
            denom = mask.sum(dim=1, keepdim=True).clamp_min(1).to(weights.dtype)
        else:
            denom = torch.full((weights.shape[0], 1), weights.shape[1], dtype=weights.dtype, device=weights.device)
        return weights.sum(dim=1) / denom

    def export_top_neighborhoods(self, assignment_weights: Tensor, *, top_k: int = 5) -> Tensor:
        """Return the top neighborhood indices per prototype for inspection."""
        if assignment_weights.ndim != 2:
            raise ValueError("export_top_neighborhoods expects assignment weights with shape (N, K).")
        _, indices = assignment_weights.topk(k=min(int(top_k), assignment_weights.shape[0]), dim=0)
        return indices.transpose(0, 1).contiguous()

    def forward(self, embeddings: Tensor, *, mask: Tensor | None = None) -> PrototypeBottleneckOutput:
        """Align embeddings to the learned prototype vocabulary."""
        if embeddings.ndim not in {2, 3}:
            raise ValueError(f"PrototypeBottleneck expected 2D or 3D embeddings, got shape={tuple(embeddings.shape)}")
        weights = self.get_assignment_weights(embeddings)
        aligned = torch.einsum("...k,kd->...d", weights, self.prototypes)
        if mask is not None:
            aligned = aligned * mask.unsqueeze(-1).to(aligned.dtype)
        if embeddings.ndim == 3:
            composition = self.export_lesion_prototype_composition(weights, mask=mask)
        else:
            composition = weights.mean(dim=0, keepdim=True)
        return PrototypeBottleneckOutput(
            aligned_embeddings=aligned,
            assignment_weights=weights,
            prototype_composition=composition,
            prototype_bank=self.prototypes,
        )


def prototype_diversity_loss(prototypes: Tensor) -> Tensor:
    """Encourage prototype diversity by penalizing similarity off the diagonal."""
    normalized = prototypes / prototypes.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    similarity = normalized @ normalized.transpose(0, 1)
    eye = torch.eye(similarity.shape[0], device=similarity.device, dtype=similarity.dtype)
    off_diag = similarity * (1.0 - eye)
    return off_diag.pow(2).mean()


def assignment_entropy_loss(assignment_weights: Tensor, *, target_entropy: float | None = None) -> Tensor:
    """Penalize overly diffuse assignment weights."""
    safe = assignment_weights.clamp_min(1e-8)
    entropy = -(safe * safe.log()).sum(dim=-1)
    if target_entropy is None:
        return entropy.mean()
    return (entropy - float(target_entropy)).pow(2).mean()


def prototype_orthogonality_loss(prototypes: Tensor) -> Tensor:
    """Encourage prototype vectors to be approximately orthogonal."""
    normalized = prototypes / prototypes.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    gram = normalized @ normalized.transpose(0, 1)
    eye = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    return (gram - eye).pow(2).mean()
