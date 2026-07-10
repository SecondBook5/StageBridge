"""Geometry and coupling diagnostics.

Interpretability metrics over semantic features and transport couplings:
effective rank of a feature cloud, entropy of a coupling, marginal error, and
mean drift/displacement alignment. All PyTorch, gradient-friendly where it makes
sense, and safe on degenerate (zero) inputs.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..contracts.errors import CCRTShapeError, CCRTValidationError

__all__ = [
    "effective_rank",
    "coupling_entropy",
    "coupling_marginal_error",
    "mean_drift_alignment",
]


def effective_rank(features: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    """Effective rank = exp(entropy of normalized singular values of centered X)."""
    if features.dim() != 2:
        raise CCRTShapeError(
            f"features must be rank 2 [N, D], got {tuple(features.shape)}"
        )
    centered = features - features.mean(dim=0, keepdim=True)
    svals = torch.linalg.svdvals(centered)
    total = svals.sum()
    if float(total) <= eps:
        return torch.zeros((), dtype=features.dtype, device=features.device)
    p = svals / total
    entropy = -(p * torch.log(p + eps)).sum()
    return torch.exp(entropy)


def coupling_entropy(
    coupling: torch.Tensor,
    *,
    normalize: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Shannon entropy of a coupling (optionally normalized to [0, 1])."""
    if coupling.dim() != 2:
        raise CCRTShapeError(
            f"coupling must be rank 2, got {tuple(coupling.shape)}"
        )
    if bool((coupling < 0).any()):
        raise CCRTValidationError("coupling must be non-negative")
    total = coupling.sum()
    if float(total) <= eps:
        return torch.zeros((), dtype=coupling.dtype, device=coupling.device)
    p = coupling / total
    entropy = -(p * torch.log(p + eps)).sum()
    if normalize:
        num_entries = coupling.numel()
        if num_entries <= 1:
            return torch.zeros((), dtype=coupling.dtype, device=coupling.device)
        entropy = entropy / torch.log(
            torch.tensor(float(num_entries), dtype=coupling.dtype, device=coupling.device)
        )
    return entropy


def coupling_marginal_error(
    *,
    coupling: torch.Tensor,
    source_weights: torch.Tensor,
    target_weights: torch.Tensor,
) -> torch.Tensor:
    """Maximum absolute discrepancy of coupling marginals from target weights."""
    if coupling.dim() != 2:
        raise CCRTShapeError("coupling must be rank 2")
    n, m = coupling.shape
    if source_weights.shape != (n,):
        raise CCRTShapeError(f"source_weights must be shape {(n,)}")
    if target_weights.shape != (m,):
        raise CCRTShapeError(f"target_weights must be shape {(m,)}")
    src_err = (coupling.sum(dim=1) - source_weights).abs().max()
    tgt_err = (coupling.sum(dim=0) - target_weights).abs().max()
    return torch.maximum(src_err, tgt_err)


def mean_drift_alignment(
    *,
    predicted_drift: torch.Tensor,
    target_displacement: torch.Tensor,
    weights: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Weighted mean row cosine between drift and target displacement.

    Rows whose target displacement is ~0 are excluded; if none remain, returns 0.
    """
    if predicted_drift.shape != target_displacement.shape:
        raise CCRTShapeError(
            "predicted_drift and target_displacement must share shape"
        )
    if predicted_drift.dim() != 2:
        raise CCRTShapeError("inputs must be rank 2 [N, D]")
    n = predicted_drift.shape[0]

    cos = F.cosine_similarity(predicted_drift, target_displacement, dim=-1, eps=eps)
    active = target_displacement.norm(dim=-1) > eps
    active_f = active.to(cos.dtype)

    if weights is None:
        w = torch.ones(n, dtype=cos.dtype, device=cos.device)
    else:
        if weights.shape != (n,):
            raise CCRTShapeError(f"weights must be shape {(n,)}")
        w = weights.to(cos.dtype)

    eff_w = w * active_f
    denom = eff_w.sum()
    if float(denom) <= eps:
        return torch.zeros((), dtype=cos.dtype, device=cos.device)
    return (cos * eff_w).sum() / denom
