"""Transport stability metrics.

Compare two couplings, two displacement fields, or the pairwise geometry of two
feature spaces — used to assess how stable transport / drift estimates are across
perturbations, seeds, or representations. All PyTorch; safe on zero inputs.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..contracts.errors import CCRTShapeError, CCRTValidationError

__all__ = [
    "coupling_frobenius_distance",
    "displacement_cosine_stability",
    "feature_geometry_alignment",
]


def _check_finite(t: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(t).all()):
        raise CCRTValidationError(f"{name} contains non-finite values")


def coupling_frobenius_distance(
    coupling_a: torch.Tensor,
    coupling_b: torch.Tensor,
    *,
    normalize: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Frobenius distance between two couplings (optionally Frobenius-normalized)."""
    if coupling_a.shape != coupling_b.shape:
        raise CCRTShapeError("couplings must share shape")
    if coupling_a.dim() != 2:
        raise CCRTShapeError("couplings must be rank 2")
    _check_finite(coupling_a, "coupling_a")
    _check_finite(coupling_b, "coupling_b")
    if bool((coupling_a < 0).any()) or bool((coupling_b < 0).any()):
        raise CCRTValidationError("couplings must be non-negative")

    a, b = coupling_a, coupling_b
    if normalize:
        a = a / (a.norm() + eps)
        b = b / (b.norm() + eps)
    return (a - b).norm()


def displacement_cosine_stability(
    displacement_a: torch.Tensor,
    displacement_b: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Weighted mean row cosine between two displacement fields [N, D].

    Rows where both displacements are ~0 are excluded; if none remain, returns 0.
    """
    if displacement_a.shape != displacement_b.shape:
        raise CCRTShapeError("displacement fields must share shape")
    if displacement_a.dim() != 2:
        raise CCRTShapeError("displacement fields must be rank 2 [N, D]")
    _check_finite(displacement_a, "displacement_a")
    _check_finite(displacement_b, "displacement_b")
    n = displacement_a.shape[0]

    cos = F.cosine_similarity(displacement_a, displacement_b, dim=-1, eps=eps)
    both_zero = (displacement_a.norm(dim=-1) <= eps) & (
        displacement_b.norm(dim=-1) <= eps
    )
    active = (~both_zero).to(cos.dtype)

    if weights is None:
        w = torch.ones(n, dtype=cos.dtype, device=cos.device)
    else:
        if weights.shape != (n,):
            raise CCRTShapeError(f"weights must be shape {(n,)}")
        w = weights.to(cos.dtype)

    eff_w = w * active
    denom = eff_w.sum()
    if float(denom) <= eps:
        return torch.zeros((), dtype=cos.dtype, device=cos.device)
    return (cos * eff_w).sum() / denom


def _condensed_pairwise_distances(features: torch.Tensor) -> torch.Tensor:
    """Upper-triangle (i<j) pairwise Euclidean distances of ``features`` [N, D]."""
    n = features.shape[0]
    dist = torch.cdist(features, features, p=2)  # [N, N]
    iu = torch.triu_indices(n, n, offset=1, device=features.device)
    return dist[iu[0], iu[1]]


def feature_geometry_alignment(
    features_a: torch.Tensor,
    features_b: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Cosine similarity of the two feature spaces' pairwise-distance vectors.

    The feature dimensions may differ; only the number of observations N must
    match (and be >= 2).
    """
    if features_a.dim() != 2 or features_b.dim() != 2:
        raise CCRTShapeError("features must be rank 2 [N, D]")
    _check_finite(features_a, "features_a")
    _check_finite(features_b, "features_b")
    if features_a.shape[0] != features_b.shape[0]:
        raise CCRTShapeError(
            f"observation count mismatch: {features_a.shape[0]} != "
            f"{features_b.shape[0]}"
        )
    if features_a.shape[0] < 2:
        raise CCRTShapeError("need at least 2 observations")

    da = _condensed_pairwise_distances(features_a)
    db = _condensed_pairwise_distances(features_b)
    if float(da.norm()) <= eps and float(db.norm()) <= eps:
        return torch.zeros((), dtype=features_a.dtype, device=features_a.device)
    return F.cosine_similarity(da.unsqueeze(0), db.unsqueeze(0), dim=-1, eps=eps).squeeze(0)
