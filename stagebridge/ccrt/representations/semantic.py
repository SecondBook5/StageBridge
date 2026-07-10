"""Semantic geometry: validation, normalization, and pairwise cost.

Defines the continuous semantic geometry CCRT transport runs in. The canonical
cost is the FULL squared Euclidean distance ``||x_i - y_j||^2`` (no implicit 1/2
factor); a cosine option is also supported. All operations are vectorized,
gradient-preserving PyTorch — no NumPy, no detach, no loops over N/M.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..contracts.errors import CCRTShapeError, CCRTValidationError
from .registry import ALLOWED_NORMALIZATIONS, ALLOWED_SEMANTIC_METRICS

__all__ = [
    "SemanticGeometryConfig",
    "validate_semantic_features",
    "prepare_semantic_features",
    "pairwise_semantic_cost",
]


@dataclass(frozen=True)
class SemanticGeometryConfig:
    """Configuration for semantic geometry."""

    metric: str = "squared_euclidean"
    normalization: str = "none"
    eps: float = 1e-8

    def __post_init__(self) -> None:
        if self.metric not in ALLOWED_SEMANTIC_METRICS:
            raise CCRTValidationError(
                f"metric '{self.metric}' invalid; allowed: "
                f"{sorted(ALLOWED_SEMANTIC_METRICS)}"
            )
        if self.normalization not in ALLOWED_NORMALIZATIONS:
            raise CCRTValidationError(
                f"normalization '{self.normalization}' invalid; allowed: "
                f"{sorted(ALLOWED_NORMALIZATIONS)}"
            )
        if self.eps <= 0.0:
            raise CCRTValidationError("eps must be > 0")


def validate_semantic_features(
    features: torch.Tensor,
    *,
    expected_dim: int | None = None,
    name: str = "semantic_features",
) -> None:
    """Validate a semantic feature tensor: floating, rank-2 [N, D], N>0, finite."""
    if not isinstance(features, torch.Tensor):
        raise CCRTValidationError(f"{name} must be a torch.Tensor")
    if not torch.is_floating_point(features):
        raise CCRTValidationError(f"{name} must be a floating-point tensor")
    if features.dim() != 2:
        raise CCRTShapeError(
            f"{name} must be rank 2 [N, D], got shape {tuple(features.shape)}"
        )
    n, d = features.shape
    if n <= 0:
        raise CCRTShapeError(f"{name}: N must be > 0")
    if d <= 0:
        raise CCRTShapeError(f"{name}: D must be > 0")
    if expected_dim is not None and d != expected_dim:
        raise CCRTShapeError(
            f"{name}: dimension {d} != expected {expected_dim}"
        )
    if not bool(torch.isfinite(features).all()):
        raise CCRTValidationError(f"{name} contains non-finite values")


def prepare_semantic_features(
    features: torch.Tensor, config: SemanticGeometryConfig
) -> torch.Tensor:
    """Apply configured normalization (identity or L2). Never detaches."""
    validate_semantic_features(features)
    if config.normalization == "none":
        return features
    if config.normalization == "l2":
        return F.normalize(features, p=2, dim=-1, eps=config.eps)
    raise CCRTValidationError(  # pragma: no cover - guarded by config
        f"unsupported normalization '{config.normalization}'"
    )


def pairwise_semantic_cost(
    source: torch.Tensor,
    target: torch.Tensor,
    config: SemanticGeometryConfig,
) -> torch.Tensor:
    """Pairwise cost ``[N, M]`` between source ``[N, D]`` and target ``[M, D]``."""
    validate_semantic_features(source, name="source")
    validate_semantic_features(target, name="target")
    if source.shape[1] != target.shape[1]:
        raise CCRTShapeError(
            f"source dim {source.shape[1]} != target dim {target.shape[1]}"
        )

    if config.metric == "squared_euclidean":
        # torch.cdist(...)**2 keeps it vectorized; guard tiny negatives.
        dist = torch.cdist(source, target, p=2)
        cost = dist * dist
        cost = cost.clamp_min(0.0)
    elif config.metric == "cosine":
        s = F.normalize(source, p=2, dim=-1, eps=config.eps)
        t = F.normalize(target, p=2, dim=-1, eps=config.eps)
        sim = s @ t.transpose(0, 1)
        cost = (1.0 - sim).clamp(min=0.0, max=2.0)
    else:  # pragma: no cover - guarded by config
        raise CCRTValidationError(f"unsupported metric '{config.metric}'")

    return cost
