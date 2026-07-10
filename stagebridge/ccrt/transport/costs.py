"""Transport cost construction.

Thin wrapper that prepares source/target semantic features per the geometry
config and builds the canonical pairwise cost matrix. It performs no transport
solve and no backend selection — just the gradient-preserving cost.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..representations.semantic import (
    SemanticGeometryConfig,
    pairwise_semantic_cost,
    prepare_semantic_features,
)

__all__ = ["TransportCostOutput", "build_transport_cost"]


@dataclass(frozen=True)
class TransportCostOutput:
    """The prepared features and their pairwise cost matrix."""

    source_features: torch.Tensor
    target_features: torch.Tensor
    cost_matrix: torch.Tensor
    metric: str
    normalization: str


def build_transport_cost(
    *,
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    geometry: SemanticGeometryConfig,
) -> TransportCostOutput:
    """Prepare features and compute the canonical [N, M] cost matrix."""
    prepared_source = prepare_semantic_features(source_features, geometry)
    prepared_target = prepare_semantic_features(target_features, geometry)
    cost = pairwise_semantic_cost(prepared_source, prepared_target, geometry)
    return TransportCostOutput(
        source_features=prepared_source,
        target_features=prepared_target,
        cost_matrix=cost,
        metric=geometry.metric,
        normalization=geometry.normalization,
    )
