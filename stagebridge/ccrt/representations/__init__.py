"""CCRT representations — registered semantic/feature spaces and geometry.

Owns the feature-space registry (which prevents an arbitrary latent from
silently becoming transport geometry) and the continuous semantic geometry
(validation, normalization, pairwise cost). System-agnostic; imports only
contracts, torch, and the standard library.
"""

from __future__ import annotations

from .registry import (
    ALLOWED_NORMALIZATIONS,
    ALLOWED_REPRESENTATION_ROLES,
    ALLOWED_SEMANTIC_METRICS,
    FeatureSpaceRegistry,
    FeatureSpaceSpec,
)
from .semantic import (
    SemanticGeometryConfig,
    pairwise_semantic_cost,
    prepare_semantic_features,
    validate_semantic_features,
)

__all__ = [
    "ALLOWED_REPRESENTATION_ROLES",
    "ALLOWED_SEMANTIC_METRICS",
    "ALLOWED_NORMALIZATIONS",
    "FeatureSpaceSpec",
    "FeatureSpaceRegistry",
    "SemanticGeometryConfig",
    "validate_semantic_features",
    "prepare_semantic_features",
    "pairwise_semantic_cost",
]
