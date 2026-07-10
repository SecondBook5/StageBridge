"""Tests for semantic geometry (validation, preparation, pairwise cost)."""

from __future__ import annotations

import pytest
import torch

from stagebridge.ccrt.contracts import CCRTShapeError, CCRTValidationError
from stagebridge.ccrt.representations import (
    SemanticGeometryConfig,
    pairwise_semantic_cost,
    prepare_semantic_features,
    validate_semantic_features,
)


def test_cost_shape():
    geo = SemanticGeometryConfig()
    cost = pairwise_semantic_cost(torch.randn(3, 2), torch.randn(4, 2), geo)
    assert cost.shape == (3, 4)


def test_identical_points_zero_diagonal():
    geo = SemanticGeometryConfig(metric="squared_euclidean")
    x = torch.randn(4, 3, dtype=torch.float64)
    cost = pairwise_semantic_cost(x, x, geo)
    assert torch.allclose(torch.diagonal(cost), torch.zeros(4, dtype=torch.float64), atol=1e-10)


def test_known_squared_euclidean_values():
    geo = SemanticGeometryConfig(metric="squared_euclidean")
    x = torch.tensor([[0.0, 0.0]])
    y = torch.tensor([[3.0, 4.0], [0.0, 0.0]])
    cost = pairwise_semantic_cost(x, y, geo)
    # full squared distance: 3^2+4^2 = 25 (no 1/2 factor)
    assert torch.allclose(cost, torch.tensor([[25.0, 0.0]]), atol=1e-6)


def test_known_cosine_values():
    geo = SemanticGeometryConfig(metric="cosine")
    x = torch.tensor([[1.0, 0.0]])
    y = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    cost = pairwise_semantic_cost(x, y, geo)
    # 1 - cos: identical=0, orthogonal=1, opposite=2
    assert torch.allclose(cost, torch.tensor([[0.0, 1.0, 2.0]]), atol=1e-6)


def test_l2_normalization_applied():
    geo = SemanticGeometryConfig(normalization="l2")
    x = torch.tensor([[3.0, 4.0]])
    prepared = prepare_semantic_features(x, geo)
    assert torch.allclose(prepared.norm(dim=-1), torch.ones(1), atol=1e-6)


def test_normalization_none_preserves():
    geo = SemanticGeometryConfig(normalization="none")
    x = torch.tensor([[3.0, 4.0]])
    assert torch.allclose(prepare_semantic_features(x, geo), x)


def test_dimension_mismatch_fails():
    geo = SemanticGeometryConfig()
    with pytest.raises(CCRTShapeError):
        pairwise_semantic_cost(torch.randn(3, 2), torch.randn(4, 3), geo)


def test_nonfinite_input_fails():
    geo = SemanticGeometryConfig()
    x = torch.randn(3, 2)
    x[0, 0] = float("inf")
    with pytest.raises(CCRTValidationError):
        pairwise_semantic_cost(x, torch.randn(4, 2), geo)


def test_validate_rejects_rank1():
    with pytest.raises(CCRTShapeError):
        validate_semantic_features(torch.randn(3))


def test_cost_nonnegative_and_finite():
    geo = SemanticGeometryConfig()
    cost = pairwise_semantic_cost(torch.randn(5, 4), torch.randn(6, 4), geo)
    assert bool((cost >= 0).all())
    assert bool(torch.isfinite(cost).all())


def test_source_gradient():
    geo = SemanticGeometryConfig()
    x = torch.randn(3, 2, requires_grad=True)
    y = torch.randn(4, 2)
    pairwise_semantic_cost(x, y, geo).sum().backward()
    assert x.grad is not None


def test_target_gradient():
    geo = SemanticGeometryConfig()
    x = torch.randn(3, 2)
    y = torch.randn(4, 2, requires_grad=True)
    pairwise_semantic_cost(x, y, geo).sum().backward()
    assert y.grad is not None


def test_config_validation():
    with pytest.raises(CCRTValidationError):
        SemanticGeometryConfig(metric="l1")
    with pytest.raises(CCRTValidationError):
        SemanticGeometryConfig(normalization="zscore")
    with pytest.raises(CCRTValidationError):
        SemanticGeometryConfig(eps=0.0)
