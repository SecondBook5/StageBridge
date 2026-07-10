"""Tests for build_transport_cost."""

from __future__ import annotations

import torch

from stagebridge.ccrt.representations import (
    SemanticGeometryConfig,
    pairwise_semantic_cost,
    prepare_semantic_features,
)
from stagebridge.ccrt.transport import build_transport_cost


def test_shapes_and_recording():
    geo = SemanticGeometryConfig(metric="squared_euclidean", normalization="l2")
    out = build_transport_cost(
        source_features=torch.randn(3, 2),
        target_features=torch.randn(4, 2),
        geometry=geo,
    )
    assert out.cost_matrix.shape == (3, 4)
    assert out.source_features.shape == (3, 2)
    assert out.target_features.shape == (4, 2)
    assert out.metric == "squared_euclidean"
    assert out.normalization == "l2"


def test_normalization_applied():
    geo = SemanticGeometryConfig(normalization="l2")
    out = build_transport_cost(
        source_features=torch.tensor([[3.0, 4.0]]),
        target_features=torch.tensor([[0.0, 5.0]]),
        geometry=geo,
    )
    assert torch.allclose(out.source_features.norm(dim=-1), torch.ones(1), atol=1e-6)


def test_cost_agrees_with_semantic_cost():
    geo = SemanticGeometryConfig(metric="squared_euclidean", normalization="none")
    src = torch.randn(3, 2)
    tgt = torch.randn(4, 2)
    out = build_transport_cost(source_features=src, target_features=tgt, geometry=geo)
    expected = pairwise_semantic_cost(
        prepare_semantic_features(src, geo), prepare_semantic_features(tgt, geo), geo
    )
    assert torch.allclose(out.cost_matrix, expected)


def test_gradients_preserved():
    geo = SemanticGeometryConfig()
    src = torch.randn(3, 2, requires_grad=True)
    tgt = torch.randn(4, 2, requires_grad=True)
    out = build_transport_cost(source_features=src, target_features=tgt, geometry=geo)
    out.cost_matrix.sum().backward()
    assert src.grad is not None
    assert tgt.grad is not None
