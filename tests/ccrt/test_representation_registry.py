"""Tests for the feature-space registry."""

from __future__ import annotations

import pytest
import torch

from stagebridge.ccrt.contracts import CCRTShapeError, CCRTValidationError
from stagebridge.ccrt.representations import (
    FeatureSpaceRegistry,
    FeatureSpaceSpec,
)


def test_valid_semantic_spec():
    spec = FeatureSpaceSpec(
        feature_space_id="z_sem", role="semantic", dimension=3, metric="squared_euclidean"
    )
    assert spec.dimension == 3


def test_valid_reconstruction_spec():
    spec = FeatureSpaceSpec(feature_space_id="z_rec", role="reconstruction", dimension=5)
    assert spec.metric is None


def test_valid_regulatory_spec():
    spec = FeatureSpaceSpec(feature_space_id="reg", role="regulatory", dimension=2)
    assert spec.role == "regulatory"


def test_semantic_metric_required():
    with pytest.raises(CCRTValidationError):
        FeatureSpaceSpec(feature_space_id="z", role="semantic", dimension=3)


def test_invalid_role_fails():
    with pytest.raises(CCRTValidationError):
        FeatureSpaceSpec(feature_space_id="z", role="nonsense", dimension=3)


def test_invalid_metric_fails():
    with pytest.raises(CCRTValidationError):
        FeatureSpaceSpec(
            feature_space_id="z", role="semantic", dimension=3, metric="manhattan"
        )


def test_invalid_normalization_fails():
    with pytest.raises(CCRTValidationError):
        FeatureSpaceSpec(
            feature_space_id="z", role="reconstruction", dimension=3,
            normalization="minmax",
        )


def test_dimension_nonpositive_fails():
    with pytest.raises(CCRTValidationError):
        FeatureSpaceSpec(feature_space_id="z", role="reconstruction", dimension=0)


def test_feature_id_length_mismatch_fails():
    with pytest.raises(CCRTValidationError):
        FeatureSpaceSpec(
            feature_space_id="z", role="reconstruction", dimension=3,
            feature_ids=("a", "b"),
        )


def test_duplicate_feature_ids_fail():
    with pytest.raises(CCRTValidationError):
        FeatureSpaceSpec(
            feature_space_id="z", role="reconstruction", dimension=2,
            feature_ids=("a", "a"),
        )


def test_feature_ids_not_lowercased():
    spec = FeatureSpaceSpec(
        feature_space_id="z", role="reconstruction", dimension=2,
        feature_ids=("GeneA", "GeneB"),
    )
    assert spec.feature_ids == ("GeneA", "GeneB")


def test_duplicate_registry_id_fails():
    reg = FeatureSpaceRegistry()
    reg.register(FeatureSpaceSpec(feature_space_id="z", role="reconstruction", dimension=2))
    with pytest.raises(CCRTValidationError):
        reg.register(FeatureSpaceSpec(feature_space_id="z", role="regulatory", dimension=3))


def test_unknown_registry_id_fails():
    reg = FeatureSpaceRegistry()
    with pytest.raises(CCRTValidationError):
        reg.get("missing")


def test_contains_and_insertion_order_preserved():
    reg = FeatureSpaceRegistry()
    reg.register(FeatureSpaceSpec(feature_space_id="b", role="reconstruction", dimension=2))
    reg.register(FeatureSpaceSpec(feature_space_id="a", role="regulatory", dimension=3))
    reg.register(
        FeatureSpaceSpec(
            feature_space_id="c", role="semantic", dimension=4, metric="cosine"
        )
    )
    assert reg.contains("a")
    assert not reg.contains("z")
    assert reg.ids() == ("b", "a", "c")


def _registry_with_semantic(dim=3):
    reg = FeatureSpaceRegistry()
    reg.register(
        FeatureSpaceSpec(
            feature_space_id="z_sem", role="semantic", dimension=dim,
            metric="squared_euclidean",
        )
    )
    return reg


def test_validate_tensor_correct_passes():
    reg = _registry_with_semantic(3)
    reg.validate_tensor("z_sem", torch.randn(5, 3))


def test_validate_tensor_wrong_dimension_fails():
    reg = _registry_with_semantic(3)
    with pytest.raises(CCRTShapeError):
        reg.validate_tensor("z_sem", torch.randn(5, 4))


def test_validate_tensor_wrong_rank_fails():
    reg = _registry_with_semantic(3)
    with pytest.raises(CCRTShapeError):
        reg.validate_tensor("z_sem", torch.randn(3))


def test_validate_tensor_integer_fails():
    reg = _registry_with_semantic(3)
    with pytest.raises(CCRTValidationError):
        reg.validate_tensor("z_sem", torch.zeros(5, 3, dtype=torch.long))


def test_validate_tensor_nonfinite_fails():
    reg = _registry_with_semantic(3)
    t = torch.randn(5, 3)
    t[0, 0] = float("nan")
    with pytest.raises(CCRTValidationError):
        reg.validate_tensor("z_sem", t)


def test_validate_tensor_role_mismatch_fails():
    reg = _registry_with_semantic(3)
    with pytest.raises(CCRTValidationError):
        reg.validate_tensor("z_sem", torch.randn(5, 3), expected_role="reconstruction")
