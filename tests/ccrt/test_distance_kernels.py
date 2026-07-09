"""Tests for continuous distance transforms (no binning)."""

from __future__ import annotations

import pytest
import torch

from stagebridge.ccrt.sender_context import (
    ContinuousDistanceTransform,
    DistanceTransformConfig,
    validate_distance_tensor,
)


@pytest.mark.parametrize("transform", ["identity", "log1p", "sqrt"])
def test_transform_preserves_shape(transform):
    d = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
    out = ContinuousDistanceTransform(DistanceTransformConfig(transform=transform))(d)
    assert out.shape == d.shape


@pytest.mark.parametrize("transform", ["identity", "log1p", "sqrt"])
def test_transform_monotonic_increasing(transform):
    d = torch.tensor([[0.0, 1.0, 2.0, 10.0]])
    out = ContinuousDistanceTransform(DistanceTransformConfig(transform=transform))(d)
    diffs = out[0, 1:] - out[0, :-1]
    assert bool((diffs > 0).all()), f"{transform} not strictly increasing: {out}"


def test_identity_values():
    d = torch.tensor([[0.0, 2.0, 4.0]])
    out = ContinuousDistanceTransform(DistanceTransformConfig("identity"))(d)
    assert torch.allclose(out, d)


def test_log1p_values():
    d = torch.tensor([[0.0, 1.0]])
    out = ContinuousDistanceTransform(DistanceTransformConfig("log1p"))(d)
    assert torch.allclose(out, torch.log1p(d), atol=1e-6)


def test_negative_distance_fails():
    d = torch.tensor([[0.0, -1.0, 2.0]])
    with pytest.raises(ValueError):
        ContinuousDistanceTransform(DistanceTransformConfig("log1p"))(d)
    with pytest.raises(ValueError):
        validate_distance_tensor(d)


def test_rank_other_than_2_fails():
    d1 = torch.tensor([0.0, 1.0, 2.0])  # rank 1
    with pytest.raises(ValueError):
        validate_distance_tensor(d1)
    d3 = torch.zeros(2, 3, 4)  # rank 3
    with pytest.raises(ValueError):
        validate_distance_tensor(d3)


def test_non_floating_fails():
    d = torch.zeros(2, 3, dtype=torch.long)
    with pytest.raises(ValueError):
        validate_distance_tensor(d)


def test_unsupported_transform_fails():
    with pytest.raises(ValueError):
        DistanceTransformConfig(transform="rings")  # not a supported transform


def test_output_is_continuous_not_categorical():
    # A tiny distance perturbation should change the output continuously,
    # confirming there is no bucketing/discretization.
    base = torch.tensor([[1.0, 1.0]])
    perturbed = torch.tensor([[1.0, 1.0001]])
    tf = ContinuousDistanceTransform(DistanceTransformConfig("log1p"))
    out_base = tf(base)
    out_pert = tf(perturbed)
    delta = float((out_pert - out_base)[0, 1].abs())
    assert 0.0 < delta < 1e-3
