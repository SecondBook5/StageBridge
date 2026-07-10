"""Tests for transport stability metrics."""

from __future__ import annotations

import pytest
import torch

from stagebridge.ccrt.contracts import CCRTShapeError
from stagebridge.ccrt.transport import (
    coupling_frobenius_distance,
    displacement_cosine_stability,
    feature_geometry_alignment,
)


def test_identical_coupling_distance_zero():
    c = torch.rand(3, 4, dtype=torch.float64)
    assert float(coupling_frobenius_distance(c, c)) == pytest.approx(0.0, abs=1e-10)


def test_perturbed_coupling_positive_distance():
    torch.manual_seed(0)
    c = torch.rand(3, 4, dtype=torch.float64)
    d = c + 0.1 * torch.rand(3, 4, dtype=torch.float64)
    assert float(coupling_frobenius_distance(c, d)) > 0.0


def test_coupling_shape_mismatch_fails():
    with pytest.raises(CCRTShapeError):
        coupling_frobenius_distance(torch.rand(3, 4), torch.rand(3, 5))


def test_identical_displacement_stability_plus_one():
    d = torch.randn(4, 3, dtype=torch.float64)
    assert float(displacement_cosine_stability(d, d)) == pytest.approx(1.0, abs=1e-6)


def test_opposite_displacement_minus_one():
    d = torch.randn(4, 3, dtype=torch.float64)
    assert float(displacement_cosine_stability(d, -d)) == pytest.approx(-1.0, abs=1e-6)


def test_zero_only_displacement_safe():
    d = torch.zeros(3, 2, dtype=torch.float64)
    assert float(displacement_cosine_stability(d, d)) == 0.0


def test_geometry_preserved_under_rotation_high_alignment():
    torch.manual_seed(0)
    x = torch.randn(6, 2, dtype=torch.float64)
    theta = 0.7
    rot = torch.tensor(
        [[torch.cos(torch.tensor(theta)), -torch.sin(torch.tensor(theta))],
         [torch.sin(torch.tensor(theta)), torch.cos(torch.tensor(theta))]],
        dtype=torch.float64,
    )
    x_rot = x @ rot.T  # rotation preserves pairwise distances
    align = feature_geometry_alignment(x, x_rot)
    assert float(align) == pytest.approx(1.0, abs=1e-6)


def test_distorted_geometry_lower_alignment():
    torch.manual_seed(1)
    x = torch.randn(8, 3, dtype=torch.float64)
    distorted = x.clone()
    distorted[:, 0] *= 10.0  # anisotropic scaling distorts pairwise distances
    align_same = feature_geometry_alignment(x, x)
    align_dist = feature_geometry_alignment(x, distorted)
    assert float(align_dist) < float(align_same)


def test_differing_feature_dimensions_supported():
    torch.manual_seed(2)
    a = torch.randn(5, 2, dtype=torch.float64)
    b = torch.randn(5, 7, dtype=torch.float64)  # different D, same N
    align = feature_geometry_alignment(a, b)
    assert bool(torch.isfinite(align))


def test_observation_mismatch_fails():
    with pytest.raises(CCRTShapeError):
        feature_geometry_alignment(torch.randn(5, 2), torch.randn(6, 2))
