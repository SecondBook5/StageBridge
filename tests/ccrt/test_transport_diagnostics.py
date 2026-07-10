"""Tests for transport / geometry diagnostics."""

from __future__ import annotations

import pytest
import torch

from stagebridge.ccrt.transport import (
    coupling_entropy,
    coupling_marginal_error,
    effective_rank,
    mean_drift_alignment,
)


def test_rank_one_effective_rank_near_one():
    # rank-1 cloud: all points on a line -> effective rank ~ 1
    direction = torch.tensor([1.0, 2.0, 0.5], dtype=torch.float64)
    scales = torch.tensor([[-2.0], [-1.0], [1.0], [2.0]], dtype=torch.float64)
    features = scales * direction  # [4, 3]
    er = effective_rank(features)
    assert float(er) == pytest.approx(1.0, abs=1e-3)


def test_isotropic_higher_rank():
    torch.manual_seed(0)
    features = torch.randn(50, 3, dtype=torch.float64)
    er = effective_rank(features)
    assert float(er) > 1.5


def test_all_zero_effective_rank_zero():
    features = torch.zeros(4, 3, dtype=torch.float64)
    assert float(effective_rank(features)) == 0.0


def test_concentrated_lower_entropy_than_diffuse():
    concentrated = torch.tensor([[0.97, 0.01], [0.01, 0.01]], dtype=torch.float64)
    diffuse = torch.full((2, 2), 0.25, dtype=torch.float64)
    assert float(coupling_entropy(concentrated)) < float(coupling_entropy(diffuse))


def test_normalized_entropy_in_unit_interval():
    torch.manual_seed(0)
    coupling = torch.rand(3, 4, dtype=torch.float64)
    ent = coupling_entropy(coupling, normalize=True)
    assert -1e-9 <= float(ent) <= 1.0 + 1e-9


def test_one_entry_coupling_entropy_zero():
    coupling = torch.tensor([[0.5]], dtype=torch.float64)
    assert float(coupling_entropy(coupling, normalize=True)) == 0.0


def test_marginal_error_exact_zero():
    coupling = torch.tensor([[0.25, 0.25], [0.25, 0.25]], dtype=torch.float64)
    a = torch.tensor([0.5, 0.5], dtype=torch.float64)
    b = torch.tensor([0.5, 0.5], dtype=torch.float64)
    err = coupling_marginal_error(coupling=coupling, source_weights=a, target_weights=b)
    assert float(err) == pytest.approx(0.0, abs=1e-12)


def test_aligned_drift_alignment_plus_one():
    drift = torch.tensor([[1.0, 0.0], [0.0, 2.0]], dtype=torch.float64)
    td = torch.tensor([[3.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    assert float(mean_drift_alignment(predicted_drift=drift, target_displacement=td)) == pytest.approx(1.0, abs=1e-6)


def test_opposite_drift_alignment_minus_one():
    drift = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    td = torch.tensor([[-2.0, 0.0]], dtype=torch.float64)
    assert float(mean_drift_alignment(predicted_drift=drift, target_displacement=td)) == pytest.approx(-1.0, abs=1e-6)


def test_orthogonal_drift_alignment_zero():
    drift = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    td = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
    assert abs(float(mean_drift_alignment(predicted_drift=drift, target_displacement=td))) < 1e-6


def test_zero_target_displacement_safe():
    drift = torch.tensor([[1.0, 1.0]], dtype=torch.float64)
    td = torch.zeros(1, 2, dtype=torch.float64)
    assert float(mean_drift_alignment(predicted_drift=drift, target_displacement=td)) == 0.0


def test_weighted_alignment():
    drift = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float64)
    td = torch.tensor([[1.0, 0.0], [-1.0, 0.0]], dtype=torch.float64)
    w = torch.tensor([1.0, 0.0], dtype=torch.float64)  # ignore the anti-aligned row
    align = mean_drift_alignment(predicted_drift=drift, target_displacement=td, weights=w)
    assert float(align) == pytest.approx(1.0, abs=1e-6)
