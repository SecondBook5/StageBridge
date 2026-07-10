"""Tests for synthetic recovery metrics."""

from __future__ import annotations

import pytest
import torch

from stagebridge.ccrt.synthetic import (
    mean_cosine_recovery,
    mean_effect_norm,
    pearson_recovery,
    rank_order_recovery,
    relative_root_mean_squared_error,
    root_mean_squared_error,
    sign_agreement,
)


def test_rmse_exact_prediction_zero():
    x = torch.randn(4, 3, dtype=torch.float64)
    assert float(root_mean_squared_error(x, x)) == pytest.approx(0.0, abs=1e-12)


def test_relative_rmse():
    pred = torch.zeros(4, 3, dtype=torch.float64)
    target = torch.ones(4, 3, dtype=torch.float64)
    # rmse=1, rms_target=1 -> ~1
    assert float(relative_root_mean_squared_error(pred, target)) == pytest.approx(1.0, abs=1e-6)


def test_cosine_opposite_and_orthogonal():
    a = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    assert float(mean_cosine_recovery(a, -a)) == pytest.approx(-1.0, abs=1e-6)
    b = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
    assert abs(float(mean_cosine_recovery(a, b))) < 1e-6


def test_cosine_zero_target_safe():
    a = torch.tensor([[1.0, 1.0]], dtype=torch.float64)
    z = torch.zeros(1, 2, dtype=torch.float64)
    assert float(mean_cosine_recovery(a, z)) == 0.0


def test_pearson_exact():
    x = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    assert float(pearson_recovery(x, 2 * x + 1)) == pytest.approx(1.0, abs=1e-6)
    assert float(pearson_recovery(x, -x)) == pytest.approx(-1.0, abs=1e-6)


def test_pearson_zero_variance_safe():
    x = torch.ones(4, dtype=torch.float64)
    y = torch.randn(4, dtype=torch.float64)
    assert float(pearson_recovery(x, y)) == 0.0


def test_sign_agreement_exact():
    pred = torch.tensor([1.0, -1.0, 1.0], dtype=torch.float64)
    target = torch.tensor([2.0, -3.0, -1.0], dtype=torch.float64)
    # first two agree, third disagrees -> 2/3
    assert float(sign_agreement(pred, target)) == pytest.approx(2.0 / 3.0, abs=1e-6)


def test_sign_agreement_all_zero_target_safe():
    pred = torch.tensor([1.0, -1.0], dtype=torch.float64)
    target = torch.zeros(2, dtype=torch.float64)
    assert float(sign_agreement(pred, target)) == 0.0


def test_mean_effect_norm():
    effect = torch.tensor([[3.0, 4.0], [0.0, 0.0]], dtype=torch.float64)
    # row norms: 5, 0 -> mean 2.5
    assert float(mean_effect_norm(effect)) == pytest.approx(2.5, abs=1e-6)


def test_rank_order_recovery_exact():
    pred = torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64)
    true = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    assert float(rank_order_recovery(pred, true)) == pytest.approx(1.0, abs=1e-6)
    assert float(rank_order_recovery(pred, torch.flip(true, [0]))) == pytest.approx(-1.0, abs=1e-6)


def test_rank_order_ties_deterministic():
    pred = torch.tensor([0.5, 0.5, 0.9], dtype=torch.float64)
    true = torch.tensor([1.0, 1.0, 2.0], dtype=torch.float64)
    # tie-handled ranks should give perfect correlation here
    assert float(rank_order_recovery(pred, true)) == pytest.approx(1.0, abs=1e-6)


def test_mismatched_shapes_fail():
    with pytest.raises(ValueError):
        root_mean_squared_error(torch.zeros(3), torch.zeros(4))


def test_nonfinite_fails():
    bad = torch.tensor([1.0, float("nan")], dtype=torch.float64)
    with pytest.raises(ValueError):
        root_mean_squared_error(bad, torch.zeros(2, dtype=torch.float64))
