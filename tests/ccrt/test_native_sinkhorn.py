"""Tests for the native differentiable Sinkhorn and its divergence."""

from __future__ import annotations

import pytest
import torch

from stagebridge.ccrt.representations import SemanticGeometryConfig
from stagebridge.ccrt.transport import (
    SinkhornConfig,
    build_transport_cost,
    normalize_measure_weights,
    sinkhorn_coupling_native,
    sinkhorn_divergence_native,
)


def _cost(n=3, m=4, d=2, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(n, d, dtype=torch.float64)
    y = torch.randn(m, d, dtype=torch.float64)
    return torch.cdist(x, y, p=2) ** 2


def _solve(cost=None, **cfg):
    if cost is None:
        cost = _cost()
    cfg.setdefault("max_iterations", 500)
    return sinkhorn_coupling_native(cost_matrix=cost, config=SinkhornConfig(**cfg))


# -- coupling structure --

def test_coupling_shape_finite_nonnegative():
    out = _solve()
    assert out.coupling.shape == (3, 4)
    assert bool(torch.isfinite(out.coupling).all())
    assert bool((out.coupling >= 0).all())


def test_uniform_marginals():
    out = _solve(epsilon=0.1)
    assert torch.allclose(out.source_marginal, torch.full((3,), 1 / 3, dtype=torch.float64), atol=1e-6)
    assert torch.allclose(out.target_marginal, torch.full((4,), 1 / 4, dtype=torch.float64), atol=1e-6)
    assert float(out.marginal_error) < 1e-6


def test_custom_marginals():
    cost = _cost()
    a = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
    b = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    out = sinkhorn_coupling_native(
        cost_matrix=cost, config=SinkhornConfig(epsilon=0.1, max_iterations=500),
        source_weights=a, target_weights=b,
    )
    assert torch.allclose(out.source_marginal, a, atol=1e-6)
    assert torch.allclose(out.target_marginal, b, atol=1e-6)


def test_weights_normalized():
    a = torch.tensor([2.0, 2.0, 4.0], dtype=torch.float64)  # sums to 8
    w = normalize_measure_weights(a, size=3, dtype=torch.float64, device=torch.device("cpu"), name="a")
    assert torch.allclose(w.sum(), torch.tensor(1.0, dtype=torch.float64))


def test_zero_weight_fails():
    a = torch.tensor([0.0, 0.5, 0.5], dtype=torch.float64)
    with pytest.raises(ValueError):
        sinkhorn_coupling_native(
            cost_matrix=_cost(), config=SinkhornConfig(), source_weights=a
        )


def test_negative_weight_fails():
    a = torch.tensor([-0.1, 0.6, 0.5], dtype=torch.float64)
    with pytest.raises(ValueError):
        sinkhorn_coupling_native(cost_matrix=_cost(), config=SinkhornConfig(), source_weights=a)


def test_wrong_weight_length_fails():
    a = torch.tensor([0.5, 0.5], dtype=torch.float64)
    with pytest.raises(ValueError):
        sinkhorn_coupling_native(cost_matrix=_cost(), config=SinkhornConfig(), source_weights=a)


def test_negative_cost_fails():
    cost = _cost()
    cost[0, 0] = -1.0
    with pytest.raises(ValueError):
        sinkhorn_coupling_native(cost_matrix=cost, config=SinkhornConfig())


def test_nonfinite_cost_fails():
    cost = _cost()
    cost[0, 0] = float("nan")
    with pytest.raises(ValueError):
        sinkhorn_coupling_native(cost_matrix=cost, config=SinkhornConfig())


def test_one_point_trivial_coupling():
    cost = torch.zeros(1, 1, dtype=torch.float64)
    out = sinkhorn_coupling_native(cost_matrix=cost, config=SinkhornConfig())
    assert torch.allclose(out.coupling, torch.ones(1, 1, dtype=torch.float64), atol=1e-8)


# -- objective --

def test_objective_scalars_and_arithmetic():
    out = _solve(epsilon=0.1)
    assert out.transport_cost.dim() == 0
    assert out.kl_regularization.dim() == 0
    assert out.regularized_objective.dim() == 0
    assert torch.allclose(
        out.regularized_objective, out.transport_cost + 0.1 * out.kl_regularization, atol=1e-10
    )


def test_backend_native():
    assert _solve().backend == "native"


# -- iteration behavior --

def test_fixed_iterations_when_no_early_stopping():
    out = _solve(max_iterations=37, early_stopping=False)
    assert out.iterations == 37


def test_early_stopping_metadata_valid():
    out = sinkhorn_coupling_native(
        cost_matrix=_cost(),
        config=SinkhornConfig(epsilon=0.1, max_iterations=500, early_stopping=True, check_interval=10),
    )
    assert out.iterations <= 500
    assert isinstance(out.converged, bool)


def test_smaller_epsilon_finite():
    out = _solve(epsilon=0.01)
    assert bool(torch.isfinite(out.coupling).all())
    assert not bool(torch.isnan(out.coupling).any())


# -- gradients --

def test_gradients_to_cost_matrix():
    torch.manual_seed(0)
    x = torch.randn(3, 2, dtype=torch.float64, requires_grad=True)
    y = torch.randn(4, 2, dtype=torch.float64)
    geo = SemanticGeometryConfig()
    cost = build_transport_cost(source_features=x, target_features=y, geometry=geo).cost_matrix
    out = sinkhorn_coupling_native(cost_matrix=cost, config=SinkhornConfig(epsilon=0.1, max_iterations=200))
    out.regularized_objective.backward()
    assert x.grad is not None
    assert bool(torch.isfinite(x.grad).all())


def test_config_validation():
    with pytest.raises(ValueError):
        SinkhornConfig(epsilon=0.0)
    with pytest.raises(ValueError):
        SinkhornConfig(max_iterations=0)
    with pytest.raises(ValueError):
        SinkhornConfig(check_interval=1000, max_iterations=10)


# -- divergence --

def test_divergence_identical_near_zero():
    src = torch.randn(4, 3, dtype=torch.float64)
    out = sinkhorn_divergence_native(
        source_features=src, target_features=src,
        geometry=SemanticGeometryConfig(), config=SinkhornConfig(epsilon=0.1, max_iterations=500),
    )
    assert abs(float(out.divergence)) < 1e-6


def test_divergence_separated_positive():
    src = torch.zeros(3, 2, dtype=torch.float64)
    tgt = torch.full((3, 2), 5.0, dtype=torch.float64) + torch.randn(3, 2, dtype=torch.float64) * 0.01
    out = sinkhorn_divergence_native(
        source_features=src, target_features=tgt,
        geometry=SemanticGeometryConfig(), config=SinkhornConfig(epsilon=0.1, max_iterations=500),
    )
    assert float(out.divergence) > 0.0


def test_divergence_approx_symmetry():
    torch.manual_seed(1)
    a = torch.randn(3, 2, dtype=torch.float64)
    b = torch.randn(4, 2, dtype=torch.float64)
    cfg = SinkhornConfig(epsilon=0.2, max_iterations=500)
    geo = SemanticGeometryConfig()
    d_ab = sinkhorn_divergence_native(source_features=a, target_features=b, geometry=geo, config=cfg)
    d_ba = sinkhorn_divergence_native(source_features=b, target_features=a, geometry=geo, config=cfg)
    assert abs(float(d_ab.divergence) - float(d_ba.divergence)) < 1e-4


def test_divergence_scalar_and_shapes():
    a = torch.randn(3, 2, dtype=torch.float64)
    b = torch.randn(4, 2, dtype=torch.float64)
    out = sinkhorn_divergence_native(
        source_features=a, target_features=b, geometry=SemanticGeometryConfig(),
        config=SinkhornConfig(epsilon=0.1, max_iterations=300),
    )
    assert out.divergence.dim() == 0
    assert out.cross.coupling.shape == (3, 4)
    assert out.source_self.coupling.shape == (3, 3)
    assert out.target_self.coupling.shape == (4, 4)
    assert out.backend == "native"


def test_divergence_gradients():
    a = torch.randn(3, 2, dtype=torch.float64, requires_grad=True)
    b = torch.randn(4, 2, dtype=torch.float64, requires_grad=True)
    out = sinkhorn_divergence_native(
        source_features=a, target_features=b, geometry=SemanticGeometryConfig(),
        config=SinkhornConfig(epsilon=0.2, max_iterations=200),
    )
    out.divergence.backward()
    assert a.grad is not None and bool(torch.isfinite(a.grad).all())
    assert b.grad is not None and bool(torch.isfinite(b.grad).all())


def test_divergence_weighted():
    a = torch.randn(3, 2, dtype=torch.float64)
    b = torch.randn(4, 2, dtype=torch.float64)
    aw = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
    bw = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    out = sinkhorn_divergence_native(
        source_features=a, target_features=b, geometry=SemanticGeometryConfig(),
        config=SinkhornConfig(epsilon=0.2, max_iterations=300),
        source_weights=aw, target_weights=bw,
    )
    assert bool(torch.isfinite(out.divergence))
