"""Tests for the optional POT coupling backend."""

from __future__ import annotations

import importlib.util

import pytest
import torch

from stagebridge.ccrt.transport import (
    OptionalTransportBackendUnavailable,
    POTSinkhornConfig,
    SinkhornConfig,
    sinkhorn_coupling_native,
    sinkhorn_coupling_pot,
)

_HAS_POT = importlib.util.find_spec("ot") is not None


def _cost(n=3, m=4, d=2, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(n, d, dtype=torch.float64)
    y = torch.randn(m, d, dtype=torch.float64)
    return torch.cdist(x, y, p=2) ** 2


# -- always-run tests --

def test_config_validation():
    with pytest.raises(ValueError):
        POTSinkhornConfig(epsilon=0.0)
    with pytest.raises(ValueError):
        POTSinkhornConfig(max_iterations=0)
    with pytest.raises(ValueError):
        POTSinkhornConfig(tolerance=0.0)
    with pytest.raises(ValueError):
        POTSinkhornConfig(method="")


@pytest.mark.skipif(_HAS_POT, reason="POT installed; test absence path only")
def test_unavailable_raises_no_fallback():
    with pytest.raises(OptionalTransportBackendUnavailable):
        sinkhorn_coupling_pot(cost_matrix=_cost(), config=POTSinkhornConfig())


# -- POT-present tests --

@pytest.mark.skipif(not _HAS_POT, reason="POT not installed")
def test_pot_coupling_valid():
    out = sinkhorn_coupling_pot(
        cost_matrix=_cost(), config=POTSinkhornConfig(epsilon=0.1, max_iterations=2000)
    )
    assert out.coupling.shape == (3, 4)
    assert bool(torch.isfinite(out.coupling).all())
    assert bool((out.coupling >= 0).all())
    assert out.backend == "pot"
    assert bool(torch.isfinite(out.transport_cost))


@pytest.mark.skipif(not _HAS_POT, reason="POT not installed")
def test_pot_marginals_uniform():
    out = sinkhorn_coupling_pot(
        cost_matrix=_cost(), config=POTSinkhornConfig(epsilon=0.1, max_iterations=5000)
    )
    assert torch.allclose(out.source_marginal, torch.full((3,), 1 / 3, dtype=torch.float64), atol=1e-4)
    assert torch.allclose(out.target_marginal, torch.full((4,), 1 / 4, dtype=torch.float64), atol=1e-4)


@pytest.mark.skipif(not _HAS_POT, reason="POT not installed")
def test_pot_native_parity():
    cost = _cost()
    eps = 0.2
    native = sinkhorn_coupling_native(
        cost_matrix=cost, config=SinkhornConfig(epsilon=eps, max_iterations=5000, tolerance=1e-12)
    )
    pot = sinkhorn_coupling_pot(
        cost_matrix=cost, config=POTSinkhornConfig(epsilon=eps, max_iterations=5000, tolerance=1e-12)
    )
    assert torch.allclose(native.coupling, pot.coupling, atol=1e-4)
    assert abs(float(native.transport_cost) - float(pot.transport_cost)) < 1e-4
    assert float(pot.marginal_error) < 1e-3
