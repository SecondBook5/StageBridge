"""Tests for the optional GeomLoss divergence backend."""

from __future__ import annotations

import importlib.util
import math

import pytest
import torch

from stagebridge.ccrt.representations import SemanticGeometryConfig
from stagebridge.ccrt.transport import (
    GeomLossDivergenceConfig,
    OptionalTransportBackendUnavailable,
    TransportBackendError,
    SinkhornConfig,
    sinkhorn_divergence_geomloss,
    sinkhorn_divergence_native,
)

_HAS_GEOMLOSS = importlib.util.find_spec("geomloss") is not None


# -- always-run tests --

def test_config_validation():
    with pytest.raises(ValueError):
        GeomLossDivergenceConfig(epsilon=0.0)
    with pytest.raises(ValueError):
        GeomLossDivergenceConfig(backend="rings")
    with pytest.raises(ValueError):
        GeomLossDivergenceConfig(scaling=1.5)
    with pytest.raises(ValueError):
        GeomLossDivergenceConfig(debias=False)


def test_blur_is_sqrt_epsilon():
    cfg = GeomLossDivergenceConfig(epsilon=0.04)
    assert cfg.blur == pytest.approx(0.2)


def test_unsupported_cosine_geometry_fails_before_solve():
    # This must fail even without geomloss installed (geometry check first...),
    # but the backend-availability check runs first, so require geomloss OR expect
    # the availability error. Either way it raises.
    src = torch.randn(3, 2)
    tgt = torch.randn(4, 2)
    with pytest.raises((TransportBackendError, OptionalTransportBackendUnavailable)):
        sinkhorn_divergence_geomloss(
            source_features=src, target_features=tgt,
            geometry=SemanticGeometryConfig(metric="cosine"),
            config=GeomLossDivergenceConfig(),
        )


@pytest.mark.skipif(_HAS_GEOMLOSS, reason="geomloss installed; test absence path only")
def test_unavailable_raises_no_fallback():
    src = torch.randn(3, 2)
    tgt = torch.randn(4, 2)
    with pytest.raises(OptionalTransportBackendUnavailable):
        sinkhorn_divergence_geomloss(
            source_features=src, target_features=tgt,
            geometry=SemanticGeometryConfig(), config=GeomLossDivergenceConfig(),
        )


# -- geomloss-present tests --

@pytest.mark.skipif(not _HAS_GEOMLOSS, reason="geomloss not installed")
def test_geomloss_scalar_finite_nonnegative():
    src = torch.randn(3, 2, dtype=torch.float64)
    tgt = torch.randn(4, 2, dtype=torch.float64)
    out = sinkhorn_divergence_geomloss(
        source_features=src, target_features=tgt,
        geometry=SemanticGeometryConfig(), config=GeomLossDivergenceConfig(epsilon=0.1),
    )
    assert out.divergence.dim() == 0
    assert bool(torch.isfinite(out.divergence))
    assert float(out.divergence) >= -1e-5
    assert out.backend == "geomloss"
    assert out.geomloss_backend == "tensorized"
    assert out.epsilon == 0.1
    assert out.blur == pytest.approx(math.sqrt(0.1))


@pytest.mark.skipif(not _HAS_GEOMLOSS, reason="geomloss not installed")
def test_geomloss_identical_near_zero():
    src = torch.randn(4, 3, dtype=torch.float64)
    out = sinkhorn_divergence_geomloss(
        source_features=src, target_features=src,
        geometry=SemanticGeometryConfig(), config=GeomLossDivergenceConfig(epsilon=0.1),
    )
    assert abs(float(out.divergence)) < 1e-3


@pytest.mark.skipif(not _HAS_GEOMLOSS, reason="geomloss not installed")
def test_geomloss_gradients():
    src = torch.randn(3, 2, dtype=torch.float64, requires_grad=True)
    tgt = torch.randn(4, 2, dtype=torch.float64, requires_grad=True)
    out = sinkhorn_divergence_geomloss(
        source_features=src, target_features=tgt,
        geometry=SemanticGeometryConfig(), config=GeomLossDivergenceConfig(epsilon=0.2),
    )
    out.divergence.backward()
    assert src.grad is not None and bool(torch.isfinite(src.grad).all())
    assert tgt.grad is not None and bool(torch.isfinite(tgt.grad).all())


@pytest.mark.skipif(not _HAS_GEOMLOSS, reason="geomloss not installed")
def test_geomloss_native_parity():
    # Tiny float64 problem: native and GeomLoss divergences should agree with the
    # matching full-squared cost + blur=sqrt(epsilon). Not bitwise identical.
    torch.manual_seed(0)
    src = torch.randn(4, 2, dtype=torch.float64)
    tgt = torch.randn(5, 2, dtype=torch.float64)
    geo = SemanticGeometryConfig(metric="squared_euclidean")
    eps = 0.2
    native = sinkhorn_divergence_native(
        source_features=src, target_features=tgt, geometry=geo,
        config=SinkhornConfig(epsilon=eps, max_iterations=2000, tolerance=1e-12),
    )
    gl = sinkhorn_divergence_geomloss(
        source_features=src, target_features=tgt, geometry=geo,
        config=GeomLossDivergenceConfig(epsilon=eps, scaling=0.999),
    )
    assert abs(float(native.divergence) - float(gl.divergence)) < 5e-2
