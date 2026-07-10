"""Tests for transport backend policy, metadata, and dispatch."""

from __future__ import annotations

import importlib.util

import pytest
import torch

from stagebridge.ccrt.representations import SemanticGeometryConfig
from stagebridge.ccrt.transport import (
    DIVERGENCE_BACKENDS,
    EXPLICIT_COUPLING_BACKENDS,
    GEOMLOSS_BACKEND,
    NATIVE_BACKEND,
    POT_BACKEND,
    OptionalTransportBackendUnavailable,
    TransportBackendError,
    compute_explicit_coupling,
    compute_sinkhorn_divergence,
    get_backend_metadata,
    optional_backend_available,
    require_optional_backend,
)

_HAS_GEOMLOSS = importlib.util.find_spec("geomloss") is not None
_HAS_POT = importlib.util.find_spec("ot") is not None


def test_backend_id_constants():
    assert NATIVE_BACKEND == "native"
    assert GEOMLOSS_BACKEND == "geomloss"
    assert POT_BACKEND == "pot"
    assert EXPLICIT_COUPLING_BACKENDS == frozenset({"native", "pot"})
    assert DIVERGENCE_BACKENDS == frozenset({"native", "geomloss"})


def test_native_always_available():
    assert optional_backend_available("native") is True
    require_optional_backend("native")  # never raises


def test_unknown_backend_fails():
    with pytest.raises(TransportBackendError):
        optional_backend_available("jax")
    with pytest.raises(TransportBackendError):
        require_optional_backend("jax")


def test_unavailable_optional_raises_no_fallback():
    if not _HAS_GEOMLOSS:
        with pytest.raises(OptionalTransportBackendUnavailable):
            require_optional_backend("geomloss")
    if not _HAS_POT:
        with pytest.raises(OptionalTransportBackendUnavailable):
            require_optional_backend("pot")


def test_native_metadata():
    md = get_backend_metadata("native")
    assert md.backend == "native"
    assert md.differentiable is True
    assert md.explicit_coupling is True
    assert md.version == torch.__version__


def test_metadata_records_backend_without_requiring_install():
    # metadata must not raise merely because an optional backend is absent
    md_g = get_backend_metadata("geomloss")
    md_p = get_backend_metadata("pot")
    assert md_g.backend == "geomloss"
    assert md_p.backend == "pot"


def test_metadata_unknown_backend_fails():
    with pytest.raises(TransportBackendError):
        get_backend_metadata("jax")


def _cost():
    torch.manual_seed(0)
    x = torch.randn(3, 2, dtype=torch.float64)
    y = torch.randn(4, 2, dtype=torch.float64)
    return torch.cdist(x, y, p=2) ** 2


def test_explicit_native_dispatch():
    out = compute_explicit_coupling(backend="native", cost_matrix=_cost())
    assert out.backend == "native"
    assert out.coupling.shape == (3, 4)


def test_explicit_coupling_rejects_geomloss():
    with pytest.raises(TransportBackendError):
        compute_explicit_coupling(backend="geomloss", cost_matrix=_cost())


def test_divergence_native_dispatch():
    src = torch.randn(3, 2, dtype=torch.float64)
    tgt = torch.randn(4, 2, dtype=torch.float64)
    out = compute_sinkhorn_divergence(
        backend="native",
        source_features=src,
        target_features=tgt,
        geometry=SemanticGeometryConfig(),
    )
    assert out.backend == "native"


def test_divergence_rejects_pot():
    src = torch.randn(3, 2)
    tgt = torch.randn(4, 2)
    with pytest.raises(TransportBackendError):
        compute_sinkhorn_divergence(
            backend="pot",
            source_features=src,
            target_features=tgt,
            geometry=SemanticGeometryConfig(),
        )


@pytest.mark.skipif(not _HAS_POT, reason="POT not installed")
def test_explicit_pot_dispatch_when_available():
    out = compute_explicit_coupling(backend="pot", cost_matrix=_cost())
    assert out.backend == "pot"


@pytest.mark.skipif(not _HAS_GEOMLOSS, reason="geomloss not installed")
def test_divergence_geomloss_dispatch_when_available():
    src = torch.randn(3, 2, dtype=torch.float64)
    tgt = torch.randn(4, 2, dtype=torch.float64)
    out = compute_sinkhorn_divergence(
        backend="geomloss",
        source_features=src,
        target_features=tgt,
        geometry=SemanticGeometryConfig(),
    )
    assert out.backend == "geomloss"
