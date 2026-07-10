"""Tests for the semantic transport loss."""

from __future__ import annotations

import importlib.util

import pytest
import torch

from stagebridge.ccrt.contracts import CCRTShapeError, CCRTValidationError
from stagebridge.ccrt.representations import SemanticGeometryConfig
from stagebridge.ccrt.transport import (
    GeomLossDivergenceConfig,
    OptionalTransportBackendUnavailable,
    SemanticTransportLoss,
    SemanticTransportLossConfig,
    SinkhornConfig,
)

_HAS_GEOMLOSS = importlib.util.find_spec("geomloss") is not None


def make_loss(**loss_overrides):
    loss_cfg = SemanticTransportLossConfig(**loss_overrides)
    return SemanticTransportLoss(
        geometry=SemanticGeometryConfig(metric="squared_euclidean"),
        native_sinkhorn=SinkhornConfig(epsilon=0.1, max_iterations=300),
        loss=loss_cfg,
    )


def make_inputs(n=3, m=4, d=2, seed=0):
    torch.manual_seed(seed)
    return (
        torch.randn(n, d, dtype=torch.float64),
        torch.randn(m, d, dtype=torch.float64),
        torch.randn(n, d, dtype=torch.float64),
    )


def test_output_shapes_and_scalars():
    mod = make_loss(direction_weight=1.0)
    src, tgt, drift = make_inputs()
    out = mod(
        source_semantic_features=src, target_semantic_features=tgt, predicted_drift=drift
    )
    assert out.total_loss.dim() == 0
    assert out.displacement_loss.dim() == 0
    assert out.direction_loss.dim() == 0
    assert out.distribution_loss.dim() == 0
    assert out.predicted_destination.shape == (3, 2)
    assert out.barycentric_target.shape == (3, 2)
    assert out.coupling.shape == (3, 4)
    assert out.coupling_backend == "native"
    assert out.distribution_backend == "native"


def test_total_arithmetic_exact():
    mod = make_loss(displacement_weight=1.5, direction_weight=0.7, distribution_weight=0.3)
    src, tgt, drift = make_inputs()
    out = mod(
        source_semantic_features=src, target_semantic_features=tgt, predicted_drift=drift
    )
    expected = (
        1.5 * out.displacement_loss + 0.7 * out.direction_loss + 0.3 * out.distribution_loss
    )
    assert torch.allclose(out.total_loss, expected, atol=1e-10)


def test_predicted_destination_arithmetic():
    mod = make_loss(delta_tau=2.0)
    src, tgt, drift = make_inputs()
    out = mod(
        source_semantic_features=src, target_semantic_features=tgt, predicted_drift=drift
    )
    # geometry normalization is "none", so prepared source == source
    assert torch.allclose(out.predicted_destination, src + 2.0 * drift, atol=1e-10)


def test_target_displacement_arithmetic():
    mod = make_loss()
    src, tgt, drift = make_inputs()
    out = mod(
        source_semantic_features=src, target_semantic_features=tgt, predicted_drift=drift
    )
    assert torch.allclose(out.target_displacement, out.barycentric_target - src, atol=1e-10)


def test_perfect_drift_lower_displacement_loss():
    mod = make_loss(delta_tau=1.0, displacement_weight=1.0, distribution_weight=0.0)
    src, tgt, drift = make_inputs()
    # first compute barycentric target with zero drift to get the ideal drift
    zero = torch.zeros_like(src)
    base = mod(source_semantic_features=src, target_semantic_features=tgt, predicted_drift=zero)
    ideal_drift = base.barycentric_target - src  # delta_tau=1
    good = mod(source_semantic_features=src, target_semantic_features=tgt, predicted_drift=ideal_drift)
    bad = mod(source_semantic_features=src, target_semantic_features=tgt, predicted_drift=-ideal_drift)
    assert float(good.displacement_loss) < float(bad.displacement_loss)


def test_aligned_drift_lower_direction_loss():
    mod = make_loss(direction_weight=1.0, displacement_weight=0.0, distribution_weight=0.0)
    src, tgt, _ = make_inputs()
    zero = torch.zeros_like(src)
    base = mod(source_semantic_features=src, target_semantic_features=tgt, predicted_drift=zero)
    td = base.barycentric_target - src
    aligned = mod(source_semantic_features=src, target_semantic_features=tgt, predicted_drift=td)
    opposite = mod(source_semantic_features=src, target_semantic_features=tgt, predicted_drift=-td)
    assert float(aligned.direction_loss) < float(opposite.direction_loss)


def test_zero_target_displacement_finite():
    # source == target distribution => barycentric target ~ source => td ~ 0
    mod = make_loss(direction_weight=1.0)
    torch.manual_seed(0)
    src = torch.randn(3, 2, dtype=torch.float64)
    out = mod(
        source_semantic_features=src, target_semantic_features=src.clone(),
        predicted_drift=torch.randn(3, 2, dtype=torch.float64),
    )
    assert bool(torch.isfinite(out.direction_loss))
    assert bool(torch.isfinite(out.total_loss))


def test_native_distribution_loss_finite():
    mod = make_loss()
    src, tgt, drift = make_inputs()
    out = mod(source_semantic_features=src, target_semantic_features=tgt, predicted_drift=drift)
    assert bool(torch.isfinite(out.distribution_loss))


def test_source_and_target_weights():
    mod = make_loss()
    src, tgt, drift = make_inputs()
    aw = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
    bw = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    out = mod(
        source_semantic_features=src, target_semantic_features=tgt, predicted_drift=drift,
        source_weights=aw, target_weights=bw,
    )
    assert bool(torch.isfinite(out.total_loss))


def test_dimension_mismatch_fails():
    mod = make_loss()
    with pytest.raises(CCRTShapeError):
        mod(
            source_semantic_features=torch.randn(3, 2, dtype=torch.float64),
            target_semantic_features=torch.randn(4, 3, dtype=torch.float64),
            predicted_drift=torch.randn(3, 2, dtype=torch.float64),
        )


def test_nonfinite_input_fails():
    mod = make_loss()
    src, tgt, drift = make_inputs()
    drift[0, 0] = float("nan")
    with pytest.raises(CCRTValidationError):
        mod(source_semantic_features=src, target_semantic_features=tgt, predicted_drift=drift)


def test_gradients_flow():
    mod = make_loss(direction_weight=1.0)
    torch.manual_seed(0)
    src = torch.randn(3, 2, dtype=torch.float64, requires_grad=True)
    tgt = torch.randn(4, 2, dtype=torch.float64, requires_grad=True)
    drift = torch.randn(3, 2, dtype=torch.float64, requires_grad=True)
    out = mod(source_semantic_features=src, target_semantic_features=tgt, predicted_drift=drift)
    out.total_loss.backward()
    assert drift.grad is not None and bool(torch.isfinite(drift.grad).all())
    assert src.grad is not None and bool(torch.isfinite(src.grad).all())
    assert tgt.grad is not None and bool(torch.isfinite(tgt.grad).all())


def test_config_validation():
    with pytest.raises(ValueError):
        SemanticTransportLossConfig(delta_tau=0.0)
    with pytest.raises(ValueError):
        SemanticTransportLossConfig(
            displacement_weight=0.0, direction_weight=0.0, distribution_weight=0.0
        )
    with pytest.raises(ValueError):
        SemanticTransportLossConfig(distribution_backend="pot")


def test_no_fallback_if_geomloss_requested_unavailable():
    if _HAS_GEOMLOSS:
        pytest.skip("geomloss installed; absence path only")
    mod = SemanticTransportLoss(
        geometry=SemanticGeometryConfig(),
        native_sinkhorn=SinkhornConfig(epsilon=0.1, max_iterations=100),
        loss=SemanticTransportLossConfig(distribution_backend="geomloss"),
        geomloss=GeomLossDivergenceConfig(),
    )
    src, tgt, drift = make_inputs()
    with pytest.raises(OptionalTransportBackendUnavailable):
        mod(source_semantic_features=src, target_semantic_features=tgt, predicted_drift=drift)


@pytest.mark.skipif(not _HAS_GEOMLOSS, reason="geomloss not installed")
def test_optional_geomloss_distribution_loss():
    mod = SemanticTransportLoss(
        geometry=SemanticGeometryConfig(),
        native_sinkhorn=SinkhornConfig(epsilon=0.1, max_iterations=200),
        loss=SemanticTransportLossConfig(distribution_backend="geomloss"),
        geomloss=GeomLossDivergenceConfig(epsilon=0.1),
    )
    src, tgt, drift = make_inputs()
    out = mod(source_semantic_features=src, target_semantic_features=tgt, predicted_drift=drift)
    assert out.distribution_backend == "geomloss"
    assert bool(torch.isfinite(out.total_loss))
