"""Tests for barycentric transport targets."""

from __future__ import annotations

import pytest
import torch

from stagebridge.ccrt.contracts import CCRTShapeError, CCRTValidationError
from stagebridge.ccrt.transport import (
    barycentric_projection,
    build_barycentric_transport_target,
)


def test_hand_calculated_barycentric():
    # source row 0 couples equally to two targets -> mean of the two.
    coupling = torch.tensor([[0.25, 0.25], [0.5, 0.0]], dtype=torch.float64)
    target = torch.tensor([[0.0, 0.0], [4.0, 2.0]], dtype=torch.float64)
    bary, row_mass = barycentric_projection(coupling=coupling, target_features=target)
    assert torch.allclose(row_mass, torch.tensor([0.5, 0.5], dtype=torch.float64))
    # row 0: (0.25*[0,0] + 0.25*[4,2]) / 0.5 = [2,1]
    # row 1: (0.5*[0,0]) / 0.5 = [0,0]
    assert torch.allclose(bary, torch.tensor([[2.0, 1.0], [0.0, 0.0]], dtype=torch.float64))


def test_build_target_shapes_and_arithmetic():
    coupling = torch.tensor([[0.5, 0.0], [0.0, 0.5]], dtype=torch.float64)
    source = torch.tensor([[1.0, 1.0], [2.0, 2.0]], dtype=torch.float64)
    target = torch.tensor([[3.0, 3.0], [4.0, 4.0]], dtype=torch.float64)
    out = build_barycentric_transport_target(
        source_features=source, target_features=target, coupling=coupling,
        coupling_backend="native",
    )
    assert out.barycentric_target.shape == (2, 2)
    assert out.target_displacement.shape == (2, 2)
    assert torch.allclose(out.barycentric_target, torch.tensor([[3.0, 3.0], [4.0, 4.0]], dtype=torch.float64))
    assert torch.allclose(out.target_displacement, out.barycentric_target - source)
    assert torch.allclose(out.transported_source_mass, torch.tensor([0.5, 0.5], dtype=torch.float64))
    assert out.coupling_backend == "native"


def test_dimension_mismatch_fails():
    coupling = torch.tensor([[0.5, 0.5]], dtype=torch.float64)  # M=2
    target = torch.randn(3, 2, dtype=torch.float64)  # M=3 mismatch
    with pytest.raises(CCRTShapeError):
        barycentric_projection(coupling=coupling, target_features=target)


def test_zero_row_mass_fails():
    coupling = torch.tensor([[0.0, 0.0], [0.5, 0.5]], dtype=torch.float64)
    target = torch.randn(2, 2, dtype=torch.float64)
    with pytest.raises(CCRTValidationError):
        barycentric_projection(coupling=coupling, target_features=target)


def test_negative_coupling_fails():
    coupling = torch.tensor([[-0.1, 0.6]], dtype=torch.float64)
    target = torch.randn(2, 2, dtype=torch.float64)
    with pytest.raises(CCRTValidationError):
        barycentric_projection(coupling=coupling, target_features=target)


def test_gradients_flow():
    coupling = torch.tensor([[0.3, 0.2], [0.1, 0.4]], dtype=torch.float64, requires_grad=True)
    source = torch.randn(2, 2, dtype=torch.float64, requires_grad=True)
    target = torch.randn(2, 2, dtype=torch.float64, requires_grad=True)
    out = build_barycentric_transport_target(
        source_features=source, target_features=target, coupling=coupling,
        coupling_backend="native",
    )
    out.target_displacement.sum().backward()
    assert coupling.grad is not None
    assert source.grad is not None
    assert target.grad is not None
