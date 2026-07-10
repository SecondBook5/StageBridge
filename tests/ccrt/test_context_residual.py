"""Tests for the shared context-residual arithmetic."""

from __future__ import annotations

import pytest
import torch

from stagebridge.ccrt.operators import compose_context_residual


def test_exact_arithmetic():
    self_c = torch.tensor([[1.0, 2.0]])
    reg = torch.tensor([[0.5, -0.5]])
    resid = torch.tensor([[0.1, 0.2]])
    out = compose_context_residual(
        self_component=self_c,
        regulatory_component=reg,
        neural_residual=resid,
    )
    assert torch.allclose(out.context_delta, reg + resid)
    assert torch.allclose(out.full_component, self_c + reg + resid)
    # returned components are the inputs unchanged
    assert torch.allclose(out.self_component, self_c)
    assert torch.allclose(out.regulatory_component, reg)
    assert torch.allclose(out.neural_residual, resid)


def test_mismatched_shapes_fail():
    with pytest.raises(ValueError):
        compose_context_residual(
            self_component=torch.zeros(2, 3),
            regulatory_component=torch.zeros(2, 4),
            neural_residual=torch.zeros(2, 3),
        )
    with pytest.raises(ValueError):
        compose_context_residual(
            self_component=torch.zeros(2, 3),
            regulatory_component=torch.zeros(2, 3),
            neural_residual=torch.zeros(3, 3),
        )


def test_gradients_flow_through_all_inputs():
    self_c = torch.randn(2, 3, requires_grad=True)
    reg = torch.randn(2, 3, requires_grad=True)
    resid = torch.randn(2, 3, requires_grad=True)
    out = compose_context_residual(
        self_component=self_c,
        regulatory_component=reg,
        neural_residual=resid,
    )
    out.full_component.sum().backward()
    assert self_c.grad is not None
    assert reg.grad is not None
    assert resid.grad is not None


def test_no_positivity_constraint():
    # negative components are preserved, not clipped
    out = compose_context_residual(
        self_component=torch.tensor([[-5.0]]),
        regulatory_component=torch.tensor([[-1.0]]),
        neural_residual=torch.tensor([[-2.0]]),
    )
    assert float(out.full_component) == pytest.approx(-8.0)
