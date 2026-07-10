"""Tests for EdgeLinear (transition-edge-conditioned linear maps)."""

from __future__ import annotations

import pytest
import torch

from stagebridge.ccrt.operators import EdgeLinear, EdgeLinearConfig


def test_shared_map_output_shape():
    torch.manual_seed(0)
    mod = EdgeLinear(EdgeLinearConfig(input_dim=3, output_dim=6))
    x = torch.randn(4, 3)
    y = mod(x)
    assert y.shape == (4, 6)


def test_edge_specific_map_output_shape():
    torch.manual_seed(0)
    mod = EdgeLinear(
        EdgeLinearConfig(input_dim=3, output_dim=6, num_transition_edges=2)
    )
    x = torch.randn(4, 3)
    edge = torch.tensor([0, 1, 0, 1])
    y = mod(x, edge)
    assert y.shape == (4, 6)


def test_missing_edge_index_fails_when_edge_specific():
    mod = EdgeLinear(
        EdgeLinearConfig(input_dim=3, output_dim=6, num_transition_edges=2)
    )
    with pytest.raises(ValueError):
        mod(torch.randn(4, 3))


def test_out_of_range_edge_index_fails():
    mod = EdgeLinear(
        EdgeLinearConfig(input_dim=3, output_dim=6, num_transition_edges=2)
    )
    with pytest.raises(ValueError):
        mod(torch.randn(2, 3), torch.tensor([0, 5]))


def test_different_edges_produce_different_outputs():
    torch.manual_seed(0)
    mod = EdgeLinear(
        EdgeLinearConfig(input_dim=3, output_dim=4, num_transition_edges=2)
    )
    x = torch.randn(1, 3)
    y0 = mod(x, torch.tensor([0]))
    y1 = mod(x, torch.tensor([1]))
    assert not torch.allclose(y0, y1)


def test_gradients_flow_shared():
    mod = EdgeLinear(EdgeLinearConfig(input_dim=3, output_dim=4))
    x = torch.randn(2, 3, requires_grad=True)
    mod(x).sum().backward()
    assert mod.shared.weight.grad is not None
    assert x.grad is not None


def test_gradients_flow_edge_specific():
    mod = EdgeLinear(
        EdgeLinearConfig(input_dim=3, output_dim=4, num_transition_edges=2)
    )
    x = torch.randn(2, 3)
    edge = torch.tensor([0, 1])
    mod(x, edge).sum().backward()
    assert mod.weight.grad is not None
    assert mod.bias.grad is not None


def test_no_bias_option():
    mod = EdgeLinear(
        EdgeLinearConfig(
            input_dim=3, output_dim=4, num_transition_edges=2, bias=False
        )
    )
    assert mod.bias is None
    y = mod(torch.randn(2, 3), torch.tensor([0, 1]))
    assert y.shape == (2, 4)


def test_config_validation():
    with pytest.raises(ValueError):
        EdgeLinearConfig(input_dim=0, output_dim=4)
    with pytest.raises(ValueError):
        EdgeLinearConfig(input_dim=3, output_dim=0)
    with pytest.raises(ValueError):
        EdgeLinearConfig(input_dim=3, output_dim=4, num_transition_edges=0)


def test_wrong_input_dim_fails():
    mod = EdgeLinear(EdgeLinearConfig(input_dim=3, output_dim=4))
    with pytest.raises(ValueError):
        mod(torch.randn(2, 5))
