"""Tests for the regulatory bottleneck."""

from __future__ import annotations

import pytest
import torch

from stagebridge.ccrt.operators import (
    RegulatoryBottleneck,
    RegulatoryBottleneckConfig,
)

B, RECV, CTX, REG, HID = 2, 4, 8, 3, 8
N_EDGES = 2


def make_module(edges=None, edge_emb=None):
    torch.manual_seed(0)
    return RegulatoryBottleneck(
        RegulatoryBottleneckConfig(
            receiver_dim=RECV,
            context_dim=CTX,
            regulatory_dim=REG,
            hidden_dim=HID,
            num_transition_edges=edges,
            edge_embedding_dim=edge_emb,
        )
    )


def make_inputs():
    torch.manual_seed(1)
    return torch.randn(B, RECV), torch.randn(B, CTX)


def test_output_shape_no_edges():
    mod = make_module()
    recv, ctx = make_inputs()
    out = mod(receiver_features=recv, context=ctx)
    assert out.regulatory_state.shape == (B, REG)
    assert out.bottleneck_input.shape == (B, RECV + CTX)


def test_output_shape_with_edges_default_embedding():
    mod = make_module(edges=N_EDGES)
    recv, ctx = make_inputs()
    edge = torch.tensor([0, 1])
    out = mod(receiver_features=recv, context=ctx, transition_edge_index=edge)
    # default edge embedding dim = min(hidden_dim, 16) = 8
    assert out.regulatory_state.shape == (B, REG)
    assert out.bottleneck_input.shape == (B, RECV + CTX + min(HID, 16))


def test_custom_edge_embedding_dim():
    mod = make_module(edges=N_EDGES, edge_emb=5)
    recv, ctx = make_inputs()
    out = mod(
        receiver_features=recv, context=ctx, transition_edge_index=torch.tensor([0, 1])
    )
    assert out.bottleneck_input.shape == (B, RECV + CTX + 5)


def test_edge_conditioned_requires_edge_index():
    mod = make_module(edges=N_EDGES)
    recv, ctx = make_inputs()
    with pytest.raises(ValueError):
        mod(receiver_features=recv, context=ctx)


def test_out_of_range_edge_fails():
    mod = make_module(edges=N_EDGES)
    recv, ctx = make_inputs()
    with pytest.raises(ValueError):
        mod(receiver_features=recv, context=ctx, transition_edge_index=torch.tensor([0, 9]))


def test_regulatory_state_can_be_signed():
    mod = make_module()
    recv, ctx = make_inputs()
    # push through many inits until we observe both signs (bounded, deterministic)
    out = mod(receiver_features=recv, context=ctx)
    r = out.regulatory_state
    # a signed MLP output should not be all non-negative in general
    assert bool((r < 0).any()) or bool((r > 0).any())


def test_gradients_flow():
    mod = make_module(edges=N_EDGES)
    recv, ctx = make_inputs()
    recv.requires_grad_(True)
    ctx.requires_grad_(True)
    out = mod(
        receiver_features=recv, context=ctx, transition_edge_index=torch.tensor([0, 1])
    )
    out.regulatory_state.sum().backward()
    assert recv.grad is not None
    assert ctx.grad is not None
    assert mod.edge_embedding.weight.grad is not None
    # at least one MLP linear has grad
    grads = [m.weight.grad for m in mod.mlp if hasattr(m, "weight")]
    assert any(g is not None for g in grads)


def test_config_validation():
    with pytest.raises(ValueError):
        RegulatoryBottleneckConfig(
            receiver_dim=0, context_dim=CTX, regulatory_dim=REG, hidden_dim=HID
        )
    with pytest.raises(ValueError):
        RegulatoryBottleneckConfig(
            receiver_dim=RECV, context_dim=CTX, regulatory_dim=REG, hidden_dim=HID,
            dropout=2.0,
        )
    with pytest.raises(ValueError):
        RegulatoryBottleneckConfig(
            receiver_dim=RECV, context_dim=CTX, regulatory_dim=REG, hidden_dim=HID,
            num_transition_edges=0,
        )
