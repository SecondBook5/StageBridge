"""Tests for the drift and growth heads."""

from __future__ import annotations

import pytest
import torch

from stagebridge.ccrt.operators import (
    DriftHead,
    DriftHeadConfig,
    GrowthHead,
    GrowthHeadConfig,
)

B, RECV, CTX, REG, HID = 2, 4, 8, 3, 8
DRIFT, GROWTH = 6, 1
N_EDGES = 2


def make_inputs():
    torch.manual_seed(1)
    return (
        torch.randn(B, RECV),
        torch.randn(B, CTX),
        torch.randn(B, REG),
    )


# ---------------------------------------------------------------------------
# Drift head
# ---------------------------------------------------------------------------


def make_drift(edges=N_EDGES):
    torch.manual_seed(0)
    return DriftHead(
        DriftHeadConfig(
            receiver_dim=RECV,
            context_dim=CTX,
            regulatory_dim=REG,
            drift_dim=DRIFT,
            hidden_dim=HID,
            num_transition_edges=edges,
        )
    )


def test_drift_shapes():
    mod = make_drift()
    recv, ctx, reg = make_inputs()
    out = mod(
        receiver_features=recv,
        context=ctx,
        regulatory_state=reg,
        transition_edge_index=torch.tensor([0, 1]),
    )
    for t in (
        out.self_drift,
        out.regulatory_drift,
        out.residual_drift,
        out.context_delta_drift,
        out.full_drift,
    ):
        assert t.shape == (B, DRIFT)


def test_drift_arithmetic_exact():
    mod = make_drift()
    recv, ctx, reg = make_inputs()
    out = mod(
        receiver_features=recv,
        context=ctx,
        regulatory_state=reg,
        transition_edge_index=torch.tensor([0, 1]),
    )
    assert torch.allclose(
        out.context_delta_drift, out.regulatory_drift + out.residual_drift, atol=1e-6
    )
    assert torch.allclose(
        out.full_drift,
        out.self_drift + out.regulatory_drift + out.residual_drift,
        atol=1e-6,
    )


def test_drift_requires_edge_when_configured():
    mod = make_drift()
    recv, ctx, reg = make_inputs()
    with pytest.raises(ValueError):
        mod(receiver_features=recv, context=ctx, regulatory_state=reg)


def test_drift_no_edge_config_works():
    mod = make_drift(edges=None)
    recv, ctx, reg = make_inputs()
    out = mod(receiver_features=recv, context=ctx, regulatory_state=reg)
    assert out.full_drift.shape == (B, DRIFT)


def test_drift_can_be_signed():
    mod = make_drift()
    recv, ctx, reg = make_inputs()
    out = mod(
        receiver_features=recv,
        context=ctx,
        regulatory_state=reg,
        transition_edge_index=torch.tensor([0, 1]),
    )
    assert bool((out.full_drift < 0).any())
    assert bool((out.full_drift > 0).any())


def test_drift_gradients_flow():
    mod = make_drift()
    recv, ctx, reg = make_inputs()
    recv.requires_grad_(True)
    ctx.requires_grad_(True)
    reg.requires_grad_(True)
    out = mod(
        receiver_features=recv,
        context=ctx,
        regulatory_state=reg,
        transition_edge_index=torch.tensor([0, 1]),
    )
    out.full_drift.sum().backward()
    assert recv.grad is not None
    assert ctx.grad is not None
    assert reg.grad is not None
    assert mod.regulatory_map.weight.grad is not None
    # self MLP + residual MLP received gradient
    assert any(
        m.weight.grad is not None for m in mod.self_mlp if hasattr(m, "weight")
    )
    assert any(
        m.weight.grad is not None for m in mod.residual_mlp if hasattr(m, "weight")
    )


# ---------------------------------------------------------------------------
# Growth head
# ---------------------------------------------------------------------------


def make_growth(edges=N_EDGES):
    torch.manual_seed(0)
    return GrowthHead(
        GrowthHeadConfig(
            receiver_dim=RECV,
            context_dim=CTX,
            regulatory_dim=REG,
            growth_dim=GROWTH,
            hidden_dim=HID,
            num_transition_edges=edges,
        )
    )


def test_growth_shapes():
    mod = make_growth()
    recv, ctx, reg = make_inputs()
    out = mod(
        receiver_features=recv,
        context=ctx,
        regulatory_state=reg,
        transition_edge_index=torch.tensor([0, 1]),
    )
    for t in (
        out.self_growth,
        out.regulatory_growth,
        out.residual_growth,
        out.context_delta_growth,
        out.full_growth,
    ):
        assert t.shape == (B, GROWTH)


def test_growth_arithmetic_exact():
    mod = make_growth()
    recv, ctx, reg = make_inputs()
    out = mod(
        receiver_features=recv,
        context=ctx,
        regulatory_state=reg,
        transition_edge_index=torch.tensor([0, 1]),
    )
    assert torch.allclose(
        out.context_delta_growth,
        out.regulatory_growth + out.residual_growth,
        atol=1e-6,
    )
    assert torch.allclose(
        out.full_growth,
        out.self_growth + out.regulatory_growth + out.residual_growth,
        atol=1e-6,
    )


def test_growth_can_be_signed():
    # growth_dim wider so both signs are likely present
    torch.manual_seed(0)
    mod = GrowthHead(
        GrowthHeadConfig(
            receiver_dim=RECV,
            context_dim=CTX,
            regulatory_dim=REG,
            growth_dim=8,
            hidden_dim=HID,
            num_transition_edges=N_EDGES,
        )
    )
    recv, ctx, reg = make_inputs()
    out = mod(
        receiver_features=recv,
        context=ctx,
        regulatory_state=reg,
        transition_edge_index=torch.tensor([0, 1]),
    )
    assert bool((out.full_growth < 0).any())
    assert bool((out.full_growth > 0).any())


def test_growth_requires_edge_when_configured():
    mod = make_growth()
    recv, ctx, reg = make_inputs()
    with pytest.raises(ValueError):
        mod(receiver_features=recv, context=ctx, regulatory_state=reg)


def test_growth_gradients_flow():
    mod = make_growth()
    recv, ctx, reg = make_inputs()
    out = mod(
        receiver_features=recv,
        context=ctx,
        regulatory_state=reg,
        transition_edge_index=torch.tensor([0, 1]),
    )
    out.full_growth.sum().backward()
    assert mod.regulatory_map.weight.grad is not None
