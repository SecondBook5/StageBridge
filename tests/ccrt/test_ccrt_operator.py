"""Tests for the full ContextResidualTransportOperator."""

from __future__ import annotations

import pytest
import torch

from stagebridge.ccrt.operators import (
    ContextResidualTransportConfig,
    ContextResidualTransportOperator,
)

# Small deterministic sizes per the milestone spec.
B, K = 2, 3
RECV, SEND = 4, 5
HIDDEN, HEADS = 8, 2
N_TYPES, EMPTY_TYPE = 4, 3
N_EDGES = 2
REG, DRIFT, GROWTH = 3, 6, 1


def make_config(**overrides):
    kwargs = dict(
        receiver_dim=RECV,
        sender_dim=SEND,
        hidden_dim=HIDDEN,
        num_heads=HEADS,
        num_sender_context_types=N_TYPES,
        empty_sender_context_type_id=EMPTY_TYPE,
        regulatory_dim=REG,
        drift_dim=DRIFT,
        growth_dim=GROWTH,
        num_transition_edges=N_EDGES,
    )
    kwargs.update(overrides)
    return ContextResidualTransportConfig(**kwargs)


def make_operator(**overrides):
    torch.manual_seed(0)
    return ContextResidualTransportOperator(make_config(**overrides))


def make_inputs(mask=None, with_uncertainty=True, with_edge=True):
    torch.manual_seed(0)
    inputs = dict(
        receiver_features=torch.randn(B, RECV),
        sender_features=torch.randn(B, K, SEND),
        sender_mask=(
            mask if mask is not None
            else torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.float32)
        ),
        distance_to_receiver=torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        sender_context_type_ids=torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long),
    )
    if with_edge:
        inputs["transition_edge_index"] = torch.tensor([0, 1], dtype=torch.long)
    if with_uncertainty:
        inputs["uncertainty"] = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    return inputs


def test_full_forward_shapes():
    op = make_operator()
    out = op(**make_inputs())
    assert out.context.shape == (B, HIDDEN)
    assert out.per_head_context.shape == (B, HEADS, HIDDEN // HEADS)
    assert out.attention_weights.shape == (B, HEADS, K + 1)
    assert out.sender_value_vectors.shape == (B, HEADS, K + 1, HIDDEN // HEADS)
    assert out.sender_effects.shape == (B, HEADS, K + 1, HIDDEN // HEADS)
    assert out.aggregated_sender_effect.shape == (B, HEADS, HIDDEN // HEADS)
    assert out.regulatory_state.shape == (B, REG)
    for t in (out.self_drift, out.regulatory_drift, out.residual_drift,
              out.context_delta_drift, out.full_drift):
        assert t.shape == (B, DRIFT)
    for t in (out.self_growth, out.regulatory_growth, out.residual_growth,
              out.context_delta_growth, out.full_growth):
        assert t.shape == (B, GROWTH)


def test_attention_weights_sum_to_one():
    op = make_operator()
    out = op(**make_inputs())
    sums = out.attention_weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_drift_arithmetic_exact():
    op = make_operator()
    out = op(**make_inputs())
    assert torch.allclose(
        out.context_delta_drift, out.regulatory_drift + out.residual_drift, atol=1e-6
    )
    assert torch.allclose(
        out.full_drift,
        out.self_drift + out.regulatory_drift + out.residual_drift,
        atol=1e-6,
    )


def test_growth_arithmetic_exact():
    op = make_operator()
    out = op(**make_inputs())
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


def test_all_real_masked_still_finite():
    op = make_operator()
    out = op(**make_inputs(mask=torch.zeros(B, K, dtype=torch.float32)))
    assert torch.isfinite(out.context).all()
    assert torch.isfinite(out.full_drift).all()
    assert torch.isfinite(out.full_growth).all()
    sums = out.attention_weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_edge_required_when_configured():
    op = make_operator()
    with pytest.raises(ValueError):
        op(**make_inputs(with_edge=False))


def test_edge_optional_when_none():
    op = make_operator(num_transition_edges=None)
    out = op(**make_inputs(with_edge=False))
    assert out.full_drift.shape == (B, DRIFT)


def test_negative_distance_fails():
    op = make_operator()
    inputs = make_inputs()
    inputs["distance_to_receiver"] = torch.tensor([[1.0, -1.0, 2.0], [3.0, 4.0, 5.0]])
    with pytest.raises(ValueError):
        op(**inputs)


def test_output_includes_sender_effects():
    op = make_operator()
    out = op(**make_inputs())
    # sender effects can be signed
    assert bool((out.sender_effects != 0).any())
    assert torch.allclose(
        out.aggregated_sender_effect, out.sender_effects.sum(dim=2), atol=1e-6
    )


def test_gradients_flow_to_all_submodules():
    op = make_operator()
    out = op(**make_inputs())
    # The signed sender effects are a first-class *exposed* output (a diagnostic
    # decomposition), not an input to the drift/growth heads — those consume the
    # attention context vector. So a loss over the full operator output includes
    # the aggregated sender effect to exercise the effect projection too.
    loss = (
        out.full_drift.sum()
        + out.full_growth.sum()
        + out.aggregated_sender_effect.sum()
    )
    loss.backward()

    # attention
    assert op.attention.query_proj.weight.grad is not None
    assert op.attention.empty_sender_feature.grad is not None
    assert op.attention.distance_lambda_raw.grad is not None
    # signed sender effects
    assert op.sender_effects.effect_proj.weight.grad is not None
    # regulatory bottleneck (at least one MLP linear)
    assert any(
        m.weight.grad is not None
        for m in op.regulatory_bottleneck.mlp
        if hasattr(m, "weight")
    )
    # drift + growth heads (edge-conditioned regulatory maps)
    assert op.drift_head.regulatory_map.weight.grad is not None
    assert op.growth_head.regulatory_map.weight.grad is not None


def test_uncertainty_optional():
    op = make_operator()
    out = op(**make_inputs(with_uncertainty=False))
    assert out.uncertainty_with_empty is None
    assert out.full_drift.shape == (B, DRIFT)


def test_config_validation():
    with pytest.raises(ValueError):
        make_config(hidden_dim=9, num_heads=2)  # not divisible
    with pytest.raises(ValueError):
        make_config(empty_sender_context_type_id=N_TYPES)  # out of range
    with pytest.raises(ValueError):
        make_config(regulatory_dim=0)


def test_sender_effect_dim_defaults_to_head_dim():
    cfg = make_config()
    assert cfg.resolved_sender_effect_dim == HIDDEN // HEADS
    cfg2 = make_config(sender_effect_dim=10)
    assert cfg2.resolved_sender_effect_dim == 10
