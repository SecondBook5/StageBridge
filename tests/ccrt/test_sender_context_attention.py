"""Tests for TypedSenderContextAttention."""

from __future__ import annotations

import pytest
import torch

from stagebridge.ccrt.sender_context import (
    TypedSenderContextAttention,
    TypedSenderContextAttentionConfig,
)

# Small deterministic problem sizes (per the milestone spec).
B, K = 2, 3
D_R, D_S = 4, 5
HIDDEN, HEADS = 8, 2
N_TYPES, EMPTY_TYPE = 4, 3
N_EDGES = 2


def make_config(**overrides):
    kwargs = dict(
        receiver_dim=D_R,
        sender_dim=D_S,
        hidden_dim=HIDDEN,
        num_heads=HEADS,
        num_sender_context_types=N_TYPES,
        empty_sender_context_type_id=EMPTY_TYPE,
        num_transition_edges=N_EDGES,
        distance_transform="log1p",
        use_uncertainty=True,
    )
    kwargs.update(overrides)
    return TypedSenderContextAttentionConfig(**kwargs)


def make_inputs(mask=None, with_uncertainty=True, with_edge=True):
    torch.manual_seed(0)
    inputs = dict(
        receiver_features=torch.randn(B, D_R),
        sender_features=torch.randn(B, K, D_S),
        sender_mask=(
            mask
            if mask is not None
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


def make_module(**cfg_overrides):
    torch.manual_seed(0)
    return TypedSenderContextAttention(make_config(**cfg_overrides))


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_config_requires_divisible_hidden_dim():
    with pytest.raises(ValueError):
        make_config(hidden_dim=9, num_heads=2)


def test_config_empty_type_in_range():
    with pytest.raises(ValueError):
        make_config(empty_sender_context_type_id=N_TYPES)


def test_config_bad_dropout():
    with pytest.raises(ValueError):
        make_config(dropout=1.5)


# ---------------------------------------------------------------------------
# Output shapes and normalization
# ---------------------------------------------------------------------------


def test_output_shapes():
    mod = make_module()
    out = mod(**make_inputs())
    assert out.context.shape == (B, HIDDEN)
    assert out.per_head_context.shape == (B, HEADS, HIDDEN // HEADS)
    assert out.attention_weights.shape == (B, HEADS, K + 1)
    assert out.attention_logits.shape == (B, HEADS, K + 1)
    assert out.sender_value_vectors.shape == (B, HEADS, K + 1, HIDDEN // HEADS)
    assert out.sender_mask_with_empty.shape == (B, K + 1)
    assert out.distance_with_empty.shape == (B, K + 1)
    assert out.sender_context_type_ids_with_empty.shape == (B, K + 1)
    assert out.uncertainty_with_empty.shape == (B, K + 1)


def test_attention_weights_sum_to_one():
    mod = make_module()
    out = mod(**make_inputs())
    sums = out.attention_weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_padded_positions_get_near_zero_attention():
    mod = make_module()
    out = mod(**make_inputs())
    # row 0: sender index 2 is padding; row 1: indices 1 and 2 are padding.
    assert float(out.attention_weights[0, :, 2].max()) < 1e-5
    assert float(out.attention_weights[1, :, 1].max()) < 1e-5
    assert float(out.attention_weights[1, :, 2].max()) < 1e-5


def test_empty_sender_receives_valid_attention():
    mod = make_module()
    out = mod(**make_inputs())
    # empty token is the last (K) index and is always unmasked
    assert bool((out.sender_mask_with_empty[:, -1] == 1).all())
    assert float(out.attention_weights[:, :, -1].min()) >= 0.0


def test_all_real_masked_still_finite_and_attends_empty():
    mod = make_module()
    inputs = make_inputs(mask=torch.zeros(B, K, dtype=torch.float32))
    out = mod(**inputs)
    assert torch.isfinite(out.context).all()
    sums = out.attention_weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
    # all attention concentrates on the empty token
    assert torch.allclose(
        out.attention_weights[:, :, -1],
        torch.ones(B, HEADS),
        atol=1e-5,
    )


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_type_ids_out_of_range_fails():
    mod = make_module()
    inputs = make_inputs()
    inputs["sender_context_type_ids"] = torch.tensor(
        [[0, 1, 9], [1, 2, 0]], dtype=torch.long
    )
    with pytest.raises(ValueError):
        mod(**inputs)


def test_missing_edge_when_required_fails():
    mod = make_module()
    inputs = make_inputs(with_edge=False)
    with pytest.raises(ValueError):
        mod(**inputs)


def test_edge_out_of_range_fails():
    mod = make_module()
    inputs = make_inputs()
    inputs["transition_edge_index"] = torch.tensor([0, 5], dtype=torch.long)
    with pytest.raises(ValueError):
        mod(**inputs)


def test_negative_distance_fails():
    mod = make_module()
    inputs = make_inputs()
    inputs["distance_to_receiver"] = torch.tensor(
        [[1.0, -2.0, 3.0], [4.0, 5.0, 6.0]]
    )
    with pytest.raises(ValueError):
        mod(**inputs)


def test_uncertainty_shape_mismatch_fails():
    mod = make_module()
    inputs = make_inputs()
    inputs["uncertainty"] = torch.zeros(B, K + 1)  # wrong K
    with pytest.raises(ValueError):
        mod(**inputs)


def test_edge_free_config_works_without_edge_index():
    mod = make_module(num_transition_edges=None)
    inputs = make_inputs(with_edge=False)
    out = mod(**inputs)
    assert out.context.shape == (B, HIDDEN)


def test_use_uncertainty_true_but_none_does_not_fail():
    mod = make_module()
    inputs = make_inputs(with_uncertainty=False)
    out = mod(**inputs)
    assert out.uncertainty_with_empty is None
    assert out.context.shape == (B, HIDDEN)


# ---------------------------------------------------------------------------
# Uncertainty downweighting
# ---------------------------------------------------------------------------


def test_higher_uncertainty_lowers_attention():
    # Two senders identical in every way except uncertainty; the higher-
    # uncertainty sender must receive strictly less attention. gamma>0 at init
    # because softplus(0) > 0.
    mod = make_module(num_transition_edges=None)
    feat = torch.randn(1, D_S)
    inputs = dict(
        receiver_features=torch.randn(1, D_R),
        sender_features=torch.stack([feat, feat], dim=1),  # [1, 2, D_S] identical
        sender_mask=torch.ones(1, 2),
        distance_to_receiver=torch.zeros(1, 2),  # identical distance
        sender_context_type_ids=torch.zeros(1, 2, dtype=torch.long),  # same type
        uncertainty=torch.tensor([[0.0, 5.0]]),  # sender 1 much more uncertain
    )
    out = mod(**inputs)
    a0 = float(out.attention_weights[0, :, 0].mean())
    a1 = float(out.attention_weights[0, :, 1].mean())
    assert a0 > a1


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


def test_gradients_flow_to_all_parameters():
    mod = make_module()
    inputs = make_inputs()
    out = mod(**inputs)
    out.context.sum().backward()

    assert mod.query_proj.weight.grad is not None
    assert mod.key_proj.weight.grad is not None
    assert mod.value_proj.weight.grad is not None
    assert mod.type_bias.weight.grad is not None
    assert mod.empty_sender_feature.grad is not None
    assert mod.distance_lambda_raw.grad is not None
    # uncertainty gamma receives gradient because uncertainty is provided
    assert mod.uncertainty_gamma_raw.grad is not None
    assert bool(mod.uncertainty_gamma_raw.grad.abs().sum() > 0)


def test_gradients_flow_edge_free_config():
    mod = make_module(num_transition_edges=None)
    inputs = make_inputs(with_edge=False)
    out = mod(**inputs)
    out.context.sum().backward()
    assert mod.distance_lambda_raw.grad is not None
