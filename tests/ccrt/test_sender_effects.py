"""Tests for signed sender effects."""

from __future__ import annotations

import pytest
import torch

from stagebridge.ccrt.sender_context import (
    SignedSenderEffects,
    SignedSenderEffectsConfig,
)


def make_module(head_dim=4, effect_dim=3, num_heads=2):
    torch.manual_seed(0)
    return SignedSenderEffects(
        SignedSenderEffectsConfig(
            head_dim=head_dim, effect_dim=effect_dim, num_heads=num_heads
        )
    )


def make_inputs(b=2, h=2, k=3, dh=4):
    torch.manual_seed(1)
    values = torch.randn(b, h, k, dh)
    attn = torch.rand(b, h, k)
    attn = attn / attn.sum(dim=-1, keepdim=True)
    return values, attn


def test_output_shapes():
    mod = make_module()
    values, attn = make_inputs()
    out = mod(sender_value_vectors=values, attention_weights=attn)
    assert out.sender_effects.shape == (2, 2, 3, 3)
    assert out.aggregated_effect.shape == (2, 2, 3)


def test_effects_can_be_positive_and_negative():
    mod = make_module()
    values, attn = make_inputs()
    out = mod(sender_value_vectors=values, attention_weights=attn)
    assert bool((out.sender_effects > 0).any())
    assert bool((out.sender_effects < 0).any())


def test_masked_senders_contribute_zero():
    mod = make_module()
    values, attn = make_inputs()
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.float32)
    out = mod(sender_value_vectors=values, attention_weights=attn, sender_mask=mask)
    # masked positions must be exactly zero across heads and effect dims
    assert torch.equal(out.sender_effects[0, :, 2, :], torch.zeros(2, 3))
    assert torch.equal(out.sender_effects[1, :, 1, :], torch.zeros(2, 3))
    assert torch.equal(out.sender_effects[1, :, 2, :], torch.zeros(2, 3))


def test_aggregated_equals_sum_over_k():
    mod = make_module()
    values, attn = make_inputs()
    out = mod(sender_value_vectors=values, attention_weights=attn)
    assert torch.allclose(out.aggregated_effect, out.sender_effects.sum(dim=2))


def test_gradients_flow():
    mod = make_module()
    values, attn = make_inputs()
    values.requires_grad_(True)
    out = mod(sender_value_vectors=values, attention_weights=attn)
    out.aggregated_effect.sum().backward()
    assert values.grad is not None
    assert mod.effect_proj.weight.grad is not None


def test_config_validation():
    with pytest.raises(ValueError):
        SignedSenderEffectsConfig(head_dim=0, effect_dim=3, num_heads=2)
    with pytest.raises(ValueError):
        SignedSenderEffectsConfig(head_dim=4, effect_dim=0, num_heads=2)
    with pytest.raises(ValueError):
        SignedSenderEffectsConfig(head_dim=4, effect_dim=3, num_heads=0)


def test_wrong_head_dim_fails():
    mod = make_module(head_dim=4)
    values = torch.randn(2, 2, 3, 5)  # head dim 5 != 4
    attn = torch.rand(2, 2, 3)
    with pytest.raises(ValueError):
        mod(sender_value_vectors=values, attention_weights=attn)
