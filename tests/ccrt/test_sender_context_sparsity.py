"""Tests for attention entropy and value L1 sparsity losses."""

from __future__ import annotations

import pytest
import torch

from stagebridge.ccrt.sender_context import (
    attention_entropy_loss,
    value_l1_sparsity_loss,
)


def test_sharp_attention_has_lower_entropy_than_uniform():
    # [B=1, H=1, K=4]
    sharp = torch.tensor([[[0.97, 0.01, 0.01, 0.01]]])
    uniform = torch.tensor([[[0.25, 0.25, 0.25, 0.25]]])
    assert float(attention_entropy_loss(sharp)) < float(
        attention_entropy_loss(uniform)
    )


def test_uniform_entropy_matches_log_k():
    k = 4
    uniform = torch.full((1, 1, k), 1.0 / k)
    ent = float(attention_entropy_loss(uniform))
    assert ent == pytest.approx(torch.log(torch.tensor(float(k))).item(), abs=1e-4)


def test_masked_positions_do_not_contribute_to_entropy():
    # Two identical distributions except one has an extra (masked) position.
    weights = torch.tensor([[[0.5, 0.5, 0.3]]])  # third is padding, masked out
    mask = torch.tensor([[1, 1, 0]])
    ent_masked = float(attention_entropy_loss(weights, sender_mask=mask))
    # equals entropy of just the first two entries
    two = torch.tensor([[[0.5, 0.5]]])
    ent_two = float(attention_entropy_loss(two))
    assert ent_masked == pytest.approx(ent_two, abs=1e-5)


def test_entropy_returns_scalar():
    w = torch.rand(2, 2, 3)
    w = w / w.sum(dim=-1, keepdim=True)
    out = attention_entropy_loss(w)
    assert out.dim() == 0


def test_value_l1_simple_value():
    # [B=1, H=1, K=2, D=2]
    effects = torch.tensor([[[[1.0, -1.0], [2.0, -2.0]]]])
    # mean abs = (1+1+2+2)/4 = 1.5
    assert float(value_l1_sparsity_loss(effects)) == pytest.approx(1.5)


def test_value_l1_masked_positions_excluded():
    effects = torch.tensor([[[[1.0, -1.0], [10.0, -10.0]]]])
    mask = torch.tensor([[1, 0]])  # second sender is padding
    # only first sender counts: mean abs over [1,-1] = 1.0
    assert float(value_l1_sparsity_loss(effects, sender_mask=mask)) == pytest.approx(1.0)


def test_value_l1_rank3_supported():
    effects = torch.tensor([[[1.0, -3.0]]])  # [B=1, H=1, K=2]
    assert float(value_l1_sparsity_loss(effects)) == pytest.approx(2.0)


def test_value_l1_returns_scalar():
    out = value_l1_sparsity_loss(torch.randn(2, 2, 3, 4))
    assert out.dim() == 0


def test_bad_attention_rank_fails():
    with pytest.raises(ValueError):
        attention_entropy_loss(torch.rand(2, 3))  # rank 2
