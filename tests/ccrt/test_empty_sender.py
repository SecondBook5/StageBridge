"""Tests for the empty sender-context token."""

from __future__ import annotations

import pytest
import torch

from stagebridge.ccrt.sender_context import append_empty_sender_context


def make_inputs(with_uncertainty=False):
    torch.manual_seed(0)
    b, k, d_s = 2, 3, 5
    inputs = dict(
        sender_features=torch.randn(b, k, d_s),
        sender_mask=torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.float32),
        distance_to_receiver=torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        sender_context_type_ids=torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long),
        empty_sender_feature=torch.arange(d_s, dtype=torch.float32),
        empty_sender_context_type_id=3,
    )
    if with_uncertainty:
        inputs["uncertainty"] = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    return inputs


def test_shape_increases_k_by_one():
    inputs = make_inputs()
    out = append_empty_sender_context(**inputs)
    assert out["sender_features"].shape == (2, 4, 5)
    assert out["sender_mask"].shape == (2, 4)
    assert out["distance_to_receiver"].shape == (2, 4)
    assert out["sender_context_type_ids"].shape == (2, 4)


def test_empty_mask_is_one():
    out = append_empty_sender_context(**make_inputs())
    assert bool((out["sender_mask"][:, -1] == 1).all())


def test_empty_distance_is_zero():
    out = append_empty_sender_context(**make_inputs())
    assert bool((out["distance_to_receiver"][:, -1] == 0).all())


def test_empty_type_id_correct():
    out = append_empty_sender_context(**make_inputs())
    assert bool((out["sender_context_type_ids"][:, -1] == 3).all())


def test_empty_feature_broadcast():
    inputs = make_inputs()
    out = append_empty_sender_context(**inputs)
    for row in range(2):
        assert torch.allclose(
            out["sender_features"][row, -1], inputs["empty_sender_feature"]
        )


def test_uncertainty_appended_as_zero():
    inputs = make_inputs(with_uncertainty=True)
    out = append_empty_sender_context(**inputs)
    assert "uncertainty" in out
    assert out["uncertainty"].shape == (2, 4)
    assert bool((out["uncertainty"][:, -1] == 0).all())


def test_uncertainty_absent_when_not_provided():
    out = append_empty_sender_context(**make_inputs())
    assert "uncertainty" not in out


def test_inputs_not_mutated():
    inputs = make_inputs(with_uncertainty=True)
    feats_clone = inputs["sender_features"].clone()
    mask_clone = inputs["sender_mask"].clone()
    dist_clone = inputs["distance_to_receiver"].clone()
    types_clone = inputs["sender_context_type_ids"].clone()
    unc_clone = inputs["uncertainty"].clone()
    append_empty_sender_context(**inputs)
    assert torch.equal(inputs["sender_features"], feats_clone)
    assert torch.equal(inputs["sender_mask"], mask_clone)
    assert torch.equal(inputs["distance_to_receiver"], dist_clone)
    assert torch.equal(inputs["sender_context_type_ids"], types_clone)
    assert torch.equal(inputs["uncertainty"], unc_clone)


def test_all_masked_real_senders_still_yield_valid_empty():
    inputs = make_inputs()
    inputs["sender_mask"] = torch.zeros(2, 3, dtype=torch.float32)  # all real masked
    out = append_empty_sender_context(**inputs)
    # empty token still valid for every receiver
    assert bool((out["sender_mask"][:, -1] == 1).all())
    # at least one valid key per row
    assert bool((out["sender_mask"].sum(dim=1) >= 1).all())


def test_bad_empty_feature_shape_fails():
    inputs = make_inputs()
    inputs["empty_sender_feature"] = torch.zeros(4)  # wrong D_S (should be 5)
    with pytest.raises(ValueError):
        append_empty_sender_context(**inputs)
