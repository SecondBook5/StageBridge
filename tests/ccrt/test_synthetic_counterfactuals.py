"""Tests for synthetic counterfactual interventions."""

from __future__ import annotations

import pytest
import torch

from stagebridge.ccrt.synthetic import (
    SyntheticSystemConfig,
    attach_teacher_targets,
    generate_synthetic_dataset,
    remove_all_sender_context,
    remove_sender_context_type,
    replace_transition_edge,
    set_sender_distances,
)

SYS = SyntheticSystemConfig(train_batches=1, validation_batches=1, test_batches=1, batch_size=4)


def make_dataset(scenario="mixed_drift_growth"):
    return generate_synthetic_dataset(system=SYS, scenario_id=scenario)


def test_no_input_mutation():
    ds = make_dataset()
    batch = ds.train[0].factual_batch
    mask_before = batch.sender_mask.clone()
    remove_all_sender_context(batch)
    assert torch.equal(batch.sender_mask, mask_before)


def test_remove_all_masks_everything():
    ds = make_dataset()
    out = remove_all_sender_context(ds.train[0].factual_batch)
    assert not bool(out.sender_mask.any())


def test_remove_one_type_leaves_others():
    ds = make_dataset()
    batch = ds.train[0].factual_batch
    # find a type index present in the batch
    types = batch.sender_context_type_ids
    target = int(types[types >= 0].min())
    out = remove_sender_context_type(batch, target)
    # positions of that type are all masked off
    assert not bool(out.sender_mask[types == target].any())
    # positions of other real types remain unaffected where originally unmasked
    other = types != target
    assert torch.equal(
        out.sender_mask[other], (batch.sender_mask.to(torch.bool) & other)[other]
    )


def test_distance_replacement():
    ds = make_dataset()
    batch = ds.train[0].factual_batch
    out = set_sender_distances(batch, 1.75, real_senders_only=True)
    mask = batch.sender_mask.to(torch.bool)
    assert torch.allclose(
        out.distance_to_receiver[mask],
        torch.full_like(out.distance_to_receiver[mask], 1.75),
    )


def test_edge_replacement():
    ds = make_dataset()
    out = replace_transition_edge(ds.train[0].factual_batch, 1)
    assert bool((out.transition_edge_index == 1).all())


def test_invalid_distance_fails():
    ds = make_dataset()
    with pytest.raises(ValueError):
        set_sender_distances(ds.train[0].factual_batch, -1.0)


def test_teacher_targets_match_truth_when_noise_zero():
    ds = make_dataset()
    batch = ds.train[0].factual_batch
    new_batch, truth = attach_teacher_targets(
        batch=batch, teacher=ds.teacher, target_noise_std=0.0, growth_noise_std=0.0, seed=1
    )
    assert torch.allclose(
        new_batch.target_semantic_features,
        truth.destination_semantic_features.to(new_batch.target_semantic_features.dtype),
        atol=1e-6,
    )
    assert torch.allclose(
        new_batch.growth_targets, truth.full_growth.to(new_batch.growth_targets.dtype), atol=1e-6
    )


def test_changed_noise_seed_changes_targets():
    ds = make_dataset()
    batch = ds.train[0].factual_batch
    b1, _ = attach_teacher_targets(batch=batch, teacher=ds.teacher, target_noise_std=0.1, seed=1)
    b2, _ = attach_teacher_targets(batch=batch, teacher=ds.teacher, target_noise_std=0.1, seed=2)
    assert not torch.equal(b1.target_semantic_features, b2.target_semantic_features)
