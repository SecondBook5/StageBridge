"""Counterfactual interventions on synthetic training batches.

Each function returns a NEW ``CCRTTrainingBatch`` (never mutates its input) so a
factual batch and its counterfactual can be compared. ``attach_teacher_targets``
re-derives the target semantic population and growth targets from the teacher for
a given (possibly intervened) batch, using independent deterministic noise.

For the single synthetic system, the batch's global sender-context-type indices
coincide with the teacher's local ids (the registry preserves order), so the
same integer identifies a type here and in the teacher.
"""

from __future__ import annotations

import torch

from ..training.batch import CCRTTrainingBatch
from .ground_truth import SyntheticGroundTruth, SyntheticTeacher

__all__ = [
    "remove_all_sender_context",
    "remove_sender_context_type",
    "set_sender_distances",
    "replace_transition_edge",
    "attach_teacher_targets",
]


def _clone_batch(batch: CCRTTrainingBatch, **overrides) -> CCRTTrainingBatch:
    fields = dict(
        receiver_features=batch.receiver_features,
        sender_features=batch.sender_features,
        sender_mask=batch.sender_mask,
        distance_to_receiver=batch.distance_to_receiver,
        sender_context_type_ids=batch.sender_context_type_ids,
        transition_edge_index=batch.transition_edge_index,
        source_semantic_features=batch.source_semantic_features,
        target_semantic_features=batch.target_semantic_features,
        uncertainty=batch.uncertainty,
        source_weights=batch.source_weights,
        target_weights=batch.target_weights,
        growth_targets=batch.growth_targets,
        growth_mask=batch.growth_mask,
        metadata=dict(batch.metadata),
    )
    fields.update(overrides)
    new = CCRTTrainingBatch(**fields)
    new.validate()
    return new


def remove_all_sender_context(batch: CCRTTrainingBatch) -> CCRTTrainingBatch:
    """Return a batch with all sender positions masked off."""
    new_mask = torch.zeros_like(batch.sender_mask, dtype=torch.bool)
    return _clone_batch(batch, sender_mask=new_mask)


def remove_sender_context_type(
    batch: CCRTTrainingBatch, sender_context_type_index: int
) -> CCRTTrainingBatch:
    """Return a batch with the given sender-context type masked off.

    Previously-masked positions stay masked.
    """
    if sender_context_type_index < 0:
        raise ValueError("sender_context_type_index must be >= 0")
    is_type = batch.sender_context_type_ids == int(sender_context_type_index)
    new_mask = batch.sender_mask.to(torch.bool) & (~is_type)
    return _clone_batch(batch, sender_mask=new_mask)


def set_sender_distances(
    batch: CCRTTrainingBatch,
    distance: float,
    *,
    real_senders_only: bool = True,
) -> CCRTTrainingBatch:
    """Return a batch with sender distances set to a constant value."""
    if distance < 0:
        raise ValueError("distance must be >= 0")
    new_dist = batch.distance_to_receiver.clone()
    if real_senders_only:
        mask = batch.sender_mask.to(torch.bool)
        new_dist = torch.where(mask, torch.full_like(new_dist, float(distance)), new_dist)
    else:
        new_dist = torch.full_like(new_dist, float(distance))
    return _clone_batch(batch, distance_to_receiver=new_dist)


def replace_transition_edge(
    batch: CCRTTrainingBatch, transition_edge_index: int
) -> CCRTTrainingBatch:
    """Return a batch with all transition-edge indices replaced."""
    if transition_edge_index < 0:
        raise ValueError("transition_edge_index must be >= 0")
    if batch.transition_edge_index is None:
        raise ValueError("batch has no transition_edge_index to replace")
    new_edges = torch.full_like(batch.transition_edge_index, int(transition_edge_index))
    return _clone_batch(batch, transition_edge_index=new_edges)


def attach_teacher_targets(
    *,
    batch: CCRTTrainingBatch,
    teacher: SyntheticTeacher,
    target_noise_std: float = 0.0,
    growth_noise_std: float = 0.0,
    seed: int,
) -> tuple[CCRTTrainingBatch, SyntheticGroundTruth]:
    """Re-derive teacher targets for a (possibly intervened) batch.

    Uses the batch's integer type/edge ids directly as the teacher's local ids
    (valid for the single synthetic system). Independent deterministic noise.
    """
    truth = teacher.evaluate(
        receiver_features=batch.receiver_features,
        sender_features=batch.sender_features,
        sender_mask=batch.sender_mask,
        distance_to_receiver=batch.distance_to_receiver,
        sender_context_type_ids=batch.sender_context_type_ids,
        transition_edge_index=batch.transition_edge_index,
        source_semantic_features=batch.source_semantic_features,
    )
    gen = torch.Generator().manual_seed(int(seed))
    dest = truth.destination_semantic_features
    tgt = dest + target_noise_std * torch.randn(*dest.shape, generator=gen, dtype=dest.dtype)
    growth = truth.full_growth
    growth_t = growth + growth_noise_std * torch.randn(
        *growth.shape, generator=gen, dtype=growth.dtype
    )
    new_batch = _clone_batch(
        batch, target_semantic_features=tgt, growth_targets=growth_t
    )
    return new_batch, truth
