"""Tests for CCRTBatch shape and leakage validation.

All tensors are plain nested Python lists so no numpy/torch is required.
"""

from __future__ import annotations

import pytest

from stagebridge.ccrt.contracts import (
    CCRTForbiddenFieldError,
    CCRTLeakageError,
    CCRTShapeError,
    CCRTValidationError,
)
from stagebridge.ccrt.data import CCRTBatch


def _zeros(*shape):
    """Build a nested list of zeros with the given rectangular shape."""
    if not shape:
        return 0.0
    return [_zeros(*shape[1:]) for _ in range(shape[0])]


def make_valid_batch(**overrides) -> CCRTBatch:
    """A valid batch with B=2, K=4, D_R=3, D_S=5, D_Z=6, D_REG=2."""
    kwargs = dict(
        receiver_features=_zeros(2, 3),
        sender_features=_zeros(2, 4, 5),
        sender_mask=_zeros(2, 4),
        distance_to_receiver=_zeros(2, 4),
        uncertainty=_zeros(2, 4),
        semantic_features=_zeros(2, 6),
        regulatory_features=_zeros(2, 2),
        biological_system_id=["system_a", "system_a"],
        transition_edge_id=["edge_1", "edge_1"],
    )
    kwargs.update(overrides)
    return CCRTBatch(**kwargs)


def test_valid_batch_passes():
    batch = make_valid_batch()
    batch.validate()


def test_batch_size_and_max_sender_context():
    batch = make_valid_batch()
    assert batch.batch_size() == 2
    assert batch.max_sender_context() == 4


def test_mismatched_sender_mask_shape_fails():
    batch = make_valid_batch(sender_mask=_zeros(2, 3))  # K=3 != 4
    with pytest.raises(CCRTShapeError):
        batch.validate()


def test_mismatched_distance_shape_fails():
    batch = make_valid_batch(distance_to_receiver=_zeros(2, 5))  # K=5 != 4
    with pytest.raises(CCRTShapeError):
        batch.validate()


def test_mismatched_semantic_batch_fails():
    batch = make_valid_batch(semantic_features=_zeros(3, 6))  # B=3 != 2
    with pytest.raises(CCRTShapeError):
        batch.validate()


def test_mismatched_regulatory_batch_fails():
    batch = make_valid_batch(regulatory_features=_zeros(3, 2))  # B=3 != 2
    with pytest.raises(CCRTShapeError):
        batch.validate()


def test_wrong_rank_receiver_features_fails():
    batch = make_valid_batch(receiver_features=_zeros(2, 3, 1))  # rank 3
    with pytest.raises(CCRTShapeError):
        batch.validate()


def test_model_inputs_with_leakage_field_fails():
    batch = make_valid_batch(model_inputs={"future_expression": _zeros(2, 3)})
    with pytest.raises(CCRTLeakageError):
        batch.validate()


def test_metadata_with_outcome_label_fails():
    batch = make_valid_batch(metadata={"outcome_label": "responder"})
    with pytest.raises(CCRTLeakageError):
        batch.validate()


def test_model_inputs_with_ring_id_fails():
    batch = make_valid_batch(model_inputs={"ring_id": _zeros(2, 4)})
    with pytest.raises(CCRTForbiddenFieldError):
        batch.validate()


def test_targets_may_not_smuggle_leakage_field():
    # Even under `targets`, the exact forbidden leakage names are rejected in
    # this milestone.
    batch = make_valid_batch(targets={"future_expression": _zeros(2, 3)})
    with pytest.raises(CCRTLeakageError):
        batch.validate()


def test_targets_allow_legitimate_target_name():
    # A non-forbidden training target is fine.
    batch = make_valid_batch(targets={"drift_velocity": _zeros(2, 3)})
    batch.validate()


def test_missing_transition_edge_id_fails():
    batch = make_valid_batch(transition_edge_id="")
    with pytest.raises(Exception):
        batch.validate()


def test_missing_biological_system_id_fails():
    batch = make_valid_batch(biological_system_id=[])
    with pytest.raises(Exception):
        batch.validate()


def test_optional_tensors_may_be_absent():
    batch = make_valid_batch(
        uncertainty=None, semantic_features=None, regulatory_features=None
    )
    batch.validate()


def test_scalar_conditioning_ids_allowed():
    batch = make_valid_batch(
        biological_system_id="system_a", transition_edge_id="edge_1"
    )
    batch.validate()


# -- Milestone 6 contract repair: sender_context_type_ids [B, K] --

def test_sender_context_type_ids_valid():
    # B=2, K=4: grammar-id strings for real senders, None for padding.
    ids = [
        ["ctx_0", "ctx_1", "ctx_0", None],
        ["ctx_1", None, None, None],
    ]
    batch = make_valid_batch(sender_context_type_ids=ids)
    batch.validate()


def test_sender_context_type_ids_absent_ok():
    batch = make_valid_batch(sender_context_type_ids=None)
    batch.validate()


def test_sender_context_type_ids_wrong_b_fails():
    ids = [["ctx_0", "ctx_1", "ctx_0", None]]  # only 1 row, B=2
    batch = make_valid_batch(sender_context_type_ids=ids)
    with pytest.raises(CCRTValidationError):
        batch.validate()


def test_sender_context_type_ids_wrong_k_fails():
    ids = [["ctx_0", "ctx_1"], ["ctx_0", "ctx_1"]]  # K=2, expected K=4
    batch = make_valid_batch(sender_context_type_ids=ids)
    with pytest.raises(CCRTValidationError):
        batch.validate()
