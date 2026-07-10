"""Tests for CCRTTrainingBatch and build_training_batch tensorization."""

from __future__ import annotations

import pytest
import torch

from stagebridge.ccrt.contracts import CCRTShapeError, CCRTValidationError
from stagebridge.ccrt.data import CCRTBatch, CCRTIndexRegistry
from stagebridge.ccrt.grammar import (
    SEMANTIC_DRIFT,
    BiologicalSystemSpec,
    ReceiverBehavior,
    ReceiverState,
    SenderContextType,
    TransitionEdge,
)
from stagebridge.ccrt.training import CCRTTrainingBatch, build_training_batch


def make_spec(sid, senders):
    return BiologicalSystemSpec(
        biological_system_id=sid,
        receiver_states=(ReceiverState("s0"), ReceiverState("s1")),
        transition_edges=(TransitionEdge("e0", "s0", "s1"),),
        sender_context_types=tuple(SenderContextType(s) for s in senders),
        receiver_behaviors=(ReceiverBehavior(SEMANTIC_DRIFT),),
    )


def make_source_batch(system="sysA"):
    # B=2, K=2, D_R=3, D_S=4, D_Z=2. r1 has 2 senders, r2 has 1 (+ padding).
    return CCRTBatch(
        receiver_features=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        sender_features=[[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
                         [[0.9, 1.0, 1.1, 1.2], [0.0, 0.0, 0.0, 0.0]]],
        sender_mask=[[1, 1], [1, 0]],
        distance_to_receiver=[[1.0, 2.0], [3.0, 0.0]],
        biological_system_id=[system, system],
        transition_edge_id=["e0", "e0"],
        semantic_features=[[0.1, 0.2], [0.3, 0.4]],
        sender_context_type_ids=[["c0", "c1"], ["c0", None]],
    )


def make_registry():
    return CCRTIndexRegistry.from_system_specs([make_spec("sysA", ["c0", "c1"])])


# -- CCRTTrainingBatch direct --

def test_training_batch_validates_and_sizes():
    tb = CCRTTrainingBatch(
        receiver_features=torch.randn(2, 3),
        sender_features=torch.randn(2, 2, 4),
        sender_mask=torch.tensor([[1, 1], [1, 0]], dtype=torch.bool),
        distance_to_receiver=torch.rand(2, 2),
        sender_context_type_ids=torch.tensor([[0, 1], [0, 0]], dtype=torch.int64),
        transition_edge_index=torch.tensor([0, 0], dtype=torch.int64),
        source_semantic_features=torch.randn(2, 2),
        target_semantic_features=torch.randn(3, 2),
    )
    tb.validate()
    assert tb.batch_size() == 2
    assert tb.target_size() == 3
    assert tb.max_sender_context() == 2
    assert tb.semantic_dim() == 2


def test_training_batch_semantic_dim_mismatch_fails():
    tb = CCRTTrainingBatch(
        receiver_features=torch.randn(2, 3),
        sender_features=torch.randn(2, 2, 4),
        sender_mask=torch.tensor([[1, 1], [1, 0]], dtype=torch.bool),
        distance_to_receiver=torch.rand(2, 2),
        sender_context_type_ids=torch.tensor([[0, 1], [0, 0]], dtype=torch.int64),
        transition_edge_index=torch.tensor([0, 0], dtype=torch.int64),
        source_semantic_features=torch.randn(2, 2),
        target_semantic_features=torch.randn(3, 3),  # mismatch
    )
    with pytest.raises(CCRTShapeError):
        tb.validate()


def test_training_batch_to_preserves_int_and_bool():
    tb = CCRTTrainingBatch(
        receiver_features=torch.randn(2, 3),
        sender_features=torch.randn(2, 2, 4),
        sender_mask=torch.tensor([[1, 1], [1, 0]], dtype=torch.bool),
        distance_to_receiver=torch.rand(2, 2),
        sender_context_type_ids=torch.tensor([[0, 1], [0, 0]], dtype=torch.int64),
        transition_edge_index=torch.tensor([0, 0], dtype=torch.int64),
        source_semantic_features=torch.randn(2, 2),
        target_semantic_features=torch.randn(3, 2),
    )
    moved = tb.to("cpu", dtype=torch.float64)
    assert moved.receiver_features.dtype == torch.float64
    assert moved.sender_context_type_ids.dtype == torch.int64
    assert moved.sender_mask.dtype == torch.bool
    # original not mutated
    assert tb.receiver_features.dtype == torch.float32


def test_growth_mask_without_targets_fails():
    tb = CCRTTrainingBatch(
        receiver_features=torch.randn(2, 3),
        sender_features=torch.randn(2, 2, 4),
        sender_mask=torch.ones(2, 2, dtype=torch.bool),
        distance_to_receiver=torch.rand(2, 2),
        sender_context_type_ids=torch.zeros(2, 2, dtype=torch.int64),
        transition_edge_index=torch.zeros(2, dtype=torch.int64),
        source_semantic_features=torch.randn(2, 2),
        target_semantic_features=torch.randn(3, 2),
        growth_mask=torch.ones(2, 1, dtype=torch.bool),
    )
    with pytest.raises(CCRTValidationError):
        tb.validate()


# -- build_training_batch --

def test_valid_one_system_tensorization():
    reg = make_registry()
    src = make_source_batch()
    tb = build_training_batch(
        source_batch=src,
        target_semantic_features=torch.randn(3, 2),
        index_registry=reg,
    )
    tb.validate()
    # c0->0, c1->1; padded position -> 0 (masked)
    assert tb.sender_context_type_ids.tolist() == [[0, 1], [0, 0]]
    assert tb.transition_edge_index.tolist() == [0, 0]


def test_valid_two_system_tensorization():
    reg = CCRTIndexRegistry.from_system_specs(
        [make_spec("sysA", ["c0", "c1"]), make_spec("sysB", ["c0"])]
    )
    src = make_source_batch(system="sysB")
    # sysB only has c0; adjust the source so real senders are all c0
    src = CCRTBatch(
        receiver_features=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        sender_features=[[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
                         [[0.9, 1.0, 1.1, 1.2], [0.0, 0.0, 0.0, 0.0]]],
        sender_mask=[[1, 1], [1, 0]],
        distance_to_receiver=[[1.0, 2.0], [3.0, 0.0]],
        biological_system_id=["sysB", "sysB"],
        transition_edge_id=["e0", "e0"],
        semantic_features=[[0.1, 0.2], [0.3, 0.4]],
        sender_context_type_ids=[["c0", "c0"], ["c0", None]],
    )
    tb = build_training_batch(
        source_batch=src, target_semantic_features=torch.randn(3, 2),
        index_registry=reg,
    )
    # sysB::c0 has a distinct global index from sysA::c0
    sysB_c0 = reg.encode_sender_context_type("sysB", "c0")
    assert tb.sender_context_type_ids[0, 0].item() == sysB_c0


def test_masked_padding_needs_no_biological_sender_id():
    reg = make_registry()
    src = make_source_batch()  # r2 second position is None + masked
    tb = build_training_batch(
        source_batch=src, target_semantic_features=torch.randn(3, 2),
        index_registry=reg,
    )
    assert tb.sender_context_type_ids[1, 1].item() == 0
    assert bool(tb.sender_mask[1, 1]) is False


def test_real_unknown_sender_id_fails():
    reg = make_registry()
    src = make_source_batch()
    src = CCRTBatch(
        receiver_features=src.receiver_features,
        sender_features=src.sender_features,
        sender_mask=src.sender_mask,
        distance_to_receiver=src.distance_to_receiver,
        biological_system_id=src.biological_system_id,
        transition_edge_id=src.transition_edge_id,
        semantic_features=src.semantic_features,
        sender_context_type_ids=[["c0", "UNKNOWN"], ["c0", None]],
    )
    with pytest.raises(CCRTValidationError):
        build_training_batch(
            source_batch=src, target_semantic_features=torch.randn(3, 2),
            index_registry=reg,
        )


def test_unknown_transition_edge_fails():
    reg = make_registry()
    src = CCRTBatch(
        receiver_features=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        sender_features=[[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
                         [[0.9, 1.0, 1.1, 1.2], [0.0, 0.0, 0.0, 0.0]]],
        sender_mask=[[1, 1], [1, 0]],
        distance_to_receiver=[[1.0, 2.0], [3.0, 0.0]],
        biological_system_id=["sysA", "sysA"],
        transition_edge_id=["e_missing", "e_missing"],
        semantic_features=[[0.1, 0.2], [0.3, 0.4]],
        sender_context_type_ids=[["c0", "c1"], ["c0", None]],
    )
    with pytest.raises(CCRTValidationError):
        build_training_batch(
            source_batch=src, target_semantic_features=torch.randn(3, 2),
            index_registry=reg,
        )


def test_semantic_features_required():
    reg = make_registry()
    src = CCRTBatch(
        receiver_features=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        sender_features=[[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
                         [[0.9, 1.0, 1.1, 1.2], [0.0, 0.0, 0.0, 0.0]]],
        sender_mask=[[1, 1], [1, 0]],
        distance_to_receiver=[[1.0, 2.0], [3.0, 0.0]],
        biological_system_id=["sysA", "sysA"],
        transition_edge_id=["e0", "e0"],
        sender_context_type_ids=[["c0", "c1"], ["c0", None]],
    )
    with pytest.raises(CCRTValidationError):
        build_training_batch(
            source_batch=src, target_semantic_features=torch.randn(3, 2),
            index_registry=reg,
        )


def test_source_target_dim_mismatch_fails():
    reg = make_registry()
    src = make_source_batch()  # D_Z = 2
    with pytest.raises(CCRTShapeError):
        build_training_batch(
            source_batch=src, target_semantic_features=torch.randn(3, 5),
            index_registry=reg,
        )


def test_type_ids_required():
    reg = make_registry()
    src = CCRTBatch(
        receiver_features=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        sender_features=[[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
                         [[0.9, 1.0, 1.1, 1.2], [0.0, 0.0, 0.0, 0.0]]],
        sender_mask=[[1, 1], [1, 0]],
        distance_to_receiver=[[1.0, 2.0], [3.0, 0.0]],
        biological_system_id=["sysA", "sysA"],
        transition_edge_id=["e0", "e0"],
        semantic_features=[[0.1, 0.2], [0.3, 0.4]],
        # no sender_context_type_ids
    )
    with pytest.raises(CCRTValidationError):
        build_training_batch(
            source_batch=src, target_semantic_features=torch.randn(3, 2),
            index_registry=reg,
        )
