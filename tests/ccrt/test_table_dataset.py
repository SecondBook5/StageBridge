"""Tests for CCRTTableDataset (data/dataset.py).

Records here are tiny, test-only, and carry feature-vector columns
(`receiver_features`, `sender_features`, ...) that are not part of the
standardized schema, so the dataset is built with allow_extra_fields=True.
"""

from __future__ import annotations

import pytest

from stagebridge.ccrt.contracts import (
    CCRTForbiddenFieldError,
    CCRTLeakageError,
    CCRTValidationError,
)
from stagebridge.ccrt.contracts.errors import CCRTSplitError
from stagebridge.ccrt.data import CCRTTableDataset


# ---------------------------------------------------------------------------
# Shared tiny fixtures (importable by test_collate.py)
# ---------------------------------------------------------------------------


def make_receivers():
    return [
        {
            "receiver_id": "r1",
            "sample_id": "sample_1",
            "biological_system_id": "sys_a",
            "receiver_state_id": "state_0",
            "x_spatial": 0.0,
            "y_spatial": 0.0,
            "transition_edge_id": "edge_1",
            "receiver_features": [1.0, 2.0, 3.0],
        },
        {
            "receiver_id": "r2",
            "sample_id": "sample_1",
            "biological_system_id": "sys_a",
            "receiver_state_id": "state_1",
            "x_spatial": 5.0,
            "y_spatial": 5.0,
            "transition_edge_id": "edge_1",
            "receiver_features": [4.0, 5.0, 6.0],
        },
    ]


def make_sender_context():
    return [
        {
            "receiver_id": "r1",
            "sender_id": "c1",
            "sample_id": "sample_1",
            "biological_system_id": "sys_a",
            "sender_context_type_id": "ctx_0",
            "distance_to_receiver": 1.5,
            "sender_context_mask": 1,
            "uncertainty": 0.1,
            "sender_features": [0.1, 0.2],
        },
        {
            "receiver_id": "r1",
            "sender_id": "c2",
            "sample_id": "sample_1",
            "biological_system_id": "sys_a",
            "sender_context_type_id": "ctx_1",
            "distance_to_receiver": 2.5,
            "sender_context_mask": 1,
            "uncertainty": 0.2,
            "sender_features": [0.3, 0.4],
        },
        {
            "receiver_id": "r2",
            "sender_id": "c3",
            "sample_id": "sample_1",
            "biological_system_id": "sys_a",
            "sender_context_type_id": "ctx_0",
            "distance_to_receiver": 3.5,
            "sender_context_mask": 1,
            "uncertainty": 0.3,
            "sender_features": [0.5, 0.6],
        },
    ]


def make_semantic_features():
    return [
        {
            "receiver_id": "r1",
            "sample_id": "sample_1",
            "biological_system_id": "sys_a",
            "feature_space_id": "z_sem_default",
            "semantic_features": [0.7, 0.8, 0.9, 1.0],
        },
        {
            "receiver_id": "r2",
            "sample_id": "sample_1",
            "biological_system_id": "sys_a",
            "feature_space_id": "z_sem_default",
            "semantic_features": [1.1, 1.2, 1.3, 1.4],
        },
    ]


def make_regulatory_features():
    return [
        {
            "receiver_id": "r1",
            "sample_id": "sample_1",
            "biological_system_id": "sys_a",
            "regulatory_feature_space_id": "reg_default",
            "regulatory_features": [0.01, 0.02],
        },
        {
            "receiver_id": "r2",
            "sample_id": "sample_1",
            "biological_system_id": "sys_a",
            "regulatory_feature_space_id": "reg_default",
            "regulatory_features": [0.03, 0.04],
        },
    ]


def make_transition_edges():
    return [
        {
            "transition_edge_id": "edge_1",
            "biological_system_id": "sys_a",
            "source_state_id": "state_0",
            "target_state_id": "state_1",
        }
    ]


def make_samples():
    return [
        {
            "sample_id": "sample_1",
            "biological_system_id": "sys_a",
            "patient_id": "patient_1",
        }
    ]


def make_dataset(**overrides):
    kwargs = dict(
        receivers=make_receivers(),
        sender_context=make_sender_context(),
        semantic_features=make_semantic_features(),
        regulatory_features=make_regulatory_features(),
        transition_edges=make_transition_edges(),
        samples=make_samples(),
        allow_extra_fields=True,
    )
    kwargs.update(overrides)
    return CCRTTableDataset(**kwargs)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_dataset_validates_and_len():
    ds = make_dataset()
    assert len(ds) == 2


def test_getitem_0_returns_r1_with_two_senders():
    ds = make_dataset()
    item = ds[0]
    assert item["receiver"]["receiver_id"] == "r1"
    assert len(item["sender_context"]) == 2
    assert item["semantic_features"]["receiver_id"] == "r1"
    assert item["regulatory_features"]["receiver_id"] == "r1"
    assert item["transition_edge"]["transition_edge_id"] == "edge_1"
    assert item["sample"]["sample_id"] == "sample_1"


def test_getitem_1_returns_r2_with_one_sender():
    ds = make_dataset()
    item = ds[1]
    assert item["receiver"]["receiver_id"] == "r2"
    assert len(item["sender_context"]) == 1


def test_receiver_order_is_preserved():
    ds = make_dataset()
    assert ds.receiver_ids() == ("r1", "r2")


def test_get_by_receiver_id():
    ds = make_dataset()
    item = ds.get_by_receiver_id("r2")
    assert item["receiver"]["receiver_id"] == "r2"
    with pytest.raises(CCRTValidationError):
        ds.get_by_receiver_id("nope")


def test_sender_order_within_receiver_preserved():
    ds = make_dataset()
    senders = ds[0]["sender_context"]
    assert [s["sender_id"] for s in senders] == ["c1", "c2"]


def test_duplicate_receiver_id_fails():
    receivers = make_receivers()
    receivers[1]["receiver_id"] = "r1"
    with pytest.raises(CCRTValidationError):
        make_dataset(receivers=receivers)


def test_sender_context_referencing_missing_receiver_fails():
    sc = make_sender_context()
    sc[0]["receiver_id"] = "ghost"
    with pytest.raises(CCRTValidationError):
        make_dataset(sender_context=sc)


def test_semantic_features_referencing_missing_receiver_fails():
    sem = make_semantic_features()
    sem[0]["receiver_id"] = "ghost"
    with pytest.raises(CCRTValidationError):
        make_dataset(semantic_features=sem)


def test_receiver_transition_edge_absent_from_edges_fails():
    receivers = make_receivers()
    receivers[0]["transition_edge_id"] = "edge_missing"
    with pytest.raises(CCRTValidationError):
        make_dataset(receivers=receivers)


def test_receiver_sample_absent_from_samples_fails():
    receivers = make_receivers()
    receivers[0]["sample_id"] = "sample_missing"
    with pytest.raises(CCRTValidationError):
        make_dataset(receivers=receivers)


def test_forbidden_field_in_records_fails():
    receivers = make_receivers()
    receivers[0]["ring_id"] = 2
    with pytest.raises(CCRTForbiddenFieldError):
        make_dataset(receivers=receivers)


def test_leakage_field_in_records_fails():
    receivers = make_receivers()
    receivers[0]["outcome_label"] = "responder"
    with pytest.raises(CCRTLeakageError):
        make_dataset(receivers=receivers)


def test_split_manifest_validated_when_provided():
    good = {
        "split_strategy": "patient_aware",
        "group_key": "patient_id",
        "train": ["patient_1"],
        "validation": ["patient_2"],
        "test": ["patient_3"],
    }
    ds = make_dataset(split_manifest=good)
    assert ds.split_manifest is not None

    bad = dict(good, split_strategy="random_spot")
    with pytest.raises(CCRTSplitError):
        make_dataset(split_manifest=bad)


def test_dataset_does_not_mutate_inputs():
    receivers = make_receivers()
    ds = make_dataset(receivers=receivers)
    # mutating a returned item's receiver must not be possible / must not leak
    item = ds[0]
    with pytest.raises(TypeError):
        item["receiver"]["receiver_id"] = "mutated"  # read-only mapping
    assert receivers[0]["receiver_id"] == "r1"


def test_missing_optional_tables_ok():
    ds = CCRTTableDataset(
        receivers=make_receivers(),
        sender_context=make_sender_context(),
        allow_extra_fields=True,
    )
    item = ds[0]
    assert item["semantic_features"] is None
    assert item["regulatory_features"] is None
    # transition_edge is None because no transition_edges table was supplied
    assert item["transition_edge"] is None
    assert item["sample"] is None
