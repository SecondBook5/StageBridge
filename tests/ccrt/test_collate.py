"""Tests for collate_ccrt_records (data/collate.py)."""

from __future__ import annotations

import pytest

from stagebridge.ccrt.contracts import CCRTForbiddenFieldError, CCRTValidationError
from stagebridge.ccrt.contracts.tensors import shape_of
from stagebridge.ccrt.data import CCRTTableDataset, collate_ccrt_records

from test_table_dataset import (
    make_dataset,
    make_receivers,
    make_sender_context,
)


def collate_default(ds, **overrides):
    kwargs = dict(
        receiver_feature_key="receiver_features",
        sender_feature_key="sender_features",
        semantic_feature_key="semantic_features",
        regulatory_feature_key="regulatory_features",
    )
    kwargs.update(overrides)
    return collate_ccrt_records([ds[0], ds[1]], **kwargs)


def test_collate_two_items_valid_batch():
    ds = make_dataset()
    batch = collate_default(ds)
    batch.validate()  # should not raise
    assert batch.batch_size() == 2


def test_receiver_features_shape():
    ds = make_dataset()
    batch = collate_default(ds)
    assert shape_of(batch.receiver_features) == (2, 3)


def test_sender_features_shape_padded_to_max_k():
    ds = make_dataset()
    batch = collate_default(ds)
    # r1 has 2 senders, r2 has 1 -> K=2, D_S=2
    assert shape_of(batch.sender_features) == (2, 2, 2)
    assert batch.max_sender_context() == 2


def test_sender_mask_padding():
    ds = make_dataset()
    batch = collate_default(ds)
    assert batch.sender_mask == [[1, 1], [1, 0]]


def test_distance_padded_correctly():
    ds = make_dataset()
    batch = collate_default(ds)
    assert batch.distance_to_receiver == [[1.5, 2.5], [3.5, 0.0]]


def test_uncertainty_included_when_all_real_rows_have_it():
    ds = make_dataset()
    batch = collate_default(ds)
    assert batch.uncertainty == [[0.1, 0.2], [0.3, 0.0]]


def test_uncertainty_excluded_when_any_real_row_missing_it():
    sc = make_sender_context()
    del sc[0]["uncertainty"]  # one real row lacks uncertainty
    ds = make_dataset(sender_context=sc)
    batch = collate_default(ds)
    assert batch.uncertainty is None


def test_semantic_and_regulatory_included_when_keys_provided():
    ds = make_dataset()
    batch = collate_default(ds)
    assert shape_of(batch.semantic_features) == (2, 4)
    assert shape_of(batch.regulatory_features) == (2, 2)


def test_semantic_regulatory_absent_when_keys_omitted():
    ds = make_dataset()
    batch = collate_ccrt_records(
        [ds[0], ds[1]],
        receiver_feature_key="receiver_features",
        sender_feature_key="sender_features",
    )
    assert batch.semantic_features is None
    assert batch.regulatory_features is None


def test_missing_receiver_feature_fails():
    ds = make_dataset()
    with pytest.raises(CCRTValidationError):
        collate_ccrt_records(
            [ds[0], ds[1]],
            receiver_feature_key="nonexistent_feature",
            sender_feature_key="sender_features",
        )


def test_inconsistent_receiver_feature_length_fails():
    receivers = make_receivers()
    receivers[1]["receiver_features"] = [1.0, 2.0]  # len 2 vs 3
    ds = make_dataset(receivers=receivers)
    with pytest.raises(CCRTValidationError):
        collate_default(ds)


def test_inconsistent_sender_feature_length_fails():
    sc = make_sender_context()
    sc[0]["sender_features"] = [0.1, 0.2, 0.3]  # len 3 vs 2
    ds = make_dataset(sender_context=sc)
    with pytest.raises(CCRTValidationError):
        collate_default(ds)


def test_missing_transition_edge_id_fails_when_required():
    receivers = make_receivers()
    for r in receivers:
        del r["transition_edge_id"]
    # no transition_edges table -> dataset is fine; collate should fail
    ds = CCRTTableDataset(
        receivers=receivers,
        sender_context=make_sender_context(),
        allow_extra_fields=True,
    )
    with pytest.raises(CCRTValidationError):
        collate_ccrt_records(
            [ds[0], ds[1]],
            receiver_feature_key="receiver_features",
            sender_feature_key="sender_features",
            require_transition_edge=True,
        )


def test_missing_transition_edge_id_allowed_when_not_required():
    receivers = make_receivers()
    for r in receivers:
        del r["transition_edge_id"]
    ds = CCRTTableDataset(
        receivers=receivers,
        sender_context=make_sender_context(),
        allow_extra_fields=True,
    )
    batch = collate_ccrt_records(
        [ds[0], ds[1]],
        receiver_feature_key="receiver_features",
        sender_feature_key="sender_features",
        require_transition_edge=False,
    )
    batch.validate()


def test_receiver_with_zero_senders_supported():
    # r2 keeps its sender; r1 has none.
    sc = [row for row in make_sender_context() if row["receiver_id"] == "r2"]
    ds = make_dataset(sender_context=sc)
    batch = collate_default(ds)
    # K determined by batch max (r2 has 1) -> K=1
    assert batch.max_sender_context() == 1
    assert batch.sender_mask == [[0], [1]]


def test_all_zero_sender_batch_returns_k1_with_mask_zeros():
    # Build items directly where every receiver has zero senders. (The dataset
    # itself requires a non-empty sender_context table, so the all-empty case is
    # exercised at the collate boundary, which operates on items.)
    item0 = {
        "receiver": {
            "receiver_id": "r1",
            "biological_system_id": "sys_a",
            "transition_edge_id": "edge_1",
            "receiver_features": [1.0, 2.0, 3.0],
        },
        "sender_context": (),
    }
    item1 = {
        "receiver": {
            "receiver_id": "r2",
            "biological_system_id": "sys_a",
            "transition_edge_id": "edge_1",
            "receiver_features": [4.0, 5.0, 6.0],
        },
        "sender_context": (),
    }
    batch = collate_ccrt_records(
        [item0, item1],
        receiver_feature_key="receiver_features",
        sender_feature_key="sender_features",
    )
    assert batch.max_sender_context() == 1
    assert batch.sender_mask == [[0], [0]]
    # uncertainty excluded when there are no real senders
    assert batch.uncertainty is None
    batch.validate()


def test_forbidden_field_in_item_fails():
    ds = make_dataset()
    item0 = dict(ds[0])
    item0["ring_id"] = 1  # forbidden mechanism key at item level
    with pytest.raises(CCRTForbiddenFieldError):
        collate_ccrt_records(
            [item0, ds[1]],
            receiver_feature_key="receiver_features",
            sender_feature_key="sender_features",
        )
