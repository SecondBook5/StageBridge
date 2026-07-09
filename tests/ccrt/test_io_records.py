"""Tests for record-level table validation (io/records.py)."""

from __future__ import annotations

import pytest

from stagebridge.ccrt.contracts import (
    CCRTForbiddenFieldError,
    CCRTLeakageError,
    CCRTTableSchemaError,
)
from stagebridge.ccrt.io import (
    infer_record_fields,
    require_non_empty_records,
    validate_records,
)


def valid_receiver_records():
    return [
        {
            "receiver_id": "r1",
            "sample_id": "s1",
            "biological_system_id": "sys_a",
            "receiver_state_id": "state_0",
            "x_spatial": 1.0,
            "y_spatial": 2.0,
        },
        {
            "receiver_id": "r2",
            "sample_id": "s1",
            "biological_system_id": "sys_a",
            "receiver_state_id": "state_1",
            "x_spatial": 3.0,
            "y_spatial": 4.0,
        },
    ]


def valid_sender_records():
    return [
        {
            "receiver_id": "r1",
            "sender_id": "c1",
            "sample_id": "s1",
            "biological_system_id": "sys_a",
            "sender_context_type_id": "ctx_0",
            "distance_to_receiver": 5.0,
            "sender_context_mask": 1,
        }
    ]


def test_validate_records_accepts_valid_receivers():
    validate_records("receivers", valid_receiver_records())


def test_validate_records_accepts_valid_sender_context():
    validate_records("sender_context", valid_sender_records())


def test_infer_record_fields_is_union_in_order():
    records = [{"a": 1, "b": 2}, {"b": 3, "c": 4}]
    assert infer_record_fields(records) == ("a", "b", "c")


def test_empty_records_fail():
    with pytest.raises(CCRTTableSchemaError):
        validate_records("receivers", [])
    with pytest.raises(CCRTTableSchemaError):
        require_non_empty_records("receivers", [])


def test_non_mapping_record_fails():
    with pytest.raises(CCRTTableSchemaError):
        validate_records("receivers", [valid_receiver_records()[0], ["not", "a", "map"]])


def test_missing_required_fields_fail():
    bad = [{"receiver_id": "r1", "sample_id": "s1"}]
    with pytest.raises(CCRTTableSchemaError):
        validate_records("receivers", bad)


def test_extra_fields_fail_by_default():
    records = valid_receiver_records()
    records[0]["some_extra_col"] = 1.0
    with pytest.raises(CCRTTableSchemaError):
        validate_records("receivers", records)


def test_extra_fields_pass_with_allow_extra():
    records = valid_receiver_records()
    records[0]["receiver_features"] = [0.1, 0.2, 0.3]
    validate_records("receivers", records, allow_extra=True)


def test_forbidden_mechanism_field_fails_even_with_allow_extra():
    records = valid_receiver_records()
    records[0]["ring_id"] = 3
    with pytest.raises(CCRTForbiddenFieldError):
        validate_records("receivers", records, allow_extra=True)


def test_model_input_leakage_field_fails():
    records = valid_receiver_records()
    records[0]["future_expression"] = [1.0]
    with pytest.raises(CCRTLeakageError):
        validate_records("receivers", records, allow_extra=True)


def test_unknown_table_fails():
    with pytest.raises(CCRTTableSchemaError):
        validate_records("not_a_table", valid_receiver_records())
