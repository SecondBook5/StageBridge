"""Tests for the standardized table schemas."""

from __future__ import annotations

import pytest

from stagebridge.ccrt.contracts import (
    CCRTForbiddenFieldError,
    CCRTLeakageError,
    CCRTTableSchemaError,
    RECEIVERS_SCHEMA,
    SENDER_CONTEXT_SCHEMA,
    TRANSITION_EDGES_SCHEMA,
    get_table_schema,
    validate_table_fields,
)


def test_receivers_schema_validates_required_fields():
    fields = [
        "receiver_id",
        "sample_id",
        "biological_system_id",
        "receiver_state_id",
        "x_spatial",
        "y_spatial",
    ]
    # Exact required set passes.
    RECEIVERS_SCHEMA.validate_fields(fields)
    # Optional fields are allowed.
    RECEIVERS_SCHEMA.validate_fields(fields + ["patient_id", "donor_id"])


def test_receivers_schema_missing_required_fails():
    fields = ["receiver_id", "sample_id"]  # missing several required
    with pytest.raises(CCRTTableSchemaError) as excinfo:
        RECEIVERS_SCHEMA.validate_fields(fields)
    assert "missing required" in str(excinfo.value)


def test_sender_context_schema_validates_continuous_distance_fields():
    fields = [
        "receiver_id",
        "sender_id",
        "sample_id",
        "biological_system_id",
        "sender_context_type_id",
        "distance_to_receiver",
        "sender_context_mask",
    ]
    SENDER_CONTEXT_SCHEMA.validate_fields(fields)
    # continuous distance-related optionals are allowed
    SENDER_CONTEXT_SCHEMA.validate_fields(
        fields + ["distance_scaled", "uncertainty"]
    )


@pytest.mark.parametrize(
    "bad", ["ring_id", "radial_bin", "radius_bin", "neighborhood_bin"]
)
def test_sender_context_rejects_binned_distance_fields(bad):
    fields = [
        "receiver_id",
        "sender_id",
        "sample_id",
        "biological_system_id",
        "sender_context_type_id",
        "distance_to_receiver",
        "sender_context_mask",
        bad,
    ]
    with pytest.raises(CCRTForbiddenFieldError):
        SENDER_CONTEXT_SCHEMA.validate_fields(fields)
    # forbidden even with allow_extra
    with pytest.raises(CCRTForbiddenFieldError):
        SENDER_CONTEXT_SCHEMA.validate_fields(fields, allow_extra=True)


def test_transition_edges_uses_transition_edge_id_not_stage_edge_id():
    required = TRANSITION_EDGES_SCHEMA.required_fields
    assert "transition_edge_id" in required
    assert "stage_edge_id" not in required
    # A table using stage_edge_id in place of transition_edge_id is missing the
    # canonical required field.
    fields = [
        "stage_edge_id",
        "biological_system_id",
        "source_state_id",
        "target_state_id",
    ]
    with pytest.raises(CCRTTableSchemaError):
        TRANSITION_EDGES_SCHEMA.validate_fields(fields)


def test_extra_fields_fail_by_default_but_pass_with_allow_extra():
    fields = [
        "sample_id",
        "biological_system_id",
        "some_custom_qc_flag",
    ]
    with pytest.raises(CCRTTableSchemaError) as excinfo:
        validate_table_fields("samples", fields)
    assert "extra" in str(excinfo.value)
    # allow_extra lets the non-forbidden extra through
    validate_table_fields("samples", fields, allow_extra=True)


def test_extra_fields_with_allow_extra_still_reject_forbidden():
    fields = ["sample_id", "biological_system_id", "outcome_label"]
    with pytest.raises(CCRTLeakageError):
        validate_table_fields("samples", fields, allow_extra=True)


def test_get_table_schema_unknown_fails():
    with pytest.raises(CCRTTableSchemaError):
        get_table_schema("not_a_table")


def test_all_allowed_fields_normalized():
    allowed = RECEIVERS_SCHEMA.all_allowed_fields()
    assert "receiver_id" in allowed
    assert "observation_weight" in allowed  # optional
