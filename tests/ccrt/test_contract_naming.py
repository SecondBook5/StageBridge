"""Tests for forbidden-field detection and canonical naming."""

from __future__ import annotations

import pytest

from stagebridge.ccrt.contracts import (
    CCRTForbiddenFieldError,
    CCRTLeakageError,
    assert_no_forbidden_fields,
    assert_no_forbidden_mechanism_fields,
    assert_no_model_input_leakage_fields,
    is_forbidden_mechanism_field,
    is_forbidden_model_input_field,
    normalize_field_name,
)

FORBIDDEN_MECHANISM = [
    "world_token",
    "ring_id",
    "radial_bin",
    "radius_bin",
    "neighborhood_bin",
]

CASE_VARIANTS = [
    "World_Token",
    "RADIAL_BIN",
    "  ring_id ",
    "Neighborhood_Bin",
]

FORBIDDEN_LEAKAGE = [
    "future_expression",
    "outcome_label",
    "patient_response",
    "test_split_label",
    "target_stage_expression",
]

SAFE_GRAMMAR_FIELDS = [
    "biological_system_id",
    "receiver_state_id",
    "transition_edge_id",
    "sender_context_type_id",
]


@pytest.mark.parametrize("name", FORBIDDEN_MECHANISM)
def test_forbidden_mechanism_fields_rejected(name):
    assert is_forbidden_mechanism_field(name)
    with pytest.raises(CCRTForbiddenFieldError):
        assert_no_forbidden_mechanism_fields([name])
    with pytest.raises(CCRTForbiddenFieldError):
        assert_no_forbidden_fields([name])


@pytest.mark.parametrize("name", CASE_VARIANTS)
def test_case_and_whitespace_variants_rejected(name):
    assert is_forbidden_mechanism_field(name)
    with pytest.raises(CCRTForbiddenFieldError):
        assert_no_forbidden_mechanism_fields([name])


@pytest.mark.parametrize("name", FORBIDDEN_LEAKAGE)
def test_model_input_leakage_fields_rejected(name):
    assert is_forbidden_model_input_field(name)
    with pytest.raises(CCRTLeakageError):
        assert_no_model_input_leakage_fields([name])
    with pytest.raises(CCRTLeakageError):
        assert_no_forbidden_fields([name])


@pytest.mark.parametrize("name", [n.upper() for n in FORBIDDEN_LEAKAGE])
def test_leakage_fields_case_insensitive(name):
    assert is_forbidden_model_input_field(name)


@pytest.mark.parametrize("name", SAFE_GRAMMAR_FIELDS)
def test_safe_grammar_fields_pass(name):
    assert not is_forbidden_mechanism_field(name)
    assert not is_forbidden_model_input_field(name)
    # Neither guard should raise for a clean field set.
    assert_no_forbidden_mechanism_fields([name])
    assert_no_model_input_leakage_fields([name])
    assert_no_forbidden_fields([name])


def test_assert_no_forbidden_fields_can_skip_leakage_check():
    # With leakage checking off, a leakage field passes but a mechanism field
    # still fails.
    assert_no_forbidden_fields(
        ["future_expression"], include_model_input_leakage=False
    )
    with pytest.raises(CCRTForbiddenFieldError):
        assert_no_forbidden_fields(
            ["ring_id"], include_model_input_leakage=False
        )


def test_normalize_field_name():
    assert normalize_field_name("  Receiver_State_ID ") == "receiver_state_id"
    with pytest.raises(TypeError):
        normalize_field_name(123)  # type: ignore[arg-type]


def test_mixed_clean_and_forbidden_reports_offender():
    with pytest.raises(CCRTForbiddenFieldError) as excinfo:
        assert_no_forbidden_mechanism_fields(
            ["receiver_id", "ring_id", "sample_id"]
        )
    assert "ring_id" in str(excinfo.value)
