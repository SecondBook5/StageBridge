"""Canonical field names and forbidden-field enforcement for CCRT.

This module is the single source of truth for:

* the **forbidden mechanism fields** (binned / world-token spatial mechanisms),
* the **forbidden model-input leakage fields** (target / outcome / split labels),
* the canonical **grammar field names** and **table names**, and
* the helpers that normalize a field name and assert a field set is clean.

This is the ONLY implementation module permitted to contain the forbidden term
strings (``world_token``, ``ring_id``, ...). Every other package detects them
through the helpers here, never by re-listing the literals. Field matching is
case-insensitive and whitespace-insensitive: ``"  RADIAL_BIN "`` is rejected
exactly as ``"radial_bin"`` is.
"""

from __future__ import annotations

from typing import Iterable

from .errors import CCRTForbiddenFieldError, CCRTLeakageError

__all__ = [
    # forbidden sets
    "FORBIDDEN_MECHANISM_FIELDS",
    "FORBIDDEN_MODEL_INPUT_FIELDS",
    # grammar field names
    "BIOLOGICAL_SYSTEM_ID",
    "RECEIVER_STATE_ID",
    "TRANSITION_EDGE_ID",
    "SENDER_CONTEXT_TYPE_ID",
    "SIGNAL_PROGRAM_ID",
    "RECEIVER_BEHAVIOR_ID",
    "REGULATORY_MEDIATOR_ID",
    "COUNTERFACTUAL_PERTURBATION_ID",
    # table names
    "TABLE_RECEIVERS",
    "TABLE_SENDER_CONTEXT",
    "TABLE_SEMANTIC_FEATURES",
    "TABLE_REGULATORY_FEATURES",
    "TABLE_TRANSITION_EDGES",
    "TABLE_SAMPLES",
    # helpers
    "normalize_field_name",
    "is_forbidden_mechanism_field",
    "is_forbidden_model_input_field",
    "assert_no_forbidden_mechanism_fields",
    "assert_no_model_input_leakage_fields",
    "assert_no_forbidden_fields",
]


# ---------------------------------------------------------------------------
# Forbidden field sets
# ---------------------------------------------------------------------------

#: Binned / world-token spatial mechanism field names. CCRT expresses local
#: influence only through continuous distance modulation, never through these.
FORBIDDEN_MECHANISM_FIELDS = frozenset(
    {
        "world_token",
        "ring_id",
        "radial_bin",
        "radius_bin",
        "neighborhood_bin",
    }
)

#: Target-stage / outcome / split fields that must never be model inputs.
#: They may exist only as explicitly separated training targets.
FORBIDDEN_MODEL_INPUT_FIELDS = frozenset(
    {
        "target_stage_expression",
        "future_expression",
        "outcome_label",
        "patient_response",
        "test_split_label",
    }
)


# ---------------------------------------------------------------------------
# Canonical grammar field names
# ---------------------------------------------------------------------------

BIOLOGICAL_SYSTEM_ID = "biological_system_id"
RECEIVER_STATE_ID = "receiver_state_id"
TRANSITION_EDGE_ID = "transition_edge_id"
SENDER_CONTEXT_TYPE_ID = "sender_context_type_id"
SIGNAL_PROGRAM_ID = "signal_program_id"
RECEIVER_BEHAVIOR_ID = "receiver_behavior_id"
REGULATORY_MEDIATOR_ID = "regulatory_mediator_id"
COUNTERFACTUAL_PERTURBATION_ID = "counterfactual_perturbation_id"


# ---------------------------------------------------------------------------
# Canonical table names
# ---------------------------------------------------------------------------
# NOTE: the core grammar uses ``transition_edges``. A domain-specific alias such
# as ``stage_edges`` is only ever a future adapter-level alias, never the core
# table name.

TABLE_RECEIVERS = "receivers"
TABLE_SENDER_CONTEXT = "sender_context"
TABLE_SEMANTIC_FEATURES = "semantic_features"
TABLE_REGULATORY_FEATURES = "regulatory_features"
TABLE_TRANSITION_EDGES = "transition_edges"
TABLE_SAMPLES = "samples"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def normalize_field_name(name: str) -> str:
    """Return the canonical comparison form of a field name.

    Strips surrounding whitespace and lowercases. Used so that forbidden-field
    matching is case-insensitive and whitespace-insensitive. Raises
    ``TypeError`` for non-string input, since silently coercing identifiers is
    itself a contract smell.
    """
    if not isinstance(name, str):
        raise TypeError(f"field name must be a str, got {type(name).__name__}")
    return name.strip().lower()


def is_forbidden_mechanism_field(name: str) -> bool:
    """True if ``name`` normalizes to a forbidden mechanism field."""
    return normalize_field_name(name) in FORBIDDEN_MECHANISM_FIELDS


def is_forbidden_model_input_field(name: str) -> bool:
    """True if ``name`` normalizes to a forbidden model-input leakage field."""
    return normalize_field_name(name) in FORBIDDEN_MODEL_INPUT_FIELDS


def assert_no_forbidden_mechanism_fields(fields: Iterable[str]) -> None:
    """Raise ``CCRTForbiddenFieldError`` if any field is a mechanism field."""
    offenders = sorted(
        {name for name in fields if is_forbidden_mechanism_field(name)}
    )
    if offenders:
        raise CCRTForbiddenFieldError(
            "forbidden mechanism field(s) present "
            f"(binned/world-token spatial mechanisms are prohibited): {offenders}. "
            f"Forbidden mechanism fields are: {sorted(FORBIDDEN_MECHANISM_FIELDS)}."
        )


def assert_no_model_input_leakage_fields(fields: Iterable[str]) -> None:
    """Raise ``CCRTLeakageError`` if any field is a model-input leakage field."""
    offenders = sorted(
        {name for name in fields if is_forbidden_model_input_field(name)}
    )
    if offenders:
        raise CCRTLeakageError(
            "forbidden model-input leakage field(s) present "
            f"(target/outcome/split labels may not be model inputs): {offenders}. "
            f"Forbidden leakage fields are: {sorted(FORBIDDEN_MODEL_INPUT_FIELDS)}."
        )


def assert_no_forbidden_fields(
    fields: Iterable[str], include_model_input_leakage: bool = True
) -> None:
    """Assert a field set contains no forbidden fields.

    Always rejects forbidden mechanism fields. Also rejects model-input leakage
    fields when ``include_model_input_leakage`` is True (the default).

    ``fields`` is materialized once so single-use iterables are safe.
    """
    field_list = list(fields)
    assert_no_forbidden_mechanism_fields(field_list)
    if include_model_input_leakage:
        assert_no_model_input_leakage_fields(field_list)
