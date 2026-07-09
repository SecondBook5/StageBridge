"""Standardized CCRT table schemas and field validation.

Every disease/system adapter emits the *same* standardized tables with the
*same* canonical columns; system-specific meaning lives only in the grammar IDs
those columns carry. This module defines:

* ``TableSchema`` — a frozen schema (required / optional / forbidden fields)
  with strict field validation, and
* the six core table schemas plus lookup/validation helpers.

Validation is strict and boring:

* required fields must be present;
* forbidden mechanism fields (rings/bins/world-token) ALWAYS fail;
* forbidden model-input leakage fields ALWAYS fail (even with ``allow_extra``);
* extra fields fail unless ``allow_extra=True``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from .errors import CCRTTableSchemaError
from .naming import (
    TABLE_RECEIVERS,
    TABLE_REGULATORY_FEATURES,
    TABLE_SAMPLES,
    TABLE_SEMANTIC_FEATURES,
    TABLE_SENDER_CONTEXT,
    TABLE_TRANSITION_EDGES,
    assert_no_forbidden_mechanism_fields,
    assert_no_model_input_leakage_fields,
    normalize_field_name,
)

__all__ = [
    "TableSchema",
    "RECEIVERS_SCHEMA",
    "SENDER_CONTEXT_SCHEMA",
    "SEMANTIC_FEATURES_SCHEMA",
    "REGULATORY_FEATURES_SCHEMA",
    "TRANSITION_EDGES_SCHEMA",
    "SAMPLES_SCHEMA",
    "TABLE_SCHEMAS",
    "get_table_schema",
    "validate_table_fields",
]


@dataclass(frozen=True)
class TableSchema:
    """A standardized-table schema.

    Attributes:
        name: canonical table name.
        required_fields: fields that must be present.
        optional_fields: fields that are allowed but not required.
        forbidden_fields: extra table-specific forbidden fields (on top of the
            global forbidden mechanism / leakage sets, which always apply).
    """

    name: str
    required_fields: tuple[str, ...]
    optional_fields: tuple[str, ...] = ()
    forbidden_fields: tuple[str, ...] = ()

    def all_allowed_fields(self) -> frozenset[str]:
        """The set of allowed field names (required ∪ optional), normalized."""
        return frozenset(
            normalize_field_name(f)
            for f in (*self.required_fields, *self.optional_fields)
        )

    def validate_fields(
        self, fields: Iterable[str], *, allow_extra: bool = False
    ) -> None:
        """Validate a set of field names against this schema.

        Order of checks (all boring, all strict):

        1. forbidden mechanism fields -> ``CCRTForbiddenFieldError``
        2. forbidden model-input leakage fields -> ``CCRTLeakageError``
        3. table-specific forbidden fields -> ``CCRTTableSchemaError``
        4. missing required fields -> ``CCRTTableSchemaError``
        5. extra fields (unless ``allow_extra``) -> ``CCRTTableSchemaError``
        """
        present = [normalize_field_name(f) for f in fields]
        present_set = set(present)

        # (1) + (2): global forbidden fields always fail, even with allow_extra.
        assert_no_forbidden_mechanism_fields(present)
        assert_no_model_input_leakage_fields(present)

        # (3): table-specific forbidden fields.
        table_forbidden = {normalize_field_name(f) for f in self.forbidden_fields}
        hit_forbidden = sorted(present_set & table_forbidden)
        if hit_forbidden:
            raise CCRTTableSchemaError(
                f"table '{self.name}': forbidden field(s) present: {hit_forbidden}"
            )

        # (4): required fields present.
        required = {normalize_field_name(f) for f in self.required_fields}
        missing = sorted(required - present_set)
        if missing:
            raise CCRTTableSchemaError(
                f"table '{self.name}': missing required field(s): {missing} "
                f"(required: {sorted(required)})"
            )

        # (5): extra fields.
        if not allow_extra:
            allowed = self.all_allowed_fields()
            extra = sorted(present_set - allowed)
            if extra:
                raise CCRTTableSchemaError(
                    f"table '{self.name}': unexpected extra field(s): {extra}. "
                    f"Pass allow_extra=True to permit non-forbidden extras. "
                    f"Allowed: {sorted(allowed)}"
                )


# ---------------------------------------------------------------------------
# The six standardized core table schemas
# ---------------------------------------------------------------------------

RECEIVERS_SCHEMA = TableSchema(
    name=TABLE_RECEIVERS,
    required_fields=(
        "receiver_id",
        "sample_id",
        "biological_system_id",
        "receiver_state_id",
        "x_spatial",
        "y_spatial",
    ),
    optional_fields=(
        "patient_id",
        "donor_id",
        "section_id",
        "receiver_type",
        "transition_edge_id",
        "observation_weight",
    ),
)

SENDER_CONTEXT_SCHEMA = TableSchema(
    name=TABLE_SENDER_CONTEXT,
    required_fields=(
        "receiver_id",
        "sender_id",
        "sample_id",
        "biological_system_id",
        "sender_context_type_id",
        "distance_to_receiver",
        "sender_context_mask",
    ),
    optional_fields=(
        "patient_id",
        "donor_id",
        "section_id",
        "x_receiver",
        "y_receiver",
        "x_sender",
        "y_sender",
        "dx",
        "dy",
        "distance_scaled",
        "abundance",
        "uncertainty",
        "sender_source",
        "signal_program_id",
    ),
)

SEMANTIC_FEATURES_SCHEMA = TableSchema(
    name=TABLE_SEMANTIC_FEATURES,
    required_fields=(
        "receiver_id",
        "sample_id",
        "biological_system_id",
        "feature_space_id",
    ),
    optional_fields=(
        "patient_id",
        "donor_id",
        "section_id",
        "feature_names",
        "feature_values",
        "representation_version",
    ),
)

REGULATORY_FEATURES_SCHEMA = TableSchema(
    name=TABLE_REGULATORY_FEATURES,
    required_fields=(
        "receiver_id",
        "sample_id",
        "biological_system_id",
        "regulatory_feature_space_id",
    ),
    optional_fields=(
        "patient_id",
        "donor_id",
        "section_id",
        "regulatory_mediator_id",
        "feature_names",
        "feature_values",
        "representation_version",
    ),
)

TRANSITION_EDGES_SCHEMA = TableSchema(
    name=TABLE_TRANSITION_EDGES,
    required_fields=(
        "transition_edge_id",
        "biological_system_id",
        "source_state_id",
        "target_state_id",
    ),
    optional_fields=(
        "source_order",
        "target_order",
        "edge_order",
        "edge_type",
    ),
)

SAMPLES_SCHEMA = TableSchema(
    name=TABLE_SAMPLES,
    required_fields=(
        "sample_id",
        "biological_system_id",
    ),
    optional_fields=(
        "patient_id",
        "donor_id",
        "section_id",
        "tissue_region",
        "platform",
        "source_dataset",
        "split_group_id",
    ),
)


#: Registry of the standardized core table schemas, keyed by canonical name.
TABLE_SCHEMAS: Mapping[str, TableSchema] = {
    schema.name: schema
    for schema in (
        RECEIVERS_SCHEMA,
        SENDER_CONTEXT_SCHEMA,
        SEMANTIC_FEATURES_SCHEMA,
        REGULATORY_FEATURES_SCHEMA,
        TRANSITION_EDGES_SCHEMA,
        SAMPLES_SCHEMA,
    )
}


def get_table_schema(name: str) -> TableSchema:
    """Return the schema for a canonical table name, or raise."""
    key = normalize_field_name(name)
    schema = TABLE_SCHEMAS.get(key)
    if schema is None:
        raise CCRTTableSchemaError(
            f"unknown table '{name}'; known tables: {sorted(TABLE_SCHEMAS)}"
        )
    return schema


def validate_table_fields(
    table_name: str, fields: Iterable[str], *, allow_extra: bool = False
) -> None:
    """Validate ``fields`` against the named standardized table schema."""
    get_table_schema(table_name).validate_fields(fields, allow_extra=allow_extra)
