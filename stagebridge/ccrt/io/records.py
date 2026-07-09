"""Record-level validation of standardized table-like data.

This milestone works with in-memory Python records — ``Sequence[Mapping[str,
Any]]`` — not files, DataFrames, or AnnData. It validates a list of records
against the standardized table schemas defined in ``contracts/tables.py``:
required fields present, no unexpected extras (unless allowed), and never any
forbidden mechanism or model-input-leakage field.

No CSV / parquet / pandas reading is implemented here; that belongs to a later
milestone.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..contracts.errors import CCRTTableSchemaError
from ..contracts.tables import get_table_schema

__all__ = [
    "infer_record_fields",
    "require_non_empty_records",
    "validate_records",
]


def infer_record_fields(records: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    """Return the union of keys across all records, in first-seen order.

    Every record must be a mapping; a non-mapping record is a schema error
    (records are the standardized table rows, not arbitrary objects).
    """
    seen: dict[str, None] = {}
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise CCRTTableSchemaError(
                f"record at index {index} must be a Mapping, got "
                f"{type(record).__name__}"
            )
        for key in record.keys():
            if key not in seen:
                seen[key] = None
    return tuple(seen.keys())


def require_non_empty_records(
    table_name: str, records: Sequence[Mapping[str, Any]]
) -> None:
    """Raise ``CCRTTableSchemaError`` if ``records`` is empty or not a sequence."""
    if isinstance(records, (str, bytes, Mapping)):
        raise CCRTTableSchemaError(
            f"table '{table_name}': records must be a sequence of mappings, "
            f"got {type(records).__name__}"
        )
    try:
        length = len(records)
    except TypeError as exc:
        raise CCRTTableSchemaError(
            f"table '{table_name}': records must be a sized sequence"
        ) from exc
    if length == 0:
        raise CCRTTableSchemaError(
            f"table '{table_name}': records must be non-empty"
        )


def validate_records(
    table_name: str,
    records: Sequence[Mapping[str, Any]],
    *,
    allow_extra: bool = False,
) -> None:
    """Validate table-like records against the named standardized schema.

    * records must be non-empty and every record a mapping;
    * inferred fields (union of keys) are validated against the schema;
    * missing required fields fail; extras fail unless ``allow_extra``;
    * forbidden mechanism / model-input-leakage fields always fail (even with
      ``allow_extra``).
    """
    require_non_empty_records(table_name, records)
    fields = infer_record_fields(records)
    # Delegates the whole forbidden/required/extra policy to the table schema,
    # which already enforces the global forbidden-field rules even under
    # allow_extra. get_table_schema raises for an unknown table name.
    schema = get_table_schema(table_name)
    schema.validate_fields(fields, allow_extra=allow_extra)
