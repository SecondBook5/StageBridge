"""Composed field-name validation helpers.

Boring, strict, reusable checks that sit on top of ``naming.py``. Grammar,
data, and table code call these so the "forbidden field", "required field", and
"no extra field" logic is written once.
"""

from __future__ import annotations

from typing import Iterable

from .errors import CCRTValidationError
from .naming import (
    assert_no_forbidden_mechanism_fields,
    assert_no_model_input_leakage_fields,
    normalize_field_name,
)

__all__ = [
    "validate_field_names",
    "validate_required_fields",
    "validate_no_extra_fields",
]


def validate_field_names(
    fields: Iterable[str], *, include_model_input_leakage: bool = True
) -> None:
    """Reject forbidden field names.

    Always rejects forbidden mechanism fields. Also rejects model-input leakage
    fields when ``include_model_input_leakage`` is True (the default).
    """
    field_list = list(fields)
    assert_no_forbidden_mechanism_fields(field_list)
    if include_model_input_leakage:
        assert_no_model_input_leakage_fields(field_list)


def validate_required_fields(
    fields: Iterable[str], required: Iterable[str], context: str
) -> None:
    """Assert every field in ``required`` is present in ``fields``."""
    present = {normalize_field_name(f) for f in fields}
    required_norm = {normalize_field_name(f) for f in required}
    missing = sorted(required_norm - present)
    if missing:
        raise CCRTValidationError(
            f"{context}: missing required field(s): {missing} "
            f"(required: {sorted(required_norm)})"
        )


def validate_no_extra_fields(
    fields: Iterable[str], allowed: Iterable[str], context: str
) -> None:
    """Assert ``fields`` contains no names outside ``allowed``."""
    present = {normalize_field_name(f) for f in fields}
    allowed_norm = {normalize_field_name(f) for f in allowed}
    extra = sorted(present - allowed_norm)
    if extra:
        raise CCRTValidationError(
            f"{context}: unexpected extra field(s): {extra} "
            f"(allowed: {sorted(allowed_norm)})"
        )
