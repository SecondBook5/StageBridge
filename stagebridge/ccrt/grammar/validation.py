"""Validation helpers for grammar specs.

These functions hold the reusable, non-recursive checks that
``BiologicalSystemSpec.validate`` composes: unique-ID enforcement and
reference-existence enforcement. ``validate_biological_system_spec`` is the
public entry point and simply delegates to ``spec.validate()``.
"""

from __future__ import annotations

from typing import Callable, Iterable, Sequence, TYPE_CHECKING, TypeVar

from ..contracts.errors import CCRTGrammarError

if TYPE_CHECKING:  # avoid a runtime import cycle with spec.py
    from .spec import BiologicalSystemSpec

__all__ = [
    "require_unique_ids",
    "require_ids_exist",
    "validate_biological_system_spec",
]

_T = TypeVar("_T")


def require_unique_ids(
    items: Sequence[_T], id_getter: Callable[[_T], str], context: str
) -> None:
    """Raise ``CCRTGrammarError`` if any ID returned by ``id_getter`` repeats."""
    seen: set[str] = set()
    duplicates: list[str] = []
    for item in items:
        item_id = id_getter(item)
        if item_id in seen and item_id not in duplicates:
            duplicates.append(item_id)
        seen.add(item_id)
    if duplicates:
        raise CCRTGrammarError(
            f"{context}: duplicate id(s): {sorted(duplicates)}"
        )


def require_ids_exist(
    referenced_ids: Iterable[str], allowed_ids: Iterable[str], context: str
) -> None:
    """Raise ``CCRTGrammarError`` if any referenced ID is not in ``allowed_ids``."""
    allowed = set(allowed_ids)
    missing = sorted({rid for rid in referenced_ids if rid not in allowed})
    if missing:
        raise CCRTGrammarError(
            f"{context}: reference(s) to unknown id(s): {missing} "
            f"(known: {sorted(allowed)})"
        )


def validate_biological_system_spec(spec: "BiologicalSystemSpec") -> None:
    """Public validation entry point; delegates to ``spec.validate()``.

    Kept as a free function so callers can validate without reaching into the
    method, while the authoritative logic lives in one place (no recursion).
    """
    spec.validate()
