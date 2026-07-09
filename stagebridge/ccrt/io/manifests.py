"""In-memory provenance manifest utility.

A manifest is a small dict describing a processed CCRT bundle: which biological
system it belongs to, its source dataset, which standardized tables it contains,
and bookkeeping. This milestone builds and validates the manifest *in memory
only* — it does not write JSON or any file (that belongs to a later IO
milestone).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..contracts.errors import CCRTValidationError
from ..contracts.naming import (
    assert_no_forbidden_mechanism_fields,
    assert_no_model_input_leakage_fields,
)
from ..contracts.tables import TABLE_SCHEMAS

__all__ = [
    "build_manifest",
    "validate_manifest",
]

#: Keys the manifest always carries; ``extra`` is merged in at the top level.
_REQUIRED_MANIFEST_KEYS = (
    "biological_system_id",
    "source_dataset",
    "table_names",
    "schema_version",
    "created_by",
)


def _require_nonempty_str(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise CCRTValidationError(f"manifest: '{name}' must be a non-empty str")


def _known_table_names() -> frozenset[str]:
    return frozenset(TABLE_SCHEMAS.keys())


def build_manifest(
    *,
    biological_system_id: str,
    source_dataset: str,
    table_names: Sequence[str],
    schema_version: str = "ccrt-0.1",
    created_by: str = "stagebridge.ccrt",
    notes: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an in-memory provenance manifest and validate it before returning.

    ``extra`` keys are merged at the top level (so they are subject to the same
    forbidden-field validation). Building always validates, so a returned
    manifest is guaranteed well-formed.
    """
    manifest: dict[str, Any] = {
        "biological_system_id": biological_system_id,
        "source_dataset": source_dataset,
        "table_names": list(table_names),
        "schema_version": schema_version,
        "created_by": created_by,
    }
    if notes is not None:
        manifest["notes"] = notes
    if extra:
        for key, value in extra.items():
            if key in manifest:
                raise CCRTValidationError(
                    f"manifest: extra key '{key}' collides with a reserved key"
                )
            manifest[key] = value

    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate a provenance manifest.

    Enforces required non-empty fields, known standardized table names, and
    forbidden-field hygiene on every manifest key (including ``extra`` keys).
    """
    if not isinstance(manifest, Mapping):
        raise CCRTValidationError(
            f"manifest must be a mapping, got {type(manifest).__name__}"
        )

    # -- forbidden-field hygiene on all keys (including any extra keys) --
    keys = list(manifest.keys())
    assert_no_forbidden_mechanism_fields(keys)
    assert_no_model_input_leakage_fields(keys)

    # -- required keys present --
    for key in _REQUIRED_MANIFEST_KEYS:
        if key not in manifest:
            raise CCRTValidationError(f"manifest: missing required key '{key}'")

    # -- required non-empty string fields --
    _require_nonempty_str(manifest["biological_system_id"], "biological_system_id")
    _require_nonempty_str(manifest["source_dataset"], "source_dataset")
    _require_nonempty_str(manifest["schema_version"], "schema_version")
    _require_nonempty_str(manifest["created_by"], "created_by")

    # -- table_names required, non-empty, and all known standardized tables --
    table_names = manifest["table_names"]
    if isinstance(table_names, (str, bytes, Mapping)):
        raise CCRTValidationError(
            "manifest: 'table_names' must be a sequence of table names, "
            f"got {type(table_names).__name__}"
        )
    try:
        names = list(table_names)
    except TypeError as exc:
        raise CCRTValidationError(
            "manifest: 'table_names' must be an iterable of table names"
        ) from exc
    if not names:
        raise CCRTValidationError("manifest: 'table_names' must be non-empty")

    known = _known_table_names()
    unknown = sorted({n for n in names if n not in known})
    if unknown:
        raise CCRTValidationError(
            f"manifest: unknown standardized table name(s): {unknown} "
            f"(known: {sorted(known)})"
        )
