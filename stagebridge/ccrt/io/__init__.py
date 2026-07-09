"""CCRT io — safe record validation and provenance manifests.

This milestone provides record-level (in-memory) validation of standardized
table-like data and an in-memory provenance manifest utility. No file reading /
writing is implemented yet. Imports only ``contracts`` and the standard library.
"""

from __future__ import annotations

from .manifests import build_manifest, validate_manifest
from .records import (
    infer_record_fields,
    require_non_empty_records,
    validate_records,
)

__all__ = [
    "infer_record_fields",
    "require_non_empty_records",
    "validate_records",
    "build_manifest",
    "validate_manifest",
]
