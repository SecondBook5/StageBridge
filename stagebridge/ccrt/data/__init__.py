"""CCRT data — validated tables to model-ready ``CCRTBatch``.

Owns the batch container, split-manifest validation, and the cross-system
categorical index registry. Contains NO disease-specific code and imports only
``contracts``, ``grammar``, and the standard library.
"""

from __future__ import annotations

from .batch import CCRTBatch
from .collate import collate_ccrt_records
from .dataset import CCRTTableDataset
from .indexing import (
    QUALIFIED_ID_SEPARATOR,
    CCRTIndexRegistry,
    qualify_grammar_id,
)
from .splits import (
    ALLOWED_SPLIT_STRATEGIES,
    FORBIDDEN_SPLIT_STRATEGIES,
    validate_split_manifest,
)

__all__ = [
    "CCRTBatch",
    "CCRTTableDataset",
    "collate_ccrt_records",
    "validate_split_manifest",
    "ALLOWED_SPLIT_STRATEGIES",
    "FORBIDDEN_SPLIT_STRATEGIES",
    "QUALIFIED_ID_SEPARATOR",
    "qualify_grammar_id",
    "CCRTIndexRegistry",
]
