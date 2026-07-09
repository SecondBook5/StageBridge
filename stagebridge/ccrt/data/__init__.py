"""CCRT data — validated tables to model-ready ``CCRTBatch``.

Owns the batch container and split-manifest validation. Contains NO
disease-specific code and imports only ``contracts`` and the standard library.
"""

from __future__ import annotations

from .batch import CCRTBatch
from .splits import (
    ALLOWED_SPLIT_STRATEGIES,
    FORBIDDEN_SPLIT_STRATEGIES,
    validate_split_manifest,
)

__all__ = [
    "CCRTBatch",
    "validate_split_manifest",
    "ALLOWED_SPLIT_STRATEGIES",
    "FORBIDDEN_SPLIT_STRATEGIES",
]
