"""Validation module for StageBridge pipeline outputs.

This module provides validation checks for:
- Artifact contracts (expected files exist with correct schemas)
- Numerical stability (no NaN, valid ranges)
- Donor/split leakage detection
- Reproducibility verification
- Comprehensive validation reports
"""

from stagebridge.validation.reference_geometry import (
    ReferenceGeometryValidator,
    ReferenceGeometryValidationReport,
    validate_reference_geometry,
    validate_reference_geometry_quick,
)

__all__ = [
    "ReferenceGeometryValidator",
    "ReferenceGeometryValidationReport",
    "validate_reference_geometry",
    "validate_reference_geometry_quick",
]
