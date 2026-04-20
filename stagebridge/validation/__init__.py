"""
StageBridge Validation Module.

Provides comprehensive validation infrastructure:
- Artifact contract validation
- Leakage detection (donor, sample, neighborhood)
- Split validation and reproducibility
- Numerical stability checks
- Reproducibility tracking
- Reference geometry validation
"""

# Reference geometry validation (existing)
from .reference_geometry import (
    ReferenceGeometryValidator,
    ReferenceGeometryValidationReport,
    validate_reference_geometry,
    validate_reference_geometry_quick,
)

# Artifact checks
from .artifact_checks import (
    check_artifact_exists,
    check_parquet_schema,
    check_json_artifact,
    check_model_checkpoint,
    validate_run_artifacts,
    save_validation_report,
)

# Leakage detection
from .leakage_checks import (
    check_donor_leakage,
    check_sample_containment,
    check_neighborhood_leakage,
    check_reference_leakage,
    validate_split_manifest,
    run_leakage_audit,
)

# Split validation
from .split_checks import (
    create_split_manifest,
    verify_split_reproducibility,
    assign_cells_to_splits,
    validate_split_balance,
    save_split_manifest,
    load_split_manifest,
)

# Numerical stability
from .numerics import (
    check_tensor_health,
    check_gradient_health,
    check_loss_stability,
    check_embedding_quality,
    check_confidence_calibration,
)

# Reproducibility
from .repro import (
    capture_environment,
    compute_config_hash,
    save_repro_manifest,
    verify_reproducibility,
    set_all_seeds,
    create_run_id,
)

# Calibration (post-hoc confidence calibration)
from .calibration import (
    CalibrationResult,
    expected_calibration_error,
    temperature_scale,
    apply_temperature,
    calibrate_reference_confidence,
)

# Marker gene validation
from .markers import (
    MARKER_GENES,
    compute_marker_enrichment,
    validate_cell_type,
    validate_all_cell_types,
)

# Split validation (donor leakage)
from .splits import (
    validate_splits,
    validate_splits_from_files,
    check_paired_sample_leakage,
)

__all__ = [
    # Reference geometry
    "ReferenceGeometryValidator",
    "ReferenceGeometryValidationReport",
    "validate_reference_geometry",
    "validate_reference_geometry_quick",
    # Artifact checks
    "check_artifact_exists",
    "check_parquet_schema",
    "check_json_artifact",
    "check_model_checkpoint",
    "validate_run_artifacts",
    "save_validation_report",
    # Leakage checks
    "check_donor_leakage",
    "check_sample_containment",
    "check_neighborhood_leakage",
    "check_reference_leakage",
    "validate_split_manifest",
    "run_leakage_audit",
    # Split checks
    "create_split_manifest",
    "verify_split_reproducibility",
    "assign_cells_to_splits",
    "validate_split_balance",
    "save_split_manifest",
    "load_split_manifest",
    # Numerics
    "check_tensor_health",
    "check_gradient_health",
    "check_loss_stability",
    "check_embedding_quality",
    "check_confidence_calibration",
    # Reproducibility
    "capture_environment",
    "compute_config_hash",
    "save_repro_manifest",
    "verify_reproducibility",
    "set_all_seeds",
    "create_run_id",
    # Calibration
    "CalibrationResult",
    "expected_calibration_error",
    "temperature_scale",
    "apply_temperature",
    "calibrate_reference_confidence",
    # Marker validation
    "MARKER_GENES",
    "compute_marker_enrichment",
    "validate_cell_type",
    "validate_all_cell_types",
    # Split validation
    "validate_splits",
    "validate_splits_from_files",
    "check_paired_sample_leakage",
]
