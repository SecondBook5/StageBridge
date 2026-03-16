"""Stage and artifact validation for StageBridge orchestration.

This module provides validation utilities for checking stage completion,
artifact integrity, and manifest consistency.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from stagebridge.orchestration.run_manager import RunContext


@dataclass
class ValidationResult:
    """Result of a validation check."""

    success: bool
    stage_name: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checked_files: list[str] = field(default_factory=list)
    missing_files: list[str] = field(default_factory=list)
    invalid_files: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        """Return True if validation succeeded."""
        return self.success

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "stage_name": self.stage_name,
            "errors": self.errors,
            "warnings": self.warnings,
            "checked_files": self.checked_files,
            "missing_files": self.missing_files,
            "invalid_files": self.invalid_files,
            "details": self.details,
        }


# Expected outputs per stage (file patterns)
STAGE_EXPECTED_OUTPUTS: dict[str, list[str]] = {
    "data_qc": [
        "qc_report.json",
        "qc_summary.html",
    ],
    "reference": [
        "reference_mapping.h5ad",
        "reference_metrics.json",
    ],
    "spatial_backend": [
        "backend_benchmark.json",
        "selected_backend.txt",
    ],
    "baselines": [
        "baseline_results.json",
    ],
    "full_model": [
        "model_checkpoint.pt",
        "training_metrics.json",
    ],
    "ablations": [
        "ablation_results.json",
    ],
    "biology": [
        "biology_validation.json",
    ],
    "figures": [
        "figures_manifest.json",
    ],
}


def _check_file_readable(path: Path) -> tuple[bool, str | None]:
    """Check if a file is readable and not corrupted.

    Returns (success, error_message).
    """
    if not path.exists():
        return False, f"File does not exist: {path}"

    if not path.is_file():
        return False, f"Not a file: {path}"

    try:
        size = path.stat().st_size
        if size == 0:
            return False, f"File is empty: {path}"

        # Try to read first bytes
        with path.open("rb") as f:
            _ = f.read(1024)

        return True, None
    except PermissionError:
        return False, f"Permission denied: {path}"
    except Exception as e:
        return False, f"Error reading file: {path} - {e}"


def _check_json_valid(path: Path) -> tuple[bool, str | None]:
    """Check if a JSON file is valid."""
    try:
        with path.open("r", encoding="utf-8") as f:
            json.load(f)
        return True, None
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {path} - {e}"
    except Exception as e:
        return False, f"Error reading JSON: {path} - {e}"


def _get_stage_dir(run_dir: Path, stage_name: str) -> Path:
    """Get the output directory for a stage."""
    stage_to_subdir = {
        "data_qc": "qc",
        "reference": "references",
        "spatial_backend": "spatial_backends",
        "baselines": "baselines",
        "full_model": "full_model",
        "ablations": "ablations",
        "biology": "biology",
        "figures": "figures",
    }
    subdir = stage_to_subdir.get(stage_name, stage_name)
    return run_dir / subdir


def validate_stage_artifacts(
    ctx: "RunContext",
    stage_name: str,
    *,
    strict: bool = False,
) -> ValidationResult:
    """Validate that all expected artifacts exist and are valid for a stage.

    Parameters
    ----------
    ctx : RunContext
        The run context
    stage_name : str
        Name of the stage to validate
    strict : bool
        If True, treat warnings as errors (default: False)

    Returns
    -------
    ValidationResult
        The validation result
    """
    result = ValidationResult(
        success=True,
        stage_name=stage_name,
    )

    stage_dir = _get_stage_dir(ctx.run_dir, stage_name)

    # Check directory exists
    if not stage_dir.exists():
        result.success = False
        result.errors.append(f"Stage directory does not exist: {stage_dir}")
        return result

    # Check completion marker
    completion_marker = stage_dir / ".completed"
    if not completion_marker.exists():
        result.warnings.append(f"Completion marker missing: {completion_marker}")

    # Check expected outputs
    expected = STAGE_EXPECTED_OUTPUTS.get(stage_name, [])
    for expected_file in expected:
        file_path = stage_dir / expected_file
        result.checked_files.append(str(file_path))

        if not file_path.exists():
            result.missing_files.append(expected_file)
            result.errors.append(f"Missing expected artifact: {expected_file}")
            result.success = False
            continue

        # Check file is readable
        readable, error = _check_file_readable(file_path)
        if not readable:
            result.invalid_files.append(expected_file)
            result.errors.append(error or f"Invalid file: {expected_file}")
            result.success = False
            continue

        # For JSON files, validate format
        if expected_file.endswith(".json"):
            valid, error = _check_json_valid(file_path)
            if not valid:
                result.invalid_files.append(expected_file)
                result.errors.append(error or f"Invalid JSON: {expected_file}")
                result.success = False

    # Check stage manifest
    manifests_dir = ctx.run_dir / "manifests"
    stage_manifest = manifests_dir / f"{stage_name}_manifest.json"
    if stage_manifest.exists():
        valid, error = _check_json_valid(stage_manifest)
        if not valid:
            result.warnings.append(f"Invalid stage manifest: {error}")
        else:
            try:
                with stage_manifest.open("r", encoding="utf-8") as f:
                    manifest_data = json.load(f)
                    manifest_status = manifest_data.get("status")
                    if manifest_status != "completed":
                        result.warnings.append(
                            f"Stage manifest status is '{manifest_status}', expected 'completed'"
                        )
            except Exception as e:
                result.warnings.append(f"Could not read manifest status: {e}")

    # In strict mode, warnings become errors
    if strict and result.warnings:
        result.errors.extend(result.warnings)
        result.success = False

    return result


def validate_run_artifacts(ctx: "RunContext") -> dict[str, ValidationResult]:
    """Validate all stages in a run.

    Parameters
    ----------
    ctx : RunContext
        The run context

    Returns
    -------
    dict
        Dictionary mapping stage names to validation results
    """
    results: dict[str, ValidationResult] = {}

    # Get list of stages that should be validated
    stages = ctx.config.get("stages", {})
    if isinstance(stages, dict):
        enabled_stages = stages.get("enabled", list(STAGE_EXPECTED_OUTPUTS.keys()))
    else:
        enabled_stages = list(STAGE_EXPECTED_OUTPUTS.keys())

    for stage_name in enabled_stages:
        results[stage_name] = validate_stage_artifacts(ctx, stage_name)

    return results


def check_stage_can_resume(
    ctx: "RunContext",
    stage_name: str,
) -> tuple[bool, str]:
    """Check if a stage can be resumed (outputs exist and are valid).

    Parameters
    ----------
    ctx : RunContext
        The run context
    stage_name : str
        Name of the stage

    Returns
    -------
    tuple of (bool, str)
        (can_resume, reason)
    """
    # If force_rerun is set, don't resume
    if ctx.force_rerun:
        return False, "force_rerun is enabled"

    # Check if stage directory exists
    stage_dir = _get_stage_dir(ctx.run_dir, stage_name)
    if not stage_dir.exists():
        return False, "stage directory does not exist"

    # Check completion marker
    completion_marker = stage_dir / ".completed"
    if not completion_marker.exists():
        return False, "completion marker missing"

    # Validate artifacts
    validation = validate_stage_artifacts(ctx, stage_name)
    if not validation.success:
        return False, f"validation failed: {'; '.join(validation.errors[:3])}"

    return True, "outputs exist and validation passed"


def should_run_stage(
    ctx: "RunContext",
    stage_name: str,
) -> tuple[bool, str]:
    """Determine if a stage should be run.

    Parameters
    ----------
    ctx : RunContext
        The run context
    stage_name : str
        Name of the stage

    Returns
    -------
    tuple of (bool, str)
        (should_run, reason)
    """
    # Check if stage is enabled
    stages = ctx.config.get("stages", {})
    if isinstance(stages, dict):
        enabled_stages = stages.get("enabled", list(STAGE_EXPECTED_OUTPUTS.keys()))
    else:
        enabled_stages = list(STAGE_EXPECTED_OUTPUTS.keys())

    if stage_name not in enabled_stages:
        return False, "stage is not enabled in config"

    # Check if we can resume
    if ctx.resume_if_possible and not ctx.force_rerun:
        can_resume, reason = check_stage_can_resume(ctx, stage_name)
        if can_resume:
            return False, f"skipping (resume): {reason}"

    return True, "stage should run"


def validate_config_for_stage(
    config: dict[str, Any],
    stage_name: str,
) -> ValidationResult:
    """Validate that config has required fields for a stage.

    Parameters
    ----------
    config : dict
        The configuration
    stage_name : str
        Name of the stage

    Returns
    -------
    ValidationResult
        The validation result
    """
    result = ValidationResult(
        success=True,
        stage_name=stage_name,
    )

    # Stage-specific config requirements
    stage_requirements: dict[str, list[str]] = {
        "data_qc": [],
        "reference": ["reference"],
        "spatial_backend": ["spatial_backends"],
        "baselines": ["baselines"],
        "full_model": [],
        "ablations": ["ablations"],
        "biology": [],
        "figures": [],
    }

    required_keys = stage_requirements.get(stage_name, [])

    for key in required_keys:
        if key not in config or config[key] is None:
            result.errors.append(f"Missing required config key for {stage_name}: {key}")
            result.success = False

    return result


def format_validation_errors(
    result: ValidationResult,
    log_path: Path | None = None,
) -> str:
    """Format validation errors for display.

    Parameters
    ----------
    result : ValidationResult
        The validation result
    log_path : Path, optional
        Path to log file

    Returns
    -------
    str
        Formatted error message
    """
    lines = [
        f"Validation failed for stage '{result.stage_name}'",
        "",
    ]

    if result.missing_files:
        lines.append("Missing files:")
        for f in result.missing_files:
            lines.append(f"  - {f}")
        lines.append("")

    if result.invalid_files:
        lines.append("Invalid files:")
        for f in result.invalid_files:
            lines.append(f"  - {f}")
        lines.append("")

    if result.errors:
        lines.append("Errors:")
        for e in result.errors:
            lines.append(f"  - {e}")
        lines.append("")

    if result.warnings:
        lines.append("Warnings:")
        for w in result.warnings:
            lines.append(f"  - {w}")
        lines.append("")

    if log_path:
        lines.append(f"See logs at: {log_path}")

    return "\n".join(lines)
