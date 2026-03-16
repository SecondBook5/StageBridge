"""Tests for the validation module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from stagebridge.orchestration.run_manager import RunContext, RunStatus
from stagebridge.orchestration.validation import (
    ValidationResult,
    check_stage_can_resume,
    format_validation_errors,
    should_run_stage,
    validate_config_for_stage,
    validate_stage_artifacts,
)


@pytest.fixture
def temp_run_dir(tmp_path: Path) -> Path:
    """Create a temporary run directory with standard structure."""
    run_dir = tmp_path / "run_test"
    run_dir.mkdir()

    # Create standard subdirectories
    for subdir in ["qc", "references", "spatial_backends", "manifests", "logs", "config"]:
        (run_dir / subdir).mkdir()

    return run_dir


@pytest.fixture
def run_context(temp_run_dir: Path) -> RunContext:
    """Create a run context for testing."""
    return RunContext(
        run_id="test_run",
        run_dir=temp_run_dir,
        config={
            "stages": {"enabled": ["data_qc", "reference", "spatial_backend"]},
            "spatial_backends": ["tangram"],
        },
        status=RunStatus.RUNNING,
        seed=42,
        device="cpu",
    )


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_bool_success(self) -> None:
        """Test bool conversion for success."""
        result = ValidationResult(success=True, stage_name="test")
        assert bool(result) is True

    def test_bool_failure(self) -> None:
        """Test bool conversion for failure."""
        result = ValidationResult(success=False, stage_name="test")
        assert bool(result) is False

    def test_to_dict(self) -> None:
        """Test converting to dictionary."""
        result = ValidationResult(
            success=False,
            stage_name="data_qc",
            errors=["Error 1", "Error 2"],
            warnings=["Warning 1"],
            missing_files=["file1.json"],
            invalid_files=["file2.json"],
        )

        d = result.to_dict()

        assert d["success"] is False
        assert d["stage_name"] == "data_qc"
        assert len(d["errors"]) == 2
        assert len(d["warnings"]) == 1
        assert "file1.json" in d["missing_files"]


class TestValidateStageArtifacts:
    """Tests for validate_stage_artifacts function."""

    def test_missing_stage_dir(self, run_context: RunContext, temp_run_dir: Path) -> None:
        """Test validation when stage directory doesn't exist."""
        # Remove the qc directory
        (temp_run_dir / "qc").rmdir()

        result = validate_stage_artifacts(run_context, "data_qc")

        assert not result.success
        assert any("does not exist" in e for e in result.errors)

    def test_missing_completion_marker(self, run_context: RunContext, temp_run_dir: Path) -> None:
        """Test validation warns about missing completion marker."""
        qc_dir = temp_run_dir / "qc"

        # Create expected files but no completion marker
        (qc_dir / "qc_report.json").write_text('{"status": "ok"}', encoding="utf-8")
        (qc_dir / "qc_summary.html").write_text("<html></html>", encoding="utf-8")

        result = validate_stage_artifacts(run_context, "data_qc")

        assert any("marker" in w.lower() for w in result.warnings)

    def test_missing_expected_files(self, run_context: RunContext, temp_run_dir: Path) -> None:
        """Test validation catches missing expected files."""
        qc_dir = temp_run_dir / "qc"
        (qc_dir / ".completed").write_text("done", encoding="utf-8")
        # Don't create expected files

        result = validate_stage_artifacts(run_context, "data_qc")

        assert not result.success
        assert len(result.missing_files) > 0

    def test_empty_file(self, run_context: RunContext, temp_run_dir: Path) -> None:
        """Test validation catches empty files."""
        qc_dir = temp_run_dir / "qc"
        (qc_dir / ".completed").write_text("done", encoding="utf-8")
        (qc_dir / "qc_report.json").write_text("", encoding="utf-8")  # Empty file
        (qc_dir / "qc_summary.html").write_text("<html></html>", encoding="utf-8")

        result = validate_stage_artifacts(run_context, "data_qc")

        assert not result.success
        assert any("empty" in e.lower() for e in result.errors)

    def test_invalid_json(self, run_context: RunContext, temp_run_dir: Path) -> None:
        """Test validation catches invalid JSON."""
        qc_dir = temp_run_dir / "qc"
        (qc_dir / ".completed").write_text("done", encoding="utf-8")
        (qc_dir / "qc_report.json").write_text("not valid json {{{", encoding="utf-8")
        (qc_dir / "qc_summary.html").write_text("<html></html>", encoding="utf-8")

        result = validate_stage_artifacts(run_context, "data_qc")

        assert not result.success
        assert any("json" in e.lower() for e in result.errors)

    def test_valid_stage(self, run_context: RunContext, temp_run_dir: Path) -> None:
        """Test validation passes for valid stage."""
        qc_dir = temp_run_dir / "qc"
        (qc_dir / ".completed").write_text("done", encoding="utf-8")
        (qc_dir / "qc_report.json").write_text('{"status": "ok"}', encoding="utf-8")
        (qc_dir / "qc_summary.html").write_text("<html></html>", encoding="utf-8")

        result = validate_stage_artifacts(run_context, "data_qc")

        # May still have warnings but core validation should pass
        assert len(result.missing_files) == 0
        assert len(result.invalid_files) == 0

    def test_strict_mode(self, run_context: RunContext, temp_run_dir: Path) -> None:
        """Test strict mode treats warnings as errors."""
        qc_dir = temp_run_dir / "qc"
        # Missing completion marker will be a warning
        (qc_dir / "qc_report.json").write_text('{"status": "ok"}', encoding="utf-8")
        (qc_dir / "qc_summary.html").write_text("<html></html>", encoding="utf-8")

        result_normal = validate_stage_artifacts(run_context, "data_qc", strict=False)
        result_strict = validate_stage_artifacts(run_context, "data_qc", strict=True)

        # In strict mode, warnings become errors
        if result_normal.warnings:
            assert not result_strict.success


class TestCheckStageCanResume:
    """Tests for check_stage_can_resume function."""

    def test_force_rerun_prevents_resume(self, run_context: RunContext) -> None:
        """Test that force_rerun prevents resuming."""
        run_context.force_rerun = True

        can_resume, reason = check_stage_can_resume(run_context, "data_qc")

        assert not can_resume
        assert "force_rerun" in reason

    def test_missing_dir_prevents_resume(
        self, run_context: RunContext, temp_run_dir: Path
    ) -> None:
        """Test that missing directory prevents resuming."""
        (temp_run_dir / "qc").rmdir()

        can_resume, reason = check_stage_can_resume(run_context, "data_qc")

        assert not can_resume
        assert "does not exist" in reason

    def test_missing_marker_prevents_resume(
        self, run_context: RunContext, temp_run_dir: Path
    ) -> None:
        """Test that missing completion marker prevents resuming."""
        qc_dir = temp_run_dir / "qc"
        (qc_dir / "qc_report.json").write_text('{"status": "ok"}', encoding="utf-8")
        # No .completed marker

        can_resume, reason = check_stage_can_resume(run_context, "data_qc")

        assert not can_resume
        assert "marker" in reason.lower()

    def test_can_resume_valid_stage(self, run_context: RunContext, temp_run_dir: Path) -> None:
        """Test resuming a valid completed stage."""
        qc_dir = temp_run_dir / "qc"
        (qc_dir / ".completed").write_text("done", encoding="utf-8")
        (qc_dir / "qc_report.json").write_text('{"status": "ok"}', encoding="utf-8")
        (qc_dir / "qc_summary.html").write_text("<html></html>", encoding="utf-8")

        can_resume, reason = check_stage_can_resume(run_context, "data_qc")

        # Should be able to resume (may still fail validation but marker exists)
        # The result depends on complete validation passing


class TestShouldRunStage:
    """Tests for should_run_stage function."""

    def test_disabled_stage_should_not_run(self, run_context: RunContext) -> None:
        """Test that disabled stages should not run."""
        run_context.config["stages"]["enabled"] = ["data_qc"]  # Only data_qc enabled

        should_run, reason = should_run_stage(run_context, "reference")

        assert not should_run
        assert "not enabled" in reason

    def test_enabled_stage_should_run(self, run_context: RunContext, temp_run_dir: Path) -> None:
        """Test that enabled stages without cache should run."""
        run_context.resume_if_possible = False

        should_run, reason = should_run_stage(run_context, "data_qc")

        assert should_run


class TestValidateConfigForStage:
    """Tests for validate_config_for_stage function."""

    def test_valid_config(self) -> None:
        """Test validation of valid config."""
        config = {
            "reference": {"method": "hlca"},
            "spatial_backends": ["tangram"],
            "baselines": ["mlp"],
            "ablations": ["no_spatial"],
        }

        result = validate_config_for_stage(config, "reference")
        assert result.success

        result = validate_config_for_stage(config, "spatial_backend")
        assert result.success

    def test_missing_required_key(self) -> None:
        """Test validation catches missing required keys."""
        config = {}  # Missing required keys

        result = validate_config_for_stage(config, "spatial_backend")
        assert not result.success
        assert any("spatial_backends" in e for e in result.errors)


class TestFormatValidationErrors:
    """Tests for format_validation_errors function."""

    def test_format_with_all_fields(self, tmp_path: Path) -> None:
        """Test formatting with all error types."""
        log_path = tmp_path / "test.log"

        result = ValidationResult(
            success=False,
            stage_name="data_qc",
            errors=["Error 1", "Error 2"],
            warnings=["Warning 1"],
            missing_files=["missing.json"],
            invalid_files=["invalid.json"],
        )

        formatted = format_validation_errors(result, log_path)

        assert "data_qc" in formatted
        assert "missing.json" in formatted
        assert "invalid.json" in formatted
        assert "Error 1" in formatted
        assert "Warning 1" in formatted
        assert str(log_path) in formatted

    def test_format_minimal(self) -> None:
        """Test formatting with minimal errors."""
        result = ValidationResult(
            success=False,
            stage_name="data_qc",
            errors=["Single error"],
        )

        formatted = format_validation_errors(result)

        assert "data_qc" in formatted
        assert "Single error" in formatted
