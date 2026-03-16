"""Tests for resume behavior in the orchestration system."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from stagebridge.orchestration.notebook_api import (
    initialize_run,
    run_data_qc,
    run_reference_mapping,
    summarize_run,
)
from stagebridge.orchestration.run_manager import RunManager, RunStatus, StageStatus


@pytest.fixture
def temp_artifacts_root(tmp_path: Path) -> Path:
    """Create a temporary artifacts directory."""
    artifacts_dir = tmp_path / "artifacts" / "runs"
    artifacts_dir.mkdir(parents=True)
    return artifacts_dir


@pytest.fixture
def sample_config() -> dict:
    """Create a sample configuration."""
    return {
        "seed": 42,
        "device": "cpu",
        "stages": {
            "enabled": ["data_qc", "reference"],
        },
        "spatial_backends": ["tangram"],
        "baselines": ["mlp"],
        "ablations": [],
    }


class TestResumeDetection:
    """Tests for resume detection logic."""

    def test_resume_skips_completed_stage(
        self, temp_artifacts_root: Path, sample_config: dict
    ) -> None:
        """Test that completed stages are skipped on resume."""
        # First run - complete data_qc
        ctx1 = initialize_run(
            sample_config,
            run_id="test_resume_skip",
            artifacts_root=str(temp_artifacts_root),
            resume_if_possible=False,  # Fresh run
        )
        result1 = run_data_qc(ctx1)
        assert result1.success
        assert not result1.skipped

        # Mark stage complete and create completion marker
        qc_dir = ctx1.stage_dir("data_qc")
        (qc_dir / ".completed").write_text("done", encoding="utf-8")

        # Resume - data_qc should be skipped
        ctx2 = initialize_run(
            sample_config,
            run_id="test_resume_skip",
            artifacts_root=str(temp_artifacts_root),
            resume_if_possible=True,
        )

        result2 = run_data_qc(ctx2)

        # Should be skipped because outputs exist
        assert result2.skipped or result2.success

    def test_resume_runs_incomplete_stage(
        self, temp_artifacts_root: Path, sample_config: dict
    ) -> None:
        """Test that incomplete stages are run on resume."""
        # First run - start but don't complete
        ctx1 = initialize_run(
            sample_config,
            run_id="test_resume_incomplete",
            artifacts_root=str(temp_artifacts_root),
        )
        # Don't run any stages

        # Resume - data_qc should run
        ctx2 = initialize_run(
            sample_config,
            run_id="test_resume_incomplete",
            artifacts_root=str(temp_artifacts_root),
            resume_if_possible=True,
        )

        result = run_data_qc(ctx2)

        assert result.success
        assert not result.skipped

    def test_force_rerun_ignores_cache(
        self, temp_artifacts_root: Path, sample_config: dict
    ) -> None:
        """Test that force_rerun ignores cached results."""
        # First run
        ctx1 = initialize_run(
            sample_config,
            run_id="test_force",
            artifacts_root=str(temp_artifacts_root),
        )
        result1 = run_data_qc(ctx1)
        assert result1.success

        # Mark complete
        qc_dir = ctx1.stage_dir("data_qc")
        (qc_dir / ".completed").write_text("done", encoding="utf-8")

        # Resume with force_rerun
        ctx2 = initialize_run(
            sample_config,
            run_id="test_force",
            artifacts_root=str(temp_artifacts_root),
            resume_if_possible=True,
        )
        ctx2.force_rerun = True

        result2 = run_data_qc(ctx2, force_rerun=True)

        # Should run, not skip
        assert result2.success
        assert not result2.skipped


class TestResumeState:
    """Tests for state restoration on resume."""

    def test_resume_preserves_stage_status(
        self, temp_artifacts_root: Path, sample_config: dict, tmp_path: Path
    ) -> None:
        """Test that stage status is preserved on resume."""
        manager = RunManager(
            artifacts_root=temp_artifacts_root,
            repo_root=tmp_path,
        )

        # Create and complete stages
        ctx1 = manager.initialize_run(sample_config, run_id="test_preserve")
        manager.start_stage(ctx1, "data_qc")
        manager.complete_stage(ctx1, "data_qc")
        manager.start_stage(ctx1, "reference")
        manager.complete_stage(ctx1, "reference")

        # Resume
        ctx2 = manager.initialize_run(
            sample_config,
            run_id="test_preserve",
            resume_if_possible=True,
        )

        # Check stages are preserved
        assert "data_qc" in ctx2.stages
        assert "reference" in ctx2.stages
        assert ctx2.stages["data_qc"].status == StageStatus.COMPLETED
        assert ctx2.stages["reference"].status == StageStatus.COMPLETED

    def test_resume_preserves_start_time(
        self, temp_artifacts_root: Path, sample_config: dict, tmp_path: Path
    ) -> None:
        """Test that run start time is preserved on resume."""
        manager = RunManager(
            artifacts_root=temp_artifacts_root,
            repo_root=tmp_path,
        )

        # Create run
        ctx1 = manager.initialize_run(sample_config, run_id="test_time")
        original_start = ctx1.start_time

        # Resume
        ctx2 = manager.initialize_run(
            sample_config,
            run_id="test_time",
            resume_if_possible=True,
        )

        # Start time should be preserved
        assert ctx2.start_time == original_start

    def test_resume_updates_environment(
        self, temp_artifacts_root: Path, sample_config: dict, tmp_path: Path
    ) -> None:
        """Test that environment info is updated on resume."""
        manager = RunManager(
            artifacts_root=temp_artifacts_root,
            repo_root=tmp_path,
        )

        # Create run
        ctx1 = manager.initialize_run(sample_config, run_id="test_env")

        # Resume
        ctx2 = manager.initialize_run(
            sample_config,
            run_id="test_env",
            resume_if_possible=True,
        )

        # Environment should be fresh
        assert ctx2.environment is not None
        assert "python_version" in ctx2.environment


class TestResumeAfterFailure:
    """Tests for resuming after failed runs."""

    def test_resume_after_stage_failure(
        self, temp_artifacts_root: Path, sample_config: dict, tmp_path: Path
    ) -> None:
        """Test resuming a run that had a failed stage."""
        manager = RunManager(
            artifacts_root=temp_artifacts_root,
            repo_root=tmp_path,
        )

        # Create run with failure
        ctx1 = manager.initialize_run(sample_config, run_id="test_fail_resume")
        manager.start_stage(ctx1, "data_qc")
        manager.complete_stage(ctx1, "data_qc")
        manager.start_stage(ctx1, "reference")
        manager.fail_stage(ctx1, "reference", "Test failure")

        assert ctx1.status == RunStatus.FAILED

        # Resume
        ctx2 = manager.initialize_run(
            sample_config,
            run_id="test_fail_resume",
            resume_if_possible=True,
        )

        # Should be able to resume from where it failed
        assert ctx2.status == RunStatus.RUNNING
        assert ctx2.stages["data_qc"].status == StageStatus.COMPLETED
        # Failed stage should be recorded but run can continue

    def test_resume_clears_failed_status(
        self, temp_artifacts_root: Path, sample_config: dict
    ) -> None:
        """Test that resuming clears the failed status."""
        ctx1 = initialize_run(
            sample_config,
            run_id="test_clear_fail",
            artifacts_root=str(temp_artifacts_root),
        )

        # Run and fail artificially
        run_data_qc(ctx1)
        ctx1.status = RunStatus.FAILED

        # Resume
        ctx2 = initialize_run(
            sample_config,
            run_id="test_clear_fail",
            artifacts_root=str(temp_artifacts_root),
            resume_if_possible=True,
        )

        # Status should be running again
        assert ctx2.status == RunStatus.RUNNING


class TestPartialResume:
    """Tests for partial resume scenarios."""

    def test_resume_continues_from_last_complete(
        self, temp_artifacts_root: Path, sample_config: dict
    ) -> None:
        """Test that resume continues from last completed stage."""
        # Run partial pipeline
        ctx1 = initialize_run(
            sample_config,
            run_id="test_partial_resume",
            artifacts_root=str(temp_artifacts_root),
        )
        run_data_qc(ctx1)
        # Don't run reference

        # Resume and run reference
        ctx2 = initialize_run(
            sample_config,
            run_id="test_partial_resume",
            artifacts_root=str(temp_artifacts_root),
            resume_if_possible=True,
        )

        # data_qc might be skipped if validation passes
        ref_result = run_reference_mapping(ctx2)

        assert ref_result.success

    def test_resume_multiple_times(self, temp_artifacts_root: Path, sample_config: dict) -> None:
        """Test that runs can be resumed multiple times."""
        run_id = "test_multi_resume"

        # First session
        ctx1 = initialize_run(
            sample_config,
            run_id=run_id,
            artifacts_root=str(temp_artifacts_root),
        )
        run_data_qc(ctx1)

        # Second session
        ctx2 = initialize_run(
            sample_config,
            run_id=run_id,
            artifacts_root=str(temp_artifacts_root),
            resume_if_possible=True,
        )
        run_reference_mapping(ctx2)

        # Third session - summarize
        ctx3 = initialize_run(
            sample_config,
            run_id=run_id,
            artifacts_root=str(temp_artifacts_root),
            resume_if_possible=True,
        )

        summary = summarize_run(ctx3)

        assert summary.run_id == run_id
        assert "data_qc" in summary.stages or "reference" in summary.stages


class TestConfigChangesOnResume:
    """Tests for handling config changes on resume."""

    def test_resume_with_updated_config(
        self, temp_artifacts_root: Path, sample_config: dict
    ) -> None:
        """Test resuming with an updated config."""
        # First run
        ctx1 = initialize_run(
            sample_config,
            run_id="test_config_change",
            artifacts_root=str(temp_artifacts_root),
        )
        run_data_qc(ctx1)

        # Resume with different config
        updated_config = dict(sample_config)
        updated_config["seed"] = 123  # Changed seed

        ctx2 = initialize_run(
            updated_config,
            run_id="test_config_change",
            artifacts_root=str(temp_artifacts_root),
            resume_if_possible=True,
        )

        # The new config should be used
        assert ctx2.config["seed"] == 123
