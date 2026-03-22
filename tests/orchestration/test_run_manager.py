"""Tests for the run manager module."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from stagebridge.orchestration.run_manager import (
    RunManager,
    RunStatus,
    StageStatus,
)


@pytest.fixture
def temp_artifacts_dir(tmp_path: Path) -> Path:
    """Create a temporary artifacts directory."""
    artifacts_dir = tmp_path / "artifacts" / "runs"
    artifacts_dir.mkdir(parents=True)
    return artifacts_dir


@pytest.fixture
def run_manager(temp_artifacts_dir: Path, tmp_path: Path) -> RunManager:
    """Create a run manager with temporary directories."""
    return RunManager(
        artifacts_root=temp_artifacts_dir,
        repo_root=tmp_path,
    )


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
    }


class TestRunManager:
    """Tests for RunManager class."""

    def test_initialize_new_run(self, run_manager: RunManager, sample_config: dict) -> None:
        """Test initializing a new run."""
        ctx = run_manager.initialize_run(sample_config, run_id="test_run")

        assert ctx.run_id == "test_run"
        assert ctx.status == RunStatus.RUNNING
        assert ctx.run_dir.exists()
        assert ctx.config == sample_config
        assert ctx.seed == 42
        assert ctx.device == "cpu"

    def test_run_directory_structure(self, run_manager: RunManager, sample_config: dict) -> None:
        """Test that run directory structure is created correctly."""
        ctx = run_manager.initialize_run(sample_config, run_id="test_structure")

        # Check all required subdirectories exist
        expected_subdirs = [
            "config",
            "splits",
            "data",
            "qc",
            "references",
            "spatial_backends",
            "baselines",
            "full_model",
            "ablations",
            "biology",
            "figures",
            "notebook_cache",
            "logs",
            "manifests",
            "checkpoints",
            "metrics",
        ]

        for subdir in expected_subdirs:
            assert (ctx.run_dir / subdir).exists(), f"Missing subdirectory: {subdir}"

    def test_metadata_saved(self, run_manager: RunManager, sample_config: dict) -> None:
        """Test that run metadata is saved correctly."""
        ctx = run_manager.initialize_run(sample_config, run_id="test_metadata")

        metadata_path = ctx.metadata_path
        assert metadata_path.exists()

        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = yaml.safe_load(f)

        assert metadata["run_id"] == "test_metadata"
        assert metadata["status"] == "running"
        assert metadata["seed"] == 42
        assert metadata["device"] == "cpu"
        assert "git_commit" in metadata
        assert "environment" in metadata

    def test_auto_generate_run_id(self, run_manager: RunManager, sample_config: dict) -> None:
        """Test that run ID is auto-generated when not provided."""
        ctx = run_manager.initialize_run(sample_config)

        assert ctx.run_id is not None
        assert ctx.run_id.startswith("run_")

    def test_start_and_complete_stage(self, run_manager: RunManager, sample_config: dict) -> None:
        """Test starting and completing a stage."""
        ctx = run_manager.initialize_run(sample_config, run_id="test_stage")

        # Start stage
        stage_info = run_manager.start_stage(ctx, "data_qc")
        assert stage_info.status == StageStatus.RUNNING
        assert stage_info.start_time is not None
        assert ctx.current_stage == "data_qc"

        # Complete stage
        run_manager.complete_stage(ctx, "data_qc", artifacts=["output.json"])
        assert ctx.stages["data_qc"].status == StageStatus.COMPLETED
        assert ctx.stages["data_qc"].end_time is not None
        assert ctx.stages["data_qc"].duration_seconds is not None
        assert "output.json" in ctx.stages["data_qc"].artifacts

    def test_fail_stage(self, run_manager: RunManager, sample_config: dict) -> None:
        """Test failing a stage."""
        ctx = run_manager.initialize_run(sample_config, run_id="test_fail")

        run_manager.start_stage(ctx, "data_qc")
        run_manager.fail_stage(ctx, "data_qc", "Test error message")

        assert ctx.stages["data_qc"].status == StageStatus.FAILED
        assert ctx.stages["data_qc"].error_message == "Test error message"
        assert ctx.status == RunStatus.FAILED

    def test_skip_stage(self, run_manager: RunManager, sample_config: dict) -> None:
        """Test skipping a stage."""
        ctx = run_manager.initialize_run(sample_config, run_id="test_skip")

        run_manager.skip_stage(ctx, "data_qc", reason="cached")

        assert ctx.stages["data_qc"].status == StageStatus.SKIPPED
        assert "cached" in ctx.stages["data_qc"].error_message

    def test_finalize_run_success(self, run_manager: RunManager, sample_config: dict) -> None:
        """Test finalizing a successful run."""
        ctx = run_manager.initialize_run(sample_config, run_id="test_finalize")

        run_manager.start_stage(ctx, "data_qc")
        run_manager.complete_stage(ctx, "data_qc")

        run_manager.finalize_run(ctx, success=True)

        assert ctx.status == RunStatus.COMPLETED
        assert ctx.end_time is not None

    def test_finalize_run_partial(self, run_manager: RunManager, sample_config: dict) -> None:
        """Test finalizing a run with some failed stages."""
        ctx = run_manager.initialize_run(sample_config, run_id="test_partial")

        run_manager.start_stage(ctx, "data_qc")
        run_manager.complete_stage(ctx, "data_qc")

        run_manager.start_stage(ctx, "reference")
        run_manager.fail_stage(ctx, "reference", "Test error")

        run_manager.finalize_run(ctx, success=True)

        assert ctx.status == RunStatus.PARTIAL

    def test_list_runs(self, run_manager: RunManager, sample_config: dict) -> None:
        """Test listing all runs."""
        run_manager.initialize_run(sample_config, run_id="run_1")
        run_manager.initialize_run(sample_config, run_id="run_2")

        runs = run_manager.list_runs()

        assert "run_1" in runs
        assert "run_2" in runs

    def test_load_run_context(self, run_manager: RunManager, sample_config: dict) -> None:
        """Test loading an existing run context."""
        # Create a run
        ctx = run_manager.initialize_run(sample_config, run_id="test_load")
        run_manager.start_stage(ctx, "data_qc")
        run_manager.complete_stage(ctx, "data_qc")

        # Load it back
        loaded_ctx = run_manager.load_run_context("test_load")

        assert loaded_ctx is not None
        assert loaded_ctx.run_id == "test_load"
        assert "data_qc" in loaded_ctx.stages

    def test_resume_run(self, run_manager: RunManager, sample_config: dict) -> None:
        """Test resuming an existing run."""
        # Create initial run
        ctx1 = run_manager.initialize_run(sample_config, run_id="test_resume")
        run_manager.start_stage(ctx1, "data_qc")
        run_manager.complete_stage(ctx1, "data_qc")

        # Resume the run
        ctx2 = run_manager.initialize_run(
            sample_config,
            run_id="test_resume",
            resume_if_possible=True,
        )

        assert ctx2.run_id == "test_resume"
        assert "data_qc" in ctx2.stages
        assert ctx2.stages["data_qc"].status == StageStatus.COMPLETED


class TestRunContext:
    """Tests for RunContext dataclass."""

    def test_stage_dir(self, run_manager: RunManager, sample_config: dict) -> None:
        """Test getting stage directories."""
        ctx = run_manager.initialize_run(sample_config, run_id="test_stage_dir")

        assert ctx.stage_dir("data_qc") == ctx.run_dir / "qc"
        assert ctx.stage_dir("reference") == ctx.run_dir / "references"
        assert ctx.stage_dir("spatial_backend") == ctx.run_dir / "spatial_backends"
        assert ctx.stage_dir("custom") == ctx.run_dir / "custom"

    def test_stage_log(self, run_manager: RunManager, sample_config: dict) -> None:
        """Test getting stage log paths."""
        ctx = run_manager.initialize_run(sample_config, run_id="test_stage_log")

        assert ctx.stage_log("data_qc") == ctx.run_dir / "logs" / "data_qc.log"
        assert ctx.stage_log("reference") == ctx.run_dir / "logs" / "reference.log"

    def test_property_paths(self, run_manager: RunManager, sample_config: dict) -> None:
        """Test property paths."""
        ctx = run_manager.initialize_run(sample_config, run_id="test_paths")

        assert ctx.config_dir == ctx.run_dir / "config"
        assert ctx.logs_dir == ctx.run_dir / "logs"
        assert ctx.manifests_dir == ctx.run_dir / "manifests"
        assert ctx.metadata_path == ctx.run_dir / "config" / "run_metadata.yaml"
        assert ctx.master_manifest_path == ctx.run_dir / "manifests" / "master_manifest.json"
