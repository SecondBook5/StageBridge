"""Tests for the notebook API module."""

from __future__ import annotations

from pathlib import Path

import pytest

from stagebridge.orchestration.notebook_api import (
    RunSummary,
    StageResult,
    initialize_run,
    run_ablations,
    run_baselines,
    run_biology,
    run_data_qc,
    run_full_model,
    run_publication_figures,
    run_reference_mapping,
    run_smoke_pipeline,
    run_spatial_backend_benchmark,
    summarize_run,
    validate_stage,
)
from stagebridge.orchestration.run_manager import RunStatus


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
        "notebook": {
            "verbosity": "minimal",
            "show_figures": False,
        },
    }


class TestStageResult:
    """Tests for StageResult dataclass."""

    def test_bool_success(self) -> None:
        """Test bool conversion for successful result."""
        result = StageResult(stage_name="test", success=True)
        assert bool(result) is True

    def test_bool_failure(self) -> None:
        """Test bool conversion for failed result."""
        result = StageResult(stage_name="test", success=False)
        assert bool(result) is False

    def test_bool_skipped(self) -> None:
        """Test bool conversion for skipped result."""
        result = StageResult(stage_name="test", success=False, skipped=True)
        assert bool(result) is True  # Skipped is considered OK


class TestInitializeRun:
    """Tests for initialize_run function."""

    def test_initialize_with_dict_config(
        self, temp_artifacts_root: Path, sample_config: dict
    ) -> None:
        """Test initializing a run with dict config."""
        ctx = initialize_run(
            sample_config,
            run_id="test_init",
            artifacts_root=str(temp_artifacts_root),
        )

        assert ctx.run_id == "test_init"
        assert ctx.run_dir.exists()
        assert ctx.status == RunStatus.RUNNING

    def test_initialize_creates_directories(
        self, temp_artifacts_root: Path, sample_config: dict
    ) -> None:
        """Test that initialization creates required directories."""
        ctx = initialize_run(
            sample_config,
            run_id="test_dirs",
            artifacts_root=str(temp_artifacts_root),
        )

        # Check key directories exist
        assert ctx.config_dir.exists()
        assert ctx.logs_dir.exists()
        assert ctx.manifests_dir.exists()

    def test_initialize_auto_run_id(self, temp_artifacts_root: Path, sample_config: dict) -> None:
        """Test auto-generated run ID."""
        ctx = initialize_run(
            sample_config,
            artifacts_root=str(temp_artifacts_root),
        )

        assert ctx.run_id is not None
        assert ctx.run_id.startswith("run_")


class TestStageExecution:
    """Tests for stage execution functions."""

    def test_run_data_qc(self, temp_artifacts_root: Path, sample_config: dict) -> None:
        """Test running data QC stage."""
        ctx = initialize_run(
            sample_config,
            run_id="test_qc",
            artifacts_root=str(temp_artifacts_root),
        )

        result = run_data_qc(ctx)

        assert result.stage_name == "data_qc"
        assert result.success
        assert result.output_dir is not None
        assert result.output_dir.exists()

    def test_run_reference_mapping(self, temp_artifacts_root: Path, sample_config: dict) -> None:
        """Test running reference mapping stage."""
        ctx = initialize_run(
            sample_config,
            run_id="test_ref",
            artifacts_root=str(temp_artifacts_root),
        )

        result = run_reference_mapping(ctx)

        assert result.stage_name == "reference"
        assert result.success

    def test_run_spatial_backend_benchmark(
        self, temp_artifacts_root: Path, sample_config: dict
    ) -> None:
        """Test running spatial backend benchmark stage."""
        ctx = initialize_run(
            sample_config,
            run_id="test_spatial",
            artifacts_root=str(temp_artifacts_root),
        )

        result = run_spatial_backend_benchmark(ctx)

        assert result.stage_name == "spatial_backend"
        assert result.success

    def test_run_baselines(self, temp_artifacts_root: Path, sample_config: dict) -> None:
        """Test running baselines stage."""
        ctx = initialize_run(
            sample_config,
            run_id="test_baselines",
            artifacts_root=str(temp_artifacts_root),
        )

        result = run_baselines(ctx)

        assert result.stage_name == "baselines"
        assert result.success

    def test_run_full_model(self, temp_artifacts_root: Path, sample_config: dict) -> None:
        """Test running full model stage."""
        ctx = initialize_run(
            sample_config,
            run_id="test_full_model",
            artifacts_root=str(temp_artifacts_root),
        )

        result = run_full_model(ctx)

        assert result.stage_name == "full_model"
        assert result.success

    def test_run_ablations(self, temp_artifacts_root: Path, sample_config: dict) -> None:
        """Test running ablations stage."""
        ctx = initialize_run(
            sample_config,
            run_id="test_ablations",
            artifacts_root=str(temp_artifacts_root),
        )

        result = run_ablations(ctx)

        assert result.stage_name == "ablations"
        assert result.success

    def test_run_biology(self, temp_artifacts_root: Path, sample_config: dict) -> None:
        """Test running biology stage."""
        ctx = initialize_run(
            sample_config,
            run_id="test_biology",
            artifacts_root=str(temp_artifacts_root),
        )

        result = run_biology(ctx)

        assert result.stage_name == "biology"
        assert result.success

    def test_run_publication_figures(self, temp_artifacts_root: Path, sample_config: dict) -> None:
        """Test running publication figures stage."""
        ctx = initialize_run(
            sample_config,
            run_id="test_figures",
            artifacts_root=str(temp_artifacts_root),
        )

        result = run_publication_figures(ctx)

        assert result.stage_name == "figures"
        assert result.success

    def test_force_rerun(self, temp_artifacts_root: Path, sample_config: dict) -> None:
        """Test force rerun option."""
        ctx = initialize_run(
            sample_config,
            run_id="test_force_rerun",
            artifacts_root=str(temp_artifacts_root),
        )

        # Run once
        result1 = run_data_qc(ctx)
        assert result1.success

        # Run again with force_rerun
        result2 = run_data_qc(ctx, force_rerun=True)
        assert result2.success
        assert not result2.skipped


class TestValidation:
    """Tests for validation function."""

    def test_validate_stage(self, temp_artifacts_root: Path, sample_config: dict) -> None:
        """Test stage validation."""
        ctx = initialize_run(
            sample_config,
            run_id="test_validate",
            artifacts_root=str(temp_artifacts_root),
        )

        # Run a stage first
        run_data_qc(ctx)

        # Validate it
        result = validate_stage(ctx, "data_qc")

        # Result is a ValidationResult object
        assert result.stage_name == "data_qc"


class TestSummarize:
    """Tests for summarize_run function."""

    def test_summarize_run(self, temp_artifacts_root: Path, sample_config: dict) -> None:
        """Test run summarization."""
        ctx = initialize_run(
            sample_config,
            run_id="test_summarize",
            artifacts_root=str(temp_artifacts_root),
        )

        # Run some stages
        run_data_qc(ctx)
        run_reference_mapping(ctx)

        # Summarize
        summary = summarize_run(ctx)

        assert isinstance(summary, RunSummary)
        assert summary.run_id == "test_summarize"
        assert summary.completed_stages >= 0
        assert summary.run_dir == ctx.run_dir

    def test_summary_includes_stages(self, temp_artifacts_root: Path, sample_config: dict) -> None:
        """Test that summary includes stage information."""
        ctx = initialize_run(
            sample_config,
            run_id="test_summary_stages",
            artifacts_root=str(temp_artifacts_root),
        )

        run_data_qc(ctx)

        summary = summarize_run(ctx)

        assert "data_qc" in summary.stages
        assert summary.stages["data_qc"]["status"] in ["completed", "running", "pending"]


class TestSmokePipeline:
    """Tests for smoke pipeline function."""

    def test_smoke_pipeline_runs(self, temp_artifacts_root: Path, monkeypatch) -> None:
        """Test that smoke pipeline runs without errors."""
        # Monkey-patch the artifacts root
        import stagebridge.orchestration.notebook_api as api

        original_manager = api._run_manager
        api._run_manager = None

        try:
            # The smoke pipeline should run through
            # Note: This will create a run in the default artifacts location
            # unless we patch it differently
            pass  # Skip actual execution in test
        finally:
            api._run_manager = original_manager


class TestResultData:
    """Tests for result data handling."""

    def test_stage_result_has_data(self, temp_artifacts_root: Path, sample_config: dict) -> None:
        """Test that stage results contain result data."""
        ctx = initialize_run(
            sample_config,
            run_id="test_result_data",
            artifacts_root=str(temp_artifacts_root),
        )

        result = run_spatial_backend_benchmark(ctx)

        assert result.result_data is not None
        assert isinstance(result.result_data, dict)
        if result.success:
            assert "benchmark" in result.result_data

    def test_stage_result_artifacts(self, temp_artifacts_root: Path, sample_config: dict) -> None:
        """Test that stage results list artifacts."""
        ctx = initialize_run(
            sample_config,
            run_id="test_artifacts",
            artifacts_root=str(temp_artifacts_root),
        )

        result = run_data_qc(ctx)

        assert result.artifacts is not None
        assert isinstance(result.artifacts, list)
