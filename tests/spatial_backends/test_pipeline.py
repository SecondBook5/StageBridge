"""
Tests for spatial backend benchmark pipeline.

These tests focus on the pipeline infrastructure, not actual backend execution.
"""

import pytest

from stagebridge.spatial_backends.pipeline import (
    SpatialBenchmarkConfig,
    BenchmarkProgress,
    run_smoke_benchmark,
    load_benchmark_results,
    get_canonical_backend_result,
    _apply_smoke_mode,
    _initialize_backends,
)


class TestSpatialBenchmarkConfig:
    """Tests for SpatialBenchmarkConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SpatialBenchmarkConfig()

        assert config.backends_to_run == ["tangram", "destvi", "tacco"]
        assert config.required_backends == ["tangram", "destvi", "tacco"]
        assert config.smoke_mode is False
        assert config.random_seed == 42

    def test_custom_config(self):
        """Test custom configuration."""
        config = SpatialBenchmarkConfig(
            backends_to_run=["tangram"],
            smoke_mode=True,
            smoke_n_spots=100,
            random_seed=123,
        )

        assert config.backends_to_run == ["tangram"]
        assert config.smoke_mode is True
        assert config.smoke_n_spots == 100
        assert config.random_seed == 123

    def test_get_backend_config(self):
        """Test getting backend-specific config."""
        config = SpatialBenchmarkConfig(
            tangram_config={"n_epochs": 500},
            destvi_config={"n_latent": 20},
        )

        tangram_cfg = config.get_backend_config("tangram")
        destvi_cfg = config.get_backend_config("destvi")

        assert tangram_cfg["n_epochs"] == 500
        assert destvi_cfg["n_latent"] == 20

    def test_smoke_mode_modifies_config(self):
        """Test that smoke mode modifies backend configs."""
        config = SpatialBenchmarkConfig(
            smoke_mode=True,
            smoke_n_epochs=50,
        )

        tangram_cfg = config.get_backend_config("tangram")
        destvi_cfg = config.get_backend_config("destvi")

        assert tangram_cfg["n_epochs"] == 50
        assert destvi_cfg["n_epochs_condsc"] == 50

    def test_selection_weights(self):
        """Test selection weights configuration."""
        config = SpatialBenchmarkConfig(
            selection_weights={
                "upstream": 0.5,
                "downstream": 0.5,
            }
        )

        assert config.selection_weights["upstream"] == 0.5
        assert config.selection_weights["downstream"] == 0.5


class TestBenchmarkProgress:
    """Tests for BenchmarkProgress tracking."""

    def test_initial_state(self):
        """Test initial progress state."""
        progress = BenchmarkProgress(total_backends=3)

        assert progress.total_backends == 3
        assert progress.completed_backends == 0
        assert progress.status == "not_started"
        assert len(progress.errors) == 0

    def test_update(self):
        """Test progress update."""
        progress = BenchmarkProgress(total_backends=3)

        progress.update(backend="tangram", status="running")

        assert progress.current_backend == "tangram"
        assert progress.status == "running"

    def test_backend_complete(self):
        """Test marking backend complete."""
        progress = BenchmarkProgress(total_backends=3)

        progress.backend_complete("tangram", success=True)

        assert progress.completed_backends == 1
        assert len(progress.errors) == 0

    def test_backend_failed(self):
        """Test marking backend as failed."""
        progress = BenchmarkProgress(total_backends=3)

        progress.backend_complete("destvi", success=False)

        assert progress.completed_backends == 1
        assert "destvi failed" in progress.errors


class TestApplySmokeMode:
    """Tests for smoke mode data subsampling."""

    def test_subsample_cells(self, synthetic_snrna, synthetic_spatial):
        """Test cell subsampling in smoke mode."""
        snrna_sub, spatial_sub = _apply_smoke_mode(
            synthetic_snrna,
            synthetic_spatial,
            n_cells=100,
            n_spots=50,
            seed=42,
        )

        assert len(snrna_sub) == 100
        assert len(spatial_sub) == 50

    def test_no_subsample_if_smaller(self, synthetic_snrna, synthetic_spatial):
        """Test no subsampling if data already smaller."""
        snrna_sub, spatial_sub = _apply_smoke_mode(
            synthetic_snrna,
            synthetic_spatial,
            n_cells=10000,  # Larger than actual
            n_spots=10000,
            seed=42,
        )

        assert len(snrna_sub) == len(synthetic_snrna)
        assert len(spatial_sub) == len(synthetic_spatial)

    def test_reproducible_subsampling(self, synthetic_snrna, synthetic_spatial):
        """Test that subsampling is reproducible with seed."""
        snrna_1, _ = _apply_smoke_mode(
            synthetic_snrna,
            synthetic_spatial,
            n_cells=100,
            n_spots=50,
            seed=42,
        )

        snrna_2, _ = _apply_smoke_mode(
            synthetic_snrna,
            synthetic_spatial,
            n_cells=100,
            n_spots=50,
            seed=42,
        )

        # Same cells selected
        assert set(snrna_1.obs_names) == set(snrna_2.obs_names)


class TestInitializeBackends:
    """Tests for backend initialization."""

    def test_initialize_all_backends(self):
        """Test initializing all backends."""
        config = SpatialBenchmarkConfig()

        backends = _initialize_backends(config)

        assert "tangram" in backends
        assert "destvi" in backends
        assert "tacco" in backends

    def test_initialize_subset(self):
        """Test initializing subset of backends."""
        config = SpatialBenchmarkConfig(
            backends_to_run=["tangram"],
        )

        backends = _initialize_backends(config)

        assert "tangram" in backends
        assert "destvi" not in backends

    def test_unknown_backend_skipped(self):
        """Test unknown backends are skipped with warning."""
        config = SpatialBenchmarkConfig(
            backends_to_run=["tangram", "unknown_backend"],
        )

        backends = _initialize_backends(config)

        assert "tangram" in backends
        assert "unknown_backend" not in backends


class TestLoadBenchmarkResults:
    """Tests for loading saved benchmark results."""

    def test_load_results(self, tmp_output_dir, synthetic_comparison_table):
        """Test loading saved results."""
        from stagebridge.spatial_backends.comparison import ComparisonResult
        from stagebridge.spatial_backends.selection import (
            BackendSelection,
            save_canonical_decision,
        )

        # Create and save mock results
        comparison = ComparisonResult(
            comparison_table=synthetic_comparison_table,
            rankings={"overall": ["tacco"]},
        )
        comparison.save(tmp_output_dir)

        selection = BackendSelection(
            canonical_backend="tacco",
            selection_score=0.75,
            justification="Test",
        )
        save_canonical_decision(selection, tmp_output_dir)

        # Load results
        loaded_comparison, loaded_selection = load_benchmark_results(tmp_output_dir)

        assert loaded_comparison.comparison_table is not None
        assert loaded_selection.canonical_backend == "tacco"


class TestGetCanonicalBackendResult:
    """Tests for getting canonical backend result."""

    def test_get_canonical_result(self, tmp_output_dir, synthetic_standardized_output):
        """Test retrieving canonical backend's standardized output."""
        from stagebridge.spatial_backends.selection import (
            BackendSelection,
            save_canonical_decision,
        )

        # Save canonical decision
        selection = BackendSelection(
            canonical_backend="tangram",
            selection_score=0.8,
            justification="Test",
        )
        save_canonical_decision(selection, tmp_output_dir)

        # Save tangram result
        tangram_dir = tmp_output_dir / "tangram"
        synthetic_standardized_output.save(tangram_dir)

        # Get canonical result
        result = get_canonical_backend_result(tmp_output_dir)

        assert result.backend_name == synthetic_standardized_output.backend_name


# Integration test (marked slow as it would run actual backends)
@pytest.mark.skip(reason="Integration test requires actual backend packages")
class TestRunSpatialBenchmark:
    """Integration tests for full benchmark pipeline."""

    def test_smoke_benchmark(self, synthetic_snrna, synthetic_spatial, tmp_output_dir):
        """Test running smoke benchmark."""
        comparison, selection = run_smoke_benchmark(
            snrna=synthetic_snrna,
            spatial=synthetic_spatial,
            output_dir=tmp_output_dir,
        )

        assert comparison is not None
        assert selection is not None
        assert (tmp_output_dir / "canonical_backend.json").exists()
        assert (tmp_output_dir / "backend_selection_report.md").exists()
