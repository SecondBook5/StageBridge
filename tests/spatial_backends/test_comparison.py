"""
Tests for spatial backend comparison module.
"""

import pandas as pd

from stagebridge.spatial_backends.comparison import (
    BackendRunResult,
    ComparisonResult,
    build_comparison_table,
    rank_backends,
)
from stagebridge.spatial_backends.metrics import MetricsReport


class TestBackendRunResult:
    """Tests for BackendRunResult dataclass."""

    def test_success_result(self, synthetic_mapping_result, synthetic_standardized_output):
        """Test successful run result."""
        metrics = MetricsReport(
            backend_name="tangram",
            upstream_metrics={"coverage": 0.8},
        )

        result = BackendRunResult(
            backend_name="tangram",
            success=True,
            result=synthetic_mapping_result,
            standardized=synthetic_standardized_output,
            metrics=metrics,
            runtime_seconds=10.5,
        )

        assert result.success
        assert result.error is None
        assert result.runtime_seconds == 10.5

    def test_failed_result(self):
        """Test failed run result."""
        result = BackendRunResult(
            backend_name="tangram",
            success=False,
            error="Test error",
            traceback="Traceback here",
            runtime_seconds=1.0,
        )

        assert not result.success
        assert result.error == "Test error"
        assert result.result is None


class TestComparisonResult:
    """Tests for ComparisonResult dataclass."""

    def test_get_successful_backends(self, synthetic_comparison_table):
        """Test getting successful backend list."""
        comparison = ComparisonResult(
            comparison_table=synthetic_comparison_table,
        )

        # Mock results
        comparison.results = {
            "tangram": BackendRunResult("tangram", True),
            "destvi": BackendRunResult("destvi", True),
            "tacco": BackendRunResult("tacco", False, error="failed"),
        }

        successful = comparison.get_successful_backends()

        assert "tangram" in successful
        assert "destvi" in successful
        assert "tacco" not in successful

    def test_get_failed_backends(self):
        """Test getting failed backend list."""
        comparison = ComparisonResult()
        comparison.results = {
            "tangram": BackendRunResult("tangram", False, error="error1"),
            "destvi": BackendRunResult("destvi", True),
        }

        failed = comparison.get_failed_backends()

        assert "tangram" in failed
        assert "destvi" not in failed

    def test_save_load(self, synthetic_comparison_table, tmp_output_dir):
        """Test save and load round-trip."""
        comparison = ComparisonResult(
            comparison_table=synthetic_comparison_table,
            rankings={"overall": ["tacco", "tangram", "destvi"]},
            metadata={"test_key": "test_value"},
        )

        # Save
        comparison.save(tmp_output_dir)

        # Load
        loaded = ComparisonResult.load(tmp_output_dir)

        assert loaded.comparison_table is not None
        assert len(loaded.comparison_table) == 3
        assert loaded.rankings["overall"] == ["tacco", "tangram", "destvi"]


class TestBuildComparisonTable:
    """Tests for comparison table building."""

    def test_build_from_results(self, synthetic_mapping_result, synthetic_standardized_output):
        """Test building comparison table from results."""
        metrics_tangram = MetricsReport(
            backend_name="tangram",
            upstream_metrics={"mean_entropy": 0.5},
            downstream_metrics={"overall_utility": 0.7},
        )
        metrics_destvi = MetricsReport(
            backend_name="destvi",
            upstream_metrics={"mean_entropy": 0.6},
            downstream_metrics={"overall_utility": 0.65},
        )

        results = {
            "tangram": BackendRunResult(
                backend_name="tangram",
                success=True,
                result=synthetic_mapping_result,
                standardized=synthetic_standardized_output,
                metrics=metrics_tangram,
                runtime_seconds=10.0,
            ),
            "destvi": BackendRunResult(
                backend_name="destvi",
                success=True,
                result=synthetic_mapping_result,
                standardized=synthetic_standardized_output,
                metrics=metrics_destvi,
                runtime_seconds=20.0,
            ),
        }

        table = build_comparison_table(results)

        assert isinstance(table, pd.DataFrame)
        assert len(table) == 2
        assert "backend" in table.columns
        assert "success" in table.columns
        assert "runtime_seconds" in table.columns

    def test_build_with_failed_backend(self, synthetic_mapping_result):
        """Test building table with failed backends."""
        results = {
            "tangram": BackendRunResult("tangram", True, runtime_seconds=10.0),
            "destvi": BackendRunResult(
                "destvi",
                False,
                error="Import error",
                runtime_seconds=0.5,
            ),
        }

        table = build_comparison_table(results)

        assert len(table) == 2
        assert not table[table["backend"] == "destvi"]["success"].iloc[0]


class TestRankBackends:
    """Tests for backend ranking."""

    def test_overall_ranking(self, synthetic_comparison_table):
        """Test overall ranking computation."""
        rankings = rank_backends(synthetic_comparison_table)

        assert "overall" in rankings
        assert "upstream" in rankings
        assert "downstream" in rankings
        assert "spatial" in rankings
        assert "runtime" in rankings

        # Each ranking should have all successful backends
        assert len(rankings["overall"]) == 3

    def test_custom_weights(self, synthetic_comparison_table):
        """Test ranking with custom weights."""
        # Weight heavily toward downstream
        weights = {
            "upstream": 0.0,
            "downstream": 1.0,
            "spatial": 0.0,
            "runtime": 0.0,
        }

        rankings = rank_backends(synthetic_comparison_table, weights=weights)

        # tacco has highest downstream score (0.75)
        assert rankings["downstream"][0] == "tacco"

    def test_empty_table(self):
        """Test ranking with no successful backends."""
        empty_table = pd.DataFrame(
            {
                "backend": ["tangram"],
                "success": [False],
                "runtime_seconds": [0.0],
            }
        )

        rankings = rank_backends(empty_table)

        assert rankings["overall"] == []

    def test_runtime_ranking(self, synthetic_comparison_table):
        """Test runtime ranking (lower is better)."""
        rankings = rank_backends(synthetic_comparison_table)

        # tangram has lowest runtime (10.5)
        assert rankings["runtime"][0] == "tangram"
