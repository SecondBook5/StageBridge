"""
Tests for spatial backend metrics module.
"""

import numpy as np
import pandas as pd

from stagebridge.spatial_backends.metrics import (
    MetricsReport,
    compute_upstream_metrics,
    compute_downstream_utility,
    compute_spatial_coherence,
    compute_donor_robustness,
    compute_comprehensive_metrics,
)


class TestMetricsReport:
    """Tests for MetricsReport dataclass."""

    def test_to_dict(self):
        """Test conversion to flat dictionary."""
        report = MetricsReport(
            backend_name="tangram",
            upstream_metrics={"mean_entropy": 0.5, "coverage": 0.8},
            downstream_metrics={"overall_utility": 0.7},
            spatial_metrics={"local_coherence": 0.6},
        )

        d = report.to_dict()

        assert d["backend"] == "tangram"
        assert d["upstream_mean_entropy"] == 0.5
        assert d["upstream_coverage"] == 0.8
        assert d["downstream_overall_utility"] == 0.7
        assert d["spatial_local_coherence"] == 0.6

    def test_get_summary_score(self):
        """Test weighted summary score computation."""
        report = MetricsReport(
            backend_name="tangram",
            upstream_metrics={"mean_entropy": 0.5, "coverage": 0.8},
            downstream_metrics={"overall_utility": 0.7, "confidence_quality": 0.6},
            spatial_metrics={"local_coherence": 0.6, "smoothness": 0.5},
        )

        score = report.get_summary_score()

        # Score should be in reasonable range
        assert 0 <= score <= 1

    def test_get_summary_score_custom_weights(self):
        """Test summary score with custom weights."""
        report = MetricsReport(
            backend_name="tangram",
            upstream_metrics={"metric1": 1.0},
            downstream_metrics={"metric1": 0.0},
        )

        # With all weight on upstream
        score_upstream = report.get_summary_score({"upstream": 1.0, "downstream": 0.0})

        # With all weight on downstream
        score_downstream = report.get_summary_score({"upstream": 0.0, "downstream": 1.0})

        assert score_upstream > score_downstream


class TestComputeUpstreamMetrics:
    """Tests for upstream metrics computation."""

    def test_basic_metrics(self, synthetic_mapping_result):
        """Test basic upstream metric computation."""
        metrics = compute_upstream_metrics(synthetic_mapping_result)

        assert "mean_entropy" in metrics
        assert "std_entropy" in metrics
        assert "sparsity" in metrics
        assert "coverage" in metrics
        assert "max_proportion_mean" in metrics
        assert "n_spots" in metrics
        assert "n_celltypes" in metrics

    def test_entropy_range(self, synthetic_mapping_result):
        """Test entropy is in valid range."""
        metrics = compute_upstream_metrics(synthetic_mapping_result)

        assert 0 <= metrics["mean_entropy"] <= 1
        assert metrics["std_entropy"] >= 0

    def test_sparsity_range(self, synthetic_mapping_result):
        """Test sparsity is in valid range."""
        metrics = compute_upstream_metrics(synthetic_mapping_result)

        assert 0 <= metrics["sparsity"] <= 1

    def test_coverage_range(self, synthetic_mapping_result):
        """Test coverage is in valid range."""
        metrics = compute_upstream_metrics(synthetic_mapping_result)

        assert 0 <= metrics["coverage"] <= 1


class TestComputeDownstreamUtility:
    """Tests for downstream utility computation."""

    def test_basic_utility(self, synthetic_mapping_result):
        """Test basic downstream utility computation."""
        metrics = compute_downstream_utility(synthetic_mapping_result)

        assert "proportion_stability" in metrics
        assert "celltype_coverage" in metrics
        assert "confidence_mean" in metrics
        assert "confidence_quality" in metrics
        assert "entropy_quality" in metrics
        assert "overall_utility" in metrics

    def test_utility_ranges(self, synthetic_mapping_result):
        """Test utility metrics are in valid ranges."""
        metrics = compute_downstream_utility(synthetic_mapping_result)

        # Most metrics should be in [0, 1] or close
        assert 0 <= metrics["confidence_mean"] <= 1
        assert 0 <= metrics["celltype_coverage"] <= 1
        assert 0 <= metrics["overall_utility"] <= 1

    def test_with_transition_data(self, synthetic_mapping_result):
        """Test utility with transition data."""
        transition_data = {
            "source_types": ["CellType_0", "CellType_1"],
            "target_types": ["CellType_2", "CellType_3"],
            "known_transitions": [("CellType_0", "CellType_2")],
        }

        metrics = compute_downstream_utility(
            synthetic_mapping_result,
            transition_data=transition_data,
        )

        assert "source_type_coverage" in metrics
        assert "target_type_coverage" in metrics


class TestComputeSpatialCoherence:
    """Tests for spatial coherence computation."""

    def test_basic_coherence(self, synthetic_mapping_result, synthetic_spatial):
        """Test basic spatial coherence computation."""
        coords = synthetic_spatial.obsm["spatial"]

        metrics = compute_spatial_coherence(
            synthetic_mapping_result,
            coords,
        )

        assert "local_coherence" in metrics
        assert "spatial_smoothness" in metrics
        assert "spatial_autocorrelation" in metrics
        assert "niche_coherence" in metrics

    def test_coherence_ranges(self, synthetic_mapping_result, synthetic_spatial):
        """Test coherence metrics are in valid ranges."""
        coords = synthetic_spatial.obsm["spatial"]

        metrics = compute_spatial_coherence(
            synthetic_mapping_result,
            coords,
        )

        # Autocorrelation should be in [0, 1]
        assert 0 <= metrics["spatial_autocorrelation"] <= 1

    def test_few_spots_handling(self, synthetic_snrna):
        """Test handling of very few spots."""
        from stagebridge.spatial_backends.base import BackendMappingResult

        # Create tiny dataset
        n_spots = 3
        cell_types = synthetic_snrna.obs["cell_type"].cat.categories.tolist()

        props = pd.DataFrame(
            np.ones((n_spots, len(cell_types))) / len(cell_types),
            index=[f"spot_{i}" for i in range(n_spots)],
            columns=cell_types,
        )

        result = BackendMappingResult(
            cell_type_proportions=props,
            confidence=pd.Series(np.ones(n_spots), index=props.index),
            upstream_metrics={},
            metadata={},
        )

        coords = np.random.rand(n_spots, 2)

        metrics = compute_spatial_coherence(result, coords, k_neighbors=2)

        # Should handle gracefully
        assert isinstance(metrics, dict)


class TestComputeDonorRobustness:
    """Tests for donor robustness computation."""

    def test_basic_robustness(self, synthetic_mapping_result):
        """Test basic robustness computation with multiple donors."""
        # Create results for multiple donors (reusing same structure)
        results_by_donor = {
            "donor_1": synthetic_mapping_result,
            "donor_2": synthetic_mapping_result,
            "donor_3": synthetic_mapping_result,
        }

        metrics = compute_donor_robustness(results_by_donor)

        assert "donor_consistency" in metrics
        assert "celltype_stability" in metrics
        assert "confidence_stability" in metrics
        assert "n_donors" in metrics

    def test_single_donor(self, synthetic_mapping_result):
        """Test robustness with single donor returns NaN."""
        results_by_donor = {"donor_1": synthetic_mapping_result}

        metrics = compute_donor_robustness(results_by_donor)

        assert np.isnan(metrics["donor_consistency"])

    def test_robustness_ranges(self, synthetic_mapping_result):
        """Test robustness metrics ranges."""
        results_by_donor = {
            "donor_1": synthetic_mapping_result,
            "donor_2": synthetic_mapping_result,
        }

        metrics = compute_donor_robustness(results_by_donor)

        # With identical results, should have high consistency
        assert metrics["donor_consistency"] > 0.9


class TestComputeComprehensiveMetrics:
    """Tests for comprehensive metrics computation."""

    def test_comprehensive_metrics(self, synthetic_mapping_result, synthetic_spatial):
        """Test comprehensive metrics report generation."""
        coords = synthetic_spatial.obsm["spatial"]

        report = compute_comprehensive_metrics(
            synthetic_mapping_result,
            spatial_coords=coords,
            runtime_seconds=10.5,
            memory_mb=512.0,
        )

        assert isinstance(report, MetricsReport)
        assert report.backend_name == "test"
        assert len(report.upstream_metrics) > 0
        assert len(report.downstream_metrics) > 0
        assert len(report.spatial_metrics) > 0
        assert report.runtime_metrics["runtime_seconds"] == 10.5
        assert report.runtime_metrics["memory_mb"] == 512.0
