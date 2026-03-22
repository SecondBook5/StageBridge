"""
Tests for spatial backend visualization module.

These are smoke tests to verify visualizations can be generated without errors.
"""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for testing
import matplotlib.pyplot as plt

from stagebridge.spatial_backends.visualize import (
    plot_spatial_maps_comparison,
    plot_metrics_comparison,
    plot_confidence_distributions,
    plot_donor_robustness,
    plot_entropy_comparison,
    create_comparison_summary_figure,
)
from stagebridge.spatial_backends.comparison import ComparisonResult


class TestPlotSpatialMapsComparison:
    """Tests for spatial maps comparison plot."""

    def test_basic_plot(self, synthetic_standardized_output, synthetic_spatial):
        """Test basic spatial maps plot generation."""
        results = {
            "tangram": synthetic_standardized_output,
            "destvi": synthetic_standardized_output,
        }
        coords = synthetic_spatial.obsm["spatial"]

        fig = plot_spatial_maps_comparison(
            results=results,
            spatial_coords=coords,
            n_types_per_backend=2,
        )

        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_with_specific_types(self, synthetic_standardized_output, synthetic_spatial):
        """Test plot with specific cell types."""
        results = {"test": synthetic_standardized_output}
        coords = synthetic_spatial.obsm["spatial"]
        cell_types = synthetic_standardized_output.cell_type_proportions.columns[:2].tolist()

        fig = plot_spatial_maps_comparison(
            results=results,
            spatial_coords=coords,
            cell_types_to_show=cell_types,
        )

        assert fig is not None
        plt.close(fig)

    def test_plot_save_to_file(
        self, synthetic_standardized_output, synthetic_spatial, tmp_output_dir
    ):
        """Test saving plot to file."""
        results = {"test": synthetic_standardized_output}
        coords = synthetic_spatial.obsm["spatial"]
        output_path = tmp_output_dir / "test_spatial_maps.png"

        fig = plot_spatial_maps_comparison(
            results=results,
            spatial_coords=coords,
            n_types_per_backend=2,
            output_path=output_path,
        )

        assert output_path.exists()
        plt.close(fig)


class TestPlotMetricsComparison:
    """Tests for metrics comparison plot."""

    def test_basic_plot(self, synthetic_comparison_table):
        """Test basic metrics comparison plot."""
        fig = plot_metrics_comparison(
            comparison_table=synthetic_comparison_table,
        )

        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_specific_metrics(self, synthetic_comparison_table):
        """Test plot with specific metrics."""
        metrics = ["upstream_mean_entropy", "downstream_overall_utility"]

        fig = plot_metrics_comparison(
            comparison_table=synthetic_comparison_table,
            metrics_to_show=metrics,
        )

        assert fig is not None
        plt.close(fig)

    def test_plot_with_failed_backends(self):
        """Test plot handles failed backends gracefully."""
        table = pd.DataFrame(
            {
                "backend": ["tangram", "destvi"],
                "success": [True, False],
                "runtime_seconds": [10.0, 0.0],
                "upstream_coverage": [0.8, np.nan],
            }
        )

        fig = plot_metrics_comparison(comparison_table=table)

        assert fig is not None
        plt.close(fig)

    def test_plot_save_to_file(self, synthetic_comparison_table, tmp_output_dir):
        """Test saving metrics plot to file."""
        output_path = tmp_output_dir / "test_metrics.png"

        fig = plot_metrics_comparison(
            comparison_table=synthetic_comparison_table,
            output_path=output_path,
        )

        assert output_path.exists()
        plt.close(fig)


class TestPlotConfidenceDistributions:
    """Tests for confidence distribution plot."""

    def test_basic_plot(self, synthetic_standardized_output):
        """Test basic confidence distribution plot."""
        results = {
            "tangram": synthetic_standardized_output,
            "destvi": synthetic_standardized_output,
            "tacco": synthetic_standardized_output,
        }

        fig = plot_confidence_distributions(results=results)

        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_save_to_file(self, synthetic_standardized_output, tmp_output_dir):
        """Test saving confidence plot to file."""
        results = {"test": synthetic_standardized_output}
        output_path = tmp_output_dir / "test_confidence.png"

        fig = plot_confidence_distributions(
            results=results,
            output_path=output_path,
        )

        assert output_path.exists()
        plt.close(fig)


class TestPlotDonorRobustness:
    """Tests for donor robustness plot."""

    def test_basic_plot(self):
        """Test basic robustness plot."""
        robustness = {
            "tangram": {
                "donor_consistency": 0.85,
                "celltype_stability": 0.78,
                "confidence_stability": 0.82,
            },
            "destvi": {
                "donor_consistency": 0.80,
                "celltype_stability": 0.75,
                "confidence_stability": 0.79,
            },
        }

        fig = plot_donor_robustness(robustness_by_backend=robustness)

        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_empty_robustness(self):
        """Test handling of empty robustness data."""
        fig = plot_donor_robustness(robustness_by_backend={})

        assert fig is not None
        plt.close(fig)

    def test_plot_save_to_file(self, tmp_output_dir):
        """Test saving robustness plot to file."""
        robustness = {
            "test": {
                "donor_consistency": 0.8,
                "celltype_stability": 0.75,
            },
        }
        output_path = tmp_output_dir / "test_robustness.png"

        fig = plot_donor_robustness(
            robustness_by_backend=robustness,
            output_path=output_path,
        )

        assert output_path.exists()
        plt.close(fig)


class TestPlotEntropyComparison:
    """Tests for entropy comparison plot."""

    def test_basic_plot(self, synthetic_standardized_output, synthetic_spatial):
        """Test basic entropy comparison plot."""
        results = {
            "tangram": synthetic_standardized_output,
            "destvi": synthetic_standardized_output,
        }
        coords = synthetic_spatial.obsm["spatial"]

        fig = plot_entropy_comparison(
            results=results,
            spatial_coords=coords,
        )

        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestCreateComparisonSummaryFigure:
    """Tests for comprehensive summary figure."""

    def test_basic_summary(
        self, synthetic_comparison_table, synthetic_standardized_output, synthetic_spatial
    ):
        """Test basic summary figure creation."""
        comparison = ComparisonResult(
            comparison_table=synthetic_comparison_table,
            rankings={
                "overall": ["tacco", "tangram", "destvi"],
                "upstream": ["tangram", "tacco", "destvi"],
                "downstream": ["tacco", "destvi", "tangram"],
            },
        )

        results = {
            "tangram": synthetic_standardized_output,
            "destvi": synthetic_standardized_output,
            "tacco": synthetic_standardized_output,
        }
        coords = synthetic_spatial.obsm["spatial"]

        fig = create_comparison_summary_figure(
            comparison_result=comparison,
            results=results,
            spatial_coords=coords,
        )

        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_summary_save_to_file(
        self,
        synthetic_comparison_table,
        synthetic_standardized_output,
        synthetic_spatial,
        tmp_output_dir,
    ):
        """Test saving summary figure to file."""
        comparison = ComparisonResult(
            comparison_table=synthetic_comparison_table,
            rankings={"overall": ["tacco", "tangram", "destvi"]},
        )

        results = {"tangram": synthetic_standardized_output}
        coords = synthetic_spatial.obsm["spatial"]
        output_path = tmp_output_dir / "test_summary.png"

        fig = create_comparison_summary_figure(
            comparison_result=comparison,
            results=results,
            spatial_coords=coords,
            output_path=output_path,
        )

        assert output_path.exists()
        plt.close(fig)
