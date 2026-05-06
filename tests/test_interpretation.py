"""Tests for the interpretation module.

Tests that interpretation analysis modules can be imported and have expected interfaces.
Full functional tests require a trained model, so these are primarily smoke tests.
"""

import pytest
import torch
import numpy as np


class TestInterpretationImports:
    """Test that all interpretation module components can be imported."""

    def test_import_ablation_module(self):
        from stagebridge.interpretation import AblationModule, compute_token_ablation
        assert AblationModule is not None
        assert compute_token_ablation is not None

    def test_import_attention_module(self):
        from stagebridge.interpretation import AttentionModule, extract_attention_patterns
        assert AttentionModule is not None
        assert extract_attention_patterns is not None

    def test_import_network_module(self):
        from stagebridge.interpretation import InteractionNetwork, build_interaction_network
        assert InteractionNetwork is not None
        assert build_interaction_network is not None

    def test_import_plotting_functions(self):
        from stagebridge.interpretation import (
            plot_interaction_network,
            plot_interaction_heatmap,
            plot_ring_attention_decay,
            plot_ablation_importance,
            plot_reference_balance,
            plot_stage_network_comparison,
        )
        assert all(f is not None for f in [
            plot_interaction_network,
            plot_interaction_heatmap,
            plot_ring_attention_decay,
            plot_ablation_importance,
            plot_reference_balance,
            plot_stage_network_comparison,
        ])

    def test_import_dynamics_module(self):
        from stagebridge.interpretation import (
            TrajectoryAnalysis,
            FateProbability,
            DynamicDriverResult,
            cluster_driver_genes,
        )
        assert TrajectoryAnalysis is not None
        assert FateProbability is not None

    def test_import_trajectory_plots(self):
        from stagebridge.interpretation import (
            plot_temporal_evolution,
            plot_fate_probability,
            plot_single_cell_trajectories,
            plot_driver_heatmap,
            plot_gene_dynamics,
            create_trajectory_animation,
        )
        assert plot_temporal_evolution is not None
        assert plot_fate_probability is not None

    def test_import_manifold_viz(self):
        from stagebridge.interpretation import (
            plot_manifold_comparison,
            plot_multi_method_comparison,
            plot_trajectory_straightness,
            plot_geodesic_comparison,
            plot_phase_map,
            plot_phase_portrait_grid,
            compute_manifold_comparison,
            ManifoldComparisonResult,
        )
        assert plot_manifold_comparison is not None
        assert ManifoldComparisonResult is not None


class TestAblationModuleInterface:
    """Test AblationModule interface."""

    def test_ablation_module_has_compute_method(self):
        from stagebridge.interpretation import AblationModule
        assert hasattr(AblationModule, "compute")
        assert callable(getattr(AblationModule, "compute"))

    def test_ablation_module_has_results_attribute(self):
        from stagebridge.interpretation import AblationModule
        module = AblationModule()
        assert hasattr(module, "results")
        assert hasattr(module, "per_sample_losses")
        assert hasattr(module, "stage_breakdown")


class TestAttentionModuleInterface:
    """Test AttentionModule interface."""

    def test_attention_module_has_compute_method(self):
        from stagebridge.interpretation import AttentionModule
        assert hasattr(AttentionModule, "compute")
        assert callable(getattr(AttentionModule, "compute"))

    def test_attention_module_has_dataframes(self):
        from stagebridge.interpretation import AttentionModule
        module = AttentionModule()
        assert hasattr(module, "attention_df")
        assert hasattr(module, "empty_attention_df")
        assert hasattr(module, "summary_stats")


class TestInteractionNetworkInterface:
    """Test InteractionNetwork interface."""

    def test_interaction_network_dataclass(self):
        from stagebridge.interpretation.networks import InteractionNetwork
        # Should be able to instantiate with default values
        assert InteractionNetwork is not None


class TestManifoldComparisonInterface:
    """Test ManifoldComparisonResult interface."""

    def test_manifold_comparison_result_dataclass(self):
        from stagebridge.interpretation import ManifoldComparisonResult
        assert ManifoldComparisonResult is not None


class TestInterpretationScript:
    """Test the interpretation CLI script."""

    def test_script_exists(self):
        from pathlib import Path
        script_path = Path(__file__).parent.parent / "scripts" / "run_interpretation.py"
        assert script_path.exists(), f"Script not found at {script_path}"

    def test_script_imports(self):
        """Test that the script can be imported without errors."""
        import importlib.util
        from pathlib import Path

        script_path = Path(__file__).parent.parent / "scripts" / "run_interpretation.py"
        spec = importlib.util.spec_from_file_location("run_interpretation", script_path)
        module = importlib.util.module_from_spec(spec)

        # This should not raise
        try:
            spec.loader.exec_module(module)
        except SystemExit:
            # Script may call sys.exit() when run without args, that's OK
            pass

        assert hasattr(module, "main")
        assert hasattr(module, "run_ablation_analysis")
        assert hasattr(module, "run_attention_analysis")
        assert hasattr(module, "run_network_analysis")
        assert hasattr(module, "run_trajectory_analysis")
        assert hasattr(module, "run_manifold_analysis")


class TestTokenAblationLogic:
    """Test token ablation computation logic."""

    def test_ablation_result_dataclass(self):
        from stagebridge.interpretation.ablation import AblationResult

        result = AblationResult(
            token_name="ring1",
            baseline_loss=0.1,
            ablated_loss=0.15,
            delta_loss=0.05,
            relative_importance=0.5,
            p_value=0.01,
            n_samples=100,
        )

        assert result.token_name == "ring1"
        assert result.delta_loss == pytest.approx(0.05)
        assert result.relative_importance == pytest.approx(0.5)

    def test_ablation_delta_positive_means_important(self):
        """Higher ablated loss than baseline means token was important."""
        from stagebridge.interpretation.ablation import AblationResult

        # Token is important if removing it increases loss
        important = AblationResult(
            token_name="hlca",
            baseline_loss=0.1,
            ablated_loss=0.2,  # Loss increased
            delta_loss=0.1,
            relative_importance=1.0,
            p_value=0.001,
            n_samples=100,
        )
        assert important.delta_loss > 0
        assert important.relative_importance > 0

        # Token is not important if removing it doesn't change loss
        unimportant = AblationResult(
            token_name="stats",
            baseline_loss=0.1,
            ablated_loss=0.1,  # Loss unchanged
            delta_loss=0.0,
            relative_importance=0.0,
            p_value=0.5,
            n_samples=100,
        )
        assert unimportant.delta_loss == 0


class TestPlottingFunctionSignatures:
    """Test that plotting functions have expected signatures."""

    def test_plot_ablation_importance_accepts_save_path(self):
        from stagebridge.interpretation import plot_ablation_importance
        import inspect
        sig = inspect.signature(plot_ablation_importance)
        params = list(sig.parameters.keys())
        assert "save_path" in params or any("path" in p.lower() for p in params)

    def test_plot_ring_attention_decay_accepts_save_path(self):
        from stagebridge.interpretation import plot_ring_attention_decay
        import inspect
        sig = inspect.signature(plot_ring_attention_decay)
        params = list(sig.parameters.keys())
        assert "save_path" in params or any("path" in p.lower() for p in params)

    def test_plot_interaction_network_accepts_save_path(self):
        from stagebridge.interpretation import plot_interaction_network
        import inspect
        sig = inspect.signature(plot_interaction_network)
        params = list(sig.parameters.keys())
        assert "save_path" in params or any("path" in p.lower() for p in params)
