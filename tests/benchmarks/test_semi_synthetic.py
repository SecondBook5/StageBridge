"""Tests for semi-synthetic benchmark generator."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from stagebridge.benchmarks.semi_synthetic.configs import (
    BenchmarkConfig,
    SmokeConfig,
    InteractionRule,
    CellGroupSpec,
)
from stagebridge.benchmarks.semi_synthetic.data_sources import (
    DataSourceLoader,
    create_fallback_data,
)
from stagebridge.benchmarks.semi_synthetic.feature_harmonization import (
    FeatureHarmonizer,
)
from stagebridge.benchmarks.semi_synthetic.world_generator import (
    WorldGenerator,
    generate_multiple_worlds,
)
from stagebridge.benchmarks.semi_synthetic.interaction_rules import (
    InteractionRuleEngine,
    compute_ground_truth_labels,
)
from stagebridge.benchmarks.semi_synthetic.benchmark_generator import (
    SemiSyntheticBenchmarkGenerator,
    generate_benchmark,
)
from stagebridge.benchmarks.semi_synthetic.metrics import (
    evaluate_receiver_state_recovery,
    evaluate_sender_attribution,
    BenchmarkMetrics,
)


class TestBenchmarkConfig:
    """Tests for benchmark configuration."""

    def test_default_config_has_cell_groups(self):
        """Default config should have cell groups."""
        config = BenchmarkConfig()
        assert len(config.cell_groups) > 0
        assert any(g.role == "receiver" for g in config.cell_groups)
        assert any(g.role == "sender" for g in config.cell_groups)

    def test_default_config_has_interaction_rules(self):
        """Default config should have interaction rules."""
        config = BenchmarkConfig()
        assert len(config.interaction_rules) > 0

    def test_smoke_config_is_smaller(self):
        """Smoke config should have fewer worlds and cells."""
        default = BenchmarkConfig()
        smoke = SmokeConfig()

        assert smoke.n_worlds_train < default.n_worlds_train
        assert smoke.cells_per_world < default.cells_per_world

    def test_interaction_rule_stage_modulation(self):
        """Interaction rule should apply stage modulation."""
        rule = InteractionRule(
            rule_id="test",
            sender_group="sender",
            receiver_group="receiver",
            interaction_radius=100.0,
            effect_strength=0.5,
            effect_name="test_effect",
            stage_modulation={"Normal": 0.5, "LUAD": 2.0},
        )

        assert rule.get_stage_effect("Normal") == 0.25  # 0.5 * 0.5
        assert rule.get_stage_effect("LUAD") == 1.0  # 0.5 * 2.0
        assert rule.get_stage_effect("Unknown") == 0.5  # Default


class TestFallbackData:
    """Tests for fallback data generation."""

    def test_create_fallback_data_shape(self):
        """Fallback data should have correct shape."""
        adata = create_fallback_data(n_cells=500, n_genes=1000)

        assert adata.n_obs == 500
        assert adata.n_vars == 1000
        assert "cell_type" in adata.obs.columns
        assert "stage" in adata.obs.columns

    def test_create_fallback_data_has_pca(self):
        """Fallback data should have PCA embedding."""
        adata = create_fallback_data(n_cells=200, n_genes=500)

        assert "X_pca" in adata.obsm
        assert adata.obsm["X_pca"].shape == (200, 32)


class TestFeatureHarmonization:
    """Tests for feature harmonization."""

    def test_harmonize_computes_shared_genes(self):
        """Harmonizer should compute shared genes."""
        harmonizer = FeatureHarmonizer(n_hvg=100)

        gene_sets = {
            "source1": {"A", "B", "C", "D", "E"},
            "source2": {"B", "C", "D", "F", "G"},
            "source3": {"C", "D", "E", "F", "H"},
        }

        report = harmonizer.harmonize(gene_sets)

        assert len(report.shared_genes) == 2  # C, D
        assert "C" in report.shared_genes
        assert "D" in report.shared_genes

    def test_harmonize_computes_overlap_matrix(self):
        """Harmonizer should compute pairwise overlap."""
        harmonizer = FeatureHarmonizer()

        gene_sets = {
            "source1": set("ABCDE"),
            "source2": set("BCDEF"),
        }

        report = harmonizer.harmonize(gene_sets)

        assert "source1" in report.overlap_matrix
        assert "source2" in report.overlap_matrix["source1"]


class TestWorldGenerator:
    """Tests for synthetic world generation."""

    @pytest.fixture
    def simple_cell_pools(self):
        """Create simple cell pools for testing."""
        n = 100
        return {
            "epithelial_receiver": pd.DataFrame({
                "cell_type": ["AT2"] * n,
                "stage": np.random.choice(["Normal", "LUAD"], n),
            }),
            "caf_sender": pd.DataFrame({
                "cell_type": ["Fibroblast"] * n,
                "stage": np.random.choice(["Normal", "LUAD"], n),
            }),
            "immune_sender": pd.DataFrame({
                "cell_type": ["Macrophage"] * n,
                "stage": np.random.choice(["Normal", "LUAD"], n),
            }),
        }

    def test_generate_world_produces_cells(self, simple_cell_pools):
        """World generator should produce cells with positions."""
        generator = WorldGenerator(width=500, height=500)

        world = generator.generate_world(
            world_id="test_world",
            split="train",
            cell_pools=simple_cell_pools,
            n_cells=100,
            seed=42,
        )

        assert len(world.cell_positions) > 0
        assert "x" in world.cell_positions.columns
        assert "y" in world.cell_positions.columns
        assert "cell_group" in world.cell_positions.columns

    def test_world_positions_in_bounds(self, simple_cell_pools):
        """Cell positions should be within world bounds."""
        width, height = 500, 500
        generator = WorldGenerator(width=width, height=height)

        world = generator.generate_world(
            world_id="test",
            split="train",
            cell_pools=simple_cell_pools,
            n_cells=100,
            seed=42,
        )

        assert world.cell_positions["x"].min() >= 0
        assert world.cell_positions["x"].max() <= width
        assert world.cell_positions["y"].min() >= 0
        assert world.cell_positions["y"].max() <= height

    def test_generate_multiple_worlds_independent(self, simple_cell_pools):
        """Multiple worlds should have different seeds."""
        worlds = generate_multiple_worlds(
            cell_pools=simple_cell_pools,
            n_worlds=3,
            cells_per_world=50,
            split="train",
            base_seed=42,
        )

        assert len(worlds) == 3
        assert all(w.seed != worlds[0].seed for w in worlds[1:])


class TestInteractionRules:
    """Tests for interaction rule application."""

    @pytest.fixture
    def simple_world_positions(self):
        """Create simple world positions for testing."""
        # Create a grid of cells with known positions
        positions = []

        # Receivers in center
        for i in range(10):
            positions.append({
                "synthetic_cell_id": f"receiver_{i}",
                "x": 250 + np.random.randn() * 10,
                "y": 250 + np.random.randn() * 10,
                "cell_group": "epithelial_receiver",
                "stage": "Normal",
            })

        # Senders nearby (within 100 radius)
        for i in range(5):
            positions.append({
                "synthetic_cell_id": f"caf_near_{i}",
                "x": 300 + np.random.randn() * 10,
                "y": 250 + np.random.randn() * 10,
                "cell_group": "caf_sender",
                "stage": "Normal",
            })

        # Senders far away (outside 100 radius)
        for i in range(5):
            positions.append({
                "synthetic_cell_id": f"caf_far_{i}",
                "x": 500,
                "y": 500,
                "cell_group": "caf_sender",
                "stage": "Normal",
            })

        return pd.DataFrame(positions)

    def test_interaction_engine_applies_rules(self, simple_world_positions):
        """Interaction engine should apply rules based on radius."""
        rules = [
            InteractionRule(
                rule_id="test_rule",
                sender_group="caf_sender",
                receiver_group="epithelial_receiver",
                interaction_radius=100.0,
                effect_strength=0.9,  # High strength for deterministic test
                effect_name="test_effect",
            )
        ]

        engine = InteractionRuleEngine(rules, seed=42)
        updated, report = engine.apply_to_world(simple_world_positions)

        # Some receivers should be interacting (near senders)
        assert report.n_interacting > 0
        assert "is_interacting" in updated.columns

    def test_ground_truth_labels_deterministic(self, simple_world_positions):
        """Ground truth computation should be deterministic."""
        rules = [
            InteractionRule(
                rule_id="test_rule",
                sender_group="caf_sender",
                receiver_group="epithelial_receiver",
                interaction_radius=100.0,
                effect_strength=0.5,
                effect_name="test_effect",
            )
        ]

        gt1 = compute_ground_truth_labels(simple_world_positions.copy(), rules)
        gt2 = compute_ground_truth_labels(simple_world_positions.copy(), rules)

        # Ground truth should be identical
        assert (gt1["gt_test_rule_strength"] == gt2["gt_test_rule_strength"]).all()


class TestMetrics:
    """Tests for benchmark metrics."""

    def test_receiver_state_metrics_basic(self):
        """Test basic receiver state metrics computation."""
        predictions = np.array([True, True, False, False, True])
        ground_truth = np.array([True, False, False, True, True])

        metrics = evaluate_receiver_state_recovery(predictions, ground_truth)

        assert metrics.n_samples == 5
        assert 0 <= metrics.accuracy <= 1
        assert 0 <= metrics.f1 <= 1

    def test_receiver_state_metrics_with_probs(self):
        """Test receiver state metrics with probability scores."""
        predictions = np.array([True, True, False, False])
        ground_truth = np.array([True, True, False, False])
        probs = np.array([0.9, 0.8, 0.2, 0.1])

        metrics = evaluate_receiver_state_recovery(predictions, ground_truth, probs)

        assert metrics.auroc == 1.0  # Perfect predictions
        assert metrics.accuracy == 1.0

    def test_sender_attribution_correlation(self):
        """Test sender attribution correlation computation."""
        model_attr = np.array([0.1, 0.5, 0.8, 0.2, 0.9])
        gt_counts = np.array([1, 3, 5, 2, 6])

        metrics = evaluate_sender_attribution(model_attr, gt_counts, "test_sender")

        assert metrics.sender_group == "test_sender"
        assert metrics.attribution_correlation > 0.5  # Should be positively correlated


@pytest.mark.slow
class TestBenchmarkGenerator:
    """Tests for full benchmark generation.

    These tests load large datasets and are marked slow.
    Run with: pytest -m slow
    Skip with: pytest -m "not slow"
    """

    def test_smoke_benchmark_generates(self, tmp_path):
        """Smoke benchmark should generate successfully with fallback data."""
        config = SmokeConfig()
        config.output_dir = tmp_path

        # Use fallback_only to skip loading huge real datasets (avoids OOM)
        report = generate_benchmark(config=config, fallback_only=True)

        assert report.success
        assert len(report.output_paths) > 0

    def test_benchmark_creates_all_splits(self, tmp_path):
        """Benchmark should create train/val/test splits."""
        config = SmokeConfig()
        config.output_dir = tmp_path

        generator = SemiSyntheticBenchmarkGenerator(config)
        generator.generate(fallback_only=True)

        assert "train" in generator.worlds
        assert "val" in generator.worlds
        assert "test" in generator.worlds
        assert len(generator.worlds["train"]) == config.n_worlds_train

    def test_benchmark_exports_ground_truth(self, tmp_path):
        """Benchmark should export ground truth files."""
        config = SmokeConfig()
        config.output_dir = tmp_path

        generate_benchmark(config=config, fallback_only=True)

        # Check for ground truth files
        benchmark_dir = tmp_path / config.benchmark_name
        assert benchmark_dir.exists()

        # Check manifest
        manifest_path = benchmark_dir / "benchmark_manifest.json"
        assert manifest_path.exists()

        # Check at least one world has ground truth
        train_dir = benchmark_dir / "train"
        if train_dir.exists():
            world_dirs = list(train_dir.iterdir())
            if world_dirs:
                gt_path = world_dirs[0] / "ground_truth.parquet"
                assert gt_path.exists()
