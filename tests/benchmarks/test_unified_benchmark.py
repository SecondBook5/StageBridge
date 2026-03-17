"""Tests for unified benchmark generator."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from stagebridge.benchmarks.unified.config import (
    UnifiedBenchmarkConfig,
    SmokeTestConfig,
    FullBenchmarkConfig,
    InteractionRule,
    NicheInfluenceSpec,
    CellGroupSpec,
    DynamicsConfig,
)
from stagebridge.benchmarks.unified.ground_truth import (
    GroundTruth,
    GroundTruthRecovery,
    RecoveryMetrics,
    build_ground_truth_from_config,
)
from stagebridge.benchmarks.unified.generator import (
    UnifiedBenchmarkGenerator,
    generate_benchmark,
)


class TestUnifiedConfig:
    """Tests for unified benchmark configuration."""

    def test_default_config_has_cell_groups(self):
        """Default config should have cell groups."""
        config = UnifiedBenchmarkConfig()
        assert len(config.cell_groups) > 0
        assert any(g.role == "receiver" for g in config.cell_groups)
        assert any(g.role == "sender" for g in config.cell_groups)

    def test_default_config_has_interaction_rules(self):
        """Default config should have interaction rules."""
        config = UnifiedBenchmarkConfig()
        assert len(config.interaction_rules) > 0

    def test_interaction_rules_have_niche_influence(self):
        """Default interaction rules should have linked niche influence."""
        config = UnifiedBenchmarkConfig()
        rules_with_influence = [r for r in config.interaction_rules if r.niche_influence is not None]
        assert len(rules_with_influence) > 0

    def test_smoke_config_is_smaller(self):
        """Smoke config should have fewer worlds and cells."""
        default = UnifiedBenchmarkConfig()
        smoke = SmokeTestConfig()

        assert smoke.n_worlds_train < default.n_worlds_train
        assert smoke.cells_per_world < default.cells_per_world
        assert smoke.n_cells < default.n_cells

    def test_full_config_is_larger(self):
        """Full config should have more worlds and cells."""
        default = UnifiedBenchmarkConfig()
        full = FullBenchmarkConfig()

        assert full.n_worlds_train > default.n_worlds_train
        assert full.n_cells > default.n_cells

    def test_difficulty_adjusts_parameters(self):
        """Difficulty level should adjust dynamics parameters."""
        easy = UnifiedBenchmarkConfig(difficulty="easy")
        hard = UnifiedBenchmarkConfig(difficulty="hard")

        assert easy.dynamics.diffusion_strength < hard.dynamics.diffusion_strength

    def test_mode_properties(self):
        """Mode properties should correctly identify generation mode."""
        synthetic = UnifiedBenchmarkConfig(mode="fully_synthetic")
        semi = UnifiedBenchmarkConfig(mode="semi_synthetic")
        hybrid = UnifiedBenchmarkConfig(mode="hybrid")

        assert synthetic.is_synthetic
        assert not synthetic.uses_real_data
        assert synthetic.applies_causal_dynamics

        assert not semi.is_synthetic
        assert semi.uses_real_data
        assert not semi.applies_causal_dynamics

        assert not hybrid.is_synthetic
        assert hybrid.uses_real_data
        assert hybrid.applies_causal_dynamics

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

    def test_niche_influence_stage_modulation(self):
        """Niche influence should apply stage modulation."""
        niche = NicheInfluenceSpec(
            sender_group="sender",
            influence_name="test",
            strength=0.6,
            stage_modulation={"Normal": 0.5, "LUAD": 1.5},
        )

        assert abs(niche.get_effective_strength("Normal") - 0.3) < 1e-9  # 0.6 * 0.5
        assert abs(niche.get_effective_strength("LUAD") - 0.9) < 1e-9  # 0.6 * 1.5
        assert niche.get_effective_strength("Unknown") == 0.6  # Default


class TestGroundTruth:
    """Tests for ground truth management."""

    def test_build_ground_truth_from_config(self):
        """Ground truth should be built from config."""
        config = SmokeTestConfig()
        rng = np.random.default_rng(42)
        gt = build_ground_truth_from_config(config, rng)

        # Should have stage centroids
        assert len(gt.stage_centroids) == len(config.stages)

        # Should have influence vectors
        assert len(gt.influence_vectors) > 0

        # Should have interaction rules
        assert len(gt.interaction_rules) > 0

    def test_ground_truth_serialization(self, tmp_path):
        """Ground truth should serialize and deserialize correctly."""
        gt = GroundTruth(
            stage_centroids={"Normal": [0.0, 0.0], "LUAD": [1.0, 0.5]},
            drift_strength=1.0,
            influence_vectors={"EMT": [0.5, 0.5, 0.0, 0.0]},
            influence_strengths={"EMT": 0.6},
            influential_celltypes=["Fibroblast"],
            generation_seed=42,
        )

        # Save
        save_path = tmp_path / "gt.json"
        gt.save(save_path)
        assert save_path.exists()

        # Load
        loaded = GroundTruth.load(save_path)
        assert loaded.drift_strength == gt.drift_strength
        assert loaded.influential_celltypes == gt.influential_celltypes
        assert loaded.generation_seed == gt.generation_seed

    def test_ground_truth_to_dict(self):
        """Ground truth should convert to dictionary."""
        gt = GroundTruth(
            stage_centroids={"Normal": [0.0], "LUAD": [1.0]},
            drift_strength=1.0,
        )

        d = gt.to_dict()
        assert "flow_field" in d
        assert "niche_influence" in d
        assert "clone_structure" in d
        assert "interaction_rules" in d


class TestGroundTruthRecovery:
    """Tests for ground truth recovery evaluation."""

    def test_evaluate_niche_influence_recovery(self):
        """Should evaluate niche influence recovery."""
        gt = GroundTruth(
            influence_vectors={"EMT": [1.0, 0.0, 0.0, 0.0]},
            influential_celltypes=["Fibroblast", "Macrophage"],
        )

        recovery = GroundTruthRecovery(gt)

        # Perfect recovery
        metrics = recovery.evaluate_niche_influence_recovery(
            predicted_influence_vectors={"EMT": np.array([1.0, 0.0, 0.0, 0.0])},
            predicted_influential_celltypes=["Fibroblast", "Macrophage"],
        )

        assert metrics["influence_direction_cosines"]["EMT"] > 0.99
        assert metrics["influential_celltype_precision"] == 1.0
        assert metrics["influential_celltype_recall"] == 1.0

    def test_evaluate_niche_influence_partial_recovery(self):
        """Should evaluate partial niche influence recovery."""
        gt = GroundTruth(
            influential_celltypes=["Fibroblast", "Macrophage", "T_cell"],
        )

        recovery = GroundTruthRecovery(gt)

        # Partial recovery
        metrics = recovery.evaluate_niche_influence_recovery(
            predicted_influential_celltypes=["Fibroblast", "Endothelial"],
        )

        assert metrics["influential_celltype_precision"] == 0.5  # 1/2
        assert abs(metrics["influential_celltype_recall"] - 1/3) < 0.01  # 1/3

    def test_compute_composite_score(self):
        """Should compute composite recovery score."""
        gt = GroundTruth()
        recovery = GroundTruthRecovery(gt)

        result = recovery.compute_composite_score(
            flow_metrics={"centroid_correlation": 0.9, "drift_direction_cosine": 0.8},
            niche_metrics={
                "mean_direction_cosine": 0.7,
                "influential_celltype_precision": 0.8,
                "influential_celltype_recall": 0.9,
            },
            clone_metrics={"clone_compatibility_auc": 0.85},
            interaction_metrics={"receiver_auroc": 0.9},
        )

        assert isinstance(result, RecoveryMetrics)
        assert result.composite_score > 0
        assert result.centroid_correlation == 0.9
        assert result.clone_compatibility_auc == 0.85


class TestUnifiedGenerator:
    """Tests for unified benchmark generator."""

    def test_smoke_generation_fully_synthetic(self, tmp_path):
        """Smoke test should generate fully synthetic benchmark."""
        config = SmokeTestConfig()
        config.mode = "fully_synthetic"
        config.output_dir = tmp_path

        report = generate_benchmark(config=config, use_fallback=True)

        assert report.success
        assert report.mode == "fully_synthetic"
        assert report.n_cells_generated > 0
        assert len(report.output_paths) > 0

    def test_smoke_generation_creates_splits(self, tmp_path):
        """Smoke test should create train/val/test splits."""
        config = SmokeTestConfig()
        config.output_dir = tmp_path

        generator = UnifiedBenchmarkGenerator(config)
        report = generator.generate(use_fallback=True)

        assert "train" in generator.worlds
        assert "val" in generator.worlds
        assert "test" in generator.worlds
        assert len(generator.worlds["train"]) == config.n_worlds_train
        assert len(generator.worlds["val"]) == config.n_worlds_val
        assert len(generator.worlds["test"]) == config.n_worlds_test

    def test_worlds_have_cells(self, tmp_path):
        """Generated worlds should have cells with positions."""
        config = SmokeTestConfig()
        config.output_dir = tmp_path

        generator = UnifiedBenchmarkGenerator(config)
        generator.generate(use_fallback=True)

        for split, worlds in generator.worlds.items():
            for world in worlds:
                assert len(world.cell_positions) > 0
                assert "x" in world.cell_positions.columns
                assert "y" in world.cell_positions.columns
                assert "cell_group" in world.cell_positions.columns

    def test_cells_have_interactions(self, tmp_path):
        """Cells should have interaction labels."""
        config = SmokeTestConfig()
        config.output_dir = tmp_path

        generator = UnifiedBenchmarkGenerator(config)
        generator.generate(use_fallback=True)

        world = generator.worlds["train"][0]
        assert "is_interacting" in world.cell_positions.columns
        assert "niche_influence_score" in world.cell_positions.columns

    def test_causal_dynamics_applied_in_hybrid_mode(self, tmp_path):
        """Causal dynamics should be applied in hybrid mode."""
        config = SmokeTestConfig()
        config.mode = "hybrid"
        config.output_dir = tmp_path

        generator = UnifiedBenchmarkGenerator(config)
        generator.generate(use_fallback=True)

        world = generator.worlds["train"][0]
        assert "z_influenced" in world.cell_positions.columns
        assert "niche_influence_score" in world.cell_positions.columns

        # Some cells should have non-zero influence
        scores = world.cell_positions["niche_influence_score"]
        assert scores.max() > 0

    def test_reference_projections_computed(self, tmp_path):
        """Reference projections should be computed."""
        config = SmokeTestConfig()
        config.output_dir = tmp_path

        generator = UnifiedBenchmarkGenerator(config)
        generator.generate(use_fallback=True)

        world = generator.worlds["train"][0]
        assert "z_hlca" in world.cell_positions.columns
        assert "z_luca" in world.cell_positions.columns
        assert "z_fused" in world.cell_positions.columns

    def test_expression_generated_in_synthetic_mode(self, tmp_path):
        """Gene expression should be generated in synthetic mode."""
        config = SmokeTestConfig()
        config.mode = "fully_synthetic"
        config.output_dir = tmp_path

        generator = UnifiedBenchmarkGenerator(config)
        generator.generate(use_fallback=True)

        world = generator.worlds["train"][0]
        assert "expression" in world.cell_positions.columns

    def test_neighborhoods_built(self, tmp_path):
        """Neighborhoods should be built for each world."""
        config = SmokeTestConfig()
        config.output_dir = tmp_path

        generator = UnifiedBenchmarkGenerator(config)
        generator.generate(use_fallback=True)

        world = generator.worlds["train"][0]
        assert hasattr(world, "neighborhoods")
        assert world.neighborhoods is not None
        assert len(world.neighborhoods) == len(world.cell_positions)

    def test_ground_truth_exported(self, tmp_path):
        """Ground truth should be exported."""
        config = SmokeTestConfig()
        config.output_dir = tmp_path

        report = generate_benchmark(config=config, use_fallback=True)

        gt_path = tmp_path / config.benchmark_name / "ground_truth.json"
        assert gt_path.exists()

    def test_manifest_exported(self, tmp_path):
        """Benchmark manifest should be exported."""
        config = SmokeTestConfig()
        config.output_dir = tmp_path

        report = generate_benchmark(config=config, use_fallback=True)

        manifest_path = tmp_path / config.benchmark_name / "benchmark_manifest.json"
        assert manifest_path.exists()

    def test_world_ground_truth_exported(self, tmp_path):
        """World-level ground truth should be exported."""
        config = SmokeTestConfig()
        config.output_dir = tmp_path

        report = generate_benchmark(config=config, use_fallback=True)

        # Check first train world
        train_dir = tmp_path / config.benchmark_name / "train"
        world_dirs = list(train_dir.iterdir())
        assert len(world_dirs) > 0

        gt_path = world_dirs[0] / "ground_truth.parquet"
        assert gt_path.exists()

    def test_seed_reproducibility(self, tmp_path):
        """Same seed should produce same results."""
        config1 = SmokeTestConfig()
        config1.output_dir = tmp_path / "run1"

        config2 = SmokeTestConfig()
        config2.output_dir = tmp_path / "run2"

        gen1 = UnifiedBenchmarkGenerator(config1)
        gen1.generate(use_fallback=True)

        gen2 = UnifiedBenchmarkGenerator(config2)
        gen2.generate(use_fallback=True)

        # Compare first world
        world1 = gen1.worlds["train"][0]
        world2 = gen2.worlds["train"][0]

        assert len(world1.cell_positions) == len(world2.cell_positions)
        # Interaction states should match
        assert (world1.cell_positions["is_interacting"] == world2.cell_positions["is_interacting"]).all()

    def test_different_splits_different_seeds(self, tmp_path):
        """Different splits should have different seeds."""
        config = SmokeTestConfig()
        config.output_dir = tmp_path

        generator = UnifiedBenchmarkGenerator(config)
        generator.generate(use_fallback=True)

        train_world = generator.worlds["train"][0]
        val_world = generator.worlds["val"][0]

        # Seeds should be different
        assert train_world.seed != val_world.seed

    def test_cell_pools_created(self, tmp_path):
        """Cell pools should be created for each group."""
        config = SmokeTestConfig()
        config.output_dir = tmp_path

        generator = UnifiedBenchmarkGenerator(config)
        generator.generate(use_fallback=True)

        assert len(generator.cell_pools) > 0
        for name, pool in generator.cell_pools.items():
            assert len(pool.cells) > 0
            assert pool.role in ("receiver", "sender", "background")


class TestInteractionRules:
    """Tests for interaction rule application."""

    def test_interaction_rate_positive(self, tmp_path):
        """Some cells should be interacting."""
        config = SmokeTestConfig()
        config.output_dir = tmp_path

        report = generate_benchmark(config=config, use_fallback=True)

        assert report.interaction_summary["interaction_rate"] > 0

    def test_ground_truth_labels_deterministic(self, tmp_path):
        """Ground truth labels should be deterministic."""
        config = SmokeTestConfig()
        config.output_dir = tmp_path

        generator = UnifiedBenchmarkGenerator(config)
        generator.generate(use_fallback=True)

        world = generator.worlds["train"][0]

        # gt_* columns should exist
        gt_cols = [c for c in world.cell_positions.columns if c.startswith("gt_")]
        assert len(gt_cols) > 0

        # gt_max_strength should exist
        assert "gt_max_strength" in world.cell_positions.columns or "gt_should_interact" in world.cell_positions.columns


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_generate_benchmark_smoke(self, tmp_path):
        """generate_benchmark with smoke=True should work."""
        config = SmokeTestConfig()
        config.output_dir = tmp_path

        report = generate_benchmark(config=config, smoke=True, use_fallback=True)
        assert report.success

    def test_generate_benchmark_mode_override(self, tmp_path):
        """generate_benchmark should allow mode override."""
        config = SmokeTestConfig()
        config.output_dir = tmp_path

        report = generate_benchmark(config=config, mode="fully_synthetic", use_fallback=True)
        assert report.success
        assert report.mode == "fully_synthetic"
