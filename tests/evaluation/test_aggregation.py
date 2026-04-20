"""Tests for stagebridge.evaluation.aggregation module."""
import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import tempfile

from stagebridge.evaluation.aggregation import (
    load_cv_results,
    aggregate_metrics,
    per_fold_summary,
    load_baseline_results,
    aggregate_by_baseline,
    compute_improvements,
    generate_latex_results_table,
    generate_latex_baseline_table,
)


@pytest.fixture
def cv_results_dir():
    """Create temporary directory with CV results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        for fold in range(3):
            for seed in [42, 123]:
                run_dir = tmpdir / f"fold{fold}_seed{seed}"
                run_dir.mkdir()

                results = {
                    "metrics": {
                        "accuracy": 0.8 + fold * 0.01 + (seed - 42) * 0.001,
                        "f1": 0.75 + fold * 0.02,
                        "auroc": 0.85,
                    }
                }

                with open(run_dir / "results.json", "w") as f:
                    json.dump(results, f)

        yield tmpdir


@pytest.fixture
def baseline_results_dir():
    """Create temporary directory with baseline results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        for baseline in ["pooling_mlp", "deep_sets"]:
            for fold in range(2):
                for seed in [42, 123]:
                    run_dir = tmpdir / f"{baseline}_fold{fold}_seed{seed}"
                    run_dir.mkdir()

                    base_acc = 0.7 if baseline == "pooling_mlp" else 0.75
                    results = {
                        "metrics": {
                            "accuracy": base_acc + fold * 0.01,
                            "f1": base_acc - 0.05,
                        }
                    }

                    with open(run_dir / "results.json", "w") as f:
                        json.dump(results, f)

        yield tmpdir


class TestLoadCVResults:
    def test_loads_all_results(self, cv_results_dir):
        df = load_cv_results(cv_results_dir, n_folds=3, seeds=[42, 123])
        assert len(df) == 6  # 3 folds * 2 seeds
        assert "fold" in df.columns
        assert "seed" in df.columns
        assert "accuracy" in df.columns

    def test_missing_files_handled(self, cv_results_dir):
        # Remove one file
        (cv_results_dir / "fold0_seed42" / "results.json").unlink()
        df = load_cv_results(cv_results_dir, n_folds=3, seeds=[42, 123])
        assert len(df) == 5  # One missing

    def test_empty_dir_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="No results found"):
                load_cv_results(Path(tmpdir), n_folds=5, seeds=[42])


class TestAggregateMetrics:
    def test_computes_mean_and_ci(self, cv_results_dir):
        df = load_cv_results(cv_results_dir, n_folds=3, seeds=[42, 123])
        agg = aggregate_metrics(df, n_bootstrap=100)

        assert "accuracy" in agg
        assert "mean" in agg["accuracy"]
        assert "ci_lower" in agg["accuracy"]
        assert "ci_upper" in agg["accuracy"]
        assert "formatted" in agg["accuracy"]

    def test_ci_contains_mean(self, cv_results_dir):
        df = load_cv_results(cv_results_dir, n_folds=3, seeds=[42, 123])
        agg = aggregate_metrics(df, n_bootstrap=100)

        for metric, stats in agg.items():
            assert stats["ci_lower"] <= stats["mean"] <= stats["ci_upper"]

    def test_reproducible(self, cv_results_dir):
        df = load_cv_results(cv_results_dir, n_folds=3, seeds=[42, 123])
        agg1 = aggregate_metrics(df, n_bootstrap=100, seed=42)
        agg2 = aggregate_metrics(df, n_bootstrap=100, seed=42)

        assert agg1["accuracy"]["ci_lower"] == agg2["accuracy"]["ci_lower"]


class TestPerFoldSummary:
    def test_computes_per_fold_stats(self, cv_results_dir):
        df = load_cv_results(cv_results_dir, n_folds=3, seeds=[42, 123])
        summary = per_fold_summary(df)

        assert "fold_0" in summary
        assert "fold_1" in summary
        assert "fold_2" in summary

        for fold_key, fold_stats in summary.items():
            assert "accuracy" in fold_stats
            assert "mean" in fold_stats["accuracy"]
            assert "n_seeds" in fold_stats["accuracy"]


class TestLoadBaselineResults:
    def test_loads_all_baselines(self, baseline_results_dir):
        df = load_baseline_results(
            baseline_results_dir,
            ["pooling_mlp", "deep_sets"],
            n_folds=2,
            seeds=[42, 123]
        )
        assert len(df) == 8  # 2 baselines * 2 folds * 2 seeds
        assert "baseline" in df.columns

    def test_filters_to_requested_baselines(self, baseline_results_dir):
        df = load_baseline_results(
            baseline_results_dir,
            ["pooling_mlp"],
            n_folds=2,
            seeds=[42, 123]
        )
        assert len(df) == 4
        assert df["baseline"].unique().tolist() == ["pooling_mlp"]


class TestAggregateByBaseline:
    def test_aggregates_per_baseline(self, baseline_results_dir):
        df = load_baseline_results(
            baseline_results_dir,
            ["pooling_mlp", "deep_sets"],
            n_folds=2,
            seeds=[42, 123]
        )
        agg = aggregate_by_baseline(df, n_bootstrap=100)

        assert "pooling_mlp" in agg
        assert "deep_sets" in agg
        assert "accuracy" in agg["pooling_mlp"]


class TestComputeImprovements:
    def test_computes_relative_improvement(self):
        main_model_agg = {
            "accuracy": {"mean": 0.90},
            "f1": {"mean": 0.85},
        }
        baseline_agg = {
            "pooling_mlp": {"accuracy": {"mean": 0.80}, "f1": {"mean": 0.75}},
            "deep_sets": {"accuracy": {"mean": 0.82}, "f1": {"mean": 0.78}},
        }

        improvements = compute_improvements(
            main_model_agg, baseline_agg,
            ["pooling_mlp", "deep_sets"],
            metrics=["accuracy", "f1"]
        )

        assert "accuracy" in improvements
        # Best baseline is 0.82, main is 0.90: (0.90-0.82)/0.82 * 100 ~ 9.76%
        assert improvements["accuracy"]["relative_improvement_pct"] == pytest.approx(9.76, abs=0.1)


class TestLatexTables:
    def test_results_table_created(self, cv_results_dir):
        df = load_cv_results(cv_results_dir, n_folds=3, seeds=[42, 123])
        agg = aggregate_metrics(df, n_bootstrap=100)

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "table.tex"
            generate_latex_results_table(agg, output)

            assert output.exists()
            content = output.read_text()
            assert "\\begin{table}" in content

    def test_baseline_table_created(self, baseline_results_dir):
        df = load_baseline_results(
            baseline_results_dir,
            ["pooling_mlp", "deep_sets"],
            n_folds=2,
            seeds=[42, 123]
        )
        baseline_agg = aggregate_by_baseline(df, n_bootstrap=100)
        main_agg = {"stage_accuracy": {"mean": 0.9, "std": 0.02}}

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "table.tex"
            generate_latex_baseline_table(baseline_agg, main_agg, output)

            assert output.exists()
            content = output.read_text()
            assert "StageBridge" in content
