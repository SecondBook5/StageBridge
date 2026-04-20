"""Tests for stagebridge.spatial.backend_comparison module."""
import json
import numpy as np
import pytest
from pathlib import Path
import tempfile

from stagebridge.spatial.backend_comparison import (
    METRICS_CONFIG,
    FALLBACK_METRICS,
    compute_composite_score,
    load_backend_metrics,
    compare_backends,
    select_canonical_backend,
)


class TestMetricsConfig:
    def test_weights_sum_to_one(self):
        total = sum(cfg["weight"] for cfg in METRICS_CONFIG.values())
        assert total == pytest.approx(1.0)

    def test_fallback_weights_sum_to_one(self):
        total = sum(cfg["weight"] for cfg in FALLBACK_METRICS.values())
        assert total == pytest.approx(1.0)

    def test_all_have_higher_better(self):
        for name, cfg in METRICS_CONFIG.items():
            assert "higher_better" in cfg, f"{name} missing higher_better"


class TestComputeCompositeScore:
    def test_perfect_metrics(self):
        metrics = {
            "types_per_spot_mean": 15.0,  # Max normalized to 1.0
            "effective_coverage": 1.0,
            "global_type_coverage": 1.0,
            "mean_entropy": 1.0,
            "gini_coefficient_mean": 0.0,  # 0 is best (inverted)
        }
        score = compute_composite_score(metrics)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_zero_metrics(self):
        metrics = {
            "types_per_spot_mean": 0.0,
            "effective_coverage": 0.0,
            "global_type_coverage": 0.0,
            "mean_entropy": 0.0,
            "gini_coefficient_mean": 1.0,  # 1 is worst
        }
        score = compute_composite_score(metrics)
        assert score == pytest.approx(0.0, abs=0.01)

    def test_falls_back_to_fallback_metrics(self):
        # Only provide fallback metrics
        metrics = {
            "coverage": 0.8,
            "mean_entropy": 0.7,
            "sparsity": 0.3,  # Lower is better
        }
        score = compute_composite_score(metrics)
        assert 0 < score < 1

    def test_handles_nan(self):
        metrics = {
            "types_per_spot_mean": np.nan,
            "effective_coverage": 0.8,
            "global_type_coverage": 0.7,
        }
        score = compute_composite_score(metrics)
        assert not np.isnan(score)

    def test_empty_metrics_returns_zero(self):
        score = compute_composite_score({})
        assert score == 0.0

    def test_custom_config(self):
        # Need at least 2 metrics to avoid fallback
        custom_config = {
            "my_metric": {"weight": 0.5, "higher_better": True},
            "other_metric": {"weight": 0.5, "higher_better": True},
        }
        metrics = {"my_metric": 0.5, "other_metric": 0.5}
        score = compute_composite_score(metrics, config=custom_config)
        assert score == pytest.approx(0.5)


class TestLoadBackendMetrics:
    @pytest.fixture
    def spatial_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            for label_source in ["hlca", "luca"]:
                for backend in ["tangram", "destvi"]:
                    backend_dir = tmpdir / label_source / backend
                    backend_dir.mkdir(parents=True)

                    metrics = {
                        "types_per_spot_mean": 8.0 + (1 if backend == "destvi" else 0),
                        "effective_coverage": 0.85,
                        "global_type_coverage": 0.9,
                    }

                    with open(backend_dir / "upstream_metrics.json", "w") as f:
                        json.dump(metrics, f)

            yield tmpdir

    def test_loads_all_metrics(self, spatial_dir):
        results = load_backend_metrics(
            spatial_dir,
            ["hlca", "luca"],
            ["tangram", "destvi"]
        )

        assert "hlca" in results
        assert "luca" in results
        assert "tangram" in results["hlca"]
        assert "destvi" in results["hlca"]

    def test_missing_backend_skipped(self, spatial_dir):
        results = load_backend_metrics(
            spatial_dir,
            ["hlca"],
            ["tangram", "nonexistent"]
        )

        assert "tangram" in results["hlca"]
        assert "nonexistent" not in results["hlca"]


class TestCompareBackends:
    def test_selects_best_backend(self):
        results = {
            "hlca": {
                "tangram": {"effective_coverage": 0.7, "mean_entropy": 0.6},
                "destvi": {"effective_coverage": 0.9, "mean_entropy": 0.8},
            }
        }

        comparison = compare_backends(results)
        assert comparison["canonical"]["backend"] == "destvi"

    def test_force_backend(self):
        results = {
            "hlca": {
                "tangram": {"effective_coverage": 0.9, "mean_entropy": 0.8},  # Better
                "destvi": {"effective_coverage": 0.7, "mean_entropy": 0.6},
            }
        }

        comparison = compare_backends(results, force_backend="destvi")
        assert comparison["canonical"]["backend"] == "destvi"
        assert comparison["canonical"]["forced"] is True

    def test_computes_label_ablation(self):
        results = {
            "hlca": {
                "tangram": {"effective_coverage": 0.8, "mean_entropy": 0.7},
            },
            "luca": {
                "tangram": {"effective_coverage": 0.6, "mean_entropy": 0.5},
            }
        }

        comparison = compare_backends(results)
        assert "label_ablation" in comparison
        assert comparison["label_ablation"]["better_label_source"] == "hlca"

    def test_empty_results_handled(self):
        comparison = compare_backends({})
        assert comparison["error"] == "No backend metrics available"


class TestSelectCanonicalBackend:
    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create test data
            backend_dir = tmpdir / "hlca" / "tangram"
            backend_dir.mkdir(parents=True)

            metrics = {
                "types_per_spot_mean": 10.0,
                "effective_coverage": 0.85,
                "global_type_coverage": 0.9,
                "mean_entropy": 0.75,
                "gini_coefficient_mean": 0.2,
            }

            with open(backend_dir / "upstream_metrics.json", "w") as f:
                json.dump(metrics, f)

            result = select_canonical_backend(
                tmpdir,
                label_sources=["hlca"],
                backends=["tangram"],
            )

            assert result["canonical"]["backend"] == "tangram"
            assert result["canonical"]["label_source"] == "hlca"
