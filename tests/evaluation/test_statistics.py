"""Tests for stagebridge.evaluation.statistics module."""
import numpy as np
import pytest
from pathlib import Path
import tempfile

from stagebridge.evaluation.statistics import (
    cohens_d,
    cliffs_delta,
    bootstrap_ci,
    interpret_effect_size,
    paired_comparison,
    apply_multiple_comparison_correction,
    generate_latex_comparison_table,
)


class TestCohensD:
    def test_identical_groups(self):
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([1, 2, 3, 4, 5])
        assert cohens_d(x, y) == pytest.approx(0.0, abs=0.01)

    def test_large_effect(self):
        x = np.array([10, 11, 12, 13, 14])
        y = np.array([1, 2, 3, 4, 5])
        d = cohens_d(x, y)
        assert d > 2.0  # Very large effect

    def test_negative_effect(self):
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([10, 11, 12, 13, 14])
        d = cohens_d(x, y)
        assert d < -2.0

    def test_small_effect(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([1.2, 2.2, 3.2, 4.2, 5.2])
        d = cohens_d(x, y)
        assert abs(d) < 0.5  # Small effect


class TestCliffsDelta:
    def test_identical_groups(self):
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([1, 2, 3, 4, 5])
        assert cliffs_delta(x, y) == pytest.approx(0.0, abs=0.01)

    def test_complete_dominance(self):
        x = np.array([10, 11, 12, 13, 14])
        y = np.array([1, 2, 3, 4, 5])
        delta = cliffs_delta(x, y)
        assert delta == pytest.approx(1.0, abs=0.01)

    def test_complete_reverse_dominance(self):
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([10, 11, 12, 13, 14])
        delta = cliffs_delta(x, y)
        assert delta == pytest.approx(-1.0, abs=0.01)

    def test_bounded(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 100)
        y = rng.normal(0.5, 1, 100)
        delta = cliffs_delta(x, y)
        assert -1.0 <= delta <= 1.0


class TestBootstrapCI:
    def test_returns_three_values(self):
        values = np.array([1, 2, 3, 4, 5])
        result = bootstrap_ci(values)
        assert len(result) == 3

    def test_mean_is_correct(self):
        values = np.array([1, 2, 3, 4, 5])
        mean, ci_lower, ci_upper = bootstrap_ci(values)
        assert mean == pytest.approx(3.0)

    def test_ci_contains_mean(self):
        values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        mean, ci_lower, ci_upper = bootstrap_ci(values)
        assert ci_lower <= mean <= ci_upper

    def test_reproducible_with_seed(self):
        values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result1 = bootstrap_ci(values, seed=42)
        result2 = bootstrap_ci(values, seed=42)
        assert result1 == result2

    def test_handles_single_value(self):
        values = np.array([5.0])
        mean, ci_lower, ci_upper = bootstrap_ci(values)
        assert mean == pytest.approx(5.0)

    def test_handles_nans(self):
        values = np.array([1, 2, np.nan, 4, 5])
        mean, ci_lower, ci_upper = bootstrap_ci(values)
        assert mean == pytest.approx(3.0)  # Mean of [1,2,4,5]


class TestInterpretEffectSize:
    def test_negligible(self):
        assert interpret_effect_size(0.1) == "negligible"
        assert interpret_effect_size(-0.1) == "negligible"

    def test_small(self):
        assert interpret_effect_size(0.3) == "small"
        assert interpret_effect_size(-0.3) == "small"

    def test_medium(self):
        assert interpret_effect_size(0.6) == "medium"
        assert interpret_effect_size(-0.6) == "medium"

    def test_large(self):
        assert interpret_effect_size(1.0) == "large"
        assert interpret_effect_size(-1.0) == "large"


class TestPairedComparison:
    def test_basic_comparison(self):
        baseline = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        model = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
        result = paired_comparison(baseline, model, "test_metric")

        assert result["metric"] == "test_metric"
        assert result["n_pairs"] == 5
        assert result["mean_difference"] == pytest.approx(0.5)
        assert "paired_ttest_pvalue" in result
        assert "cohens_d" in result

    def test_insufficient_samples(self):
        baseline = np.array([1.0, 2.0])
        model = np.array([1.5, 2.5])
        result = paired_comparison(baseline, model, "test_metric")
        assert result["error"] == "insufficient_samples"

    def test_handles_nans(self):
        baseline = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        model = np.array([1.5, 2.5, np.nan, 4.5, 5.5])
        result = paired_comparison(baseline, model, "test_metric")
        assert result["n_pairs"] == 3  # Only 3 valid pairs

    def test_relative_improvement(self):
        baseline = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
        model = np.array([11.0, 11.0, 11.0, 11.0, 11.0])
        result = paired_comparison(baseline, model, "test_metric")
        assert result["relative_improvement_pct"] == pytest.approx(10.0)


class TestMultipleComparisonCorrection:
    def test_bonferroni(self):
        pvalues = [0.01, 0.02, 0.03, 0.04, 0.05]
        corrected, reject = apply_multiple_comparison_correction(pvalues, "bonferroni")
        assert len(corrected) == 5
        assert len(reject) == 5
        assert corrected[0] == pytest.approx(0.05)  # 0.01 * 5

    def test_fdr_bh(self):
        pvalues = [0.001, 0.01, 0.02, 0.03, 0.5]
        corrected, reject = apply_multiple_comparison_correction(pvalues, "fdr_bh")
        assert len(corrected) == 5
        assert reject[0] is True  # Smallest p-value should be significant

    def test_no_correction(self):
        pvalues = [0.01, 0.06]
        corrected, reject = apply_multiple_comparison_correction(pvalues, "none")
        assert corrected[0] == pytest.approx(0.01)
        assert reject[0] is True
        assert reject[1] is False


class TestGenerateLatexTable:
    def test_creates_file(self):
        comparisons = [
            {
                "metric": "accuracy",
                "baseline_mean": 0.8,
                "baseline_std": 0.05,
                "model_mean": 0.9,
                "model_std": 0.03,
                "mean_difference": 0.1,
                "paired_ttest_pvalue": 0.001,
                "effect_size_interpretation": "large",
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "table.tex"
            generate_latex_comparison_table(comparisons, output_path)

            assert output_path.exists()
            content = output_path.read_text()
            assert "accuracy" in content
            assert "\\begin{table}" in content
            assert "\\end{table}" in content

    def test_skips_errors(self):
        comparisons = [
            {"error": "insufficient_samples"},
            {
                "metric": "accuracy",
                "baseline_mean": 0.8,
                "baseline_std": 0.05,
                "model_mean": 0.9,
                "model_std": 0.03,
                "mean_difference": 0.1,
                "paired_ttest_pvalue": 0.001,
                "effect_size_interpretation": "large",
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "table.tex"
            generate_latex_comparison_table(comparisons, output_path)
            content = output_path.read_text()
            assert "accuracy" in content
