"""Tests for stagebridge.evaluation.power module."""
import numpy as np
import pandas as pd
import pytest

from stagebridge.evaluation.power import (
    compute_icc,
    compute_design_effect,
    compute_effective_sample_size,
    compute_power,
    compute_min_detectable_effect,
    run_power_analysis,
    power_analysis_from_cv_results,
    generate_power_report,
    PowerAnalysisResult,
)


class TestComputeICC:
    def test_no_clustering(self):
        """Random data should have ICC near 0."""
        rng = np.random.default_rng(42)
        values = rng.normal(0, 1, 1000)
        groups = np.repeat(np.arange(10), 100)

        icc = compute_icc(values, groups)
        assert icc < 0.1  # Should be near 0

    def test_perfect_clustering(self):
        """When groups have identical values, ICC should be high."""
        # Each group has same value for all members
        group_means = [1.0, 5.0, 10.0, 15.0, 20.0]
        values = np.concatenate([np.full(20, m) for m in group_means])
        groups = np.repeat(np.arange(5), 20)

        icc = compute_icc(values, groups)
        assert icc > 0.95  # Should be near 1

    def test_moderate_clustering(self):
        """Groups with some shared variance."""
        rng = np.random.default_rng(42)
        group_means = [0, 1, 2, 3, 4]  # Smaller spread for moderate ICC
        values = []
        groups = []
        for i, m in enumerate(group_means):
            values.extend(rng.normal(m, 1.5, 50))  # Higher within-group variance
            groups.extend([i] * 50)

        icc = compute_icc(np.array(values), np.array(groups))
        assert 0.2 < icc < 0.95  # Moderate to high clustering

    def test_single_group(self):
        """Single group should return 0."""
        values = np.array([1, 2, 3, 4, 5])
        groups = np.array([0, 0, 0, 0, 0])

        icc = compute_icc(values, groups)
        assert icc == 0.0

    def test_bounded_output(self):
        """ICC should always be in [0, 1]."""
        rng = np.random.default_rng(42)
        for _ in range(10):
            values = rng.normal(0, 1, 100)
            groups = rng.integers(0, 5, 100)
            icc = compute_icc(values, groups)
            assert 0 <= icc <= 1


class TestComputeDesignEffect:
    def test_no_clustering(self):
        """DEFF = 1 when ICC = 0."""
        deff = compute_design_effect(icc=0.0, avg_cluster_size=100)
        assert deff == pytest.approx(1.0)

    def test_high_clustering(self):
        """DEFF increases with ICC and cluster size."""
        deff = compute_design_effect(icc=0.5, avg_cluster_size=100)
        # DEFF = 1 + (100-1) * 0.5 = 50.5
        assert deff == pytest.approx(50.5)

    def test_small_clusters(self):
        """Small clusters reduce DEFF impact."""
        deff = compute_design_effect(icc=0.5, avg_cluster_size=2)
        # DEFF = 1 + (2-1) * 0.5 = 1.5
        assert deff == pytest.approx(1.5)


class TestComputeEffectiveSampleSize:
    def test_no_design_effect(self):
        """N_eff = N when DEFF = 1."""
        n_eff = compute_effective_sample_size(n_total=1000, design_effect=1.0)
        assert n_eff == pytest.approx(1000)

    def test_with_design_effect(self):
        """N_eff < N when DEFF > 1."""
        n_eff = compute_effective_sample_size(n_total=1000, design_effect=10.0)
        assert n_eff == pytest.approx(100)


class TestComputePower:
    def test_large_effect_high_power(self):
        """Large effect with large N should have high power."""
        power = compute_power(effect_size=0.8, n_effective=100)
        assert power > 0.95

    def test_small_effect_low_power(self):
        """Small effect with small N should have low power."""
        power = compute_power(effect_size=0.2, n_effective=20)
        assert power < 0.5

    def test_zero_effect(self):
        """Zero effect should have zero power."""
        power = compute_power(effect_size=0.0, n_effective=100)
        assert power == 0.0

    def test_bounded(self):
        """Power should be in [0, 1]."""
        for es in [0.1, 0.5, 1.0, 2.0]:
            for n in [10, 50, 100, 500]:
                power = compute_power(es, n)
                assert 0 <= power <= 1


class TestComputeMinDetectableEffect:
    def test_large_sample(self):
        """Large sample can detect small effects."""
        mde = compute_min_detectable_effect(n_effective=500)
        assert mde < 0.2  # Can detect small effects

    def test_small_sample(self):
        """Small sample needs large effects."""
        mde = compute_min_detectable_effect(n_effective=10)
        assert mde > 0.8  # Needs large effects

    def test_higher_power_needs_larger_mde(self):
        """Higher power requirements increase MDE."""
        mde_80 = compute_min_detectable_effect(n_effective=50, power=0.80)
        mde_95 = compute_min_detectable_effect(n_effective=50, power=0.95)
        assert mde_95 > mde_80


class TestRunPowerAnalysis:
    @pytest.fixture
    def sample_data(self):
        """Generate sample data with known clustering."""
        rng = np.random.default_rng(42)
        donors = ["D1", "D2", "D3", "D4", "D5"]
        donor_effects = [0, 1, 2, 3, 4]

        values = []
        donor_ids = []
        for donor, effect in zip(donors, donor_effects):
            n_cells = rng.integers(80, 120)
            values.extend(rng.normal(effect, 1, n_cells))
            donor_ids.extend([donor] * n_cells)

        return np.array(values), np.array(donor_ids)

    def test_returns_result(self, sample_data):
        values, donors = sample_data
        result = run_power_analysis(values, donors, observed_effect=0.5)

        assert isinstance(result, PowerAnalysisResult)
        assert result.n_donors == 5
        assert result.n_cells > 400

    def test_computes_icc(self, sample_data):
        values, donors = sample_data
        result = run_power_analysis(values, donors)

        # Should detect the donor clustering
        assert result.icc > 0.3

    def test_design_effect_gt_1(self, sample_data):
        values, donors = sample_data
        result = run_power_analysis(values, donors)

        # Clustering should inflate design effect
        assert result.design_effect > 1.0

    def test_effective_n_less_than_total(self, sample_data):
        values, donors = sample_data
        result = run_power_analysis(values, donors)

        assert result.effective_sample_size < result.n_cells

    def test_handles_nans(self):
        values = np.array([1, 2, np.nan, 4, 5, 6, np.nan, 8, 9, 10])
        donors = np.array(["A"] * 5 + ["B"] * 5)

        result = run_power_analysis(values, donors)
        assert result.n_cells == 8  # 2 NaNs removed

    def test_interpretation_included(self, sample_data):
        values, donors = sample_data
        result = run_power_analysis(values, donors)

        assert len(result.interpretation) > 0
        assert "clustering" in result.interpretation.lower()


class TestPowerAnalysisFromCVResults:
    def test_analyzes_cv_results(self):
        cv_results = pd.DataFrame({
            "fold": [0, 0, 0, 1, 1, 1, 2, 2, 2],
            "seed": [42, 123, 456] * 3,
            "accuracy": [0.80, 0.81, 0.82, 0.85, 0.84, 0.86, 0.78, 0.79, 0.77],
        })

        result = power_analysis_from_cv_results(cv_results, "accuracy")

        assert "n_folds" in result
        assert "power_analysis" in result
        assert result["n_folds"] == 3
        assert result["n_runs"] == 9


class TestGeneratePowerReport:
    def test_generates_report(self):
        results = [
            PowerAnalysisResult(
                n_donors=10,
                n_cells=1000,
                cells_per_donor_mean=100,
                cells_per_donor_std=20,
                icc=0.2,
                design_effect=20.8,
                effective_sample_size=48.1,
                observed_effect_size=0.5,
                power=0.75,
                min_detectable_effect=0.58,
                interpretation="moderate clustering; underpowered",
            )
        ]

        report = generate_power_report(results, ["accuracy"])

        assert "POWER ANALYSIS REPORT" in report
        assert "accuracy" in report
        assert "10 donors" in report
        assert "ICC" in report
