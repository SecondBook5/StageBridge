"""Tests for plasticity scoring module (H1.3 hypothesis testing).

These tests verify:
1. Entropy calculation is mathematically correct
2. Uniform distribution has maximum entropy
3. Deterministic distribution has zero entropy
4. Niche resolution metrics are computed correctly
5. Bifurcation cell identification works
6. Fate commitment analysis produces expected results
"""

import numpy as np
import pytest
import torch

from stagebridge.evaluation.plasticity import (
    compute_plasticity_score,
    compute_normalized_plasticity,
    compute_niche_resolution,
    compute_niche_resolution_effect_size,
    identify_bifurcation_cells,
    compute_bifurcation_enrichment,
    analyze_fate_commitment,
    compute_plasticity_by_cell_type,
    rank_cell_types_by_plasticity,
    stage_logits_to_probs,
    compute_plasticity_from_logits,
    generate_plasticity_report,
    report_to_dict,
    BifurcationCellResult,
    FateCommitmentResult,
    DEFAULT_STAGES,
)


class TestPlasticityScore:
    """Test core plasticity score computation."""

    def test_uniform_distribution_max_entropy(self):
        """Uniform distribution should have maximum entropy."""
        n_stages = 5
        n_cells = 10

        # Uniform distribution
        probs = torch.ones(n_cells, n_stages) / n_stages

        plasticity = compute_plasticity_score(probs)

        # Max entropy = log(n_stages)
        expected_max_entropy = np.log(n_stages)

        assert plasticity.shape == (n_cells,)
        assert torch.allclose(plasticity, torch.full((n_cells,), expected_max_entropy), atol=1e-5)

    def test_deterministic_distribution_zero_entropy(self):
        """Deterministic (one-hot) distribution should have near-zero entropy."""
        n_stages = 5
        n_cells = 10

        # One-hot distributions (each cell committed to different stage)
        probs = torch.zeros(n_cells, n_stages)
        for i in range(n_cells):
            probs[i, i % n_stages] = 1.0

        plasticity = compute_plasticity_score(probs)

        # Entropy should be very close to zero (small epsilon for numerical stability)
        assert plasticity.shape == (n_cells,)
        assert torch.all(plasticity < 0.01)

    def test_binary_entropy_formula(self):
        """Test binary entropy formula: H(p) = -p*log(p) - (1-p)*log(1-p)."""
        # For p=0.5, H = log(2) = 0.693...
        probs = torch.tensor([[0.5, 0.5]])
        plasticity = compute_plasticity_score(probs)
        expected = np.log(2)
        assert torch.isclose(plasticity[0], torch.tensor(expected, dtype=torch.float32), atol=1e-5)

        # For p=0.9, H = -0.9*log(0.9) - 0.1*log(0.1) = 0.325...
        probs = torch.tensor([[0.9, 0.1]])
        plasticity = compute_plasticity_score(probs)
        expected = -0.9 * np.log(0.9) - 0.1 * np.log(0.1)
        assert torch.isclose(plasticity[0], torch.tensor(expected, dtype=torch.float32), atol=1e-5)

    def test_ordering_by_uncertainty(self):
        """More uncertain distributions should have higher plasticity."""
        # Create distributions with varying uncertainty
        probs = torch.tensor([
            [0.01, 0.01, 0.01, 0.02, 0.95],  # Very committed (low plasticity)
            [0.1, 0.1, 0.2, 0.3, 0.3],        # Moderate uncertainty
            [0.2, 0.2, 0.2, 0.2, 0.2],        # Uniform (max plasticity)
        ])

        plasticity = compute_plasticity_score(probs)

        # Plasticity should increase: cell_0 < cell_1 < cell_2
        assert plasticity[0] < plasticity[1] < plasticity[2]

    def test_numpy_input(self):
        """Should accept numpy arrays."""
        probs = np.array([[0.5, 0.5], [0.9, 0.1]])
        plasticity = compute_plasticity_score(probs)
        assert isinstance(plasticity, torch.Tensor)
        assert plasticity.shape == (2,)

    def test_single_cell_input(self):
        """Should handle single cell (1D input)."""
        probs = torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2])
        plasticity = compute_plasticity_score(probs)
        assert plasticity.shape == (1,)


class TestNormalizedPlasticity:
    """Test normalized plasticity in [0, 1] range."""

    def test_uniform_gives_one(self):
        """Uniform distribution should give normalized plasticity of 1."""
        n_stages = 5
        probs = torch.ones(10, n_stages) / n_stages

        normalized = compute_normalized_plasticity(probs)

        assert torch.allclose(normalized, torch.ones(10), atol=1e-5)

    def test_deterministic_gives_zero(self):
        """Deterministic distribution should give normalized plasticity near 0."""
        probs = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]])

        normalized = compute_normalized_plasticity(probs)

        assert normalized[0] < 0.01

    def test_range_zero_to_one(self):
        """All normalized plasticity values should be in [0, 1]."""
        # Random probabilities
        probs = torch.rand(100, 5)
        probs = probs / probs.sum(dim=-1, keepdim=True)

        normalized = compute_normalized_plasticity(probs)

        assert torch.all(normalized >= 0)
        assert torch.all(normalized <= 1)


class TestNicheResolution:
    """Test niche resolution computation."""

    def test_positive_resolution(self):
        """Positive resolution when niche reduces uncertainty."""
        # Without niche: high entropy (uncertain)
        plasticity_without = torch.tensor([1.5, 1.4, 1.6, 1.5, 1.5])
        # With niche: lower entropy (more certain)
        plasticity_with = torch.tensor([0.5, 0.6, 0.4, 0.5, 0.5])

        resolution = compute_niche_resolution(plasticity_with, plasticity_without)

        # All positive (niche helps)
        assert torch.all(resolution > 0)
        assert torch.allclose(resolution, plasticity_without - plasticity_with)

    def test_negative_resolution(self):
        """Negative resolution when niche increases uncertainty (contradicts H1.3)."""
        plasticity_without = torch.tensor([0.5, 0.5])
        plasticity_with = torch.tensor([1.5, 1.5])

        resolution = compute_niche_resolution(plasticity_with, plasticity_without)

        assert torch.all(resolution < 0)

    def test_effect_size_statistics(self):
        """Test effect size computation."""
        # Setup: niche reduces plasticity
        plasticity_without = torch.tensor([1.5, 1.4, 1.6, 1.5, 1.5, 1.3, 1.7, 1.4])
        plasticity_with = torch.tensor([0.5, 0.6, 0.4, 0.5, 0.5, 0.7, 0.3, 0.6])

        effect = compute_niche_resolution_effect_size(plasticity_with, plasticity_without)

        assert "mean_resolution" in effect
        assert "median_resolution" in effect
        assert "std_resolution" in effect
        assert "fraction_resolved" in effect
        assert "cohens_d" in effect
        assert "mean_relative_resolution" in effect

        # Mean resolution should be ~1.0 (1.5 - 0.5)
        assert abs(effect["mean_resolution"] - 1.0) < 0.2

        # Fraction resolved should be 1.0 (all cells benefit)
        assert effect["fraction_resolved"] == 1.0


class TestBifurcationCells:
    """Test bifurcation cell identification."""

    def test_identifies_high_plasticity_cells(self):
        """Should identify cells with highest plasticity scores."""
        # 100 cells, mostly low plasticity, some high
        scores = torch.cat([
            torch.full((90,), 0.5),  # Low plasticity
            torch.full((10,), 1.5),  # High plasticity
        ])

        result = identify_bifurcation_cells(scores, threshold_percentile=90)

        # Should identify the 10 high-plasticity cells
        assert result.n_bifurcation_cells == 10
        assert result.n_total_cells == 100
        assert np.all(result.indices >= 90)  # Indices 90-99

    def test_cell_type_distribution(self):
        """Should compute cell type distribution among bifurcation cells."""
        scores = torch.cat([
            torch.full((50,), 0.5),   # Low
            torch.full((30,), 1.0),   # Medium
            torch.full((20,), 1.5),   # High
        ])
        cell_types = (
            ["AT2"] * 50 +      # Low plasticity cells
            ["AT1"] * 30 +      # Medium plasticity cells
            ["KAC"] * 20        # High plasticity cells (should be in bifurcation)
        )

        result = identify_bifurcation_cells(scores, cell_types=cell_types, threshold_percentile=80)

        # KACs should dominate bifurcation cells
        assert "KAC" in result.cell_type_distribution
        assert result.cell_type_distribution["KAC"] == 20

    def test_minimum_plasticity_threshold(self):
        """Should respect minimum plasticity threshold."""
        scores = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])

        # Percentile threshold would be 0.4 (80th percentile)
        # But minimum is 0.45, so only 0.5 should be selected
        result = identify_bifurcation_cells(
            scores,
            threshold_percentile=80,
            min_plasticity=0.45
        )

        assert result.n_bifurcation_cells == 1
        assert result.indices[0] == 4


class TestBifurcationEnrichment:
    """Test cell type enrichment in bifurcation cells."""

    def test_enrichment_calculation(self):
        """Test fold enrichment calculation."""
        # Create scenario where KACs are enriched in bifurcation cells
        all_types = np.array(
            ["AT2"] * 80 +   # 80% in population
            ["KAC"] * 20    # 20% in population
        )

        # But in bifurcation cells: 50% KAC (2.5x enrichment)
        bifurcation = BifurcationCellResult(
            indices=np.array([80, 81, 82, 83, 84, 85, 86, 87, 88, 89]),  # 10 cells
            plasticity_scores=np.ones(10),
            threshold=1.0,
            cell_type_distribution={"AT2": 5, "KAC": 5},  # 50% KAC
            n_total_cells=100,
            n_bifurcation_cells=10,
        )

        enrichment = compute_bifurcation_enrichment(
            bifurcation,
            all_types,
            target_cell_types=["KAC"]
        )

        # KAC expected: 20%, observed: 50%, fold enrichment: 2.5
        assert abs(enrichment["KAC_expected_fraction"] - 0.2) < 0.01
        assert abs(enrichment["KAC_observed_fraction"] - 0.5) < 0.01
        assert abs(enrichment["KAC_fold_enrichment"] - 2.5) < 0.1


class TestFateCommitment:
    """Test fate commitment analysis."""

    def test_il1b_high_tumor_correlation(self):
        """High IL1B should correlate with tumor fate probability."""
        n_cells = 100

        # Simulate: high IL1B -> high P(tumor)
        il1b_scores = np.linspace(0, 1, n_cells)

        # Transition probs: P(tumor) increases with IL1B
        probs = np.zeros((n_cells, 5))  # Normal, AAH, AIS, MIA, LUAD
        probs[:, 0] = 0.5 - 0.4 * il1b_scores  # P(Normal/repair) decreases
        probs[:, 4] = 0.1 + 0.5 * il1b_scores  # P(LUAD/tumor) increases
        probs[:, 1:4] = 0.4 / 3  # Others constant
        probs = probs / probs.sum(axis=1, keepdims=True)

        result = analyze_fate_commitment(
            probs,
            il1b_scores,
            stages=list(DEFAULT_STAGES),
            feature_name="IL1B",
            tumor_stages=["MIA", "LUAD"],
            repair_stages=["Normal"],
        )

        # Strong positive correlation with tumor
        assert result.correlation_tumor > 0.5
        # Negative correlation with repair
        assert result.correlation_repair < 0
        # Higher tumor prob in IL1B-high group
        assert result.mean_tumor_prob_high > result.mean_tumor_prob_low

    def test_odds_ratio_calculation(self):
        """Test odds ratio calculation for tumor vs repair."""
        # Setup: clear difference between high and low IL1B
        n = 50
        probs = np.zeros((100, 5))

        # Low IL1B (first 50): P(Normal)=0.5, P(LUAD)=0.1
        probs[:n, 0] = 0.5  # Normal (repair)
        probs[:n, 4] = 0.1  # LUAD (tumor)
        probs[:n, 1:4] = 0.4 / 3

        # High IL1B (last 50): P(Normal)=0.1, P(LUAD)=0.5
        probs[n:, 0] = 0.1
        probs[n:, 4] = 0.5
        probs[n:, 1:4] = 0.4 / 3

        probs = probs / probs.sum(axis=1, keepdims=True)

        il1b = np.concatenate([np.zeros(n), np.ones(n)])

        result = analyze_fate_commitment(
            probs,
            il1b,
            threshold_percentile=50,  # Split at median
        )

        # Odds ratio should be > 1 (tumor favored in high IL1B)
        assert result.odds_ratio > 1
        assert result.log_odds > 0


class TestCellTypeAnalysis:
    """Test cell type stratification."""

    def test_stratified_statistics(self):
        """Should compute correct statistics per cell type."""
        # KACs: high plasticity, AT2: low plasticity
        scores = np.concatenate([
            np.full(50, 1.5),   # KAC
            np.full(50, 0.5),   # AT2
        ])
        types = ["KAC"] * 50 + ["AT2"] * 50

        result = compute_plasticity_by_cell_type(scores, types)

        assert "KAC" in result
        assert "AT2" in result
        assert result["KAC"]["mean"] > result["AT2"]["mean"]
        assert result["KAC"]["n_cells"] == 50

    def test_ranking(self):
        """Should rank cell types by mean plasticity."""
        by_type = {
            "KAC": {"mean": 1.5, "median": 1.4},
            "AT2": {"mean": 0.5, "median": 0.5},
            "AT1": {"mean": 0.3, "median": 0.3},
        }

        ranking = rank_cell_types_by_plasticity(by_type, metric="mean")

        assert ranking[0][0] == "KAC"  # Highest
        assert ranking[-1][0] == "AT1"  # Lowest


class TestLogitsConversion:
    """Test stage logits to probability conversion."""

    def test_softmax_conversion(self):
        """Should apply softmax correctly."""
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        probs = stage_logits_to_probs(logits)

        # Check sums to 1
        assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-5)

        # Highest logit -> highest prob
        assert probs[0, 2] > probs[0, 1] > probs[0, 0]

    def test_temperature_scaling(self):
        """Higher temperature should produce softer distributions."""
        logits = torch.tensor([[1.0, 2.0, 3.0]])

        probs_t1 = stage_logits_to_probs(logits, temperature=1.0)
        probs_t2 = stage_logits_to_probs(logits, temperature=2.0)

        # Higher temperature -> more uniform -> higher entropy
        entropy_t1 = compute_plasticity_score(probs_t1)
        entropy_t2 = compute_plasticity_score(probs_t2)

        assert entropy_t2 > entropy_t1

    def test_plasticity_from_logits(self):
        """Test combined logits-to-plasticity function."""
        logits = torch.randn(10, 5)

        plasticity = compute_plasticity_from_logits(logits)

        assert plasticity.shape == (10,)
        assert torch.all(plasticity >= 0)


class TestPlasticityReport:
    """Test comprehensive report generation."""

    def test_basic_report(self):
        """Should generate report with basic statistics."""
        probs = torch.rand(100, 5)
        probs = probs / probs.sum(dim=-1, keepdim=True)

        report = generate_plasticity_report(probs)

        assert report.mean_plasticity > 0
        assert report.std_plasticity >= 0
        assert report.bifurcation_result is not None
        assert report.bifurcation_result.n_total_cells == 100

    def test_report_with_niche_resolution(self):
        """Should include niche resolution when provided."""
        n = 100
        probs_with = torch.rand(n, 5)
        probs_with = probs_with / probs_with.sum(dim=-1, keepdim=True)

        probs_without = probs_with + 0.1 * torch.randn(n, 5)
        probs_without = probs_without.abs()
        probs_without = probs_without / probs_without.sum(dim=-1, keepdim=True)

        report = generate_plasticity_report(
            probs_with,
            transition_probs_no_niche=probs_without,
        )

        assert report.niche_resolution is not None
        assert "mean_resolution" in report.niche_resolution

    def test_report_with_cell_types(self):
        """Should include cell type analysis when provided."""
        probs = torch.rand(100, 5)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        cell_types = ["AT2"] * 50 + ["KAC"] * 50

        report = generate_plasticity_report(probs, cell_types=cell_types)

        assert report.plasticity_by_cell_type is not None
        assert report.cell_type_ranking is not None
        assert report.bifurcation_enrichment is not None

    def test_report_with_niche_features(self):
        """Should include fate commitment when niche features provided."""
        probs = torch.rand(100, 5)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        il1b = torch.rand(100)

        report = generate_plasticity_report(
            probs,
            niche_features=il1b,
            niche_feature_name="IL1B",
        )

        assert report.fate_commitment is not None
        assert report.fate_commitment.feature_name == "IL1B"

    def test_report_to_dict_serialization(self):
        """Report should be JSON-serializable."""
        import json

        probs = torch.rand(50, 5)
        probs = probs / probs.sum(dim=-1, keepdim=True)

        report = generate_plasticity_report(
            probs,
            cell_types=["AT2"] * 25 + ["KAC"] * 25,
            niche_features=torch.rand(50),
        )

        d = report_to_dict(report)

        # Should be JSON serializable
        json_str = json.dumps(d)
        assert len(json_str) > 0

        # Should contain expected keys
        assert "mean_plasticity" in d
        assert "bifurcation" in d
        assert "fate_commitment" in d
