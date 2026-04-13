"""Tests for clonal validation (H3 hypothesis testing)."""

import numpy as np
import pytest

from stagebridge.evaluation.clonal_validation import (
    validate_h3_1,
    validate_h3_2,
    run_clonal_validation,
    compute_transition_probability,
    compute_niche_influence_from_attention,
    SHARED_CLONE_PATTERNS,
    INDEPENDENT_PATTERNS,
)


class TestH3_1:
    """Tests for H3.1: transition probability vs shared clones."""

    def test_perfect_separation(self):
        """Test with perfect separation: high trans = shared clone."""
        n_cells = 100
        # Shared clone cells have high transition prob
        transition_probs = np.concatenate([
            np.random.uniform(0.7, 1.0, 50),  # shared clones (1a/1b)
            np.random.uniform(0.0, 0.3, 50),  # independent (2)
        ])
        patterns = ["1a"] * 25 + ["1b"] * 25 + ["2"] * 50

        result = validate_h3_1(transition_probs, patterns)

        assert result.auc > 0.8, "AUC should be high with perfect separation"
        assert result.odds_ratio > 2.0, "OR should be high with perfect separation"
        assert result.h3_1_supported, "H3.1 should be supported"

    def test_no_separation(self):
        """Test with no separation: random transition probs."""
        n_cells = 100
        transition_probs = np.random.uniform(0, 1, n_cells)
        patterns = ["1a"] * 25 + ["1b"] * 25 + ["2"] * 50

        result = validate_h3_1(transition_probs, patterns)

        # AUC should be around 0.5 (random)
        assert 0.3 < result.auc < 0.7, "AUC should be near 0.5 with no separation"

    def test_insufficient_data(self):
        """Test with insufficient data."""
        transition_probs = np.array([0.5, 0.6])
        patterns = ["1a", "unknown"]

        result = validate_h3_1(transition_probs, patterns)

        assert result.auc == 0.5, "AUC should be 0.5 with insufficient data"
        assert not result.h3_1_supported, "H3.1 should not be supported"

    def test_excludes_invalid_patterns(self):
        """Test that invalid patterns are excluded."""
        n_cells = 100
        transition_probs = np.random.uniform(0, 1, n_cells)
        # Mix of valid and invalid patterns
        patterns = (
            ["1a"] * 20 + ["1b"] * 20 + ["2"] * 20 +
            ["stable"] * 20 + ["unknown"] * 20
        )

        result = validate_h3_1(transition_probs, patterns)

        # Should only count 1a, 1b, 2 as valid (60 cells)
        assert result.n_cells_with_pattern == 60
        assert result.n_cells_total == 100


class TestH3_2:
    """Tests for H3.2: niche influence by pattern."""

    def test_pattern_1a_higher(self):
        """Test when pattern 1a has higher niche influence."""
        niche_influence = np.concatenate([
            np.random.uniform(0.7, 1.0, 30),  # 1a: high influence
            np.random.uniform(0.4, 0.6, 30),  # 1b: medium
            np.random.uniform(0.0, 0.3, 30),  # 2: low influence
        ])
        patterns = ["1a"] * 30 + ["1b"] * 30 + ["2"] * 30

        result = validate_h3_2(niche_influence, patterns)

        assert result.mean_influence_1a > result.mean_influence_2
        assert result.pvalue_1a_vs_2 < 0.05
        assert result.h3_2_supported, "H3.2 should be supported"

    def test_no_difference(self):
        """Test when no difference between patterns."""
        n_cells = 90
        niche_influence = np.random.uniform(0.4, 0.6, n_cells)
        patterns = ["1a"] * 30 + ["1b"] * 30 + ["2"] * 30

        result = validate_h3_2(niche_influence, patterns)

        # No significant difference expected
        assert result.pvalue_1a_vs_2 > 0.05

    def test_effect_size(self):
        """Test effect size calculation."""
        # Strong effect: 1a much higher than 2
        niche_influence = np.concatenate([
            np.ones(30) * 0.9,  # 1a
            np.ones(30) * 0.5,  # 1b
            np.ones(30) * 0.1,  # 2
        ])
        patterns = ["1a"] * 30 + ["1b"] * 30 + ["2"] * 30

        result = validate_h3_2(niche_influence, patterns)

        # Effect size should be large positive (1a > 2)
        assert result.effect_size > 0.5


class TestRunClonalValidation:
    """Tests for the main validation function."""

    def test_full_validation(self):
        """Test complete clonal validation pipeline."""
        n_cells = 150
        transition_probs = np.concatenate([
            np.random.uniform(0.6, 0.9, 50),  # shared clones
            np.random.uniform(0.1, 0.4, 50),  # independent
            np.random.uniform(0.3, 0.7, 50),  # ambiguous
        ])
        niche_influence = np.concatenate([
            np.random.uniform(0.6, 0.9, 50),  # 1a high
            np.random.uniform(0.3, 0.5, 50),  # 1b medium
            np.random.uniform(0.1, 0.3, 50),  # 2 low
        ])
        patterns = ["1a"] * 50 + ["1b"] * 50 + ["2"] * 50
        donor_ids = [f"P{i//10}" for i in range(n_cells)]

        report = run_clonal_validation(
            transition_probs,
            niche_influence,
            patterns,
            donor_ids,
        )

        assert report.h3_1 is not None
        assert report.h3_2 is not None
        assert report.n_cells_analyzed == 150
        assert report.n_donors_analyzed == 15
        assert set(report.patterns_found) == {"1a", "1b", "2"}

    def test_to_dict(self):
        """Test JSON serialization."""
        n_cells = 90
        transition_probs = np.random.uniform(0, 1, n_cells)
        niche_influence = np.random.uniform(0, 1, n_cells)
        patterns = ["1a"] * 30 + ["1b"] * 30 + ["2"] * 30

        report = run_clonal_validation(transition_probs, niche_influence, patterns)
        report_dict = report.to_dict()

        assert "h3_1" in report_dict
        assert "h3_2" in report_dict
        assert "h3_supported" in report_dict
        assert isinstance(report_dict["h3_1"]["auc"], float)


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_compute_transition_probability(self):
        """Test transition probability computation."""
        # 5 stages: Normal, AAH, AIS, MIA, LUAD
        n_cells = 10
        n_stages = 5
        stage_logits = np.random.randn(n_cells, n_stages)
        current_stage = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])

        trans_probs = compute_transition_probability(stage_logits, current_stage)

        assert trans_probs.shape == (n_cells,)
        assert np.all(trans_probs >= 0)
        assert np.all(trans_probs <= 1)
        # Stage 4 (LUAD) cells should have 0 transition prob
        assert trans_probs[4] == 0.0
        assert trans_probs[9] == 0.0

    def test_compute_niche_influence_from_attention(self):
        """Test niche influence from attention weights."""
        n_cells = 5
        n_heads = 4
        n_tokens = 9

        # Random attention weights
        attn = np.random.rand(n_cells, n_heads, n_tokens, n_tokens)
        # Normalize to valid attention (sum to 1 along last dim)
        attn = attn / attn.sum(axis=-1, keepdims=True)

        influence = compute_niche_influence_from_attention(attn)

        assert influence.shape == (n_cells,)
        assert np.all(influence >= 0)


class TestConstants:
    """Tests for module constants."""

    def test_pattern_sets(self):
        """Test pattern set definitions."""
        assert "1a" in SHARED_CLONE_PATTERNS
        assert "1b" in SHARED_CLONE_PATTERNS
        assert "2" in INDEPENDENT_PATTERNS
        assert "2" not in SHARED_CLONE_PATTERNS
        assert "1a" not in INDEPENDENT_PATTERNS
