"""Tests for transition enrichment analysis."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from stagebridge.genomics.transition_enrichment import (
    load_transition_scores,
    define_high_transition_group,
    define_low_transition_group,
    compute_continuous_enrichment,
    compute_binary_enrichment,
    compute_feature_enrichment,
    adjust_pvalues_bh,
    generate_transition_genomic_enrichment_table,
)


FIXTURES_DIR = Path(__file__).parent / "fixtures" / "genomics"


class TestLoadTransitionScores:
    """Tests for loading transition scores."""

    def test_load_from_tsv(self):
        """Test loading transition scores from TSV."""
        df = load_transition_scores(FIXTURES_DIR / "mini_transition_scores.tsv")

        assert "barcode" in df.columns
        assert "transition_score" in df.columns
        assert len(df) == 16


class TestDefineGroups:
    """Tests for defining transition groups."""

    def test_high_transition_group(self):
        """Test high-transition group selection."""
        df = pd.DataFrame({
            "barcode": [f"b{i}" for i in range(10)],
            "transition_score": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        })

        high = define_high_transition_group(df, quantile=0.80)

        assert len(high) == 2
        assert all(high["transition_score"] >= 0.9)

    def test_low_transition_group(self):
        """Test low-transition group selection."""
        df = pd.DataFrame({
            "barcode": [f"b{i}" for i in range(10)],
            "transition_score": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        })

        low = define_low_transition_group(df, quantile=0.50)

        assert len(low) == 5
        assert all(low["transition_score"] < 0.55)


class TestContinuousEnrichment:
    """Tests for continuous feature enrichment."""

    def test_significant_difference(self):
        """Test detection of significant difference."""
        np.random.seed(42)
        high_values = np.random.normal(10, 1, 100)
        low_values = np.random.normal(5, 1, 100)

        p_value, effect_size, _ = compute_continuous_enrichment(high_values, low_values)

        assert p_value < 0.05
        assert effect_size > 0

    def test_no_difference(self):
        """Test non-significant when no difference."""
        np.random.seed(42)
        high_values = np.random.normal(5, 1, 100)
        low_values = np.random.normal(5, 1, 100)

        p_value, effect_size, _ = compute_continuous_enrichment(high_values, low_values)

        assert abs(effect_size) < 1


class TestBinaryEnrichment:
    """Tests for binary feature enrichment."""

    def test_enriched_binary(self):
        """Test detection of binary enrichment."""
        p_value, odds_ratio = compute_binary_enrichment(
            high_positive=80, high_total=100,
            low_positive=20, low_total=100,
        )

        assert p_value < 0.05
        assert odds_ratio > 1

    def test_no_enrichment(self):
        """Test no enrichment detected when equal."""
        p_value, odds_ratio = compute_binary_enrichment(
            high_positive=50, high_total=100,
            low_positive=50, low_total=100,
        )

        assert odds_ratio == pytest.approx(1.0, rel=0.1)


class TestFeatureEnrichment:
    """Tests for general feature enrichment."""

    def test_continuous_feature(self):
        """Test continuous feature enrichment."""
        np.random.seed(42)
        high_df = pd.DataFrame({"feature": np.random.normal(10, 1, 50)})
        low_df = pd.DataFrame({"feature": np.random.normal(5, 1, 150)})

        result = compute_feature_enrichment(high_df, low_df, "feature", "continuous")

        assert "p_value" in result
        assert "effect_size" in result
        assert result["n_high"] == 50
        assert result["n_low"] == 150

    def test_binary_feature(self):
        """Test binary feature enrichment."""
        high_df = pd.DataFrame({"feature": [1] * 40 + [0] * 10})
        low_df = pd.DataFrame({"feature": [1] * 30 + [0] * 70})

        result = compute_feature_enrichment(high_df, low_df, "feature", "binary")

        assert "p_value" in result
        assert "odds_ratio" in result

    def test_missing_feature(self):
        """Test handling of missing feature."""
        high_df = pd.DataFrame({"other": [1, 2, 3]})
        low_df = pd.DataFrame({"other": [4, 5, 6]})

        result = compute_feature_enrichment(high_df, low_df, "feature", "continuous")

        assert result["p_value"] == 1.0
        assert "not found" in result.get("error", "")


class TestAdjustPvalues:
    """Tests for FDR correction."""

    def test_bh_correction(self):
        """Test Benjamini-Hochberg correction."""
        df = pd.DataFrame({
            "feature": ["a", "b", "c"],
            "p_value": [0.01, 0.03, 0.05],
        })

        result = adjust_pvalues_bh(df)

        assert "q_value" in result.columns
        assert all(result["q_value"] >= result["p_value"])

    def test_handles_nan(self):
        """Test handling of NaN p-values."""
        df = pd.DataFrame({
            "feature": ["a", "b"],
            "p_value": [0.01, np.nan],
        })

        result = adjust_pvalues_bh(df)

        assert not pd.isna(result["q_value"].iloc[0])
        assert pd.isna(result["q_value"].iloc[1])


class TestGenerateEnrichmentTable:
    """Tests for comprehensive enrichment table generation."""

    def test_generates_table(self):
        """Test table generation with all inputs."""
        transition_df = pd.read_csv(FIXTURES_DIR / "mini_transition_scores.tsv", sep="\t")
        variant_counts = pd.read_csv(FIXTURES_DIR / "mini_spatial_variant_counts.tsv", sep="\t")

        from stagebridge.genomics.spatial_variant_evidence import annotate_spatial_variant_evidence
        variant_evidence = annotate_spatial_variant_evidence(variant_counts)

        result = generate_transition_genomic_enrichment_table(
            transition_df=transition_df,
            variant_evidence_df=variant_evidence,
        )

        assert len(result) > 0
        assert "feature_name" in result.columns
        assert "p_value" in result.columns
        assert "q_value" in result.columns
        assert "effect_size" in result.columns

    def test_empty_with_no_inputs(self):
        """Test returns empty DataFrame with no inputs."""
        transition_df = pd.read_csv(FIXTURES_DIR / "mini_transition_scores.tsv", sep="\t")

        result = generate_transition_genomic_enrichment_table(
            transition_df=transition_df,
        )

        assert len(result) == 0
