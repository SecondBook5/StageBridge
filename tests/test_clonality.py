"""Tests for clonality estimation."""

import pytest
import pandas as pd
from pathlib import Path

from stagebridge.genomics.clonality import (
    estimate_cancer_cell_fraction,
    classify_clonality,
    estimate_clonality_for_variants,
)


FIXTURES_DIR = Path(__file__).parent / "fixtures" / "genomics"


class TestCancerCellFraction:
    """Tests for CCF estimation."""

    def test_ccf_with_purity_and_cn(self):
        """Test CCF calculation with purity and copy number."""
        ccf = estimate_cancer_cell_fraction(
            vaf=0.5,
            purity=0.8,
            local_copy_number=2.0,
        )
        assert ccf is not None
        assert 0 <= ccf <= 1.5

    def test_ccf_none_without_purity(self):
        """Test CCF is None without purity."""
        ccf = estimate_cancer_cell_fraction(
            vaf=0.5,
            purity=None,
            local_copy_number=2.0,
        )
        assert ccf is None

    def test_ccf_none_without_cn(self):
        """Test CCF is None without copy number."""
        ccf = estimate_cancer_cell_fraction(
            vaf=0.5,
            purity=0.8,
            local_copy_number=None,
        )
        assert ccf is None


class TestClassifyClonality:
    """Tests for clonality classification."""

    def test_clonal_high_vaf(self):
        """Test high VAF classified as clonal_like."""
        label, conf, method = classify_clonality(vaf=0.45)
        assert label == "clonal_like"
        assert method == "naive_vaf"

    def test_intermediate_vaf(self):
        """Test intermediate VAF classification."""
        label, conf, method = classify_clonality(vaf=0.20)
        assert label == "intermediate"
        assert method == "naive_vaf"

    def test_subclonal_low_vaf(self):
        """Test low VAF classified as subclonal_like."""
        label, conf, method = classify_clonality(vaf=0.05)
        assert label == "subclonal_like"
        assert method == "naive_vaf"

    def test_very_low_vaf(self):
        """Test very low VAF classified as low_confidence."""
        label, conf, method = classify_clonality(vaf=0.01)
        assert label == "low_confidence"

    def test_ccf_based_classification(self):
        """Test CCF-based classification."""
        label, conf, method = classify_clonality(ccf=0.95)
        assert label == "clonal"
        assert method == "ccf"
        assert conf == "high"

    def test_ccf_subclonal(self):
        """Test CCF subclonal classification."""
        label, conf, method = classify_clonality(ccf=0.15)
        assert label == "subclonal"


class TestEstimateClonalityForVariants:
    """Tests for batch clonality estimation."""

    def test_estimate_without_purity(self):
        """Test clonality estimation without purity data."""
        variant_df = pd.read_csv(FIXTURES_DIR / "mini_somatic_variants.tsv", sep="\t")

        from stagebridge.genomics.vcf_io import add_variant_ids
        variant_df = add_variant_ids(variant_df)

        result = estimate_clonality_for_variants(variant_df)

        assert len(result) == len(variant_df)
        assert "clonality_label" in result.columns
        assert "method" in result.columns
        assert all(result["method"] == "naive_vaf")

    def test_estimate_with_purity(self):
        """Test clonality estimation with purity data."""
        variant_df = pd.read_csv(FIXTURES_DIR / "mini_somatic_variants.tsv", sep="\t")
        purity_df = pd.read_csv(FIXTURES_DIR / "mini_purity.tsv", sep="\t")

        from stagebridge.genomics.vcf_io import add_variant_ids
        variant_df = add_variant_ids(variant_df)

        result = estimate_clonality_for_variants(
            variant_df,
            purity_df=purity_df,
        )

        assert len(result) == len(variant_df)
        assert "purity" in result.columns

    def test_clonality_labels_correct(self):
        """Test clonality labels are assigned correctly."""
        variant_df = pd.DataFrame({
            "variant_id": ["v1", "v2", "v3"],
            "sample_id": ["S001", "S001", "S001"],
            "tumor_vaf": [0.50, 0.15, 0.03],
        })

        result = estimate_clonality_for_variants(variant_df)

        labels = result["clonality_label"].tolist()
        assert labels[0] == "clonal_like"
        assert labels[1] == "intermediate"
        assert labels[2] == "subclonal_like"
