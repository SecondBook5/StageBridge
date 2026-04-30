"""Tests for spatial variant evidence module."""

import pytest
import pandas as pd
from pathlib import Path

from stagebridge.genomics.spatial_variant_evidence import (
    classify_spatial_variant_evidence,
    annotate_spatial_variant_evidence,
    merge_variant_counts_with_spatial_metadata,
)


FIXTURES_DIR = Path(__file__).parent / "fixtures" / "genomics"


class TestClassifySpatialEvidence:
    """Tests for spatial evidence classification."""

    def test_alt_supported(self):
        """Test alt_supported classification."""
        label = classify_spatial_variant_evidence(ref_count=5, alt_count=3)
        assert label == "alt_supported"

    def test_weak_alt_evidence(self):
        """Test weak_alt_evidence classification."""
        label = classify_spatial_variant_evidence(ref_count=4, alt_count=1)
        assert label == "weak_alt_evidence"

    def test_ref_only_observed(self):
        """Test ref_only_observed classification."""
        label = classify_spatial_variant_evidence(ref_count=10, alt_count=0)
        assert label == "ref_only_observed"

    def test_no_coverage(self):
        """Test no_coverage classification."""
        label = classify_spatial_variant_evidence(ref_count=0, alt_count=0)
        assert label == "no_coverage"

    def test_low_coverage(self):
        """Test low_coverage classification."""
        label = classify_spatial_variant_evidence(ref_count=1, alt_count=0)
        assert label == "low_coverage"


class TestAnnotateSpatialEvidence:
    """Tests for spatial evidence annotation."""

    def test_annotate_counts(self):
        """Test annotation of variant counts."""
        counts_df = pd.read_csv(FIXTURES_DIR / "mini_spatial_variant_counts.tsv", sep="\t")

        result = annotate_spatial_variant_evidence(counts_df)

        assert "evidence_label" in result.columns
        assert "total_count" in result.columns
        assert "expressed_alt_fraction" in result.columns
        assert "caution" in result.columns

    def test_correct_labels_assigned(self):
        """Test correct evidence labels are assigned."""
        counts_df = pd.DataFrame({
            "variant_id": ["v1", "v2", "v3", "v4"],
            "barcode": ["b1", "b2", "b3", "b4"],
            "ref_count": [5, 4, 10, 0],
            "alt_count": [3, 1, 0, 0],
        })

        result = annotate_spatial_variant_evidence(counts_df)

        labels = result["evidence_label"].tolist()
        assert labels[0] == "alt_supported"
        assert labels[1] == "weak_alt_evidence"
        assert labels[2] == "ref_only_observed"
        assert labels[3] == "no_coverage"

    def test_caution_included(self):
        """Test caution language is included."""
        counts_df = pd.DataFrame({
            "variant_id": ["v1"],
            "barcode": ["b1"],
            "ref_count": [5],
            "alt_count": [3],
        })

        result = annotate_spatial_variant_evidence(counts_df)

        assert "RNA coverage absence" in result["caution"].iloc[0]


class TestMergeWithMetadata:
    """Tests for merging counts with spatial metadata."""

    def test_merge_adds_coordinates(self):
        """Test merge adds spatial coordinates."""
        counts_df = pd.read_csv(FIXTURES_DIR / "mini_spatial_variant_counts.tsv", sep="\t")
        meta_df = pd.read_csv(FIXTURES_DIR / "mini_spatial_metadata.tsv", sep="\t")

        result = merge_variant_counts_with_spatial_metadata(counts_df, meta_df)

        assert "x" in result.columns
        assert "y" in result.columns
        assert "sample_id" in result.columns

    def test_merge_handles_missing_barcodes(self):
        """Test merge handles barcodes not in metadata."""
        counts_df = pd.DataFrame({
            "variant_id": ["v1"],
            "barcode": ["UNKNOWN-1"],
            "ref_count": [5],
            "alt_count": [3],
        })
        meta_df = pd.read_csv(FIXTURES_DIR / "mini_spatial_metadata.tsv", sep="\t")

        result = merge_variant_counts_with_spatial_metadata(counts_df, meta_df)

        assert len(result) == 1
        assert pd.isna(result["x"].iloc[0])
