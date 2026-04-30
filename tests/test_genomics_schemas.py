"""Tests for genomics schemas."""

import pytest

from stagebridge.genomics.schemas import (
    VariantRecord,
    GermlineAnnotation,
    SomaticActionability,
    ClonalityEstimate,
    SpatialVariantEvidence,
    TransitionGenomicEnrichment,
)


class TestVariantRecord:
    """Tests for VariantRecord dataclass."""

    def test_basic_creation(self):
        """Test basic variant record creation."""
        record = VariantRecord(
            sample_id="S001",
            donor_id="D001",
            chromosome="chr7",
            position=55249071,
            reference_allele="C",
            alternate_allele="T",
            gene="EGFR",
        )
        assert record.chromosome == "chr7"
        assert record.position == 55249071
        assert record.variant_id == "chr7:55249071:C:T"

    def test_variant_id_property(self):
        """Test variant_id property format."""
        record = VariantRecord(
            sample_id="S001",
            donor_id="D001",
            chromosome="chr17",
            position=7577121,
            reference_allele="G",
            alternate_allele="A",
        )
        assert record.variant_id == "chr17:7577121:G:A"

    def test_missing_chromosome_raises(self):
        """Test that missing chromosome raises ValueError."""
        with pytest.raises(ValueError, match="chromosome cannot be empty"):
            VariantRecord(
                sample_id="S001",
                donor_id="D001",
                chromosome="",
                position=100,
                reference_allele="A",
                alternate_allele="T",
            )

    def test_invalid_position_raises(self):
        """Test that non-positive position raises ValueError."""
        with pytest.raises(ValueError, match="position must be positive"):
            VariantRecord(
                sample_id="S001",
                donor_id="D001",
                chromosome="chr1",
                position=0,
                reference_allele="A",
                alternate_allele="T",
            )

    def test_empty_allele_raises(self):
        """Test that empty alleles raise ValueError."""
        with pytest.raises(ValueError, match="reference_allele cannot be empty"):
            VariantRecord(
                sample_id="S001",
                donor_id="D001",
                chromosome="chr1",
                position=100,
                reference_allele="",
                alternate_allele="T",
            )


class TestGermlineAnnotation:
    """Tests for GermlineAnnotation dataclass."""

    def test_basic_creation(self):
        """Test basic germline annotation creation."""
        ann = GermlineAnnotation(
            variant_id="chr17:43093850:C:T",
            gene="BRCA1",
            acmg_aligned_classification="pathogenic",
        )
        assert ann.gene == "BRCA1"
        assert ann.acmg_aligned_classification == "pathogenic"

    def test_default_notes_included(self):
        """Test that default caution notes are included."""
        ann = GermlineAnnotation(
            variant_id="chr17:43093850:C:T",
            gene="BRCA1",
        )
        assert "ACMG/AMP-aligned" in ann.notes
        assert "clinical genetics" in ann.notes

    def test_cancer_predisposition_gene_flag(self):
        """Test cancer predisposition gene flag."""
        ann = GermlineAnnotation(
            variant_id="chr17:43093850:C:T",
            gene="BRCA1",
            cancer_predisposition_gene=True,
        )
        assert ann.cancer_predisposition_gene is True


class TestSomaticActionability:
    """Tests for SomaticActionability dataclass."""

    def test_basic_creation(self):
        """Test basic somatic actionability creation."""
        ann = SomaticActionability(
            variant_id="chr7:55249071:C:T",
            gene="EGFR",
            oncogenicity="oncogenic",
            actionability_level="level_1",
        )
        assert ann.gene == "EGFR"
        assert ann.actionability_level == "level_1"

    def test_therapeutic_implication(self):
        """Test therapeutic implication field."""
        ann = SomaticActionability(
            variant_id="chr7:55249071:C:T",
            gene="EGFR",
            therapeutic_implication="Osimertinib",
            knowledgebase_source="OncoKB",
        )
        assert ann.therapeutic_implication == "Osimertinib"
        assert ann.knowledgebase_source == "OncoKB"


class TestClonalityEstimate:
    """Tests for ClonalityEstimate dataclass."""

    def test_basic_creation(self):
        """Test basic clonality estimate creation."""
        est = ClonalityEstimate(
            variant_id="chr17:7577121:G:A",
            sample_id="S001",
            tumor_vaf=0.45,
            clonality_label="clonal_like",
        )
        assert est.tumor_vaf == 0.45
        assert est.clonality_label == "clonal_like"

    def test_naive_vaf_notes(self):
        """Test that naive VAF method gets appropriate notes."""
        est = ClonalityEstimate(
            variant_id="chr17:7577121:G:A",
            sample_id="S001",
            tumor_vaf=0.45,
            method="naive_vaf",
        )
        assert "naive VAF" in est.notes
        assert "Approximate" in est.notes


class TestSpatialVariantEvidence:
    """Tests for SpatialVariantEvidence dataclass."""

    def test_basic_creation(self):
        """Test basic spatial evidence creation."""
        ev = SpatialVariantEvidence(
            variant_id="chr7:55249071:C:T",
            sample_id="S001",
            barcode="AAACGC-1",
            ref_count=5,
            alt_count=3,
        )
        assert ev.total_count == 8
        assert ev.expressed_alt_fraction == 3 / 8

    def test_caution_included(self):
        """Test that caution language is included."""
        ev = SpatialVariantEvidence(
            variant_id="chr7:55249071:C:T",
            sample_id="S001",
            barcode="AAACGC-1",
        )
        assert "RNA coverage absence" in ev.caution
        assert "NOT mutation absence" in ev.caution

    def test_no_coverage_fraction(self):
        """Test that no coverage results in None fraction."""
        ev = SpatialVariantEvidence(
            variant_id="chr7:55249071:C:T",
            sample_id="S001",
            barcode="AAACGC-1",
            ref_count=0,
            alt_count=0,
        )
        assert ev.total_count == 0
        assert ev.expressed_alt_fraction is None


class TestTransitionGenomicEnrichment:
    """Tests for TransitionGenomicEnrichment dataclass."""

    def test_basic_creation(self):
        """Test basic enrichment result creation."""
        result = TransitionGenomicEnrichment(
            sample_id="S001",
            donor_id="D001",
            comparison="high_vs_low_transition",
            feature_name="TP53_mutated",
            high_transition_mean=0.45,
            low_transition_mean=0.12,
            effect_size=0.85,
            p_value=0.001,
            q_value=0.005,
            n_high=50,
            n_low=150,
        )
        assert result.feature_name == "TP53_mutated"
        assert result.effect_size == 0.85
        assert result.p_value == 0.001
