"""Tests for variant annotation modules."""

import pytest
import pandas as pd
from pathlib import Path

from stagebridge.genomics.acmg import (
    map_clinvar_to_acmg_aligned,
    annotate_germline_acmg_aligned,
    load_cancer_predisposition_genes,
)
from stagebridge.genomics.somatic_actionability import (
    map_oncokb_level,
    map_oncokb_oncogenicity,
    annotate_somatic_actionability,
)


FIXTURES_DIR = Path(__file__).parent / "fixtures" / "genomics"


class TestACMGMapping:
    """Tests for ClinVar to ACMG-aligned mapping."""

    def test_pathogenic_mapping(self):
        """Test pathogenic maps correctly."""
        assert map_clinvar_to_acmg_aligned("Pathogenic") == "pathogenic"
        assert map_clinvar_to_acmg_aligned("pathogenic") == "pathogenic"

    def test_likely_pathogenic_mapping(self):
        """Test likely pathogenic maps correctly."""
        assert map_clinvar_to_acmg_aligned("Likely pathogenic") == "likely_pathogenic"
        assert map_clinvar_to_acmg_aligned("Likely_pathogenic") == "likely_pathogenic"

    def test_benign_mapping(self):
        """Test benign maps correctly."""
        assert map_clinvar_to_acmg_aligned("Benign") == "benign"
        assert map_clinvar_to_acmg_aligned("Likely benign") == "likely_benign"

    def test_vus_mapping(self):
        """Test VUS maps correctly."""
        assert map_clinvar_to_acmg_aligned("Uncertain significance") == "vus"
        assert map_clinvar_to_acmg_aligned("VUS") == "vus"

    def test_conflicting_maps_to_vus(self):
        """Test conflicting interpretations map to VUS."""
        assert map_clinvar_to_acmg_aligned("Conflicting interpretations") == "vus"
        assert map_clinvar_to_acmg_aligned("Pathogenic/Conflicting") == "vus"

    def test_none_maps_to_unknown(self):
        """Test None maps to unknown."""
        assert map_clinvar_to_acmg_aligned(None) == "unknown"
        assert map_clinvar_to_acmg_aligned("") == "unknown"


class TestCancerPredispositionGenes:
    """Tests for cancer predisposition gene loading."""

    def test_default_genes_loaded(self):
        """Test default gene list is loaded."""
        genes = load_cancer_predisposition_genes()
        assert "BRCA1" in genes
        assert "BRCA2" in genes
        assert "TP53" in genes
        assert "ATM" in genes


class TestAnnotateGermline:
    """Tests for germline annotation."""

    def test_annotate_with_clinvar(self):
        """Test germline annotation with ClinVar."""
        variant_df = pd.read_csv(FIXTURES_DIR / "mini_germline_variants.tsv", sep="\t")
        clinvar_df = pd.read_csv(FIXTURES_DIR / "mini_clinvar.tsv", sep="\t")

        from stagebridge.genomics.vcf_io import add_variant_ids
        variant_df = add_variant_ids(variant_df)

        result = annotate_germline_acmg_aligned(variant_df, clinvar_df=clinvar_df)

        assert len(result) > 0
        assert "acmg_aligned_classification" in result.columns

        brca1 = result[result["gene"] == "BRCA1"]
        if len(brca1) > 0:
            assert brca1["acmg_aligned_classification"].iloc[0] == "pathogenic"

    def test_cancer_gene_flagged(self):
        """Test cancer predisposition genes are flagged."""
        variant_df = pd.DataFrame({
            "variant_id": ["chr17:43093850:C:T"],
            "sample_id": ["S001"],
            "donor_id": ["D001"],
            "gene": ["BRCA1"],
        })

        result = annotate_germline_acmg_aligned(variant_df)

        assert result["cancer_predisposition_gene"].iloc[0] == True


class TestOncoKBMapping:
    """Tests for OncoKB level and oncogenicity mapping."""

    def test_level_mapping(self):
        """Test OncoKB level mapping."""
        assert map_oncokb_level("1") == "level_1"
        assert map_oncokb_level("LEVEL_1") == "level_1"
        assert map_oncokb_level("3A") == "level_3A"
        assert map_oncokb_level("R1") == "level_R1"

    def test_oncogenicity_mapping(self):
        """Test OncoKB oncogenicity mapping."""
        assert map_oncokb_oncogenicity("Oncogenic") == "oncogenic"
        assert map_oncokb_oncogenicity("Likely Oncogenic") == "likely_oncogenic"
        assert map_oncokb_oncogenicity("Unknown") == "unknown"


class TestAnnotateSomatic:
    """Tests for somatic actionability annotation."""

    def test_annotate_with_oncokb(self):
        """Test somatic annotation with OncoKB."""
        variant_df = pd.read_csv(FIXTURES_DIR / "mini_somatic_variants.tsv", sep="\t")
        oncokb_df = pd.read_csv(FIXTURES_DIR / "mini_oncokb.tsv", sep="\t")

        from stagebridge.genomics.vcf_io import add_variant_ids
        variant_df = add_variant_ids(variant_df)

        result = annotate_somatic_actionability(variant_df, oncokb_df=oncokb_df)

        assert len(result) > 0
        assert "actionability_level" in result.columns
        assert "oncogenicity" in result.columns

    def test_driver_gene_fallback(self):
        """Test driver gene fallback annotation."""
        variant_df = pd.DataFrame({
            "variant_id": ["chr11:534287:A:G"],
            "sample_id": ["S001"],
            "donor_id": ["D001"],
            "gene": ["STK11"],
        })

        result = annotate_somatic_actionability(variant_df)

        assert result["actionability_level"].iloc[0] == "gene_level_cancer_relevance"
        assert result["knowledgebase_source"].iloc[0] == "driver_gene_list"

    def test_unknown_gene_not_actionable(self):
        """Test unknown gene is not marked actionable."""
        variant_df = pd.DataFrame({
            "variant_id": ["chr1:100:A:T"],
            "sample_id": ["S001"],
            "donor_id": ["D001"],
            "gene": ["UNKNOWN_GENE"],
        })

        result = annotate_somatic_actionability(variant_df)

        assert result["actionability_level"].iloc[0] == "unknown"
