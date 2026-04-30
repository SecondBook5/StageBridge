"""Tests for VCF I/O utilities."""

import pytest
import pandas as pd
from pathlib import Path

from stagebridge.genomics.vcf_io import (
    normalize_chromosome,
    normalize_variant_id,
    read_annotated_variant_table,
    validate_variant_table,
    add_variant_ids,
    handle_multiallelic,
)


FIXTURES_DIR = Path(__file__).parent / "fixtures" / "genomics"


class TestNormalizeChromosome:
    """Tests for chromosome normalization."""

    def test_already_prefixed(self):
        """Test chromosomes already with chr prefix."""
        assert normalize_chromosome("chr7") == "chr7"
        assert normalize_chromosome("chrX") == "chrX"

    def test_numeric_chromosome(self):
        """Test numeric chromosomes get prefixed."""
        assert normalize_chromosome("7") == "chr7"
        assert normalize_chromosome("17") == "chr17"

    def test_sex_chromosomes(self):
        """Test sex chromosomes."""
        assert normalize_chromosome("X") == "chrX"
        assert normalize_chromosome("Y") == "chrY"

    def test_mitochondrial(self):
        """Test mitochondrial chromosome."""
        assert normalize_chromosome("M") == "chrM"
        assert normalize_chromosome("MT") == "chrM"


class TestNormalizeVariantId:
    """Tests for variant ID normalization."""

    def test_basic_format(self):
        """Test basic variant ID format."""
        vid = normalize_variant_id("chr7", 55249071, "C", "T")
        assert vid == "chr7:55249071:C:T"

    def test_uppercase_alleles(self):
        """Test alleles are uppercased."""
        vid = normalize_variant_id("chr7", 100, "a", "t")
        assert vid == "chr7:100:A:T"

    def test_chromosome_normalized(self):
        """Test chromosome is normalized."""
        vid = normalize_variant_id("7", 55249071, "C", "T")
        assert vid == "chr7:55249071:C:T"


class TestReadAnnotatedVariantTable:
    """Tests for reading annotated variant tables."""

    def test_read_somatic_tsv(self):
        """Test reading somatic variants TSV."""
        df = read_annotated_variant_table(FIXTURES_DIR / "mini_somatic_variants.tsv")
        assert len(df) == 7
        assert "chromosome" in df.columns
        assert "gene" in df.columns

    def test_chromosome_normalization(self):
        """Test chromosomes are normalized on load."""
        df = read_annotated_variant_table(FIXTURES_DIR / "mini_somatic_variants.tsv")
        assert all(df["chromosome"].str.startswith("chr"))


class TestValidateVariantTable:
    """Tests for variant table validation."""

    def test_valid_table(self):
        """Test validation of valid table."""
        df = pd.DataFrame({
            "chromosome": ["chr7", "chr17"],
            "position": [55249071, 7577121],
            "reference_allele": ["C", "G"],
            "alternate_allele": ["T", "A"],
            "sample_id": ["S001", "S001"],
        })
        is_valid, errors = validate_variant_table(df)
        assert is_valid
        assert len(errors) == 0

    def test_missing_column(self):
        """Test validation catches missing columns."""
        df = pd.DataFrame({
            "chromosome": ["chr7"],
            "position": [55249071],
        })
        is_valid, errors = validate_variant_table(df)
        assert not is_valid
        assert any("reference_allele" in e for e in errors)

    def test_invalid_position(self):
        """Test validation catches invalid positions."""
        df = pd.DataFrame({
            "chromosome": ["chr7"],
            "position": [0],
            "reference_allele": ["C"],
            "alternate_allele": ["T"],
            "sample_id": ["S001"],
        })
        is_valid, errors = validate_variant_table(df)
        assert not is_valid
        assert any("non-positive" in e for e in errors)


class TestAddVariantIds:
    """Tests for adding variant IDs."""

    def test_adds_variant_id_column(self):
        """Test variant_id column is added."""
        df = pd.DataFrame({
            "chromosome": ["chr7", "chr17"],
            "position": [55249071, 7577121],
            "reference_allele": ["C", "G"],
            "alternate_allele": ["T", "A"],
        })
        result = add_variant_ids(df)
        assert "variant_id" in result.columns
        assert result["variant_id"].iloc[0] == "chr7:55249071:C:T"


class TestHandleMultiallelic:
    """Tests for multi-allelic record handling."""

    def test_splits_multiallelic(self):
        """Test multi-allelic records are split."""
        df = pd.DataFrame({
            "chromosome": ["chr7"],
            "position": [100],
            "reference_allele": ["C"],
            "alternate_allele": ["T,G"],
            "sample_id": ["S001"],
        })
        result = handle_multiallelic(df)
        assert len(result) == 2
        assert set(result["alternate_allele"]) == {"T", "G"}

    def test_single_allele_unchanged(self):
        """Test single-allele records are unchanged."""
        df = pd.DataFrame({
            "chromosome": ["chr7", "chr17"],
            "position": [100, 200],
            "reference_allele": ["C", "G"],
            "alternate_allele": ["T", "A"],
        })
        result = handle_multiallelic(df)
        assert len(result) == 2
