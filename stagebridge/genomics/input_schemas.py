"""Input schema contracts for genomic interpretation layer.

These define the required columns and dtypes for each input file type.
WES pipelines should produce outputs matching these schemas.

Uses the same ColumnSchema/ParquetSchema from contracts.py for consistency.
"""

from __future__ import annotations

from stagebridge.contracts import ColumnSchema, ParquetSchema


# -----------------------------------------------------------------------------
# Variant input schemas (from WES pipeline)
# -----------------------------------------------------------------------------

SOMATIC_VARIANT_SCHEMA = ParquetSchema(
    name="somatic_variants.parquet",
    description="Somatic variant calls from WES tumor-normal pipeline",
    columns=[
        # Required
        ColumnSchema("sample_id", "str", required=True, nullable=False, description="Sample identifier"),
        ColumnSchema("donor_id", "str", required=True, nullable=False, description="Donor/patient identifier"),
        ColumnSchema("chromosome", "str", required=True, nullable=False, description="Chromosome (chr1-22, chrX, chrY, chrM)"),
        ColumnSchema("position", "int", required=True, nullable=False, description="1-based genomic position"),
        ColumnSchema("reference_allele", "str", required=True, nullable=False, description="Reference allele"),
        ColumnSchema("alternate_allele", "str", required=True, nullable=False, description="Alternate allele"),
        ColumnSchema("gene", "str", required=True, nullable=True, description="Gene symbol"),
        # Recommended
        ColumnSchema("tumor_vaf", "float", required=False, nullable=True, description="Tumor VAF [0,1]"),
        ColumnSchema("tumor_depth", "int", required=False, nullable=True, description="Tumor read depth"),
        # Optional
        ColumnSchema("variant_id", "str", required=False, nullable=True, description="chr:pos:ref:alt format"),
        ColumnSchema("transcript", "str", required=False, nullable=True, description="Transcript ID"),
        ColumnSchema("consequence", "str", required=False, nullable=True, description="VEP consequence"),
        ColumnSchema("protein_change", "str", required=False, nullable=True, description="HGVS protein change"),
        ColumnSchema("quality", "float", required=False, nullable=True, description="Variant quality"),
        ColumnSchema("filter_status", "str", required=False, nullable=True, description="FILTER field"),
    ],
)

GERMLINE_VARIANT_SCHEMA = ParquetSchema(
    name="germline_variants.parquet",
    description="Germline variant calls from WES (matched normal or germline-only)",
    columns=[
        # Required
        ColumnSchema("donor_id", "str", required=True, nullable=False, description="Donor/patient identifier"),
        ColumnSchema("chromosome", "str", required=True, nullable=False, description="Chromosome"),
        ColumnSchema("position", "int", required=True, nullable=False, description="1-based genomic position"),
        ColumnSchema("reference_allele", "str", required=True, nullable=False, description="Reference allele"),
        ColumnSchema("alternate_allele", "str", required=True, nullable=False, description="Alternate allele"),
        ColumnSchema("gene", "str", required=True, nullable=True, description="Gene symbol"),
        # Recommended
        ColumnSchema("normal_vaf", "float", required=False, nullable=True, description="Normal VAF [0,1]"),
        ColumnSchema("normal_depth", "int", required=False, nullable=True, description="Normal read depth"),
        # Optional
        ColumnSchema("sample_id", "str", required=False, nullable=True, description="Sample identifier"),
        ColumnSchema("variant_id", "str", required=False, nullable=True, description="chr:pos:ref:alt format"),
    ],
)


# -----------------------------------------------------------------------------
# Annotation input schemas (external knowledgebases)
# -----------------------------------------------------------------------------

CLINVAR_SCHEMA = ParquetSchema(
    name="clinvar.parquet",
    description="ClinVar annotation lookup table",
    columns=[
        ColumnSchema("variant_id", "str", required=True, nullable=False, description="chr:pos:ref:alt format"),
        ColumnSchema("gene", "str", required=False, nullable=True, description="Gene symbol"),
        ColumnSchema("clinical_significance", "str", required=True, nullable=False, description="ClinVar significance"),
        ColumnSchema("review_status", "str", required=False, nullable=True, description="Review status"),
        ColumnSchema("conditions", "str", required=False, nullable=True, description="Associated conditions"),
    ],
)

ONCOKB_SCHEMA = ParquetSchema(
    name="oncokb.parquet",
    description="OncoKB annotation lookup table",
    columns=[
        ColumnSchema("gene", "str", required=True, nullable=False, description="Gene symbol"),
        ColumnSchema("alteration", "str", required=True, nullable=False, description="Mutation (e.g., V600E)"),
        ColumnSchema("oncogenicity", "str", required=True, nullable=False, description="Oncogenic/Likely Oncogenic/etc"),
        ColumnSchema("level", "str", required=False, nullable=True, description="Actionability level"),
        ColumnSchema("drugs", "str", required=False, nullable=True, description="Associated therapies"),
        ColumnSchema("cancer_type", "str", required=False, nullable=True, description="Cancer type context"),
    ],
)


# -----------------------------------------------------------------------------
# Spatial variant evidence schemas
# -----------------------------------------------------------------------------

SPATIAL_VARIANT_COUNTS_SCHEMA = ParquetSchema(
    name="spatial_variant_counts.parquet",
    description="Spatial variant read counts from VarTrix/cellSNP-lite",
    columns=[
        ColumnSchema("variant_id", "str", required=True, nullable=False, description="chr:pos:ref:alt format"),
        ColumnSchema("barcode", "str", required=True, nullable=False, description="Cell/spot barcode"),
        ColumnSchema("ref_count", "int", required=True, nullable=False, description="Reference read count"),
        ColumnSchema("alt_count", "int", required=True, nullable=False, description="Alternate read count"),
        ColumnSchema("sample_id", "str", required=False, nullable=True, description="Sample identifier"),
    ],
)

SPATIAL_METADATA_SCHEMA = ParquetSchema(
    name="spatial_metadata.parquet",
    description="Spatial metadata (coordinates, annotations)",
    columns=[
        ColumnSchema("barcode", "str", required=True, nullable=False, description="Cell/spot barcode"),
        ColumnSchema("sample_id", "str", required=True, nullable=False, description="Sample identifier"),
        ColumnSchema("x", "float", required=True, nullable=False, description="Spatial X coordinate"),
        ColumnSchema("y", "float", required=True, nullable=False, description="Spatial Y coordinate"),
        ColumnSchema("cell_type", "str", required=False, nullable=True, description="Cell type annotation"),
        ColumnSchema("region", "str", required=False, nullable=True, description="Tissue region"),
    ],
)


# -----------------------------------------------------------------------------
# Purity/ploidy schema (for proper clonality estimation)
# -----------------------------------------------------------------------------

PURITY_PLOIDY_SCHEMA = ParquetSchema(
    name="purity_ploidy.parquet",
    description="Tumor purity and ploidy estimates from ABSOLUTE/ASCAT/PureCN",
    columns=[
        ColumnSchema("sample_id", "str", required=True, nullable=False, description="Sample identifier"),
        ColumnSchema("purity", "float", required=True, nullable=False, description="Tumor purity [0,1]"),
        ColumnSchema("ploidy", "float", required=False, nullable=True, description="Tumor ploidy"),
        ColumnSchema("method", "str", required=False, nullable=True, description="Tool used"),
    ],
)


# -----------------------------------------------------------------------------
# Genomics output schemas
# -----------------------------------------------------------------------------

GERMLINE_ANNOTATION_SCHEMA = ParquetSchema(
    name="germline_acmg_aligned.parquet",
    description="ACMG/AMP-aligned germline pathogenicity annotations (NOT clinical-grade)",
    columns=[
        ColumnSchema("variant_id", "str", required=True, nullable=False, description="chr:pos:ref:alt"),
        ColumnSchema("gene", "str", required=True, nullable=True, description="Gene symbol"),
        ColumnSchema("clinvar_significance", "str", required=False, nullable=True, description="Raw ClinVar significance"),
        ColumnSchema("acmg_aligned_classification", "str", required=True, nullable=False, description="pathogenic/likely_pathogenic/vus/likely_benign/benign/unknown"),
        ColumnSchema("cancer_predisposition_gene", "bool", required=True, nullable=False, description="Is known cancer predisposition gene"),
        ColumnSchema("notes", "str", required=True, nullable=False, description="Caution/interpretation notes"),
    ],
)

SOMATIC_ACTIONABILITY_SCHEMA = ParquetSchema(
    name="somatic_actionability.parquet",
    description="OncoKB/CIViC-style somatic actionability annotations",
    columns=[
        ColumnSchema("variant_id", "str", required=True, nullable=False, description="chr:pos:ref:alt"),
        ColumnSchema("gene", "str", required=True, nullable=True, description="Gene symbol"),
        ColumnSchema("oncogenicity", "str", required=True, nullable=False, description="oncogenic/likely_oncogenic/unknown/etc"),
        ColumnSchema("actionability_level", "str", required=True, nullable=False, description="level_1-4/R1-R2/unknown"),
        ColumnSchema("therapeutic_implication", "str", required=False, nullable=True, description="Drug/therapy"),
        ColumnSchema("knowledgebase_source", "str", required=True, nullable=False, description="OncoKB/CIViC/driver_gene_list/etc"),
    ],
)

CLONALITY_ESTIMATE_SCHEMA = ParquetSchema(
    name="clonality_estimates.parquet",
    description="Clonality estimates from WES VAF (naive or CCF-corrected)",
    columns=[
        ColumnSchema("variant_id", "str", required=True, nullable=False, description="chr:pos:ref:alt"),
        ColumnSchema("sample_id", "str", required=True, nullable=False, description="Sample identifier"),
        ColumnSchema("tumor_vaf", "float", required=True, nullable=False, description="Tumor VAF"),
        ColumnSchema("cancer_cell_fraction", "float", required=False, nullable=True, description="CCF if purity/CN available"),
        ColumnSchema("clonality_label", "str", required=True, nullable=False, description="clonal/clonal_like/intermediate/subclonal/subclonal_like"),
        ColumnSchema("method", "str", required=True, nullable=False, description="naive_vaf or ccf_corrected"),
        ColumnSchema("notes", "str", required=True, nullable=False, description="Method caveats"),
    ],
)

SPATIAL_VARIANT_EVIDENCE_SCHEMA = ParquetSchema(
    name="spatial_variant_evidence.parquet",
    description="RNA-based variant evidence (NOT mutation calling - localization only)",
    columns=[
        ColumnSchema("variant_id", "str", required=True, nullable=False, description="chr:pos:ref:alt"),
        ColumnSchema("barcode", "str", required=True, nullable=False, description="Cell/spot barcode"),
        ColumnSchema("sample_id", "str", required=True, nullable=False, description="Sample identifier"),
        ColumnSchema("ref_count", "int", required=True, nullable=False, description="Reference read count"),
        ColumnSchema("alt_count", "int", required=True, nullable=False, description="Alternate read count"),
        ColumnSchema("evidence_label", "str", required=True, nullable=False, description="alt_supported/weak_alt_evidence/ref_only_observed/no_coverage"),
        ColumnSchema("caution", "str", required=True, nullable=False, description="RNA absence != mutation absence warning"),
    ],
)

TRANSITION_ENRICHMENT_SCHEMA = ParquetSchema(
    name="transition_genomic_enrichment.parquet",
    description="Enrichment of genomic features in high vs low transition zones",
    columns=[
        ColumnSchema("feature_name", "str", required=True, nullable=False, description="Feature tested (e.g., TP53_mutated)"),
        ColumnSchema("category", "str", required=True, nullable=False, description="variant_evidence/actionability/clonality"),
        ColumnSchema("comparison", "str", required=True, nullable=False, description="high_vs_low_transition"),
        ColumnSchema("high_transition_mean", "float", required=True, nullable=False, description="Mean in high-transition group"),
        ColumnSchema("low_transition_mean", "float", required=True, nullable=False, description="Mean in low-transition group"),
        ColumnSchema("effect_size", "float", required=True, nullable=False, description="Cohen's d or log2(OR)"),
        ColumnSchema("p_value", "float", required=True, nullable=False, description="Raw p-value"),
        ColumnSchema("q_value", "float", required=True, nullable=False, description="BH-corrected FDR"),
        ColumnSchema("n_high", "int", required=True, nullable=False, description="N in high-transition group"),
        ColumnSchema("n_low", "int", required=True, nullable=False, description="N in low-transition group"),
        ColumnSchema("test_method", "str", required=True, nullable=False, description="mann_whitney/fisher_exact/chi_square"),
    ],
)


# -----------------------------------------------------------------------------
# Schema registry
# -----------------------------------------------------------------------------

GENOMICS_INPUT_SCHEMAS = {
    "somatic_variants": SOMATIC_VARIANT_SCHEMA,
    "germline_variants": GERMLINE_VARIANT_SCHEMA,
    "clinvar": CLINVAR_SCHEMA,
    "oncokb": ONCOKB_SCHEMA,
    "spatial_variant_counts": SPATIAL_VARIANT_COUNTS_SCHEMA,
    "spatial_metadata": SPATIAL_METADATA_SCHEMA,
    "purity_ploidy": PURITY_PLOIDY_SCHEMA,
}

GENOMICS_OUTPUT_SCHEMAS = {
    "germline_annotation": GERMLINE_ANNOTATION_SCHEMA,
    "somatic_actionability": SOMATIC_ACTIONABILITY_SCHEMA,
    "clonality_estimate": CLONALITY_ESTIMATE_SCHEMA,
    "spatial_variant_evidence": SPATIAL_VARIANT_EVIDENCE_SCHEMA,
    "transition_enrichment": TRANSITION_ENRICHMENT_SCHEMA,
}

ALL_GENOMICS_SCHEMAS = {**GENOMICS_INPUT_SCHEMAS, **GENOMICS_OUTPUT_SCHEMAS}
