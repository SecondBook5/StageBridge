"""Unified variant annotation module.

Combines germline and somatic annotations with variant metadata.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from stagebridge.genomics.vcf_io import (
    read_vcf,
    read_annotated_variant_table,
    add_variant_ids,
    handle_multiallelic,
    validate_variant_table,
)
from stagebridge.genomics.acmg import annotate_germline_acmg_aligned
from stagebridge.genomics.somatic_actionability import annotate_somatic_actionability

logger = logging.getLogger(__name__)


def build_variant_master_table(
    somatic_vcf: str | Path | None = None,
    germline_vcf: str | Path | None = None,
    annotated_table: str | Path | None = None,
    sample_id: str | None = None,
    donor_id: str | None = None,
) -> pd.DataFrame:
    """Build unified variant master table from multiple sources.

    Args:
        somatic_vcf: Path to somatic VCF
        germline_vcf: Path to germline VCF
        annotated_table: Path to pre-annotated variant table
        sample_id: Sample ID to assign
        donor_id: Donor ID to assign

    Returns:
        Combined variant DataFrame
    """
    all_variants = []

    if somatic_vcf:
        logger.info(f"Loading somatic variants from {somatic_vcf}")
        for record in read_vcf(somatic_vcf, sample_id=sample_id, donor_id=donor_id, is_germline=False):
            all_variants.append({
                "sample_id": record.sample_id,
                "donor_id": record.donor_id,
                "chromosome": record.chromosome,
                "position": record.position,
                "reference_allele": record.reference_allele,
                "alternate_allele": record.alternate_allele,
                "gene": record.gene,
                "transcript": record.transcript,
                "consequence": record.consequence,
                "protein_change": record.protein_change,
                "variant_type": record.variant_type,
                "source": "somatic_vcf",
                "is_germline": False,
                "is_somatic": True,
                "tumor_vaf": record.tumor_vaf,
                "normal_vaf": record.normal_vaf,
                "tumor_depth": record.tumor_depth,
                "normal_depth": record.normal_depth,
                "filter_status": record.filter_status,
            })

    if germline_vcf:
        logger.info(f"Loading germline variants from {germline_vcf}")
        for record in read_vcf(germline_vcf, sample_id=sample_id, donor_id=donor_id, is_germline=True):
            all_variants.append({
                "sample_id": record.sample_id,
                "donor_id": record.donor_id,
                "chromosome": record.chromosome,
                "position": record.position,
                "reference_allele": record.reference_allele,
                "alternate_allele": record.alternate_allele,
                "gene": record.gene,
                "transcript": record.transcript,
                "consequence": record.consequence,
                "protein_change": record.protein_change,
                "variant_type": record.variant_type,
                "source": "germline_vcf",
                "is_germline": True,
                "is_somatic": False,
                "tumor_vaf": record.tumor_vaf,
                "normal_vaf": record.normal_vaf,
                "tumor_depth": record.tumor_depth,
                "normal_depth": record.normal_depth,
                "filter_status": record.filter_status,
            })

    if annotated_table:
        logger.info(f"Loading annotated variants from {annotated_table}")
        table_df = read_annotated_variant_table(annotated_table)

        is_valid, errors = validate_variant_table(table_df)
        if not is_valid:
            for err in errors:
                logger.warning(f"Validation warning: {err}")

        if sample_id and "sample_id" not in table_df.columns:
            table_df["sample_id"] = sample_id
        if donor_id and "donor_id" not in table_df.columns:
            table_df["donor_id"] = donor_id

        table_df["source"] = "annotated_table"
        all_variants.extend(table_df.to_dict(orient="records"))

    if not all_variants:
        logger.warning("No variants loaded from any source")
        return pd.DataFrame()

    df = pd.DataFrame(all_variants)

    df = handle_multiallelic(df)
    df = add_variant_ids(df)

    n_dupes = df.duplicated(subset=["variant_id", "sample_id"]).sum()
    if n_dupes > 0:
        logger.info(f"Removing {n_dupes} duplicate variant-sample pairs")
        df = df.drop_duplicates(subset=["variant_id", "sample_id"], keep="first")

    logger.info(f"Built variant master table with {len(df)} variants")

    return df


def annotate_all_variants(
    variant_df: pd.DataFrame,
    clinvar_df: pd.DataFrame | None = None,
    oncokb_df: pd.DataFrame | None = None,
    civic_df: pd.DataFrame | None = None,
    cancer_gene_list: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Annotate variants with germline and somatic interpretations.

    Args:
        variant_df: Variant master table
        clinvar_df: Optional ClinVar annotations
        oncokb_df: Optional OncoKB annotations
        civic_df: Optional CIViC annotations
        cancer_gene_list: Optional cancer gene list path

    Returns:
        Tuple of (germline_annotations_df, somatic_annotations_df)
    """
    germline_df = None
    somatic_df = None

    germline_mask = variant_df.get("is_germline", pd.Series(False, index=variant_df.index))
    if germline_mask.any():
        germline_variants = variant_df[germline_mask].copy()
        logger.info(f"Annotating {len(germline_variants)} germline variants")
        germline_df = annotate_germline_acmg_aligned(
            germline_variants,
            clinvar_df=clinvar_df,
            cancer_gene_list=cancer_gene_list,
        )

    somatic_mask = variant_df.get("is_somatic", pd.Series(False, index=variant_df.index))
    if not somatic_mask.any():
        somatic_mask = ~germline_mask

    if somatic_mask.any():
        somatic_variants = variant_df[somatic_mask].copy()
        logger.info(f"Annotating {len(somatic_variants)} somatic variants")
        somatic_df = annotate_somatic_actionability(
            somatic_variants,
            oncokb_df=oncokb_df,
            civic_df=civic_df,
        )

    return germline_df, somatic_df


def merge_annotations_with_master(
    variant_df: pd.DataFrame,
    germline_df: pd.DataFrame | None = None,
    somatic_df: pd.DataFrame | None = None,
    clonality_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge all annotations back to variant master table.

    Args:
        variant_df: Variant master table
        germline_df: Germline annotations
        somatic_df: Somatic annotations
        clonality_df: Clonality estimates

    Returns:
        Merged DataFrame with all annotations
    """
    df = variant_df.copy()

    if germline_df is not None:
        germline_cols = [
            "variant_id", "acmg_aligned_classification",
            "cancer_predisposition_gene", "clinvar_significance"
        ]
        germline_cols = [c for c in germline_cols if c in germline_df.columns]
        if len(germline_cols) > 1:
            df = df.merge(
                germline_df[germline_cols].drop_duplicates(subset=["variant_id"]),
                on="variant_id",
                how="left",
            )

    if somatic_df is not None:
        somatic_cols = [
            "variant_id", "oncogenicity", "actionability_level",
            "therapeutic_implication", "knowledgebase_source"
        ]
        somatic_cols = [c for c in somatic_cols if c in somatic_df.columns]
        if len(somatic_cols) > 1:
            df = df.merge(
                somatic_df[somatic_cols].drop_duplicates(subset=["variant_id"]),
                on="variant_id",
                how="left",
            )

    if clonality_df is not None:
        clonality_cols = [
            "variant_id", "sample_id", "clonality_label",
            "cancer_cell_fraction", "confidence"
        ]
        clonality_cols = [c for c in clonality_cols if c in clonality_df.columns]
        if len(clonality_cols) > 2:
            df = df.merge(
                clonality_df[clonality_cols],
                on=["variant_id", "sample_id"],
                how="left",
            )

    return df
