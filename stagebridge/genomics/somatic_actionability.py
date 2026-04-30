"""OncoKB/CIViC-style somatic actionability annotation.

Uses cancer-specific actionability frameworks for somatic variants.
Does NOT use ACMG as the primary somatic framework.

When OncoKB annotations are provided, uses those directly.
When not available, provides conservative gene-level annotations only.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from stagebridge.genomics.schemas import (
    SomaticActionability,
    OncogenicityLevel,
    ActionabilityLevel,
)

logger = logging.getLogger(__name__)

LUNG_CANCER_DRIVER_GENES = {
    "TP53", "EGFR", "KRAS", "STK11", "KEAP1",
    "ALK", "ROS1", "BRAF", "MET", "ERBB2", "RET",
    "NTRK1", "NTRK2", "NTRK3",
    "PIK3CA", "NF1", "RB1", "CDKN2A",
    "SMARCA4", "ARID1A", "SETD2",
}

ONCOKB_LEVEL_MAPPING = {
    "1": "level_1",
    "2": "level_2",
    "3A": "level_3A",
    "3B": "level_3B",
    "4": "level_4",
    "R1": "level_R1",
    "R2": "level_R2",
    "LEVEL_1": "level_1",
    "LEVEL_2": "level_2",
    "LEVEL_3A": "level_3A",
    "LEVEL_3B": "level_3B",
    "LEVEL_4": "level_4",
    "LEVEL_R1": "level_R1",
    "LEVEL_R2": "level_R2",
}

ONCOKB_ONCOGENICITY_MAPPING = {
    "Oncogenic": "oncogenic",
    "Likely Oncogenic": "likely_oncogenic",
    "Predicted Oncogenic": "predicted_oncogenic",
    "Likely Neutral": "likely_neutral",
    "Inconclusive": "inconclusive",
    "Unknown": "unknown",
}


def map_oncokb_level(level_str: str | None) -> ActionabilityLevel:
    """Map OncoKB level string to ActionabilityLevel."""
    if not level_str or pd.isna(level_str):
        return "unknown"
    level_str = str(level_str).strip().upper()
    return ONCOKB_LEVEL_MAPPING.get(level_str, "unknown")


def map_oncokb_oncogenicity(onco_str: str | None) -> OncogenicityLevel:
    """Map OncoKB oncogenicity string to OncogenicityLevel."""
    if not onco_str or pd.isna(onco_str):
        return "unknown"
    onco_str = str(onco_str).strip()
    return ONCOKB_ONCOGENICITY_MAPPING.get(onco_str, "unknown")


def classify_oncogenicity(
    row: pd.Series,
    gene_col: str = "gene",
    oncokb_onco_col: str = "oncogenicity",
    driver_genes: set[str] | None = None,
) -> OncogenicityLevel:
    """Classify oncogenicity for a variant.

    Uses OncoKB if available, otherwise conservative gene-level inference.
    """
    if driver_genes is None:
        driver_genes = LUNG_CANCER_DRIVER_GENES

    if oncokb_onco_col in row.index and not pd.isna(row.get(oncokb_onco_col)):
        return map_oncokb_oncogenicity(row[oncokb_onco_col])

    gene = row.get(gene_col, "") or ""
    gene = gene.upper() if isinstance(gene, str) else ""

    if gene in driver_genes:
        return "unknown"

    return "unknown"


def annotate_single_variant(
    row: pd.Series,
    oncokb_row: pd.Series | None = None,
    driver_genes: set[str] | None = None,
) -> SomaticActionability:
    """Annotate a single somatic variant.

    Args:
        row: Variant DataFrame row
        oncokb_row: Optional matching OncoKB annotation row
        driver_genes: Set of driver genes for gene-level annotation

    Returns:
        SomaticActionability annotation
    """
    if driver_genes is None:
        driver_genes = LUNG_CANCER_DRIVER_GENES

    variant_id = row.get("variant_id", "unknown")
    gene = row.get("gene", "") or ""
    gene = gene.upper() if isinstance(gene, str) else ""

    oncogenicity: OncogenicityLevel = "unknown"
    actionability: ActionabilityLevel = "unknown"
    therapeutic = None
    diagnostic = None
    prognostic = None
    resistance = None
    source = "none"
    notes = ""

    if oncokb_row is not None:
        source = "OncoKB"

        for col in ["oncogenicity", "ONCOGENICITY", "Oncogenicity"]:
            if col in oncokb_row.index and not pd.isna(oncokb_row.get(col)):
                oncogenicity = map_oncokb_oncogenicity(oncokb_row[col])
                break

        for col in ["HIGHEST_LEVEL", "highest_level", "Level", "level"]:
            if col in oncokb_row.index and not pd.isna(oncokb_row.get(col)):
                actionability = map_oncokb_level(oncokb_row[col])
                break

        for col in ["THERAPEUTIC", "therapeutic", "Treatments"]:
            if col in oncokb_row.index and not pd.isna(oncokb_row.get(col)):
                therapeutic = str(oncokb_row[col])
                break

        for col in ["DIAGNOSTIC", "diagnostic"]:
            if col in oncokb_row.index and not pd.isna(oncokb_row.get(col)):
                diagnostic = str(oncokb_row[col])
                break

        for col in ["PROGNOSTIC", "prognostic"]:
            if col in oncokb_row.index and not pd.isna(oncokb_row.get(col)):
                prognostic = str(oncokb_row[col])
                break

        for col in ["RESISTANCE", "resistance", "Resistance"]:
            if col in oncokb_row.index and not pd.isna(oncokb_row.get(col)):
                resistance = str(oncokb_row[col])
                break

    elif gene in driver_genes:
        source = "driver_gene_list"
        actionability = "gene_level_cancer_relevance"
        notes = (
            f"{gene} is a known cancer driver gene. However, variant-level "
            "actionability evidence is not available. Gene-level annotation only."
        )

    return SomaticActionability(
        variant_id=variant_id,
        gene=gene,
        oncogenicity=oncogenicity,
        actionability_level=actionability,
        therapeutic_implication=therapeutic,
        diagnostic_implication=diagnostic,
        prognostic_implication=prognostic,
        resistance_implication=resistance,
        knowledgebase_source=source,
        notes=notes,
    )


def annotate_somatic_actionability(
    variant_df: pd.DataFrame,
    oncokb_df: pd.DataFrame | None = None,
    civic_df: pd.DataFrame | None = None,
    manual_driver_list: str | Path | None = None,
) -> pd.DataFrame:
    """Annotate somatic variants with actionability information.

    Args:
        variant_df: DataFrame with somatic variants
        oncokb_df: Optional OncoKB annotations
        civic_df: Optional CIViC annotations (future support)
        manual_driver_list: Optional path to driver gene list

    Returns:
        DataFrame with actionability annotations
    """
    df = variant_df.copy()

    if manual_driver_list:
        driver_genes = set()
        with open(manual_driver_list) as f:
            for line in f:
                gene = line.strip().upper()
                if gene and not gene.startswith("#"):
                    driver_genes.add(gene)
    else:
        driver_genes = LUNG_CANCER_DRIVER_GENES

    oncokb_lookup = {}
    if oncokb_df is not None:
        oncokb_df = oncokb_df.copy()

        if "variant_id" not in oncokb_df.columns:
            if all(c in oncokb_df.columns for c in ["chromosome", "position", "reference_allele", "alternate_allele"]):
                from stagebridge.genomics.vcf_io import normalize_variant_id
                oncokb_df["variant_id"] = oncokb_df.apply(
                    lambda r: normalize_variant_id(
                        r["chromosome"], r["position"],
                        r["reference_allele"], r["alternate_allele"]
                    ),
                    axis=1,
                )

        if "variant_id" in oncokb_df.columns:
            for _, row in oncokb_df.iterrows():
                oncokb_lookup[row["variant_id"]] = row

    annotations = []
    for _, row in df.iterrows():
        variant_id = row.get("variant_id", "")
        oncokb_row = oncokb_lookup.get(variant_id)

        ann = annotate_single_variant(
            row,
            oncokb_row=oncokb_row,
            driver_genes=driver_genes,
        )

        annotations.append({
            "variant_id": ann.variant_id,
            "gene": ann.gene,
            "oncogenicity": ann.oncogenicity,
            "actionability_level": ann.actionability_level,
            "therapeutic_implication": ann.therapeutic_implication,
            "diagnostic_implication": ann.diagnostic_implication,
            "prognostic_implication": ann.prognostic_implication,
            "resistance_implication": ann.resistance_implication,
            "knowledgebase_source": ann.knowledgebase_source,
            "notes": ann.notes,
        })

    result = pd.DataFrame(annotations)

    result = df[["variant_id", "sample_id", "donor_id"]].merge(
        result,
        on="variant_id",
        how="left",
    )

    return result


def summarize_actionability_by_stage(
    variant_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize actionability by disease stage.

    Args:
        variant_df: DataFrame with actionability annotations
        metadata_df: DataFrame with stage labels

    Returns:
        Summary DataFrame
    """
    if "stage" in variant_df.columns:
        df = variant_df.copy()
    else:
        df = variant_df.merge(
            metadata_df[["sample_id", "stage"]].drop_duplicates(),
            on="sample_id",
            how="left",
        )

    summary = df.groupby(["stage", "actionability_level"]).size().reset_index(name="count")
    total_by_stage = df.groupby("stage").size().reset_index(name="total")
    summary = summary.merge(total_by_stage, on="stage")
    summary["fraction"] = summary["count"] / summary["total"]

    return summary


def summarize_actionability_by_sample(variant_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize actionability by sample.

    Returns:
        Summary DataFrame with one row per sample
    """
    summary = []
    for sample_id, group in variant_df.groupby("sample_id"):
        row = {
            "sample_id": sample_id,
            "n_variants": len(group),
            "n_actionable": (group["actionability_level"].isin([
                "level_1", "level_2", "level_3A", "level_3B", "level_4"
            ])).sum(),
            "n_oncogenic": (group["oncogenicity"].isin([
                "oncogenic", "likely_oncogenic"
            ])).sum(),
            "n_resistance": (group["actionability_level"].isin([
                "level_R1", "level_R2"
            ])).sum(),
            "genes_affected": ";".join(sorted(group["gene"].dropna().unique())),
        }
        summary.append(row)

    return pd.DataFrame(summary)
