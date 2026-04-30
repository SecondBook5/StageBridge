"""ACMG/AMP-aligned germline pathogenicity annotation.

IMPORTANT: This module provides ACMG-aligned interpretations, NOT
clinical-grade ACMG adjudications. Automated output is labeled as
"ACMG-aligned" and requires clinical genetics review.

Uses ClinVar significance when available, with conservative fallbacks
for variants without ClinVar annotations.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from stagebridge.genomics.schemas import GermlineAnnotation, ACMGClassification

logger = logging.getLogger(__name__)

CANCER_PREDISPOSITION_GENES_DEFAULT = {
    "BRCA1", "BRCA2", "TP53", "PTEN", "STK11", "CDH1", "PALB2",
    "CHEK2", "ATM", "NBN", "BARD1", "BRIP1", "RAD51C", "RAD51D",
    "MLH1", "MSH2", "MSH6", "PMS2", "EPCAM",  # Lynch syndrome
    "APC", "MUTYH",  # FAP/MAP
    "RB1", "CDKN2A", "CDK4",  # Melanoma/retinoblastoma
    "MEN1", "RET", "VHL", "SDHB", "SDHD",  # Endocrine
    "BAP1", "NF1", "NF2",  # Other
    "EGFR",  # Lung cancer familial
}


def load_cancer_predisposition_genes(path: str | Path | None = None) -> set[str]:
    """Load cancer predisposition gene list.

    Args:
        path: Optional path to gene list file (one gene per line)

    Returns:
        Set of gene symbols
    """
    if path is None:
        return CANCER_PREDISPOSITION_GENES_DEFAULT.copy()

    path = Path(path)
    if not path.exists():
        logger.warning(f"Gene list not found: {path}, using defaults")
        return CANCER_PREDISPOSITION_GENES_DEFAULT.copy()

    genes = set()
    with open(path) as f:
        for line in f:
            gene = line.strip().upper()
            if gene and not gene.startswith("#"):
                genes.add(gene)
    return genes


def map_clinvar_to_acmg_aligned(clinvar_sig: str | None) -> ACMGClassification:
    """Map ClinVar clinical significance to ACMG-aligned classification.

    Args:
        clinvar_sig: ClinVar clinical significance string

    Returns:
        ACMG-aligned classification
    """
    if not clinvar_sig:
        return "unknown"

    sig_lower = clinvar_sig.lower()

    if "pathogenic" in sig_lower and "likely" not in sig_lower:
        if "conflicting" in sig_lower:
            return "vus"
        return "pathogenic"

    if "likely_pathogenic" in sig_lower or "likely pathogenic" in sig_lower:
        if "conflicting" in sig_lower:
            return "vus"
        return "likely_pathogenic"

    if "benign" in sig_lower and "likely" not in sig_lower:
        if "conflicting" in sig_lower:
            return "vus"
        return "benign"

    if "likely_benign" in sig_lower or "likely benign" in sig_lower:
        if "conflicting" in sig_lower:
            return "vus"
        return "likely_benign"

    if "uncertain" in sig_lower or "vus" in sig_lower:
        return "vus"

    if "conflicting" in sig_lower:
        return "vus"

    if "risk_factor" in sig_lower or "risk factor" in sig_lower:
        return "vus"

    return "unknown"


def classify_acmg_aligned(
    row: pd.Series,
    clinvar_col: str = "clinvar_significance",
    gene_col: str = "gene",
    cancer_genes: set[str] | None = None,
) -> GermlineAnnotation:
    """Classify a single variant using ACMG-aligned rules.

    Args:
        row: DataFrame row with variant information
        clinvar_col: Column name for ClinVar significance
        gene_col: Column name for gene symbol
        cancer_genes: Set of cancer predisposition genes

    Returns:
        GermlineAnnotation with ACMG-aligned classification
    """
    if cancer_genes is None:
        cancer_genes = CANCER_PREDISPOSITION_GENES_DEFAULT

    variant_id = row.get("variant_id", "unknown")
    gene = row.get(gene_col, "") or ""
    gene = gene.upper() if isinstance(gene, str) else ""

    clinvar_sig = row.get(clinvar_col)
    if pd.isna(clinvar_sig):
        clinvar_sig = None

    acmg_class = map_clinvar_to_acmg_aligned(clinvar_sig)

    evidence_codes = []
    if clinvar_sig:
        evidence_codes.append(f"ClinVar:{clinvar_sig}")

    is_cancer_gene = gene in cancer_genes
    if is_cancer_gene:
        evidence_codes.append("cancer_predisposition_gene")

    inheritance = row.get("inheritance")
    if pd.isna(inheritance):
        inheritance = None

    return GermlineAnnotation(
        variant_id=variant_id,
        gene=gene,
        clinvar_significance=clinvar_sig,
        acmg_aligned_classification=acmg_class,
        acmg_evidence_codes=evidence_codes,
        inheritance=inheritance,
        cancer_predisposition_gene=is_cancer_gene,
        notes=(
            "ACMG/AMP-aligned interpretation based on ClinVar. "
            "Not a clinical-grade classification. Requires clinical genetics review."
        ),
    )


def annotate_germline_acmg_aligned(
    variant_df: pd.DataFrame,
    clinvar_df: pd.DataFrame | None = None,
    cancer_gene_list: str | Path | None = None,
) -> pd.DataFrame:
    """Annotate germline variants with ACMG-aligned classifications.

    Args:
        variant_df: DataFrame with germline variants
        clinvar_df: Optional ClinVar annotations DataFrame
        cancer_gene_list: Optional path to cancer gene list

    Returns:
        DataFrame with ACMG-aligned annotations
    """
    df = variant_df.copy()
    cancer_genes = load_cancer_predisposition_genes(cancer_gene_list)

    if clinvar_df is not None:
        clinvar_df = clinvar_df.copy()

        if "variant_id" not in clinvar_df.columns:
            if all(c in clinvar_df.columns for c in ["chromosome", "position", "reference_allele", "alternate_allele"]):
                from stagebridge.genomics.vcf_io import normalize_variant_id
                clinvar_df["variant_id"] = clinvar_df.apply(
                    lambda r: normalize_variant_id(
                        r["chromosome"], r["position"],
                        r["reference_allele"], r["alternate_allele"]
                    ),
                    axis=1,
                )

        clinvar_cols = ["variant_id"]
        for col in ["clinvar_significance", "CLNSIG", "ClinicalSignificance", "clinical_significance"]:
            if col in clinvar_df.columns:
                clinvar_cols.append(col)
                break

        if len(clinvar_cols) > 1:
            clinvar_merge = clinvar_df[clinvar_cols].drop_duplicates(subset=["variant_id"])
            sig_col = clinvar_cols[1]
            clinvar_merge = clinvar_merge.rename(columns={sig_col: "clinvar_significance"})
            df = df.merge(clinvar_merge, on="variant_id", how="left")

    if "clinvar_significance" not in df.columns:
        df["clinvar_significance"] = None

    annotations = []
    for _, row in df.iterrows():
        ann = classify_acmg_aligned(row, cancer_genes=cancer_genes)
        annotations.append({
            "variant_id": ann.variant_id,
            "gene": ann.gene,
            "clinvar_significance": ann.clinvar_significance,
            "acmg_aligned_classification": ann.acmg_aligned_classification,
            "acmg_evidence_codes": ";".join(ann.acmg_evidence_codes),
            "inheritance": ann.inheritance,
            "cancer_predisposition_gene": ann.cancer_predisposition_gene,
            "notes": ann.notes,
        })

    result = pd.DataFrame(annotations)

    result = df[["variant_id", "sample_id", "donor_id", "gene"]].merge(
        result.drop(columns=["gene"], errors="ignore"),
        on="variant_id",
        how="left",
    )

    return result


def summarize_germline_annotations(df: pd.DataFrame) -> dict:
    """Summarize germline ACMG-aligned annotations.

    Returns:
        Dictionary with summary statistics
    """
    summary = {
        "total_variants": len(df),
        "by_classification": {},
        "cancer_predisposition_genes": 0,
        "pathogenic_or_likely": 0,
    }

    if "acmg_aligned_classification" in df.columns:
        counts = df["acmg_aligned_classification"].value_counts().to_dict()
        summary["by_classification"] = counts
        summary["pathogenic_or_likely"] = counts.get("pathogenic", 0) + counts.get("likely_pathogenic", 0)

    if "cancer_predisposition_gene" in df.columns:
        summary["cancer_predisposition_genes"] = df["cancer_predisposition_gene"].sum()

    return summary
