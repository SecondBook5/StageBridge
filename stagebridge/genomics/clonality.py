"""Clonality estimation from WES variant allele fractions.

Estimates clonal/subclonal status from tumor VAF, optionally corrected
for purity, copy number, and ploidy when available.

IMPORTANT: When purity/CNV/ploidy are unavailable, uses naive VAF
thresholds with explicit labeling as approximate.
"""

from __future__ import annotations

import logging
from typing import Literal

import pandas as pd

from stagebridge.genomics.schemas import ClonalityLabel

logger = logging.getLogger(__name__)

NAIVE_VAF_THRESHOLDS = {
    "clonal_like": 0.30,
    "intermediate": 0.10,
    "subclonal_like": 0.02,
}


def estimate_cancer_cell_fraction(
    vaf: float,
    purity: float | None = None,
    local_copy_number: float | None = None,
    multiplicity: int = 1,
    ploidy: float | None = None,
) -> float | None:
    """Estimate cancer cell fraction from VAF with purity/CNV correction.

    Uses the formula:
    CCF = VAF * (purity * local_CN + (1 - purity) * 2) / (purity * multiplicity)

    Args:
        vaf: Variant allele fraction in tumor
        purity: Tumor purity estimate (0-1)
        local_copy_number: Local copy number at variant site
        multiplicity: Number of copies carrying the mutation (usually 1)
        ploidy: Tumor ploidy (optional, for validation)

    Returns:
        Estimated cancer cell fraction, or None if insufficient data
    """
    if purity is None or local_copy_number is None:
        return None

    if purity <= 0 or purity > 1:
        logger.warning(f"Invalid purity: {purity}")
        return None

    if local_copy_number <= 0:
        logger.warning(f"Invalid copy number: {local_copy_number}")
        return None

    normal_cn = 2
    expected_vaf_clonal = (purity * multiplicity) / (
        purity * local_copy_number + (1 - purity) * normal_cn
    )

    if expected_vaf_clonal <= 0:
        return None

    ccf = vaf / expected_vaf_clonal

    ccf = min(1.5, max(0, ccf))

    return ccf


def classify_clonality(
    ccf: float | None = None,
    vaf: float | None = None,
    method: Literal["ccf", "naive_vaf", "auto"] = "auto",
    thresholds: dict | None = None,
) -> tuple[ClonalityLabel, str, str]:
    """Classify clonality from CCF or VAF.

    Args:
        ccf: Cancer cell fraction (if available)
        vaf: Variant allele fraction
        method: Classification method
        thresholds: Custom VAF thresholds

    Returns:
        Tuple of (clonality_label, confidence, method_used)
    """
    if thresholds is None:
        thresholds = NAIVE_VAF_THRESHOLDS

    if method == "auto":
        method = "ccf" if ccf is not None else "naive_vaf"

    if method == "ccf" and ccf is not None:
        if ccf >= 0.9:
            return "clonal", "high", "ccf"
        elif ccf >= 0.5:
            return "clonal", "medium", "ccf"
        elif ccf >= 0.2:
            return "subclonal", "medium", "ccf"
        elif ccf >= 0.05:
            return "subclonal", "low", "ccf"
        else:
            return "low_confidence", "low", "ccf"

    if vaf is None:
        return "unknown", "none", "none"

    if vaf >= thresholds["clonal_like"]:
        return "clonal_like", "low", "naive_vaf"
    elif vaf >= thresholds["intermediate"]:
        return "intermediate", "low", "naive_vaf"
    elif vaf >= thresholds["subclonal_like"]:
        return "subclonal_like", "low", "naive_vaf"
    else:
        return "low_confidence", "low", "naive_vaf"


def estimate_clonality_for_variants(
    variant_df: pd.DataFrame,
    purity_df: pd.DataFrame | None = None,
    cna_df: pd.DataFrame | None = None,
    vaf_col: str = "tumor_vaf",
    depth_col: str = "tumor_depth",
    min_depth: int = 20,
) -> pd.DataFrame:
    """Estimate clonality for all variants.

    Args:
        variant_df: DataFrame with variants
        purity_df: Optional purity estimates per sample
        cna_df: Optional copy number segments
        vaf_col: Column name for tumor VAF
        depth_col: Column name for tumor depth
        min_depth: Minimum depth for confident calls

    Returns:
        DataFrame with clonality estimates
    """
    df = variant_df.copy()

    purity_lookup = {}
    if purity_df is not None:
        for _, row in purity_df.iterrows():
            sample_id = row.get("sample_id")
            purity = row.get("purity")
            if sample_id and purity and not pd.isna(purity):
                purity_lookup[sample_id] = float(purity)

    cn_lookup = {}
    if cna_df is not None:
        for _, row in cna_df.iterrows():
            sample_id = row.get("sample_id")
            chrom = row.get("chromosome")
            start = row.get("start")
            end = row.get("end")
            cn = row.get("copy_number") or row.get("cn") or row.get("total_cn")
            if all(v is not None and not pd.isna(v) for v in [sample_id, chrom, start, end, cn]):
                key = (sample_id, chrom)
                if key not in cn_lookup:
                    cn_lookup[key] = []
                cn_lookup[key].append((int(start), int(end), float(cn)))

    def get_local_cn(sample_id: str, chrom: str, pos: int) -> float | None:
        key = (sample_id, chrom)
        if key not in cn_lookup:
            return None
        for start, end, cn in cn_lookup[key]:
            if start <= pos <= end:
                return cn
        return None

    results = []
    for _, row in df.iterrows():
        variant_id = row.get("variant_id", "unknown")
        sample_id = row.get("sample_id", "unknown")

        vaf = row.get(vaf_col)
        if pd.isna(vaf):
            vaf = None
        else:
            vaf = float(vaf)

        depth = row.get(depth_col)
        if pd.isna(depth):
            depth = None
        else:
            depth = int(depth)

        purity = purity_lookup.get(sample_id)
        local_cn = None
        ploidy = None

        if cna_df is not None and "chromosome" in row.index:
            local_cn = get_local_cn(
                sample_id,
                row.get("chromosome", ""),
                row.get("position", 0),
            )

        ccf = None
        if vaf is not None and purity is not None and local_cn is not None:
            ccf = estimate_cancer_cell_fraction(
                vaf=vaf,
                purity=purity,
                local_copy_number=local_cn,
            )

        label, confidence, method = classify_clonality(ccf=ccf, vaf=vaf)

        if depth is not None and depth < min_depth:
            if confidence != "none":
                confidence = "low"

        notes = ""
        if method == "naive_vaf":
            notes = (
                "Clonality estimated from naive VAF thresholds without "
                "purity/CNV correction. Approximate only."
            )
        elif method == "ccf":
            notes = f"CCF-based clonality (purity={purity:.2f}, CN={local_cn:.1f})"

        results.append({
            "variant_id": variant_id,
            "sample_id": sample_id,
            "tumor_vaf": vaf,
            "local_copy_number": local_cn,
            "purity": purity,
            "ploidy": ploidy,
            "cancer_cell_fraction": ccf,
            "clonality_label": label,
            "confidence": confidence,
            "method": method,
            "notes": notes,
        })

    return pd.DataFrame(results)


def summarize_clonality_by_stage(
    clonality_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize clonality distribution by disease stage.

    Args:
        clonality_df: DataFrame with clonality estimates
        metadata_df: DataFrame with stage labels

    Returns:
        Summary DataFrame
    """
    if "stage" in clonality_df.columns:
        df = clonality_df.copy()
    else:
        df = clonality_df.merge(
            metadata_df[["sample_id", "stage"]].drop_duplicates(),
            on="sample_id",
            how="left",
        )

    summary = df.groupby(["stage", "clonality_label"]).size().reset_index(name="count")

    total_by_stage = df.groupby("stage").size().reset_index(name="total")
    summary = summary.merge(total_by_stage, on="stage")
    summary["fraction"] = summary["count"] / summary["total"]

    return summary


def summarize_clonality_by_sample(clonality_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize clonality by sample.

    Returns:
        Summary DataFrame with one row per sample
    """
    summary = []
    for sample_id, group in clonality_df.groupby("sample_id"):
        n_total = len(group)
        clonal_labels = ["clonal", "clonal_like"]
        subclonal_labels = ["subclonal", "subclonal_like"]

        n_clonal = group["clonality_label"].isin(clonal_labels).sum()
        n_subclonal = group["clonality_label"].isin(subclonal_labels).sum()

        mean_vaf = group["tumor_vaf"].mean() if "tumor_vaf" in group.columns else None
        mean_ccf = None
        if "cancer_cell_fraction" in group.columns:
            ccf_vals = group["cancer_cell_fraction"].dropna()
            if len(ccf_vals) > 0:
                mean_ccf = ccf_vals.mean()

        summary.append({
            "sample_id": sample_id,
            "n_variants": n_total,
            "n_clonal": n_clonal,
            "n_subclonal": n_subclonal,
            "clonal_fraction": n_clonal / n_total if n_total > 0 else 0,
            "mean_tumor_vaf": mean_vaf,
            "mean_ccf": mean_ccf,
        })

    return pd.DataFrame(summary)
