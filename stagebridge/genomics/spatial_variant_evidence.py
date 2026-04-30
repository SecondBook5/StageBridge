"""Spatial variant evidence from RNA-based data.

CRITICAL SCIENTIFIC FRAMING:
- This module does NOT call new variants from spatial/snRNA data
- It only counts evidence for WES-confirmed variants
- Output is called "expressed variant evidence", not "mutation calls"
- Absence of alternate reads does NOT mean the mutation is absent
  (RNA coverage is sparse)
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import numpy as np

from stagebridge.genomics.schemas import SpatialVariantEvidence, EvidenceLabel

logger = logging.getLogger(__name__)

EVIDENCE_THRESHOLDS = {
    "alt_supported_min_alt_count": 2,
    "alt_supported_min_total_count": 5,
    "weak_alt_min_total_count": 3,
    "low_coverage_threshold": 3,
}


def classify_spatial_variant_evidence(
    ref_count: int,
    alt_count: int,
    thresholds: dict | None = None,
) -> EvidenceLabel:
    """Classify spatial variant evidence based on read counts.

    Args:
        ref_count: Number of reference-supporting reads
        alt_count: Number of alternate-supporting reads
        thresholds: Custom thresholds

    Returns:
        Evidence label
    """
    if thresholds is None:
        thresholds = EVIDENCE_THRESHOLDS

    total_count = ref_count + alt_count

    if total_count == 0:
        return "no_coverage"

    if total_count < thresholds["low_coverage_threshold"]:
        return "low_coverage"

    if (alt_count >= thresholds["alt_supported_min_alt_count"] and
            total_count >= thresholds["alt_supported_min_total_count"]):
        return "alt_supported"

    if alt_count >= 1 and total_count >= thresholds["weak_alt_min_total_count"]:
        return "weak_alt_evidence"

    if ref_count > 0 and alt_count == 0:
        return "ref_only_observed"

    return "low_coverage"


def prepare_variant_sites_for_counting(
    variant_df: pd.DataFrame,
    output_path: str | Path,
    format: str = "bed",
) -> Path:
    """Prepare variant sites file for counting tools.

    Args:
        variant_df: DataFrame with variant records
        output_path: Output file path
        format: Output format ('bed' or 'vcf')

    Returns:
        Path to output file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    required_cols = ["chromosome", "position", "reference_allele", "alternate_allele"]
    for col in required_cols:
        if col not in variant_df.columns:
            raise ValueError(f"Missing required column: {col}")

    df = variant_df[required_cols].drop_duplicates()

    if format == "bed":
        bed_df = pd.DataFrame({
            "chrom": df["chromosome"],
            "start": df["position"] - 1,
            "end": df["position"],
            "name": df.apply(
                lambda r: f"{r['chromosome']}:{r['position']}:{r['reference_allele']}:{r['alternate_allele']}",
                axis=1,
            ),
            "score": 0,
            "strand": ".",
        })
        bed_df.to_csv(output_path, sep="\t", header=False, index=False)

    elif format == "vcf":
        with open(output_path, "w") as f:
            f.write("##fileformat=VCFv4.2\n")
            f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
            for _, row in df.iterrows():
                vid = f"{row['chromosome']}:{row['position']}:{row['reference_allele']}:{row['alternate_allele']}"
                f.write(f"{row['chromosome']}\t{row['position']}\t{vid}\t"
                       f"{row['reference_allele']}\t{row['alternate_allele']}\t.\t.\t.\n")

    logger.info(f"Wrote {len(df)} variant sites to {output_path}")
    return output_path


def load_vartrix_output(
    matrix_dir: str | Path,
    barcodes_path: str | Path | None = None,
    variants_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load VarTrix output matrices.

    VarTrix outputs:
    - ref_matrix.mtx: Reference allele counts
    - alt_matrix.mtx: Alternate allele counts
    - barcodes.tsv: Cell barcodes
    - variants.tsv: Variant positions

    Args:
        matrix_dir: Directory with VarTrix output
        barcodes_path: Optional path to barcodes file
        variants_path: Optional path to variants file

    Returns:
        DataFrame with barcode, variant_id, ref_count, alt_count
    """
    from scipy.io import mmread

    matrix_dir = Path(matrix_dir)

    ref_path = matrix_dir / "ref_matrix.mtx"
    alt_path = matrix_dir / "alt_matrix.mtx"

    if not ref_path.exists() or not alt_path.exists():
        raise FileNotFoundError(f"VarTrix matrices not found in {matrix_dir}")

    ref_matrix = mmread(ref_path).tocsr()
    alt_matrix = mmread(alt_path).tocsr()

    if barcodes_path is None:
        barcodes_path = matrix_dir / "barcodes.tsv"
    barcodes = pd.read_csv(barcodes_path, header=None, sep="\t")[0].tolist()

    if variants_path is None:
        variants_path = matrix_dir / "variants.tsv"
    variants = pd.read_csv(variants_path, header=None, sep="\t")[0].tolist()

    records = []
    for var_idx, variant_id in enumerate(variants):
        for bc_idx, barcode in enumerate(barcodes):
            ref_count = int(ref_matrix[var_idx, bc_idx])
            alt_count = int(alt_matrix[var_idx, bc_idx])

            if ref_count > 0 or alt_count > 0:
                records.append({
                    "variant_id": variant_id,
                    "barcode": barcode,
                    "ref_count": ref_count,
                    "alt_count": alt_count,
                })

    return pd.DataFrame(records)


def load_cellsnp_output(path: str | Path) -> pd.DataFrame:
    """Load cellSNP-lite output.

    Args:
        path: Path to cellSNP output directory or file

    Returns:
        DataFrame with barcode, variant_id, ref_count, alt_count
    """
    path = Path(path)

    if path.is_dir():
        ad_path = path / "cellSNP.tag.AD.mtx"
        dp_path = path / "cellSNP.tag.DP.mtx"
        samples_path = path / "cellSNP.samples.tsv"
        base_path = path / "cellSNP.base.vcf"
    else:
        raise ValueError("cellSNP path must be a directory")

    if not ad_path.exists():
        raise FileNotFoundError(f"cellSNP AD matrix not found: {ad_path}")

    from scipy.io import mmread

    ad_matrix = mmread(ad_path).tocsr()
    dp_matrix = mmread(dp_path).tocsr() if dp_path.exists() else None

    barcodes = pd.read_csv(samples_path, header=None, sep="\t")[0].tolist()

    variants = []
    with open(base_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            chrom, pos, _, ref, alt = parts[:5]
            variants.append(f"{chrom}:{pos}:{ref}:{alt}")

    records = []
    for var_idx, variant_id in enumerate(variants):
        for bc_idx, barcode in enumerate(barcodes):
            alt_count = int(ad_matrix[var_idx, bc_idx])

            if dp_matrix is not None:
                total_count = int(dp_matrix[var_idx, bc_idx])
                ref_count = total_count - alt_count
            else:
                ref_count = 0

            if ref_count > 0 or alt_count > 0:
                records.append({
                    "variant_id": variant_id,
                    "barcode": barcode,
                    "ref_count": max(0, ref_count),
                    "alt_count": alt_count,
                })

    return pd.DataFrame(records)


def merge_variant_counts_with_spatial_metadata(
    counts_df: pd.DataFrame,
    spatial_metadata_df: pd.DataFrame,
    sample_id: str | None = None,
) -> pd.DataFrame:
    """Merge variant counts with spatial metadata.

    Args:
        counts_df: DataFrame with variant counts per barcode
        spatial_metadata_df: DataFrame with spatial coordinates
        sample_id: Sample ID to assign

    Returns:
        DataFrame with counts and spatial coordinates
    """
    df = counts_df.copy()

    meta_cols = ["barcode"]
    for col in ["x", "y", "sample_id", "donor_id", "stage"]:
        if col in spatial_metadata_df.columns:
            meta_cols.append(col)

    df = df.merge(
        spatial_metadata_df[meta_cols].drop_duplicates(subset=["barcode"]),
        on="barcode",
        how="left",
    )

    if sample_id and "sample_id" not in df.columns:
        df["sample_id"] = sample_id

    return df


def annotate_spatial_variant_evidence(
    counts_df: pd.DataFrame,
    thresholds: dict | None = None,
) -> pd.DataFrame:
    """Annotate spatial variant evidence with labels.

    Args:
        counts_df: DataFrame with variant counts
        thresholds: Custom evidence thresholds

    Returns:
        DataFrame with evidence labels
    """
    df = counts_df.copy()

    df["total_count"] = df["ref_count"] + df["alt_count"]
    df["expressed_alt_fraction"] = np.where(
        df["total_count"] > 0,
        df["alt_count"] / df["total_count"],
        np.nan,
    )

    df["evidence_label"] = df.apply(
        lambda r: classify_spatial_variant_evidence(
            r["ref_count"], r["alt_count"], thresholds
        ),
        axis=1,
    )

    df["caution"] = (
        "RNA coverage absence is NOT mutation absence. "
        "Expression-based evidence only."
    )

    return df


def summarize_variant_evidence_by_transition_score(
    evidence_df: pd.DataFrame,
    transition_df: pd.DataFrame,
    high_quantile: float = 0.90,
) -> pd.DataFrame:
    """Summarize variant evidence by transition score groups.

    Args:
        evidence_df: DataFrame with spatial variant evidence
        transition_df: DataFrame with transition scores
        high_quantile: Quantile threshold for high-transition

    Returns:
        Summary DataFrame
    """
    df = evidence_df.merge(
        transition_df[["barcode", "transition_score"]],
        on="barcode",
        how="left",
    )

    threshold = df["transition_score"].quantile(high_quantile)
    df["transition_group"] = np.where(
        df["transition_score"] >= threshold,
        "high_transition",
        "low_transition",
    )

    summary = df.groupby(["transition_group", "evidence_label"]).agg({
        "variant_id": "count",
        "alt_count": "sum",
        "total_count": "sum",
    }).reset_index()

    summary = summary.rename(columns={"variant_id": "n_observations"})

    return summary


def create_spatial_evidence_records(
    counts_df: pd.DataFrame,
    thresholds: dict | None = None,
) -> list[SpatialVariantEvidence]:
    """Create SpatialVariantEvidence records from counts DataFrame.

    Args:
        counts_df: DataFrame with annotated variant counts
        thresholds: Custom evidence thresholds

    Returns:
        List of SpatialVariantEvidence records
    """
    df = annotate_spatial_variant_evidence(counts_df, thresholds)

    records = []
    for _, row in df.iterrows():
        records.append(SpatialVariantEvidence(
            variant_id=row["variant_id"],
            sample_id=row.get("sample_id", "unknown"),
            barcode=row["barcode"],
            x=row.get("x"),
            y=row.get("y"),
            ref_count=row["ref_count"],
            alt_count=row["alt_count"],
            total_count=row["total_count"],
            expressed_alt_fraction=row.get("expressed_alt_fraction"),
            evidence_label=row["evidence_label"],
            caution=row.get("caution", ""),
        ))

    return records
