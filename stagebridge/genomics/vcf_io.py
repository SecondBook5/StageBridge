"""VCF and variant table I/O utilities.

Handles reading VCF files and annotated variant tables, normalizing
variant identifiers, and validating variant data.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterator

import pandas as pd

from stagebridge.genomics.schemas import VariantRecord

logger = logging.getLogger(__name__)


def normalize_chromosome(chrom: str) -> str:
    """Normalize chromosome names to chr-prefixed format."""
    chrom = str(chrom).strip()
    if chrom.startswith("chr"):
        return chrom
    if chrom.isdigit() or chrom in ("X", "Y", "M", "MT"):
        if chrom == "MT":
            chrom = "M"
        return f"chr{chrom}"
    return chrom


def normalize_variant_id(
    chromosome: str,
    position: int,
    ref: str,
    alt: str,
) -> str:
    """Create normalized variant identifier.

    Format: chromosome:position:reference:alternate
    Example: chr7:55249071:C:T
    """
    chrom = normalize_chromosome(chromosome)
    return f"{chrom}:{position}:{ref.upper()}:{alt.upper()}"


def parse_vcf_info(info_str: str) -> dict[str, str]:
    """Parse VCF INFO field into dictionary."""
    if not info_str or info_str == ".":
        return {}
    result = {}
    for item in info_str.split(";"):
        if "=" in item:
            key, value = item.split("=", 1)
            result[key] = value
        else:
            result[item] = "True"
    return result


def parse_vcf_format(
    format_str: str,
    sample_str: str,
) -> dict[str, str]:
    """Parse VCF FORMAT and sample fields."""
    if not format_str or not sample_str:
        return {}
    keys = format_str.split(":")
    values = sample_str.split(":")
    return dict(zip(keys, values))


def read_vcf(
    path: str | Path,
    sample_id: str | None = None,
    donor_id: str | None = None,
    is_germline: bool | None = None,
) -> Iterator[VariantRecord]:
    """Read variants from a VCF file.

    Args:
        path: Path to VCF file (can be gzipped)
        sample_id: Sample ID to assign (defaults to first sample in VCF)
        donor_id: Donor ID to assign (defaults to sample_id)
        is_germline: Whether variants are germline (None = unknown)

    Yields:
        VariantRecord for each variant
    """
    path = Path(path)

    if path.suffix == ".gz":
        import gzip
        opener = gzip.open
    else:
        opener = open

    vcf_samples = []
    with opener(path, "rt") as f:
        for line in f:
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                parts = line.strip().split("\t")
                if len(parts) > 9:
                    vcf_samples = parts[9:]
                break

    if not sample_id:
        sample_id = vcf_samples[0] if vcf_samples else path.stem
    if not donor_id:
        donor_id = sample_id

    with opener(path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue

            parts = line.strip().split("\t")
            if len(parts) < 8:
                continue

            chrom = normalize_chromosome(parts[0])
            try:
                pos = int(parts[1])
            except ValueError:
                logger.warning(f"Invalid position: {parts[1]}")
                continue

            ref = parts[3].upper()
            alts = parts[4].upper().split(",")
            filter_status = parts[6]

            info = parse_vcf_info(parts[7])

            tumor_vaf = None
            tumor_depth = None
            normal_vaf = None
            normal_depth = None

            if len(parts) > 9:
                fmt = parse_vcf_format(parts[8], parts[9])
                if "AF" in fmt:
                    try:
                        tumor_vaf = float(fmt["AF"].split(",")[0])
                    except (ValueError, IndexError):
                        pass
                if "DP" in fmt:
                    try:
                        tumor_depth = int(fmt["DP"])
                    except ValueError:
                        pass
                if "AD" in fmt:
                    try:
                        ads = [int(x) for x in fmt["AD"].split(",")]
                        tumor_depth = sum(ads)
                        if len(ads) > 1 and tumor_depth > 0:
                            tumor_vaf = ads[1] / tumor_depth
                    except (ValueError, IndexError):
                        pass

            if len(parts) > 10:
                fmt_normal = parse_vcf_format(parts[8], parts[10])
                if "AF" in fmt_normal:
                    try:
                        normal_vaf = float(fmt_normal["AF"].split(",")[0])
                    except (ValueError, IndexError):
                        pass
                if "DP" in fmt_normal:
                    try:
                        normal_depth = int(fmt_normal["DP"])
                    except ValueError:
                        pass

            gene = info.get("GENE") or info.get("Gene") or info.get("gene")
            consequence = info.get("Consequence") or info.get("CSQ")
            protein_change = info.get("HGVSp") or info.get("AAChange")

            for alt in alts:
                if alt == "." or alt == "*":
                    continue

                if len(ref) == 1 and len(alt) == 1:
                    variant_type = "SNV"
                elif len(ref) > len(alt):
                    variant_type = "DEL"
                elif len(ref) < len(alt):
                    variant_type = "INS"
                else:
                    variant_type = "MNV"

                yield VariantRecord(
                    sample_id=sample_id,
                    donor_id=donor_id,
                    chromosome=chrom,
                    position=pos,
                    reference_allele=ref,
                    alternate_allele=alt,
                    gene=gene,
                    consequence=consequence,
                    protein_change=protein_change,
                    variant_type=variant_type,
                    source="VCF",
                    is_germline=is_germline,
                    is_somatic=not is_germline if is_germline is not None else None,
                    tumor_vaf=tumor_vaf,
                    normal_vaf=normal_vaf,
                    tumor_depth=tumor_depth,
                    normal_depth=normal_depth,
                    filter_status=filter_status,
                )


def read_annotated_variant_table(
    path: str | Path,
    column_mapping: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Read an annotated variant table (TSV/CSV/Parquet).

    Args:
        path: Path to variant table
        column_mapping: Optional mapping from file columns to standard names

    Returns:
        DataFrame with standardized column names
    """
    path = Path(path)

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix in (".tsv", ".txt"):
        df = pd.read_csv(path, sep="\t")
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_csv(path, sep="\t")

    if column_mapping:
        df = df.rename(columns=column_mapping)

    standard_columns = {
        "chrom": "chromosome",
        "chr": "chromosome",
        "CHROM": "chromosome",
        "pos": "position",
        "POS": "position",
        "ref": "reference_allele",
        "REF": "reference_allele",
        "alt": "alternate_allele",
        "ALT": "alternate_allele",
        "GENE": "gene",
        "Gene": "gene",
        "Hugo_Symbol": "gene",
        "SYMBOL": "gene",
        "Tumor_Sample_Barcode": "sample_id",
        "sample": "sample_id",
        "SAMPLE": "sample_id",
        "patient": "donor_id",
        "Patient": "donor_id",
        "donor": "donor_id",
        "VAF": "tumor_vaf",
        "t_vaf": "tumor_vaf",
        "Tumor_VAF": "tumor_vaf",
        "t_depth": "tumor_depth",
        "Tumor_Depth": "tumor_depth",
        "n_vaf": "normal_vaf",
        "Normal_VAF": "normal_vaf",
        "n_depth": "normal_depth",
        "Normal_Depth": "normal_depth",
        "HGVSp_Short": "protein_change",
        "Protein_Change": "protein_change",
        "Consequence": "consequence",
        "Variant_Classification": "consequence",
        "Transcript_ID": "transcript",
    }

    for old, new in standard_columns.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    if "chromosome" in df.columns:
        df["chromosome"] = df["chromosome"].apply(normalize_chromosome)

    return df


def validate_variant_table(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """Validate a variant table for required fields.

    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []

    required = ["chromosome", "position", "reference_allele", "alternate_allele"]
    for col in required:
        if col not in df.columns:
            errors.append(f"Missing required column: {col}")

    if not errors:
        if df["chromosome"].isna().any():
            errors.append("chromosome contains missing values")
        if df["position"].isna().any():
            errors.append("position contains missing values")
        if (df["position"] <= 0).any():
            errors.append("position contains non-positive values")
        if df["reference_allele"].isna().any() or (df["reference_allele"] == "").any():
            errors.append("reference_allele contains empty values")
        if df["alternate_allele"].isna().any() or (df["alternate_allele"] == "").any():
            errors.append("alternate_allele contains empty values")

    if "sample_id" not in df.columns and "donor_id" not in df.columns:
        errors.append("Neither sample_id nor donor_id column present")

    return len(errors) == 0, errors


def add_variant_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Add normalized variant_id column to DataFrame."""
    df = df.copy()
    df["variant_id"] = df.apply(
        lambda r: normalize_variant_id(
            r["chromosome"],
            r["position"],
            r["reference_allele"],
            r["alternate_allele"],
        ),
        axis=1,
    )
    return df


def handle_multiallelic(df: pd.DataFrame) -> pd.DataFrame:
    """Split or flag multi-allelic records.

    Multi-allelic records (multiple alts at same position) are split
    into separate rows, each with its own variant_id.
    """
    df = df.copy()

    if "alternate_allele" not in df.columns:
        return df

    mask = df["alternate_allele"].str.contains(",", na=False)
    if not mask.any():
        return df

    single = df[~mask].copy()
    multi = df[mask].copy()

    expanded = []
    for _, row in multi.iterrows():
        alts = row["alternate_allele"].split(",")
        for alt in alts:
            new_row = row.copy()
            new_row["alternate_allele"] = alt.strip()
            expanded.append(new_row)

    if expanded:
        expanded_df = pd.DataFrame(expanded)
        result = pd.concat([single, expanded_df], ignore_index=True)
    else:
        result = single

    return result


def write_variant_master_table(
    df: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Write variant master table to parquet.

    Args:
        df: Variant DataFrame
        output_path: Output path

    Returns:
        Path to written file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if "variant_id" not in df.columns:
        df = add_variant_ids(df)

    df = handle_multiallelic(df)

    n_dupes = df.duplicated(subset=["variant_id", "sample_id"]).sum()
    if n_dupes > 0:
        logger.warning(
            f"Found {n_dupes} duplicate variant_id+sample_id combinations. "
            "Keeping first occurrence."
        )
        df = df.drop_duplicates(subset=["variant_id", "sample_id"], keep="first")

    df.to_parquet(output_path, index=False)
    logger.info(f"Wrote {len(df)} variants to {output_path}")

    return output_path
