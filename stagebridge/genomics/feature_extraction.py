"""Feature extraction for WES, clonal, and evolution data.

Provides functions to create feature parquets from raw variant and clone data.

Usage:
    from stagebridge.genomics.feature_extraction import (
        create_wes_features,
        create_clonal_features,
        create_evolution_features,
    )

    # Create WES features
    wes_df = create_wes_features(variants_path, output_path)

    # Create clonal features
    cell_df, patient_df = create_clonal_features(clone_assignments_path, output_path)

    # Combine into evolution features
    evo_df = create_evolution_features(wes_path, clonal_path, output_path)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ONCOKB_LEVELS = {
    ("EGFR", "L858R"): ("1", "Osimertinib, Erlotinib, Gefitinib, Afatinib"),
    ("EGFR", "exon19del"): ("1", "Osimertinib, Erlotinib, Gefitinib, Afatinib"),
    ("EGFR", "T790M"): ("1", "Osimertinib"),
    ("KRAS", "G12C"): ("1", "Sotorasib, Adagrasib"),
    ("BRAF", "V600E"): ("1", "Dabrafenib + Trametinib"),
    ("ALK", "fusion"): ("1", "Alectinib, Brigatinib, Lorlatinib"),
    ("ROS1", "fusion"): ("1", "Entrectinib, Crizotinib"),
    ("MET", "exon14skip"): ("1", "Capmatinib, Tepotinib"),
    ("RET", "fusion"): ("1", "Selpercatinib, Pralsetinib"),
    ("NTRK", "fusion"): ("1", "Larotrectinib, Entrectinib"),
    ("MET", "amplification"): ("2", "Capmatinib, Tepotinib"),
    ("ERBB2", "amplification"): ("2", "Trastuzumab deruxtecan"),
    ("KRAS", "G12D"): ("4", ""),
    ("KRAS", "G12V"): ("4", ""),
    ("KRAS", "G12S"): ("4", ""),
    ("KRAS", "G13D"): ("4", ""),
    ("EGFR", "C797S"): ("R1", ""),
}


def compute_tmb(variants_df: pd.DataFrame, exome_mb: float = 38.0) -> float:
    """Compute tumor mutation burden (mutations per Mb)."""
    if "consequence" in variants_df.columns:
        coding = variants_df[
            variants_df["consequence"].str.contains(
                "missense|nonsense|frameshift|splice", case=False, na=False
            )
        ]
        return len(coding) / exome_mb
    return len(variants_df) / exome_mb


def create_wes_features(
    variants_path: str | Path,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Create per-sample WES features from annotated variants.

    Args:
        variants_path: Path to annotated_variants.parquet
        output_path: Path to save wes_features.parquet (optional)

    Returns:
        DataFrame with WES features per sample
    """
    variants_path = Path(variants_path)

    logger.info(f"Loading variants from {variants_path}")
    df = pd.read_parquet(variants_path)
    logger.info(f"Loaded {len(df):,} variants from {df['donor_id'].nunique()} patients")

    df["sample_id"] = df["donor_id"] + "_" + df["stage"]
    samples = df["sample_id"].unique()

    records = []
    for sample_id in samples:
        sample_df = df[df["sample_id"] == sample_id]
        donor_id = sample_df["donor_id"].iloc[0]
        stage = sample_df["stage"].iloc[0]

        record = {
            "sample_id": sample_id,
            "donor_id": donor_id,
            "stage": stage,
            "tmb": compute_tmb(sample_df),
        }

        genes_to_check = [
            "KRAS", "EGFR", "TP53", "STK11", "KEAP1", "SMAD4", "BRAF",
            "ALK", "ROS1", "MET", "RET", "ERBB2", "PIK3CA", "PTEN", "NF1",
        ]
        for gene in genes_to_check:
            record[f"{gene.lower()}_mut"] = int((sample_df["gene"] == gene).any())

        hotspots = sample_df[sample_df["is_hotspot"] == True]

        record["egfr_L858R"] = int(
            ((hotspots["gene"] == "EGFR") & (hotspots["hotspot_type"] == "L858R")).any()
        )
        record["egfr_exon19del"] = int(
            ((hotspots["gene"] == "EGFR") & (hotspots["hotspot_type"] == "exon19del")).any()
        )
        record["egfr_T790M"] = int(
            ((hotspots["gene"] == "EGFR") & (hotspots["hotspot_type"] == "T790M")).any()
        )
        record["kras_G12C"] = int(
            ((hotspots["gene"] == "KRAS") & (hotspots["hotspot_type"] == "G12C")).any()
        )
        record["kras_G12D"] = int(
            ((hotspots["gene"] == "KRAS") & (hotspots["hotspot_type"] == "G12D")).any()
        )
        record["kras_G12V"] = int(
            ((hotspots["gene"] == "KRAS") & (hotspots["hotspot_type"] == "G12V")).any()
        )
        record["kras_G12S"] = int(
            ((hotspots["gene"] == "KRAS") & (hotspots["hotspot_type"] == "G12S")).any()
        )
        record["braf_V600E"] = int(
            ((hotspots["gene"] == "BRAF") & (hotspots["hotspot_type"] == "V600E")).any()
        )

        oncokb_level = None
        therapies = []
        for _, row in hotspots.iterrows():
            key = (row["gene"], row["hotspot_type"])
            if key in ONCOKB_LEVELS:
                level, drugs = ONCOKB_LEVELS[key]
                if oncokb_level is None or level < oncokb_level:
                    oncokb_level = level
                if drugs:
                    therapies.extend(drugs.split(", "))

        record["oncokb_highest_level"] = oncokb_level
        record["has_level1_mutation"] = int(oncokb_level == "1")
        record["has_actionable_mutation"] = int(oncokb_level in ("1", "2", "3A", "3B"))
        record["recommended_therapies"] = (
            ", ".join(sorted(set(therapies))) if therapies else None
        )

        record["egfr_kras_comut"] = int(record["egfr_mut"] and record["kras_mut"])
        record["tp53_comut"] = int(
            record["tp53_mut"] and (record["egfr_mut"] or record["kras_mut"])
        )
        record["stk11_keap1_comut"] = int(record["stk11_mut"] and record["keap1_mut"])

        records.append(record)

    result = pd.DataFrame(records)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(output_path, index=False)
        logger.info(f"Saved WES features to {output_path}")

    return result


def _compute_cell_level_clonal_features(clone_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-cell clonal features."""
    df = clone_df.copy()

    clone_sizes = df.groupby("cnv_leiden").size().rename("clone_size")
    df = df.merge(clone_sizes, left_on="cnv_leiden", right_index=True)

    df["clone_rank"] = (
        df.groupby("patient_id")["clone_size"]
        .rank(method="dense", ascending=False)
        .astype(int)
        - 1
    )
    df["is_major_clone"] = (df["clone_rank"] == 0).astype(int)

    patient_sizes = df.groupby("patient_id").size().rename("patient_n_cells")
    df = df.merge(patient_sizes, left_on="patient_id", right_index=True)
    df["clone_fraction"] = df["clone_size"] / df["patient_n_cells"]

    df["cnv_score_z"] = df.groupby("patient_id")["cnv_score"].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-8)
    )

    return df


def _compute_patient_level_clonal_features(
    clone_df: pd.DataFrame,
    clonal_details: pd.DataFrame | None,
    clonal_patterns: dict[str, str],
) -> pd.DataFrame:
    """Compute per-patient clonal features."""
    patient_stats = (
        clone_df.groupby("patient_id")
        .agg(
            n_cells=("cell_id", "count"),
            n_clones=("cnv_leiden", "nunique"),
            cnv_score_mean=("cnv_score", "mean"),
            cnv_score_std=("cnv_score", "std"),
        )
        .reset_index()
    )

    def shannon_entropy(group):
        counts = group["cnv_leiden"].value_counts(normalize=True)
        return -np.sum(counts * np.log(counts + 1e-10))

    entropy = clone_df.groupby("patient_id").apply(
        shannon_entropy, include_groups=False
    ).rename("clonal_entropy")
    patient_stats = patient_stats.merge(entropy, left_on="patient_id", right_index=True)

    def gini_simpson(group):
        counts = group["cnv_leiden"].value_counts(normalize=True)
        return 1 - np.sum(counts**2)

    diversity = clone_df.groupby("patient_id").apply(
        gini_simpson, include_groups=False
    ).rename("clonal_diversity")
    patient_stats = patient_stats.merge(diversity, left_on="patient_id", right_index=True)

    if clonal_details is not None:
        details_cols = [
            "patient_id",
            "n_clones_precursor",
            "n_clones_invasive",
            "n_clones_shared",
            "aneuploidy_score",
            "confidence",
        ]
        available_cols = [c for c in details_cols if c in clonal_details.columns]
        patient_stats = patient_stats.merge(
            clonal_details[available_cols], on="patient_id", how="left"
        )

        if "n_clones_shared" in patient_stats.columns:
            patient_stats["clone_sharing_ratio"] = (
                patient_stats["n_clones_shared"]
                / patient_stats[["n_clones_precursor", "n_clones_invasive"]].max(axis=1)
            ).fillna(0)

        if "n_clones_invasive" in patient_stats.columns:
            patient_stats["has_invasive_only_clones"] = (
                patient_stats["n_clones_invasive"] > patient_stats["n_clones_shared"]
            ).astype(int)

    patient_stats["clonal_pattern"] = patient_stats["patient_id"].map(clonal_patterns)
    patient_stats["clonal_pattern"] = patient_stats["clonal_pattern"].fillna("unknown")

    pattern_map = {"1a": 0, "1b": 1, "uncategorized": 2, "unknown": 3}
    patient_stats["clonal_pattern_idx"] = patient_stats["clonal_pattern"].map(pattern_map)

    return patient_stats


def create_clonal_features(
    clone_assignments_path: str | Path,
    output_path: str | Path | None = None,
    clonal_details_path: str | Path | None = None,
    clonal_patterns_path: str | Path | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create clonal features for evolution branch.

    Args:
        clone_assignments_path: Path to clone_assignments.parquet
        output_path: Output path for clonal_features.parquet
        clonal_details_path: Path to clonal_analysis_details.csv
        clonal_patterns_path: Path to clonal_patterns.json

    Returns:
        (cell_features, patient_features) DataFrames
    """
    logger.info("Loading clone assignments")
    clone_df = pd.read_parquet(clone_assignments_path)
    logger.info(f"Loaded {len(clone_df):,} cells")

    clonal_details = None
    if clonal_details_path and Path(clonal_details_path).exists():
        clonal_details = pd.read_csv(clonal_details_path)

    clonal_patterns = {}
    if clonal_patterns_path and Path(clonal_patterns_path).exists():
        with open(clonal_patterns_path) as f:
            clonal_patterns = json.load(f)

    cell_features = _compute_cell_level_clonal_features(clone_df)
    patient_features = _compute_patient_level_clonal_features(
        clone_df, clonal_details, clonal_patterns
    )

    patient_cols = [
        "patient_id",
        "n_clones",
        "clonal_entropy",
        "clonal_diversity",
        "clonal_pattern",
        "clonal_pattern_idx",
        "aneuploidy_score",
        "clone_sharing_ratio",
        "has_invasive_only_clones",
    ]
    available_patient_cols = [c for c in patient_cols if c in patient_features.columns]

    cell_features = cell_features.merge(
        patient_features[available_patient_cols],
        on="patient_id",
        how="left",
        suffixes=("", "_patient"),
    )

    output_cols = [
        "cell_id",
        "patient_id",
        "stage",
        "sample_id",
        "cnv_leiden",
        "cnv_score",
        "cnv_score_z",
        "clone_size",
        "clone_rank",
        "is_major_clone",
        "clone_fraction",
        "n_clones",
        "clonal_entropy",
        "clonal_diversity",
        "clonal_pattern",
        "clonal_pattern_idx",
    ]
    for col in ["aneuploidy_score", "clone_sharing_ratio", "has_invasive_only_clones"]:
        if col in cell_features.columns:
            output_cols.append(col)

    output_df = cell_features[[c for c in output_cols if c in cell_features.columns]]

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_df.to_parquet(output_path, index=False)
        logger.info(f"Saved cell-level features to {output_path}")

        patient_output = output_path.parent / "clonal_patient_features.parquet"
        patient_features.to_parquet(patient_output, index=False)
        logger.info(f"Saved patient-level features to {patient_output}")

    return output_df, patient_features


def create_evolution_features(
    wes_path: str | Path,
    clonal_path: str | Path,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Merge WES and clonal features into evolution features.

    Args:
        wes_path: Path to wes_features.parquet (patient-level)
        clonal_path: Path to clonal_features.parquet (cell-level)
        output_path: Output path for evolution_features.parquet

    Returns:
        DataFrame with combined evolution features
    """
    logger.info("Loading WES features")
    wes_df = pd.read_parquet(wes_path)
    logger.info(f"Loaded {len(wes_df)} patients with WES data")

    if "patient_id" not in wes_df.columns and "donor_id" in wes_df.columns:
        wes_df = wes_df.rename(columns={"donor_id": "patient_id"})

    logger.info("Loading clonal features")
    clonal_df = pd.read_parquet(clonal_path)
    logger.info(f"Loaded {len(clonal_df):,} cells with clonal data")

    clonal_cols = [
        "cell_id",
        "patient_id",
        "stage",
        "sample_id",
        "cnv_score",
        "cnv_score_z",
        "clone_size",
        "clone_rank",
        "is_major_clone",
        "clone_fraction",
        "n_clones",
        "clonal_entropy",
        "clonal_diversity",
        "clonal_pattern_idx",
        "aneuploidy_score",
        "clone_sharing_ratio",
        "has_invasive_only_clones",
    ]
    clonal_cols = [c for c in clonal_cols if c in clonal_df.columns]
    clonal_df = clonal_df[clonal_cols]

    if "stage" in wes_df.columns and "stage" in clonal_df.columns:
        result = clonal_df.merge(wes_df, on=["patient_id", "stage"], how="left")
    else:
        wes_cols = [c for c in wes_df.columns if c not in ["stage", "sample_id"]]
        result = clonal_df.merge(wes_df[wes_cols], on="patient_id", how="left")

    wes_feature_cols = [
        c for c in wes_df.columns if c not in ["patient_id", "stage", "sample_id"]
    ]
    for col in wes_feature_cols:
        if col in result.columns:
            result[col] = result[col].fillna(0)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(output_path, index=False)
        logger.info(f"Saved evolution features to {output_path}")

    return result
