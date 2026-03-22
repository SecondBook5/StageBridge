"""
Data Harmonization - unify metadata across donors, samples, stages, and modalities.

This module ensures consistent identifiers and vocabularies across all datasets
processed by StageBridge.
"""

import logging
from typing import Any

import pandas as pd
import anndata as ad

log = logging.getLogger(__name__)


# =============================================================================
# Canonical Vocabularies
# =============================================================================

CANONICAL_STAGES = ["Normal", "AAH", "AIS", "MIA", "LUAD"]

STAGE_ALIASES = {
    # Common variations
    "normal": "Normal",
    "healthy": "Normal",
    "control": "Normal",
    "aah": "AAH",
    "atypical adenomatous hyperplasia": "AAH",
    "ais": "AIS",
    "adenocarcinoma in situ": "AIS",
    "mia": "MIA",
    "minimally invasive adenocarcinoma": "MIA",
    "luad": "LUAD",
    "lung adenocarcinoma": "LUAD",
    "adenocarcinoma": "LUAD",
    "invasive": "LUAD",
}

CANONICAL_MODALITIES = ["snRNA-seq", "scRNA-seq", "Visium", "WES"]

MODALITY_ALIASES = {
    "snrna": "snRNA-seq",
    "snrna-seq": "snRNA-seq",
    "single-nucleus": "snRNA-seq",
    "scrna": "scRNA-seq",
    "scrna-seq": "scRNA-seq",
    "single-cell": "scRNA-seq",
    "visium": "Visium",
    "spatial": "Visium",
    "10x_visium": "Visium",
    "wes": "WES",
    "whole-exome": "WES",
    "exome": "WES",
}


# =============================================================================
# Harmonization Functions
# =============================================================================

def harmonize_stage(stage: str) -> str:
    """
    Map stage label to canonical vocabulary.

    Parameters
    ----------
    stage : str
        Raw stage label

    Returns
    -------
    str
        Canonical stage label

    Raises
    ------
    ValueError
        If stage cannot be mapped
    """
    if stage in CANONICAL_STAGES:
        return stage

    normalized = stage.lower().strip()
    if normalized in STAGE_ALIASES:
        return STAGE_ALIASES[normalized]

    raise ValueError(
        f"Unknown stage '{stage}'. Valid stages: {CANONICAL_STAGES}. "
        f"Add mapping to STAGE_ALIASES if this is a valid alias."
    )


def harmonize_modality(modality: str) -> str:
    """Map modality label to canonical vocabulary."""
    if modality in CANONICAL_MODALITIES:
        return modality

    normalized = modality.lower().strip().replace(" ", "-")
    if normalized in MODALITY_ALIASES:
        return MODALITY_ALIASES[normalized]

    raise ValueError(
        f"Unknown modality '{modality}'. Valid modalities: {CANONICAL_MODALITIES}"
    )


def harmonize_donor_ids(
    df: pd.DataFrame,
    donor_col: str = "donor_id",
    mapping: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Harmonize donor IDs across datasets.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with donor column
    donor_col : str
        Name of donor ID column
    mapping : dict, optional
        Explicit mapping of raw -> canonical donor IDs

    Returns
    -------
    pd.DataFrame
        DataFrame with harmonized donor IDs
    """
    df = df.copy()

    if mapping:
        df[donor_col] = df[donor_col].map(lambda x: mapping.get(x, x))

    # Standardize format: uppercase, no spaces, underscores
    df[donor_col] = (
        df[donor_col]
        .astype(str)
        .str.upper()
        .str.strip()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )

    return df


def harmonize_adata_obs(
    adata: ad.AnnData,
    stage_col: str | None = None,
    donor_col: str | None = None,
    modality: str | None = None,
    donor_mapping: dict[str, str] | None = None,
) -> ad.AnnData:
    """
    Harmonize AnnData obs columns to canonical vocabulary.

    Parameters
    ----------
    adata : AnnData
        AnnData object to harmonize
    stage_col : str, optional
        Name of stage column to harmonize
    donor_col : str, optional
        Name of donor column to harmonize
    modality : str, optional
        Modality to assign (will be harmonized)
    donor_mapping : dict, optional
        Donor ID mapping

    Returns
    -------
    AnnData
        Harmonized AnnData (modified in place)
    """
    # Harmonize stages
    if stage_col and stage_col in adata.obs.columns:
        try:
            adata.obs["stage"] = adata.obs[stage_col].apply(harmonize_stage)
            log.info(f"Harmonized stages from '{stage_col}'")
        except ValueError as e:
            log.warning(f"Stage harmonization failed: {e}")

    # Harmonize donor IDs
    if donor_col and donor_col in adata.obs.columns:
        obs_df = adata.obs.reset_index()
        obs_df = harmonize_donor_ids(obs_df, donor_col, donor_mapping)
        adata.obs["donor_id"] = obs_df[donor_col].values
        log.info(f"Harmonized donor IDs from '{donor_col}'")

    # Add modality
    if modality:
        adata.obs["modality"] = harmonize_modality(modality)

    return adata


def create_donor_mapping_table(
    datasets: dict[str, pd.DataFrame],
    donor_cols: dict[str, str],
) -> pd.DataFrame:
    """
    Create a mapping table linking donor IDs across datasets.

    Parameters
    ----------
    datasets : dict
        Dict of dataset name -> DataFrame
    donor_cols : dict
        Dict of dataset name -> donor column name

    Returns
    -------
    pd.DataFrame
        Mapping table with columns [dataset, raw_id, canonical_id]
    """
    rows = []

    for dataset_name, df in datasets.items():
        col = donor_cols.get(dataset_name)
        if col and col in df.columns:
            for raw_id in df[col].unique():
                # Generate canonical ID
                canonical = str(raw_id).upper().strip().replace(" ", "_").replace("-", "_")
                rows.append({
                    "dataset": dataset_name,
                    "raw_id": raw_id,
                    "canonical_id": canonical,
                })

    mapping_df = pd.DataFrame(rows)

    # Flag potential duplicates (same canonical ID from different raw IDs)
    dup_check = mapping_df.groupby("canonical_id")["raw_id"].nunique()
    duplicates = dup_check[dup_check > 1].index.tolist()

    if duplicates:
        log.warning(
            f"Potential donor ID conflicts detected for: {duplicates}. "
            "Review mapping table and provide explicit mapping if needed."
        )

    return mapping_df


def validate_harmonization(adata: ad.AnnData) -> dict[str, Any]:
    """
    Validate that AnnData has been properly harmonized.

    Returns
    -------
    dict
        Validation report with any issues found
    """
    issues = []

    # Check stage column
    if "stage" in adata.obs.columns:
        unknown_stages = set(adata.obs["stage"].unique()) - set(CANONICAL_STAGES)
        if unknown_stages:
            issues.append(f"Non-canonical stages found: {unknown_stages}")
    else:
        issues.append("Missing 'stage' column")

    # Check donor_id column
    if "donor_id" not in adata.obs.columns:
        issues.append("Missing 'donor_id' column")

    # Check modality column
    if "modality" in adata.obs.columns:
        unknown_modalities = set(adata.obs["modality"].unique()) - set(CANONICAL_MODALITIES)
        if unknown_modalities:
            issues.append(f"Non-canonical modalities found: {unknown_modalities}")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "n_cells": adata.n_obs,
        "n_donors": adata.obs["donor_id"].nunique() if "donor_id" in adata.obs.columns else 0,
        "stages": list(adata.obs["stage"].unique()) if "stage" in adata.obs.columns else [],
    }
