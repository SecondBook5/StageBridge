"""Split validation for donor-held-out cross-validation.

Critical pre-training check for scientific integrity - detects donor leakage
that would invalidate model evaluation.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def validate_splits(
    cells_df: pd.DataFrame,
    splits: dict[str, Any],
    donor_col: str = "donor_id",
) -> dict[str, Any]:
    """Validate that splits have no donor leakage.

    Args:
        cells_df: DataFrame with cell data including donor_id column
        splits: Split manifest dictionary with 'folds' key
        donor_col: Column name for donor IDs

    Returns:
        Dictionary with validation results
    """
    results = {
        "valid": True,
        "issues": [],
        "warnings": [],
        "summary": {},
    }

    n_cells = len(cells_df)
    n_donors = cells_df[donor_col].nunique()

    results["summary"]["n_cells"] = n_cells
    results["summary"]["n_donors"] = n_donors
    results["summary"]["n_folds"] = len(splits.get("folds", []))

    all_donors = set(cells_df[donor_col].unique())

    for fold_idx, fold_spec in enumerate(splits.get("folds", [])):
        train_donors = set(fold_spec.get("train_donors", fold_spec.get("train", [])))
        val_donors = set(fold_spec.get("val_donors", fold_spec.get("val", [])))
        test_donors = set(fold_spec.get("test_donors", fold_spec.get("test", [])))

        train_val_overlap = train_donors & val_donors
        if train_val_overlap:
            results["valid"] = False
            results["issues"].append(
                f"Fold {fold_idx}: Train-Val donor overlap: {train_val_overlap}"
            )

        train_test_overlap = train_donors & test_donors
        if train_test_overlap:
            results["valid"] = False
            results["issues"].append(
                f"Fold {fold_idx}: Train-Test donor overlap: {train_test_overlap}"
            )

        val_test_overlap = val_donors & test_donors
        if val_test_overlap:
            results["valid"] = False
            results["issues"].append(
                f"Fold {fold_idx}: Val-Test donor overlap: {val_test_overlap}"
            )

        covered_donors = train_donors | val_donors | test_donors
        missing_donors = all_donors - covered_donors
        extra_donors = covered_donors - all_donors

        if missing_donors:
            results["warnings"].append(
                f"Fold {fold_idx}: Missing donors: {missing_donors}"
            )

        if extra_donors:
            results["warnings"].append(
                f"Fold {fold_idx}: Extra donors not in data: {extra_donors}"
            )

        train_cells = cells_df[cells_df[donor_col].isin(train_donors)]
        val_cells = cells_df[cells_df[donor_col].isin(val_donors)]
        test_cells = cells_df[cells_df[donor_col].isin(test_donors)]

        fold_summary = {
            "train_donors": len(train_donors),
            "val_donors": len(val_donors),
            "test_donors": len(test_donors),
            "train_cells": len(train_cells),
            "val_cells": len(val_cells),
            "test_cells": len(test_cells),
            "no_leakage": (
                len(train_val_overlap) == 0
                and len(train_test_overlap) == 0
                and len(val_test_overlap) == 0
            ),
        }
        results["summary"][f"fold_{fold_idx}"] = fold_summary

    return results


def validate_splits_from_files(
    cells_path: Path,
    splits_path: Path,
    donor_col: str = "donor_id",
) -> dict[str, Any]:
    """Validate splits by loading from files.

    Args:
        cells_path: Path to cells.parquet
        splits_path: Path to split_manifest.json
        donor_col: Column name for donor IDs

    Returns:
        Dictionary with validation results
    """
    cells_df = pd.read_parquet(cells_path)

    with open(splits_path) as f:
        splits = json.load(f)

    return validate_splits(cells_df, splits, donor_col=donor_col)


def check_paired_sample_leakage(
    cells_df: pd.DataFrame,
    splits: dict[str, Any],
    sample_col: str = "sample_id",
    donor_col: str = "donor_id",
) -> dict[str, Any]:
    """Check for paired sample leakage (samples from same donor in different splits).

    This is a stronger check than donor leakage - even if donors are split correctly,
    paired samples (e.g., tumor and adjacent normal from same patient) could leak.

    Args:
        cells_df: DataFrame with cell data
        splits: Split manifest dictionary
        sample_col: Column name for sample IDs
        donor_col: Column name for donor IDs

    Returns:
        Dictionary with leakage analysis
    """
    results = {
        "has_paired_leakage": False,
        "leakage_pairs": [],
        "summary": {},
    }

    if sample_col not in cells_df.columns:
        results["summary"]["error"] = f"Sample column '{sample_col}' not found"
        return results

    sample_to_donor = cells_df.groupby(sample_col)[donor_col].first().to_dict()

    for fold_idx, fold_spec in enumerate(splits.get("folds", [])):
        train_donors = set(fold_spec.get("train_donors", fold_spec.get("train", [])))
        test_donors = set(fold_spec.get("test_donors", fold_spec.get("test", [])))

        train_samples = set(
            cells_df[cells_df[donor_col].isin(train_donors)][sample_col].unique()
        )
        test_samples = set(
            cells_df[cells_df[donor_col].isin(test_donors)][sample_col].unique()
        )

        for train_sample in train_samples:
            train_donor = sample_to_donor.get(train_sample)
            for test_sample in test_samples:
                test_donor = sample_to_donor.get(test_sample)
                if train_donor == test_donor:
                    results["has_paired_leakage"] = True
                    results["leakage_pairs"].append({
                        "fold": fold_idx,
                        "train_sample": train_sample,
                        "test_sample": test_sample,
                        "shared_donor": train_donor,
                    })

    results["summary"]["n_leakage_pairs"] = len(results["leakage_pairs"])
    return results
