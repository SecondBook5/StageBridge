#!/usr/bin/env python3
"""Validate donor-held-out splits for data leakage before training.

This is a critical pre-training check for scientific integrity.
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def validate_splits(cells_path: Path, splits_path: Path, output_path: Path) -> dict:
    """Validate that splits have no donor leakage.

    Returns:
        dict with validation results
    """
    results = {
        "valid": True,
        "issues": [],
        "warnings": [],
        "summary": {},
    }

    # Load data
    cells_df = pd.read_parquet(cells_path)
    with open(splits_path) as f:
        splits = json.load(f)

    n_cells = len(cells_df)
    n_donors = cells_df['donor_id'].nunique()

    results["summary"]["n_cells"] = n_cells
    results["summary"]["n_donors"] = n_donors
    results["summary"]["n_folds"] = len(splits.get("folds", []))

    print(f"Validating splits: {n_cells:,} cells from {n_donors} donors")
    print(f"  Split manifest: {len(splits.get('folds', []))} folds")

    # Check each fold for donor leakage
    all_donors = set(cells_df['donor_id'].unique())

    for fold_idx, fold_spec in enumerate(splits.get("folds", [])):
        train_donors = set(fold_spec.get("train_donors", []))
        val_donors = set(fold_spec.get("val_donors", []))
        test_donors = set(fold_spec.get("test_donors", []))

        # Check 1: No overlap between train and val
        train_val_overlap = train_donors & val_donors
        if train_val_overlap:
            results["valid"] = False
            results["issues"].append(f"Fold {fold_idx}: Train-Val donor overlap: {train_val_overlap}")
            print(f"  [FAIL] Fold {fold_idx}: Train-Val donor overlap!")

        # Check 2: No overlap between train and test
        train_test_overlap = train_donors & test_donors
        if train_test_overlap:
            results["valid"] = False
            results["issues"].append(f"Fold {fold_idx}: Train-Test donor overlap: {train_test_overlap}")
            print(f"  [FAIL] Fold {fold_idx}: Train-Test donor overlap!")

        # Check 3: No overlap between val and test
        val_test_overlap = val_donors & test_donors
        if val_test_overlap:
            results["valid"] = False
            results["issues"].append(f"Fold {fold_idx}: Val-Test donor overlap: {val_test_overlap}")
            print(f"  [FAIL] Fold {fold_idx}: Val-Test donor overlap!")

        # Check 4: All donors accounted for
        covered_donors = train_donors | val_donors | test_donors
        missing_donors = all_donors - covered_donors
        extra_donors = covered_donors - all_donors

        if missing_donors:
            results["warnings"].append(f"Fold {fold_idx}: Missing donors: {missing_donors}")
            print(f"  [WARN] Fold {fold_idx}: {len(missing_donors)} donors not in any split")

        if extra_donors:
            results["warnings"].append(f"Fold {fold_idx}: Extra donors not in data: {extra_donors}")
            print(f"  [WARN] Fold {fold_idx}: {len(extra_donors)} split donors not in data")

        # Calculate cell counts per split
        train_cells = cells_df[cells_df['donor_id'].isin(train_donors)]
        val_cells = cells_df[cells_df['donor_id'].isin(val_donors)]
        test_cells = cells_df[cells_df['donor_id'].isin(test_donors)]

        fold_summary = {
            "train_donors": len(train_donors),
            "val_donors": len(val_donors),
            "test_donors": len(test_donors),
            "train_cells": len(train_cells),
            "val_cells": len(val_cells),
            "test_cells": len(test_cells),
            "no_leakage": len(train_val_overlap) == 0 and len(train_test_overlap) == 0 and len(val_test_overlap) == 0,
        }
        results["summary"][f"fold_{fold_idx}"] = fold_summary

        if fold_summary["no_leakage"]:
            print(f"  [PASS] Fold {fold_idx}: No leakage (train={len(train_cells):,}, val={len(val_cells):,}, test={len(test_cells):,})")

    # Final status
    if results["valid"]:
        print("\n[PASS] All splits validated - no donor leakage detected")
    else:
        print(f"\n[FAIL] Validation failed with {len(results['issues'])} issues")
        for issue in results["issues"]:
            print(f"  - {issue}")

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nValidation report saved to: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Validate donor-held-out splits")
    parser.add_argument("--cells", type=str, required=True, help="cells.parquet path")
    parser.add_argument("--splits", type=str, required=True, help="split_manifest.json path")
    parser.add_argument("--output", type=str, required=True, help="Output validation report path")
    args = parser.parse_args()

    results = validate_splits(
        cells_path=Path(args.cells),
        splits_path=Path(args.splits),
        output_path=Path(args.output),
    )

    # Exit with error if validation failed
    if not results["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
