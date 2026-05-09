#!/usr/bin/env python3
"""Generate donor-held-out cross-validation splits.

Creates split_manifest.json with NO LEAKAGE:
- Each donor appears in exactly ONE of train/val/test per fold
- Rotates through folds so each donor gets tested

Usage:
    python scripts/create_split_manifest.py \
        --cells /path/to/cells.parquet \
        --output /path/to/split_manifest.json \
        --n-folds 5
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def create_split_manifest(
    cells_path: Path,
    output_path: Path,
    n_folds: int = 5,
) -> dict:
    """Create donor-held-out CV splits.

    Args:
        cells_path: Path to cells.parquet
        output_path: Where to save split_manifest.json
        n_folds: Number of CV folds

    Returns:
        Split manifest dict
    """
    print(f"Loading cells from {cells_path}...")
    cells = pd.read_parquet(cells_path, columns=["donor_id"])

    donors = sorted(cells["donor_id"].unique().tolist())
    n_donors = len(donors)
    print(f"  Found {n_donors} unique donors: {donors}")

    if n_donors < 3:
        raise ValueError(f"Need at least 3 donors for train/val/test, got {n_donors}")

    n_folds = min(n_folds, n_donors)
    print(f"  Creating {n_folds}-fold CV splits")

    folds = []
    for fold_idx in range(n_folds):
        # Rotate: fold_idx is test, fold_idx+1 is val, rest is train
        test_idx = fold_idx
        val_idx = (fold_idx + 1) % n_donors
        train_indices = [i for i in range(n_donors) if i != test_idx and i != val_idx]

        fold = {
            "fold": fold_idx,
            "train_donors": [donors[i] for i in train_indices],
            "val_donors": [donors[val_idx]],
            "test_donors": [donors[test_idx]],
        }

        # Verify NO LEAKAGE
        train_set = set(fold["train_donors"])
        val_set = set(fold["val_donors"])
        test_set = set(fold["test_donors"])

        assert len(train_set & val_set) == 0, f"Fold {fold_idx}: Train/val overlap!"
        assert len(train_set & test_set) == 0, f"Fold {fold_idx}: Train/test overlap!"
        assert len(val_set & test_set) == 0, f"Fold {fold_idx}: Val/test overlap!"
        assert len(train_set) + len(val_set) + len(test_set) == n_donors, f"Fold {fold_idx}: Missing donors!"

        folds.append(fold)
        print(f"    Fold {fold_idx}: train={len(fold['train_donors'])}, val=1, test=1")

    split_manifest = {"n_folds": n_folds, "folds": folds}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(split_manifest, f, indent=2)

    print(f"\nSaved: {output_path}")
    return split_manifest


def main():
    parser = argparse.ArgumentParser(description="Create donor-held-out CV splits")
    parser.add_argument("--cells", type=Path, required=True, help="Path to cells.parquet")
    parser.add_argument("--output", type=Path, required=True, help="Output split_manifest.json path")
    parser.add_argument("--n-folds", type=int, default=5, help="Number of CV folds")
    args = parser.parse_args()

    create_split_manifest(args.cells, args.output, args.n_folds)


if __name__ == "__main__":
    main()
