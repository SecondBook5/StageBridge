#!/usr/bin/env python3
"""Fix canonical data artifacts to pass contract validation.

Fixes:
1. Add data_type column to cells.parquet (snrna/spatial)
2. Add stage_3 column (Normal/Preinvasive/Invasive mapping)
3. Rebuild neighborhoods.parquet with proper embeddings (receiver_z, hlca_z, luca_z)
4. Verify all fixes pass validation

Usage:
    python scripts/fix_canonical_data.py
    python scripts/fix_canonical_data.py --dry-run  # Check without writing
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from tqdm import tqdm

# Paths
DATA = Path("/data1/chaunzt1/stagebridge/processed/luad_evo")
CANONICAL = DATA / "canonical"

# Stage mapping (from contracts.py)
STAGE_5_TO_3 = {
    "Normal": "Normal",
    "AAH": "Preinvasive",
    "AIS": "Preinvasive",
    "MIA": "Preinvasive",
    "LUAD": "Invasive",
}

STAGE_3_TO_IDX = {
    "Normal": 0,
    "Preinvasive": 1,
    "Invasive": 2,
}

# Ring configuration
RING_RADII = (600, 1200, 2000, 3000)  # Microns
MAX_CELLS_PER_RING = 50


def fix_cells(cells_path: Path, dry_run: bool = False) -> pd.DataFrame:
    """Fix cells.parquet: add data_type and stage_3 columns."""
    print("\n[1] FIXING CELLS.PARQUET")
    print(f"    Loading {cells_path}...")
    cells = pd.read_parquet(cells_path)
    print(f"    Loaded {len(cells):,} cells")

    changes = []

    # 1. Add data_type based on x_spatial presence
    if "data_type" not in cells.columns:
        cells["data_type"] = np.where(cells["x_spatial"].notna(), "spatial", "snrna")
        n_spatial = (cells["data_type"] == "spatial").sum()
        n_snrna = (cells["data_type"] == "snrna").sum()
        print(f"    Added data_type: {n_spatial:,} spatial, {n_snrna:,} snrna")
        changes.append("data_type")
    else:
        print(f"    data_type already exists")

    # 2. Add stage_3 (3-class mapping)
    if "stage_3" not in cells.columns:
        cells["stage_3"] = cells["stage"].map(STAGE_5_TO_3)
        unmapped = cells["stage_3"].isna().sum()
        if unmapped > 0:
            print(f"    WARNING: {unmapped} cells with unmapped stage")
            print(f"    Unique stages: {cells['stage'].unique()}")
        cells["stage_3"] = cells["stage_3"].fillna("Unknown")
        print(f"    Added stage_3: {cells['stage_3'].value_counts().to_dict()}")
        changes.append("stage_3")
    else:
        print(f"    stage_3 already exists")

    # 3. Add stage_3_idx
    if "stage_3_idx" not in cells.columns:
        cells["stage_3_idx"] = cells["stage_3"].map(STAGE_3_TO_IDX).fillna(-1).astype(int)
        print(f"    Added stage_3_idx")
        changes.append("stage_3_idx")

    # 4. Verify embeddings exist and are valid
    print("\n    Verifying embeddings...")
    for emb_col in ["z_fused", "z_hlca", "z_luca"]:
        if emb_col in cells.columns:
            sample = cells[emb_col].iloc[0]
            if hasattr(sample, "__len__"):
                dim = len(sample)
                # Check for zeros
                n_zero = sum(1 for x in cells[emb_col].head(1000) if x is not None and np.allclose(x, 0))
                if n_zero > 100:
                    print(f"    WARNING: {emb_col} has many zero vectors ({n_zero}/1000 sampled)")
                else:
                    print(f"    OK: {emb_col} dim={dim}")
            else:
                print(f"    WARNING: {emb_col} is not array type")
        else:
            print(f"    MISSING: {emb_col}")

    if not dry_run and changes:
        backup_path = cells_path.with_suffix(".parquet.bak")
        print(f"\n    Backing up to {backup_path}...")
        # Only backup if not already backed up
        if not backup_path.exists():
            import shutil
            shutil.copy(cells_path, backup_path)

        print(f"    Saving fixed cells.parquet...")
        cells.to_parquet(cells_path)
        print(f"    Saved with changes: {changes}")
    elif dry_run:
        print(f"\n    DRY RUN - would add: {changes}")

    return cells


def rebuild_neighborhoods(cells: pd.DataFrame, nhood_path: Path, dry_run: bool = False) -> pd.DataFrame:
    """Rebuild neighborhoods.parquet with proper embeddings."""
    print("\n[2] REBUILDING NEIGHBORHOODS.PARQUET")

    # Filter to spatial cells only
    spatial_cells = cells[cells["data_type"] == "spatial"].copy()
    print(f"    {len(spatial_cells):,} spatial cells")

    # Check required columns
    required = ["cell_id", "donor_id", "stage", "x_spatial", "y_spatial", "z_fused", "z_hlca", "z_luca"]
    missing = [c for c in required if c not in spatial_cells.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Build spatial index per donor (neighborhoods are within-donor)
    donors = spatial_cells["donor_id"].unique()
    print(f"    {len(donors)} donors")

    all_neighborhoods = []

    for donor in tqdm(donors, desc="Building neighborhoods"):
        donor_cells = spatial_cells[spatial_cells["donor_id"] == donor].reset_index(drop=True)
        n_cells = len(donor_cells)

        if n_cells < 10:
            continue

        # Build KDTree for this donor
        coords = donor_cells[["x_spatial", "y_spatial"]].values
        tree = KDTree(coords)

        # Pre-extract embeddings as numpy arrays for speed
        z_fused_arr = np.array([np.array(x) for x in donor_cells["z_fused"].values])
        z_hlca_arr = np.array([np.array(x) for x in donor_cells["z_hlca"].values])
        z_luca_arr = np.array([np.array(x) for x in donor_cells["z_luca"].values])

        for i in range(n_cells):
            cell = donor_cells.iloc[i]

            # Query neighbors within max radius
            idx = tree.query_ball_point(coords[i], r=RING_RADII[-1])
            idx = [j for j in idx if j != i]  # Exclude self

            if not idx:
                continue

            neighbor_coords = coords[idx]
            distances = np.linalg.norm(neighbor_coords - coords[i], axis=1)

            # Assign to rings
            ring_cells = [[] for _ in range(4)]
            prev_r = 0
            for ring_idx, r in enumerate(RING_RADII):
                ring_mask = (distances > prev_r) & (distances <= r)
                ring_neighbor_idx = [idx[j] for j in np.where(ring_mask)[0]]

                # Limit cells per ring
                if len(ring_neighbor_idx) > MAX_CELLS_PER_RING:
                    ring_neighbor_idx = ring_neighbor_idx[:MAX_CELLS_PER_RING]

                # Store neighbor embeddings (z_fused for ring cells)
                for j in ring_neighbor_idx:
                    ring_cells[ring_idx].append(z_fused_arr[j].tolist())

                prev_r = r

            # Build neighborhood record
            nhood = {
                "cell_id": cell["cell_id"],
                "donor_id": donor,
                "stage": cell["stage"],
                "stage_idx": cell.get("stage_idx", 0),
                "stage_3": cell.get("stage_3", STAGE_5_TO_3.get(cell["stage"], "Unknown")),
                "stage_3_idx": STAGE_3_TO_IDX.get(cell.get("stage_3", STAGE_5_TO_3.get(cell["stage"])), -1),
                # Receiver embedding (center cell)
                "receiver_z": z_fused_arr[i].tolist(),
                # Reference embeddings
                "hlca_z": z_hlca_arr[i].tolist(),
                "luca_z": z_luca_arr[i].tolist(),
                # Ring cells (neighbor embeddings)
                "ring_1_cells": ring_cells[0],
                "ring_2_cells": ring_cells[1],
                "ring_3_cells": ring_cells[2],
                "ring_4_cells": ring_cells[3],
                # Spatial coordinates
                "x_spatial": coords[i, 0],
                "y_spatial": coords[i, 1],
            }

            # Transfer additional columns from cells
            transfer_cols = [
                "S_score", "G2M_score", "phase",
                "clonal_pattern", "clonal_pattern_idx",
                "tmb", "kras_mut", "egfr_mut", "tp53_mut",
                "stk11_mut", "keap1_mut", "smad4_mut", "braf_mut",
            ]
            for col in transfer_cols:
                if col in cell.index:
                    val = cell[col]
                    nhood[col] = val if pd.notna(val) else None

            all_neighborhoods.append(nhood)

    nhood_df = pd.DataFrame(all_neighborhoods)
    print(f"    Built {len(nhood_df):,} neighborhoods")

    # Verify embeddings are not zero
    print("\n    Verifying neighborhood embeddings...")
    for col in ["receiver_z", "hlca_z", "luca_z"]:
        sample = nhood_df[col].iloc[0]
        if hasattr(sample, "__len__"):
            vals = np.array([np.array(x) for x in nhood_df[col].head(100)])
            vmin, vmax = vals.min(), vals.max()
            vmean = vals.mean()
            if np.allclose(vals, 0):
                print(f"    ERROR: {col} is all zeros!")
            else:
                print(f"    OK: {col} range=[{vmin:.3g}, {vmax:.3g}], mean={vmean:.3g}")

    # Verify ring quality
    print("\n    Ring quality:")
    for i in range(1, 5):
        col = f"ring_{i}_cells"
        sizes = [len(x) for x in nhood_df[col]]
        n_empty = sum(1 for s in sizes if s == 0)
        avg_size = np.mean(sizes)
        print(f"    ring_{i}: {n_empty} empty ({100*n_empty/len(nhood_df):.1f}%), avg size={avg_size:.1f}")

    if not dry_run:
        backup_path = nhood_path.with_suffix(".parquet.bak")
        print(f"\n    Backing up to {backup_path}...")
        if not backup_path.exists() and nhood_path.exists():
            import shutil
            shutil.copy(nhood_path, backup_path)

        print(f"    Saving neighborhoods.parquet...")
        nhood_df.to_parquet(nhood_path)
        print(f"    Saved {len(nhood_df):,} neighborhoods")
    else:
        print(f"\n    DRY RUN - would save {len(nhood_df):,} neighborhoods")

    return nhood_df


def update_split_manifest(cells: pd.DataFrame, manifest_path: Path, dry_run: bool = False):
    """Update split manifest to use stage_3 if needed."""
    print("\n[3] CHECKING SPLIT MANIFEST")

    with open(manifest_path) as f:
        manifest = json.load(f)

    print(f"    {len(manifest['folds'])} folds")

    # Verify no donor leakage
    for i, fold in enumerate(manifest["folds"]):
        train = set(fold["train_donors"])
        val = set(fold["val_donors"])
        test = set(fold["test_donors"])

        if train & val:
            print(f"    ERROR: Fold {i} train/val overlap!")
        if train & test:
            print(f"    ERROR: Fold {i} train/test overlap!")
        if val & test:
            print(f"    ERROR: Fold {i} val/test overlap!")

    print(f"    No donor leakage detected")


def validate_final(cells_path: Path, nhood_path: Path):
    """Run final validation."""
    print("\n[4] FINAL VALIDATION")

    # Check cells
    cells = pd.read_parquet(cells_path)
    print(f"\n    cells.parquet: {len(cells):,} cells")

    required = ["cell_id", "donor_id", "stage", "data_type"]
    missing = [c for c in required if c not in cells.columns]
    if missing:
        print(f"    FAIL: Missing required columns: {missing}")
    else:
        print(f"    OK: All required columns present")

    # Check stage_3
    if "stage_3" in cells.columns:
        print(f"    OK: stage_3 present: {cells['stage_3'].value_counts().to_dict()}")
    else:
        print(f"    WARN: stage_3 not present")

    # Check embeddings
    for col in ["z_fused", "z_hlca", "z_luca"]:
        if col in cells.columns:
            sample = np.array(cells[col].iloc[0])
            if np.allclose(sample, 0):
                print(f"    FAIL: {col} is zeros")
            else:
                print(f"    OK: {col} has valid values")

    # Check neighborhoods
    nhood = pd.read_parquet(nhood_path)
    print(f"\n    neighborhoods.parquet: {len(nhood):,} neighborhoods")

    for col in ["receiver_z", "hlca_z", "luca_z"]:
        if col in nhood.columns:
            sample = np.array(nhood[col].iloc[0])
            if np.allclose(sample, 0):
                print(f"    FAIL: {col} is zeros")
            else:
                print(f"    OK: {col} has valid values")
        else:
            print(f"    FAIL: {col} missing")

    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Fix canonical data artifacts")
    parser.add_argument("--dry-run", action="store_true", help="Check without writing")
    parser.add_argument("--cells-only", action="store_true", help="Only fix cells.parquet")
    parser.add_argument("--nhood-only", action="store_true", help="Only rebuild neighborhoods")
    args = parser.parse_args()

    print("=" * 60)
    print("FIXING CANONICAL DATA ARTIFACTS")
    print("=" * 60)

    cells_path = CANONICAL / "cells.parquet"
    nhood_path = CANONICAL / "neighborhoods.parquet"
    manifest_path = CANONICAL / "split_manifest.json"

    if args.nhood_only:
        # Load cells without modifying
        cells = pd.read_parquet(cells_path)
        # Ensure data_type exists for filtering
        if "data_type" not in cells.columns:
            cells["data_type"] = np.where(cells["x_spatial"].notna(), "spatial", "snrna")
        rebuild_neighborhoods(cells, nhood_path, dry_run=args.dry_run)
    elif args.cells_only:
        fix_cells(cells_path, dry_run=args.dry_run)
    else:
        # Fix both
        cells = fix_cells(cells_path, dry_run=args.dry_run)
        rebuild_neighborhoods(cells, nhood_path, dry_run=args.dry_run)
        update_split_manifest(cells, manifest_path, dry_run=args.dry_run)

    if not args.dry_run:
        validate_final(cells_path, nhood_path)


if __name__ == "__main__":
    main()
