#!/usr/bin/env python3
"""Create evolution features combining WES and clonal data.

Merges WES somatic mutations with inferCNV clonal features for
the evolution branch conditioning.

Usage:
    python scripts/create_evolution_features.py \
        --wes /path/to/wes_features.parquet \
        --clonal /path/to/clonal_features.parquet \
        --output /path/to/evolution_features.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import numpy as np


def create_evolution_features(
    wes_path: str | Path,
    clonal_path: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:
    """Merge WES and clonal features into evolution features.

    Args:
        wes_path: Path to wes_features.parquet (patient-level)
        clonal_path: Path to clonal_features.parquet (cell-level)
        output_path: Output path for evolution_features.parquet

    Returns:
        DataFrame with combined evolution features
    """
    output_path = Path(output_path)

    # Load WES (patient-level)
    print(f"Loading WES features...")
    wes_df = pd.read_parquet(wes_path)
    print(f"  {len(wes_df)} patients with WES data")
    print(f"  Columns: {wes_df.columns.tolist()}")

    # Standardize patient_id column
    if 'patient_id' not in wes_df.columns and 'donor_id' in wes_df.columns:
        wes_df = wes_df.rename(columns={'donor_id': 'patient_id'})

    # Load clonal (cell-level)
    print(f"\nLoading clonal features...")
    clonal_df = pd.read_parquet(clonal_path)
    print(f"  {len(clonal_df):,} cells with clonal data")

    # Select clonal columns (avoid duplicates with WES)
    clonal_cols = [
        'cell_id', 'patient_id', 'stage', 'sample_id',
        # Cell-level
        'cnv_score', 'cnv_score_z', 'clone_size', 'clone_rank',
        'is_major_clone', 'clone_fraction',
        # Patient-level
        'n_clones', 'clonal_entropy', 'clonal_diversity',
        'clonal_pattern_idx', 'aneuploidy_score',
        'clone_sharing_ratio', 'has_invasive_only_clones',
    ]
    clonal_cols = [c for c in clonal_cols if c in clonal_df.columns]
    clonal_df = clonal_df[clonal_cols]

    # Merge WES onto clonal
    # WES can be patient-level or sample-level (patient+stage)
    # Try sample-level first, fall back to patient-level
    if 'stage' in wes_df.columns and 'stage' in clonal_df.columns:
        # Sample-level merge (patient + stage)
        print("  Merging on patient_id + stage (sample-level)")
        result = clonal_df.merge(
            wes_df,
            on=['patient_id', 'stage'],
            how='left'
        )
    else:
        # Patient-level merge
        print("  Merging on patient_id only (patient-level)")
        wes_cols = [c for c in wes_df.columns if c not in ['stage', 'sample_id']]
        result = clonal_df.merge(
            wes_df[wes_cols],
            on='patient_id',
            how='left'
        )

    # Fill missing WES values (patients without WES data)
    wes_feature_cols = [c for c in wes_df.columns if c not in ['patient_id', 'stage', 'sample_id']]
    for col in wes_feature_cols:
        if col in result.columns:
            result[col] = result[col].fillna(0)

    # Summary
    print(f"\n=== Evolution Features Summary ===")
    print(f"Cells: {len(result):,}")
    print(f"Patients: {result['patient_id'].nunique()}")

    print(f"\nWES coverage:")
    wes_patients = set(wes_df['patient_id'].unique())
    clonal_patients = set(clonal_df['patient_id'].unique())
    overlap = wes_patients & clonal_patients
    print(f"  WES patients: {len(wes_patients)}")
    print(f"  Clonal patients: {len(clonal_patients)}")
    print(f"  Overlap: {len(overlap)}")

    print(f"\nFeature dimensions:")
    print(f"  WES features: {len(wes_feature_cols)}")
    clonal_feature_cols = [c for c in clonal_cols if c not in ['cell_id', 'patient_id', 'stage', 'sample_id']]
    print(f"  Clonal features: {len(clonal_feature_cols)}")
    print(f"  Total: {len(wes_feature_cols) + len(clonal_feature_cols)}")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    print(f"\nSaved to {output_path}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create evolution features")
    parser.add_argument("--wes", required=True, help="Path to wes_features.parquet")
    parser.add_argument("--clonal", required=True, help="Path to clonal_features.parquet")
    parser.add_argument("--output", required=True, help="Output path")
    args = parser.parse_args()

    create_evolution_features(args.wes, args.clonal, args.output)
