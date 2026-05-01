#!/usr/bin/env python3
"""Create clonal features from inferCNV clone assignments.

Computes cell-level and patient-level clonal features for the evolution branch.

Usage:
    python scripts/create_clonal_features.py \
        --clone-assignments /path/to/clone_assignments.parquet \
        --clonal-details /path/to/clonal_analysis_details.csv \
        --clonal-patterns /path/to/clonal_patterns.json \
        --output /path/to/clonal_features.parquet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def compute_cell_level_features(clone_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-cell clonal features.

    Args:
        clone_df: DataFrame with cell_id, patient_id, cnv_leiden, cnv_score

    Returns:
        DataFrame with cell-level clonal features
    """
    df = clone_df.copy()

    # Clone size: number of cells in each clone
    clone_sizes = df.groupby('cnv_leiden').size().rename('clone_size')
    df = df.merge(clone_sizes, left_on='cnv_leiden', right_index=True)

    # Clone rank within patient (0 = largest clone)
    df['clone_rank'] = df.groupby('patient_id')['clone_size'].rank(
        method='dense', ascending=False
    ).astype(int) - 1

    # Is major clone (largest in patient)
    df['is_major_clone'] = (df['clone_rank'] == 0).astype(int)

    # Clone fraction within patient
    patient_sizes = df.groupby('patient_id').size().rename('patient_n_cells')
    df = df.merge(patient_sizes, left_on='patient_id', right_index=True)
    df['clone_fraction'] = df['clone_size'] / df['patient_n_cells']

    # Normalized CNV score (z-score within patient)
    df['cnv_score_z'] = df.groupby('patient_id')['cnv_score'].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-8)
    )

    return df


def compute_patient_level_features(
    clone_df: pd.DataFrame,
    clonal_details: pd.DataFrame,
    clonal_patterns: dict[str, str],
) -> pd.DataFrame:
    """Compute per-patient clonal features.

    Args:
        clone_df: DataFrame with cell-level clone assignments
        clonal_details: DataFrame with clonal analysis details
        clonal_patterns: Dict mapping patient_id -> pattern (1a, 1b, uncategorized)

    Returns:
        DataFrame with patient-level clonal features
    """
    # Basic counts from cell data
    patient_stats = clone_df.groupby('patient_id').agg(
        n_cells=('cell_id', 'count'),
        n_clones=('cnv_leiden', 'nunique'),
        cnv_score_mean=('cnv_score', 'mean'),
        cnv_score_std=('cnv_score', 'std'),
    ).reset_index()

    # Clonal diversity (Shannon entropy)
    def shannon_entropy(group):
        counts = group['cnv_leiden'].value_counts(normalize=True)
        return -np.sum(counts * np.log(counts + 1e-10))

    entropy = clone_df.groupby('patient_id').apply(
        shannon_entropy, include_groups=False
    ).rename('clonal_entropy')
    patient_stats = patient_stats.merge(entropy, left_on='patient_id', right_index=True)

    # Gini-Simpson diversity (1 - sum(p^2))
    def gini_simpson(group):
        counts = group['cnv_leiden'].value_counts(normalize=True)
        return 1 - np.sum(counts ** 2)

    diversity = clone_df.groupby('patient_id').apply(
        gini_simpson, include_groups=False
    ).rename('clonal_diversity')
    patient_stats = patient_stats.merge(diversity, left_on='patient_id', right_index=True)

    # Merge clonal analysis details
    if clonal_details is not None:
        details_cols = [
            'patient_id', 'n_clones_precursor', 'n_clones_invasive',
            'n_clones_shared', 'aneuploidy_score', 'confidence'
        ]
        available_cols = [c for c in details_cols if c in clonal_details.columns]
        patient_stats = patient_stats.merge(
            clonal_details[available_cols],
            on='patient_id',
            how='left'
        )

        # Computed features from details
        if 'n_clones_shared' in patient_stats.columns:
            patient_stats['clone_sharing_ratio'] = (
                patient_stats['n_clones_shared'] /
                patient_stats[['n_clones_precursor', 'n_clones_invasive']].max(axis=1)
            ).fillna(0)

        if 'n_clones_invasive' in patient_stats.columns:
            patient_stats['has_invasive_only_clones'] = (
                patient_stats['n_clones_invasive'] > patient_stats['n_clones_shared']
            ).astype(int)

    # Add clonal pattern
    patient_stats['clonal_pattern'] = patient_stats['patient_id'].map(clonal_patterns)
    patient_stats['clonal_pattern'] = patient_stats['clonal_pattern'].fillna('unknown')

    # Encode pattern as numeric
    pattern_map = {'1a': 0, '1b': 1, 'uncategorized': 2, 'unknown': 3}
    patient_stats['clonal_pattern_idx'] = patient_stats['clonal_pattern'].map(pattern_map)

    return patient_stats


def create_clonal_features(
    clone_assignments_path: str | Path,
    clonal_details_path: str | Path | None,
    clonal_patterns_path: str | Path | None,
    output_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create clonal features for evolution branch.

    Args:
        clone_assignments_path: Path to clone_assignments.parquet
        clonal_details_path: Path to clonal_analysis_details.csv
        clonal_patterns_path: Path to clonal_patterns.json
        output_path: Output path for clonal_features.parquet

    Returns:
        (cell_features, patient_features) DataFrames
    """
    output_path = Path(output_path)

    # Load data
    print(f"Loading clone assignments...")
    clone_df = pd.read_parquet(clone_assignments_path)
    print(f"  {len(clone_df):,} cells")

    clonal_details = None
    if clonal_details_path and Path(clonal_details_path).exists():
        clonal_details = pd.read_csv(clonal_details_path)
        print(f"  {len(clonal_details)} patients in clonal details")

    clonal_patterns = {}
    if clonal_patterns_path and Path(clonal_patterns_path).exists():
        with open(clonal_patterns_path) as f:
            clonal_patterns = json.load(f)
        print(f"  {len(clonal_patterns)} patients in clonal patterns")

    # Compute features
    print("\nComputing cell-level features...")
    cell_features = compute_cell_level_features(clone_df)

    print("Computing patient-level features...")
    patient_features = compute_patient_level_features(
        clone_df, clonal_details, clonal_patterns
    )

    # Merge patient features back to cells
    patient_cols = [
        'patient_id', 'n_clones', 'clonal_entropy', 'clonal_diversity',
        'clonal_pattern', 'clonal_pattern_idx', 'aneuploidy_score',
        'clone_sharing_ratio', 'has_invasive_only_clones'
    ]
    available_patient_cols = [c for c in patient_cols if c in patient_features.columns]

    cell_features = cell_features.merge(
        patient_features[available_patient_cols],
        on='patient_id',
        how='left',
        suffixes=('', '_patient')
    )

    # Select output columns
    output_cols = [
        'cell_id', 'patient_id', 'stage', 'sample_id',
        # Cell-level
        'cnv_leiden', 'cnv_score', 'cnv_score_z',
        'clone_size', 'clone_rank', 'is_major_clone', 'clone_fraction',
        # Patient-level
        'n_clones', 'clonal_entropy', 'clonal_diversity',
        'clonal_pattern', 'clonal_pattern_idx',
    ]
    # Add optional columns if present
    for col in ['aneuploidy_score', 'clone_sharing_ratio', 'has_invasive_only_clones']:
        if col in cell_features.columns:
            output_cols.append(col)

    output_df = cell_features[[c for c in output_cols if c in cell_features.columns]]

    # Summary
    print(f"\n=== Clonal Features Summary ===")
    print(f"Cells: {len(output_df):,}")
    print(f"Patients: {output_df['patient_id'].nunique()}")
    print(f"\nClonal patterns:")
    print(output_df.groupby('clonal_pattern')['patient_id'].nunique())
    print(f"\nClone counts per patient:")
    print(patient_features[['patient_id', 'n_clones', 'clonal_entropy']].to_string())

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_parquet(output_path, index=False)
    print(f"\nSaved cell-level features to {output_path}")

    # Also save patient-level summary
    patient_output = output_path.parent / "clonal_patient_features.parquet"
    patient_features.to_parquet(patient_output, index=False)
    print(f"Saved patient-level features to {patient_output}")

    return output_df, patient_features


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create clonal features")
    parser.add_argument("--clone-assignments", required=True)
    parser.add_argument("--clonal-details", default=None)
    parser.add_argument("--clonal-patterns", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    create_clonal_features(
        args.clone_assignments,
        args.clonal_details,
        args.clonal_patterns,
        args.output,
    )
