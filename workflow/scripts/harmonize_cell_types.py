#!/usr/bin/env python3
"""
Harmonize cell type labels from dual references (HLCA + LuCA).

For lung cancer progression studies, we need labels that capture:
- Normal cell types (from HLCA) - for Normal/early stages
- Cancer-specific types (from LuCA) - for late stages
- Transitional states - cells mapping to both

Strategy:
1. Load predictions from both references with confidence scores
2. For each cell, determine dominant reference based on confidence
3. Create harmonized label vocabulary
4. Flag ambiguous/transitional cells

This avoids biasing toward either reference and lets the data
reveal the actual progression states.
"""

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


def load_reference_labels(labels_path: Path) -> pd.DataFrame:
    """Load labels parquet with predictions and confidence."""
    df = pd.read_parquet(labels_path)
    return df


def harmonize_labels(
    hlca_labels: pd.DataFrame,
    luca_labels: pd.DataFrame,
    confidence_threshold: float = 0.5,
    ambiguity_threshold: float = 0.3,
) -> pd.DataFrame:
    """
    Create harmonized cell type labels from dual references.

    Args:
        hlca_labels: DataFrame with 'cell_type' and 'confidence' from HLCA
        luca_labels: DataFrame with 'cell_type' and 'confidence' from LuCA
        confidence_threshold: Min confidence to use a reference's label
        ambiguity_threshold: If both confidences within this range, mark as transitional

    Returns:
        DataFrame with harmonized labels and metadata
    """
    # Ensure same index
    common_cells = hlca_labels.index.intersection(luca_labels.index)
    print(f"Common cells: {len(common_cells)}")

    hlca = hlca_labels.loc[common_cells].copy()
    luca = luca_labels.loc[common_cells].copy()

    # Get confidence columns (might be named differently)
    # Check common column names for confidence/probability
    hlca_conf_candidates = ['confidence', 'mapping_confidence', 'hlca_max_prob', 'max_prob', 'prob']
    luca_conf_candidates = ['confidence', 'mapping_confidence', 'luca_max_prob', 'max_prob', 'prob']

    hlca_conf_col = None
    for col in hlca_conf_candidates:
        if col in hlca.columns:
            hlca_conf_col = col
            break

    luca_conf_col = None
    for col in luca_conf_candidates:
        if col in luca.columns:
            luca_conf_col = col
            break

    # Handle missing confidence columns
    if hlca_conf_col is None:
        print("Warning: No confidence column in HLCA labels, using 1.0")
        hlca['confidence'] = 1.0
        hlca_conf_col = 'confidence'
    if luca_conf_col is None:
        print("Warning: No confidence column in LuCA labels, using 1.0")
        luca['confidence'] = 1.0
        luca_conf_col = 'confidence'

    # Get label columns
    hlca_label_col = 'cell_type' if 'cell_type' in hlca.columns else 'predicted_label'
    luca_label_col = 'cell_type' if 'cell_type' in luca.columns else 'predicted_label'

    if hlca_label_col not in hlca.columns:
        # Try to find any column that looks like a label
        for col in hlca.columns:
            if 'label' in col.lower() or 'type' in col.lower():
                hlca_label_col = col
                break

    if luca_label_col not in luca.columns:
        for col in luca.columns:
            if 'label' in col.lower() or 'type' in col.lower():
                luca_label_col = col
                break

    print(f"HLCA label column: {hlca_label_col}, confidence column: {hlca_conf_col}")
    print(f"LuCA label column: {luca_label_col}, confidence column: {luca_conf_col}")

    hlca_conf = hlca[hlca_conf_col].values
    luca_conf = luca[luca_conf_col].values
    hlca_type = hlca[hlca_label_col].values
    luca_type = luca[luca_label_col].values

    # Normalize confidences to [0, 1] if needed
    if hlca_conf.max() > 1:
        hlca_conf = hlca_conf / hlca_conf.max()
    if luca_conf.max() > 1:
        luca_conf = luca_conf / luca_conf.max()

    # Determine dominant reference for each cell
    # ALWAYS pick ONE label (highest confidence wins) - no compound labels
    results = []
    for i in range(len(common_cells)):
        h_conf = hlca_conf[i]
        l_conf = luca_conf[i]
        h_type = hlca_type[i]
        l_type = luca_type[i]

        # Check if confidences are similar (ambiguous)
        conf_diff = abs(h_conf - l_conf)
        is_ambiguous = (conf_diff < ambiguity_threshold and
                        h_conf > confidence_threshold and
                        l_conf > confidence_threshold and
                        h_type != l_type)

        # Always pick the label with higher confidence
        if h_conf >= l_conf and h_conf > confidence_threshold:
            # HLCA wins or tie goes to HLCA (healthy reference)
            cell_type = h_type
            source = "hlca"
        elif l_conf > h_conf and l_conf > confidence_threshold:
            # LuCA wins
            cell_type = l_type
            source = "luca"
        elif h_conf > l_conf:
            # HLCA higher but below threshold - use it anyway
            cell_type = h_type
            source = "hlca_low_conf"
        elif l_conf > h_conf:
            # LuCA higher but below threshold
            cell_type = l_type
            source = "luca_low_conf"
        else:
            # Equal and both low - default to HLCA
            cell_type = h_type
            source = "hlca_low_conf"

        results.append({
            "cell_type": cell_type,
            "source": source,
            "is_ambiguous": is_ambiguous,
            "hlca_confidence": h_conf,
            "luca_confidence": l_conf,
            "hlca_label": h_type,
            "luca_label": l_type,
        })

    result_df = pd.DataFrame(results, index=common_cells)

    # Print summary
    print("\n=== Harmonization Summary ===")
    print(f"Total cells: {len(result_df)}")
    print(f"\nSource distribution:")
    print(result_df['source'].value_counts())
    print(f"\nAmbiguous cells (similar confidence, different labels): {result_df['is_ambiguous'].sum()} ({100*result_df['is_ambiguous'].mean():.1f}%)")
    print(f"\nTop 20 cell types:")
    print(result_df['cell_type'].value_counts().head(20))

    return result_df


def main():
    parser = argparse.ArgumentParser(description="Harmonize dual-reference cell type labels")
    parser.add_argument("--snrna", required=True, help="Path to snRNA h5ad")
    parser.add_argument("--hlca-labels", required=True, help="Path to HLCA labels parquet")
    parser.add_argument("--luca-labels", required=True, help="Path to LuCA labels parquet")
    parser.add_argument("--output", required=True, help="Output snRNA h5ad with harmonized labels")
    parser.add_argument("--confidence-threshold", type=float, default=0.5,
                        help="Min confidence to use reference label")
    parser.add_argument("--ambiguity-threshold", type=float, default=0.3,
                        help="Max confidence diff to flag as transitional")
    args = parser.parse_args()

    print(f"Loading snRNA data from {args.snrna}...")
    adata = ad.read_h5ad(args.snrna)
    print(f"  Shape: {adata.shape}")

    print(f"\nLoading HLCA labels from {args.hlca_labels}...")
    hlca_labels = load_reference_labels(Path(args.hlca_labels))
    print(f"  Shape: {hlca_labels.shape}")
    print(f"  Columns: {hlca_labels.columns.tolist()}")

    print(f"\nLoading LuCA labels from {args.luca_labels}...")
    luca_labels = load_reference_labels(Path(args.luca_labels))
    print(f"  Shape: {luca_labels.shape}")
    print(f"  Columns: {luca_labels.columns.tolist()}")

    # Harmonize
    print("\nHarmonizing labels...")
    harmonized = harmonize_labels(
        hlca_labels,
        luca_labels,
        confidence_threshold=args.confidence_threshold,
        ambiguity_threshold=args.ambiguity_threshold,
    )

    # Add to adata
    common_cells = harmonized.index.intersection(adata.obs_names)
    print(f"\nCells in adata: {len(common_cells)}")

    # Add harmonized columns to obs
    adata.obs["cell_type"] = harmonized.loc[adata.obs_names, "cell_type"]
    adata.obs["cell_type_source"] = harmonized.loc[adata.obs_names, "source"]
    adata.obs["is_ambiguous"] = harmonized.loc[adata.obs_names, "is_ambiguous"]
    adata.obs["hlca_confidence"] = harmonized.loc[adata.obs_names, "hlca_confidence"]
    adata.obs["luca_confidence"] = harmonized.loc[adata.obs_names, "luca_confidence"]
    adata.obs["hlca_label"] = harmonized.loc[adata.obs_names, "hlca_label"]
    adata.obs["luca_label"] = harmonized.loc[adata.obs_names, "luca_label"]

    # Make cell_type categorical
    adata.obs["cell_type"] = adata.obs["cell_type"].astype("category")

    # Save
    print(f"\nSaving to {args.output}...")
    adata.write_h5ad(args.output)
    print("Done!")


if __name__ == "__main__":
    main()
