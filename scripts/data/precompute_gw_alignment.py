#!/usr/bin/env python
"""Precompute Gromov-Wasserstein alignment between HLCA and LuCA.

This script:
1. Loads the cells.parquet with HLCA and LuCA embeddings
2. Samples representative cells (stratified by stage/celltype)
3. Computes GW coupling to find structure-preserving alignment
4. Trains a neural transport map for inference on new cells
5. Saves everything for use during training

Usage:
    python scripts/precompute_gw_alignment.py \\
        --cells /path/to/cells.parquet \\
        --output-dir /path/to/gw_alignment \\
        --n-reference-cells 5000
"""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np

from stagebridge.reference.gw_precompute import (
    precompute_gw_alignment,
    GWPrecomputeConfig,
)


def main():
    parser = argparse.ArgumentParser(description="Precompute GW alignment")
    parser.add_argument("--cells", type=Path, required=True, help="Path to cells.parquet")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--n-reference-cells", type=int, default=5000)
    parser.add_argument("--sinkhorn-reg", type=float, default=0.05)
    parser.add_argument("--gw-iters", type=int, default=50)
    parser.add_argument("--no-stratify", action="store_true", help="Disable stratification")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    print(f"Loading cells from {args.cells}...")
    cells = pd.read_parquet(args.cells)

    # Extract HLCA embeddings (z_hlca_0 through z_hlca_29)
    hlca_cols = [f"z_hlca_{i}" for i in range(30)]
    if hlca_cols[0] not in cells.columns:
        # Try alternative format
        if "z_hlca" in cells.columns:
            hlca = np.stack(cells["z_hlca"].values)
        else:
            raise ValueError("Cannot find HLCA embeddings in cells.parquet")
    else:
        hlca = cells[hlca_cols].values

    # Extract LuCA embeddings (z_luca_0 through z_luca_9)
    luca_cols = [f"z_luca_{i}" for i in range(10)]
    if luca_cols[0] not in cells.columns:
        if "z_luca" in cells.columns:
            luca = np.stack(cells["z_luca"].values)
        else:
            raise ValueError("Cannot find LuCA embeddings in cells.parquet")
    else:
        luca = cells[luca_cols].values

    print(f"  HLCA shape: {hlca.shape}")
    print(f"  LuCA shape: {luca.shape}")

    # Get stratification labels
    stages = None
    cell_types = None
    if not args.no_stratify:
        if "stage" in cells.columns:
            stages = cells["stage"].values
            print(f"  Stages: {np.unique(stages)}")
        if "cell_type" in cells.columns:
            cell_types = cells["cell_type"].values
            print(f"  Cell types: {len(np.unique(cell_types))} unique")

    config = GWPrecomputeConfig(
        n_reference_cells=args.n_reference_cells,
        sinkhorn_reg=args.sinkhorn_reg,
        gw_iters=args.gw_iters,
        stratify_by_stage=stages is not None,
        stratify_by_celltype=cell_types is not None,
    )

    result = precompute_gw_alignment(
        hlca_embeddings=hlca,
        luca_embeddings=luca,
        output_dir=args.output_dir,
        config=config,
        stages=stages,
        cell_types=cell_types,
        device=args.device,
    )

    print(f"\nGW alignment precomputed!")
    print(f"  GW cost: {result['gw_cost']:.4f}")
    print(f"  Reference cells: {len(result['reference_indices'])}")
    print(f"  Output: {result['output_dir']}")


if __name__ == "__main__":
    main()
