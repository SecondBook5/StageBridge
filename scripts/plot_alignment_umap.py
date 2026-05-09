#!/usr/bin/env python
"""Plot UMAP of spatial + snRNA embeddings to verify alignment.

Usage:
    python scripts/plot_alignment_umap.py \
        --spatial-emb $DATA/processed/luad_evo/reference_geometry/spatial_fused_embedding.parquet \
        --snrna-emb $DATA/processed/luad_evo/reference_geometry/snrna_fused_embedding.parquet \
        --output alignment_umap.png

    # Or use HLCA only:
    python scripts/plot_alignment_umap.py \
        --spatial-emb $DATA/processed/luad_evo/reference_geometry/spatial_hlca_embedding.parquet \
        --snrna-emb $DATA/processed/luad_evo/snrna_with_celltypes.h5ad \
        --emb-key X_scANVI \
        --output alignment_umap_hlca.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from umap import UMAP


def load_embeddings(path: Path, emb_key: str = None, max_cells: int = None):
    """Load embeddings from parquet or h5ad."""
    path = Path(path)

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
        # Get numeric columns (embeddings)
        emb_cols = [c for c in df.columns if c.startswith(("hlca_", "luca_", "z_", "X_"))]
        if not emb_cols:
            # Try all numeric
            emb_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        emb = df[emb_cols].values
        index = df.index
    elif path.suffix in (".h5ad", ".h5"):
        import anndata
        adata = anndata.read_h5ad(path)
        if emb_key and emb_key in adata.obsm:
            emb = adata.obsm[emb_key]
        elif "X_scANVI" in adata.obsm:
            emb = adata.obsm["X_scANVI"]
        elif "X_scVI" in adata.obsm:
            emb = adata.obsm["X_scVI"]
        else:
            raise ValueError(f"No embedding found in {path}. Available: {list(adata.obsm.keys())}")
        index = adata.obs_names
    else:
        raise ValueError(f"Unknown file type: {path.suffix}")

    if max_cells and len(emb) > max_cells:
        idx = np.random.choice(len(emb), max_cells, replace=False)
        emb = emb[idx]
        index = index[idx]

    return emb, index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spatial-emb", type=Path, required=True)
    parser.add_argument("--snrna-emb", type=Path, required=True)
    parser.add_argument("--emb-key", type=str, default=None, help="Key for h5ad obsm")
    parser.add_argument("--output", type=Path, default=Path("alignment_umap.png"))
    parser.add_argument("--max-cells", type=int, default=50000, help="Subsample for speed")
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--min-dist", type=float, default=0.3)
    args = parser.parse_args()

    print("Loading embeddings...")
    spatial_emb, spatial_idx = load_embeddings(args.spatial_emb, args.emb_key, args.max_cells)
    snrna_emb, snrna_idx = load_embeddings(args.snrna_emb, args.emb_key, args.max_cells)

    print(f"  Spatial: {spatial_emb.shape}")
    print(f"  snRNA: {snrna_emb.shape}")

    # Check dimensions match
    if spatial_emb.shape[1] != snrna_emb.shape[1]:
        print(f"  WARNING: Dimension mismatch! Spatial={spatial_emb.shape[1]}, snRNA={snrna_emb.shape[1]}")
        # Try to use minimum
        min_dim = min(spatial_emb.shape[1], snrna_emb.shape[1])
        spatial_emb = spatial_emb[:, :min_dim]
        snrna_emb = snrna_emb[:, :min_dim]
        print(f"  Using first {min_dim} dimensions")

    # Combine
    combined = np.vstack([spatial_emb, snrna_emb])
    modality = ["Spatial"] * len(spatial_emb) + ["snRNA"] * len(snrna_emb)

    print(f"  Combined: {combined.shape}")

    # Run UMAP
    print("Running UMAP...")
    reducer = UMAP(n_neighbors=args.n_neighbors, min_dist=args.min_dist, random_state=42)
    umap_emb = reducer.fit_transform(combined)

    # Plot
    print("Plotting...")
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = {"Spatial": "#1f77b4", "snRNA": "#ff7f0e"}

    for mod in ["snRNA", "Spatial"]:  # Plot snRNA first so spatial is on top
        mask = np.array(modality) == mod
        ax.scatter(
            umap_emb[mask, 0],
            umap_emb[mask, 1],
            c=colors[mod],
            label=f"{mod} (n={mask.sum():,})",
            s=1,
            alpha=0.3,
        )

    ax.legend(markerscale=10)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title("Spatial + snRNA Embedding Alignment")

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved: {args.output}")

    # Also save coordinates
    coords_path = args.output.with_suffix(".csv")
    coords_df = pd.DataFrame({
        "UMAP1": umap_emb[:, 0],
        "UMAP2": umap_emb[:, 1],
        "modality": modality,
    })
    coords_df.to_csv(coords_path, index=False)
    print(f"Saved coordinates: {coords_path}")


if __name__ == "__main__":
    main()
