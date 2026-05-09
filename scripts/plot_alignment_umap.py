#!/usr/bin/env python3
"""Plot UMAP of spatial and snRNA embeddings to verify alignment.

Usage:
    python scripts/plot_alignment_umap.py \
        --embeddings-dir $DATA/processed/luad_evo/reference_embeddings \
        --output alignment_umap.png
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from umap import UMAP


def main():
    parser = argparse.ArgumentParser(description="Plot alignment UMAP")
    parser.add_argument("--embeddings-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--n-sample", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.output is None:
        args.output = args.embeddings_dir / "alignment_umap.png"

    print(f"Loading embeddings from {args.embeddings_dir}")

    spatial = pd.read_parquet(args.embeddings_dir / "spatial_fused_direct.parquet")
    snrna = pd.read_parquet(args.embeddings_dir / "snrna_fused_direct.parquet")

    print(f"  Spatial: {len(spatial):,}")
    print(f"  snRNA: {len(snrna):,}")

    # Columns are hlca_latent_* + luca_latent_* (40d fused)
    emb_cols = [c for c in spatial.columns if c.startswith("hlca_latent_") or c.startswith("luca_latent_")]
    print(f"  Embedding dim: {len(emb_cols)}")

    n_spatial = min(args.n_sample, len(spatial))
    n_snrna = min(args.n_sample, len(snrna))

    X_spatial = spatial[emb_cols].sample(n_spatial, random_state=args.seed).values
    X_snrna = snrna[emb_cols].sample(n_snrna, random_state=args.seed).values

    X = np.vstack([X_spatial, X_snrna])
    labels = ["Spatial"] * n_spatial + ["snRNA"] * n_snrna

    print(f"Running UMAP on {len(X):,} cells...")
    umap = UMAP(n_neighbors=30, min_dist=0.3, random_state=args.seed)
    emb = umap.fit_transform(X)

    print("Plotting...")
    fig, ax = plt.subplots(figsize=(10, 8))
    for modality, color in [("Spatial", "#1f77b4"), ("snRNA", "#ff7f0e")]:
        mask = np.array(labels) == modality
        ax.scatter(emb[mask, 0], emb[mask, 1], s=1, alpha=0.5, label=modality, c=color)
    ax.legend(markerscale=5)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title("Direct Inference Alignment (Fused 40d)")

    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
