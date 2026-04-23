#!/usr/bin/env python3
"""
Compute CytoTRACE and pseudotime progression scores for StageBridge.

This script adds:
1. CytoTRACE - stemness/differentiation potential
2. Diffusion pseudotime (DPT) - continuous progression coordinate

These scores complement discrete stage labels for transition modeling.

Usage:
    python scripts/compute_progression_scores.py \
        --snrna $DATA/processed/luad_evo/snrna_with_celltypes.h5ad \
        --cells $DATA/processed/luad_evo/canonical/cells.parquet \
        --output_dir $DATA/processed/luad_evo/canonical/progression
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import warnings

warnings.filterwarnings("ignore")


def compute_cytotrace(adata: sc.AnnData) -> np.ndarray:
    """
    Compute CytoTRACE-like score based on gene count correlation.

    CytoTRACE principle: more differentiated cells express fewer genes,
    and genes correlating with gene count indicate differentiation state.

    Returns array of scores (higher = more stem-like/less differentiated).
    """
    print("Computing CytoTRACE scores...")

    # Gene counts per cell (number of detected genes)
    if "n_genes" not in adata.obs.columns:
        adata.obs["n_genes"] = (adata.X > 0).sum(axis=1).A1 if hasattr(adata.X, 'A1') else (adata.X > 0).sum(axis=1)

    gene_counts = adata.obs["n_genes"].values.astype(float)

    # Normalize gene counts to 0-1 range
    gc_min, gc_max = gene_counts.min(), gene_counts.max()
    if gc_max > gc_min:
        cytotrace_score = (gene_counts - gc_min) / (gc_max - gc_min)
    else:
        cytotrace_score = np.zeros_like(gene_counts)

    # CytoTRACE: higher gene count = more stem-like
    # (inverse of differentiation)
    return cytotrace_score


def compute_diffusion_pseudotime(
    adata: sc.AnnData,
    root_stage: str = "Normal",
    n_neighbors: int = 30,
    n_pcs: int = 50,
) -> np.ndarray:
    """
    Compute diffusion pseudotime rooted at Normal epithelium.

    Returns array of pseudotime values (0 = root/Normal, higher = more progressed).
    """
    print(f"Computing diffusion pseudotime (root: {root_stage})...")

    # Work on a copy to avoid modifying original
    adata = adata.copy()

    # Ensure we have PCA
    if "X_pca" not in adata.obsm:
        print("  Computing PCA...")
        sc.pp.pca(adata, n_comps=min(n_pcs, adata.n_vars - 1))

    # Compute neighbors
    print("  Computing neighbors...")
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep="X_pca")

    # Compute diffusion map
    print("  Computing diffusion map...")
    sc.tl.diffmap(adata, n_comps=15)

    # Find root cell (most central Normal cell)
    stage_col = "stage" if "stage" in adata.obs.columns else None
    if stage_col and root_stage in adata.obs[stage_col].values:
        root_mask = adata.obs[stage_col] == root_stage
        root_indices = np.where(root_mask)[0]

        # Pick the Normal cell closest to centroid of Normal cells in diffmap
        diffmap = adata.obsm["X_diffmap"]
        normal_diffmap = diffmap[root_mask]
        centroid = normal_diffmap.mean(axis=0)
        distances = np.linalg.norm(normal_diffmap - centroid, axis=1)
        root_cell_local = np.argmin(distances)
        root_cell = root_indices[root_cell_local]
    else:
        # Fallback: use first cell
        print(f"  Warning: '{root_stage}' not found, using first cell as root")
        root_cell = 0

    adata.uns["iroot"] = root_cell

    # Compute DPT
    print("  Computing DPT...")
    sc.tl.dpt(adata, n_branchings=0)

    return adata.obs["dpt_pseudotime"].values


def create_visualizations(
    cells_df: pd.DataFrame,
    output_dir: Path,
):
    """Create progression score visualizations."""
    print("Creating visualizations...")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    stages = ["Normal", "AAH", "AIS", "MIA", "LUAD"]
    stage_order = {s: i for i, s in enumerate(stages)}

    # Filter to cells with stage info
    if "stage" in cells_df.columns:
        plot_df = cells_df[cells_df["stage"].isin(stages)].copy()
        plot_df["stage_order"] = plot_df["stage"].map(stage_order)
        plot_df = plot_df.sort_values("stage_order")
    else:
        plot_df = cells_df.copy()

    # 1. CytoTRACE by stage (boxplot)
    ax = axes[0, 0]
    if "cytotrace" in plot_df.columns and "stage" in plot_df.columns:
        stage_data = [plot_df[plot_df["stage"] == s]["cytotrace"].values for s in stages if s in plot_df["stage"].values]
        stage_labels = [s for s in stages if s in plot_df["stage"].values]
        ax.boxplot(stage_data, labels=stage_labels)
        ax.set_ylabel("CytoTRACE Score")
        ax.set_title("CytoTRACE by Stage")
        ax.tick_params(axis='x', rotation=45)
    else:
        ax.text(0.5, 0.5, "No CytoTRACE data", ha='center', va='center')
        ax.set_title("CytoTRACE by Stage")

    # 2. Pseudotime by stage (boxplot)
    ax = axes[0, 1]
    if "pseudotime" in plot_df.columns and "stage" in plot_df.columns:
        stage_data = [plot_df[plot_df["stage"] == s]["pseudotime"].dropna().values for s in stages if s in plot_df["stage"].values]
        stage_labels = [s for s in stages if s in plot_df["stage"].values]
        ax.boxplot(stage_data, labels=stage_labels)
        ax.set_ylabel("Pseudotime")
        ax.set_title("Pseudotime by Stage")
        ax.tick_params(axis='x', rotation=45)
    else:
        ax.text(0.5, 0.5, "No pseudotime data", ha='center', va='center')
        ax.set_title("Pseudotime by Stage")

    # 3. CytoTRACE vs Pseudotime scatter
    ax = axes[0, 2]
    if "cytotrace" in plot_df.columns and "pseudotime" in plot_df.columns:
        valid = plot_df["pseudotime"].notna()
        sample = plot_df[valid].sample(min(10000, valid.sum()), random_state=42)
        ax.scatter(sample["pseudotime"], sample["cytotrace"], alpha=0.3, s=1)
        ax.set_xlabel("Pseudotime")
        ax.set_ylabel("CytoTRACE")
        ax.set_title("CytoTRACE vs Pseudotime")
        # Add correlation
        corr = sample[["pseudotime", "cytotrace"]].corr().iloc[0, 1]
        ax.text(0.05, 0.95, f"r = {corr:.3f}", transform=ax.transAxes, va='top')
    else:
        ax.text(0.5, 0.5, "Missing data", ha='center', va='center')
        ax.set_title("CytoTRACE vs Pseudotime")

    # 4. CytoTRACE distribution
    ax = axes[1, 0]
    if "cytotrace" in plot_df.columns:
        ax.hist(plot_df["cytotrace"], bins=50, edgecolor='black', alpha=0.7)
        ax.set_xlabel("CytoTRACE Score")
        ax.set_ylabel("Count")
        ax.set_title("CytoTRACE Distribution")
        ax.axvline(plot_df["cytotrace"].median(), color='red', linestyle='--', label=f'Median: {plot_df["cytotrace"].median():.3f}')
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No CytoTRACE data", ha='center', va='center')
        ax.set_title("CytoTRACE Distribution")

    # 5. Pseudotime distribution
    ax = axes[1, 1]
    if "pseudotime" in plot_df.columns:
        valid_pt = plot_df["pseudotime"].dropna()
        ax.hist(valid_pt, bins=50, edgecolor='black', alpha=0.7)
        ax.set_xlabel("Pseudotime")
        ax.set_ylabel("Count")
        ax.set_title("Pseudotime Distribution")
        ax.axvline(valid_pt.median(), color='red', linestyle='--', label=f'Median: {valid_pt.median():.3f}')
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No pseudotime data", ha='center', va='center')
        ax.set_title("Pseudotime Distribution")

    # 6. Stage progression summary
    ax = axes[1, 2]
    if "stage" in plot_df.columns:
        summary = []
        for s in stages:
            if s in plot_df["stage"].values:
                stage_df = plot_df[plot_df["stage"] == s]
                row = {"stage": s, "n_cells": len(stage_df)}
                if "cytotrace" in stage_df.columns:
                    row["cytotrace_mean"] = stage_df["cytotrace"].mean()
                if "pseudotime" in stage_df.columns:
                    row["pseudotime_mean"] = stage_df["pseudotime"].mean()
                summary.append(row)

        summary_df = pd.DataFrame(summary)

        if "cytotrace_mean" in summary_df.columns and "pseudotime_mean" in summary_df.columns:
            x = range(len(summary_df))
            width = 0.35
            ax.bar([i - width/2 for i in x], summary_df["cytotrace_mean"], width, label="CytoTRACE", alpha=0.8)
            ax.bar([i + width/2 for i in x], summary_df["pseudotime_mean"], width, label="Pseudotime", alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(summary_df["stage"])
            ax.set_ylabel("Mean Score")
            ax.set_title("Mean Scores by Stage")
            ax.legend()
            ax.tick_params(axis='x', rotation=45)
        else:
            ax.text(0.5, 0.5, "Insufficient data", ha='center', va='center')
            ax.set_title("Mean Scores by Stage")
    else:
        ax.text(0.5, 0.5, "No stage data", ha='center', va='center')
        ax.set_title("Mean Scores by Stage")

    plt.tight_layout()
    plt.savefig(output_dir / "progression_scores.png", dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / "progression_scores.pdf", bbox_inches='tight')
    print(f"  Saved: {output_dir / 'progression_scores.png'}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Compute CytoTRACE and pseudotime scores")
    parser.add_argument("--snrna", required=True, help="Path to snRNA h5ad file")
    parser.add_argument("--cells", required=True, help="Path to cells.parquet")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--root_stage", default="Normal", help="Root stage for pseudotime")
    parser.add_argument("--skip_pseudotime", action="store_true", help="Skip pseudotime (slow)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load cells.parquet for metadata
    print(f"Loading cells.parquet: {args.cells}")
    cells_df = pd.read_parquet(args.cells)
    print(f"  {len(cells_df)} cells")

    # Filter to snRNA only (pseudotime needs expression data)
    if "data_type" in cells_df.columns:
        snrna_mask = cells_df["data_type"] == "snRNA"
        snrna_cell_ids = set(cells_df.loc[snrna_mask, "cell_id"])
        print(f"  {len(snrna_cell_ids)} snRNA cells")
    else:
        snrna_cell_ids = set(cells_df["cell_id"])

    # Load snRNA h5ad
    print(f"Loading snRNA: {args.snrna}")
    adata = sc.read_h5ad(args.snrna)
    print(f"  {adata.n_obs} cells, {adata.n_vars} genes")

    # Subset to cells in canonical parquet
    common_cells = [c for c in adata.obs_names if c in snrna_cell_ids]
    print(f"  {len(common_cells)} cells in common with canonical parquet")
    adata = adata[common_cells].copy()

    # Add stage info from cells_df
    cell_to_stage = dict(zip(cells_df["cell_id"], cells_df.get("stage", ["Unknown"] * len(cells_df))))
    adata.obs["stage"] = [cell_to_stage.get(c, "Unknown") for c in adata.obs_names]

    # Compute CytoTRACE
    cytotrace_scores = compute_cytotrace(adata)

    # Compute pseudotime
    if not args.skip_pseudotime:
        try:
            pseudotime_scores = compute_diffusion_pseudotime(adata, root_stage=args.root_stage)
        except Exception as e:
            print(f"  Pseudotime failed: {e}")
            pseudotime_scores = np.full(adata.n_obs, np.nan)
    else:
        print("Skipping pseudotime computation")
        pseudotime_scores = np.full(adata.n_obs, np.nan)

    # Build results DataFrame
    results_df = pd.DataFrame({
        "cell_id": adata.obs_names,
        "cytotrace": cytotrace_scores,
        "pseudotime": pseudotime_scores,
    })

    # Save scores
    results_df.to_parquet(output_dir / "progression_scores.parquet", index=False)
    print(f"Saved: {output_dir / 'progression_scores.parquet'}")

    # Merge with cells_df for visualization
    vis_df = cells_df.merge(results_df, on="cell_id", how="left")

    # Create visualizations
    create_visualizations(vis_df, output_dir)

    # Print summary
    print("\n" + "=" * 60)
    print("Progression Score Summary")
    print("=" * 60)
    print(f"CytoTRACE: min={cytotrace_scores.min():.3f}, max={cytotrace_scores.max():.3f}, median={np.median(cytotrace_scores):.3f}")
    valid_pt = pseudotime_scores[~np.isnan(pseudotime_scores)]
    if len(valid_pt) > 0:
        print(f"Pseudotime: min={valid_pt.min():.3f}, max={valid_pt.max():.3f}, median={np.median(valid_pt):.3f}")

    if "stage" in cells_df.columns:
        print("\nBy Stage:")
        for stage in ["Normal", "AAH", "AIS", "MIA", "LUAD"]:
            mask = vis_df["stage"] == stage
            if mask.any():
                ct = vis_df.loc[mask, "cytotrace"].mean()
                pt = vis_df.loc[mask, "pseudotime"].mean()
                print(f"  {stage}: CytoTRACE={ct:.3f}, Pseudotime={pt:.3f}")

    print("\nTo join with cells.parquet:")
    print(f"  python -c \"")
    print(f"  import pandas as pd")
    print(f"  cells = pd.read_parquet('{args.cells}')")
    print(f"  scores = pd.read_parquet('{output_dir}/progression_scores.parquet')")
    print(f"  merged = cells.merge(scores, on='cell_id', how='left')")
    print(f"  merged.to_parquet('{args.cells}', index=False)")
    print(f"  \"")


if __name__ == "__main__":
    main()
