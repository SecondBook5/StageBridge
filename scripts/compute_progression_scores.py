#!/usr/bin/env python3
"""
Compute CytoTRACE and pseudotime progression scores for StageBridge.

Generates LungPCA paper-style UMAP visualizations (Figure 3B-D style):
1. UMAP colored by cell type
2. UMAP colored by pseudotime with trajectory
3. UMAP colored by CytoTRACE (turbo colormap)

Usage:
    python scripts/compute_progression_scores.py \
        --snrna $DATA/processed/luad_evo/snrna_with_celltypes.h5ad \
        --cells $DATA/processed/luad_evo/canonical/cells.parquet \
        --output_dir $DATA/processed/luad_evo/canonical/progression
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import scanpy as sc
import warnings

warnings.filterwarnings("ignore")

# LungPCA paper color scheme
STAGE_COLORS = {
    'Normal': '#33a02c',
    'AAH': '#b2df8a',
    'AIS': '#fdbf6f',
    'MIA': '#fb9a99',
    'LUAD': '#ff7f00',
}

CELLTYPE_COLORS = {
    'Basal': '#1f77b4',
    'AT1': '#ff7f0e',
    'AT2': '#e377c2',
    'KAC': '#2ca02c',
    'Ciliated': '#ffbb78',
    'Club': '#ff9896',
    'Club.secretory': '#9467bd',
    'AIC': '#d62728',
    'Tumor': '#f7b6d2',
    # Additional cell types
    'Epithelial': '#1f78b4',
    'Fibroblast': '#cab2d6',
    'Myeloid': '#6a3d9a',
    'Lymphoid': '#ffff99',
    'Endothelial': '#b15928',
}


def compute_cytotrace(adata: sc.AnnData) -> np.ndarray:
    """
    Compute CytoTRACE-like score based on gene count correlation.

    CytoTRACE principle: more differentiated cells express fewer genes.
    Higher score = more stem-like/less differentiated.
    """
    print("Computing CytoTRACE scores...")

    # Gene counts per cell
    if "n_genes" not in adata.obs.columns:
        adata.obs["n_genes"] = (adata.X > 0).sum(axis=1).A1 if hasattr(adata.X, 'A1') else (adata.X > 0).sum(axis=1)

    gene_counts = adata.obs["n_genes"].values.astype(float)

    # Normalize to 0-1
    gc_min, gc_max = gene_counts.min(), gene_counts.max()
    if gc_max > gc_min:
        cytotrace_score = (gene_counts - gc_min) / (gc_max - gc_min)
    else:
        cytotrace_score = np.zeros_like(gene_counts)

    return cytotrace_score


def compute_diffusion_pseudotime(
    adata: sc.AnnData,
    root_stage: str = "Normal",
    n_neighbors: int = 30,
    n_pcs: int = 50,
) -> np.ndarray:
    """
    Compute diffusion pseudotime rooted at Normal epithelium.
    """
    print(f"Computing diffusion pseudotime (root: {root_stage})...")

    adata = adata.copy()

    # PCA
    if "X_pca" not in adata.obsm:
        print("  Computing PCA...")
        sc.pp.pca(adata, n_comps=min(n_pcs, adata.n_vars - 1))

    # Neighbors
    print("  Computing neighbors...")
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep="X_pca")

    # Diffusion map
    print("  Computing diffusion map...")
    sc.tl.diffmap(adata, n_comps=15)

    # Find root cell
    stage_col = "stage" if "stage" in adata.obs.columns else None
    if stage_col and root_stage in adata.obs[stage_col].values:
        root_mask = adata.obs[stage_col] == root_stage
        root_indices = np.where(root_mask)[0]
        diffmap = adata.obsm["X_diffmap"]
        normal_diffmap = diffmap[root_mask]
        centroid = normal_diffmap.mean(axis=0)
        distances = np.linalg.norm(normal_diffmap - centroid, axis=1)
        root_cell_local = np.argmin(distances)
        root_cell = root_indices[root_cell_local]
    else:
        print(f"  Warning: '{root_stage}' not found, using first cell as root")
        root_cell = 0

    adata.uns["iroot"] = root_cell

    # DPT
    print("  Computing DPT...")
    sc.tl.dpt(adata, n_branchings=0)

    return adata.obs["dpt_pseudotime"].values


def create_lungpca_style_figures(
    adata: sc.AnnData,
    cytotrace: np.ndarray,
    pseudotime: np.ndarray,
    output_dir: Path,
):
    """
    Create LungPCA paper-style UMAP visualizations (Figure 3B-D).

    Three-panel figure:
    - Left: UMAP by cell type / stage
    - Middle: UMAP by pseudotime
    - Right: UMAP by CytoTRACE (turbo colormap)
    """
    print("Creating LungPCA-style UMAP figures...")

    # Ensure UMAP exists
    if "X_umap" not in adata.obsm:
        print("  Computing UMAP...")
        if "neighbors" not in adata.uns:
            sc.pp.neighbors(adata, n_neighbors=30)
        sc.tl.umap(adata)

    umap = adata.obsm["X_umap"]

    # Add scores to adata
    adata.obs["CytoTRACE"] = cytotrace
    adata.obs["Pseudotime"] = pseudotime

    # Subsample for plotting if too large
    max_cells = 50000
    if adata.n_obs > max_cells:
        print(f"  Subsampling to {max_cells} cells for visualization...")
        idx = np.random.choice(adata.n_obs, max_cells, replace=False)
        umap_plot = umap[idx]
        cytotrace_plot = cytotrace[idx]
        pseudotime_plot = pseudotime[idx]
        stage_plot = adata.obs["stage"].values[idx] if "stage" in adata.obs else None
        celltype_plot = adata.obs["celltype"].values[idx] if "celltype" in adata.obs else None
    else:
        umap_plot = umap
        cytotrace_plot = cytotrace
        pseudotime_plot = pseudotime
        stage_plot = adata.obs["stage"].values if "stage" in adata.obs else None
        celltype_plot = adata.obs["celltype"].values if "celltype" in adata.obs else None

    # =========================================================================
    # Figure 1: Three-panel LungPCA style (Stage, Pseudotime, CytoTRACE)
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel A: UMAP by stage
    ax = axes[0]
    if stage_plot is not None:
        unique_stages = list(set(stage_plot))
        colors = [STAGE_COLORS.get(s, '#cccccc') for s in unique_stages]
        for i, stage in enumerate(unique_stages):
            mask = stage_plot == stage
            ax.scatter(umap_plot[mask, 0], umap_plot[mask, 1],
                      c=colors[i], label=stage, s=0.5, alpha=0.6, rasterized=True)
        ax.legend(markerscale=5, frameon=False, fontsize=10)
    else:
        ax.scatter(umap_plot[:, 0], umap_plot[:, 1], c='gray', s=0.5, alpha=0.6, rasterized=True)
    ax.set_xlabel("UMAP1", fontsize=12)
    ax.set_ylabel("UMAP2", fontsize=12)
    ax.set_title("Stage", fontsize=14, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel B: UMAP by pseudotime
    ax = axes[1]
    valid_pt = ~np.isnan(pseudotime_plot)
    sc = ax.scatter(umap_plot[valid_pt, 0], umap_plot[valid_pt, 1],
                    c=pseudotime_plot[valid_pt], cmap='viridis', s=0.5, alpha=0.6, rasterized=True)
    plt.colorbar(sc, ax=ax, label='Pseudotime', shrink=0.6)
    ax.set_xlabel("UMAP1", fontsize=12)
    ax.set_ylabel("UMAP2", fontsize=12)
    ax.set_title("Pseudotime", fontsize=14, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel C: UMAP by CytoTRACE (turbo colormap like LungPCA paper)
    ax = axes[2]
    sc = ax.scatter(umap_plot[:, 0], umap_plot[:, 1],
                    c=cytotrace_plot, cmap='turbo', s=0.5, alpha=0.6, rasterized=True)
    plt.colorbar(sc, ax=ax, label='CytoTRACE', shrink=0.6)
    ax.set_xlabel("UMAP1", fontsize=12)
    ax.set_ylabel("UMAP2", fontsize=12)
    ax.set_title("CytoTRACE", fontsize=14, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / "umap_progression_3panel.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "umap_progression_3panel.pdf", bbox_inches='tight')
    print(f"  Saved: {output_dir / 'umap_progression_3panel.png'}")
    plt.close()

    # =========================================================================
    # Figure 2: Individual high-res UMAP panels
    # =========================================================================

    # Stage UMAP
    fig, ax = plt.subplots(figsize=(8, 8))
    if stage_plot is not None:
        for stage in ["Normal", "AAH", "AIS", "MIA", "LUAD"]:
            if stage in stage_plot:
                mask = stage_plot == stage
                ax.scatter(umap_plot[mask, 0], umap_plot[mask, 1],
                          c=STAGE_COLORS.get(stage, '#cccccc'), label=stage,
                          s=1, alpha=0.7, rasterized=True)
        ax.legend(markerscale=8, frameon=False, fontsize=12, loc='upper right')
    ax.set_xlabel("UMAP1", fontsize=14)
    ax.set_ylabel("UMAP2", fontsize=14)
    ax.set_title("LUAD Progression Stages", fontsize=16, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout()
    plt.savefig(output_dir / "umap_stage.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "umap_stage.pdf", bbox_inches='tight')
    plt.close()

    # CytoTRACE UMAP (turbo colormap)
    fig, ax = plt.subplots(figsize=(8, 8))
    sc = ax.scatter(umap_plot[:, 0], umap_plot[:, 1],
                    c=cytotrace_plot, cmap='turbo', s=1, alpha=0.7, rasterized=True)
    cbar = plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label('CytoTRACE Score', fontsize=12)
    ax.set_xlabel("UMAP1", fontsize=14)
    ax.set_ylabel("UMAP2", fontsize=14)
    ax.set_title("CytoTRACE (Differentiation Potential)", fontsize=16, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout()
    plt.savefig(output_dir / "umap_cytotrace.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "umap_cytotrace.pdf", bbox_inches='tight')
    plt.close()

    # Pseudotime UMAP
    fig, ax = plt.subplots(figsize=(8, 8))
    valid_pt = ~np.isnan(pseudotime_plot)
    sc = ax.scatter(umap_plot[valid_pt, 0], umap_plot[valid_pt, 1],
                    c=pseudotime_plot[valid_pt], cmap='viridis', s=1, alpha=0.7, rasterized=True)
    cbar = plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label('Pseudotime', fontsize=12)
    ax.set_xlabel("UMAP1", fontsize=14)
    ax.set_ylabel("UMAP2", fontsize=14)
    ax.set_title("Diffusion Pseudotime", fontsize=16, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout()
    plt.savefig(output_dir / "umap_pseudotime.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "umap_pseudotime.pdf", bbox_inches='tight')
    plt.close()

    # =========================================================================
    # Figure 3: Summary statistics by stage
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    stages = ["Normal", "AAH", "AIS", "MIA", "LUAD"]
    stage_order = {s: i for i, s in enumerate(stages)}

    if stage_plot is not None:
        # CytoTRACE by stage boxplot
        ax = axes[0]
        stage_data = []
        stage_labels = []
        for s in stages:
            if s in stage_plot:
                stage_data.append(cytotrace_plot[stage_plot == s])
                stage_labels.append(s)
        bp = ax.boxplot(stage_data, labels=stage_labels, patch_artist=True)
        for patch, label in zip(bp['boxes'], stage_labels):
            patch.set_facecolor(STAGE_COLORS.get(label, '#cccccc'))
            patch.set_alpha(0.7)
        ax.set_ylabel("CytoTRACE Score", fontsize=12)
        ax.set_title("CytoTRACE by Stage", fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)

        # Pseudotime by stage boxplot
        ax = axes[1]
        stage_data = []
        stage_labels = []
        for s in stages:
            if s in stage_plot:
                pt_vals = pseudotime_plot[stage_plot == s]
                pt_vals = pt_vals[~np.isnan(pt_vals)]
                if len(pt_vals) > 0:
                    stage_data.append(pt_vals)
                    stage_labels.append(s)
        bp = ax.boxplot(stage_data, labels=stage_labels, patch_artist=True)
        for patch, label in zip(bp['boxes'], stage_labels):
            patch.set_facecolor(STAGE_COLORS.get(label, '#cccccc'))
            patch.set_alpha(0.7)
        ax.set_ylabel("Pseudotime", fontsize=12)
        ax.set_title("Pseudotime by Stage", fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)

        # Mean scores by stage
        ax = axes[2]
        means_ct = [cytotrace_plot[stage_plot == s].mean() for s in stages if s in stage_plot]
        means_pt = [np.nanmean(pseudotime_plot[stage_plot == s]) for s in stages if s in stage_plot]
        labels = [s for s in stages if s in stage_plot]
        x = np.arange(len(labels))
        width = 0.35
        ax.bar(x - width/2, means_ct, width, label='CytoTRACE',
               color=[STAGE_COLORS.get(s, '#cccccc') for s in labels], alpha=0.7, edgecolor='black')
        ax.bar(x + width/2, means_pt, width, label='Pseudotime',
               color=[STAGE_COLORS.get(s, '#cccccc') for s in labels], alpha=0.4, edgecolor='black', hatch='//')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45)
        ax.set_ylabel("Mean Score", fontsize=12)
        ax.set_title("Mean Progression Scores", fontsize=14, fontweight='bold')
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "progression_summary.png", dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / "progression_summary.pdf", bbox_inches='tight')
    print(f"  Saved: {output_dir / 'progression_summary.png'}")
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

    # Filter to snRNA only
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

    # Add metadata from cells_df
    cell_meta = cells_df.set_index("cell_id")
    for col in ["stage", "donor_id", "celltype", "hlca_celltype", "luca_celltype"]:
        if col in cell_meta.columns:
            adata.obs[col] = cell_meta.loc[adata.obs_names, col].values

    # Use HLCA celltype if celltype not available
    if "celltype" not in adata.obs.columns and "hlca_celltype" in adata.obs.columns:
        adata.obs["celltype"] = adata.obs["hlca_celltype"]

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

    # Create LungPCA-style visualizations
    create_lungpca_style_figures(adata, cytotrace_scores, pseudotime_scores, output_dir)

    # Print summary
    print("\n" + "=" * 60)
    print("Progression Score Summary")
    print("=" * 60)
    print(f"CytoTRACE: min={cytotrace_scores.min():.3f}, max={cytotrace_scores.max():.3f}, median={np.median(cytotrace_scores):.3f}")
    valid_pt = pseudotime_scores[~np.isnan(pseudotime_scores)]
    if len(valid_pt) > 0:
        print(f"Pseudotime: min={valid_pt.min():.3f}, max={valid_pt.max():.3f}, median={np.median(valid_pt):.3f}")

    if "stage" in adata.obs.columns:
        print("\nBy Stage:")
        for stage in ["Normal", "AAH", "AIS", "MIA", "LUAD"]:
            mask = adata.obs["stage"] == stage
            if mask.any():
                ct = cytotrace_scores[mask].mean()
                pt = np.nanmean(pseudotime_scores[mask])
                print(f"  {stage}: CytoTRACE={ct:.3f}, Pseudotime={pt:.3f}")

    print(f"\nOutput files in: {output_dir}")
    print("  - progression_scores.parquet")
    print("  - umap_progression_3panel.png/pdf (LungPCA Figure 3B-D style)")
    print("  - umap_stage.png/pdf")
    print("  - umap_cytotrace.png/pdf")
    print("  - umap_pseudotime.png/pdf")
    print("  - progression_summary.png/pdf")


if __name__ == "__main__":
    main()
