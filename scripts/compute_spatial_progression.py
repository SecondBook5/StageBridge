#!/usr/bin/env python3
"""
Compute spatial progression scores from deconvolution + snRNA pseudotime.

This script creates spatially-resolved progression maps by:
1. Computing mean CytoTRACE/pseudotime per cell type from snRNA
2. Using deconvolution proportions (gamma) to compute weighted spatial scores
3. Generating spatial heatmaps showing progression gradients across tissue

Usage:
    python scripts/compute_spatial_progression.py \
        --cells $DATA/processed/luad_evo/canonical/cells.parquet \
        --progression $DATA/processed/luad_evo/canonical/progression/progression_scores.parquet \
        --spatial $DATA/processed/luad_evo/spatial_merged.h5ad \
        --output_dir $DATA/processed/luad_evo/canonical/spatial_progression
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import scanpy as scp
import warnings

warnings.filterwarnings("ignore")

# LungPCA color scheme
STAGE_COLORS = {
    'Normal': '#33a02c',
    'AAH': '#b2df8a',
    'AIS': '#fdbf6f',
    'MIA': '#fb9a99',
    'LUAD': '#ff7f00',
}


def compute_celltype_progression_means(
    cells_df: pd.DataFrame,
    progression_df: pd.DataFrame,
) -> tuple[dict, dict]:
    """
    Compute mean CytoTRACE and pseudotime per cell type from snRNA data.

    Returns:
        (cytotrace_means, pseudotime_means) - dicts mapping celltype -> mean score
    """
    print("Computing cell type progression means from snRNA...")

    # Merge progression scores with cell metadata
    merged = cells_df.merge(progression_df, on="cell_id", how="inner")

    # Filter to snRNA only
    if "data_type" in merged.columns:
        merged = merged[merged["data_type"] == "snRNA"]

    # Determine cell type column
    celltype_col = None
    for col in ["hlca_celltype", "luca_celltype", "celltype", "cell_type"]:
        if col in merged.columns:
            celltype_col = col
            break

    if celltype_col is None:
        raise ValueError("No cell type column found in cells.parquet")

    print(f"  Using cell type column: {celltype_col}")

    # Compute means per cell type
    cytotrace_means = merged.groupby(celltype_col)["cytotrace"].mean().to_dict()
    pseudotime_means = merged.groupby(celltype_col)["pseudotime"].mean().to_dict()

    print(f"  Computed means for {len(cytotrace_means)} cell types")

    # Print top/bottom
    sorted_ct = sorted(cytotrace_means.items(), key=lambda x: x[1], reverse=True)
    print("  Top 5 CytoTRACE (most stem-like):")
    for ct, val in sorted_ct[:5]:
        print(f"    {ct}: {val:.3f}")
    print("  Bottom 5 CytoTRACE (most differentiated):")
    for ct, val in sorted_ct[-5:]:
        print(f"    {ct}: {val:.3f}")

    return cytotrace_means, pseudotime_means


def compute_spatial_progression_scores(
    cells_df: pd.DataFrame,
    cytotrace_means: dict,
    pseudotime_means: dict,
    proportions_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Compute weighted progression scores for spatial spots using deconvolution proportions.

    spatial_score = Σ(proportion_celltype × mean_score_celltype)
    """
    print("Computing spatial progression scores from deconvolution...")

    # Filter to spatial data
    if "data_type" in cells_df.columns:
        spatial_df = cells_df[cells_df["data_type"] == "spatial"].copy()
    else:
        # Infer from cell_id pattern
        spatial_df = cells_df[cells_df["cell_id"].str.startswith("spatial_")].copy()

    print(f"  {len(spatial_df)} spatial spots")

    # Get cell type proportions - either from proportions_df or cells_df columns
    if proportions_df is not None:
        # Use provided proportions file (already normalized 0-1)
        prop_cols = [c for c in proportions_df.columns if c not in ["sample", "sample_id", "spot_id", "barcode"]]
        print(f"  Using {len(prop_cols)} cell types from proportions file: {prop_cols}")

        # Match spots - proportions_df index should be spot barcodes
        # spatial_df cell_id is "spatial_{barcode}"
        spatial_df["spot_barcode"] = spatial_df["cell_id"].str.replace("spatial_", "", regex=False)

        # Merge proportions - also get sample column if present
        prop_subset = proportions_df[prop_cols].copy()
        prop_subset.index = prop_subset.index.astype(str)

        # Add sample info from proportions file if available
        if "sample" in proportions_df.columns:
            prop_subset["_prop_sample"] = proportions_df["sample"].values

        matched_spots = spatial_df["spot_barcode"].isin(prop_subset.index)
        print(f"  Matched {matched_spots.sum()} / {len(spatial_df)} spots to proportions")

        if matched_spots.sum() == 0:
            raise ValueError("No spots matched between cells.parquet and proportions file")

        spatial_df = spatial_df[matched_spots].copy()
        matched_props = prop_subset.loc[spatial_df["spot_barcode"].values]
        prop_matrix = matched_props[prop_cols].values
        celltype_names = prop_cols

        # Add sample_id from proportions if not in spatial_df
        if "sample_id" not in spatial_df.columns and "_prop_sample" in matched_props.columns:
            spatial_df["sample_id"] = matched_props["_prop_sample"].values
    else:
        # Fall back to looking for proportion columns in cells_df
        # Try common cell type names
        candidate_cols = ["AT2", "Basal", "Capillary", "Ciliated", "Fibroblast lineage",
                         "Macrophages", "Mast cells", "Secretory", "T cell lineage"]
        prop_cols = [c for c in candidate_cols if c in spatial_df.columns]

        if not prop_cols:
            # Try gamma columns as fallback (will be normalized)
            gamma_cols = sorted([c for c in spatial_df.columns if c.startswith("gamma_")])
            if gamma_cols:
                print(f"  Warning: Using gamma columns (latent factors, not proportions)")
                prop_cols = gamma_cols
                celltype_names = [c.replace("gamma_", "") for c in gamma_cols]
            else:
                raise ValueError("No cell type proportion columns found")
        else:
            celltype_names = prop_cols

        prop_matrix = spatial_df[prop_cols].values.copy()

    # Normalize to proportions (ensure sum to 1)
    prop_matrix = np.clip(prop_matrix, 0, None)
    row_sums = prop_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    prop_matrix = prop_matrix / row_sums

    print(f"  Proportions: min={prop_matrix.min():.3f}, max={prop_matrix.max():.3f}")

    # Compute weighted scores
    spatial_cytotrace = np.zeros(len(spatial_df))
    spatial_pseudotime = np.zeros(len(spatial_df))
    matched_celltypes = []

    for i, celltype in enumerate(celltype_names):
        # Match cell type name (case insensitive, handle variations)
        ct_cytotrace = None
        ct_pseudotime = None

        for key in cytotrace_means.keys():
            if key.lower() == celltype.lower() or celltype.lower() in key.lower():
                ct_cytotrace = cytotrace_means[key]
                ct_pseudotime = pseudotime_means.get(key, np.nan)
                matched_celltypes.append((celltype, key))
                break

        if ct_cytotrace is not None:
            spatial_cytotrace += prop_matrix[:, i] * ct_cytotrace
            if not np.isnan(ct_pseudotime):
                spatial_pseudotime += prop_matrix[:, i] * ct_pseudotime

    print(f"  Matched {len(matched_celltypes)}/{len(celltype_names)} cell types")

    # Build results
    results = pd.DataFrame({
        "cell_id": spatial_df["cell_id"].values,
        "spatial_cytotrace": spatial_cytotrace,
        "spatial_pseudotime": spatial_pseudotime,
    })

    # Add metadata
    for col in ["sample_id", "donor_id", "stage", "x", "y", "array_row", "array_col"]:
        if col in spatial_df.columns:
            results[col] = spatial_df[col].values

    print(f"  Spatial CytoTRACE: min={spatial_cytotrace.min():.3f}, max={spatial_cytotrace.max():.3f}")
    print(f"  Spatial Pseudotime: min={spatial_pseudotime.min():.3f}, max={spatial_pseudotime.max():.3f}")

    return results


def create_spatial_progression_figures(
    spatial_scores: pd.DataFrame,
    spatial_adata: scp.AnnData,
    output_dir: Path,
    n_samples: int = 6,
):
    """
    Create spatial progression heatmap figures.

    Shows tissue sections colored by progression score.
    """
    print("Creating spatial progression figures...")

    # Merge scores with adata obs
    # Handle both "spatial_{barcode}" and raw barcode formats
    score_dict = spatial_scores.set_index("cell_id")[["spatial_cytotrace", "spatial_pseudotime"]].to_dict('index')

    # Also create lookup by raw barcode (without "spatial_" prefix)
    score_dict_barcode = {}
    for cell_id, scores in score_dict.items():
        if cell_id.startswith("spatial_"):
            barcode = cell_id[8:]
            score_dict_barcode[barcode] = scores

    def get_score(obs_name, key):
        # Try full cell_id first, then raw barcode
        if obs_name in score_dict:
            return score_dict[obs_name].get(key, np.nan)
        elif obs_name in score_dict_barcode:
            return score_dict_barcode[obs_name].get(key, np.nan)
        # Try adding "spatial_" prefix
        prefixed = f"spatial_{obs_name}"
        if prefixed in score_dict:
            return score_dict[prefixed].get(key, np.nan)
        return np.nan

    spatial_adata.obs["spatial_cytotrace"] = [
        get_score(c, "spatial_cytotrace") for c in spatial_adata.obs_names
    ]
    spatial_adata.obs["spatial_pseudotime"] = [
        get_score(c, "spatial_pseudotime") for c in spatial_adata.obs_names
    ]

    n_matched = (~spatial_adata.obs["spatial_cytotrace"].isna()).sum()
    print(f"  Matched {n_matched} / {spatial_adata.n_obs} spots to scores")

    # Get unique samples
    sample_col = None
    for col in ["sample_id", "sample", "batch"]:
        if col in spatial_adata.obs.columns:
            sample_col = col
            break

    if sample_col is None:
        print("  Warning: No sample column found, treating as single sample")
        samples = ["all"]
    else:
        samples = spatial_adata.obs[sample_col].unique()
        print(f"  Found {len(samples)} samples")

    # Select diverse samples (by stage if available)
    # Use same column name that was found in adata
    score_sample_col = sample_col if sample_col in spatial_scores.columns else None
    if score_sample_col is None:
        for col in ["sample_id", "sample", "batch"]:
            if col in spatial_scores.columns:
                score_sample_col = col
                break

    if "stage" in spatial_scores.columns and score_sample_col and len(samples) > n_samples:
        # Pick samples from each stage
        selected = []
        for stage in ["Normal", "AAH", "AIS", "MIA", "LUAD"]:
            stage_samples = spatial_scores[spatial_scores["stage"] == stage][score_sample_col].unique()
            if len(stage_samples) > 0:
                selected.append(stage_samples[0])
                if len(selected) >= n_samples:
                    break
        # Fill remaining
        for s in samples:
            if s not in selected and len(selected) < n_samples:
                selected.append(s)
        samples = selected[:n_samples]
    else:
        samples = list(samples)[:n_samples]

    print(f"  Creating figures for {len(samples)} samples")

    # =========================================================================
    # Figure 1: Multi-sample spatial CytoTRACE grid
    # =========================================================================
    n_cols = min(3, len(samples))
    n_rows = (len(samples) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 6*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, sample in enumerate(samples):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]

        if sample_col and sample != "all":
            mask = spatial_adata.obs[sample_col] == sample
            sample_adata = spatial_adata[mask]
        else:
            sample_adata = spatial_adata

        # Get coordinates
        if "spatial" in sample_adata.obsm:
            coords = sample_adata.obsm["spatial"]
        elif "X_spatial" in sample_adata.obsm:
            coords = sample_adata.obsm["X_spatial"]
        else:
            # Try obs columns
            if "x" in sample_adata.obs and "y" in sample_adata.obs:
                coords = sample_adata.obs[["x", "y"]].values
            elif "array_col" in sample_adata.obs and "array_row" in sample_adata.obs:
                coords = sample_adata.obs[["array_col", "array_row"]].values
            else:
                print(f"  Warning: No spatial coordinates for {sample}")
                continue

        scores = sample_adata.obs["spatial_cytotrace"].values

        # Get stage for title
        if "stage" in spatial_scores.columns:
            sample_stage = spatial_scores[spatial_scores["sample_id"] == sample]["stage"].iloc[0] if sample != "all" else "All"
        else:
            sample_stage = ""

        sc = ax.scatter(coords[:, 0], coords[:, 1],
                       c=scores, cmap='turbo', s=8, alpha=0.8, rasterized=True)
        ax.set_title(f"{sample}\n{sample_stage}", fontsize=12, fontweight='bold')
        ax.set_aspect('equal')
        ax.axis('off')
        plt.colorbar(sc, ax=ax, shrink=0.6, label='CytoTRACE')

    # Hide empty subplots
    for idx in range(len(samples), n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].axis('off')

    plt.suptitle("Spatial CytoTRACE (Differentiation Potential)", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "spatial_cytotrace_grid.png", dpi=200, bbox_inches='tight')
    plt.savefig(output_dir / "spatial_cytotrace_grid.pdf", bbox_inches='tight')
    print(f"  Saved: {output_dir / 'spatial_cytotrace_grid.png'}")
    plt.close()

    # =========================================================================
    # Figure 2: Multi-sample spatial Pseudotime grid
    # =========================================================================
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 6*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, sample in enumerate(samples):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]

        if sample_col and sample != "all":
            mask = spatial_adata.obs[sample_col] == sample
            sample_adata = spatial_adata[mask]
        else:
            sample_adata = spatial_adata

        if "spatial" in sample_adata.obsm:
            coords = sample_adata.obsm["spatial"]
        elif "X_spatial" in sample_adata.obsm:
            coords = sample_adata.obsm["X_spatial"]
        else:
            if "x" in sample_adata.obs and "y" in sample_adata.obs:
                coords = sample_adata.obs[["x", "y"]].values
            elif "array_col" in sample_adata.obs and "array_row" in sample_adata.obs:
                coords = sample_adata.obs[["array_col", "array_row"]].values
            else:
                continue

        scores = sample_adata.obs["spatial_pseudotime"].values

        if "stage" in spatial_scores.columns:
            sample_stage = spatial_scores[spatial_scores["sample_id"] == sample]["stage"].iloc[0] if sample != "all" else "All"
        else:
            sample_stage = ""

        sc = ax.scatter(coords[:, 0], coords[:, 1],
                       c=scores, cmap='viridis', s=8, alpha=0.8, rasterized=True)
        ax.set_title(f"{sample}\n{sample_stage}", fontsize=12, fontweight='bold')
        ax.set_aspect('equal')
        ax.axis('off')
        plt.colorbar(sc, ax=ax, shrink=0.6, label='Pseudotime')

    for idx in range(len(samples), n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].axis('off')

    plt.suptitle("Spatial Pseudotime (Progression)", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "spatial_pseudotime_grid.png", dpi=200, bbox_inches='tight')
    plt.savefig(output_dir / "spatial_pseudotime_grid.pdf", bbox_inches='tight')
    print(f"  Saved: {output_dir / 'spatial_pseudotime_grid.png'}")
    plt.close()

    # =========================================================================
    # Figure 3: Stage comparison - mean spatial scores by pathological stage
    # =========================================================================
    if "stage" in spatial_scores.columns:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        stages = ["Normal", "AAH", "AIS", "MIA", "LUAD"]

        # CytoTRACE by stage
        ax = axes[0]
        stage_data = []
        stage_labels = []
        for s in stages:
            vals = spatial_scores[spatial_scores["stage"] == s]["spatial_cytotrace"].dropna()
            if len(vals) > 0:
                stage_data.append(vals)
                stage_labels.append(s)

        bp = ax.boxplot(stage_data, labels=stage_labels, patch_artist=True)
        for patch, label in zip(bp['boxes'], stage_labels):
            patch.set_facecolor(STAGE_COLORS.get(label, '#cccccc'))
            patch.set_alpha(0.7)
        ax.set_ylabel("Spatial CytoTRACE", fontsize=12)
        ax.set_title("Spatial Differentiation by Stage", fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)

        # Pseudotime by stage
        ax = axes[1]
        stage_data = []
        stage_labels = []
        for s in stages:
            vals = spatial_scores[spatial_scores["stage"] == s]["spatial_pseudotime"].dropna()
            if len(vals) > 0:
                stage_data.append(vals)
                stage_labels.append(s)

        bp = ax.boxplot(stage_data, labels=stage_labels, patch_artist=True)
        for patch, label in zip(bp['boxes'], stage_labels):
            patch.set_facecolor(STAGE_COLORS.get(label, '#cccccc'))
            patch.set_alpha(0.7)
        ax.set_ylabel("Spatial Pseudotime", fontsize=12)
        ax.set_title("Spatial Progression by Stage", fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(output_dir / "spatial_progression_by_stage.png", dpi=150, bbox_inches='tight')
        plt.savefig(output_dir / "spatial_progression_by_stage.pdf", bbox_inches='tight')
        print(f"  Saved: {output_dir / 'spatial_progression_by_stage.png'}")
        plt.close()

    # =========================================================================
    # Figure 4: Single sample showcase with all annotations
    # =========================================================================
    if len(samples) > 0:
        # Pick a LUAD sample for showcase if available
        showcase_sample = samples[0]
        for s in samples:
            if "LUAD" in str(spatial_scores[spatial_scores.get("sample_id", "") == s].get("stage", "").values):
                showcase_sample = s
                break

        if sample_col and showcase_sample != "all":
            mask = spatial_adata.obs[sample_col] == showcase_sample
            showcase_adata = spatial_adata[mask]
        else:
            showcase_adata = spatial_adata

        if "spatial" in showcase_adata.obsm:
            coords = showcase_adata.obsm["spatial"]
        elif "X_spatial" in showcase_adata.obsm:
            coords = showcase_adata.obsm["X_spatial"]
        elif "x" in showcase_adata.obs and "y" in showcase_adata.obs:
            coords = showcase_adata.obs[["x", "y"]].values
        elif "array_col" in showcase_adata.obs and "array_row" in showcase_adata.obs:
            coords = showcase_adata.obs[["array_col", "array_row"]].values
        else:
            coords = None

        if coords is not None:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            # Panel A: Stage annotation
            ax = axes[0]
            if "stage" in showcase_adata.obs.columns:
                for stage in ["Normal", "AAH", "AIS", "MIA", "LUAD"]:
                    stage_mask = showcase_adata.obs["stage"] == stage
                    if stage_mask.any():
                        ax.scatter(coords[stage_mask, 0], coords[stage_mask, 1],
                                  c=STAGE_COLORS.get(stage, '#cccccc'), label=stage,
                                  s=12, alpha=0.8, rasterized=True)
                ax.legend(markerscale=3, frameon=False, fontsize=10)
            ax.set_title("Pathological Stage", fontsize=14, fontweight='bold')
            ax.set_aspect('equal')
            ax.axis('off')

            # Panel B: Spatial CytoTRACE
            ax = axes[1]
            scores = showcase_adata.obs["spatial_cytotrace"].values
            sc = ax.scatter(coords[:, 0], coords[:, 1],
                           c=scores, cmap='turbo', s=12, alpha=0.8, rasterized=True)
            plt.colorbar(sc, ax=ax, shrink=0.6, label='CytoTRACE')
            ax.set_title("Spatial CytoTRACE", fontsize=14, fontweight='bold')
            ax.set_aspect('equal')
            ax.axis('off')

            # Panel C: Spatial Pseudotime
            ax = axes[2]
            scores = showcase_adata.obs["spatial_pseudotime"].values
            sc = ax.scatter(coords[:, 0], coords[:, 1],
                           c=scores, cmap='viridis', s=12, alpha=0.8, rasterized=True)
            plt.colorbar(sc, ax=ax, shrink=0.6, label='Pseudotime')
            ax.set_title("Spatial Pseudotime", fontsize=14, fontweight='bold')
            ax.set_aspect('equal')
            ax.axis('off')

            plt.suptitle(f"Sample: {showcase_sample}", fontsize=16, fontweight='bold', y=1.02)
            plt.tight_layout()
            plt.savefig(output_dir / "spatial_showcase_3panel.png", dpi=300, bbox_inches='tight')
            plt.savefig(output_dir / "spatial_showcase_3panel.pdf", bbox_inches='tight')
            print(f"  Saved: {output_dir / 'spatial_showcase_3panel.png'}")
            plt.close()


def main():
    parser = argparse.ArgumentParser(description="Compute spatial progression scores")
    parser.add_argument("--cells", required=True, help="Path to cells.parquet")
    parser.add_argument("--progression", required=True, help="Path to progression_scores.parquet from snRNA")
    parser.add_argument("--spatial", required=True, help="Path to spatial_merged.h5ad")
    parser.add_argument("--proportions", required=False, default=None,
                       help="Path to cell_type_proportions.parquet from deconvolution (recommended)")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--n_samples", type=int, default=6, help="Number of samples for grid figures")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading cells.parquet: {args.cells}")
    cells_df = pd.read_parquet(args.cells)
    print(f"  {len(cells_df)} cells")

    print(f"Loading progression scores: {args.progression}")
    progression_df = pd.read_parquet(args.progression)
    print(f"  {len(progression_df)} snRNA cells with scores")

    print(f"Loading spatial data: {args.spatial}")
    spatial_adata = scp.read_h5ad(args.spatial)
    print(f"  {spatial_adata.n_obs} spatial spots")

    # Load proportions if provided
    proportions_df = None
    if args.proportions:
        print(f"Loading cell type proportions: {args.proportions}")
        proportions_df = pd.read_parquet(args.proportions)
        # Set index to spot barcode if 'sample' column exists
        if "sample" in proportions_df.columns:
            proportions_df = proportions_df.set_index(proportions_df.index.astype(str))
        print(f"  {len(proportions_df)} spots with proportions")

    # Compute cell type means from snRNA
    cytotrace_means, pseudotime_means = compute_celltype_progression_means(cells_df, progression_df)

    # Compute spatial progression scores
    spatial_scores = compute_spatial_progression_scores(
        cells_df, cytotrace_means, pseudotime_means, proportions_df
    )

    # Save scores
    spatial_scores.to_parquet(output_dir / "spatial_progression_scores.parquet", index=False)
    print(f"Saved: {output_dir / 'spatial_progression_scores.parquet'}")

    # Create figures
    create_spatial_progression_figures(spatial_scores, spatial_adata, output_dir, n_samples=args.n_samples)

    # Print summary
    print("\n" + "=" * 60)
    print("Spatial Progression Summary")
    print("=" * 60)
    print(f"Spatial spots: {len(spatial_scores)}")
    print(f"Spatial CytoTRACE range: {spatial_scores['spatial_cytotrace'].min():.3f} - {spatial_scores['spatial_cytotrace'].max():.3f}")
    print(f"Spatial Pseudotime range: {spatial_scores['spatial_pseudotime'].min():.3f} - {spatial_scores['spatial_pseudotime'].max():.3f}")

    if "stage" in spatial_scores.columns:
        print("\nBy Stage:")
        for stage in ["Normal", "AAH", "AIS", "MIA", "LUAD"]:
            mask = spatial_scores["stage"] == stage
            if mask.any():
                ct = spatial_scores.loc[mask, "spatial_cytotrace"].mean()
                pt = spatial_scores.loc[mask, "spatial_pseudotime"].mean()
                n = mask.sum()
                print(f"  {stage} (n={n}): CytoTRACE={ct:.3f}, Pseudotime={pt:.3f}")

    print(f"\nOutput files in: {output_dir}")
    print("  - spatial_progression_scores.parquet")
    print("  - spatial_cytotrace_grid.png/pdf")
    print("  - spatial_pseudotime_grid.png/pdf")
    print("  - spatial_progression_by_stage.png/pdf")
    print("  - spatial_showcase_3panel.png/pdf")


if __name__ == "__main__":
    main()
