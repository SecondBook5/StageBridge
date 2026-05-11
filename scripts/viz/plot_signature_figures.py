#!/usr/bin/env python3
"""Generate snRNA UMAP and spatial figures for biological signatures.

Creates publication-quality figures showing:
1. snRNA UMAPs colored by signature scores
2. Spatial maps of signatures on Visium sections
3. Colocalization analysis of KAC with IL1B_mac niches
"""

import argparse
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# Publication color scheme
STAGE_COLORS = {
    'Normal': '#228B22',  # forest green
    'AAH': '#4682B4',     # steel blue
    'AIS': '#4169E1',     # royal blue
    'MIA': '#8B008B',     # dark magenta
    'LUAD': '#CB4154',    # brick red
}

STAGE_ORDER = ['Normal', 'AAH', 'AIS', 'MIA', 'LUAD']

# Key signatures to visualize
KEY_SIGNATURES = [
    'KAC_score',
    'KAC_extended_score',
    'IL1_axis_score',
    'IL1B_mac_score',
    'stress_like_score',
    'AP1_score',
    'NFkB_score',
    'entropic_score',
    'AT2_score',
]


def load_snrna_with_scores(snrna_path: Path, scores_path: Path) -> sc.AnnData:
    """Load snRNA data and merge signature scores."""
    print(f"Loading snRNA: {snrna_path}")
    adata = sc.read_h5ad(snrna_path)
    print(f"  {adata.n_obs} cells, {adata.n_vars} genes")

    print(f"Loading scores: {scores_path}")
    scores = pd.read_parquet(scores_path)
    print(f"  {len(scores)} cells, {len([c for c in scores.columns if '_score' in c])} signatures")

    # Get score columns
    score_cols = [c for c in scores.columns if c.endswith('_score')]

    # Match by index
    common = adata.obs_names.intersection(scores.index)
    print(f"  {len(common)} cells in common")

    if len(common) == 0:
        print("WARNING: No common cells. Trying to match by position...")
        if len(scores) == adata.n_obs:
            scores.index = adata.obs_names
            common = adata.obs_names

    # Add scores to adata.obs
    for col in score_cols:
        if col in scores.columns:
            adata.obs[col] = scores.loc[adata.obs_names, col].values

    return adata


def score_spatial_spots(spatial_path: Path, signatures: dict) -> sc.AnnData:
    """Score Visium spots directly using signature gene sets."""
    print(f"Loading spatial: {spatial_path}")
    adata = sc.read_h5ad(spatial_path)
    print(f"  {adata.n_obs} spots, {adata.n_vars} genes")

    # Score each signature
    for name, genes in signatures.items():
        present = [g for g in genes if g in adata.var_names]
        if len(present) >= 3:
            sc.tl.score_genes(adata, present, score_name=f'{name}_score', use_raw=False)
            print(f"  {name}: {len(present)}/{len(genes)} genes")
        else:
            print(f"  {name}: skipped ({len(present)}/{len(genes)} genes)")

    return adata


def plot_snrna_umaps(adata: sc.AnnData, output_dir: Path, signatures: list):
    """Plot UMAPs colored by stage and signatures."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check for UMAP
    if 'X_umap' not in adata.obsm:
        print("Computing UMAP...")
        sc.pp.neighbors(adata, n_neighbors=15)
        sc.tl.umap(adata)

    # 1. UMAP by stage
    if 'stage' in adata.obs.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        for stage in STAGE_ORDER:
            mask = adata.obs['stage'] == stage
            if mask.sum() > 0:
                ax.scatter(
                    adata.obsm['X_umap'][mask, 0],
                    adata.obsm['X_umap'][mask, 1],
                    c=STAGE_COLORS.get(stage, 'gray'),
                    s=1, alpha=0.5, label=stage, rasterized=True
                )
        ax.legend(markerscale=5, frameon=False)
        ax.set_xlabel('UMAP1')
        ax.set_ylabel('UMAP2')
        ax.set_title('snRNA by Stage')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.savefig(output_dir / 'umap_stage.png', dpi=150, bbox_inches='tight')
        fig.savefig(output_dir / 'umap_stage.pdf', bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved umap_stage.png/pdf")

    # 2. UMAPs by signature score
    for sig in signatures:
        if sig not in adata.obs.columns:
            continue

        fig, ax = plt.subplots(figsize=(8, 6))
        vals = adata.obs[sig].values
        vmin, vmax = np.percentile(vals[~np.isnan(vals)], [2, 98])

        scatter = ax.scatter(
            adata.obsm['X_umap'][:, 0],
            adata.obsm['X_umap'][:, 1],
            c=vals, cmap='viridis', s=1, alpha=0.7,
            vmin=vmin, vmax=vmax, rasterized=True
        )
        plt.colorbar(scatter, ax=ax, label=sig, shrink=0.6)
        ax.set_xlabel('UMAP1')
        ax.set_ylabel('UMAP2')
        ax.set_title(sig.replace('_score', '').replace('_', ' ').title())
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        fname = sig.replace('_score', '')
        fig.savefig(output_dir / f'umap_{fname}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved umap_{fname}.png")


def plot_spatial_signatures(adata: sc.AnnData, output_dir: Path, signatures: list, n_samples: int = 4):
    """Plot spatial maps of signatures on tissue sections."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get sample column
    sample_col = None
    for col in ['sample_id', 'sample', 'library_id']:
        if col in adata.obs.columns:
            sample_col = col
            break

    if sample_col is None:
        print("WARNING: No sample column found, plotting all spots together")
        samples = ['all']
    else:
        samples = adata.obs[sample_col].unique()[:n_samples]

    for sig in signatures:
        if sig not in adata.obs.columns:
            continue

        fig, axes = plt.subplots(1, len(samples), figsize=(4*len(samples), 4))
        if len(samples) == 1:
            axes = [axes]

        for ax, sample in zip(axes, samples):
            if sample == 'all':
                mask = np.ones(adata.n_obs, dtype=bool)
                title = 'All samples'
            else:
                mask = adata.obs[sample_col] == sample
                title = str(sample)

            subset = adata[mask]

            # Get spatial coordinates
            if 'spatial' in subset.obsm:
                coords = subset.obsm['spatial']
            elif 'X_spatial' in subset.obsm:
                coords = subset.obsm['X_spatial']
            else:
                print(f"  No spatial coords for {sample}")
                continue

            vals = subset.obs[sig].values
            vmin, vmax = np.percentile(vals[~np.isnan(vals)], [2, 98])

            scatter = ax.scatter(
                coords[:, 0], coords[:, 1],
                c=vals, cmap='viridis', s=10, alpha=0.8,
                vmin=vmin, vmax=vmax, rasterized=True
            )
            ax.set_aspect('equal')
            ax.set_title(f'{title}\n{sig.replace("_score", "")}', fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.invert_yaxis()

        plt.colorbar(scatter, ax=axes[-1], shrink=0.6)
        fname = sig.replace('_score', '')
        fig.savefig(output_dir / f'spatial_{fname}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved spatial_{fname}.png")


def plot_signature_by_stage(scores_path: Path, output_dir: Path):
    """Plot signature scores by stage as violin plots with colored boxes."""
    output_dir.mkdir(parents=True, exist_ok=True)

    scores = pd.read_parquet(scores_path)
    if 'stage' not in scores.columns:
        print("No stage column in scores, skipping violin plots")
        return

    # Filter to key signatures
    sig_cols = [c for c in KEY_SIGNATURES if c in scores.columns]

    # Create figure grid
    n_sigs = len(sig_cols)
    n_cols = 3
    n_rows = (n_sigs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3.5*n_rows))
    axes = axes.flatten()

    for i, sig in enumerate(sig_cols):
        ax = axes[i]

        # Prepare data in stage order
        plot_data = []
        positions = []
        colors = []
        for j, stage in enumerate(STAGE_ORDER):
            if stage in scores['stage'].values:
                vals = scores[scores['stage'] == stage][sig].dropna().values
                plot_data.append(vals)
                positions.append(j)
                colors.append(STAGE_COLORS.get(stage, 'gray'))

        # Violin plot
        parts = ax.violinplot(plot_data, positions=positions, showmedians=False, showextrema=False)
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)

        # Add boxplots inside
        bp = ax.boxplot(plot_data, positions=positions, widths=0.15,
                        patch_artist=True, showfliers=False)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.9)
        for element in ['whiskers', 'caps', 'medians']:
            plt.setp(bp[element], color='black', linewidth=1)

        ax.set_xticks(range(len(STAGE_ORDER)))
        ax.set_xticklabels(STAGE_ORDER, rotation=45, ha='right')
        ax.set_title(sig.replace('_score', '').replace('_', ' ').title())
        ax.set_ylabel('Score')

        # Remove top/right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Hide unused axes
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_dir / 'signatures_by_stage.png', dpi=150, bbox_inches='tight')
    fig.savefig(output_dir / 'signatures_by_stage.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved signatures_by_stage.png/pdf")


def main():
    parser = argparse.ArgumentParser(description='Plot signature figures')
    parser.add_argument('--snrna', type=Path, help='Path to snRNA h5ad')
    parser.add_argument('--spatial', type=Path, help='Path to spatial h5ad')
    parser.add_argument('--scores', type=Path, required=True, help='Path to caf_kac_scores.parquet')
    parser.add_argument('--output_dir', type=Path, required=True, help='Output directory')
    parser.add_argument('--skip_umap', action='store_true', help='Skip UMAP plots')
    parser.add_argument('--skip_spatial', action='store_true', help='Skip spatial plots')
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Define signatures for direct scoring
    SIGNATURES = {
        'KAC': ['KRT8', 'CLDN4', 'CDKN1A', 'CDKN2A', 'PLAUR'],
        'KAC_extended': ['KRT8', 'CLDN4', 'LGALS3', 'AREG', 'CLDN7', 'KRT18', 'SFN', 'TACSTD2'],
        'IL1_axis': ['IL1B', 'IL1R1', 'IL1R2', 'IL1RAP', 'IL1RN'],
        'IL1B_mac': ['IL1B', 'CCL2', 'IL18', 'NFKB1', 'CSF1', 'TNF', 'IL6'],
        'stress_like': ['FOS', 'JUN', 'FOSB', 'ATF3', 'HSPA5', 'DNAJB9', 'SQSTM1', 'TXNIP'],
        'AP1': ['FOS', 'FOSB', 'FOSL1', 'FOSL2', 'JUN', 'JUNB', 'JUND'],
        'NFkB': ['RELA', 'RELB', 'NFKB1', 'NFKB2', 'NFKBIA', 'NFKBIZ'],
        'entropic': ['TOP2A', 'MKI67', 'PCNA', 'CDK1', 'CCNB1', 'UBE2C'],
        'AT2': ['SFTPC', 'SFTPA1', 'SFTPA2', 'SFTPB', 'ABCA3', 'LAMP3', 'NKX2-1'],
    }

    # 1. Plot signature by stage (from pre-computed scores)
    print("\n=== Signature by stage plots ===")
    plot_signature_by_stage(args.scores, args.output_dir)

    # 2. snRNA UMAP plots
    if args.snrna and not args.skip_umap:
        print("\n=== snRNA UMAP plots ===")
        adata = load_snrna_with_scores(args.snrna, args.scores)
        plot_snrna_umaps(adata, args.output_dir / 'snrna', KEY_SIGNATURES)

    # 3. Spatial plots
    if args.spatial and not args.skip_spatial:
        print("\n=== Spatial plots ===")
        spatial = score_spatial_spots(args.spatial, SIGNATURES)
        plot_spatial_signatures(spatial, args.output_dir / 'spatial',
                               [f'{k}_score' for k in SIGNATURES.keys()])

    print("\nDone!")


if __name__ == '__main__':
    main()
