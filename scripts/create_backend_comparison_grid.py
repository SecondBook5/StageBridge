#!/usr/bin/env python3
"""
Create TRUE multi-backend comparison figures.

Shows the same cell types across all backends on the same spatial coordinates.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm


def load_backend_proportions(
    results_dir: Path,
    sample_id: str,
    label_source: str = "hlca",
) -> dict[str, pd.DataFrame]:
    """Load cell type proportions from all backends for a sample."""
    base = results_dir / label_source
    proportions = {}

    for backend_dir in base.iterdir():
        if not backend_dir.is_dir():
            continue
        backend = backend_dir.name

        props_path = backend_dir / "samples" / sample_id / "cell_type_proportions.parquet"
        if props_path.exists():
            df = pd.read_parquet(props_path)
            # Standardize cell2location column names
            df.columns = [c.replace("q05cell_abundance_w_sf_means_per_cluster_mu_fg_", "") for c in df.columns]
            proportions[backend] = df

    return proportions


def load_spatial_coords(
    results_dir: Path,
    sample_id: str,
    label_source: str = "hlca",
) -> tuple[np.ndarray, pd.Index] | None:
    """Load spatial coordinates and index from a consistent backend.

    Uses destvi as primary source (canonical for StageBridge), with tangram
    as fallback. This ensures all backends use the same spot indices for
    alignment in comparison figures.
    """
    base = results_dir / label_source

    # Prioritize marker_scoring (baseline reference) for consistent coords
    # All other backends should align to marker_scoring indices
    preferred_backends = ["marker_scoring", "destvi", "tangram", "tacco", "cell2location"]

    for backend in preferred_backends:
        backend_dir = base / backend
        if not backend_dir.is_dir():
            continue

        sample_dir = backend_dir / "samples" / sample_id
        for h5ad_file in sample_dir.glob("*_spatial_annotated.h5ad"):
            try:
                adata = ad.read_h5ad(h5ad_file)
                if "spatial" in adata.obsm:
                    return adata.obsm["spatial"], adata.obs_names
            except Exception:
                continue

    # Fallback: try any backend
    for backend_dir in base.iterdir():
        if not backend_dir.is_dir():
            continue

        sample_dir = backend_dir / "samples" / sample_id
        for h5ad_file in sample_dir.glob("*_spatial_annotated.h5ad"):
            try:
                adata = ad.read_h5ad(h5ad_file)
                if "spatial" in adata.obsm:
                    return adata.obsm["spatial"], adata.obs_names
            except Exception:
                continue

    return None


def get_common_celltypes(proportions: dict[str, pd.DataFrame]) -> list[str]:
    """Get cell types present in all backends."""
    if not proportions:
        return []

    common = set(list(proportions.values())[0].columns)
    for df in proportions.values():
        common &= set(df.columns)

    return sorted(list(common))


def create_backend_comparison_figure(
    proportions: dict[str, pd.DataFrame],
    spatial_coords: np.ndarray,
    spatial_index: pd.Index,
    cell_types: list[str],
    sample_id: str,
    output_path: Path,
    max_celltypes: int = 6,
    spot_size: float = 8,
):
    """
    Create publication-quality multi-backend comparison figure.

    Rows = cell types, Columns = backends
    Marker scoring is separated as a baseline method at the end.
    """
    # Parse sample_id for metadata (e.g., GSM9226168_P1_AAH -> P1, AAH)
    parts = sample_id.split('_')
    donor = parts[1] if len(parts) > 1 else ""
    stage = parts[2] if len(parts) > 2 else ""

    # Stage colors for annotation
    STAGE_COLORS = {
        "Normal": "#2ecc71",
        "AAH": "#f39c12",
        "AIS": "#e74c3c",
        "MIA": "#9b59b6",
        "LUAD": "#1a1a2e",
    }
    stage_color = STAGE_COLORS.get(stage.split('-')[0], "#333333")

    # Separate deconvolution methods from baselines
    BASELINE_BACKENDS = {"marker_scoring"}

    all_backends = set(proportions.keys())
    deconv_backends = sorted(all_backends - BASELINE_BACKENDS)
    baseline_backends = sorted(all_backends & BASELINE_BACKENDS)

    # Order: deconvolution methods first, then baselines at the end
    backends = deconv_backends + baseline_backends
    n_backends = len(backends)
    baseline_start_idx = len(deconv_backends)

    # Limit cell types shown
    cell_types = cell_types[:max_celltypes]
    n_celltypes = len(cell_types)

    if n_celltypes == 0 or n_backends == 0:
        print(f"  Skipping {sample_id}: no common cell types or backends")
        return

    # Create spatial coords DataFrame for alignment
    spatial_df = pd.DataFrame(spatial_coords, index=spatial_index, columns=["x", "y"])

    # Publication figure setup
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 10,
        'axes.linewidth': 0.8,
    })

    # Create figure with space for colorbars
    fig, axes = plt.subplots(
        n_celltypes, n_backends + 1,  # +1 for colorbar column
        figsize=(2.5 * n_backends + 0.8, 2.5 * n_celltypes + 0.5),
        squeeze=False,
        gridspec_kw={'width_ratios': [1] * n_backends + [0.05], 'wspace': 0.05, 'hspace': 0.15}
    )

    # Color normalization per cell type (shared across backends)
    for row, ct in enumerate(cell_types):
        all_vals = []
        for backend in backends:
            if ct in proportions[backend].columns:
                all_vals.extend(proportions[backend][ct].values)

        if not all_vals:
            continue

        vmin = np.percentile(all_vals, 2)
        vmax = np.percentile(all_vals, 98)
        norm = Normalize(vmin=vmin, vmax=vmax)

        scatter_ref = None
        for col, backend in enumerate(backends):
            ax = axes[row, col]

            if ct in proportions[backend].columns:
                props_df = proportions[backend]
                common_idx = props_df.index.intersection(spatial_df.index)

                if len(common_idx) == 0:
                    ax.text(0.5, 0.5, "No\noverlap", ha='center', va='center',
                            transform=ax.transAxes, fontsize=9, color='#999999')
                    ax.set_facecolor('#f8f8f8')
                else:
                    values = props_df.loc[common_idx, ct].values
                    coords = spatial_df.loc[common_idx, ["x", "y"]].values

                    scatter = ax.scatter(
                        coords[:, 0],
                        coords[:, 1],
                        c=values,
                        cmap="magma",
                        norm=norm,
                        s=spot_size,
                        edgecolors='none',
                        rasterized=True,
                        alpha=0.9,
                    )
                    if scatter_ref is None:
                        scatter_ref = scatter
                    ax.set_facecolor('black')
            else:
                ax.text(0.5, 0.5, "N/A", ha='center', va='center',
                        transform=ax.transAxes, fontsize=9, color='#999999')
                ax.set_facecolor('#f8f8f8')

            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])

            # Clean spines
            for spine in ax.spines.values():
                spine.set_visible(False)

            # Column header (backend name) - only on first row
            if row == 0:
                is_baseline = backend in BASELINE_BACKENDS
                title_text = backend.replace('_', ' ').title()
                if is_baseline:
                    title_text += "\n(Baseline)"
                ax.set_title(title_text, fontsize=9, fontweight='bold',
                            color='#666666' if is_baseline else '#333333', pad=4)

            # Row label (cell type) - only on first column
            if col == 0:
                ax.set_ylabel(ct.replace('_', ' '), fontsize=9, fontweight='bold',
                             labelpad=2, color='#333333')

        # Add colorbar in the last column for this row
        cax = axes[row, -1]
        if scatter_ref is not None:
            cbar = plt.colorbar(scatter_ref, cax=cax)
            cbar.ax.tick_params(labelsize=7, length=2, width=0.5)
            cbar.outline.set_linewidth(0.5)
        else:
            cax.axis('off')

    # Overall title with stage color accent
    title_text = f"{donor} - {stage}" if donor and stage else sample_id
    fig.suptitle(
        title_text,
        fontsize=12,
        fontweight='bold',
        color=stage_color,
        y=0.98,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white',
                edgecolor='none', pad_inches=0.1)
    plt.close()

    print(f"  Saved: {output_path}")


def create_correlation_heatmap(
    proportions: dict[str, pd.DataFrame],
    cell_types: list[str],
    sample_id: str,
    output_path: Path,
):
    """Create publication-quality correlation heatmap between backends."""
    # Parse sample_id for metadata
    parts = sample_id.split('_')
    donor = parts[1] if len(parts) > 1 else ""
    stage = parts[2] if len(parts) > 2 else ""

    STAGE_COLORS = {
        "Normal": "#2ecc71", "AAH": "#f39c12", "AIS": "#e74c3c",
        "MIA": "#9b59b6", "LUAD": "#1a1a2e",
    }
    stage_color = STAGE_COLORS.get(stage.split('-')[0], "#333333")

    # Separate deconvolution methods from baselines
    BASELINE_BACKENDS = {"marker_scoring"}

    all_backends = set(proportions.keys())
    deconv_backends = sorted(all_backends - BASELINE_BACKENDS)
    baseline_backends = sorted(all_backends & BASELINE_BACKENDS)
    backends = deconv_backends + baseline_backends
    n_backends = len(backends)
    baseline_start_idx = len(deconv_backends)

    if n_backends < 2:
        return

    # Compute pairwise correlations
    corr_matrix = np.zeros((n_backends, n_backends))

    for i, b1 in enumerate(backends):
        for j, b2 in enumerate(backends):
            common_idx = proportions[b1].index.intersection(proportions[b2].index)
            common_cols = [c for c in cell_types if c in proportions[b1].columns and c in proportions[b2].columns]

            if len(common_idx) > 0 and len(common_cols) > 0:
                v1 = proportions[b1].loc[common_idx, common_cols].values.flatten()
                v2 = proportions[b2].loc[common_idx, common_cols].values.flatten()
                corr = np.corrcoef(v1, v2)[0, 1]
                corr_matrix[i, j] = corr
            else:
                corr_matrix[i, j] = np.nan

    # Publication figure setup
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 10,
    })

    fig, ax = plt.subplots(figsize=(7, 6))

    # Use a perceptually uniform diverging colormap
    im = ax.imshow(corr_matrix, cmap='RdYlBu_r', vmin=0, vmax=1, aspect='equal')

    ax.set_xticks(range(n_backends))
    ax.set_yticks(range(n_backends))

    # Clean backend labels
    labels = [b.replace('_', ' ').title() for b in backends]
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    # Add separator lines before baseline section
    if baseline_start_idx > 0 and baseline_start_idx < n_backends:
        ax.axhline(y=baseline_start_idx - 0.5, color='#333333', linewidth=1.5, linestyle='--')
        ax.axvline(x=baseline_start_idx - 0.5, color='#333333', linewidth=1.5, linestyle='--')

    # Add correlation values with smart text coloring
    for i in range(n_backends):
        for j in range(n_backends):
            val = corr_matrix[i, j]
            if not np.isnan(val):
                # White text on dark backgrounds, black on light
                text_color = 'white' if val > 0.65 or val < 0.35 else '#333333'
                fontweight = 'bold' if i == j else 'normal'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                       color=text_color, fontsize=8, fontweight=fontweight)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Pearson r', fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    # Title with stage color
    title_text = f"{donor} - {stage}" if donor and stage else sample_id
    ax.set_title(f'Backend Agreement\n{title_text}', fontsize=11, fontweight='bold',
                color=stage_color, pad=10)

    # Clean up spines
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white',
               edgecolor='none', pad_inches=0.1)
    plt.close()


def process_sample(
    results_dir: Path,
    sample_id: str,
    output_dir: Path,
    label_source: str = "hlca",
):
    """Process a single sample."""
    print(f"Processing {sample_id}...")

    # Load data
    proportions = load_backend_proportions(results_dir, sample_id, label_source)
    if not proportions:
        print(f"  No proportions found for {sample_id}")
        return

    spatial_result = load_spatial_coords(results_dir, sample_id, label_source)
    if spatial_result is None:
        print(f"  No spatial coordinates found for {sample_id}")
        return
    spatial_coords, spatial_index = spatial_result

    cell_types = get_common_celltypes(proportions)
    if not cell_types:
        print(f"  No common cell types for {sample_id}")
        return

    print(f"  Found {len(proportions)} backends, {len(cell_types)} common cell types")

    # Create spatial comparison grid
    create_backend_comparison_figure(
        proportions,
        spatial_coords,
        spatial_index,
        cell_types,
        sample_id,
        output_dir / f"{sample_id}_spatial_comparison.png",
    )

    # Create correlation heatmap
    create_correlation_heatmap(
        proportions,
        cell_types,
        sample_id,
        output_dir / f"{sample_id}_backend_correlation.png",
    )


def get_all_samples(results_dir: Path, label_source: str = "hlca") -> list[str]:
    """Get all sample IDs."""
    base = results_dir / label_source
    for backend_dir in base.iterdir():
        if backend_dir.is_dir():
            samples_dir = backend_dir / "samples"
            if samples_dir.exists():
                return [s.name for s in samples_dir.iterdir() if s.is_dir()]
    return []


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create multi-backend comparison figures")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/spatial_benchmark"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/spatial_benchmark/figures/backend_comparison"),
    )
    parser.add_argument("--label-source", default="hlca")
    parser.add_argument("--sample", type=str, default=None, help="Single sample to process")

    args = parser.parse_args()

    if args.sample:
        process_sample(args.results_dir, args.sample, args.output_dir, args.label_source)
    else:
        samples = get_all_samples(args.results_dir, args.label_source)
        print(f"Processing {len(samples)} samples...")
        for sample in sorted(samples):
            process_sample(args.results_dir, sample, args.output_dir, args.label_source)
