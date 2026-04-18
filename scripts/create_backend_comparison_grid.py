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
    spot_size: float = 5,
):
    """
    Create multi-backend comparison figure.

    Rows = cell types, Columns = backends
    Marker scoring is separated as a baseline method at the end.
    """
    # Separate deconvolution methods from baselines
    BASELINE_BACKENDS = {"marker_scoring"}

    all_backends = set(proportions.keys())
    deconv_backends = sorted(all_backends - BASELINE_BACKENDS)
    baseline_backends = sorted(all_backends & BASELINE_BACKENDS)

    # Order: deconvolution methods first, then baselines at the end
    backends = deconv_backends + baseline_backends
    n_backends = len(backends)
    baseline_start_idx = len(deconv_backends)  # Column index where baselines start

    # Limit cell types shown
    cell_types = cell_types[:max_celltypes]
    n_celltypes = len(cell_types)

    if n_celltypes == 0 or n_backends == 0:
        print(f"  Skipping {sample_id}: no common cell types or backends")
        return

    # Create spatial coords DataFrame for alignment
    spatial_df = pd.DataFrame(spatial_coords, index=spatial_index, columns=["x", "y"])

    # Create figure: rows = cell types, cols = backends
    fig, axes = plt.subplots(
        n_celltypes, n_backends,
        figsize=(3 * n_backends, 3 * n_celltypes),
        squeeze=False,
    )

    # Color normalization per cell type (shared across backends)
    for row, ct in enumerate(cell_types):
        # Get global min/max across all backends for this cell type
        all_vals = []
        for backend in backends:
            if ct in proportions[backend].columns:
                all_vals.extend(proportions[backend][ct].values)

        if not all_vals:
            continue

        vmin = np.percentile(all_vals, 1)
        vmax = np.percentile(all_vals, 99)
        norm = Normalize(vmin=vmin, vmax=vmax)

        for col, backend in enumerate(backends):
            ax = axes[row, col]

            if ct in proportions[backend].columns:
                props_df = proportions[backend]

                # Align by index - get common spots
                common_idx = props_df.index.intersection(spatial_df.index)

                if len(common_idx) == 0:
                    ax.text(0.5, 0.5, "No\noverlap", ha='center', va='center', transform=ax.transAxes)
                    ax.set_xticks([])
                    ax.set_yticks([])
                else:
                    # Get aligned values and coords
                    values = props_df.loc[common_idx, ct].values
                    coords = spatial_df.loc[common_idx, ["x", "y"]].values

                    scatter = ax.scatter(
                        coords[:, 0],
                        coords[:, 1],
                        c=values,
                        cmap="Reds",
                        norm=norm,
                        s=spot_size,
                        edgecolors='none',
                        rasterized=True,
                    )
                    ax.set_aspect('equal')
                    ax.set_xticks([])
                    ax.set_yticks([])
            else:
                ax.text(0.5, 0.5, "N/A", ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])

            # Column header (backend name)
            if row == 0:
                is_baseline = backend in BASELINE_BACKENDS
                title_text = f"{backend.upper()}\n(Baseline)" if is_baseline else backend.upper()
                title_color = '#666666' if is_baseline else 'black'
                ax.set_title(title_text, fontsize=11, fontweight='bold', color=title_color)

            # Row label (cell type)
            if col == 0:
                ax.set_ylabel(ct, fontsize=10, fontweight='bold')

            # Add vertical separator line before baseline section
            if col == baseline_start_idx and baseline_start_idx > 0:
                # Draw line on left edge of baseline columns
                ax.axvline(x=ax.get_xlim()[0], color='#333333', linewidth=2, linestyle='--')

    # Add a visible separator between deconvolution and baseline sections
    if baseline_start_idx > 0 and baseline_start_idx < n_backends:
        # Add text annotation for the sections
        fig.text(
            baseline_start_idx / n_backends - 0.02, 0.5,
            '|', fontsize=40, ha='center', va='center',
            transform=fig.transFigure, color='#999999'
        )

    # Overall title
    fig.suptitle(
        f"Backend Comparison: {sample_id}",
        fontsize=14,
        fontweight='bold',
        y=1.02,
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  Saved: {output_path}")


def create_correlation_heatmap(
    proportions: dict[str, pd.DataFrame],
    cell_types: list[str],
    sample_id: str,
    output_path: Path,
):
    """Create correlation heatmap between backends."""
    # Separate deconvolution methods from baselines (same ordering as spatial figure)
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
            # Get common indices
            common_idx = proportions[b1].index.intersection(proportions[b2].index)
            common_cols = [c for c in cell_types if c in proportions[b1].columns and c in proportions[b2].columns]

            if len(common_idx) > 0 and len(common_cols) > 0:
                v1 = proportions[b1].loc[common_idx, common_cols].values.flatten()
                v2 = proportions[b2].loc[common_idx, common_cols].values.flatten()

                # Pearson correlation
                corr = np.corrcoef(v1, v2)[0, 1]
                corr_matrix[i, j] = corr
            else:
                corr_matrix[i, j] = np.nan

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=0, vmax=1)

    ax.set_xticks(range(n_backends))
    ax.set_yticks(range(n_backends))

    # Format labels - mark baselines
    xlabels = [f"{b.upper()}\n(Baseline)" if b in BASELINE_BACKENDS else b.upper() for b in backends]
    ylabels = [f"{b.upper()} (B)" if b in BASELINE_BACKENDS else b.upper() for b in backends]
    ax.set_xticklabels(xlabels, rotation=45, ha='right')
    ax.set_yticklabels(ylabels)

    # Add separator lines before baseline section
    if baseline_start_idx > 0 and baseline_start_idx < n_backends:
        ax.axhline(y=baseline_start_idx - 0.5, color='white', linewidth=3)
        ax.axvline(x=baseline_start_idx - 0.5, color='white', linewidth=3)

    # Add correlation values
    for i in range(n_backends):
        for j in range(n_backends):
            val = corr_matrix[i, j]
            if not np.isnan(val):
                color = 'white' if val < 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=10)

    plt.colorbar(im, ax=ax, label='Pearson Correlation')
    ax.set_title(f'Backend Agreement: {sample_id}', fontsize=12, fontweight='bold')

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
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
