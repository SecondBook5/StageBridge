"""
Visualization utilities for spatial mapping backends.

Provides publication-quality visualizations for Tangram and DestVI outputs,
including spatial patterns, cell-type-specific analysis, and gamma space exploration.
"""

from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_proportions_spatial(
    spatial_adata,
    cell_types: list[str],
    ncols: int = 3,
    cmap: str = "Reds",
    figsize_per_plot: tuple[float, float] = (4, 4),
    save_path: Path | None = None,
):
    """
    Plot multiple cell type proportions in spatial coordinates.

    Args:
        spatial_adata: AnnData with spatial coordinates and cell type proportions
        cell_types: List of cell types to visualize
        ncols: Number of columns in subplot grid
        cmap: Matplotlib colormap
        figsize_per_plot: Size of each subplot
        save_path: If provided, saves figure
    """
    import scanpy as sc

    nrows = int(np.ceil(len(cell_types) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows),
    )
    axes = np.atleast_1d(axes).flatten()

    for idx, ct in enumerate(cell_types):
        ax = axes[idx]

        # Try different column naming conventions
        col = None
        for prefix in ["tangram_", "destvi_", ""]:
            candidate = f"{prefix}{ct}"
            if candidate in spatial_adata.obs.columns:
                col = candidate
                break

        if col is None:
            ax.text(0.5, 0.5, f"Not found:\n{ct}", ha="center", va="center")
            ax.axis("off")
            continue

        # Plot
        coords = spatial_adata.obsm["spatial"]
        values = spatial_adata.obs[col].values
        scatter = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=values,
            cmap=cmap,
            s=10,
            vmin=0,
            vmax=np.quantile(values, 0.99),
        )
        plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(ct)
        ax.axis("equal")

    # Hide unused subplots
    for idx in range(len(cell_types), len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

    plt.close()


def plot_gamma_pca_spatial(
    gamma_pca: np.ndarray,
    coords: np.ndarray,
    explained_variance: np.ndarray,
    cell_type: str,
    save_path: Path | None = None,
):
    """
    Plot gamma PCA components in spatial coordinates.

    Args:
        gamma_pca: PCA-transformed gamma values (n_spots, n_components)
        coords: Spatial coordinates (n_spots, 2)
        explained_variance: Explained variance ratio per component
        cell_type: Cell type name
        save_path: If provided, saves figure
    """
    n_components = min(gamma_pca.shape[1], 3)

    fig, axes = plt.subplots(1, n_components, figsize=(5 * n_components, 4))
    if n_components == 1:
        axes = [axes]

    for i in range(n_components):
        ax = axes[i]
        pc_values = gamma_pca[:, i]
        scatter = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=pc_values,
            cmap="RdBu_r",
            s=20,
        )
        plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        var_exp = explained_variance[i]
        ax.set_title(f"{cell_type}\nSpatial PC{i+1} ({var_exp:.1%} var)")
        ax.set_xlabel("Spatial X")
        ax.set_ylabel("Spatial Y")
        ax.axis("equal")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

    plt.close()


def plot_projected_genes_spatial(
    spatial_adata,
    gene_expression: pd.DataFrame,
    gene_names: list[str],
    ncols: int = 3,
    cmap: str = "viridis",
    log_scale: bool = True,
    figsize_per_plot: tuple[float, float] = (4, 4),
    save_path: Path | None = None,
):
    """
    Plot projected gene expression in spatial coordinates.

    Args:
        spatial_adata: AnnData with spatial coordinates
        gene_expression: DataFrame with gene expression (spots × genes)
        gene_names: List of genes to plot
        ncols: Number of columns in subplot grid
        cmap: Matplotlib colormap
        log_scale: Apply log1p transform
        figsize_per_plot: Size of each subplot
        save_path: If provided, saves figure
    """
    nrows = int(np.ceil(len(gene_names) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows),
    )
    axes = np.atleast_1d(axes).flatten()

    coords = spatial_adata.obsm["spatial"]

    for idx, gene in enumerate(gene_names):
        ax = axes[idx]

        if gene not in gene_expression.columns:
            ax.text(0.5, 0.5, f"Not found:\n{gene}", ha="center", va="center")
            ax.axis("off")
            continue

        values = gene_expression[gene].values
        if log_scale:
            values = np.log1p(1e4 * values)

        scatter = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=values,
            cmap=cmap,
            s=10,
        )
        plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(gene)
        ax.axis("equal")

    # Hide unused subplots
    for idx in range(len(gene_names), len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

    plt.close()


def plot_proportion_distribution(
    proportions: pd.DataFrame,
    cell_types: list[str] | None = None,
    kind: str = "violin",
    figsize: tuple[float, float] = (12, 5),
    save_path: Path | None = None,
):
    """
    Plot distribution of cell type proportions.

    Args:
        proportions: DataFrame with cell type proportions (spots × cell_types)
        cell_types: Cell types to plot. If None, plots all.
        kind: Plot kind - 'violin', 'box', or 'hist'
        figsize: Figure size
        save_path: If provided, saves figure
    """
    if cell_types is None:
        cell_types = proportions.columns.tolist()

    prop_data = proportions[cell_types]

    fig, ax = plt.subplots(figsize=figsize)

    if kind == "violin":
        ax.violinplot(
            [prop_data[ct].values for ct in cell_types],
            positions=range(len(cell_types)),
            showmeans=True,
            showextrema=True,
        )
    elif kind == "box":
        ax.boxplot(
            [prop_data[ct].values for ct in cell_types],
            labels=cell_types,
        )
    elif kind == "hist":
        for ct in cell_types:
            ax.hist(prop_data[ct].values, bins=50, alpha=0.5, label=ct)
        ax.legend()
    else:
        raise ValueError(f"Unknown kind: {kind}")

    ax.set_xticks(range(len(cell_types)))
    ax.set_xticklabels(cell_types, rotation=45, ha="right")
    ax.set_ylabel("Proportion")
    ax.set_title("Cell Type Proportion Distribution")
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

    plt.close()


def plot_proportion_heatmap(
    proportions: pd.DataFrame,
    cell_types: list[str] | None = None,
    cluster_spots: bool = True,
    cluster_celltypes: bool = True,
    cmap: str = "YlOrRd",
    figsize: tuple[float, float] = (10, 8),
    save_path: Path | None = None,
):
    """
    Plot cell type proportions as a clustered heatmap.

    Args:
        proportions: DataFrame with cell type proportions (spots × cell_types)
        cell_types: Cell types to include. If None, uses all.
        cluster_spots: Cluster spots (rows)
        cluster_celltypes: Cluster cell types (columns)
        cmap: Matplotlib colormap
        figsize: Figure size
        save_path: If provided, saves figure
    """
    if cell_types is None:
        cell_types = proportions.columns.tolist()

    prop_data = proportions[cell_types]

    # Use seaborn clustermap
    g = sns.clustermap(
        prop_data.T,  # Transpose for cell types as rows
        cmap=cmap,
        row_cluster=cluster_celltypes,
        col_cluster=cluster_spots,
        figsize=figsize,
        cbar_kws={"label": "Proportion"},
        yticklabels=True,
        xticklabels=False,  # Too many spots to show
    )

    g.ax_heatmap.set_xlabel("Spots")
    g.ax_heatmap.set_ylabel("Cell Types")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

    plt.close()


def plot_entropy_vs_sparsity(
    proportions: pd.DataFrame,
    figsize: tuple[float, float] = (6, 5),
    save_path: Path | None = None,
):
    """
    Plot entropy vs sparsity scatter for spots.

    Args:
        proportions: DataFrame with cell type proportions (spots × cell_types)
        figsize: Figure size
        save_path: If provided, saves figure
    """
    from ..spatial_backends.base import compute_cell_type_entropy

    # Compute metrics per spot
    entropy = compute_cell_type_entropy(proportions)

    # Sparsity: fraction of near-zero proportions
    sparsity = (proportions.values < 0.05).sum(axis=1) / proportions.shape[1]

    fig, ax = plt.subplots(figsize=figsize)

    scatter = ax.scatter(
        entropy,
        sparsity,
        alpha=0.5,
        s=10,
        c=entropy,
        cmap="viridis",
    )
    plt.colorbar(scatter, ax=ax, label="Entropy")

    ax.set_xlabel("Cell Type Entropy")
    ax.set_ylabel("Sparsity (fraction of cell types < 5%)")
    ax.set_title("Spot-level Entropy vs Sparsity")
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

    plt.close()


def plot_spatial_autocorrelation(
    morans_i: dict[str, float],
    figsize: tuple[float, float] = (8, 5),
    save_path: Path | None = None,
):
    """
    Plot Moran's I spatial autocorrelation for cell types.

    Args:
        morans_i: Dictionary mapping cell types to Moran's I values
        figsize: Figure size
        save_path: If provided, saves figure
    """
    cell_types = list(morans_i.keys())
    values = [morans_i[ct] for ct in cell_types]

    fig, ax = plt.subplots(figsize=figsize)

    colors = ["green" if v > 0 else "red" for v in values]
    ax.barh(range(len(cell_types)), values, color=colors, alpha=0.7)
    ax.set_yticks(range(len(cell_types)))
    ax.set_yticklabels(cell_types)
    ax.set_xlabel("Moran's I")
    ax.set_title("Spatial Autocorrelation (Moran's I)")
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.grid(alpha=0.3, axis="x")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

    plt.close()


def create_comprehensive_report(
    backend_result,
    spatial_adata,
    output_dir: Path,
    cell_types: list[str] | None = None,
    n_genes_to_plot: int = 6,
):
    """
    Create comprehensive visualization report for spatial mapping.

    Generates a full suite of publication-quality figures:
    - Spatial proportions for each cell type
    - Proportion distributions
    - Entropy vs sparsity
    - Clustered heatmap
    - Spatial autocorrelation (if available)

    Args:
        backend_result: BackendMappingResult from mapping
        spatial_adata: Annotated spatial AnnData
        output_dir: Directory to save figures
        cell_types: Cell types to include. If None, uses all.
        n_genes_to_plot: Number of genes to plot (if available)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    proportions = backend_result.cell_type_proportions

    if cell_types is None:
        cell_types = proportions.columns.tolist()

    print(f"Generating comprehensive visualization report in {output_dir}...")

    # 1. Spatial proportions
    print("  Plotting spatial proportions...")
    plot_proportions_spatial(
        spatial_adata,
        cell_types=cell_types,
        save_path=output_dir / "spatial_proportions.png",
    )

    # 2. Proportion distributions
    print("  Plotting proportion distributions...")
    plot_proportion_distribution(
        proportions,
        cell_types=cell_types,
        kind="violin",
        save_path=output_dir / "proportion_distributions.png",
    )

    # 3. Entropy vs sparsity
    print("  Plotting entropy vs sparsity...")
    plot_entropy_vs_sparsity(
        proportions,
        save_path=output_dir / "entropy_vs_sparsity.png",
    )

    # 4. Clustered heatmap
    print("  Plotting clustered heatmap...")
    plot_proportion_heatmap(
        proportions,
        cell_types=cell_types,
        save_path=output_dir / "proportion_heatmap.png",
    )

    # 5. Spatial autocorrelation (if available)
    if "morans_i" in backend_result.metadata:
        print("  Plotting spatial autocorrelation...")
        plot_spatial_autocorrelation(
            backend_result.metadata["morans_i"],
            save_path=output_dir / "spatial_autocorrelation.png",
        )

    print(f"  Report complete: {len(list(output_dir.glob('*.png')))} figures generated")
