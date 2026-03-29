"""
Visualization utilities for spatial mapping backends.

Provides publication-quality visualizations for Tangram and DestVI outputs,
including spatial patterns, cell-type-specific analysis, and gamma space exploration.
"""

from pathlib import Path
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
        ax.set_title(f"{cell_type}\nSpatial PC{i + 1} ({var_exp:.1%} var)")
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
            tick_labels=cell_types,
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
    from .backend_base import compute_cell_type_entropy

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


# =============================================================================
# Advanced Visualizations (inspired by scvi-tools tutorials)
# =============================================================================


def plot_backend_comparison_spatial(
    spatial_adata,
    backend_results: dict[str, pd.DataFrame],
    cell_type: str,
    cmap: str = "Reds",
    figsize_per_plot: tuple[float, float] = (5, 5),
    save_path: Path | None = None,
):
    """
    Compare cell type proportions across backends in spatial coordinates.

    Shows the same tissue section with different backend predictions side by side,
    making it easy to see where backends agree or disagree.

    Args:
        spatial_adata: AnnData with spatial coordinates in obsm['spatial']
        backend_results: Dict mapping backend name to proportions DataFrame
        cell_type: Cell type to visualize
        cmap: Matplotlib colormap
        figsize_per_plot: Size per subplot
        save_path: If provided, saves figure
    """
    n_backends = len(backend_results)
    fig, axes = plt.subplots(
        1, n_backends,
        figsize=(figsize_per_plot[0] * n_backends, figsize_per_plot[1]),
    )
    if n_backends == 1:
        axes = [axes]

    coords = spatial_adata.obsm["spatial"]

    # Find global min/max for consistent color scaling
    all_values = []
    for props in backend_results.values():
        if cell_type in props.columns:
            all_values.extend(props[cell_type].values)
    vmax = np.quantile(all_values, 0.99) if all_values else 1.0

    for idx, (backend_name, props) in enumerate(backend_results.items()):
        ax = axes[idx]

        if cell_type not in props.columns:
            ax.text(0.5, 0.5, f"{cell_type}\nnot in {backend_name}",
                   ha="center", va="center", transform=ax.transAxes)
            ax.set_title(backend_name.upper())
            ax.axis("off")
            continue

        values = props[cell_type].values
        scatter = ax.scatter(
            coords[:, 0], coords[:, 1],
            c=values, cmap=cmap, s=15,
            vmin=0, vmax=vmax,
        )
        plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"{backend_name.upper()}\n{cell_type}")
        ax.axis("equal")
        ax.axis("off")

    plt.suptitle(f"Backend Comparison: {cell_type}", fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def plot_cell_type_colocalization(
    proportions: pd.DataFrame,
    cell_types: list[str] | None = None,
    method: str = "spearman",
    cmap: str = "RdBu_r",
    figsize: tuple[float, float] = (10, 8),
    save_path: Path | None = None,
):
    """
    Plot cell type co-localization correlation matrix.

    Shows which cell types tend to appear together in the same spots.
    Positive correlation = co-localization, negative = mutual exclusion.

    Args:
        proportions: DataFrame with cell type proportions (spots x cell_types)
        cell_types: Subset of cell types. If None, uses all.
        method: Correlation method ('spearman' or 'pearson')
        cmap: Colormap (diverging recommended)
        figsize: Figure size
        save_path: If provided, saves figure
    """
    if cell_types is None:
        cell_types = proportions.columns.tolist()

    prop_subset = proportions[cell_types]
    corr_matrix = prop_subset.corr(method=method)

    fig, ax = plt.subplots(figsize=figsize)

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap=cmap,
        center=0,
        vmin=-1, vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": f"{method.title()} Correlation"},
        annot=True,
        fmt=".2f",
        annot_kws={"size": 8},
        ax=ax,
    )

    ax.set_title("Cell Type Co-localization", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def plot_dominant_cell_type_map(
    spatial_adata,
    proportions: pd.DataFrame,
    min_proportion: float = 0.2,
    figsize: tuple[float, float] = (10, 10),
    save_path: Path | None = None,
):
    """
    Plot spatial map colored by dominant cell type per spot.

    Each spot is colored by its most abundant cell type, with spots
    below min_proportion shown in gray (ambiguous/mixed).

    Args:
        spatial_adata: AnnData with spatial coordinates
        proportions: DataFrame with cell type proportions
        min_proportion: Minimum proportion to be considered dominant
        figsize: Figure size
        save_path: If provided, saves figure
    """
    coords = spatial_adata.obsm["spatial"]

    # Find dominant cell type per spot
    dominant_ct = proportions.idxmax(axis=1)
    max_prop = proportions.max(axis=1)

    # Mark spots below threshold as "Mixed"
    dominant_ct = dominant_ct.where(max_prop >= min_proportion, "Mixed")

    # Create color palette
    cell_types = sorted(proportions.columns.tolist())
    n_colors = len(cell_types) + 1  # +1 for Mixed
    palette = sns.color_palette("husl", n_colors)
    color_map = {ct: palette[i] for i, ct in enumerate(cell_types)}
    color_map["Mixed"] = (0.7, 0.7, 0.7)  # Gray for mixed

    colors = [color_map[ct] for ct in dominant_ct]

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=20, alpha=0.8)

    # Create legend
    handles = [plt.Line2D([0], [0], marker='o', color='w',
               markerfacecolor=color_map[ct], markersize=10, label=ct)
               for ct in cell_types + ["Mixed"]]
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.02, 1),
              title="Dominant Cell Type")

    ax.set_title(f"Dominant Cell Type Map (threshold={min_proportion:.0%})")
    ax.axis("equal")
    ax.axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def plot_cell_type_pie_summary(
    proportions: pd.DataFrame,
    spatial_adata=None,
    top_n: int = 10,
    figsize: tuple[float, float] = (14, 6),
    save_path: Path | None = None,
):
    """
    Plot overall cell type composition as pie chart with spatial context.

    Shows two panels:
    1. Pie chart of average cell type proportions
    2. If spatial_adata provided, spatial scatter colored by total cell density

    Args:
        proportions: DataFrame with cell type proportions
        spatial_adata: Optional AnnData with spatial coordinates
        top_n: Number of top cell types to show (rest grouped as "Other")
        figsize: Figure size
        save_path: If provided, saves figure
    """
    # Calculate mean proportions
    mean_props = proportions.mean().sort_values(ascending=False)

    # Group small cell types as "Other"
    top_cts = mean_props.head(top_n)
    if len(mean_props) > top_n:
        other_sum = mean_props.iloc[top_n:].sum()
        top_cts = pd.concat([top_cts, pd.Series({"Other": other_sum})])

    n_panels = 2 if spatial_adata is not None else 1
    fig, axes = plt.subplots(1, n_panels, figsize=figsize)
    if n_panels == 1:
        axes = [axes]

    # Panel 1: Pie chart
    ax = axes[0]
    colors = sns.color_palette("husl", len(top_cts))
    wedges, texts, autotexts = ax.pie(
        top_cts.values,
        labels=top_cts.index,
        autopct=lambda p: f'{p:.1f}%' if p > 3 else '',
        colors=colors,
        pctdistance=0.75,
    )
    ax.set_title("Average Cell Type Composition", fontsize=12)

    # Panel 2: Spatial density (if available)
    if spatial_adata is not None and len(axes) > 1:
        ax = axes[1]
        coords = spatial_adata.obsm["spatial"]
        # Total "confidence" or sum of top proportions as density proxy
        density = proportions[mean_props.head(top_n).index].sum(axis=1)

        scatter = ax.scatter(
            coords[:, 0], coords[:, 1],
            c=density, cmap="viridis", s=15, alpha=0.8
        )
        plt.colorbar(scatter, ax=ax, label="Cell Type Confidence")
        ax.set_title("Spatial Cell Type Density", fontsize=12)
        ax.axis("equal")
        ax.axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def plot_multi_backend_radar(
    backend_metrics: dict[str, dict[str, float]],
    metrics_to_show: list[str] | None = None,
    figsize: tuple[float, float] = (8, 8),
    save_path: Path | None = None,
):
    """
    Plot radar chart comparing multiple backends across metrics.

    Like the backend_comparison radar but with more customization options.

    Args:
        backend_metrics: Dict of {backend_name: {metric_name: value}}
        metrics_to_show: Subset of metrics to plot. If None, uses all.
        figsize: Figure size
        save_path: If provided, saves figure
    """
    # Get all metrics
    all_metrics = set()
    for metrics in backend_metrics.values():
        all_metrics.update(metrics.keys())

    if metrics_to_show is None:
        metrics_to_show = sorted(all_metrics)

    # Create angles for radar
    n_metrics = len(metrics_to_show)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))

    colors = sns.color_palette("husl", len(backend_metrics))

    for idx, (backend_name, metrics) in enumerate(backend_metrics.items()):
        values = [metrics.get(m, 0) for m in metrics_to_show]
        values += values[:1]  # Close the polygon

        ax.plot(angles, values, 'o-', linewidth=2, label=backend_name, color=colors[idx])
        ax.fill(angles, values, alpha=0.25, color=colors[idx])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_to_show, size=10)
    ax.set_ylim(0, 1)
    ax.set_title("Backend Comparison", fontsize=14, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def plot_spatial_gene_signature(
    spatial_adata,
    gene_expression: pd.DataFrame | None = None,
    genes: list[str] | None = None,
    cell_type: str | None = None,
    aggregate: bool = True,
    cmap: str = "Reds",
    figsize: tuple[float, float] = (8, 8),
    title: str | None = None,
    save_path: Path | None = None,
):
    """
    Plot gene signature score in spatial coordinates.

    Can plot from cell-type-specific imputed expression (DestVI style)
    or from the spatial adata directly.

    Args:
        spatial_adata: AnnData with spatial coordinates
        gene_expression: Optional DataFrame with imputed expression
        genes: List of genes to aggregate as signature
        cell_type: Cell type name (for title)
        aggregate: If True, sum genes; if False, plot first gene
        cmap: Colormap
        figsize: Figure size
        title: Custom title
        save_path: If provided, saves figure
    """
    coords = spatial_adata.obsm["spatial"]

    if gene_expression is not None and genes is not None:
        # Use imputed expression
        available_genes = [g for g in genes if g in gene_expression.columns]
        if not available_genes:
            raise ValueError(f"None of {genes} found in gene_expression")

        if aggregate:
            values = gene_expression[available_genes].sum(axis=1).values
            values = np.log1p(1e4 * values)
        else:
            values = gene_expression[available_genes[0]].values
            values = np.log1p(1e4 * values)
    elif genes is not None:
        # Use spatial adata directly
        available_genes = [g for g in genes if g in spatial_adata.var_names]
        if not available_genes:
            raise ValueError(f"None of {genes} found in spatial_adata")

        expr_matrix = spatial_adata[:, available_genes].X
        if hasattr(expr_matrix, 'toarray'):
            expr_matrix = expr_matrix.toarray()

        if aggregate:
            values = np.log1p(expr_matrix.sum(axis=1))
        else:
            values = np.log1p(expr_matrix[:, 0])
    else:
        raise ValueError("Must provide either gene_expression DataFrame or genes list")

    fig, ax = plt.subplots(figsize=figsize)

    # Background spots
    ax.scatter(coords[:, 0], coords[:, 1], c='lightgray', s=5, alpha=0.3)

    # Gene signature
    scatter = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=values, cmap=cmap, s=20, alpha=0.8,
    )
    plt.colorbar(scatter, ax=ax, label="log(expression)")

    if title:
        ax.set_title(title, fontsize=12)
    elif cell_type and genes:
        gene_str = ", ".join(genes[:3])
        if len(genes) > 3:
            gene_str += f" (+{len(genes)-3})"
        ax.set_title(f"{cell_type}: {gene_str}", fontsize=12)

    ax.axis("equal")
    ax.axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def create_publication_figure_panel(
    spatial_adata,
    proportions: pd.DataFrame,
    cell_types: list[str],
    output_path: Path,
    include_legend: bool = True,
    cmap: str = "Reds",
    n_cols: int = 4,
):
    """
    Create publication-ready multi-panel figure of cell type spatial distributions.

    Produces a clean figure suitable for Nature Methods with:
    - Consistent coloring across panels
    - Proper aspect ratios
    - Optional shared legend

    Args:
        spatial_adata: AnnData with spatial coordinates
        proportions: DataFrame with cell type proportions
        cell_types: List of cell types to include
        output_path: Path to save figure (PNG and PDF)
        include_legend: Whether to add a shared colorbar
        cmap: Colormap
        n_cols: Number of columns
    """
    n_cts = len(cell_types)
    n_rows = int(np.ceil(n_cts / n_cols))

    # Calculate figure size
    panel_size = 3
    fig_width = panel_size * n_cols + (1.5 if include_legend else 0)
    fig_height = panel_size * n_rows

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    axes = np.atleast_2d(axes)

    coords = spatial_adata.obsm["spatial"]

    # Global normalization for consistent colors
    vmax = proportions[cell_types].values.max()
    vmax = min(vmax, np.quantile(proportions[cell_types].values.flatten(), 0.99))

    for idx, ct in enumerate(cell_types):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]

        values = proportions[ct].values
        scatter = ax.scatter(
            coords[:, 0], coords[:, 1],
            c=values, cmap=cmap, s=8,
            vmin=0, vmax=vmax,
            rasterized=True,  # Better PDF rendering
        )
        ax.set_title(ct, fontsize=10, fontweight='bold')
        ax.axis("equal")
        ax.axis("off")

    # Hide unused panels
    for idx in range(n_cts, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].axis("off")

    # Add shared colorbar
    if include_legend:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, vmax))
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label("Proportion", fontsize=10)

    plt.tight_layout(rect=[0, 0, 0.9 if include_legend else 1, 1])

    # Save both PNG and PDF
    output_path = Path(output_path)
    plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    print(f"Saved publication figure to {output_path.with_suffix('.png')} and .pdf")
    plt.close()
