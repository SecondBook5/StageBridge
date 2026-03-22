"""Recreation of LungPCA paper figures using StageBridge data.

This module provides functions to generate publication-quality figures
that match the style and content of the original Peng et al. LungPCA paper.

Figures included:
- Figure 1B-style: Sankey diagram of cell type composition by stage
- Figure 1C-style: UMAP with stage coloring and marker genes
- Figure 2C-style: Spatial plots with histology overlay
- Figure 3A-style: Cell type UMAP
- Figure 3H-style: MP composition boxplots by stage
- Figure 3L-style: Alluvial plot of MP transitions
- Figure 4B-style: Correlation heatmap
- Figure 5D-style: Neighborhood composition stacked bars
- Figure 5E-style: Violin + boxplot comparisons
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
from typing import Any

from .lungpca_style import (
    STAGE_COLORS,
    STAGE_ORDER,
    EPITHELIAL_COLORS,
    STROMAL_COLORS,
    MP_COLORS,
    MAJOR_CELLTYPE_COLORS,
    configure_lungpca_style,
    create_lungpca_figure,
    save_lungpca_figure,
    plot_sankey_diagram,
    plot_violin_boxplot,
    plot_boxplot_jitter,
    plot_stacked_bar,
    plot_heatmap,
    plot_alluvial,
    get_magma_white,
)

from stagebridge.logging_utils import get_logger

log = get_logger(__name__)


# =============================================================================
# Figure 1B: Sankey Diagram - Cell Type Flow by Stage
# =============================================================================

def figure_1b_sankey(
    adata: Any,
    stage_col: str = "stage",
    celltype_col: str = "cell_type",
    output_path: Path | str | None = None,
) -> Any:
    """Create Figure 1B-style Sankey diagram showing cell type composition by stage.

    Parameters
    ----------
    adata : AnnData
        Annotated data with stage and cell type columns in obs
    stage_col : str
        Column name for stage labels
    celltype_col : str
        Column name for cell type labels
    output_path : Path, optional
        If provided, save figure to this path

    Returns
    -------
    fig : Figure or plotly Figure
        The generated figure
    """
    configure_lungpca_style()

    # Count cells per stage-celltype combination
    df = adata.obs[[stage_col, celltype_col]].copy()
    counts = df.groupby([stage_col, celltype_col]).size().unstack(fill_value=0)

    # Order stages
    ordered_stages = [s for s in STAGE_ORDER if s in counts.index]
    counts = counts.loc[ordered_stages]

    # Build flow matrix (stage -> celltype)
    flow_matrix = counts.values.astype(float)
    source_labels = list(counts.index)
    target_labels = list(counts.columns)

    # Get colors
    target_colors = {**EPITHELIAL_COLORS, **STROMAL_COLORS, **MAJOR_CELLTYPE_COLORS}

    fig = plot_sankey_diagram(
        source_labels=source_labels,
        target_labels=target_labels,
        flow_matrix=flow_matrix,
        source_colors=STAGE_COLORS,
        target_colors=target_colors,
        output_path=output_path,
        title="Cell Type Composition by Stage",
    )

    log.info(f"Generated Figure 1B-style Sankey diagram")
    return fig


# =============================================================================
# Figure 1C: UMAP with Stage Coloring
# =============================================================================

def figure_1c_umap(
    adata: Any,
    stage_col: str = "stage",
    output_path: Path | str | None = None,
    marker_genes: list[str] | None = None,
    figsize: tuple[float, float] = (15, 6),
) -> plt.Figure:
    """Create Figure 1C-style UMAP colored by stage with optional marker genes.

    Parameters
    ----------
    adata : AnnData
        Annotated data with UMAP coordinates and stage labels
    stage_col : str
        Column name for stage labels
    output_path : Path, optional
        If provided, save figure to this path
    marker_genes : list of str, optional
        Genes to show as feature plots (default: SFTPC, CEACAM5, etc.)
    figsize : tuple
        Figure size in inches

    Returns
    -------
    fig : Figure
        Matplotlib figure
    """
    import scanpy as sc

    configure_lungpca_style()

    if marker_genes is None:
        # Default markers from LungPCA Figure 1C
        marker_genes = ["SFTPC", "CEACAM5", "KRT7", "MUC5B"]
        # Filter to genes that exist
        marker_genes = [g for g in marker_genes if g in adata.var_names]

    n_panels = 1 + len(marker_genes)
    fig, axes = plt.subplots(1, n_panels, figsize=figsize)
    if n_panels == 1:
        axes = [axes]

    # Main UMAP colored by stage
    ax = axes[0]
    for stage in STAGE_ORDER:
        mask = adata.obs[stage_col] == stage
        if mask.sum() == 0:
            continue
        coords = adata.obsm["X_umap"][mask]
        ax.scatter(
            coords[:, 0], coords[:, 1],
            c=STAGE_COLORS.get(stage, "#d9d9d9"),
            label=stage, s=0.8, alpha=0.6, rasterized=True
        )

    ax.set_xlabel("UMAP1", fontsize=6)
    ax.set_ylabel("UMAP2", fontsize=6)
    ax.legend(markerscale=5, fontsize=6, frameon=False)
    ax.set_title("Stage", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Feature plots for marker genes
    for i, gene in enumerate(marker_genes):
        ax = axes[i + 1]
        if gene in adata.var_names:
            gene_idx = adata.var_names.get_loc(gene)
            expr = adata.X[:, gene_idx]
            if hasattr(expr, "toarray"):
                expr = expr.toarray().ravel()
            expr = np.asarray(expr).ravel()

            sc_plot = ax.scatter(
                adata.obsm["X_umap"][:, 0],
                adata.obsm["X_umap"][:, 1],
                c=expr, cmap="Reds", s=0.5, alpha=0.7, rasterized=True
            )
            plt.colorbar(sc_plot, ax=ax, fraction=0.046, pad=0.04)

        ax.set_xlabel("UMAP1", fontsize=6)
        ax.set_ylabel("UMAP2", fontsize=6)
        ax.set_title(gene, fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if output_path:
        save_lungpca_figure(fig, output_path)
        log.info(f"Saved Figure 1C-style UMAP to {output_path}")

    return fig


# =============================================================================
# Figure 3A: Cell Type UMAP
# =============================================================================

def figure_3a_celltype_umap(
    adata: Any,
    celltype_col: str = "cell_type",
    output_path: Path | str | None = None,
    figsize: tuple[float, float] = (5, 5),
) -> plt.Figure:
    """Create Figure 3A-style UMAP colored by cell type.

    Parameters
    ----------
    adata : AnnData
        Annotated data with UMAP coordinates and cell type labels
    celltype_col : str
        Column name for cell type labels
    output_path : Path, optional
        If provided, save figure to this path
    figsize : tuple
        Figure size in inches

    Returns
    -------
    fig : Figure
        Matplotlib figure
    """
    configure_lungpca_style()
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    fig.patch.set_facecolor("white")

    celltypes = adata.obs[celltype_col].unique()
    colors = {**EPITHELIAL_COLORS, **STROMAL_COLORS, **MAJOR_CELLTYPE_COLORS}

    for ct in celltypes:
        mask = adata.obs[celltype_col] == ct
        coords = adata.obsm["X_umap"][mask]
        ax.scatter(
            coords[:, 0], coords[:, 1],
            c=colors.get(ct, "#d9d9d9"),
            label=ct, s=1.5, alpha=0.7, rasterized=True
        )

    ax.set_xlabel("UMAP1", fontsize=6)
    ax.set_ylabel("UMAP2", fontsize=6)
    ax.legend(markerscale=3, fontsize=5, frameon=False, loc="center left", bbox_to_anchor=(1, 0.5))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if output_path:
        save_lungpca_figure(fig, output_path)
        log.info(f"Saved Figure 3A-style cell type UMAP to {output_path}")

    return fig


# =============================================================================
# Figure 3H: Boxplot of Scores by Stage
# =============================================================================

def figure_3h_stage_boxplot(
    adata: Any,
    score_col: str,
    stage_col: str = "stage",
    output_path: Path | str | None = None,
    ylabel: str = "Score",
    title: str = "",
    figsize: tuple[float, float] = (3, 3),
) -> plt.Figure:
    """Create Figure 3H-style boxplot with jitter points.

    Parameters
    ----------
    adata : AnnData
        Annotated data with score and stage columns
    score_col : str
        Column name for the score to plot
    stage_col : str
        Column name for stage labels
    output_path : Path, optional
        If provided, save figure to this path
    ylabel : str
        Y-axis label
    title : str
        Plot title
    figsize : tuple
        Figure size in inches

    Returns
    -------
    fig : Figure
        Matplotlib figure
    """
    configure_lungpca_style()
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    fig.patch.set_facecolor("white")

    # Prepare data
    stages = [s for s in STAGE_ORDER if s in adata.obs[stage_col].unique()]
    data = [adata.obs.loc[adata.obs[stage_col] == s, score_col].values for s in stages]
    colors = [STAGE_COLORS.get(s, "#d9d9d9") for s in stages]

    plot_boxplot_jitter(
        ax, data, positions=list(range(len(stages))),
        colors=colors, labels=stages, jitter_size=1
    )

    ax.set_ylabel(ylabel, fontsize=6)
    ax.set_title(title, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if output_path:
        save_lungpca_figure(fig, output_path)
        log.info(f"Saved Figure 3H-style boxplot to {output_path}")

    return fig


# =============================================================================
# Figure 3L: Alluvial Plot of Composition by Stage
# =============================================================================

def figure_3l_alluvial(
    adata: Any,
    category_col: str,
    stage_col: str = "stage",
    color_map: dict[str, str] | None = None,
    output_path: Path | str | None = None,
    figsize: tuple[float, float] = (4, 3),
) -> plt.Figure:
    """Create Figure 3L-style alluvial/area plot.

    Parameters
    ----------
    adata : AnnData
        Annotated data
    category_col : str
        Column for category (e.g., MP, cell_type)
    stage_col : str
        Column name for stage labels
    color_map : dict, optional
        Color mapping for categories
    output_path : Path, optional
        If provided, save figure to this path
    figsize : tuple
        Figure size in inches

    Returns
    -------
    fig : Figure
        Matplotlib figure
    """
    configure_lungpca_style()
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    fig.patch.set_facecolor("white")

    if color_map is None:
        color_map = MP_COLORS

    # Calculate proportions
    stages = [s for s in STAGE_ORDER if s in adata.obs[stage_col].unique()]
    categories = adata.obs[category_col].unique()

    data = {}
    for stage in stages:
        stage_data = adata.obs[adata.obs[stage_col] == stage]
        counts = stage_data[category_col].value_counts()
        data[stage] = {cat: counts.get(cat, 0) for cat in categories}

    plot_alluvial(ax, data, stages, color_map)

    ax.set_ylabel("Fraction", fontsize=6)
    ax.legend(fontsize=5, frameon=False, loc="center left", bbox_to_anchor=(1, 0.5))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if output_path:
        save_lungpca_figure(fig, output_path)
        log.info(f"Saved Figure 3L-style alluvial plot to {output_path}")

    return fig


# =============================================================================
# Figure 4B: Correlation Heatmap
# =============================================================================

def figure_4b_correlation(
    data: pd.DataFrame | np.ndarray,
    labels: list[str],
    output_path: Path | str | None = None,
    title: str = "Correlation",
    figsize: tuple[float, float] = (6, 5),
    method: str = "spearman",
) -> plt.Figure:
    """Create Figure 4B-style correlation heatmap.

    Parameters
    ----------
    data : DataFrame or ndarray
        Data to compute correlation on (features as columns)
    labels : list of str
        Labels for rows/columns
    output_path : Path, optional
        If provided, save figure to this path
    title : str
        Plot title
    figsize : tuple
        Figure size in inches
    method : str
        Correlation method ('spearman' or 'pearson')

    Returns
    -------
    fig : Figure
        Matplotlib figure
    """
    configure_lungpca_style()
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    fig.patch.set_facecolor("white")

    if isinstance(data, pd.DataFrame):
        corr = data.corr(method=method)
    else:
        from scipy.stats import spearmanr, pearsonr
        if method == "spearman":
            corr, _ = spearmanr(data, axis=0)
        else:
            corr = np.corrcoef(data.T)

    if isinstance(corr, pd.DataFrame):
        corr = corr.values

    im = plot_heatmap(
        ax, corr,
        row_labels=labels,
        col_labels=labels,
        cmap="RdBu_r",
        vmin=-1, vmax=1,
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"{method.capitalize()} r", fontsize=6)

    ax.set_title(title, fontsize=8)

    plt.tight_layout()

    if output_path:
        save_lungpca_figure(fig, output_path)
        log.info(f"Saved Figure 4B-style correlation heatmap to {output_path}")

    return fig


# =============================================================================
# Figure 5D: Neighborhood Composition Stacked Bar
# =============================================================================

def figure_5d_neighborhood(
    neighborhood_df: pd.DataFrame,
    center_col: str = "center_cell_type",
    neighbor_col: str = "neighborhood_cell_type",
    prop_col: str = "neighborhood_cell_prop",
    output_path: Path | str | None = None,
    figsize: tuple[float, float] = (5, 5),
) -> plt.Figure:
    """Create Figure 5D-style neighborhood composition stacked bar chart.

    Parameters
    ----------
    neighborhood_df : DataFrame
        DataFrame with columns for center cell type, neighbor type, and proportion
    center_col : str
        Column name for center cell type
    neighbor_col : str
        Column name for neighbor cell type
    prop_col : str
        Column name for proportion values
    output_path : Path, optional
        If provided, save figure to this path
    figsize : tuple
        Figure size in inches

    Returns
    -------
    fig : Figure
        Matplotlib figure
    """
    configure_lungpca_style()
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    fig.patch.set_facecolor("white")

    # Pivot to get stacked bar format
    pivot = neighborhood_df.pivot_table(
        index=center_col, columns=neighbor_col, values=prop_col, aggfunc="mean"
    ).fillna(0)

    # Prepare data
    groups = list(pivot.index)
    categories = list(pivot.columns)
    data = {cat: pivot[cat].values.tolist() for cat in categories}

    colors = {**MAJOR_CELLTYPE_COLORS, **EPITHELIAL_COLORS, **STROMAL_COLORS}

    plot_stacked_bar(ax, data, groups, colors, normalize=True)

    ax.set_ylabel("Proportion", fontsize=6)
    ax.set_xlabel("Center Cell Type", fontsize=6)
    ax.legend(fontsize=4, frameon=False, loc="center left", bbox_to_anchor=(1, 0.5))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if output_path:
        save_lungpca_figure(fig, output_path)
        log.info(f"Saved Figure 5D-style neighborhood chart to {output_path}")

    return fig


# =============================================================================
# Figure 5E: Violin + Boxplot by Cell Type
# =============================================================================

def figure_5e_violin(
    adata: Any,
    score_col: str,
    celltype_col: str = "cell_type",
    output_path: Path | str | None = None,
    ylabel: str = "Score",
    title: str = "",
    figsize: tuple[float, float] = (5, 5),
) -> plt.Figure:
    """Create Figure 5E-style violin + boxplot by cell type.

    Parameters
    ----------
    adata : AnnData
        Annotated data
    score_col : str
        Column name for score
    celltype_col : str
        Column name for cell type
    output_path : Path, optional
        If provided, save figure to this path
    ylabel : str
        Y-axis label
    title : str
        Plot title
    figsize : tuple
        Figure size in inches

    Returns
    -------
    fig : Figure
        Matplotlib figure
    """
    configure_lungpca_style()
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    fig.patch.set_facecolor("white")

    celltypes = sorted(adata.obs[celltype_col].unique())
    data = [adata.obs.loc[adata.obs[celltype_col] == ct, score_col].values for ct in celltypes]
    colors_dict = {**MAJOR_CELLTYPE_COLORS, **EPITHELIAL_COLORS, **STROMAL_COLORS}
    colors = [colors_dict.get(ct, "#d9d9d9") for ct in celltypes]

    plot_violin_boxplot(
        ax, data, positions=list(range(len(celltypes))),
        colors=colors, labels=celltypes
    )

    ax.set_ylabel(ylabel, fontsize=6)
    ax.set_title(title, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if output_path:
        save_lungpca_figure(fig, output_path)
        log.info(f"Saved Figure 5E-style violin plot to {output_path}")

    return fig


# =============================================================================
# Multi-Panel Figure Generation
# =============================================================================

def generate_stagebridge_figure_panel(
    adata: Any,
    output_dir: Path | str,
    stage_col: str = "stage",
    celltype_col: str = "cell_type",
) -> dict[str, Path]:
    """Generate a complete panel of LungPCA-style figures from StageBridge data.

    Parameters
    ----------
    adata : AnnData
        Annotated data with all required columns
    output_dir : Path
        Directory to save figures
    stage_col : str
        Column name for stage labels
    celltype_col : str
        Column name for cell type labels

    Returns
    -------
    paths : dict
        Dictionary mapping figure names to saved paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {}

    # Figure 1B: Sankey
    try:
        fig = figure_1b_sankey(
            adata, stage_col=stage_col, celltype_col=celltype_col,
            output_path=output_dir / "fig1b_sankey.png"
        )
        paths["fig1b_sankey"] = output_dir / "fig1b_sankey.png"
    except Exception as e:
        log.warning(f"Failed to generate Figure 1B: {e}")

    # Figure 1C: Stage UMAP
    if "X_umap" in adata.obsm:
        try:
            fig = figure_1c_umap(
                adata, stage_col=stage_col,
                output_path=output_dir / "fig1c_umap.png"
            )
            paths["fig1c_umap"] = output_dir / "fig1c_umap.png"
            plt.close(fig)
        except Exception as e:
            log.warning(f"Failed to generate Figure 1C: {e}")

    # Figure 3A: Cell type UMAP
    if "X_umap" in adata.obsm:
        try:
            fig = figure_3a_celltype_umap(
                adata, celltype_col=celltype_col,
                output_path=output_dir / "fig3a_celltype_umap.png"
            )
            paths["fig3a_celltype_umap"] = output_dir / "fig3a_celltype_umap.png"
            plt.close(fig)
        except Exception as e:
            log.warning(f"Failed to generate Figure 3A: {e}")

    log.info(f"Generated {len(paths)} LungPCA-style figures in {output_dir}")
    return paths
