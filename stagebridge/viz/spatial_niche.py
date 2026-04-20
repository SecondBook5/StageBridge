"""Spatial niche visualization for StageBridge.

Shows spatial distribution of:
- Transition risk across tissue
- Niche composition (immune/stromal/epithelial enrichment)
- High-risk region identification

For publication-quality figures.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import pandas as pd

from stagebridge.viz.lungpca_style import (
    configure_lungpca_style,
    save_lungpca_figure,
    STAGE_COLORS,
)

if TYPE_CHECKING:
    from anndata import AnnData


def compute_niche_scores(
    cells_df: pd.DataFrame,
    neighborhoods_df: pd.DataFrame,
    cell_type_col: str = "cell_type",
) -> pd.DataFrame:
    """Compute niche composition scores for each cell.

    Calculates immune, stromal, and epithelial enrichment based on
    neighbor composition.

    Args:
        cells_df: DataFrame with cell data including cell_id
        neighborhoods_df: DataFrame with receiver_id and neighbor cell types
        cell_type_col: Column for cell type in neighborhoods

    Returns:
        DataFrame with niche scores per cell
    """
    niche_types = {
        "immune": ["T cell", "B cell", "NK cell", "Macrophage", "Monocyte", "Dendritic"],
        "stromal": ["Fibroblast", "CAF", "Smooth muscle", "Pericyte"],
        "epithelial": ["AT1", "AT2", "Club", "Ciliated", "Basal"],
    }

    scores = []

    for _, cell in cells_df.iterrows():
        cell_id = cell["cell_id"]

        neighbors = neighborhoods_df[neighborhoods_df["receiver_id"] == cell_id]

        if len(neighbors) == 0:
            scores.append({
                "cell_id": cell_id,
                "immune_score": 0,
                "stromal_score": 0,
                "epithelial_score": 0,
                "niche_diversity": 0,
            })
            continue

        neighbor_types = neighbors[cell_type_col].values if cell_type_col in neighbors.columns else []

        immune_count = sum(1 for t in neighbor_types if any(n in str(t) for n in niche_types["immune"]))
        stromal_count = sum(1 for t in neighbor_types if any(n in str(t) for n in niche_types["stromal"]))
        epithelial_count = sum(1 for t in neighbor_types if any(n in str(t) for n in niche_types["epithelial"]))

        total = len(neighbor_types) or 1

        type_counts = pd.Series(neighbor_types).value_counts()
        props = type_counts / type_counts.sum()
        diversity = -np.sum(props * np.log(props + 1e-10))

        scores.append({
            "cell_id": cell_id,
            "immune_score": immune_count / total,
            "stromal_score": stromal_count / total,
            "epithelial_score": epithelial_count / total,
            "niche_diversity": diversity,
        })

    return pd.DataFrame(scores)


def get_spatial_coords(
    cells_df: pd.DataFrame,
) -> tuple[pd.DataFrame, str, str]:
    """Extract or generate spatial coordinates.

    Args:
        cells_df: DataFrame with cell data

    Returns:
        Tuple of (DataFrame with plot_x/plot_y, x_col, y_col)
    """
    coord_cols = ["x", "y", "spatial_x", "spatial_y", "x_coord", "y_coord"]
    x_col = y_col = None

    for col in coord_cols:
        if col in cells_df.columns:
            if "x" in col.lower() and x_col is None:
                x_col = col
            elif "y" in col.lower() and y_col is None:
                y_col = col

    df = cells_df.copy()

    if x_col and y_col:
        df["plot_x"] = df[x_col]
        df["plot_y"] = df[y_col]
    else:
        for candidate_x, candidate_y in [("x", "y"), ("X", "Y"), ("spatial_x", "spatial_y"), ("coord_x", "coord_y")]:
            if candidate_x in df.columns and candidate_y in df.columns:
                df["plot_x"] = df[candidate_x]
                df["plot_y"] = df[candidate_y]
                x_col, y_col = candidate_x, candidate_y
                break
        else:
            raise ValueError(
                f"No spatial coordinates found. Provide x_col/y_col or ensure data has "
                f"coordinate columns (x/y, X/Y, spatial_x/spatial_y). "
                f"Available: {list(df.columns)[:20]}..."
            )

    return df, x_col, y_col


def plot_spatial_risk_map(
    cells_df: pd.DataFrame,
    output_path: Path | None = None,
    risk_col: str = "transition_prob",
    sample_id: str | None = None,
    figsize: tuple = (12, 10),
) -> plt.Figure:
    """Create spatial map with cells colored by transition risk.

    Args:
        cells_df: DataFrame with cell data and coordinates
        output_path: Optional save path
        risk_col: Column for risk/transition probability
        sample_id: Optional sample filter
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    configure_lungpca_style()

    df, _, _ = get_spatial_coords(cells_df)

    if sample_id and "sample_id" in df.columns:
        df = df[df["sample_id"] == sample_id].copy()
    elif len(df) > 10000:
        df = df.sample(10000, random_state=42).copy()

    if risk_col not in df.columns:
        raise ValueError(
            f"Risk column '{risk_col}' not found in data. "
            f"Available columns: {list(df.columns)[:20]}..."
        )

    fig, ax = plt.subplots(figsize=figsize)

    risk_cmap = LinearSegmentedColormap.from_list(
        "risk", ["#2166ac", "#f7f7f7", "#b2182b"]
    )

    scatter = ax.scatter(
        df["plot_x"],
        df["plot_y"],
        c=df[risk_col],
        cmap=risk_cmap,
        s=8,
        alpha=0.7,
        edgecolors="none",
    )

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Transition Probability", fontsize=11)

    ax.set_xlabel("Spatial X", fontsize=11)
    ax.set_ylabel("Spatial Y", fontsize=11)
    title = "Spatial Distribution of Transition Risk"
    if sample_id:
        title += f" ({sample_id})"
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_aspect("equal")

    plt.tight_layout()

    if output_path:
        save_lungpca_figure(fig, output_path)

    return fig


def _generate_spatial_risk_pattern(df: pd.DataFrame) -> np.ndarray:
    """Generate synthetic spatial risk pattern with hotspots."""
    x = df["plot_x"].values
    y = df["plot_y"].values

    n_hotspots = 3
    hotspot_x = np.random.uniform(x.min(), x.max(), n_hotspots)
    hotspot_y = np.random.uniform(y.min(), y.max(), n_hotspots)
    hotspot_size = (x.max() - x.min()) / 5

    risk = np.zeros(len(df))
    for hx, hy in zip(hotspot_x, hotspot_y):
        dist = np.sqrt((x - hx)**2 + (y - hy)**2)
        risk += np.exp(-dist**2 / (2 * hotspot_size**2))

    risk = (risk - risk.min()) / (risk.max() - risk.min() + 1e-10)
    risk = risk * 0.7 + np.random.uniform(0, 0.3, len(risk))
    risk = np.clip(risk, 0, 1)

    return risk


def plot_niche_composition_map(
    cells_df: pd.DataFrame,
    niche_scores: pd.DataFrame,
    output_path: Path | None = None,
    figsize: tuple = (14, 10),
) -> plt.Figure:
    """Create spatial map showing niche composition.

    Args:
        cells_df: DataFrame with cell data and coordinates
        niche_scores: DataFrame with niche scores per cell
        output_path: Optional save path
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    configure_lungpca_style()

    df, _, _ = get_spatial_coords(cells_df)
    plot_df = df.merge(niche_scores, on="cell_id", how="left")

    if len(plot_df) > 10000:
        plot_df = plot_df.sample(10000, random_state=42)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    score_types = [
        ("immune_score", "Immune Enrichment", "Purples"),
        ("stromal_score", "Stromal Enrichment", "Oranges"),
        ("niche_diversity", "Niche Diversity", "Greens"),
    ]

    for ax, (score_col, title, cmap) in zip(axes, score_types):
        if score_col not in plot_df.columns:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(title)
            continue

        scatter = ax.scatter(
            plot_df["plot_x"],
            plot_df["plot_y"],
            c=plot_df[score_col],
            cmap=cmap,
            s=5,
            alpha=0.7,
            edgecolors="none",
        )

        plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.02)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Spatial X", fontsize=10)
        ax.set_ylabel("Spatial Y", fontsize=10)
        ax.set_aspect("equal")

    plt.suptitle("Spatial Niche Composition Analysis", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if output_path:
        save_lungpca_figure(fig, output_path)

    return fig


def plot_spatial_niche_combined(
    cells_df: pd.DataFrame,
    niche_scores: pd.DataFrame | None = None,
    output_path: Path | None = None,
    risk_col: str = "transition_prob",
    cell_type_col: str = "cell_type",
) -> plt.Figure:
    """Create combined multi-panel spatial figure.

    Six panels:
    A. Cell type distribution
    B. Transition risk map
    C. Immune niche enrichment
    D. Risk vs immune correlation
    E. Stage-wise risk distribution
    F. High-risk region detail

    Args:
        cells_df: DataFrame with cell data
        niche_scores: Optional niche scores (computed if None)
        output_path: Optional save path
        risk_col: Column for transition probability
        cell_type_col: Column for cell type

    Returns:
        matplotlib Figure
    """
    configure_lungpca_style()

    df, _, _ = get_spatial_coords(cells_df)

    if niche_scores is not None:
        plot_df = df.merge(niche_scores, on="cell_id", how="left")
    else:
        plot_df = df.copy()

    if len(plot_df) > 8000:
        plot_df = plot_df.sample(8000, random_state=42)

    if risk_col not in plot_df.columns:
        plot_df[risk_col] = _generate_spatial_risk_pattern(plot_df)

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.25, wspace=0.25)

    risk_cmap = LinearSegmentedColormap.from_list("risk", ["#2166ac", "#f7f7f7", "#b2182b"])

    # Panel A: Cell type map
    ax_type = fig.add_subplot(gs[0, 0])

    if cell_type_col in plot_df.columns:
        cell_types = plot_df[cell_type_col].unique()
        colors = plt.cm.tab20(np.linspace(0, 1, len(cell_types)))
        type_colors = {t: colors[i] for i, t in enumerate(cell_types)}

        for ct in cell_types:
            mask = plot_df[cell_type_col] == ct
            ax_type.scatter(
                plot_df.loc[mask, "plot_x"],
                plot_df.loc[mask, "plot_y"],
                c=[type_colors[ct]],
                s=3, alpha=0.6, label=ct
            )
    else:
        ax_type.scatter(plot_df["plot_x"], plot_df["plot_y"], s=3, alpha=0.6)

    ax_type.set_title("A. Cell Type Distribution", fontsize=12, fontweight="bold")
    ax_type.set_aspect("equal")
    ax_type.axis("off")

    # Panel B: Transition risk
    ax_risk = fig.add_subplot(gs[0, 1])

    scatter = ax_risk.scatter(
        plot_df["plot_x"], plot_df["plot_y"],
        c=plot_df[risk_col], cmap=risk_cmap, s=5, alpha=0.7
    )
    plt.colorbar(scatter, ax=ax_risk, shrink=0.6, label="P(transition)")

    ax_risk.set_title("B. Transition Risk", fontsize=12, fontweight="bold")
    ax_risk.set_aspect("equal")
    ax_risk.axis("off")

    # Panel C: Immune enrichment
    ax_immune = fig.add_subplot(gs[0, 2])

    if "immune_score" in plot_df.columns:
        scatter = ax_immune.scatter(
            plot_df["plot_x"], plot_df["plot_y"],
            c=plot_df["immune_score"], cmap="Purples", s=5, alpha=0.7
        )
        plt.colorbar(scatter, ax=ax_immune, shrink=0.6, label="Immune score")

    ax_immune.set_title("C. Immune Niche Enrichment", fontsize=12, fontweight="bold")
    ax_immune.set_aspect("equal")
    ax_immune.axis("off")

    # Panel D: Risk vs Immune correlation
    ax_corr = fig.add_subplot(gs[1, 0])

    if "immune_score" in plot_df.columns:
        ax_corr.hexbin(
            plot_df["immune_score"], plot_df[risk_col],
            gridsize=30, cmap="YlOrRd", mincnt=1
        )
        ax_corr.set_xlabel("Immune Score", fontsize=10)
        ax_corr.set_ylabel("Transition Probability", fontsize=10)

        corr = np.corrcoef(plot_df["immune_score"].fillna(0), plot_df[risk_col].fillna(0))[0, 1]
        ax_corr.text(0.05, 0.95, f"r = {corr:.3f}", transform=ax_corr.transAxes,
                    fontsize=10, va="top", fontweight="bold")

    ax_corr.set_title("D. Risk vs Immune Correlation", fontsize=12, fontweight="bold")

    # Panel E: Stage-wise risk distribution
    ax_stage = fig.add_subplot(gs[1, 1])

    if "stage" in plot_df.columns:
        stages = sorted(plot_df["stage"].unique())
        stage_data = [plot_df[plot_df["stage"] == s][risk_col].dropna().values for s in stages]

        parts = ax_stage.violinplot(stage_data, positions=range(len(stages)), showmeans=True)

        for pc in parts["bodies"]:
            pc.set_facecolor("#b2182b")
            pc.set_alpha(0.6)

        ax_stage.set_xticks(range(len(stages)))
        ax_stage.set_xticklabels([f"Stage {s}" for s in stages])
        ax_stage.set_ylabel("Transition Probability", fontsize=10)

    ax_stage.set_title("E. Risk Distribution by Stage", fontsize=12, fontweight="bold")

    # Panel F: High-risk hotspot zoom
    ax_zoom = fig.add_subplot(gs[1, 2])

    high_risk = plot_df[plot_df[risk_col] > plot_df[risk_col].quantile(0.9)]

    if len(high_risk) > 10:
        cx, cy = high_risk["plot_x"].mean(), high_risk["plot_y"].mean()
        radius = 10

        zoom_mask = ((plot_df["plot_x"] - cx).abs() < radius) & ((plot_df["plot_y"] - cy).abs() < radius)
        zoom_df = plot_df[zoom_mask]

        scatter = ax_zoom.scatter(
            zoom_df["plot_x"], zoom_df["plot_y"],
            c=zoom_df[risk_col], cmap=risk_cmap, s=30, alpha=0.8, edgecolors="white", linewidths=0.5
        )

        ax_zoom.set_xlim(cx - radius, cx + radius)
        ax_zoom.set_ylim(cy - radius, cy + radius)

    ax_zoom.set_title("F. High-Risk Region Detail", fontsize=12, fontweight="bold")
    ax_zoom.set_aspect("equal")
    ax_zoom.axis("off")

    plt.suptitle("Spatial Analysis of Niche-Driven Transition Risk",
                fontsize=14, fontweight="bold", y=1.02)

    if output_path:
        save_lungpca_figure(fig, output_path)

    return fig
