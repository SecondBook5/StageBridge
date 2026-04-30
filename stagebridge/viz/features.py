"""Publication-quality feature visualization for StageBridge.

Generates LungPCA-style figures for biological features.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stagebridge.viz.theme import (
    configure_publication_style,
    STAGE_COLORS,
    CELLTYPE_COLORS,
    save_figure,
)


def plot_feature_distributions(
    df: pd.DataFrame,
    features: list[str],
    stage_col: str = "stage",
    output_dir: Path | None = None,
    figname: str = "feature_distributions",
) -> plt.Figure:
    """Plot feature distributions by disease stage.

    Creates violin/box plots showing how biological features vary across stages.

    Args:
        df: DataFrame with feature columns and stage
        features: List of feature column names to plot
        stage_col: Column containing stage labels
        output_dir: Directory to save figure
        figname: Output filename

    Returns:
        Figure object
    """
    configure_publication_style()

    n_features = len(features)
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    stages = ["Normal", "Preinvasive", "Invasive"]
    stage_order = [s for s in stages if s in df[stage_col].values]

    for ax, feat in zip(axes, features):
        if feat not in df.columns:
            ax.set_visible(False)
            continue

        data = []
        labels = []
        colors = []

        for stage in stage_order:
            vals = df.loc[df[stage_col] == stage, feat].dropna()
            if len(vals) > 0:
                data.append(vals)
                labels.append(stage)
                colors.append(STAGE_COLORS.get(stage, "#999999"))

        if data:
            bp = ax.boxplot(data, labels=labels, patch_artist=True)
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

        ax.set_ylabel(feat.replace("_", " ").title())
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45)

    # Hide unused axes
    for ax in axes[len(features):]:
        ax.set_visible(False)

    fig.suptitle("Biological Features by Disease Stage", fontsize=16, fontweight="bold")
    plt.tight_layout()

    if output_dir:
        save_figure(fig, output_dir, figname)

    return fig


def plot_spatial_features(
    df: pd.DataFrame,
    features: list[str],
    x_col: str = "x_spatial",
    y_col: str = "y_spatial",
    donor_col: str = "donor_id",
    output_dir: Path | None = None,
    figname: str = "spatial_features",
    sample_donor: str | None = None,
) -> plt.Figure:
    """Plot spatial heatmaps of biological features.

    Args:
        df: DataFrame with spatial coordinates and features
        features: List of feature columns to plot
        x_col: Column for x coordinate
        y_col: Column for y coordinate
        donor_col: Column for donor/sample ID
        output_dir: Directory to save figure
        figname: Output filename
        sample_donor: Specific donor to plot (default: most cells)

    Returns:
        Figure object
    """
    configure_publication_style()

    # Select donor
    if sample_donor is None:
        sample_donor = df[donor_col].value_counts().index[0]

    donor_df = df[df[donor_col] == sample_donor]

    n_features = len(features)
    n_cols = min(2, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for ax, feat in zip(axes, features):
        if feat not in donor_df.columns:
            ax.set_visible(False)
            continue

        vals = donor_df[feat].fillna(0)
        sc = ax.scatter(
            donor_df[x_col],
            donor_df[y_col],
            c=vals,
            s=2,
            cmap="viridis",
            alpha=0.8,
            rasterized=True,
        )
        cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
        cbar.set_label(feat.replace("_", " ").title())

        ax.set_xlabel("Spatial X")
        ax.set_ylabel("Spatial Y")
        ax.set_title(f"{feat.replace('_', ' ').title()}")
        ax.set_aspect("equal")

    for ax in axes[len(features):]:
        ax.set_visible(False)

    fig.suptitle(f"Spatial Features - {sample_donor}", fontsize=16, fontweight="bold")
    plt.tight_layout()

    if output_dir:
        save_figure(fig, output_dir, figname)

    return fig


def plot_umap_features(
    df: pd.DataFrame,
    features: list[str],
    umap_cols: tuple[str, str] = ("UMAP1", "UMAP2"),
    output_dir: Path | None = None,
    figname: str = "umap_features",
    max_cells: int = 50000,
) -> plt.Figure:
    """Plot UMAP colored by biological features.

    Args:
        df: DataFrame with UMAP coordinates and features
        features: List of feature columns to plot
        umap_cols: Column names for UMAP coordinates
        output_dir: Directory to save figure
        figname: Output filename
        max_cells: Maximum cells to plot (subsampled if larger)

    Returns:
        Figure object
    """
    configure_publication_style()

    # Subsample if needed
    if len(df) > max_cells:
        df = df.sample(n=max_cells, random_state=42)

    n_features = len(features)
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    u1, u2 = umap_cols

    for ax, feat in zip(axes, features):
        if feat not in df.columns or u1 not in df.columns:
            ax.set_visible(False)
            continue

        vals = df[feat].fillna(0)

        # Use categorical colormap for cell types
        if df[feat].dtype == "object" or feat.startswith("cell_type"):
            unique_vals = df[feat].dropna().unique()
            color_map = {v: CELLTYPE_COLORS.get(v, f"C{i}") for i, v in enumerate(unique_vals)}
            colors = df[feat].map(color_map)

            for val in unique_vals[:15]:  # Limit legend
                mask = df[feat] == val
                ax.scatter(
                    df.loc[mask, u1],
                    df.loc[mask, u2],
                    c=color_map[val],
                    s=1,
                    alpha=0.6,
                    label=val[:20],
                    rasterized=True,
                )
            ax.legend(markerscale=5, fontsize=6, loc="upper right")
        else:
            sc = ax.scatter(
                df[u1],
                df[u2],
                c=vals,
                s=1,
                cmap="viridis",
                alpha=0.6,
                rasterized=True,
            )
            plt.colorbar(sc, ax=ax, shrink=0.6)

        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        ax.set_title(feat.replace("_", " ").title())
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes[len(features):]:
        ax.set_visible(False)

    plt.tight_layout()

    if output_dir:
        save_figure(fig, output_dir, figname)

    return fig


def plot_progression_panel(
    df: pd.DataFrame,
    umap_cols: tuple[str, str] = ("UMAP1", "UMAP2"),
    output_dir: Path | None = None,
    figname: str = "progression_panel",
) -> plt.Figure:
    """Create LungPCA-style 3-panel progression figure.

    Panel A: UMAP by stage
    Panel B: UMAP by pseudotime
    Panel C: UMAP by CytoTRACE

    Args:
        df: DataFrame with UMAP, stage, pseudotime, cytotrace
        umap_cols: UMAP coordinate columns
        output_dir: Output directory
        figname: Output filename

    Returns:
        Figure object
    """
    configure_publication_style()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    u1, u2 = umap_cols

    # Subsample for plotting
    if len(df) > 50000:
        df = df.sample(n=50000, random_state=42)

    # Panel A: Stage
    ax = axes[0]
    for stage in ["Normal", "Preinvasive", "Invasive"]:
        if stage in df["stage"].values:
            mask = df["stage"] == stage
            ax.scatter(
                df.loc[mask, u1],
                df.loc[mask, u2],
                c=STAGE_COLORS.get(stage, "#999999"),
                s=1,
                alpha=0.6,
                label=stage,
                rasterized=True,
            )
    ax.legend(markerscale=8, frameon=False)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title("Disease Stage", fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])

    # Panel B: Pseudotime
    ax = axes[1]
    if "pseudotime" in df.columns:
        valid = df["pseudotime"].notna()
        sc = ax.scatter(
            df.loc[valid, u1],
            df.loc[valid, u2],
            c=df.loc[valid, "pseudotime"],
            s=1,
            cmap="viridis",
            alpha=0.6,
            rasterized=True,
        )
        plt.colorbar(sc, ax=ax, shrink=0.6, label="Pseudotime")
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title("Pseudotime", fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])

    # Panel C: CytoTRACE (turbo colormap like LungPCA)
    ax = axes[2]
    if "cytotrace" in df.columns:
        sc = ax.scatter(
            df[u1],
            df[u2],
            c=df["cytotrace"],
            s=1,
            cmap="turbo",
            alpha=0.6,
            rasterized=True,
        )
        plt.colorbar(sc, ax=ax, shrink=0.6, label="CytoTRACE")
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title("CytoTRACE", fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()

    if output_dir:
        save_figure(fig, output_dir, figname)

    return fig
