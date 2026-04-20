"""Dual-reference trajectory visualization for StageBridge.

Shows:
- 3D embedding trajectories with flow vectors
- HLCA vs LuCA reference contributions
- Trajectory paths through fused embedding space

For publication-quality figures.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from stagebridge.viz.lungpca_style import (
    configure_lungpca_style,
    save_lungpca_figure,
    STAGE_COLORS,
)

if TYPE_CHECKING:
    from anndata import AnnData


def compute_reduced_embedding(
    embedding_df: pd.DataFrame,
    method: str = "umap",
    n_components: int = 3,
    embedding_prefix: str = "fused_",
    n_samples: int = 10000,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Compute dimensionality reduction on embeddings.

    Args:
        embedding_df: DataFrame with embedding columns
        method: Reduction method (umap, tsne, pca)
        n_components: Target dimensions
        embedding_prefix: Prefix for embedding columns
        n_samples: Max samples to use

    Returns:
        Tuple of (reduced coordinates, sampled DataFrame)
    """
    embedding_cols = [c for c in embedding_df.columns if c.startswith(embedding_prefix)]

    if not embedding_cols:
        embedding_cols = [c for c in embedding_df.columns if c.startswith("embedding_")]

    if not embedding_cols:
        raise ValueError(
            f"No embedding columns found with prefix '{embedding_prefix}' or 'embedding_'. "
            f"Available columns: {list(embedding_df.columns)[:20]}..."
        )

    if len(embedding_df) > n_samples:
        sample_df = embedding_df.sample(n_samples, random_state=42)
    else:
        sample_df = embedding_df.copy()

    X = sample_df[embedding_cols].values

    # Filter out rows with NaN values
    nan_mask = ~np.isnan(X).any(axis=1)
    if nan_mask.sum() < len(X):
        print(f"Warning: Filtering {len(X) - nan_mask.sum()} rows with NaN values")
        X = X[nan_mask]
        sample_df = sample_df.iloc[nan_mask].copy()

    if method == "umap":
        try:
            import umap
            reducer = umap.UMAP(n_components=n_components, random_state=42, n_neighbors=30, min_dist=0.3)
            X_reduced = reducer.fit_transform(X)
        except ImportError:
            reducer = PCA(n_components=n_components, random_state=42)
            X_reduced = reducer.fit_transform(X)
    elif method == "tsne":
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=min(n_components, 3), random_state=42, perplexity=30)
        X_reduced = reducer.fit_transform(X)
    else:
        reducer = PCA(n_components=n_components, random_state=42)
        X_reduced = reducer.fit_transform(X)

    return X_reduced, sample_df


def plot_3d_trajectory(
    X_reduced: np.ndarray,
    sample_df: pd.DataFrame,
    output_path: Path | None = None,
    stage_col: str = "stage",
    figsize: tuple = (12, 10),
) -> plt.Figure:
    """Create 3D trajectory plot with stage coloring.

    Args:
        X_reduced: Reduced coordinates (N, 3)
        sample_df: DataFrame with metadata
        output_path: Optional save path
        stage_col: Stage column name
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    configure_lungpca_style()

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    if stage_col in sample_df.columns:
        stages = sorted(sample_df[stage_col].unique())
        colors = plt.cm.viridis(np.linspace(0, 1, len(stages)))
        stage_colors = {s: colors[i] for i, s in enumerate(stages)}

        for stage in stages:
            mask = sample_df[stage_col] == stage
            ax.scatter(
                X_reduced[mask, 0],
                X_reduced[mask, 1],
                X_reduced[mask, 2],
                c=[stage_colors[stage]],
                s=10,
                alpha=0.6,
                label=f"Stage {stage}",
            )

        centroids = []
        for stage in stages:
            mask = sample_df[stage_col] == stage
            centroid = X_reduced[mask].mean(axis=0)
            centroids.append(centroid)

        centroids = np.array(centroids)

        for i in range(len(centroids) - 1):
            ax.quiver(
                centroids[i, 0], centroids[i, 1], centroids[i, 2],
                centroids[i+1, 0] - centroids[i, 0],
                centroids[i+1, 1] - centroids[i, 1],
                centroids[i+1, 2] - centroids[i, 2],
                color="red", arrow_length_ratio=0.15, linewidth=2.5
            )

        ax.legend(loc="upper left", fontsize=9)
    else:
        ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], s=10, alpha=0.6)

    ax.set_xlabel("UMAP 1", fontsize=10)
    ax.set_ylabel("UMAP 2", fontsize=10)
    ax.set_zlabel("UMAP 3", fontsize=10)
    ax.set_title("3D Embedding Trajectory Across Disease Stages",
                fontsize=14, fontweight="bold")

    plt.tight_layout()

    if output_path:
        save_lungpca_figure(fig, output_path)

    return fig


def plot_reference_contribution(
    cells_df: pd.DataFrame,
    output_path: Path | None = None,
    figsize: tuple = (14, 6),
) -> plt.Figure:
    """Create plot showing HLCA vs LuCA embedding contributions.

    Args:
        cells_df: DataFrame with HLCA and LuCA embedding columns
        output_path: Optional save path
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    configure_lungpca_style()

    hlca_cols = [c for c in cells_df.columns if c.startswith("hlca_")]
    luca_cols = [c for c in cells_df.columns if c.startswith("luca_")]

    if not hlca_cols or not luca_cols:
        raise ValueError(
            f"Missing reference embedding columns. Need hlca_* and luca_* columns. "
            f"Found hlca: {len(hlca_cols)}, luca: {len(luca_cols)}. "
            f"Available: {[c for c in cells_df.columns if 'hlca' in c.lower() or 'luca' in c.lower()]}"
        )

    if len(cells_df) > 5000:
        plot_df = cells_df.sample(5000, random_state=42)
    else:
        plot_df = cells_df.copy()

    # Filter rows with NaN in HLCA or LuCA columns
    hlca_valid = ~plot_df[hlca_cols].isna().any(axis=1)
    luca_valid = ~plot_df[luca_cols].isna().any(axis=1)
    valid_mask = hlca_valid & luca_valid
    if valid_mask.sum() < len(plot_df):
        print(f"  Filtering {len(plot_df) - valid_mask.sum()} rows with NaN in reference embeddings")
        plot_df = plot_df[valid_mask].copy()

    if len(plot_df) == 0:
        print("  ERROR: No valid rows after NaN filtering")
        return

    hlca_var = plot_df[hlca_cols].var().sum()
    luca_var = plot_df[luca_cols].var().sum()
    total_var = hlca_var + luca_var

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Panel A: Variance contribution
    ax_var = axes[0]
    bars = ax_var.bar(["HLCA\n(Healthy)", "LuCA\n(Cancer)"],
                     [hlca_var / total_var * 100, luca_var / total_var * 100],
                     color=["#2166ac", "#b2182b"], edgecolor="black", linewidth=1)
    ax_var.set_ylabel("Variance Contribution (%)", fontsize=11)
    ax_var.set_title("A. Reference Contribution", fontsize=12, fontweight="bold")
    ax_var.set_ylim(0, 100)

    for bar in bars:
        height = bar.get_height()
        ax_var.text(bar.get_x() + bar.get_width()/2, height + 1,
                   f"{height:.1f}%", ha="center", va="bottom", fontsize=10)

    # Panel B: PCA of HLCA
    ax_hlca = axes[1]
    pca_hlca = PCA(n_components=2, random_state=42)
    hlca_2d = pca_hlca.fit_transform(plot_df[hlca_cols].values)

    if "stage" in plot_df.columns:
        stages = sorted(plot_df["stage"].unique())
        colors = plt.cm.viridis(np.linspace(0, 1, len(stages)))
        for i, s in enumerate(stages):
            mask = plot_df["stage"] == s
            ax_hlca.scatter(hlca_2d[mask, 0], hlca_2d[mask, 1], c=[colors[i]], s=5, alpha=0.5, label=f"Stage {s}")
        ax_hlca.legend(fontsize=8)
    else:
        ax_hlca.scatter(hlca_2d[:, 0], hlca_2d[:, 1], s=5, alpha=0.5, c="#2166ac")

    ax_hlca.set_xlabel("HLCA PC1", fontsize=10)
    ax_hlca.set_ylabel("HLCA PC2", fontsize=10)
    ax_hlca.set_title("B. HLCA Embedding (Healthy Reference)", fontsize=12, fontweight="bold")

    # Panel C: PCA of LuCA
    ax_luca = axes[2]
    pca_luca = PCA(n_components=2, random_state=42)
    luca_2d = pca_luca.fit_transform(plot_df[luca_cols].values)

    if "stage" in plot_df.columns:
        for i, s in enumerate(stages):
            mask = plot_df["stage"] == s
            ax_luca.scatter(luca_2d[mask, 0], luca_2d[mask, 1], c=[colors[i]], s=5, alpha=0.5, label=f"Stage {s}")
        ax_luca.legend(fontsize=8)
    else:
        ax_luca.scatter(luca_2d[:, 0], luca_2d[:, 1], s=5, alpha=0.5, c="#b2182b")

    ax_luca.set_xlabel("LuCA PC1", fontsize=10)
    ax_luca.set_ylabel("LuCA PC2", fontsize=10)
    ax_luca.set_title("C. LuCA Embedding (Cancer Reference)", fontsize=12, fontweight="bold")

    plt.suptitle("Dual-Reference Embedding Analysis", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if output_path:
        save_lungpca_figure(fig, output_path)

    return fig


def plot_flow_field(
    X_reduced: np.ndarray,
    sample_df: pd.DataFrame,
    output_path: Path | None = None,
    stage_col: str = "stage",
    figsize: tuple = (12, 10),
) -> plt.Figure:
    """Create 2D embedding with flow field vectors.

    Args:
        X_reduced: Reduced coordinates (N, 2+)
        sample_df: DataFrame with metadata
        output_path: Optional save path
        stage_col: Stage column name
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    configure_lungpca_style()

    fig, ax = plt.subplots(figsize=figsize)

    X_2d = X_reduced[:, :2]

    if stage_col in sample_df.columns:
        stages = sorted(sample_df[stage_col].unique())
        colors = plt.cm.viridis(np.linspace(0, 1, len(stages)))
        stage_colors = {s: colors[i] for i, s in enumerate(stages)}

        for stage in stages:
            mask = (sample_df[stage_col] == stage).values
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=[stage_colors[stage]],
                      s=8, alpha=0.4, label=f"Stage {stage}")

        x_range = X_2d[:, 0].max() - X_2d[:, 0].min()
        y_range = X_2d[:, 1].max() - X_2d[:, 1].min()

        grid_x = np.linspace(X_2d[:, 0].min() - 0.1*x_range, X_2d[:, 0].max() + 0.1*x_range, 20)
        grid_y = np.linspace(X_2d[:, 1].min() - 0.1*y_range, X_2d[:, 1].max() + 0.1*y_range, 20)

        U = np.zeros((len(grid_y), len(grid_x)))
        V = np.zeros((len(grid_y), len(grid_x)))

        for i, gx in enumerate(grid_x):
            for j, gy in enumerate(grid_y):
                dist = np.sqrt((X_2d[:, 0] - gx)**2 + (X_2d[:, 1] - gy)**2)
                radius = 0.1 * max(x_range, y_range)
                nearby = dist < radius

                if nearby.sum() < 5:
                    continue

                nearby_stages = sample_df.loc[nearby.values if hasattr(nearby, 'values') else nearby, stage_col].values
                nearby_pos = X_2d[nearby]

                for s_idx, stage in enumerate(stages[:-1]):
                    next_stage = stages[s_idx + 1]

                    current_mask = nearby_stages == stage
                    next_mask = nearby_stages == next_stage

                    if current_mask.sum() > 0 and next_mask.sum() > 0:
                        current_center = nearby_pos[current_mask].mean(axis=0)
                        next_center = nearby_pos[next_mask].mean(axis=0)

                        direction = next_center - current_center
                        weight = current_mask.sum() / nearby.sum()

                        U[j, i] += direction[0] * weight
                        V[j, i] += direction[1] * weight

        magnitude = np.sqrt(U**2 + V**2)
        magnitude[magnitude == 0] = 1
        U = U / magnitude
        V = V / magnitude

        X_grid, Y_grid = np.meshgrid(grid_x, grid_y)
        ax.quiver(X_grid, Y_grid, U, V, color="gray", alpha=0.6, scale=25, width=0.003)

        ax.legend(loc="upper right", fontsize=10)
    else:
        ax.scatter(X_2d[:, 0], X_2d[:, 1], s=8, alpha=0.4)

    ax.set_xlabel("UMAP 1", fontsize=11)
    ax.set_ylabel("UMAP 2", fontsize=11)
    ax.set_title("Cell State Embedding with Transition Flow Field",
                fontsize=14, fontweight="bold")

    plt.tight_layout()

    if output_path:
        save_lungpca_figure(fig, output_path)

    return fig


def plot_dual_reference_combined(
    cells_df: pd.DataFrame,
    output_path: Path | None = None,
    stage_col: str = "stage",
    cell_type_col: str = "cell_type",
    n_samples: int = 10000,
) -> plt.Figure:
    """Create combined multi-panel embedding figure.

    Four panels:
    A. 3D trajectory
    B. 2D with flow arrows
    C. Cell type distribution
    D. Early vs late stage density

    Args:
        cells_df: DataFrame with cell data and embeddings
        output_path: Optional save path
        stage_col: Stage column
        cell_type_col: Cell type column
        n_samples: Max samples

    Returns:
        matplotlib Figure
    """
    configure_lungpca_style()

    X_reduced, sample_df = compute_reduced_embedding(cells_df, n_samples=n_samples)

    fig = plt.figure(figsize=(16, 12))

    # Panel A: 3D view
    ax_3d = fig.add_subplot(2, 2, 1, projection="3d")

    if stage_col in sample_df.columns:
        stages = sorted(sample_df[stage_col].unique())
        colors = plt.cm.viridis(np.linspace(0, 1, len(stages)))

        for i, stage in enumerate(stages):
            mask = sample_df[stage_col] == stage
            ax_3d.scatter(X_reduced[mask, 0], X_reduced[mask, 1], X_reduced[mask, 2],
                         c=[colors[i]], s=5, alpha=0.5, label=f"Stage {stage}")

        centroids = np.array([X_reduced[sample_df[stage_col] == s].mean(axis=0) for s in stages])
        ax_3d.plot(centroids[:, 0], centroids[:, 1], centroids[:, 2], 'r-', linewidth=3, label="Trajectory")
        ax_3d.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c="red", s=100, marker="*")

        ax_3d.legend(fontsize=8)

    ax_3d.set_xlabel("UMAP 1")
    ax_3d.set_ylabel("UMAP 2")
    ax_3d.set_zlabel("UMAP 3")
    ax_3d.set_title("A. 3D Trajectory", fontsize=12, fontweight="bold")

    # Panel B: 2D with flow
    ax_flow = fig.add_subplot(2, 2, 2)
    X_2d = X_reduced[:, :2]

    if stage_col in sample_df.columns:
        for i, stage in enumerate(stages):
            mask = sample_df[stage_col] == stage
            ax_flow.scatter(X_2d[mask, 0], X_2d[mask, 1], c=[colors[i]], s=5, alpha=0.4, label=f"Stage {stage}")

        centroids_2d = np.array([X_2d[sample_df[stage_col] == s].mean(axis=0) for s in stages])
        for i in range(len(centroids_2d) - 1):
            ax_flow.annotate("", centroids_2d[i+1], centroids_2d[i],
                           arrowprops=dict(arrowstyle="->", color="red", lw=2.5))

        ax_flow.legend(fontsize=8)

    ax_flow.set_xlabel("UMAP 1")
    ax_flow.set_ylabel("UMAP 2")
    ax_flow.set_title("B. 2D Embedding with Flow", fontsize=12, fontweight="bold")

    # Panel C: Cell type coloring
    ax_type = fig.add_subplot(2, 2, 3)

    if cell_type_col in sample_df.columns:
        cell_types = sample_df[cell_type_col].value_counts().head(10).index.tolist()
        type_colors = plt.cm.tab10(np.linspace(0, 1, len(cell_types)))

        for i, ct in enumerate(cell_types):
            mask = sample_df[cell_type_col] == ct
            ax_type.scatter(X_2d[mask, 0], X_2d[mask, 1], c=[type_colors[i]], s=5, alpha=0.5, label=ct)

        ax_type.legend(fontsize=7, ncol=2, loc="upper right")

    ax_type.set_xlabel("UMAP 1")
    ax_type.set_ylabel("UMAP 2")
    ax_type.set_title("C. Cell Type Distribution", fontsize=12, fontweight="bold")

    # Panel D: Stage density
    ax_density = fig.add_subplot(2, 2, 4)

    if stage_col in sample_df.columns and len(stages) > 1:
        early_mask = sample_df[stage_col] == stages[0]
        late_mask = sample_df[stage_col] == stages[-1]

        ax_density.hexbin(X_2d[early_mask, 0], X_2d[early_mask, 1], gridsize=25, cmap="Blues", alpha=0.6)
        ax_density.hexbin(X_2d[late_mask, 0], X_2d[late_mask, 1], gridsize=25, cmap="Reds", alpha=0.4)

        ax_density.legend(handles=[
            mpatches.Patch(facecolor="blue", alpha=0.6, label=f"Stage {stages[0]}"),
            mpatches.Patch(facecolor="red", alpha=0.4, label=f"Stage {stages[-1]}")
        ], fontsize=9)

    ax_density.set_xlabel("UMAP 1")
    ax_density.set_ylabel("UMAP 2")
    ax_density.set_title("D. Early vs Late Stage Density", fontsize=12, fontweight="bold")

    plt.suptitle("Dual-Reference Embedding Analysis: Cell State Trajectories",
                fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()

    if output_path:
        save_lungpca_figure(fig, output_path)

    return fig
