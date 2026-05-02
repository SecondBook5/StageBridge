"""Visualization API for StageBridge.

Provides publication-quality plotting functions for StageBridge results,
following scanpy conventions.

Example usage:
    import stagebridge as sb

    # Plot embedding colored by stage
    sb.pl.embedding(adata, color="stage")

    # Plot velocity field
    sb.pl.flow_field(embeddings, velocities)

    # Plot attention heatmap
    sb.pl.niche_attention(model, cell_idx=0)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Colormap
from matplotlib.figure import Figure

if TYPE_CHECKING:
    pass

# Default color palette for stages
STAGE_COLORS = {
    "Normal": "#2ecc71",      # Green
    "Preinvasive": "#f39c12", # Orange
    "AAH": "#e67e22",         # Dark orange
    "AIS": "#f1c40f",         # Yellow
    "MIA": "#e74c3c",         # Red
    "Invasive": "#c0392b",    # Dark red
    "LUAD": "#8e44ad",        # Purple
}


def _setup_style():
    """Set publication-quality matplotlib style."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "axes.grid": False,
        "legend.frameon": False,
        "legend.fontsize": 10,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })


def embedding(
    embeddings: np.ndarray,
    stages: np.ndarray | Sequence[str] | None = None,
    colors: np.ndarray | Sequence[float] | None = None,
    method: Literal["umap", "pca", "phate", "tsne"] = "umap",
    stage_colors: dict[str, str] | None = None,
    cmap: str | Colormap = "viridis",
    title: str | None = None,
    point_size: int = 15,
    alpha: float = 0.7,
    show_legend: bool = True,
    figsize: tuple[float, float] = (8, 6),
    save_path: str | Path | None = None,
    show: bool = True,
    ax: plt.Axes | None = None,
    **embedding_kwargs,
) -> Figure:
    """Plot embedding colored by stage or continuous variable.

    Args:
        embeddings: Cell embeddings [N, D] (will be reduced to 2D)
        stages: Stage labels for coloring (categorical)
        colors: Continuous values for coloring (alternative to stages)
        method: Dimensionality reduction method
        stage_colors: Custom colors for stages
        cmap: Colormap for continuous coloring
        title: Plot title
        point_size: Size of scatter points
        alpha: Point transparency
        show_legend: Whether to show legend
        figsize: Figure size
        save_path: Path to save figure
        show: Whether to display figure
        ax: Existing axes to plot on
        **embedding_kwargs: Passed to embedding method

    Returns:
        Figure

    Example:
        import stagebridge as sb

        # Color by stage
        sb.pl.embedding(embeddings, stages=adata.obs["stage"])

        # Color by continuous variable
        sb.pl.embedding(embeddings, colors=adata.obs["pseudotime"])
    """
    _setup_style()

    # Compute 2D embedding
    if embeddings.shape[1] == 2:
        coords = embeddings
    else:
        coords = _compute_2d_embedding(embeddings, method, **embedding_kwargs)

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    # Plot points
    if stages is not None:
        # Categorical coloring
        stages = np.array(stages)
        unique_stages = np.unique(stages)

        if stage_colors is None:
            stage_colors = STAGE_COLORS

        for stage in unique_stages:
            mask = stages == stage
            color = stage_colors.get(stage, "#999999")
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                c=color,
                s=point_size,
                alpha=alpha,
                label=stage,
                rasterized=True,
            )

        if show_legend:
            ax.legend(loc="best", markerscale=1.5)

    elif colors is not None:
        # Continuous coloring
        sc = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=colors,
            s=point_size,
            alpha=alpha,
            cmap=cmap,
            rasterized=True,
        )
        plt.colorbar(sc, ax=ax)

    else:
        # No coloring
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c="#3498db",
            s=point_size,
            alpha=alpha,
            rasterized=True,
        )

    # Labels
    method_labels = {
        "umap": "UMAP",
        "pca": "PCA",
        "phate": "PHATE",
        "tsne": "t-SNE",
    }
    label = method_labels.get(method, method.upper())
    ax.set_xlabel(f"{label} 1")
    ax.set_ylabel(f"{label} 2")
    ax.set_xticks([])
    ax.set_yticks([])

    if title:
        ax.set_title(title, fontweight="bold")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def flow_field(
    embeddings: np.ndarray,
    velocities: np.ndarray,
    stages: np.ndarray | None = None,
    method: Literal["umap", "pca", "phate", "tsne"] = "umap",
    stage_colors: dict[str, str] | None = None,
    arrow_scale: float = 0.1,
    arrow_width: float = 0.003,
    n_arrows: int = 500,
    stream: bool = False,
    grid_density: int = 25,
    title: str | None = None,
    point_size: int = 10,
    alpha: float = 0.5,
    figsize: tuple[float, float] = (10, 8),
    save_path: str | Path | None = None,
    show: bool = True,
    **embedding_kwargs,
) -> Figure:
    """Plot velocity/flow field on embedding.

    Args:
        embeddings: Cell embeddings [N, D]
        velocities: Velocity vectors [N, D]
        stages: Stage labels for coloring points
        method: Embedding method
        stage_colors: Custom stage colors
        arrow_scale: Scale factor for arrows
        arrow_width: Width of arrows
        n_arrows: Number of arrows to show (subsampled)
        stream: If True, use streamplot instead of quiver
        grid_density: Grid density for streamplot
        title: Plot title
        point_size: Size of scatter points
        alpha: Point transparency
        figsize: Figure size
        save_path: Path to save figure
        show: Whether to display
        **embedding_kwargs: Passed to embedding method

    Returns:
        Figure

    Example:
        sb.pl.flow_field(
            embeddings,
            velocities,
            stages=adata.obs["stage"]
        )
    """
    _setup_style()

    # Compute 2D embedding
    if embeddings.shape[1] == 2:
        coords = embeddings
        vel_2d = velocities
    else:
        # Need to project velocities too
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        coords = pca.fit_transform(embeddings)
        vel_2d = pca.transform(embeddings + velocities) - coords

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Plot cells
    if stages is not None:
        stages = np.array(stages)
        unique_stages = np.unique(stages)
        if stage_colors is None:
            stage_colors = STAGE_COLORS

        for stage in unique_stages:
            mask = stages == stage
            color = stage_colors.get(stage, "#999999")
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                c=color,
                s=point_size,
                alpha=alpha,
                label=stage,
                rasterized=True,
            )
        ax.legend(loc="best")
    else:
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c="lightgrey",
            s=point_size,
            alpha=alpha,
            rasterized=True,
        )

    # Draw velocity field
    has_velocity = np.linalg.norm(vel_2d, axis=1) > 1e-6
    valid_idx = np.where(has_velocity)[0]

    if len(valid_idx) > n_arrows:
        selected = np.random.choice(valid_idx, n_arrows, replace=False)
    else:
        selected = valid_idx

    if stream:
        # Streamplot requires gridded data
        from scipy.interpolate import griddata

        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
        margin = 0.1
        x_range = x_max - x_min
        y_range = y_max - y_min

        xi = np.linspace(x_min - margin * x_range, x_max + margin * x_range, grid_density)
        yi = np.linspace(y_min - margin * y_range, y_max + margin * y_range, grid_density)
        Xi, Yi = np.meshgrid(xi, yi)

        U = griddata(coords[has_velocity], vel_2d[has_velocity, 0], (Xi, Yi), method="linear", fill_value=0)
        V = griddata(coords[has_velocity], vel_2d[has_velocity, 1], (Xi, Yi), method="linear", fill_value=0)

        speed = np.sqrt(U**2 + V**2)
        U_norm = np.where(speed > 0, U / speed, 0)
        V_norm = np.where(speed > 0, V / speed, 0)

        ax.streamplot(
            xi, yi, U_norm, V_norm,
            color="black", density=1.5, linewidth=0.8,
            arrowsize=1.2, zorder=2,
        )
    else:
        # Quiver plot
        ax.quiver(
            coords[selected, 0],
            coords[selected, 1],
            vel_2d[selected, 0],
            vel_2d[selected, 1],
            angles="xy",
            scale_units="xy",
            scale=1 / arrow_scale,
            width=arrow_width,
            color="black",
            alpha=0.7,
            zorder=2,
        )

    # Labels
    ax.set_xlabel("Embedding 1")
    ax.set_ylabel("Embedding 2")
    ax.set_xticks([])
    ax.set_yticks([])

    if title:
        ax.set_title(title, fontweight="bold")
    else:
        ax.set_title("Velocity Field", fontweight="bold")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def niche_attention(
    attention_weights: np.ndarray,
    token_names: list[str] | None = None,
    cell_idx: int | Sequence[int] | None = None,
    title: str | None = None,
    cmap: str = "Blues",
    figsize: tuple[float, float] = (10, 6),
    save_path: str | Path | None = None,
    show: bool = True,
) -> Figure:
    """Plot attention heatmap for niche tokens.

    Args:
        attention_weights: Attention weights [N, K] or [K] for single cell
        token_names: Names for each token (default: ring/reference names)
        cell_idx: Cell indices to show (default: average across all)
        title: Plot title
        cmap: Colormap
        figsize: Figure size
        save_path: Path to save figure
        show: Whether to display

    Returns:
        Figure

    Example:
        embeddings = model.embed_niches(neighborhoods)
        sb.pl.niche_attention(embeddings.attention_weights)
    """
    _setup_style()

    if token_names is None:
        token_names = [
            "Ring 1 (0-50um)",
            "Ring 2 (50-100um)",
            "Ring 3 (100-150um)",
            "Ring 4 (150-200um)",
            "HLCA Ref",
            "LuCA Ref",
            "Pathway",
            "Stats",
        ]

    # Handle different input shapes
    if attention_weights.ndim == 1:
        weights = attention_weights.reshape(1, -1)
        single_cell = True
    else:
        weights = attention_weights
        single_cell = False

    # Select cells
    if cell_idx is not None:
        if isinstance(cell_idx, int):
            cell_idx = [cell_idx]
        weights = weights[cell_idx]

    # Truncate token names if needed
    n_tokens = weights.shape[1]
    token_names = token_names[:n_tokens]

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    if single_cell or len(weights) == 1:
        # Bar plot for single cell
        ax.bar(range(n_tokens), weights[0], color=plt.cm.get_cmap(cmap)(0.6))
        ax.set_xticks(range(n_tokens))
        ax.set_xticklabels(token_names, rotation=45, ha="right")
        ax.set_ylabel("Attention Weight")
        ax.set_ylim(0, max(weights[0]) * 1.1)

    else:
        # Heatmap for multiple cells
        im = ax.imshow(weights, aspect="auto", cmap=cmap)
        plt.colorbar(im, ax=ax, label="Attention Weight")

        ax.set_xticks(range(n_tokens))
        ax.set_xticklabels(token_names, rotation=45, ha="right")
        ax.set_ylabel("Cell Index")

        if len(weights) <= 50:
            ax.set_yticks(range(len(weights)))
        else:
            ax.set_yticks([])

    if title:
        ax.set_title(title, fontweight="bold")
    else:
        ax.set_title("Niche Token Attention", fontweight="bold")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def trajectory(
    trajectories: np.ndarray,
    stages: np.ndarray | None = None,
    stage_colors: dict[str, str] | None = None,
    method: Literal["umap", "pca", "phate", "tsne"] = "pca",
    n_trajectories: int = 100,
    alpha: float = 0.3,
    linewidth: float = 1.0,
    title: str | None = None,
    show_endpoints: bool = True,
    figsize: tuple[float, float] = (10, 8),
    save_path: str | Path | None = None,
    show: bool = True,
    **embedding_kwargs,
) -> Figure:
    """Plot cell trajectories through latent space.

    Args:
        trajectories: Trajectories [N, T, D] from compute_transitions
        stages: Stage labels for start points
        stage_colors: Custom stage colors
        method: Embedding method
        n_trajectories: Max number of trajectories to show
        alpha: Line transparency
        linewidth: Line width
        title: Plot title
        show_endpoints: Mark start/end points
        figsize: Figure size
        save_path: Path to save figure
        show: Whether to display
        **embedding_kwargs: Passed to embedding method

    Returns:
        Figure

    Example:
        transitions = model.compute_transitions(embeddings, context)
        sb.pl.trajectory(transitions.trajectories)
    """
    _setup_style()

    n_cells, n_steps, dim = trajectories.shape

    # Flatten for embedding
    traj_flat = trajectories.reshape(-1, dim)

    if dim == 2:
        coords_flat = traj_flat
    else:
        coords_flat = _compute_2d_embedding(traj_flat, method, **embedding_kwargs)

    coords = coords_flat.reshape(n_cells, n_steps, 2)

    # Subsample if needed
    if n_cells > n_trajectories:
        idx = np.random.choice(n_cells, n_trajectories, replace=False)
        coords = coords[idx]
        if stages is not None:
            stages = np.array(stages)[idx]

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    if stage_colors is None:
        stage_colors = STAGE_COLORS

    # Plot trajectories
    for i in range(len(coords)):
        if stages is not None:
            color = stage_colors.get(stages[i], "#999999")
        else:
            color = "#3498db"

        ax.plot(
            coords[i, :, 0],
            coords[i, :, 1],
            c=color,
            alpha=alpha,
            linewidth=linewidth,
        )

        if show_endpoints:
            # Start point
            ax.scatter(coords[i, 0, 0], coords[i, 0, 1], c=color, s=20, marker="o", zorder=5)
            # End point
            ax.scatter(coords[i, -1, 0], coords[i, -1, 1], c=color, s=30, marker="^", zorder=5)

    # Legend for stages
    if stages is not None:
        unique_stages = np.unique(stages)
        for stage in unique_stages:
            color = stage_colors.get(stage, "#999999")
            ax.plot([], [], c=color, linewidth=2, label=stage)
        ax.legend(loc="best")

    ax.set_xlabel("Embedding 1")
    ax.set_ylabel("Embedding 2")
    ax.set_xticks([])
    ax.set_yticks([])

    if title:
        ax.set_title(title, fontweight="bold")
    else:
        ax.set_title("Cell Trajectories", fontweight="bold")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def stage_centroids(
    embeddings: np.ndarray,
    stages: np.ndarray,
    stage_colors: dict[str, str] | None = None,
    method: Literal["umap", "pca", "phate", "tsne"] = "umap",
    show_paths: bool = True,
    stage_order: list[str] | None = None,
    point_size: int = 15,
    centroid_size: int = 200,
    title: str | None = None,
    figsize: tuple[float, float] = (10, 8),
    save_path: str | Path | None = None,
    show: bool = True,
    **embedding_kwargs,
) -> Figure:
    """Plot embedding with stage centroids and progression path.

    Args:
        embeddings: Cell embeddings [N, D]
        stages: Stage labels
        stage_colors: Custom stage colors
        method: Embedding method
        show_paths: Draw arrows between stage centroids
        stage_order: Order of stages for path (default: alphabetical)
        point_size: Size of cell points
        centroid_size: Size of centroid markers
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        show: Whether to display
        **embedding_kwargs: Passed to embedding method

    Returns:
        Figure

    Example:
        sb.pl.stage_centroids(
            embeddings,
            stages=adata.obs["stage"],
            stage_order=["Normal", "Preinvasive", "Invasive"]
        )
    """
    _setup_style()

    # Compute 2D embedding
    if embeddings.shape[1] == 2:
        coords = embeddings
    else:
        coords = _compute_2d_embedding(embeddings, method, **embedding_kwargs)

    stages = np.array(stages)
    unique_stages = np.unique(stages)

    if stage_colors is None:
        stage_colors = STAGE_COLORS

    if stage_order is None:
        stage_order = sorted(unique_stages)
    else:
        stage_order = [s for s in stage_order if s in unique_stages]

    # Compute centroids
    centroids = {}
    for stage in stage_order:
        mask = stages == stage
        centroids[stage] = coords[mask].mean(axis=0)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Plot cells
    for stage in stage_order:
        mask = stages == stage
        color = stage_colors.get(stage, "#999999")
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=color,
            s=point_size,
            alpha=0.5,
            label=stage,
            rasterized=True,
        )

    # Plot path between centroids
    if show_paths and len(stage_order) > 1:
        centroid_array = np.array([centroids[s] for s in stage_order])

        for i in range(len(centroid_array) - 1):
            ax.annotate(
                "",
                xy=centroid_array[i + 1],
                xytext=centroid_array[i],
                arrowprops=dict(
                    arrowstyle="->",
                    color="black",
                    lw=2.5,
                    connectionstyle="arc3,rad=0.1",
                ),
                zorder=10,
            )

    # Plot centroids
    for stage in stage_order:
        color = stage_colors.get(stage, "#999999")
        c = centroids[stage]
        ax.scatter(
            c[0], c[1],
            c=color,
            s=centroid_size,
            edgecolors="black",
            linewidths=2,
            zorder=11,
        )

    ax.legend(loc="best", markerscale=1.5)
    ax.set_xlabel("Embedding 1")
    ax.set_ylabel("Embedding 2")
    ax.set_xticks([])
    ax.set_yticks([])

    if title:
        ax.set_title(title, fontweight="bold")
    else:
        ax.set_title("Stage Centroids", fontweight="bold")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def uncertainty(
    embeddings: np.ndarray,
    uncertainty_values: np.ndarray,
    stages: np.ndarray | None = None,
    method: Literal["umap", "pca", "phate", "tsne"] = "umap",
    cmap: str = "magma",
    percentile_cap: float = 95,
    point_size: int = 15,
    alpha: float = 0.8,
    title: str | None = None,
    figsize: tuple[float, float] = (10, 8),
    save_path: str | Path | None = None,
    show: bool = True,
    colorbar_label: str = "Uncertainty",
    **embedding_kwargs,
) -> Figure:
    """Plot embedding colored by prediction uncertainty.

    High uncertainty cells are typically in transitional states or heterogeneous
    niches where the model is less confident about the predicted trajectory.

    Args:
        embeddings: Cell embeddings [N, D]
        uncertainty_values: Per-cell uncertainty values [N]
        stages: Optional stage labels for panel comparison
        method: Embedding method (umap, pca, phate, tsne)
        cmap: Colormap for uncertainty
        percentile_cap: Cap values at this percentile for better visualization
        point_size: Size of scatter points
        alpha: Point transparency
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        show: Whether to display
        colorbar_label: Label for colorbar
        **embedding_kwargs: Passed to embedding method

    Returns:
        Figure

    Example:
        output = model.predict_with_uncertainty(neighborhoods)
        sb.pl.uncertainty(
            embeddings,
            output.uncertainty_scalar,
            stages=adata.obs["stage"]
        )
    """
    _setup_style()

    # Compute 2D embedding
    coords = _compute_2d_embedding(embeddings, method=method, **embedding_kwargs)

    # Cap extreme values for visualization
    cap_value = np.percentile(uncertainty_values, percentile_cap)
    uncertainty_capped = np.clip(uncertainty_values, 0, cap_value)

    if stages is None:
        # Single panel
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        scatter = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=uncertainty_capped,
            cmap=cmap,
            s=point_size,
            alpha=alpha,
            rasterized=True,
        )
        plt.colorbar(scatter, ax=ax, label=colorbar_label, shrink=0.8)

        ax.set_xlabel("Embedding 1")
        ax.set_ylabel("Embedding 2")
        ax.set_xticks([])
        ax.set_yticks([])

        if title:
            ax.set_title(title, fontweight="bold")
        else:
            ax.set_title("Prediction Uncertainty", fontweight="bold")

    else:
        # Two panels: stages + uncertainty
        fig, axes = plt.subplots(1, 2, figsize=(figsize[0] * 1.8, figsize[1]))

        # Panel 1: Stages
        stages = np.array(stages)
        unique_stages = np.unique(stages)

        for stage in unique_stages:
            mask = stages == stage
            color = STAGE_COLORS.get(stage, "#999999")
            axes[0].scatter(
                coords[mask, 0],
                coords[mask, 1],
                c=color,
                s=point_size,
                alpha=alpha,
                label=stage,
                rasterized=True,
            )
        axes[0].legend(loc="best")
        axes[0].set_xlabel("Embedding 1")
        axes[0].set_ylabel("Embedding 2")
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        axes[0].set_title("Disease Stage", fontweight="bold")

        # Panel 2: Uncertainty
        scatter = axes[1].scatter(
            coords[:, 0],
            coords[:, 1],
            c=uncertainty_capped,
            cmap=cmap,
            s=point_size,
            alpha=alpha,
            rasterized=True,
        )
        plt.colorbar(scatter, ax=axes[1], label=colorbar_label, shrink=0.8)
        axes[1].set_xlabel("Embedding 1")
        axes[1].set_ylabel("Embedding 2")
        axes[1].set_xticks([])
        axes[1].set_yticks([])

        if title:
            axes[1].set_title(title, fontweight="bold")
        else:
            axes[1].set_title("Prediction Uncertainty", fontweight="bold")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def uncertainty_by_stage(
    uncertainty_values: np.ndarray,
    stages: np.ndarray,
    stage_colors: dict[str, str] | None = None,
    stage_order: list[str] | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (8, 6),
    save_path: str | Path | None = None,
    show: bool = True,
) -> Figure:
    """Plot uncertainty distribution by disease stage.

    Useful for showing that transitional stages (e.g., Preinvasive) have
    higher uncertainty than stable stages (Normal, Invasive).

    Args:
        uncertainty_values: Per-cell uncertainty values [N]
        stages: Stage labels [N]
        stage_colors: Custom stage colors
        stage_order: Order of stages on x-axis
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        show: Whether to display

    Returns:
        Figure
    """
    _setup_style()

    stages = np.array(stages)
    if stage_colors is None:
        stage_colors = STAGE_COLORS

    if stage_order is None:
        stage_order = ["Normal", "Preinvasive", "Invasive"]
        stage_order = [s for s in stage_order if s in np.unique(stages)]

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    box_data = []
    positions = []
    colors = []

    for i, stage in enumerate(stage_order):
        mask = stages == stage
        if mask.sum() > 0:
            box_data.append(uncertainty_values[mask])
            positions.append(i)
            colors.append(stage_colors.get(stage, "#999999"))

    bp = ax.boxplot(box_data, positions=positions, patch_artist=True, widths=0.6)

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticks(positions)
    ax.set_xticklabels(stage_order)
    ax.set_ylabel("Prediction Uncertainty")
    ax.set_xlabel("Disease Stage")

    if title:
        ax.set_title(title, fontweight="bold")
    else:
        ax.set_title("Uncertainty by Stage", fontweight="bold")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def _compute_2d_embedding(
    data: np.ndarray,
    method: str = "umap",
    **kwargs,
) -> np.ndarray:
    """Compute 2D embedding."""
    from sklearn.preprocessing import StandardScaler

    # Standardize
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    if method == "pca":
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, random_state=42)
        return reducer.fit_transform(data_scaled)

    elif method == "umap":
        try:
            import umap
        except ImportError:
            raise ImportError("Install umap-learn: pip install umap-learn")

        reducer = umap.UMAP(
            n_components=2,
            random_state=42,
            n_neighbors=kwargs.get("n_neighbors", 15),
            min_dist=kwargs.get("min_dist", 0.1),
        )
        return reducer.fit_transform(data_scaled)

    elif method == "tsne":
        from sklearn.manifold import TSNE
        reducer = TSNE(
            n_components=2,
            random_state=42,
            perplexity=kwargs.get("perplexity", 30),
        )
        return reducer.fit_transform(data_scaled)

    elif method == "phate":
        try:
            import phate
        except ImportError:
            raise ImportError("Install phate: pip install phate")

        reducer = phate.PHATE(
            n_components=2,
            random_state=42,
            knn=kwargs.get("knn", 5),
        )
        return reducer.fit_transform(data_scaled)

    else:
        raise ValueError(f"Unknown method: {method}")


# Module-level namespace for scanpy-style access (sb.pl.embedding)
class PlottingNamespace:
    """Namespace for plotting functions (sb.pl.*)."""

    embedding = staticmethod(embedding)
    flow_field = staticmethod(flow_field)
    niche_attention = staticmethod(niche_attention)
    trajectory = staticmethod(trajectory)
    stage_centroids = staticmethod(stage_centroids)
    uncertainty = staticmethod(uncertainty)
    uncertainty_by_stage = staticmethod(uncertainty_by_stage)


pl = PlottingNamespace()
