"""Publication-quality trajectory visualization for StageBridge.

Implements the visual style for dynamic trajectory analysis:
- Temporal evolution panels (density clouds over time)
- Fate probability coloring on embeddings
- Single-cell trajectory plots with velocity arrows
- Dynamic driver gene heatmaps with temporal clustering
- Gene expression dynamics along trajectories
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.animation import FuncAnimation
from scipy.cluster.hierarchy import linkage, leaves_list, fcluster

if TYPE_CHECKING:
    from stagebridge.interpretation.dynamics import (
        FateProbability,
        DynamicDriverResult,
    )


def set_trajectory_style():
    """Set matplotlib style for trajectory figures."""
    plt.rcParams.update({
        'axes.axisbelow': False,
        'axes.edgecolor': 'lightgrey',
        'axes.facecolor': 'None',
        'axes.grid': False,
        'axes.labelcolor': 'dimgrey',
        'axes.spines.right': False,
        'axes.spines.top': False,
        'figure.facecolor': 'white',
        'lines.solid_capstyle': 'round',
        'patch.edgecolor': 'w',
        'patch.force_edgecolor': True,
        'text.color': 'dimgrey',
        'xtick.bottom': False,
        'xtick.color': 'dimgrey',
        'xtick.direction': 'out',
        'xtick.top': False,
        'ytick.color': 'dimgrey',
        'ytick.direction': 'out',
        'ytick.left': False,
        'ytick.right': False,
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'svg.fonttype': 'none',
        'pdf.fonttype': 42,
    })


def plot_temporal_evolution(
    data_pca: np.ndarray,
    time_labels: np.ndarray,
    interpolated_points: np.ndarray | None = None,
    interpolated_colors: list[str] | None = None,
    figsize: tuple[float, float] = (15, 10),
    n_cols: int = 5,
    cmap: str = 'plasma',
    spot_size: int = 5,
    alpha_bg: float = 0.1,
    alpha_fg: float = 0.8,
    title: str | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Plot temporal evolution as panel grid.

    Shows data distribution at each timepoint with optional
    interpolated trajectory overlay.

    Args:
        data_pca: PCA coordinates (n_cells, 2+)
        time_labels: Time/stage labels per cell
        interpolated_points: Optional interpolated trajectory (n_steps, n_cells, 2)
        interpolated_colors: Colors for each interpolated lineage
        figsize: Figure size
        n_cols: Number of columns in grid
        cmap: Colormap for time coloring
        spot_size: Point size
        alpha_bg: Background alpha
        alpha_fg: Foreground alpha
        title: Overall title
        save_path: Path to save
        show: Whether to display

    Returns:
        matplotlib Figure
    """
    set_trajectory_style()

    unique_times = np.unique(time_labels)
    n_times = len(unique_times)
    n_rows = int(np.ceil(n_times / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_times > 1 else [axes]

    colormap = plt.cm.get_cmap(cmap)
    time_colors = {t: colormap(i / (n_times - 1)) for i, t in enumerate(unique_times)}

    for ax_idx, time_val in enumerate(unique_times):
        ax = axes[ax_idx]

        for t in unique_times:
            mask = time_labels == t
            color = time_colors[t]
            alpha = alpha_fg if t == time_val else alpha_bg
            ax.scatter(
                data_pca[mask, 0],
                data_pca[mask, 1],
                c=[color],
                s=spot_size,
                alpha=alpha,
            )

        if interpolated_points is not None and interpolated_colors is not None:
            step_idx = int(ax_idx * len(interpolated_points) / n_times)
            step_idx = min(step_idx, len(interpolated_points) - 1)

            for lineage_idx, color in enumerate(interpolated_colors):
                if len(interpolated_points.shape) == 3:
                    points = interpolated_points[step_idx]
                else:
                    points = interpolated_points[step_idx]
                ax.scatter(points[:, 0], points[:, 1], c=color, s=spot_size * 2, alpha=0.8)

        ax.set_title(f'{time_val}', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0)

    for ax_idx in range(n_times, len(axes)):
        axes[ax_idx].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=16, y=1.02)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()

    return fig


def plot_fate_probability(
    data_pca: np.ndarray,
    fate_probs: "FateProbability",
    stage_labels: np.ndarray,
    target_stages: list[str],
    target_colors: dict[str, str],
    figsize: tuple[float, float] = (8, 6),
    cmap: str = 'viridis',
    spot_size: int = 15,
    plot_type: Literal['weights', 'assigned'] = 'weights',
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Plot fate probability on PCA embedding.

    Args:
        data_pca: PCA coordinates
        fate_probs: FateProbability object
        stage_labels: Stage labels for all cells
        target_stages: Target stage names
        target_colors: Colors for target stages
        figsize: Figure size
        cmap: Colormap for probability gradient
        spot_size: Point size
        plot_type: 'weights' for probability gradient, 'assigned' for discrete fates
        save_path: Path to save
        show: Whether to display

    Returns:
        matplotlib Figure
    """
    set_trajectory_style()

    fig, ax = plt.subplots(figsize=figsize)

    if plot_type == 'weights' and len(target_stages) > 0:
        weights = fate_probs.stage_probs[target_stages[0]]
        sc = ax.scatter(
            data_pca[:len(weights), 0],
            data_pca[:len(weights), 1],
            c=weights,
            cmap=cmap,
            s=spot_size,
            alpha=0.8,
        )
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(f'{target_stages[0]} Probability')

    else:
        fate_colors = {
            'Undetermined': 'grey',
            'Intermediate': 'yellow',
            'Low_Confidence': 'lightgrey',
        }
        fate_colors.update(target_colors)

        for fate in np.unique(fate_probs.assigned_fate):
            mask = fate_probs.assigned_fate == fate
            color = fate_colors.get(fate, 'grey')
            ax.scatter(
                data_pca[mask, 0],
                data_pca[mask, 1],
                c=color,
                s=spot_size,
                alpha=0.8,
                label=fate,
            )
        ax.legend(loc='best')

    for stage, color in target_colors.items():
        mask = stage_labels == stage
        ax.scatter(
            data_pca[mask, 0],
            data_pca[mask, 1],
            c=color,
            s=spot_size,
            alpha=0.2,
        )

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()

    return fig


def plot_single_cell_trajectories(
    data_pca: np.ndarray,
    trajectories: np.ndarray,
    stage_labels: np.ndarray,
    stage_colors: dict[str, str],
    n_trajectories: int = 30,
    figsize: tuple[float, float] = (8, 6),
    traj_color: str = 'orange',
    traj_alpha: float = 0.5,
    traj_width: float = 1.5,
    avg_width: float = 3.0,
    arrow_size: float = 0.5,
    spot_size: int = 5,
    show_avg: bool = True,
    cmap: str = 'plasma',
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Plot single-cell trajectories with average path.

    Args:
        data_pca: Background PCA coordinates
        trajectories: Trajectory array (n_steps, n_cells, 2)
        stage_labels: Stage labels for background
        stage_colors: Colors for stages
        n_trajectories: Number of individual trajectories to show
        figsize: Figure size
        traj_color: Color for individual trajectories
        traj_alpha: Alpha for individual trajectories
        traj_width: Line width for trajectories
        avg_width: Line width for average trajectory
        arrow_size: Arrow head size
        spot_size: Background point size
        show_avg: Whether to show average trajectory
        cmap: Colormap for time coloring
        save_path: Path to save
        show: Whether to display

    Returns:
        matplotlib Figure
    """
    set_trajectory_style()

    fig, axes = plt.subplots(1, 2, figsize=(figsize[0] * 2, figsize[1]))

    for ax_idx, (ax, title) in enumerate(zip(axes, ['Gene Expression Manifold', 'Latent Manifold'])):
        colormap = plt.cm.get_cmap(cmap)

        for stage, color in stage_colors.items():
            mask = stage_labels == stage
            ax.scatter(
                data_pca[mask, 0],
                data_pca[mask, 1],
                c=color,
                s=spot_size,
                alpha=0.2,
            )

        n_show = min(n_trajectories, trajectories.shape[1])
        indices = np.random.choice(trajectories.shape[1], n_show, replace=False)

        for idx in indices:
            traj = trajectories[:, idx, :]
            ax.plot(traj[:, 0], traj[:, 1], color=traj_color, alpha=traj_alpha, linewidth=traj_width)

        if show_avg:
            avg_traj = trajectories.mean(axis=1)
            ax.plot(avg_traj[:, 0], avg_traj[:, 1], 'k--', linewidth=avg_width)

            dx = avg_traj[-1, 0] - avg_traj[-2, 0]
            dy = avg_traj[-1, 1] - avg_traj[-2, 1]
            ax.arrow(
                avg_traj[-2, 0], avg_traj[-2, 1], dx, dy,
                shape='full', head_width=arrow_size, color='black',
                length_includes_head=True, zorder=10,
            )

        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()

    return fig


def plot_driver_heatmap(
    driver_result: "DynamicDriverResult",
    n_genes: int = 100,
    n_clusters: int = 3,
    figsize: tuple[float, float] = (4, 8),
    cmap: str = 'RdBu_r',
    fontsize: float = 6,
    title: str | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Plot driver gene heatmap with temporal clustering.

    Shows top driver genes clustered by temporal activity pattern.
    Rows are genes, columns are time points.

    Args:
        driver_result: DynamicDriverResult object
        n_genes: Number of top genes to show
        n_clusters: Number of temporal clusters
        figsize: Figure size
        cmap: Colormap
        fontsize: Font size for gene labels
        title: Plot title
        save_path: Path to save
        show: Whether to display

    Returns:
        matplotlib Figure
    """
    set_trajectory_style()

    top_genes = driver_result.top_genes[:n_genes]
    top_idx = [driver_result.gene_names.index(g) for g in top_genes]
    data = driver_result.driver_index_matrix[:, top_idx].T

    data_norm = (data - data.mean(axis=1, keepdims=True)) / (data.std(axis=1, keepdims=True) + 1e-8)

    Z = linkage(data_norm, method='ward')
    ordered_idx = leaves_list(Z)
    sorted_data = data_norm[ordered_idx, :]
    sorted_genes = [top_genes[i] for i in ordered_idx]

    cluster_ids = fcluster(Z, t=n_clusters, criterion='maxclust')
    palette = sns.color_palette("tab10", n_clusters)
    row_colors = [palette[cluster_ids[i] - 1] for i in ordered_idx]

    n_timepoints = sorted_data.shape[1]
    col_colors = ['purple'] * n_timepoints

    g = sns.clustermap(
        sorted_data,
        row_colors=row_colors,
        col_colors=col_colors,
        row_cluster=False,
        col_cluster=False,
        cmap=cmap,
        xticklabels=False,
        yticklabels=sorted_genes,
        figsize=figsize,
        cbar_pos=(0.02, 0.8, 0.03, 0.15),
        dendrogram_ratio=(0.001, 0.001),
    )

    if title:
        g.fig.suptitle(title, fontsize=12, y=1.02, color='dimgrey')

    plt.setp(g.ax_heatmap.get_yticklabels(), fontsize=fontsize)
    g.ax_heatmap.set_xlabel("Time", fontsize=10)
    g.ax_heatmap.set_ylabel("")

    if save_path:
        g.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()

    return g.fig


def plot_gene_dynamics(
    time_values: np.ndarray,
    gene_trajectories: dict[str, tuple[np.ndarray, np.ndarray]],
    true_means: np.ndarray | None = None,
    true_times: np.ndarray | None = None,
    colors: dict[str, str] | None = None,
    figsize: tuple[float, float] = (6, 4),
    n_cols: int = 3,
    save_dir: str | Path | None = None,
    show: bool = True,
) -> list[plt.Figure]:
    """Plot gene expression dynamics along trajectories.

    Shows mean expression with confidence bands for each gene.

    Args:
        time_values: Time points for trajectory
        gene_trajectories: Dict mapping gene -> (mean, std) arrays
        true_means: Optional true mean expression at observed timepoints
        true_times: Optional observed timepoints
        colors: Colors for each trajectory
        figsize: Figure size per gene
        n_cols: Number of columns for multi-gene layout
        save_dir: Directory to save individual plots
        show: Whether to display

    Returns:
        List of Figures
    """
    set_trajectory_style()

    if colors is None:
        colors = {'trajectory': 'steelblue'}

    figures = []

    for gene_name, (mean_expr, std_expr) in gene_trajectories.items():
        fig, ax = plt.subplots(figsize=figsize)

        for traj_name, color in colors.items():
            ax.plot(time_values, mean_expr, color=color, linewidth=2, alpha=0.8, label=traj_name)
            ax.fill_between(
                time_values,
                mean_expr - std_expr,
                mean_expr + std_expr,
                color=color,
                alpha=0.2,
            )

        if true_means is not None and true_times is not None:
            gene_idx = list(gene_trajectories.keys()).index(gene_name)
            ax.scatter(true_times, true_means[:, gene_idx], c='red', s=40, label='Observed', zorder=10)

        ax.set_title(gene_name, fontsize=12)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_edgecolor('grey')
            spine.set_linewidth(1)

        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            safe_name = gene_name.replace('/', '_').replace('\\', '_')
            fig.savefig(save_dir / f'{safe_name}.png', dpi=300, bbox_inches='tight')

        if show:
            plt.show()

        figures.append(fig)

    return figures


def create_trajectory_animation(
    data_pca: np.ndarray,
    stage_labels: np.ndarray,
    trajectories: list[tuple[np.ndarray, str, str]],
    n_frames: int = 100,
    figsize: tuple[float, float] = (8, 6),
    spot_size: int = 5,
    velocity_arrows: int = 10,
    arrow_length: float = 0.5,
    cmap: str = 'plasma',
    save_path: str | Path | None = None,
    fps: int = 10,
) -> FuncAnimation:
    """Create animated trajectory visualization.

    Args:
        data_pca: Background PCA coordinates
        stage_labels: Stage labels (numeric for colormap)
        trajectories: List of (trajectory_array, color, name) tuples
        n_frames: Number of animation frames
        figsize: Figure size
        spot_size: Background point size
        velocity_arrows: Number of velocity arrows to show
        arrow_length: Arrow length scale
        cmap: Colormap for background
        save_path: Path to save animation
        fps: Frames per second

    Returns:
        FuncAnimation object
    """
    set_trajectory_style()
    matplotlib.use('Agg')

    fig, ax = plt.subplots(figsize=figsize)

    colormap = plt.cm.get_cmap(cmap)
    ax.scatter(
        data_pca[:, 0], data_pca[:, 1],
        c=stage_labels, cmap=colormap,
        s=spot_size, alpha=0.1,
    )

    scatter_objects = []
    quiver_objects = []
    for traj, color, name in trajectories:
        sc = ax.scatter([], [], c=color, s=spot_size * 2, label=name)
        scatter_objects.append(sc)
        quiv = ax.quiver(
            [0] * velocity_arrows, [0] * velocity_arrows,
            [0] * velocity_arrows, [0] * velocity_arrows,
            angles='xy', scale_units='xy', scale=1, color='black',
        )
        quiver_objects.append(quiv)

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0)
    ax.legend(loc='upper left')

    def init():
        for sc in scatter_objects:
            sc.set_offsets(np.empty((0, 2)))
        return scatter_objects + quiver_objects

    def update(frame):
        t = frame / (n_frames - 1)

        for i, (traj, color, name) in enumerate(trajectories):
            n_steps = traj.shape[0]
            step = int(t * (n_steps - 1))
            points = traj[step]
            scatter_objects[i].set_offsets(points)

            if step < n_steps - 1:
                velocity = traj[step + 1] - traj[step]
                norms = np.linalg.norm(velocity, axis=1, keepdims=True)
                norms[norms == 0] = 1
                velocity_norm = (velocity / norms) * arrow_length

                quiver_objects[i].set_offsets(points[:velocity_arrows])
                quiver_objects[i].set_UVC(
                    velocity_norm[:velocity_arrows, 0],
                    velocity_norm[:velocity_arrows, 1],
                )

        return scatter_objects + quiver_objects

    ani = FuncAnimation(
        fig, update, frames=n_frames,
        init_func=init, blit=True, interval=1000 / fps,
    )

    if save_path:
        ani.save(str(save_path), writer='imagemagick', fps=fps)

    return ani
