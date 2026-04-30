"""Advanced manifold comparison visualizations for StageBridge.

Publication-quality figures comparing expression vs latent manifolds using
multiple embedding methods (PHATE, UMAP, t-SNE, PCA, diffusion maps).

Key visualizations:
- Side-by-side manifold comparison (expression vs latent)
- Phase maps / velocity fields showing progression direction
- Trajectory curvature analysis (quantify "straightening")
- Multi-method embedding comparison grid
- Geodesic path visualization on both manifolds
- Stage centroid connectivity analysis

Default stages: Normal → Preinvasive → Invasive (3-stage system)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from scipy.interpolate import splprep, splev
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

if TYPE_CHECKING:
    pass


EmbeddingMethod = Literal['pca', 'umap', 'tsne', 'phate', 'diffmap']


def _compute_embedding(
    data: np.ndarray,
    method: EmbeddingMethod = 'pca',
    n_components: int = 2,
    random_state: int = 42,
    **kwargs,
) -> np.ndarray:
    """Compute 2D embedding using specified method.

    Args:
        data: Input data (n_samples, n_features)
        method: Embedding method
        n_components: Number of components (usually 2)
        random_state: Random seed
        **kwargs: Method-specific parameters

    Returns:
        Embedded coordinates (n_samples, n_components)
    """
    if method == 'pca':
        pca = PCA(n_components=n_components, random_state=random_state)
        return pca.fit_transform(data)

    elif method == 'umap':
        try:
            import umap
        except ImportError:
            raise ImportError("Install umap-learn: pip install umap-learn")

        reducer = umap.UMAP(
            n_components=n_components,
            random_state=random_state,
            n_neighbors=kwargs.get('n_neighbors', 15),
            min_dist=kwargs.get('min_dist', 0.1),
            metric=kwargs.get('metric', 'euclidean'),
        )
        return reducer.fit_transform(data)

    elif method == 'tsne':
        from sklearn.manifold import TSNE

        tsne = TSNE(
            n_components=n_components,
            random_state=random_state,
            perplexity=kwargs.get('perplexity', 30),
            learning_rate=kwargs.get('learning_rate', 'auto'),
            init=kwargs.get('init', 'pca'),
        )
        return tsne.fit_transform(data)

    elif method == 'phate':
        try:
            import phate
        except ImportError:
            raise ImportError("Install phate: pip install phate")

        phate_op = phate.PHATE(
            n_components=n_components,
            random_state=random_state,
            knn=kwargs.get('knn', 5),
            decay=kwargs.get('decay', 40),
            t=kwargs.get('t', 'auto'),
        )
        return phate_op.fit_transform(data)

    elif method == 'diffmap':
        try:
            from sklearn.manifold import SpectralEmbedding
        except ImportError:
            raise ImportError("sklearn required for diffusion maps")

        dm = SpectralEmbedding(
            n_components=n_components,
            random_state=random_state,
            affinity=kwargs.get('affinity', 'nearest_neighbors'),
            n_neighbors=kwargs.get('n_neighbors', 10),
        )
        return dm.fit_transform(data)

    else:
        raise ValueError(f"Unknown method: {method}")


def _compute_path_curvature(points: np.ndarray) -> float:
    """Compute curvature of a path through points.

    Lower curvature = more linear = better for trajectory inference.

    Returns:
        Average curvature (0 = perfectly straight line)
    """
    if len(points) < 3:
        return 0.0

    # Fit spline through points
    try:
        tck, u = splprep([points[:, 0], points[:, 1]], s=0, k=min(3, len(points)-1))

        # Sample many points along spline
        u_fine = np.linspace(0, 1, 100)
        x_fine, y_fine = splev(u_fine, tck)

        # Compute curvature at each point
        dx = np.gradient(x_fine)
        dy = np.gradient(y_fine)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)

        curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2 + 1e-10)**1.5
        return np.mean(curvature)
    except Exception:
        # Fallback: compute angle changes
        vectors = np.diff(points, axis=0)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10
        unit_vectors = vectors / norms

        if len(unit_vectors) < 2:
            return 0.0

        dot_products = np.sum(unit_vectors[:-1] * unit_vectors[1:], axis=1)
        dot_products = np.clip(dot_products, -1, 1)
        angles = np.arccos(dot_products)

        return np.mean(np.abs(angles))


def _set_manifold_style():
    """Set publication-quality style for manifold plots."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.facecolor': 'white',
        'figure.facecolor': 'white',
        'axes.grid': False,
        'legend.frameon': False,
        'legend.fontsize': 10,
        'svg.fonttype': 'none',
        'pdf.fonttype': 42,
    })


@dataclass
class ManifoldComparisonResult:
    """Results from manifold comparison analysis.

    Attributes:
        expression_embedding: 2D coords for expression space
        latent_embedding: 2D coords for latent space
        method: Embedding method used
        expression_curvature: Path curvature in expression space
        latent_curvature: Path curvature in latent space
        linearity_improvement: Ratio of curvatures (>1 = latent is straighter)
        stage_centroids_expr: Stage centroids in expression embedding
        stage_centroids_latent: Stage centroids in latent embedding
    """
    expression_embedding: np.ndarray
    latent_embedding: np.ndarray
    method: str
    expression_curvature: float
    latent_curvature: float
    linearity_improvement: float
    stage_centroids_expr: np.ndarray
    stage_centroids_latent: np.ndarray


def compute_manifold_comparison(
    expression_data: np.ndarray,
    latent_data: np.ndarray,
    stage_labels: np.ndarray,
    method: EmbeddingMethod = 'phate',
    **kwargs,
) -> ManifoldComparisonResult:
    """Compute manifold embeddings and curvature metrics.

    Args:
        expression_data: Raw expression matrix (n_cells, n_genes)
        latent_data: Latent representations (n_cells, latent_dim)
        stage_labels: Stage labels per cell
        method: Embedding method
        **kwargs: Method-specific parameters

    Returns:
        ManifoldComparisonResult with embeddings and metrics
    """
    # Standardize expression data
    scaler = StandardScaler()
    expr_scaled = scaler.fit_transform(expression_data)

    # Compute embeddings
    expr_embed = _compute_embedding(expr_scaled, method=method, **kwargs)
    latent_embed = _compute_embedding(latent_data, method=method, **kwargs)

    # Compute stage centroids
    unique_stages = np.unique(stage_labels)
    n_stages = len(unique_stages)

    centroids_expr = np.zeros((n_stages, 2))
    centroids_latent = np.zeros((n_stages, 2))

    for i, stage in enumerate(unique_stages):
        mask = stage_labels == stage
        centroids_expr[i] = expr_embed[mask].mean(axis=0)
        centroids_latent[i] = latent_embed[mask].mean(axis=0)

    # Compute path curvature through centroids
    curv_expr = _compute_path_curvature(centroids_expr)
    curv_latent = _compute_path_curvature(centroids_latent)

    linearity_improvement = (curv_expr + 1e-10) / (curv_latent + 1e-10)

    return ManifoldComparisonResult(
        expression_embedding=expr_embed,
        latent_embedding=latent_embed,
        method=method,
        expression_curvature=curv_expr,
        latent_curvature=curv_latent,
        linearity_improvement=linearity_improvement,
        stage_centroids_expr=centroids_expr,
        stage_centroids_latent=centroids_latent,
    )


def plot_manifold_comparison(
    expression_data: np.ndarray,
    latent_data: np.ndarray,
    stage_labels: np.ndarray,
    stage_order: list[str] | None = None,
    stage_colors: dict[str, str] | None = None,
    method: EmbeddingMethod = 'phate',
    figsize: tuple[float, float] = (14, 6),
    point_size: int = 15,
    centroid_size: int = 200,
    show_paths: bool = True,
    show_curvature: bool = True,
    title: str | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
    **kwargs,
) -> tuple[plt.Figure, ManifoldComparisonResult]:
    """Plot side-by-side manifold comparison with trajectory paths.

    Creates publication figure showing:
    - Left: Expression space embedding with curved stage progression
    - Right: Latent space embedding with (ideally) straighter progression
    - Stage centroids connected to show trajectory
    - Curvature metrics annotated

    Args:
        expression_data: Raw expression matrix
        latent_data: Latent representations
        stage_labels: Stage labels per cell
        stage_order: Ordered list of stages for trajectory
        stage_colors: Dict mapping stage -> color
        method: Embedding method ('phate', 'umap', 'tsne', 'pca', 'diffmap')
        figsize: Figure size
        point_size: Size of cell points
        centroid_size: Size of centroid markers
        show_paths: Whether to show trajectory paths
        show_curvature: Whether to annotate curvature values
        title: Overall figure title
        save_path: Path to save figure
        show: Whether to display
        **kwargs: Method-specific parameters

    Returns:
        (Figure, ManifoldComparisonResult)
    """
    _set_manifold_style()

    # Compute comparison
    result = compute_manifold_comparison(
        expression_data, latent_data, stage_labels, method=method, **kwargs
    )

    # Set up stage ordering and colors
    unique_stages = np.unique(stage_labels)
    if stage_order is None:
        stage_order = list(unique_stages)

    if stage_colors is None:
        cmap = plt.cm.viridis
        stage_colors = {s: cmap(i / (len(stage_order) - 1)) for i, s in enumerate(stage_order)}

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    method_names = {
        'pca': 'PCA',
        'umap': 'UMAP',
        'tsne': 't-SNE',
        'phate': 'PHATE',
        'diffmap': 'Diffusion Map',
    }
    method_display = method_names.get(method, method.upper())

    for ax_idx, (ax, embed, centroids, space_name) in enumerate([
        (axes[0], result.expression_embedding, result.stage_centroids_expr, 'Gene Expression Manifold'),
        (axes[1], result.latent_embedding, result.stage_centroids_latent, 'Latent Manifold'),
    ]):
        # Plot cells colored by stage
        for stage in stage_order:
            mask = stage_labels == stage
            color = stage_colors[stage]
            ax.scatter(
                embed[mask, 0], embed[mask, 1],
                c=[color], s=point_size, alpha=0.6,
                label=stage, rasterized=True,
            )

        # Reorder centroids by stage_order
        stage_idx_map = {s: i for i, s in enumerate(np.unique(stage_labels))}
        ordered_centroids = np.array([centroids[stage_idx_map[s]] for s in stage_order if s in stage_idx_map])

        if show_paths and len(ordered_centroids) > 1:
            # Draw trajectory path through centroids
            for i in range(len(ordered_centroids) - 1):
                ax.annotate(
                    '', xy=ordered_centroids[i+1], xytext=ordered_centroids[i],
                    arrowprops=dict(
                        arrowstyle='->', color='black', lw=2.5,
                        connectionstyle='arc3,rad=0.1',
                    ),
                    zorder=10,
                )

            # Plot centroids as large markers
            for i, stage in enumerate(stage_order):
                if stage in stage_idx_map:
                    idx = stage_idx_map[stage]
                    ax.scatter(
                        centroids[idx, 0], centroids[idx, 1],
                        c=[stage_colors[stage]], s=centroid_size,
                        edgecolors='black', linewidths=2, zorder=11,
                        marker='o',
                    )

        ax.set_title(space_name, fontsize=14, fontweight='bold')
        ax.set_xlabel(f'{method_display} 1')
        ax.set_ylabel(f'{method_display} 2')

        # Remove ticks for cleaner look
        ax.set_xticks([])
        ax.set_yticks([])

        # Add curvature annotation
        if show_curvature:
            curv = result.expression_curvature if ax_idx == 0 else result.latent_curvature
            ax.text(
                0.02, 0.98, f'Curvature: {curv:.3f}',
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            )

    # Add legend to first axis
    axes[0].legend(loc='lower right', markerscale=1.5)

    # Add linearity improvement annotation
    if show_curvature:
        fig.text(
            0.5, 0.02,
            f'Linearity Improvement: {result.linearity_improvement:.2f}x '
            f'(higher = latent space is straighter)',
            ha='center', fontsize=11, style='italic',
        )

    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()

    return fig, result


def plot_multi_method_comparison(
    expression_data: np.ndarray,
    latent_data: np.ndarray,
    stage_labels: np.ndarray,
    stage_order: list[str] | None = None,
    stage_colors: dict[str, str] | None = None,
    methods: list[EmbeddingMethod] = ['pca', 'umap', 'phate'],
    figsize_per_panel: tuple[float, float] = (5, 4),
    point_size: int = 10,
    show_paths: bool = True,
    title: str | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Plot grid comparing multiple embedding methods.

    Creates a grid with rows = methods, columns = [expression, latent].

    Args:
        expression_data: Raw expression matrix
        latent_data: Latent representations
        stage_labels: Stage labels
        stage_order: Ordered stages
        stage_colors: Stage colors
        methods: List of embedding methods to compare
        figsize_per_panel: Size per panel
        point_size: Point size
        show_paths: Show trajectory paths
        title: Overall title
        save_path: Save path
        show: Display figure

    Returns:
        Figure
    """
    _set_manifold_style()

    n_methods = len(methods)
    fig, axes = plt.subplots(
        n_methods, 2,
        figsize=(figsize_per_panel[0] * 2, figsize_per_panel[1] * n_methods),
    )

    if n_methods == 1:
        axes = axes.reshape(1, -1)

    # Set up colors
    unique_stages = np.unique(stage_labels)
    if stage_order is None:
        stage_order = list(unique_stages)
    if stage_colors is None:
        cmap = plt.cm.viridis
        stage_colors = {s: cmap(i / (len(stage_order) - 1)) for i, s in enumerate(stage_order)}

    method_names = {
        'pca': 'PCA', 'umap': 'UMAP', 'tsne': 't-SNE',
        'phate': 'PHATE', 'diffmap': 'Diffusion Map',
    }

    results = []

    for row_idx, method in enumerate(methods):
        result = compute_manifold_comparison(
            expression_data, latent_data, stage_labels, method=method
        )
        results.append(result)

        for col_idx, (embed, centroids, space_type) in enumerate([
            (result.expression_embedding, result.stage_centroids_expr, 'Expression'),
            (result.latent_embedding, result.stage_centroids_latent, 'Latent'),
        ]):
            ax = axes[row_idx, col_idx]

            # Plot cells
            for stage in stage_order:
                mask = stage_labels == stage
                ax.scatter(
                    embed[mask, 0], embed[mask, 1],
                    c=[stage_colors[stage]], s=point_size, alpha=0.5,
                    rasterized=True,
                )

            # Plot trajectory
            if show_paths:
                stage_idx_map = {s: i for i, s in enumerate(unique_stages)}
                ordered_centroids = np.array([
                    centroids[stage_idx_map[s]] for s in stage_order if s in stage_idx_map
                ])

                if len(ordered_centroids) > 1:
                    ax.plot(
                        ordered_centroids[:, 0], ordered_centroids[:, 1],
                        'k-', lw=2, zorder=10,
                    )
                    ax.scatter(
                        ordered_centroids[:, 0], ordered_centroids[:, 1],
                        c='black', s=80, zorder=11,
                    )

            # Labels
            if row_idx == 0:
                ax.set_title(f'{space_type} Space', fontsize=12, fontweight='bold')
            if col_idx == 0:
                ax.set_ylabel(method_names.get(method, method), fontsize=12, fontweight='bold')

            ax.set_xticks([])
            ax.set_yticks([])

            # Curvature annotation
            curv = result.expression_curvature if col_idx == 0 else result.latent_curvature
            ax.text(
                0.02, 0.98, f'κ={curv:.3f}',
                transform=ax.transAxes, fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7),
            )

    # Add legend
    handles = [plt.Line2D([0], [0], marker='o', color='w',
               markerfacecolor=stage_colors[s], markersize=10, label=s)
               for s in stage_order]
    fig.legend(handles=handles, loc='center right', bbox_to_anchor=(1.1, 0.5))

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()

    return fig


def plot_trajectory_straightness(
    expression_data: np.ndarray,
    latent_data: np.ndarray,
    stage_labels: np.ndarray,
    stage_order: list[str],
    n_samples: int = 50,
    method: EmbeddingMethod = 'pca',
    figsize: tuple[float, float] = (12, 5),
    colors: tuple[str, str] = ('#E74C3C', '#3498DB'),
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Plot trajectory straightness comparison.

    Shows sampled trajectories from individual cells in both spaces,
    highlighting the "straightening" effect of the latent space.

    Args:
        expression_data: Expression matrix
        latent_data: Latent representations
        stage_labels: Stage labels
        stage_order: Ordered stages
        n_samples: Number of trajectory samples
        method: Embedding method
        figsize: Figure size
        colors: (expression_color, latent_color)
        save_path: Save path
        show: Display

    Returns:
        Figure
    """
    _set_manifold_style()

    # Compute embeddings
    scaler = StandardScaler()
    expr_scaled = scaler.fit_transform(expression_data)
    expr_embed = _compute_embedding(expr_scaled, method=method)
    latent_embed = _compute_embedding(latent_data, method=method)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Get cells per stage
    stage_cells = {s: np.where(stage_labels == s)[0] for s in stage_order}

    # Sample trajectories (one cell per stage, connected)
    expr_curvatures = []
    latent_curvatures = []

    for _ in range(n_samples):
        # Sample one cell from each stage
        sampled_idx = [np.random.choice(stage_cells[s]) for s in stage_order]

        expr_path = expr_embed[sampled_idx]
        latent_path = latent_embed[sampled_idx]

        # Plot on first two panels
        axes[0].plot(expr_path[:, 0], expr_path[:, 1],
                    color=colors[0], alpha=0.3, lw=1)
        axes[1].plot(latent_path[:, 0], latent_path[:, 1],
                    color=colors[1], alpha=0.3, lw=1)

        # Compute curvatures
        expr_curvatures.append(_compute_path_curvature(expr_path))
        latent_curvatures.append(_compute_path_curvature(latent_path))

    # Add all points as background
    axes[0].scatter(expr_embed[:, 0], expr_embed[:, 1], c='grey', s=5, alpha=0.1)
    axes[1].scatter(latent_embed[:, 0], latent_embed[:, 1], c='grey', s=5, alpha=0.1)

    axes[0].set_title('Expression Space\n(Curved Trajectories)', fontsize=12)
    axes[1].set_title('Latent Space\n(Straighter Trajectories)', fontsize=12)

    for ax in axes[:2]:
        ax.set_xticks([])
        ax.set_yticks([])

    # Third panel: curvature distribution comparison
    axes[2].hist(expr_curvatures, bins=20, alpha=0.7, color=colors[0],
                 label='Expression', density=True)
    axes[2].hist(latent_curvatures, bins=20, alpha=0.7, color=colors[1],
                 label='Latent', density=True)
    axes[2].axvline(np.mean(expr_curvatures), color=colors[0], ls='--', lw=2)
    axes[2].axvline(np.mean(latent_curvatures), color=colors[1], ls='--', lw=2)
    axes[2].set_xlabel('Path Curvature')
    axes[2].set_ylabel('Density')
    axes[2].set_title('Curvature Distribution')
    axes[2].legend()

    # Add stats
    improvement = np.mean(expr_curvatures) / (np.mean(latent_curvatures) + 1e-10)
    fig.text(
        0.5, 0.02,
        f'Mean curvature: Expression={np.mean(expr_curvatures):.3f}, '
        f'Latent={np.mean(latent_curvatures):.3f} | '
        f'Improvement: {improvement:.2f}x',
        ha='center', fontsize=11,
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()

    return fig


def plot_geodesic_comparison(
    expression_data: np.ndarray,
    latent_data: np.ndarray,
    stage_labels: np.ndarray,
    source_stage: str,
    target_stage: str,
    n_paths: int = 30,
    n_interp: int = 50,
    method: EmbeddingMethod = 'phate',
    figsize: tuple[float, float] = (14, 6),
    expr_color: str = '#E74C3C',
    latent_color: str = '#3498DB',
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Plot geodesic paths between two stages.

    Shows how straight-line interpolation in latent space maps
    to curved paths in expression space (and vice versa).

    Args:
        expression_data: Expression matrix
        latent_data: Latent representations
        stage_labels: Stage labels
        source_stage: Starting stage
        target_stage: Ending stage
        n_paths: Number of paths to show
        n_interp: Interpolation steps
        method: Embedding method
        figsize: Figure size
        expr_color: Expression path color
        latent_color: Latent path color
        save_path: Save path
        show: Display

    Returns:
        Figure
    """
    _set_manifold_style()

    # Get source and target cells
    source_mask = stage_labels == source_stage
    target_mask = stage_labels == target_stage

    source_expr = expression_data[source_mask]
    target_expr = expression_data[target_mask]
    source_latent = latent_data[source_mask]
    target_latent = latent_data[target_mask]

    # Sample pairs
    n_source = len(source_expr)
    n_target = len(target_expr)
    n_paths = min(n_paths, n_source, n_target)

    source_idx = np.random.choice(n_source, n_paths, replace=False)
    target_idx = np.random.choice(n_target, n_paths, replace=False)

    # Interpolate in LATENT space (straight lines)
    t_values = np.linspace(0, 1, n_interp)

    latent_paths = []
    for si, ti in zip(source_idx, target_idx):
        path = np.array([
            (1 - t) * source_latent[si] + t * target_latent[ti]
            for t in t_values
        ])
        latent_paths.append(path)

    latent_paths = np.array(latent_paths)  # (n_paths, n_interp, latent_dim)

    # Also create expression-space interpolation for comparison
    expr_paths = []
    for si, ti in zip(source_idx, target_idx):
        path = np.array([
            (1 - t) * source_expr[si] + t * target_expr[ti]
            for t in t_values
        ])
        expr_paths.append(path)

    expr_paths = np.array(expr_paths)

    # Compute embeddings
    scaler = StandardScaler()
    all_expr = np.vstack([expression_data, expr_paths.reshape(-1, expression_data.shape[1])])
    all_expr_scaled = scaler.fit_transform(all_expr)

    expr_embed_all = _compute_embedding(all_expr_scaled[:len(expression_data)], method=method)

    # Re-embed the interpolated paths
    expr_paths_scaled = scaler.transform(expr_paths.reshape(-1, expression_data.shape[1]))
    expr_paths_embed = _compute_embedding(
        np.vstack([all_expr_scaled[:len(expression_data)], expr_paths_scaled]),
        method=method,
    )[len(expression_data):]
    expr_paths_embed = expr_paths_embed.reshape(n_paths, n_interp, 2)

    # Embed latent paths
    all_latent = np.vstack([latent_data, latent_paths.reshape(-1, latent_data.shape[1])])
    latent_embed_all = _compute_embedding(all_latent, method=method)
    latent_paths_embed = latent_embed_all[len(latent_data):].reshape(n_paths, n_interp, 2)
    latent_embed = latent_embed_all[:len(latent_data)]

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot background points
    for ax, embed in zip(axes, [expr_embed_all[:len(expression_data)], latent_embed]):
        ax.scatter(embed[:, 0], embed[:, 1], c='lightgrey', s=10, alpha=0.3, rasterized=True)

    # Plot paths
    for i in range(n_paths):
        # Expression space: interpolation is in expression space
        axes[0].plot(
            expr_paths_embed[i, :, 0], expr_paths_embed[i, :, 1],
            color=expr_color, alpha=0.5, lw=1.5,
        )

        # Latent space: interpolation is in latent space (straight)
        axes[1].plot(
            latent_paths_embed[i, :, 0], latent_paths_embed[i, :, 1],
            color=latent_color, alpha=0.5, lw=1.5,
        )

    # Highlight source and target
    for ax, embed in zip(axes, [expr_embed_all[:len(expression_data)], latent_embed]):
        ax.scatter(embed[source_mask, 0], embed[source_mask, 1],
                  c='green', s=30, alpha=0.8, label=source_stage, zorder=5)
        ax.scatter(embed[target_mask, 0], embed[target_mask, 1],
                  c='red', s=30, alpha=0.8, label=target_stage, zorder=5)

    axes[0].set_title(f'Expression Space\n(Linear interp in expression)', fontsize=12)
    axes[1].set_title(f'Latent Space\n(Linear interp in latent = geodesic)', fontsize=12)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(loc='lower right')

    fig.suptitle(f'Geodesic Paths: {source_stage} → {target_stage}', fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()

    return fig


def _compute_velocity_field(
    embedding: np.ndarray,
    stage_labels: np.ndarray,
    stage_order: list[str],
    n_neighbors: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute velocity field based on stage progression.

    For each cell, velocity points toward cells of the next stage.

    Args:
        embedding: 2D coordinates
        stage_labels: Stage labels
        stage_order: Ordered stages (progression direction)
        n_neighbors: Neighbors to average for velocity

    Returns:
        (positions, velocities) arrays
    """
    from sklearn.neighbors import NearestNeighbors

    n_cells = len(embedding)
    velocities = np.zeros((n_cells, 2))

    stage_to_idx = {s: i for i, s in enumerate(stage_order)}

    for i, stage in enumerate(stage_order[:-1]):
        next_stage = stage_order[i + 1]

        current_mask = stage_labels == stage
        next_mask = stage_labels == next_stage

        if not np.any(current_mask) or not np.any(next_mask):
            continue

        current_cells = embedding[current_mask]
        next_cells = embedding[next_mask]

        # For each current cell, find nearest neighbors in next stage
        nn = NearestNeighbors(n_neighbors=min(n_neighbors, len(next_cells)))
        nn.fit(next_cells)
        _, indices = nn.kneighbors(current_cells)

        # Velocity = average direction to next-stage neighbors
        for j, cell_idx in enumerate(np.where(current_mask)[0]):
            neighbor_positions = next_cells[indices[j]]
            direction = neighbor_positions.mean(axis=0) - embedding[cell_idx]
            # Normalize
            norm = np.linalg.norm(direction)
            if norm > 0:
                velocities[cell_idx] = direction / norm

    return embedding, velocities


def plot_phase_map(
    expression_data: np.ndarray,
    latent_data: np.ndarray,
    stage_labels: np.ndarray,
    stage_order: list[str] | None = None,
    stage_colors: dict[str, str] | None = None,
    method: EmbeddingMethod = 'phate',
    figsize: tuple[float, float] = (14, 6),
    point_size: int = 20,
    arrow_scale: float = 0.15,
    arrow_width: float = 0.003,
    n_arrows: int = 500,
    stream: bool = False,
    grid_density: int = 25,
    title: str | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
    **kwargs,
) -> plt.Figure:
    """Plot phase map / velocity field showing progression direction.

    Shows the direction of disease progression at each point in the manifold.
    Left panel: Expression space. Right panel: Latent space.

    In a well-learned latent space, the velocity field should be more
    uniform and aligned (all pointing in same general direction).

    Args:
        expression_data: Expression matrix
        latent_data: Latent representations
        stage_labels: Stage labels
        stage_order: Ordered stages for progression direction
        stage_colors: Colors per stage
        method: Embedding method
        figsize: Figure size
        point_size: Size of scatter points
        arrow_scale: Scale factor for arrows
        arrow_width: Width of arrows
        n_arrows: Number of arrows to show (subsampled)
        stream: If True, use streamplot instead of quiver
        grid_density: Grid density for streamplot
        title: Figure title
        save_path: Path to save
        show: Display figure
        **kwargs: Passed to embedding method

    Returns:
        Figure
    """
    _set_manifold_style()

    unique_stages = np.unique(stage_labels)
    if stage_order is None:
        stage_order = list(unique_stages)

    if stage_colors is None:
        cmap = plt.cm.viridis
        stage_colors = {s: cmap(i / max(1, len(stage_order) - 1)) for i, s in enumerate(stage_order)}

    # Compute embeddings
    scaler = StandardScaler()
    expr_scaled = scaler.fit_transform(expression_data)
    expr_embed = _compute_embedding(expr_scaled, method=method, **kwargs)
    latent_embed = _compute_embedding(latent_data, method=method, **kwargs)

    # Compute velocity fields
    _, expr_velocity = _compute_velocity_field(expr_embed, stage_labels, stage_order)
    _, latent_velocity = _compute_velocity_field(latent_embed, stage_labels, stage_order)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    method_names = {
        'pca': 'PCA', 'umap': 'UMAP', 'tsne': 't-SNE',
        'phate': 'PHATE', 'diffmap': 'Diffusion Map',
    }
    method_display = method_names.get(method, method.upper())

    for ax_idx, (ax, embed, velocity, space_name) in enumerate([
        (axes[0], expr_embed, expr_velocity, 'Expression Space'),
        (axes[1], latent_embed, latent_velocity, 'Latent Space'),
    ]):
        # Plot cells colored by stage
        for stage in stage_order:
            mask = stage_labels == stage
            ax.scatter(
                embed[mask, 0], embed[mask, 1],
                c=[stage_colors[stage]], s=point_size, alpha=0.5,
                label=stage, rasterized=True, zorder=1,
            )

        # Get cells with non-zero velocity (not in final stage)
        has_velocity = np.linalg.norm(velocity, axis=1) > 0
        valid_idx = np.where(has_velocity)[0]

        if len(valid_idx) > n_arrows:
            # Subsample
            selected = np.random.choice(valid_idx, n_arrows, replace=False)
        else:
            selected = valid_idx

        if stream:
            # Create grid for streamplot
            x_min, x_max = embed[:, 0].min(), embed[:, 0].max()
            y_min, y_max = embed[:, 1].min(), embed[:, 1].max()
            margin = 0.1
            x_range = x_max - x_min
            y_range = y_max - y_min

            xi = np.linspace(x_min - margin * x_range, x_max + margin * x_range, grid_density)
            yi = np.linspace(y_min - margin * y_range, y_max + margin * y_range, grid_density)
            Xi, Yi = np.meshgrid(xi, yi)

            # Interpolate velocity to grid
            from scipy.interpolate import griddata

            U = griddata(embed[has_velocity], velocity[has_velocity, 0], (Xi, Yi), method='linear', fill_value=0)
            V = griddata(embed[has_velocity], velocity[has_velocity, 1], (Xi, Yi), method='linear', fill_value=0)

            # Normalize for uniform arrow length
            speed = np.sqrt(U**2 + V**2)
            U_norm = np.where(speed > 0, U / speed, 0)
            V_norm = np.where(speed > 0, V / speed, 0)

            ax.streamplot(
                xi, yi, U_norm, V_norm,
                color='black', density=1.5, linewidth=0.8,
                arrowsize=1.2, zorder=2,
            )
        else:
            # Quiver plot
            ax.quiver(
                embed[selected, 0], embed[selected, 1],
                velocity[selected, 0], velocity[selected, 1],
                angles='xy', scale_units='xy',
                scale=1/arrow_scale, width=arrow_width,
                color='black', alpha=0.7, zorder=2,
            )

        # Compute velocity alignment metric
        valid_velocities = velocity[has_velocity]
        if len(valid_velocities) > 1:
            # Normalize velocities
            norms = np.linalg.norm(valid_velocities, axis=1, keepdims=True)
            norms[norms == 0] = 1
            unit_vecs = valid_velocities / norms

            # Mean velocity direction
            mean_dir = unit_vecs.mean(axis=0)
            mean_dir_norm = mean_dir / (np.linalg.norm(mean_dir) + 1e-10)

            # Alignment = mean dot product with mean direction
            alignment = np.mean(np.dot(unit_vecs, mean_dir_norm))

            ax.text(
                0.02, 0.98, f'Alignment: {alignment:.3f}',
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            )

        ax.set_title(f'{space_name}\n({method_display})', fontsize=12, fontweight='bold')
        ax.set_xlabel(f'{method_display} 1')
        ax.set_ylabel(f'{method_display} 2')
        ax.set_xticks([])
        ax.set_yticks([])

    # Legend on first panel
    axes[0].legend(loc='lower right', markerscale=1.2)

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    else:
        fig.suptitle('Phase Map: Progression Velocity Field', fontsize=14, fontweight='bold', y=1.02)

    # Add interpretation note
    fig.text(
        0.5, 0.02,
        'Arrows show direction of progression. Higher alignment = more coherent progression.',
        ha='center', fontsize=10, style='italic',
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()

    return fig


def plot_phase_portrait_grid(
    expression_data: np.ndarray,
    latent_data: np.ndarray,
    stage_labels: np.ndarray,
    stage_order: list[str] | None = None,
    methods: list[EmbeddingMethod] = ['pca', 'umap', 'phate'],
    figsize_per_panel: tuple[float, float] = (5, 4),
    point_size: int = 15,
    arrow_scale: float = 0.12,
    n_arrows: int = 300,
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Plot grid of phase portraits across embedding methods.

    Rows: Embedding methods (PCA, UMAP, PHATE, etc.)
    Columns: Expression space vs Latent space

    Args:
        expression_data: Expression matrix
        latent_data: Latent representations
        stage_labels: Stage labels
        stage_order: Ordered stages
        methods: Embedding methods to compare
        figsize_per_panel: Size per panel
        point_size: Point size
        arrow_scale: Arrow scale
        n_arrows: Number of arrows
        save_path: Save path
        show: Display

    Returns:
        Figure
    """
    _set_manifold_style()

    unique_stages = np.unique(stage_labels)
    if stage_order is None:
        stage_order = list(unique_stages)

    cmap = plt.cm.viridis
    stage_colors = {s: cmap(i / max(1, len(stage_order) - 1)) for i, s in enumerate(stage_order)}

    n_methods = len(methods)
    fig, axes = plt.subplots(
        n_methods, 2,
        figsize=(figsize_per_panel[0] * 2, figsize_per_panel[1] * n_methods),
    )
    if n_methods == 1:
        axes = axes.reshape(1, -1)

    method_names = {
        'pca': 'PCA', 'umap': 'UMAP', 'tsne': 't-SNE',
        'phate': 'PHATE', 'diffmap': 'Diffusion Map',
    }

    scaler = StandardScaler()
    expr_scaled = scaler.fit_transform(expression_data)

    for row_idx, method in enumerate(methods):
        # Compute embeddings
        expr_embed = _compute_embedding(expr_scaled, method=method)
        latent_embed = _compute_embedding(latent_data, method=method)

        # Compute velocities
        _, expr_vel = _compute_velocity_field(expr_embed, stage_labels, stage_order)
        _, latent_vel = _compute_velocity_field(latent_embed, stage_labels, stage_order)

        for col_idx, (embed, velocity) in enumerate([
            (expr_embed, expr_vel),
            (latent_embed, latent_vel),
        ]):
            ax = axes[row_idx, col_idx]

            # Plot cells
            for stage in stage_order:
                mask = stage_labels == stage
                ax.scatter(
                    embed[mask, 0], embed[mask, 1],
                    c=[stage_colors[stage]], s=point_size, alpha=0.4,
                    rasterized=True,
                )

            # Plot velocity arrows
            has_velocity = np.linalg.norm(velocity, axis=1) > 0
            valid_idx = np.where(has_velocity)[0]
            if len(valid_idx) > n_arrows:
                selected = np.random.choice(valid_idx, n_arrows, replace=False)
            else:
                selected = valid_idx

            ax.quiver(
                embed[selected, 0], embed[selected, 1],
                velocity[selected, 0], velocity[selected, 1],
                angles='xy', scale_units='xy',
                scale=1/arrow_scale, width=0.004,
                color='black', alpha=0.6,
            )

            # Alignment metric
            valid_vels = velocity[has_velocity]
            if len(valid_vels) > 1:
                norms = np.linalg.norm(valid_vels, axis=1, keepdims=True)
                norms[norms == 0] = 1
                unit_vecs = valid_vels / norms
                mean_dir = unit_vecs.mean(axis=0)
                mean_dir_norm = mean_dir / (np.linalg.norm(mean_dir) + 1e-10)
                alignment = np.mean(np.dot(unit_vecs, mean_dir_norm))
                ax.text(0.02, 0.98, f'A={alignment:.2f}', transform=ax.transAxes,
                       fontsize=9, va='top', bbox=dict(facecolor='white', alpha=0.7))

            # Labels
            if row_idx == 0:
                ax.set_title('Expression' if col_idx == 0 else 'Latent', fontsize=11, fontweight='bold')
            if col_idx == 0:
                ax.set_ylabel(method_names.get(method, method), fontsize=11, fontweight='bold')

            ax.set_xticks([])
            ax.set_yticks([])

    # Legend
    handles = [plt.Line2D([0], [0], marker='o', color='w',
               markerfacecolor=stage_colors[s], markersize=8, label=s)
               for s in stage_order]
    fig.legend(handles=handles, loc='center right', bbox_to_anchor=(1.08, 0.5))

    fig.suptitle('Phase Portraits: Velocity Field Comparison', fontsize=13, fontweight='bold', y=1.01)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()

    return fig
