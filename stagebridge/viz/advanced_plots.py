"""Advanced visualization utilities for StageBridge.

Provides specialized plotting functions for:
  - Radar/spider plots for multi-metric comparisons
  - Parallel coordinates for high-dimensional data
  - Ridge plots for distribution comparisons
  - Correlation matrices with significance
  - 3D scatter plots for embeddings
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stagebridge.logging_utils import get_logger

log = get_logger(__name__)


def plot_radar_chart(
    df: pd.DataFrame,
    metrics: list[str],
    labels_col: str = "label",
    output_path: Path | None = None,
    title: str = "Multi-Metric Comparison",
    normalize: bool = True,
) -> plt.Figure:
    """Create radar/spider chart for comparing multiple metrics across models.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with metrics
    metrics : list of str
        Column names for metrics to include
    labels_col : str
        Column name for model labels
    output_path : Path, optional
        Where to save the figure
    title : str
        Plot title
    normalize : bool
        Whether to normalize metrics to [0, 1] range
        
    Returns
    -------
    fig : Figure
        Matplotlib figure object
    """
    if df.empty or not metrics:
        raise ValueError("DataFrame is empty or no metrics provided")

    # Extract data
    labels = df[labels_col].values
    values = df[metrics].values.astype(float)

    # Normalize if requested
    if normalize:
        mins = values.min(axis=0, keepdims=True)
        maxs = values.max(axis=0, keepdims=True)
        values = (values - mins) / (maxs - mins + 1e-8)

    # Number of variables
    num_vars = len(metrics)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # Close the plot
    angles += angles[:1]

    # Set up figure
    fig, ax = plt.subplots(figsize=(9, 8), subplot_kw=dict(projection='polar'), dpi=150)
    fig.patch.set_facecolor('white')

    # Color palette
    colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))

    # Plot each model
    for idx, (label, vals) in enumerate(zip(labels, values)):
        vals_plot = vals.tolist()
        vals_plot += vals_plot[:1]  # Close the plot
        ax.plot(angles, vals_plot, 'o-', linewidth=2, label=label,
               color=colors[idx], alpha=0.7)
        ax.fill(angles, vals_plot, alpha=0.15, color=colors[idx])

    # Fix axis to go from 0 to 1 (or data range if not normalized)
    if normalize:
        ax.set_ylim(0, 1)

    # Set labels
    metric_labels = [m.replace('_', ' ').title() for m in metrics]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, size=11)

    # Add grid
    ax.grid(True, linestyle='--', alpha=0.3)

    # Title and legend
    ax.set_title(title, size=15, fontweight='bold', pad=20)
    legend = ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0),
                      framealpha=0.95, fontsize=10)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('gray')

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        if output_path.suffix.lower() != ".pdf":
            fig.savefig(output_path.with_suffix(".pdf"), bbox_inches='tight')
        log.info("Radar chart saved to: %s", output_path)

    return fig


def plot_parallel_coordinates(
    df: pd.DataFrame,
    metrics: list[str],
    labels_col: str = "label",
    output_path: Path | None = None,
    title: str = "Parallel Coordinates Plot",
    normalize: bool = True,
) -> plt.Figure:
    """Create parallel coordinates plot for high-dimensional metric comparison.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with metrics
    metrics : list of str
        Column names for metrics to plot
    labels_col : str
        Column name for model labels
    output_path : Path, optional
        Where to save the figure
    title : str
        Plot title
    normalize : bool
        Whether to normalize each metric to [0, 1]
        
    Returns
    -------
    fig : Figure
        Matplotlib figure object
    """
    if df.empty or not metrics:
        raise ValueError("DataFrame is empty or no metrics provided")

    # Extract data
    labels = df[labels_col].values
    values = df[metrics].values.astype(float)

    # Normalize to [0, 1] for each metric
    if normalize:
        mins = values.min(axis=0, keepdims=True)
        maxs = values.max(axis=0, keepdims=True)
        values = (values - mins) / (maxs - mins + 1e-8)

    # Set up figure
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    ax.set_facecolor('#FAFAFA')
    fig.patch.set_facecolor('white')

    # X positions for each metric
    x = np.arange(len(metrics))

    # Color palette
    colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))

    # Plot lines for each model
    for idx, (label, vals) in enumerate(zip(labels, values)):
        ax.plot(x, vals, marker='o', markersize=8, linewidth=2.5,
               label=label, color=colors[idx], alpha=0.7)

    # Styling
    ax.set_xticks(x)
    metric_labels = [m.replace('_', ' ').title() for m in metrics]
    ax.set_xticklabels(metric_labels, rotation=25, ha='right', fontsize=11)

    if normalize:
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("Normalized Value", fontsize=13, fontweight='bold')
    else:
        ax.set_ylabel("Value", fontsize=13, fontweight='bold')

    ax.set_title(title, fontsize=15, fontweight='bold', pad=15)
    ax.grid(axis='both', alpha=0.3, linestyle=':', linewidth=1)

    # Legend
    legend = ax.legend(loc='best', framealpha=0.95, fontsize=10)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('gray')
    legend.get_frame().set_linewidth(1.5)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        if output_path.suffix.lower() != ".pdf":
            fig.savefig(output_path.with_suffix(".pdf"), bbox_inches='tight')
        log.info("Parallel coordinates plot saved to: %s", output_path)

    return fig


def plot_correlation_matrix(
    df: pd.DataFrame,
    metrics: list[str] | None = None,
    output_path: Path | None = None,
    title: str = "Metric Correlation Matrix",
    method: str = "pearson",
    show_values: bool = True,
) -> plt.Figure:
    """Create correlation matrix heatmap with hierarchical clustering.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with metrics
    metrics : list of str, optional
        Column names to include (if None, use all numeric columns)
    output_path : Path, optional
        Where to save the figure
    title : str
        Plot title
    method : str
        Correlation method: 'pearson', 'spearman', or 'kendall'
    show_values : bool
        Whether to annotate cells with correlation values
        
    Returns
    -------
    fig : Figure
        Matplotlib figure object
    """
    if df.empty:
        raise ValueError("DataFrame is empty")

    # Select metrics
    if metrics is None:
        metrics = df.select_dtypes(include=[np.number]).columns.tolist()

    if not metrics:
        raise ValueError("No numeric columns found")

    # Compute correlation matrix
    corr = df[metrics].corr(method=method)

    # Set up figure
    fig, ax = plt.subplots(figsize=(10, 9), dpi=150)
    fig.patch.set_facecolor('white')

    # Draw heatmap
    im = ax.imshow(corr, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1,
                   interpolation='nearest')

    # Add grid lines
    for i in range(len(metrics) + 1):
        ax.axhline(i - 0.5, color='white', linewidth=1.5)
        ax.axvline(i - 0.5, color='white', linewidth=1.5)

    # Set ticks and labels
    metric_labels = [m.replace('_', ' ').title() for m in metrics]
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(metric_labels, rotation=45, ha='right', fontsize=11)
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_yticklabels(metric_labels, fontsize=11)
    ax.set_title(title, fontsize=15, fontweight='bold', pad=15)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(f"{method.title()} Correlation", fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)

    # Annotate cells with correlation values
    if show_values:
        for i in range(len(metrics)):
            for j in range(len(metrics)):
                val = corr.iloc[i, j]
                text_color = "white" if abs(val) > 0.7 else "black"
                ax.text(j, i, f"{val:.2f}",
                       ha="center", va="center", fontsize=9,
                       color=text_color, fontweight='bold')

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        if output_path.suffix.lower() != ".pdf":
            fig.savefig(output_path.with_suffix(".pdf"), bbox_inches='tight')
        log.info("Correlation matrix saved to: %s", output_path)

    return fig


def plot_3d_embedding(
    coords: np.ndarray,
    labels: np.ndarray | None = None,
    output_path: Path | None = None,
    title: str = "3D Embedding Visualization",
    point_size: float = 3.0,
    alpha: float = 0.7,
) -> plt.Figure:
    """Create 3D scatter plot for embedding visualization.
    
    Parameters
    ----------
    coords : ndarray, shape (n_samples, 3)
        3D coordinates for each point
    labels : ndarray, optional
        Labels for coloring points
    output_path : Path, optional
        Where to save the figure
    title : str
        Plot title
    point_size : float
        Size of scatter points
    alpha : float
        Point transparency
        
    Returns
    -------
    fig : Figure
        Matplotlib figure object
    """

    coords = np.asarray(coords, dtype=float)
    if coords.shape[1] != 3:
        raise ValueError(f"Expected 3D coordinates, got shape {coords.shape}")

    # Set up 3D figure
    fig = plt.figure(figsize=(10, 9), dpi=150)
    fig.patch.set_facecolor('white')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#F8F8F8')

    if labels is not None:
        # Color by labels
        unique_labels = np.unique(labels)
        colors = plt.cm.Set2(np.linspace(0, 1, len(unique_labels)))

        for idx, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(coords[mask, 0], coords[mask, 1], coords[mask, 2],
                      c=[colors[idx]], s=point_size, alpha=alpha,
                      label=str(label), edgecolors='white', linewidths=0.3)

        ax.legend(loc='best', framealpha=0.95, fontsize=10)
    else:
        # Single color
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                  c='#0E7490', s=point_size, alpha=alpha,
                  edgecolors='white', linewidths=0.3)

    # Styling
    ax.set_xlabel("Dim 1", fontsize=12, fontweight='bold')
    ax.set_ylabel("Dim 2", fontsize=12, fontweight='bold')
    ax.set_zlabel("Dim 3", fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=15, fontweight='bold', pad=15)

    # Grid
    ax.grid(alpha=0.2, linestyle=':', linewidth=0.5)

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        if output_path.suffix.lower() != ".pdf":
            fig.savefig(output_path.with_suffix(".pdf"), bbox_inches='tight')
        log.info("3D embedding plot saved to: %s", output_path)

    return fig


def plot_ridge_distributions(
    data_dict: dict[str, np.ndarray],
    output_path: Path | None = None,
    title: str = "Distribution Comparison",
    colors: list[str] | None = None,
) -> plt.Figure:
    """Create ridge plot (joyplot) for comparing distributions.
    
    Parameters
    ----------
    data_dict : dict
        Dictionary mapping labels to 1D arrays of values
    output_path : Path, optional
        Where to save the figure
    title : str
        Plot title
    colors : list of str, optional
        Colors for each distribution
        
    Returns
    -------
    fig : Figure
        Matplotlib figure object
    """
    if not data_dict:
        raise ValueError("data_dict is empty")

    n_distributions = len(data_dict)
    labels = list(data_dict.keys())

    # Set up colors
    if colors is None:
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, n_distributions))

    # Set up figure
    fig, axes = plt.subplots(n_distributions, 1,
                            figsize=(11, 2 * n_distributions),
                            sharex=True, dpi=150)
    fig.patch.set_facecolor('white')

    if n_distributions == 1:
        axes = [axes]

    # Plot each distribution
    for idx, (label, data) in enumerate(data_dict.items()):
        ax = axes[idx]
        ax.set_facecolor('#FAFAFA')

        # Density plot
        data_clean = np.asarray(data, dtype=float)
        data_clean = data_clean[np.isfinite(data_clean)]

        if len(data_clean) > 0:
            ax.hist(data_clean, bins=50, density=True,
                   alpha=0.6, color=colors[idx], edgecolor='white')

            # Add KDE if scipy available
            try:
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(data_clean)
                x_range = np.linspace(data_clean.min(), data_clean.max(), 200)
                ax.plot(x_range, kde(x_range), color=colors[idx],
                       linewidth=2.5, alpha=0.9)
            except Exception:
                pass

        # Styling
        ax.set_ylabel(label, fontsize=11, fontweight='bold', rotation=0,
                     ha='right', va='center')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_yticks([])
        ax.grid(axis='x', alpha=0.2, linestyle=':', linewidth=0.5)

    # Only show x-label on bottom plot
    axes[-1].set_xlabel("Value", fontsize=13, fontweight='bold')

    # Overall title
    fig.suptitle(title, fontsize=15, fontweight='bold', y=0.995)

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        if output_path.suffix.lower() != ".pdf":
            fig.savefig(output_path.with_suffix(".pdf"), bbox_inches='tight')
        log.info("Ridge plot saved to: %s", output_path)

    return fig
