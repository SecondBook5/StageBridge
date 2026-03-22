"""
Visualization for spatial backend comparison.

Provides plots for comparing backend outputs and metrics.
"""

from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

from .base import BackendMappingResult
from .comparison import ComparisonResult
from .standardize import StandardizedOutput


# Custom colormap for cell type proportions
CELLTYPE_CMAP = LinearSegmentedColormap.from_list("celltype", ["#f7fbff", "#08519c"])


def plot_spatial_maps_comparison(
    results: dict[str, BackendMappingResult | StandardizedOutput],
    spatial_coords: np.ndarray,
    cell_types_to_show: list[str] | None = None,
    n_types_per_backend: int = 4,
    figsize: tuple[float, float] | None = None,
    output_path: Path | None = None,
) -> plt.Figure:
    """
    Create side-by-side spatial maps comparing backends.

    Args:
        results: Dictionary mapping backend name to result
        spatial_coords: (n_spots, 2) array of spatial coordinates
        cell_types_to_show: Specific cell types to show (auto-select if None)
        n_types_per_backend: Number of cell types to show per backend
        figsize: Figure size (auto if None)
        output_path: Optional path to save figure

    Returns:
        Matplotlib Figure
    """
    backends = list(results.keys())
    n_backends = len(backends)

    if n_backends == 0:
        raise ValueError("No results to plot")

    # Get cell types to show
    if cell_types_to_show is None:
        # Auto-select most variable cell types
        all_proportions = []
        for result in results.values():
            props = _get_proportions(result)
            all_proportions.append(props)

        # Find most variable cell types across backends
        combined = pd.concat(all_proportions, axis=0)
        type_variance = combined.var(axis=0)
        cell_types_to_show = type_variance.nlargest(n_types_per_backend).index.tolist()

    n_types = len(cell_types_to_show)

    # Create figure
    if figsize is None:
        figsize = (4 * n_backends, 3 * n_types)

    fig, axes = plt.subplots(n_types, n_backends, figsize=figsize, squeeze=False)

    for col, backend_name in enumerate(backends):
        result = results[backend_name]
        proportions = _get_proportions(result)

        for row, cell_type in enumerate(cell_types_to_show):
            ax = axes[row, col]

            if cell_type in proportions.columns:
                values = proportions[cell_type].values

                # Plot spatial scatter
                scatter = ax.scatter(
                    spatial_coords[:, 0],
                    spatial_coords[:, 1],
                    c=values,
                    cmap=CELLTYPE_CMAP,
                    s=10,
                    vmin=0,
                    vmax=1,
                    alpha=0.8,
                )

                if row == 0:
                    ax.set_title(backend_name.upper(), fontsize=12, fontweight="bold")

                if col == 0:
                    ax.set_ylabel(cell_type, fontsize=10)

                # Add colorbar
                if col == n_backends - 1:
                    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
                    cbar.set_label("Proportion", fontsize=8)
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def plot_metrics_comparison(
    comparison_table: pd.DataFrame,
    metrics_to_show: list[str] | None = None,
    figsize: tuple[float, float] = (12, 6),
    output_path: Path | None = None,
) -> plt.Figure:
    """
    Create bar charts comparing metrics across backends.

    Args:
        comparison_table: Comparison DataFrame with metrics
        metrics_to_show: Specific metrics to show (auto-select if None)
        figsize: Figure size
        output_path: Optional path to save figure

    Returns:
        Matplotlib Figure
    """
    # Filter to successful backends
    df = comparison_table[comparison_table["success"]].copy()

    if len(df) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No successful backends", ha="center", va="center")
        return fig

    # Select metrics
    if metrics_to_show is None:
        # Auto-select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        exclude = ["success", "runtime_seconds"]
        metrics_to_show = [c for c in numeric_cols if c not in exclude and not c.endswith("_n_")][
            :8
        ]  # Limit to 8 metrics

    if not metrics_to_show:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No metrics to display", ha="center", va="center")
        return fig

    n_metrics = len(metrics_to_show)
    n_backends = len(df)

    # Create subplots
    n_cols = min(4, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

    backends = df["backend"].values
    colors = plt.cm.Set2(np.linspace(0, 1, n_backends))

    for idx, metric in enumerate(metrics_to_show):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        if metric in df.columns:
            values = df[metric].values

            bars = ax.bar(
                range(n_backends),
                values,
                color=colors,
                edgecolor="black",
                linewidth=0.5,
            )

            ax.set_xticks(range(n_backends))
            ax.set_xticklabels(backends, rotation=45, ha="right", fontsize=9)
            ax.set_title(_format_metric_name(metric), fontsize=10)
            ax.set_ylim(0, max(values.max() * 1.1, 0.1))

            # Add value labels
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        else:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center")
            ax.set_title(_format_metric_name(metric), fontsize=10)

    # Hide unused axes
    for idx in range(n_metrics, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def plot_confidence_distributions(
    results: dict[str, BackendMappingResult | StandardizedOutput],
    figsize: tuple[float, float] = (10, 4),
    output_path: Path | None = None,
) -> plt.Figure:
    """
    Create histograms of confidence distributions per backend.

    Args:
        results: Dictionary mapping backend name to result
        figsize: Figure size
        output_path: Optional path to save figure

    Returns:
        Matplotlib Figure
    """
    backends = list(results.keys())
    n_backends = len(backends)

    fig, axes = plt.subplots(1, n_backends, figsize=figsize, squeeze=False)

    colors = plt.cm.Set2(np.linspace(0, 1, n_backends))

    for idx, backend_name in enumerate(backends):
        ax = axes[0, idx]
        result = results[backend_name]

        confidence = _get_confidence(result)

        ax.hist(
            confidence,
            bins=30,
            color=colors[idx],
            edgecolor="black",
            linewidth=0.5,
            alpha=0.8,
        )

        ax.axvline(
            confidence.mean(),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {confidence.mean():.2f}",
        )

        ax.set_xlabel("Confidence", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.set_title(backend_name.upper(), fontsize=12, fontweight="bold")
        ax.set_xlim(0, 1)
        ax.legend(loc="upper left", fontsize=8)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def plot_donor_robustness(
    robustness_by_backend: dict[str, dict[str, float]],
    figsize: tuple[float, float] = (10, 5),
    output_path: Path | None = None,
) -> plt.Figure:
    """
    Create robustness comparison plots across donors.

    Args:
        robustness_by_backend: Dictionary mapping backend to robustness metrics
        figsize: Figure size
        output_path: Optional path to save figure

    Returns:
        Matplotlib Figure
    """
    if not robustness_by_backend:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No robustness data", ha="center", va="center")
        return fig

    # Convert to DataFrame
    df = pd.DataFrame(robustness_by_backend).T
    df.index.name = "backend"

    # Select key metrics
    metrics = [
        "donor_consistency",
        "celltype_stability",
        "confidence_stability",
        "entropy_stability",
    ]
    metrics = [m for m in metrics if m in df.columns]

    if not metrics:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No robustness metrics", ha="center", va="center")
        return fig

    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(df))
    width = 0.8 / len(metrics)

    colors = plt.cm.Set3(np.linspace(0, 1, len(metrics)))

    for idx, metric in enumerate(metrics):
        offset = (idx - len(metrics) / 2 + 0.5) * width
        values = df[metric].fillna(0).values

        ax.bar(
            x + offset,
            values,
            width,
            label=_format_metric_name(metric),
            color=colors[idx],
            edgecolor="black",
            linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(df.index, fontsize=10)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Donor Robustness Comparison", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.legend(loc="upper right", fontsize=9)
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def plot_entropy_comparison(
    results: dict[str, BackendMappingResult | StandardizedOutput],
    spatial_coords: np.ndarray,
    figsize: tuple[float, float] | None = None,
    output_path: Path | None = None,
) -> plt.Figure:
    """
    Create spatial entropy maps for each backend.

    Args:
        results: Dictionary mapping backend name to result
        spatial_coords: Spatial coordinates
        figsize: Figure size
        output_path: Optional path to save figure

    Returns:
        Matplotlib Figure
    """
    from .base import compute_cell_type_entropy

    backends = list(results.keys())
    n_backends = len(backends)

    if figsize is None:
        figsize = (4 * n_backends, 4)

    fig, axes = plt.subplots(1, n_backends, figsize=figsize, squeeze=False)

    for idx, backend_name in enumerate(backends):
        ax = axes[0, idx]
        result = results[backend_name]
        proportions = _get_proportions(result)

        entropy = compute_cell_type_entropy(proportions)

        scatter = ax.scatter(
            spatial_coords[:, 0],
            spatial_coords[:, 1],
            c=entropy,
            cmap="viridis",
            s=10,
            vmin=0,
            vmax=1,
            alpha=0.8,
        )

        ax.set_title(
            f"{backend_name.upper()}\n(mean: {entropy.mean():.2f})", fontsize=11, fontweight="bold"
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

        plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label="Entropy")

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def create_comparison_summary_figure(
    comparison_result: ComparisonResult,
    results: dict[str, BackendMappingResult | StandardizedOutput],
    spatial_coords: np.ndarray,
    output_path: Path | None = None,
) -> plt.Figure:
    """
    Create comprehensive summary figure with all comparison visualizations.

    Args:
        comparison_result: Full comparison results
        results: Dictionary of backend results
        spatial_coords: Spatial coordinates
        output_path: Optional path to save figure

    Returns:
        Matplotlib Figure
    """
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

    backends = list(results.keys())
    len(backends)

    # Row 1: Spatial maps for dominant cell type
    for idx, backend_name in enumerate(backends[:4]):
        ax = fig.add_subplot(gs[0, idx])
        result = results[backend_name]
        proportions = _get_proportions(result)

        # Show dominant cell type
        dominant = proportions.idxmax(axis=1)
        unique_types = dominant.unique()
        type_to_int = {t: i for i, t in enumerate(unique_types)}
        colors = [type_to_int[t] for t in dominant]

        ax.scatter(
            spatial_coords[:, 0],
            spatial_coords[:, 1],
            c=colors,
            cmap="tab20",
            s=8,
            alpha=0.7,
        )
        ax.set_title(backend_name.upper(), fontsize=10, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

    # Row 2: Metrics comparison
    ax_metrics = fig.add_subplot(gs[1, :2])
    if comparison_result.comparison_table is not None:
        df = comparison_result.comparison_table[comparison_result.comparison_table["success"]]

        # Key metrics
        key_metrics = []
        for prefix in ["upstream_", "downstream_", "spatial_"]:
            cols = [c for c in df.columns if c.startswith(prefix)]
            key_metrics.extend(cols[:2])

        if key_metrics:
            metric_df = df[["backend"] + key_metrics].set_index("backend")
            metric_df.plot(kind="bar", ax=ax_metrics, width=0.8)
            ax_metrics.set_xticklabels(metric_df.index, rotation=45, ha="right")
            ax_metrics.legend(fontsize=8, loc="upper right")
            ax_metrics.set_title("Key Metrics Comparison", fontsize=10, fontweight="bold")

    # Row 2: Confidence distributions
    ax_conf = fig.add_subplot(gs[1, 2:])
    for idx, backend_name in enumerate(backends):
        result = results[backend_name]
        confidence = _get_confidence(result)
        ax_conf.hist(
            confidence,
            bins=30,
            alpha=0.5,
            label=backend_name,
        )
    ax_conf.set_xlabel("Confidence")
    ax_conf.set_ylabel("Count")
    ax_conf.set_title("Confidence Distributions", fontsize=10, fontweight="bold")
    ax_conf.legend(fontsize=9)

    # Row 3: Runtime and Rankings
    ax_runtime = fig.add_subplot(gs[2, :2])
    if comparison_result.comparison_table is not None:
        df = comparison_result.comparison_table
        colors = ["green" if s else "red" for s in df["success"]]
        ax_runtime.barh(df["backend"], df["runtime_seconds"], color=colors, alpha=0.7)
        ax_runtime.set_xlabel("Runtime (seconds)")
        ax_runtime.set_title("Runtime Comparison", fontsize=10, fontweight="bold")

    # Rankings text
    ax_rank = fig.add_subplot(gs[2, 2:])
    ax_rank.axis("off")

    ranking_text = ["RANKINGS", "=" * 30]
    for criterion, ranking in comparison_result.rankings.items():
        ranking_text.append(f"{criterion.upper()}: {' > '.join(ranking)}")

    ax_rank.text(
        0.1,
        0.9,
        "\n".join(ranking_text),
        transform=ax_rank.transAxes,
        fontsize=10,
        fontfamily="monospace",
        verticalalignment="top",
    )

    plt.suptitle("Spatial Backend Comparison Summary", fontsize=14, fontweight="bold", y=0.98)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def _get_proportions(
    result: BackendMappingResult | StandardizedOutput,
) -> pd.DataFrame:
    """Extract proportions from either result type."""
    if isinstance(result, StandardizedOutput):
        return result.cell_type_proportions
    return result.cell_type_proportions


def _get_confidence(
    result: BackendMappingResult | StandardizedOutput,
) -> pd.Series:
    """Extract confidence from either result type."""
    if isinstance(result, StandardizedOutput):
        return result.confidence
    return result.confidence


def _format_metric_name(name: str) -> str:
    """Format metric name for display."""
    # Remove prefixes
    for prefix in ["upstream_", "downstream_", "spatial_", "robustness_"]:
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break

    # Convert underscores to spaces and title case
    return name.replace("_", " ").title()
