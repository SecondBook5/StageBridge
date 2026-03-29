"""
Visualizations for abundance-stratified deconvolution benchmarking.

This module contains functions to create visualizations that compare deconvolution backends across cell types stratified by abundance. It also includes unique visualizations that show how backend performance varies across progression stages and which rare cell types are most relevant to disease progression.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List
from .abundance_metrics import AbundanceStratification


def plot_backend_comparison_boxplot(
    backend_results: Dict[str, pd.DataFrame],
    stratification: AbundanceStratification,
    figsize: tuple = (12, 6),
    save_path: str | None = None,
) -> plt.Figure:
    """
    Create boxplot comparing backends across abundance categories.

    Args:
        backend_results: Dict mapping backend name to correlation DataFrame
                        (output from compute_correlation_by_abundance)
        stratification: Abundance stratification
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    # Combine all backend results
    combined = []
    for backend_name, corr_df in backend_results.items():
        df = corr_df.copy()
        df["backend"] = backend_name
        combined.append(df)

    combined_df = pd.concat(combined, ignore_index=True)

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)

    abundance_categories = ["abundant", "medium", "rare"]
    colors = {"abundant": "#2ecc71", "medium": "#f39c12", "rare": "#e74c3c"}

    for i, category in enumerate(abundance_categories):
        ax = axes[i]

        # Filter to this abundance category
        cat_data = combined_df[combined_df["abundance_category"] == category]

        # Boxplot
        sns.boxplot(
            data=cat_data,
            x="backend",
            y="correlation",
            ax=ax,
            color=colors[category],
            showfliers=True,
        )

        # Overlay points
        sns.stripplot(
            data=cat_data,
            x="backend",
            y="correlation",
            ax=ax,
            color="black",
            alpha=0.3,
            size=3,
        )

        ax.set_title(f"{category.capitalize()} cell types", fontsize=12, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Correlation with ground truth" if i == 0 else "")
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_ylim(-0.2, 1.0)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_backend_comparison_heatmap(
    backend_results: Dict[str, pd.DataFrame],
    stratification: AbundanceStratification,
    metric: str = "correlation",
    figsize: tuple = (10, 6),
    save_path: str | None = None,
) -> plt.Figure:
    """
    Create heatmap of backend performance by cell type.

    Args:
        backend_results: Dict mapping backend name to correlation DataFrame
        stratification: Abundance stratification
        metric: Which metric to plot ("correlation", "f1_score", etc.)
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    # Build matrix: cell types × backends
    all_cell_types = stratification.mean_proportions.index
    backend_names = list(backend_results.keys())

    matrix = pd.DataFrame(
        index=all_cell_types,
        columns=backend_names,
        dtype=float
    )

    for backend_name, corr_df in backend_results.items():
        for _, row in corr_df.iterrows():
            ct = row["cell_type"]
            matrix.loc[ct, backend_name] = row[metric]

    # Sort by abundance category and proportion
    matrix["abundance_category"] = [
        stratification.get_category(ct) for ct in matrix.index
    ]
    matrix["mean_proportion"] = [
        stratification.mean_proportions[ct] for ct in matrix.index
    ]

    # Sort: abundant first, then by mean proportion within category
    category_order = {"abundant": 0, "medium": 1, "rare": 2}
    matrix["sort_key"] = matrix["abundance_category"].map(category_order)
    matrix = matrix.sort_values(["sort_key", "mean_proportion"], ascending=[True, False])

    # Drop auxiliary columns
    plot_matrix = matrix.drop(columns=["abundance_category", "mean_proportion", "sort_key"])

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        plot_matrix,
        cmap="RdYlGn",
        center=0.5,
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".2f",
        cbar_kws={"label": metric.replace("_", " ").title()},
        ax=ax,
    )

    ax.set_xlabel("Backend", fontsize=12)
    ax.set_ylabel("Cell Type", fontsize=12)
    ax.set_title(f"Backend Performance by Cell Type\n({metric})", fontsize=14, fontweight="bold")

    # Add abundance category separators
    abundant_end = len(stratification.abundant)
    medium_end = abundant_end + len(stratification.medium)

    ax.axhline(abundant_end, color="black", linewidth=2)
    ax.axhline(medium_end, color="black", linewidth=2)

    # Add abundance labels on right
    ax.text(
        len(backend_names) + 0.5,
        abundant_end / 2,
        "Abundant",
        va="center",
        fontweight="bold",
        rotation=270,
    )
    ax.text(
        len(backend_names) + 0.5,
        abundant_end + (medium_end - abundant_end) / 2,
        "Medium",
        va="center",
        fontweight="bold",
        rotation=270,
    )
    ax.text(
        len(backend_names) + 0.5,
        medium_end + (len(plot_matrix) - medium_end) / 2,
        "Rare",
        va="center",
        fontweight="bold",
        rotation=270,
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_progression_specific_comparison(
    backend_results: Dict[str, pd.DataFrame],
    stage_order: List[str],
    figsize: tuple = (14, 10),
    save_path: str | None = None,
) -> plt.Figure:
    """
    UNIQUE CONTRIBUTION: Plot backend performance across progression stages.

    Creates multi-panel figure showing how each backend performs
    for rare cell types at different stages.

    Args:
        backend_results: Dict mapping backend name to stage-specific correlation DataFrame
                        (output from compute_progression_specific_metrics)
        stage_order: Ordered list of stages (e.g., ["AAH", "AIS", "MIA", "LUAD"])
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    n_backends = len(backend_results)
    n_stages = len(stage_order)

    fig, axes = plt.subplots(n_backends, n_stages, figsize=figsize, sharex=True, sharey=True)

    if n_backends == 1:
        axes = axes.reshape(1, -1)

    for i, (backend_name, corr_df) in enumerate(backend_results.items()):
        for j, stage in enumerate(stage_order):
            ax = axes[i, j]

            # Filter to this stage and rare cell types only
            stage_data = corr_df[
                (corr_df["stage"] == stage) &
                (corr_df["abundance_category"] == "rare")
            ]

            if len(stage_data) > 0:
                # Bar plot of correlations
                stage_data_sorted = stage_data.sort_values("correlation", ascending=False)

                colors = ["#e74c3c" if corr < 0.5 else "#2ecc71"
                         for corr in stage_data_sorted["correlation"]]

                ax.barh(
                    range(len(stage_data_sorted)),
                    stage_data_sorted["correlation"],
                    color=colors,
                    alpha=0.7,
                )

                ax.set_yticks(range(len(stage_data_sorted)))
                ax.set_yticklabels(stage_data_sorted["cell_type"], fontsize=8)
                ax.axvline(0.5, color="black", linestyle="--", linewidth=0.5)
                ax.set_xlim(0, 1)

            # Labels
            if i == 0:
                ax.set_title(stage, fontsize=12, fontweight="bold")
            if j == 0:
                ax.set_ylabel(backend_name, fontsize=10, fontweight="bold")
            if i == n_backends - 1:
                ax.set_xlabel("Correlation")

    plt.suptitle("Rare Cell Type Detection by Stage", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_rare_type_stage_enrichment(
    rare_types_df: pd.DataFrame,
    top_n: int = 10,
    figsize: tuple = (10, 6),
    save_path: str | None = None,
) -> plt.Figure:
    """
    UNIQUE CONTRIBUTION: Visualize rare cell types with stage-specific enrichment.

    Shows which rare cell types are most differentially expressed across stages
    (these are the most challenging for deconvolution).

    Args:
        rare_types_df: Output from identify_progression_relevant_rare_types
        top_n: Number of top rare types to show
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    # Take top N by fold change
    top_rare = rare_types_df.head(top_n)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Panel 1: Fold change
    colors = ["#e74c3c" if fc > 10 else "#f39c12" for fc in top_rare["fold_change"]]
    ax1.barh(range(len(top_rare)), top_rare["fold_change"], color=colors, alpha=0.7)
    ax1.set_yticks(range(len(top_rare)))
    ax1.set_yticklabels(top_rare["cell_type"])
    ax1.set_xlabel("Fold Change (max/min stage)")
    ax1.set_title("Stage-Specific Enrichment", fontweight="bold")
    ax1.axvline(1, color="black", linestyle="--", linewidth=0.5)
    ax1.set_xscale("log")

    # Panel 2: Trend correlation
    colors = ["#2ecc71" if tc > 0 else "#e74c3c" for tc in top_rare["trend_correlation"]]
    ax2.barh(range(len(top_rare)), top_rare["trend_correlation"], color=colors, alpha=0.7)
    ax2.set_yticks(range(len(top_rare)))
    ax2.set_yticklabels(top_rare["cell_type"])
    ax2.set_xlabel("Spearman Correlation with Stage Order")
    ax2.set_title("Monotonic Progression Trend", fontweight="bold")
    ax2.axvline(0, color="black", linestyle="--", linewidth=0.5)
    ax2.set_xlim(-1, 1)

    plt.suptitle(
        "Progression-Relevant Rare Cell Types\n(Most Challenging for Deconvolution)",
        fontsize=14,
        fontweight="bold",
        y=1.02
    )
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_abundance_summary_table(
    backend_comparison: pd.DataFrame,
    figsize: tuple = (8, 4),
    save_path: str | None = None,
) -> plt.Figure:
    """
    Create summary table figure showing median performance by abundance.

    Args:
        backend_comparison: Output from compare_backends_by_abundance
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")

    # Format as table
    table_data = backend_comparison.round(3)

    # Add ranking column
    table_data["Overall"] = table_data.mean(axis=1).round(3)
    table_data["Rank"] = table_data["Overall"].rank(ascending=False).astype(int)

    # Sort by rank
    table_data = table_data.sort_values("Rank")

    # Create table
    table = ax.table(
        cellText=table_data.values,
        colLabels=table_data.columns,
        rowLabels=table_data.index,
        cellLoc="center",
        loc="center",
        colWidths=[0.15] * len(table_data.columns),
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Color code cells
    for i in range(len(table_data)):
        for j in range(len(table_data.columns)):
            cell = table[(i + 1, j)]
            value = table_data.iloc[i, j]

            if j < 3:  # Abundance columns
                if value > 0.7:
                    cell.set_facecolor("#d5f4e6")
                elif value > 0.5:
                    cell.set_facecolor("#fff9e6")
                else:
                    cell.set_facecolor("#ffeaa7")

    # Bold headers
    for j in range(len(table_data.columns)):
        table[(0, j)].set_facecolor("#3498db")
        table[(0, j)].set_text_props(weight="bold", color="white")

    plt.title("Backend Performance Summary (Median Correlation)", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig
