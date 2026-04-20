"""SHAP-style niche influence visualizations.

Renders attention-based interpretability from model outputs.
Uses first-class interpretability data from stagebridge.evaluation.interpretability.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

if TYPE_CHECKING:
    from stagebridge.evaluation.interpretability import BatchInterpretability

from stagebridge.viz.lungpca_style import (
    configure_lungpca_style,
    save_lungpca_figure,
    STAGE_COLORS,
    get_celltype_color,
)


def plot_niche_beeswarm(
    influence_df: pd.DataFrame,
    output_path: Path | None = None,
    top_k: int = 15,
    figsize: tuple = (10, 8),
    stage_col: str = "stage",
    type_col: str = "neighbor_type",
    attention_col: str = "attention",
) -> plt.Figure:
    """Create SHAP-style beeswarm plot of niche influence.

    Each point is a cell, x-axis is attention weight (influence),
    y-axis is neighbor cell type, color is stage.

    Args:
        influence_df: DataFrame with columns [cell_id, stage, neighbor_type, attention]
        output_path: Optional path to save figure
        top_k: Number of top cell types to show
        figsize: Figure size
        stage_col: Column name for stage
        type_col: Column name for neighbor cell type
        attention_col: Column name for attention weight

    Returns:
        matplotlib Figure
    """
    configure_lungpca_style()

    # Get top cell types by mean influence
    type_importance = influence_df.groupby(type_col)[attention_col].mean().sort_values(ascending=False)
    top_types = type_importance.head(top_k).index.tolist()

    # Filter and order
    plot_df = influence_df[influence_df[type_col].isin(top_types)].copy()
    plot_df[type_col] = pd.Categorical(
        plot_df[type_col],
        categories=top_types[::-1],
        ordered=True
    )

    fig, ax = plt.subplots(figsize=figsize)

    # Stage colors
    stages = sorted(plot_df[stage_col].unique())
    stage_colors = {s: STAGE_COLORS.get(s, plt.cm.viridis(i / len(stages)))
                   for i, s in enumerate(stages)}

    # Jitter for beeswarm effect
    np.random.seed(42)
    y_jitter = np.random.normal(0, 0.15, len(plot_df))

    for stage in stages:
        mask = plot_df[stage_col] == stage
        stage_df = plot_df[mask]
        y_pos = stage_df[type_col].cat.codes + y_jitter[mask]

        ax.scatter(
            stage_df[attention_col],
            y_pos,
            c=[stage_colors[stage]],
            alpha=0.6,
            s=20,
            label=f"Stage {stage}",
            edgecolors="none",
        )

    # Styling
    ax.set_yticks(range(len(top_types)))
    ax.set_yticklabels(top_types[::-1], fontsize=10)
    ax.set_xlabel("Niche Influence (Attention Weight)", fontsize=12)
    ax.set_ylabel("")
    ax.set_title("Niche Cell Type Influence on Transition Prediction",
                fontsize=14, fontweight="bold")

    # Reference line
    mean_attn = plot_df[attention_col].mean()
    ax.axvline(mean_attn, color="gray", linestyle="--", alpha=0.5)

    ax.legend(loc="upper right", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(left=0)

    plt.tight_layout()

    if output_path:
        save_lungpca_figure(fig, output_path)

    return fig


def plot_niche_importance_bar(
    influence_df: pd.DataFrame,
    output_path: Path | None = None,
    top_k: int = 15,
    figsize: tuple = (10, 6),
    type_col: str = "neighbor_type",
    attention_col: str = "attention",
) -> plt.Figure:
    """Create summary bar plot showing mean influence by cell type.

    Args:
        influence_df: DataFrame with influence data
        output_path: Optional path to save
        top_k: Number of top types to show
        figsize: Figure size
        type_col: Column for cell type
        attention_col: Column for attention

    Returns:
        matplotlib Figure
    """
    configure_lungpca_style()

    # Aggregate
    summary = influence_df.groupby(type_col).agg({
        attention_col: ["mean", "std", "count"]
    }).reset_index()
    summary.columns = [type_col, "mean", "std", "count"]
    summary = summary.sort_values("mean", ascending=True).tail(top_k)

    fig, ax = plt.subplots(figsize=figsize)

    # Color gradient
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(summary)))

    bars = ax.barh(
        summary[type_col],
        summary["mean"],
        xerr=summary["std"] / np.sqrt(summary["count"]),  # SEM
        color=colors,
        edgecolor="black",
        linewidth=0.5,
        capsize=3,
    )

    # Value labels
    for bar, val in zip(bars, summary["mean"]):
        ax.text(
            val + 0.002, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center", ha="left", fontsize=9
        )

    ax.set_xlabel("Mean Niche Influence (Attention Weight)", fontsize=12)
    ax.set_title("Cell Type Contribution to Transition Prediction",
                fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if output_path:
        save_lungpca_figure(fig, output_path)

    return fig


def plot_niche_stage_heatmap(
    influence_df: pd.DataFrame,
    output_path: Path | None = None,
    top_k: int = 12,
    figsize: tuple = (10, 8),
    stage_col: str = "stage",
    type_col: str = "neighbor_type",
    attention_col: str = "attention",
) -> plt.Figure:
    """Create heatmap showing cell type influence by stage.

    Args:
        influence_df: DataFrame with influence data
        output_path: Optional path to save
        top_k: Number of cell types
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    configure_lungpca_style()

    # Pivot table
    pivot = influence_df.groupby([stage_col, type_col])[attention_col].mean().unstack(fill_value=0)

    # Top types
    type_importance = influence_df.groupby(type_col)[attention_col].mean().sort_values(ascending=False)
    top_types = type_importance.head(top_k).index.tolist()
    pivot = pivot[[c for c in top_types if c in pivot.columns]]

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        pivot,
        cmap="YlOrRd",
        annot=True,
        fmt=".3f",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Mean Attention Weight"},
    )

    ax.set_xlabel("Neighbor Cell Type", fontsize=12)
    ax.set_ylabel("Receiver Stage", fontsize=12)
    ax.set_title("Stage-Specific Niche Influence Patterns",
                fontsize=14, fontweight="bold")

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if output_path:
        save_lungpca_figure(fig, output_path)

    return fig


def plot_niche_influence_combined(
    influence_df: pd.DataFrame,
    output_path: Path | None = None,
    top_k: int = 12,
) -> plt.Figure:
    """Create combined multi-panel niche influence figure.

    4 panels:
    A. Beeswarm (SHAP-style)
    B. Importance ranking
    C. Stage heatmap
    D. Transition-associated changes

    Args:
        influence_df: DataFrame with influence data
        output_path: Optional path to save

    Returns:
        matplotlib Figure
    """
    configure_lungpca_style()

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    stage_col = "stage"
    type_col = "neighbor_type"
    attention_col = "attention" if "attention" in influence_df.columns else "influence"

    # Get top types
    type_importance = influence_df.groupby(type_col)[attention_col].mean().sort_values(ascending=False)
    top_types = type_importance.head(top_k).index.tolist()
    plot_df = influence_df[influence_df[type_col].isin(top_types)].copy()
    plot_df[type_col] = pd.Categorical(plot_df[type_col], categories=top_types[::-1], ordered=True)

    stages = sorted(plot_df[stage_col].unique())
    stage_colors = plt.cm.viridis(np.linspace(0, 1, len(stages)))

    # Panel A: Beeswarm
    ax_bee = fig.add_subplot(gs[0, 0])
    np.random.seed(42)
    y_jitter = np.random.normal(0, 0.15, len(plot_df))

    for i, stage in enumerate(stages):
        mask = plot_df[stage_col] == stage
        stage_df = plot_df[mask]
        y_pos = stage_df[type_col].cat.codes + y_jitter[mask]
        ax_bee.scatter(stage_df[attention_col], y_pos, c=[stage_colors[i]],
                      alpha=0.5, s=15, label=f"Stage {stage}")

    ax_bee.set_yticks(range(len(top_types)))
    ax_bee.set_yticklabels(top_types[::-1], fontsize=9)
    ax_bee.set_xlabel("Attention Weight", fontsize=10)
    ax_bee.set_title("A. Niche Influence Distribution", fontsize=12, fontweight="bold")
    ax_bee.legend(loc="upper right", fontsize=8)
    ax_bee.spines["top"].set_visible(False)
    ax_bee.spines["right"].set_visible(False)

    # Panel B: Summary bars
    ax_bar = fig.add_subplot(gs[0, 1])
    summary = influence_df.groupby(type_col)[attention_col].agg(["mean", "std", "count"]).reset_index()
    summary = summary.sort_values("mean", ascending=True).tail(top_k)

    colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(summary)))
    ax_bar.barh(summary[type_col], summary["mean"],
                xerr=summary["std"] / np.sqrt(summary["count"]),
                color=colors, edgecolor="black", linewidth=0.5, capsize=2)
    ax_bar.set_xlabel("Mean Attention Weight", fontsize=10)
    ax_bar.set_title("B. Cell Type Importance Ranking", fontsize=12, fontweight="bold")
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    # Panel C: Stage heatmap
    ax_heat = fig.add_subplot(gs[1, 0])
    pivot = influence_df.groupby([stage_col, type_col])[attention_col].mean().unstack(fill_value=0)
    pivot = pivot[[c for c in top_types if c in pivot.columns]]

    sns.heatmap(pivot, cmap="YlOrRd", annot=True, fmt=".2f", linewidths=0.5, ax=ax_heat,
                cbar_kws={"label": "Attention", "shrink": 0.8}, annot_kws={"size": 8})
    ax_heat.set_xlabel("Neighbor Cell Type", fontsize=10)
    ax_heat.set_ylabel("Stage", fontsize=10)
    ax_heat.set_title("C. Stage-Specific Niche Patterns", fontsize=12, fontweight="bold")
    plt.setp(ax_heat.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Panel D: Stage transition differences
    ax_trans = fig.add_subplot(gs[1, 1])

    if len(stages) > 1:
        stage_means = influence_df.groupby([stage_col, type_col])[attention_col].mean().unstack(fill_value=0)

        diffs = []
        for i in range(len(stages) - 1):
            if stages[i] in stage_means.index and stages[i+1] in stage_means.index:
                diff = stage_means.loc[stages[i+1]] - stage_means.loc[stages[i]]
                diff.name = f"{stages[i]}→{stages[i+1]}"
                diffs.append(diff)

        if diffs:
            diff_df = pd.DataFrame(diffs).T
            diff_df = diff_df.loc[diff_df.abs().sum(axis=1).sort_values(ascending=False).head(top_k).index]

            sns.heatmap(diff_df, cmap="RdBu_r", center=0, annot=True, fmt=".2f",
                       linewidths=0.5, ax=ax_trans, cbar_kws={"label": "Δ Attention", "shrink": 0.8},
                       annot_kws={"size": 8})
            ax_trans.set_xlabel("Stage Transition", fontsize=10)
            ax_trans.set_ylabel("Cell Type", fontsize=10)

    ax_trans.set_title("D. Transition-Associated Changes", fontsize=12, fontweight="bold")

    plt.suptitle("Niche Influence Analysis: Attention-Based Cell Type Attribution",
                fontsize=14, fontweight="bold", y=1.02)

    if output_path:
        save_lungpca_figure(fig, output_path)

    return fig


def from_interpretability(
    batch_interp: "BatchInterpretability",
) -> pd.DataFrame:
    """Convert BatchInterpretability to DataFrame for plotting.

    Args:
        batch_interp: BatchInterpretability from model inference

    Returns:
        DataFrame suitable for niche influence plots
    """
    return batch_interp.to_dataframe()
