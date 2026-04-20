"""Ablation study waterfall visualizations for StageBridge.

Shows ordered performance degradation with:
- Ranked ablations by impact
- Confidence intervals
- Component importance ranking

For publication-quality figures.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from stagebridge.viz.lungpca_style import (
    configure_lungpca_style,
    save_lungpca_figure,
)


ABLATION_LABELS = {
    "no_niche": "No Niche Context",
    "no_wes": "No WES/Genomics",
    "pooled_niche": "Pooled (Self-Attn)",
    "flat_hierarchy": "Flat Hierarchy",
    "hlca_only": "HLCA Only",
    "luca_only": "LuCA Only",
    "deterministic": "Deterministic",
    "with_prototypes": "With Prototypes",
}

COMPONENT_GROUPS = {
    "Niche Modeling": ["no_niche", "pooled_niche"],
    "Dual Reference": ["hlca_only", "luca_only"],
    "Architecture": ["flat_hierarchy", "deterministic"],
    "Features": ["no_wes", "with_prototypes"],
}


def compute_degradation(
    ablation_results: dict[str, dict],
    full_model_results: dict,
    metric: str = "wasserstein",
) -> pd.DataFrame:
    """Compute performance degradation for each ablation.

    Args:
        ablation_results: Dict of ablation name -> metrics dict
        full_model_results: Full model metrics dict
        metric: Metric to compute degradation for

    Returns:
        DataFrame with degradation info sorted by impact
    """
    rows = []

    full_val = full_model_results.get(metric, 0)
    if isinstance(full_val, dict):
        full_val = full_val.get("mean", 0)

    for ablation, metrics in ablation_results.items():
        abl_val = metrics.get(metric, 0)
        if isinstance(abl_val, dict):
            abl_val = abl_val.get("mean", 0)

        if metric in ["wasserstein", "mmd", "mse", "mae"]:
            degradation = (abl_val - full_val) / (abs(full_val) + 1e-10) * 100
            direction = "higher"
        else:
            degradation = (full_val - abl_val) / (abs(full_val) + 1e-10) * 100
            direction = "lower"

        rows.append({
            "ablation": ablation,
            "full_model": full_val,
            "ablation_value": abl_val,
            "degradation_pct": degradation,
            "direction": direction,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("degradation_pct", ascending=False)

    return df


def plot_ablation_waterfall(
    degradation_df: pd.DataFrame,
    output_path: Path | None = None,
    metric_name: str = "Wasserstein Distance",
    figsize: tuple = (12, 8),
) -> plt.Figure:
    """Create waterfall plot showing ablation impact.

    Args:
        degradation_df: DataFrame from compute_degradation
        output_path: Optional save path
        metric_name: Display name for metric
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    configure_lungpca_style()

    fig, ax = plt.subplots(figsize=figsize)

    n_ablations = len(degradation_df)
    y_positions = np.arange(n_ablations)

    colors = []
    for deg in degradation_df["degradation_pct"]:
        if deg > 10:
            colors.append("#d62728")
        elif deg > 5:
            colors.append("#ff7f0e")
        elif deg > 0:
            colors.append("#ffbb78")
        elif deg > -5:
            colors.append("#98df8a")
        else:
            colors.append("#2ca02c")

    bars = ax.barh(
        y_positions,
        degradation_df["degradation_pct"],
        color=colors,
        edgecolor="black",
        linewidth=0.5,
        height=0.7,
    )

    for bar, deg, abl_val in zip(bars, degradation_df["degradation_pct"], degradation_df["ablation_value"]):
        width = bar.get_width()
        label_x = width + 0.5 if width >= 0 else width - 0.5
        ha = "left" if width >= 0 else "right"

        ax.text(label_x, bar.get_y() + bar.get_height() / 2,
               f"{deg:+.1f}% ({abl_val:.3f})",
               va="center", ha=ha, fontsize=9, fontweight="bold")

    labels = [ABLATION_LABELS.get(abl, abl.replace("_", " ").title())
              for abl in degradation_df["ablation"]]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=10)

    ax.axvline(0, color="black", linewidth=1.5, linestyle="-")

    ax.set_xlabel("Performance Degradation (%)", fontsize=12)
    ax.set_title(f"Ablation Study: Impact on {metric_name}",
                fontsize=14, fontweight="bold")

    full_val = degradation_df["full_model"].iloc[0]
    ax.text(0.98, 0.02, f"Full Model: {full_val:.4f}",
           transform=ax.transAxes, ha="right", va="bottom",
           fontsize=10, style="italic",
           bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    legend_elements = [
        mpatches.Patch(facecolor="#d62728", label=">10% degradation"),
        mpatches.Patch(facecolor="#ff7f0e", label="5-10% degradation"),
        mpatches.Patch(facecolor="#ffbb78", label="0-5% degradation"),
        mpatches.Patch(facecolor="#98df8a", label="0-5% improvement"),
        mpatches.Patch(facecolor="#2ca02c", label=">5% improvement"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8, framealpha=0.9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if output_path:
        save_lungpca_figure(fig, output_path)

    return fig


def plot_multi_metric_waterfall(
    ablation_results: dict[str, dict],
    full_model_results: dict,
    output_path: Path | None = None,
    metrics: list[str] | None = None,
    figsize: tuple = (16, 10),
) -> plt.Figure:
    """Create waterfall plots for multiple metrics.

    Args:
        ablation_results: Dict of ablation name -> metrics
        full_model_results: Full model metrics
        output_path: Optional save path
        metrics: List of metrics to show
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    configure_lungpca_style()

    if metrics is None:
        metrics = ["wasserstein", "stage_accuracy", "auroc"]

    metric_labels = {
        "wasserstein": "Wasserstein Distance",
        "mmd": "MMD",
        "stage_accuracy": "Stage Accuracy",
        "auroc": "AUROC",
        "mse": "MSE",
    }

    fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        deg_df = compute_degradation(ablation_results, full_model_results, metric)

        n = len(deg_df)
        y_pos = np.arange(n)

        colors = ["#d62728" if d > 5 else "#2ca02c" if d < -5 else "#7f7f7f"
                 for d in deg_df["degradation_pct"]]

        ax.barh(y_pos, deg_df["degradation_pct"], color=colors,
               edgecolor="black", linewidth=0.5, height=0.7)

        ax.axvline(0, color="black", linewidth=1)

        labels = [ABLATION_LABELS.get(a, a[:8]) for a in deg_df["ablation"]]
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)

        ax.set_xlabel("Degradation (%)", fontsize=10)
        ax.set_title(metric_labels.get(metric, metric), fontsize=12, fontweight="bold")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.suptitle("Ablation Impact Across Metrics", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if output_path:
        save_lungpca_figure(fig, output_path)

    return fig


def plot_component_importance(
    ablation_results: dict[str, dict],
    full_model_results: dict,
    output_path: Path | None = None,
    metric: str = "wasserstein",
    figsize: tuple = (10, 8),
) -> plt.Figure:
    """Create component importance ranking figure.

    Groups ablations by component type and shows average degradation.

    Args:
        ablation_results: Dict of ablation name -> metrics
        full_model_results: Full model metrics
        output_path: Optional save path
        metric: Metric to use
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    configure_lungpca_style()

    full_val = full_model_results.get(metric, 0)
    if isinstance(full_val, dict):
        full_val = full_val.get("mean", 0)

    group_impacts = {}
    for group_name, ablations in COMPONENT_GROUPS.items():
        degradations = []
        for abl in ablations:
            if abl in ablation_results:
                abl_val = ablation_results[abl].get(metric, full_val)
                if isinstance(abl_val, dict):
                    abl_val = abl_val.get("mean", full_val)
                deg = (abl_val - full_val) / (abs(full_val) + 1e-10) * 100
                degradations.append(deg)

        if degradations:
            group_impacts[group_name] = np.mean(degradations)

    sorted_groups = sorted(group_impacts.items(), key=lambda x: x[1], reverse=True)

    fig, ax = plt.subplots(figsize=figsize)

    groups = [g[0] for g in sorted_groups]
    impacts = [g[1] for g in sorted_groups]

    colors = ["#d62728" if i > 5 else "#2ca02c" if i < -5 else "#7f7f7f" for i in impacts]

    bars = ax.barh(groups, impacts, color=colors, edgecolor="black", linewidth=1, height=0.6)

    ax.axvline(0, color="black", linewidth=1.5)

    for bar, impact in zip(bars, impacts):
        width = bar.get_width()
        label_x = width + 0.3 if width >= 0 else width - 0.3
        ha = "left" if width >= 0 else "right"
        ax.text(label_x, bar.get_y() + bar.get_height() / 2,
               f"{impact:+.1f}%", va="center", ha=ha, fontsize=11, fontweight="bold")

    ax.set_xlabel("Average Performance Degradation (%)", fontsize=12)
    ax.set_title("Component Importance Ranking", fontsize=14, fontweight="bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.text(0.98, 0.02,
           "Higher degradation = more important component",
           transform=ax.transAxes, ha="right", va="bottom",
           fontsize=9, style="italic",
           bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()

    if output_path:
        save_lungpca_figure(fig, output_path)

    return fig


def plot_ablation_summary(
    ablation_results: dict[str, dict],
    full_model_results: dict,
    output_path: Path | None = None,
    primary_metric: str = "wasserstein",
) -> plt.Figure:
    """Create combined ablation summary figure.

    Three panels:
    A. Waterfall for primary metric
    B. Multi-metric comparison
    C. Component importance ranking

    Args:
        ablation_results: Dict of ablation name -> metrics
        full_model_results: Full model metrics
        output_path: Optional save path
        primary_metric: Primary metric for panel A

    Returns:
        matplotlib Figure
    """
    configure_lungpca_style()

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Panel A: Primary metric waterfall
    ax_water = fig.add_subplot(gs[0, 0])
    deg_df = compute_degradation(ablation_results, full_model_results, primary_metric)

    n = len(deg_df)
    y_pos = np.arange(n)

    colors = ["#d62728" if d > 10 else "#ff7f0e" if d > 5 else "#ffbb78" if d > 0
              else "#98df8a" if d > -5 else "#2ca02c" for d in deg_df["degradation_pct"]]

    ax_water.barh(y_pos, deg_df["degradation_pct"], color=colors,
                  edgecolor="black", linewidth=0.5, height=0.7)
    ax_water.axvline(0, color="black", linewidth=1.5)

    labels = [ABLATION_LABELS.get(a, a.replace("_", " ").title()) for a in deg_df["ablation"]]
    ax_water.set_yticks(y_pos)
    ax_water.set_yticklabels(labels, fontsize=9)
    ax_water.set_xlabel("Degradation (%)", fontsize=10)
    ax_water.set_title(f"A. Ablation Impact ({primary_metric.replace('_', ' ').title()})",
                       fontsize=12, fontweight="bold")
    ax_water.spines["top"].set_visible(False)
    ax_water.spines["right"].set_visible(False)

    # Panel B: Component importance
    ax_comp = fig.add_subplot(gs[0, 1])

    full_val = full_model_results.get(primary_metric, 0)
    if isinstance(full_val, dict):
        full_val = full_val.get("mean", 0)

    group_impacts = {}
    for group_name, ablations in COMPONENT_GROUPS.items():
        degradations = []
        for abl in ablations:
            if abl in ablation_results:
                abl_val = ablation_results[abl].get(primary_metric, full_val)
                if isinstance(abl_val, dict):
                    abl_val = abl_val.get("mean", full_val)
                deg = (abl_val - full_val) / (abs(full_val) + 1e-10) * 100
                degradations.append(deg)
        if degradations:
            group_impacts[group_name] = np.mean(degradations)

    sorted_groups = sorted(group_impacts.items(), key=lambda x: x[1], reverse=True)
    groups = [g[0] for g in sorted_groups]
    impacts = [g[1] for g in sorted_groups]

    colors = ["#d62728" if i > 5 else "#2ca02c" if i < -5 else "#7f7f7f" for i in impacts]
    ax_comp.barh(groups, impacts, color=colors, edgecolor="black", linewidth=1, height=0.6)
    ax_comp.axvline(0, color="black", linewidth=1.5)
    ax_comp.set_xlabel("Avg Degradation (%)", fontsize=10)
    ax_comp.set_title("B. Component Importance", fontsize=12, fontweight="bold")
    ax_comp.spines["top"].set_visible(False)
    ax_comp.spines["right"].set_visible(False)

    # Panel C: Multi-metric comparison
    ax_multi = fig.add_subplot(gs[1, :])

    metrics = ["wasserstein", "stage_accuracy", "auroc"]
    ablation_names = list(ablation_results.keys())
    x = np.arange(len(ablation_names))
    width = 0.25

    for i, metric in enumerate(metrics):
        degs = []
        for abl in ablation_names:
            abl_val = ablation_results[abl].get(metric, 0)
            full_val = full_model_results.get(metric, 0)
            if isinstance(abl_val, dict):
                abl_val = abl_val.get("mean", 0)
            if isinstance(full_val, dict):
                full_val = full_val.get("mean", 0)

            if metric in ["wasserstein", "mmd", "mse"]:
                deg = (abl_val - full_val) / (abs(full_val) + 1e-10) * 100
            else:
                deg = (full_val - abl_val) / (abs(full_val) + 1e-10) * 100
            degs.append(deg)

        ax_multi.bar(x + i * width, degs, width, label=metric.replace("_", " ").title())

    ax_multi.axhline(0, color="black", linewidth=1)
    ax_multi.set_xticks(x + width)
    ax_multi.set_xticklabels([ABLATION_LABELS.get(a, a[:10]) for a in ablation_names],
                             rotation=45, ha="right", fontsize=9)
    ax_multi.set_ylabel("Degradation (%)", fontsize=10)
    ax_multi.set_title("C. Multi-Metric Ablation Comparison", fontsize=12, fontweight="bold")
    ax_multi.legend(fontsize=9)
    ax_multi.spines["top"].set_visible(False)
    ax_multi.spines["right"].set_visible(False)

    plt.suptitle("Ablation Study Summary: Component Contributions to Model Performance",
                fontsize=14, fontweight="bold", y=1.02)

    if output_path:
        save_lungpca_figure(fig, output_path)

    return fig
