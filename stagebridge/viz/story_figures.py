"""Poster- and manuscript-facing benchmark figures for the StageBridge story."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PALETTE = {
    "bg": "#F8F5EE",
    "text": "#172033",
    "grid": "#C9C4B8",
    "stagebridge": "#0E7490",
    "pooled": "#C2410C",
    "graph": "#7C3AED",
    "baseline": "#475569",
    "ablation": "#94A3B8",
    "positive": "#15803D",
    "negative": "#B91C1C",
}


def _save(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240, bbox_inches="tight", facecolor=PALETTE["bg"])
    if output_path.suffix.lower() != ".pdf":
        fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)


def _model_color(label: str) -> str:
    key = str(label).lower()
    if key in {"set_only", "stagebridge"}:
        return PALETTE["stagebridge"]
    if key == "pooled":
        return PALETTE["pooled"]
    if "graph" in key:
        return PALETTE["graph"]
    if "transformer" in key or "relay" in key:
        return PALETTE["ablation"]
    return PALETTE["baseline"]


def plot_transition_vs_communication(
    transition_df: pd.DataFrame,
    communication_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot the core positive and negative benchmark stories side by side."""
    if transition_df.empty:
        raise ValueError("transition_df is empty")
    if communication_df.empty:
        raise ValueError("communication_df is empty")

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), facecolor=PALETTE["bg"])
    for ax in axes:
        ax.set_facecolor(PALETTE["bg"])
        ax.grid(axis="y", alpha=0.25, color=PALETTE["grid"])
        ax.spines[["top", "right"]].set_visible(False)

    transition_plot = transition_df.copy().sort_values("primary_metric", ascending=True)
    x_left = np.arange(transition_plot.shape[0])
    left_colors = [_model_color(label) for label in transition_plot["mode"]]
    axes[0].bar(x_left, transition_plot["primary_metric"].astype(float).values, color=left_colors, alpha=0.92)
    axes[0].set_xticks(x_left)
    axes[0].set_xticklabels(transition_plot["mode"].astype(str), rotation=25, ha="right")
    axes[0].set_ylabel("Sinkhorn distance")
    axes[0].set_title("Transition Benchmark: AIS->MIA\nLower is better")

    communication_plot = communication_df.copy().sort_values("auroc_mean", ascending=False)
    x_right = np.arange(communication_plot.shape[0])
    right_colors = [_model_color(label) for label in communication_plot["model_name"]]
    axes[1].bar(
        x_right,
        communication_plot["auroc_mean"].astype(float).values,
        yerr=communication_plot["auroc_std"].fillna(0.0).astype(float).values,
        color=right_colors,
        alpha=0.92,
        capsize=3,
    )
    axes[1].set_xticks(x_right)
    axes[1].set_xticklabels(communication_plot["model_name"].astype(str), rotation=35, ha="right")
    axes[1].set_ylabel("AUROC")
    axes[1].set_title("Communication Benchmark: AIS proxy\nHigher is better")

    fig.suptitle("StageBridge Story: Compact Set Attention Helps, Rich CCC Attention Does Not Yet", fontsize=15, color=PALETTE["text"])
    fig.tight_layout()
    _save(fig, output_path)


def plot_communication_metric_panels(
    communication_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot AUROC and AUPRC panels for the communication benchmark."""
    if communication_df.empty:
        raise ValueError("communication_df is empty")
    plot_df = communication_df.copy().sort_values("auroc_mean", ascending=False)
    x = np.arange(plot_df.shape[0])
    colors = [_model_color(label) for label in plot_df["model_name"]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), facecolor=PALETTE["bg"])
    for ax, metric, title in [
        (axes[0], "auroc_mean", "Communication Benchmark AUROC"),
        (axes[1], "auprc_mean", "Communication Benchmark AUPRC"),
    ]:
        err = plot_df[metric.replace("_mean", "_std")].fillna(0.0).astype(float).values if metric.replace("_mean", "_std") in plot_df.columns else None
        ax.bar(x, plot_df[metric].astype(float).values, yerr=err, color=colors, alpha=0.92, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(plot_df["model_name"].astype(str), rotation=35, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25, color=PALETTE["grid"])
        ax.set_facecolor(PALETTE["bg"])
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("AUROC")
    axes[1].set_ylabel("AUPRC")
    fig.tight_layout()
    _save(fig, output_path)


def plot_context_shuffle_deltas(
    shuffle_df: pd.DataFrame,
    output_path: Path,
    metric_col: str = "context_shuffle_auroc_delta_mean",
) -> None:
    """Plot context-shuffle degradation by model family."""
    if shuffle_df.empty:
        raise ValueError("shuffle_df is empty")
    plot_df = shuffle_df.copy().sort_values(metric_col, ascending=False)
    x = np.arange(plot_df.shape[0])
    colors = [
        PALETTE["positive"] if float(val) >= 0.0 else PALETTE["negative"]
        for val in plot_df[metric_col].astype(float).values
    ]

    fig, ax = plt.subplots(figsize=(10.5, 5.4), facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])
    ax.bar(x, plot_df[metric_col].astype(float).values, color=colors, alpha=0.92)
    ax.axhline(0.0, color=PALETTE["text"], linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["model_name"].astype(str), rotation=35, ha="right")
    ax.set_ylabel("AUROC drop after context shuffle")
    ax.set_title("Context Reliance Diagnostic")
    ax.grid(axis="y", alpha=0.25, color=PALETTE["grid"])
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, output_path)


def plot_label_balance(label_balance_df: pd.DataFrame, output_path: Path) -> None:
    """Plot positive/negative bag counts by edge."""
    if label_balance_df.empty:
        raise ValueError("label_balance_df is empty")
    plot_df = label_balance_df.copy()
    x = np.arange(plot_df.shape[0])
    negatives = plot_df["negative_bags"].astype(float).values
    positives = plot_df["positive_bags"].astype(float).values

    fig, ax = plt.subplots(figsize=(8.5, 5.0), facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])
    ax.bar(x, negatives, color=PALETTE["negative"], alpha=0.9, label="negative")
    ax.bar(x, positives, bottom=negatives, color=PALETTE["positive"], alpha=0.9, label="positive")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["edge_label"].astype(str))
    ax.set_ylabel("Number of sample-edge bags")
    ax.set_title("Communication Label Balance")
    ax.grid(axis="y", alpha=0.25, color=PALETTE["grid"])
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, output_path)


__all__ = [
    "plot_communication_metric_panels",
    "plot_context_shuffle_deltas",
    "plot_label_balance",
    "plot_transition_vs_communication",
]
