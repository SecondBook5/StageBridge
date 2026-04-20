"""Transition flow visualizations - Sankey and alluvial diagrams.

Renders cell state transitions from model outputs.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MPath
import numpy as np
import pandas as pd

from stagebridge.viz.lungpca_style import (
    configure_lungpca_style,
    save_lungpca_figure,
    STAGE_COLORS,
    MAJOR_CELLTYPE_COLORS,
)


def compute_transition_flows(
    cells_df: pd.DataFrame,
    stage_col: str = "stage",
    cell_type_col: str = "cell_type",
    transition_prob_col: str = "transition_prob",
) -> dict:
    """Compute flow statistics between stages.

    Args:
        cells_df: DataFrame with cell data
        stage_col: Column for stage
        cell_type_col: Column for cell type
        transition_prob_col: Column for transition probability (optional)

    Returns:
        Dictionary with flow data
    """
    stages = sorted(cells_df[stage_col].unique())
    cell_types = cells_df[cell_type_col].unique()

    flows = {
        "stages": stages,
        "cell_types": list(cell_types),
        "stage_counts": {},
        "type_by_stage": {},
        "transitions": [],
    }

    for stage in stages:
        stage_df = cells_df[cells_df[stage_col] == stage]
        flows["stage_counts"][stage] = len(stage_df)
        flows["type_by_stage"][stage] = stage_df[cell_type_col].value_counts().to_dict()

    # Compute transitions
    for i in range(len(stages) - 1):
        src_stage = stages[i]
        tgt_stage = stages[i + 1]
        src_df = cells_df[cells_df[stage_col] == src_stage]

        if transition_prob_col in src_df.columns:
            for cell_type in cell_types:
                type_mask = src_df[cell_type_col] == cell_type
                if type_mask.sum() == 0:
                    continue

                mean_prob = src_df.loc[type_mask, transition_prob_col].mean()
                count = type_mask.sum()

                flows["transitions"].append({
                    "source": f"{src_stage}_{cell_type}",
                    "target": f"{tgt_stage}_{cell_type}",
                    "value": count * mean_prob,
                    "cell_type": cell_type,
                    "source_stage": src_stage,
                    "target_stage": tgt_stage,
                    "mean_prob": mean_prob,
                })
        else:
            for cell_type in cell_types:
                src_count = (src_df[cell_type_col] == cell_type).sum()
                tgt_df = cells_df[cells_df[stage_col] == tgt_stage]
                tgt_count = (tgt_df[cell_type_col] == cell_type).sum()

                if src_count > 0 and tgt_count > 0:
                    flows["transitions"].append({
                        "source": f"{src_stage}_{cell_type}",
                        "target": f"{tgt_stage}_{cell_type}",
                        "value": min(src_count, tgt_count),
                        "cell_type": cell_type,
                        "source_stage": src_stage,
                        "target_stage": tgt_stage,
                    })

    return flows


def _draw_flow_band(ax, x0, y0, x1, y1, width, color, alpha=0.6, n_points=50):
    """Draw a curved flow band between two points."""
    cx = (x0 + x1) / 2
    t = np.linspace(0, 1, n_points)

    x_curve = (1-t)**2 * x0 + 2*(1-t)*t * cx + t**2 * x1
    y_offset = (y1 - y0) * 0.1
    y_top = y0 + width/2 + (y1 - y0 + y_offset) * t
    y_bot = y0 - width/2 + (y1 - y0 + y_offset) * t

    verts = list(zip(x_curve, y_top)) + list(zip(x_curve[::-1], y_bot[::-1]))
    verts.append(verts[0])

    codes = [MPath.MOVETO] + [MPath.LINETO] * (len(verts) - 2) + [MPath.CLOSEPOLY]
    path = MPath(verts, codes)
    patch = PathPatch(path, facecolor=color, edgecolor="none", alpha=alpha)
    ax.add_patch(patch)


def plot_sankey_flow(
    cells_df: pd.DataFrame,
    output_path: Path | None = None,
    stage_col: str = "stage",
    cell_type_col: str = "cell_type",
    transition_prob_col: str = "transition_prob",
    top_k_types: int = 8,
    figsize: tuple = (14, 10),
) -> plt.Figure:
    """Create Sankey-style flow diagram.

    Args:
        cells_df: DataFrame with cell data
        output_path: Optional save path
        stage_col: Stage column
        cell_type_col: Cell type column
        transition_prob_col: Transition probability column
        top_k_types: Number of cell types to show
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    configure_lungpca_style()

    flows = compute_transition_flows(cells_df, stage_col, cell_type_col, transition_prob_col)
    stages = flows["stages"]
    transitions = flows["transitions"]

    # Get top types
    type_flows = defaultdict(float)
    for t in transitions:
        type_flows[t["cell_type"]] += t["value"]

    top_types = sorted(type_flows.keys(), key=lambda x: type_flows[x], reverse=True)[:top_k_types]
    transitions = [t for t in transitions if t["cell_type"] in top_types]

    fig, ax = plt.subplots(figsize=figsize)

    # Positions
    stage_x = {s: i / (len(stages) - 1) for i, s in enumerate(stages)}
    type_y = {t: (i + 0.5) / len(top_types) for i, t in enumerate(top_types)}

    # Colors
    colors = plt.cm.tab20(np.linspace(0, 1, len(top_types)))
    type_colors = {t: MAJOR_CELLTYPE_COLORS.get(t, colors[i]) for i, t in enumerate(top_types)}

    # Normalize flows
    max_flow = max(t["value"] for t in transitions) if transitions else 1
    min_width, max_width = 0.005, 0.08

    # Draw flows
    for trans in transitions:
        x0 = stage_x[trans["source_stage"]]
        x1 = stage_x[trans["target_stage"]]
        y0 = type_y[trans["cell_type"]]
        y1 = type_y[trans["cell_type"]]
        width = min_width + (trans["value"] / max_flow) * (max_width - min_width)

        _draw_flow_band(ax, x0, y0, x1, y1, width, type_colors[trans["cell_type"]], alpha=0.6)

    # Stage labels
    for stage in stages:
        x = stage_x[stage]
        ax.axvline(x, color="gray", linestyle="--", alpha=0.3, zorder=0)
        ax.text(x, 1.05, f"Stage {stage}", ha="center", va="bottom",
               fontsize=12, fontweight="bold")

    # Cell type labels
    for cell_type in top_types:
        y = type_y[cell_type]
        ax.text(-0.05, y, cell_type, ha="right", va="center", fontsize=10,
               color=type_colors[cell_type], fontweight="bold")

    ax.set_xlim(-0.15, 1.1)
    ax.set_ylim(-0.05, 1.15)
    ax.axis("off")
    ax.set_title("Cell State Transition Flow Across Disease Stages",
                fontsize=14, fontweight="bold")

    plt.tight_layout()

    if output_path:
        save_lungpca_figure(fig, output_path)

    return fig


def _draw_ribbon(ax, x0, y0, h0, x1, y1, h1, color, alpha=0.4):
    """Draw a ribbon connecting two stacked bar segments."""
    cx = (x0 + x1) / 2
    n_points = 30
    t = np.linspace(0, 1, n_points)

    x_top = (1-t)**2 * x0 + 2*(1-t)*t * cx + t**2 * x1
    y_top = (1-t) * (y0 + h0) + t * (y1 + h1)
    y_bot = (1-t) * y0 + t * y1

    verts = list(zip(x_top, y_top)) + list(zip(x_top[::-1], y_bot[::-1]))
    verts.append(verts[0])

    codes = [MPath.MOVETO] + [MPath.LINETO] * (len(verts) - 2) + [MPath.CLOSEPOLY]
    path = MPath(verts, codes)
    patch = PathPatch(path, facecolor=color, edgecolor="none", alpha=alpha)
    ax.add_patch(patch)


def plot_alluvial(
    cells_df: pd.DataFrame,
    output_path: Path | None = None,
    stage_col: str = "stage",
    cell_type_col: str = "cell_type",
    top_k_types: int = 10,
    figsize: tuple = (12, 8),
) -> plt.Figure:
    """Create alluvial/parallel sets diagram.

    Shows cell type composition changes across stages.

    Args:
        cells_df: DataFrame with cell data
        output_path: Optional save path
        stage_col: Stage column
        cell_type_col: Cell type column
        top_k_types: Number of types to show
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    configure_lungpca_style()

    stages = sorted(cells_df[stage_col].unique())
    type_counts = cells_df[cell_type_col].value_counts()
    top_types = type_counts.head(top_k_types).index.tolist()

    # Proportions per stage
    stage_props = {}
    for stage in stages:
        stage_df = cells_df[cells_df[stage_col] == stage]
        props = stage_df[cell_type_col].value_counts(normalize=True)
        stage_props[stage] = {t: props.get(t, 0) for t in top_types}

    fig, ax = plt.subplots(figsize=figsize)

    x_positions = np.linspace(0, 1, len(stages))
    bar_width = 0.08

    colors = plt.cm.tab20(np.linspace(0, 1, len(top_types)))
    type_colors = {t: MAJOR_CELLTYPE_COLORS.get(t, colors[i]) for i, t in enumerate(top_types)}

    # Draw stacked bars
    for i, stage in enumerate(stages):
        x = x_positions[i]
        bottom = 0

        for cell_type in top_types:
            prop = stage_props[stage][cell_type]
            ax.bar(x, prop, bottom=bottom, width=bar_width,
                  color=type_colors[cell_type], edgecolor="white", linewidth=0.5)
            bottom += prop

    # Draw flow ribbons
    for i in range(len(stages) - 1):
        x0 = x_positions[i] + bar_width / 2
        x1 = x_positions[i + 1] - bar_width / 2

        bottom0 = 0
        bottom1 = 0

        for cell_type in top_types:
            prop0 = stage_props[stages[i]][cell_type]
            prop1 = stage_props[stages[i + 1]][cell_type]

            _draw_ribbon(ax, x0, bottom0, prop0, x1, bottom1, prop1,
                        type_colors[cell_type], alpha=0.4)

            bottom0 += prop0
            bottom1 += prop1

    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"Stage {s}" for s in stages], fontsize=11, fontweight="bold")
    ax.set_ylabel("Cell Type Proportion", fontsize=12)
    ax.set_title("Cell Type Composition Across Disease Stages",
                fontsize=14, fontweight="bold")

    legend_handles = [mpatches.Patch(color=type_colors[t], label=t) for t in top_types]
    ax.legend(handles=legend_handles, loc="center left", bbox_to_anchor=(1.02, 0.5),
             framealpha=0.9, fontsize=9)

    ax.set_xlim(-0.1, 1.15)
    ax.set_ylim(0, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if output_path:
        save_lungpca_figure(fig, output_path)

    return fig
