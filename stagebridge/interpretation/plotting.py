"""Plotting utilities for StageBridge interpretation.

Publication-quality figures adapted from AMICI for:
- Interaction network visualizations (directed graphs, chord diagrams)
- Attention heatmaps and decay profiles
- Stage comparison plots
- Ablation importance rankings
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import seaborn as sns

if TYPE_CHECKING:
    from stagebridge.interpretation.networks import InteractionNetwork
    from stagebridge.interpretation.ablation import AblationModule
    from stagebridge.interpretation.attention import AttentionModule


def set_publication_style():
    """Set matplotlib style for publication figures."""
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.titlesize": 18,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def plot_interaction_network(
    network: "InteractionNetwork",
    palette: dict[str, str] | None = None,
    weight_threshold: float = 0.0,
    node_size: int = 1500,
    figsize: tuple[float, float] = (10, 10),
    title: str | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Plot interaction network as directed graph.

    Args:
        network: InteractionNetwork to visualize
        palette: Dict mapping cell types to colors
        weight_threshold: Minimum edge weight to show
        node_size: Size of nodes
        figsize: Figure size
        title: Plot title
        save_path: Path to save figure
        show: Whether to display

    Returns:
        matplotlib Figure
    """
    import networkx as nx

    set_publication_style()

    G = network.to_networkx(weight_threshold=weight_threshold)

    fig, ax = plt.subplots(figsize=figsize)

    if len(G.nodes()) == 0:
        ax.text(0.5, 0.5, "No significant interactions", ha="center", va="center")
        return fig

    pos = nx.circular_layout(G)

    if palette is None:
        cmap = plt.cm.tab20
        palette = {ct: cmap(i / len(G.nodes())) for i, ct in enumerate(G.nodes())}

    node_colors = [palette.get(node, "#888888") for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_colors, ax=ax)

    if G.edges():
        weights = [G[u][v].get("weight", 0.5) for u, v in G.edges()]
        max_weight = max(abs(w) for w in weights) if weights else 1

        for (u, v), w in zip(G.edges(), weights):
            width = 1 + 4 * abs(w) / max_weight
            alpha = 0.3 + 0.6 * abs(w) / max_weight
            color = "darkred" if w > 0 else "darkblue"

            nx.draw_networkx_edges(
                G, pos,
                edgelist=[(u, v)],
                width=width,
                alpha=alpha,
                edge_color=color,
                arrows=True,
                arrowsize=20,
                connectionstyle="arc3,rad=0.15",
                ax=ax,
            )

    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold", ax=ax)

    if title:
        ax.set_title(title)
    ax.axis("off")

    legend_elements = [
        Line2D([0], [0], color="darkred", lw=3, label="Positive"),
        Line2D([0], [0], color="darkblue", lw=3, label="Negative"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", title="Interaction")

    plt.tight_layout()

    if save_path:
        for ext in [".png", ".svg", ".pdf"]:
            fig.savefig(str(save_path).replace(".png", ext), dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_interaction_heatmap(
    network: "InteractionNetwork",
    figsize: tuple[float, float] = (10, 8),
    cmap: str = "RdBu_r",
    title: str | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Plot interaction weight matrix as heatmap.

    Args:
        network: InteractionNetwork to visualize
        figsize: Figure size
        cmap: Colormap
        title: Plot title
        save_path: Path to save
        show: Whether to display

    Returns:
        matplotlib Figure
    """
    set_publication_style()

    if network.weight_matrix is None or network.weight_matrix.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No weight matrix available", ha="center", va="center")
        return fig

    fig, ax = plt.subplots(figsize=figsize)

    vmax = max(abs(network.weight_matrix.values.max()), abs(network.weight_matrix.values.min()))

    sns.heatmap(
        network.weight_matrix,
        cmap=cmap,
        center=0,
        vmin=-vmax,
        vmax=vmax,
        annot=True,
        fmt=".2f",
        square=True,
        cbar_kws={"label": "Interaction Weight"},
        ax=ax,
    )

    ax.set_xlabel("Receiver Cell Type")
    ax.set_ylabel("Sender Cell Type")
    ax.set_title(title or "Cell-Cell Interaction Weights")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_ring_attention_decay(
    attention_module: "AttentionModule",
    by_stage: bool = True,
    palette: dict[str, str] | None = None,
    figsize: tuple[float, float] = (8, 5),
    title: str | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Plot attention decay across spatial rings.

    Key validation: attention should decrease with ring number
    (AMICI's monotonic distance decay constraint).

    Args:
        attention_module: AttentionModule with computed patterns
        by_stage: Break down by disease stage
        palette: Stage colors
        figsize: Figure size
        title: Plot title
        save_path: Path to save
        show: Whether to display

    Returns:
        matplotlib Figure
    """
    set_publication_style()

    if attention_module.attention_df is None:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No attention data", ha="center", va="center")
        return fig

    ring_cols = sorted([c for c in attention_module.attention_df.columns if c.startswith("attn_ring")])
    if not ring_cols:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No ring attention columns", ha="center", va="center")
        return fig

    fig, ax = plt.subplots(figsize=figsize)

    ring_numbers = list(range(1, len(ring_cols) + 1))

    if by_stage and "stage_idx" in attention_module.attention_df.columns:
        from stagebridge.contracts import IDX_TO_STAGE

        for stage_idx in sorted(attention_module.attention_df["stage_idx"].unique()):
            stage_data = attention_module.attention_df[
                attention_module.attention_df["stage_idx"] == stage_idx
            ]
            stage_name = IDX_TO_STAGE.get(stage_idx, f"Stage {stage_idx}")

            means = [stage_data[col].mean() for col in ring_cols]
            stds = [stage_data[col].std() for col in ring_cols]

            color = palette.get(stage_name) if palette else None
            ax.errorbar(
                ring_numbers, means, yerr=stds,
                marker="o", capsize=4, label=stage_name, color=color,
            )
    else:
        means = [attention_module.attention_df[col].mean() for col in ring_cols]
        stds = [attention_module.attention_df[col].std() for col in ring_cols]
        ax.errorbar(ring_numbers, means, yerr=stds, marker="o", capsize=4, color="steelblue")

    ax.set_xlabel("Spatial Ring (1=closest, 4=farthest)")
    ax.set_ylabel("Mean Attention Weight")
    ax.set_title(title or "Attention Decay Across Spatial Rings")
    ax.set_xticks(ring_numbers)
    ax.legend(title="Stage")
    ax.set_ylim(bottom=0)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_ablation_importance(
    ablation_module: "AblationModule",
    figsize: tuple[float, float] = (8, 5),
    color: str = "steelblue",
    title: str | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Plot token importance from ablation analysis.

    Args:
        ablation_module: AblationModule with computed results
        figsize: Figure size
        color: Bar color
        title: Plot title
        save_path: Path to save
        show: Whether to display

    Returns:
        matplotlib Figure
    """
    set_publication_style()

    df = ablation_module.to_dataframe()
    if df.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No ablation data", ha="center", va="center")
        return fig

    df = df.sort_values("relative_importance", ascending=True)

    fig, ax = plt.subplots(figsize=figsize)

    ax.barh(df["token"], df["relative_importance"], color=color)

    for i, (idx, row) in enumerate(df.iterrows()):
        if row.get("p_adj", 1) < 0.05:
            ax.text(
                row["relative_importance"] + 0.01,
                i,
                "*" if row["p_adj"] < 0.05 else "",
                va="center",
                fontsize=14,
            )

    ax.set_xlabel("Relative Importance (delta_loss / baseline_loss)")
    ax.set_ylabel("Token")
    ax.set_title(title or "Token Importance from Ablation Analysis")
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_reference_balance(
    attention_module: "AttentionModule",
    figsize: tuple[float, float] = (8, 5),
    colors: tuple[str, str] = ("#2E86AB", "#A23B72"),
    title: str | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Plot HLCA vs LuCA attention balance across stages.

    Interpretation:
    - Higher HLCA attention = cell resembles healthy reference
    - Higher LuCA attention = cell resembles disease reference
    - Progression should show shift toward LuCA

    Args:
        attention_module: AttentionModule with computed patterns
        figsize: Figure size
        colors: (HLCA_color, LuCA_color)
        title: Plot title
        save_path: Path to save
        show: Whether to display

    Returns:
        matplotlib Figure
    """
    set_publication_style()

    ref_df = attention_module.get_reference_balance()
    if ref_df.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No reference attention data", ha="center", va="center")
        return fig

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(ref_df))
    width = 0.35

    ax.bar(x - width/2, ref_df["hlca_attention"], width, label="HLCA (healthy)", color=colors[0])
    ax.bar(x + width/2, ref_df["luca_attention"], width, label="LuCA (cancer)", color=colors[1])

    ax.set_xlabel("Disease Stage")
    ax.set_ylabel("Mean Attention Weight")
    ax.set_title(title or "Reference Atlas Attention Balance")
    ax.set_xticks(x)
    ax.set_xticklabels(ref_df["stage"])
    ax.legend()

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_stage_network_comparison(
    networks: dict[str, "InteractionNetwork"],
    figsize: tuple[float, float] = (15, 5),
    palette: dict[str, str] | None = None,
    weight_threshold: float = 0.0,
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Plot interaction networks side-by-side for stage comparison.

    Args:
        networks: Dict mapping stage name to InteractionNetwork
        figsize: Figure size
        palette: Cell type colors
        weight_threshold: Minimum edge weight
        save_path: Path to save
        show: Whether to display

    Returns:
        matplotlib Figure
    """
    import networkx as nx

    set_publication_style()

    n_stages = len(networks)
    if n_stages == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No networks to compare", ha="center", va="center")
        return fig

    fig, axes = plt.subplots(1, n_stages, figsize=figsize)
    if n_stages == 1:
        axes = [axes]

    all_cell_types = set()
    for network in networks.values():
        all_cell_types.update(network.cell_types)
    all_cell_types = sorted(all_cell_types)

    if palette is None:
        cmap = plt.cm.tab20
        palette = {ct: cmap(i / len(all_cell_types)) for i, ct in enumerate(all_cell_types)}

    for ax, (stage_name, network) in zip(axes, networks.items()):
        G = network.to_networkx(weight_threshold=weight_threshold)
        G.add_nodes_from(all_cell_types)

        pos = nx.circular_layout(G)
        node_colors = [palette.get(node, "#888888") for node in G.nodes()]

        nx.draw_networkx_nodes(G, pos, node_size=800, node_color=node_colors, ax=ax)

        if G.edges():
            weights = [G[u][v].get("weight", 0.5) for u, v in G.edges()]
            max_weight = max(abs(w) for w in weights) if weights else 1

            for (u, v), w in zip(G.edges(), weights):
                width = 1 + 3 * abs(w) / max_weight
                alpha = 0.3 + 0.5 * abs(w) / max_weight
                color = "darkred" if w > 0 else "darkblue"

                nx.draw_networkx_edges(
                    G, pos,
                    edgelist=[(u, v)],
                    width=width,
                    alpha=alpha,
                    edge_color=color,
                    arrows=True,
                    arrowsize=15,
                    connectionstyle="arc3,rad=0.15",
                    ax=ax,
                )

        nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
        ax.set_title(stage_name)
        ax.axis("off")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig
