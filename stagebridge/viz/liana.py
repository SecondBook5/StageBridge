"""LIANA cell-cell communication visualization with Ollivier-Ricci curvature.

Publication-quality figures for ligand-receptor interaction analysis.

Example usage:
    from stagebridge.viz import liana

    # Load data and compute curvature
    df = liana.load_liana_data("path/to/interactions.parquet")
    G = liana.compute_ricci_curvature(df)

    # Generate individual figures
    liana.plot_ricci_network(df, G, save_path="ricci_network.pdf")
    liana.plot_chord_diagram(df, save_path="chord.pdf")
    liana.plot_il1b_network(df, save_path="il1b.pdf")

    # Or generate all at once
    liana.generate_all_figures(df, output_dir="figures/")
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Circle, Wedge, PathPatch
from matplotlib.path import Path as MPath
import matplotlib.colors as mcolors
from pathlib import Path
from typing import Optional, Dict, Union
import networkx as nx

try:
    from GraphRicciCurvature.OllivierRicci import OllivierRicci
    HAS_RICCI = True
except ImportError:
    HAS_RICCI = False


# Publication style defaults
PUBLICATION_STYLE = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
}

# Cell type color palette
CELL_COLORS = {
    'AT2': '#0D3B66',
    'Basal': '#1E6091',
    'Secretory': '#168AAD',
    'Ciliated': '#34A0A4',
    'Macrophages': '#9D0208',
    'T cell lineage': '#DC2F02',
    'Mast cells': '#F48C06',
    'Fibroblast lineage': '#006D32',
    'Capillary': '#6A0DAD',
}

COMPARTMENTS = {
    'Epithelial': ['AT2', 'Basal', 'Secretory', 'Ciliated'],
    'Immune': ['Macrophages', 'T cell lineage', 'Mast cells'],
    'Stromal': ['Fibroblast lineage', 'Capillary'],
}

COMP_COLORS = {'Epithelial': '#1E6091', 'Immune': '#DC2F02', 'Stromal': '#006D32'}


def _short_name(ct: str) -> str:
    """Get short cell type name."""
    return ct.replace(' lineage', '').replace(' cells', '')


def _apply_style():
    """Apply publication style to matplotlib."""
    plt.rcParams.update(PUBLICATION_STYLE)


def load_liana_data(path: Union[str, Path]) -> pd.DataFrame:
    """Load LIANA interaction data from parquet file.

    Parameters
    ----------
    path : str or Path
        Path to parquet file with LIANA results.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: source, target, ligand_complex,
        receptor_complex, lrscore, etc.
    """
    return pd.read_parquet(path)


def compute_ricci_curvature(
    df: pd.DataFrame,
    alpha: float = 0.5,
    threshold_percentile: float = 30,
    use_sum: bool = True,
) -> nx.Graph:
    """Compute Ollivier-Ricci curvature on cell-cell communication graph.

    Negative curvature indicates bottleneck/unique pathways.
    Positive curvature indicates redundant/robust pathways.

    Parameters
    ----------
    df : pd.DataFrame
        LIANA interaction data with source, target, lrscore columns.
    alpha : float
        Laziness parameter for Ricci curvature (0-1). Default 0.5.
    threshold_percentile : float
        Remove edges below this percentile to create sparse graph (0-100).
        Sparse graphs give more meaningful curvature variation.
    use_sum : bool
        If True, use sum of scores (more differentiation). If False, use mean.

    Returns
    -------
    nx.Graph
        NetworkX graph with 'ricciCurvature' and 'weight' edge attributes.

    Raises
    ------
    ImportError
        If GraphRicciCurvature is not installed.
    """
    if not HAS_RICCI:
        raise ImportError(
            "GraphRicciCurvature not installed. "
            "Install with: pip install GraphRicciCurvature"
        )

    # Aggregate interactions
    agg = df.groupby(['source', 'target']).agg({
        'lrscore': ['mean', 'sum']
    }).reset_index()
    agg.columns = ['source', 'target', 'mean_score', 'sum_score']

    # Use sum for more weight differentiation, or mean
    weight_col = 'sum_score' if use_sum else 'mean_score'

    # Threshold to create sparse graph for meaningful curvature
    if threshold_percentile > 0:
        threshold = agg[weight_col].quantile(threshold_percentile / 100)
        agg = agg[agg[weight_col] > threshold]

    G = nx.Graph()

    for _, row in agg.iterrows():
        src, tgt = row['source'], row['target']
        weight = row[weight_col]
        if src != tgt:
            if G.has_edge(src, tgt):
                G[src][tgt]['weight'] = max(G[src][tgt]['weight'], weight)
            else:
                G.add_edge(src, tgt, weight=weight)

    # Normalize weights to 0-1
    if G.number_of_edges() > 0:
        max_weight = max(d['weight'] for _, _, d in G.edges(data=True))
        for u, v in G.edges():
            G[u][v]['weight'] = G[u][v]['weight'] / max_weight

    orc = OllivierRicci(G, alpha=alpha, verbose="ERROR")
    orc.compute_ricci_curvature()

    return orc.G


def plot_ricci_network(
    df: pd.DataFrame,
    G: Optional[nx.Graph] = None,
    figsize: tuple = (12, 10),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> plt.Figure:
    """Network diagram colored by Ollivier-Ricci curvature.

    Parameters
    ----------
    df : pd.DataFrame
        LIANA interaction data.
    G : nx.Graph, optional
        Pre-computed graph with Ricci curvature. If None, computed from df.
    figsize : tuple
        Figure size.
    save_path : str or Path, optional
        Path to save figure (PDF and PNG).
    show : bool
        Whether to display the figure.

    Returns
    -------
    matplotlib.Figure
    """
    _apply_style()

    if G is None:
        G = compute_ricci_curvature(df)

    fig, ax = plt.subplots(figsize=figsize)

    pos = nx.spring_layout(G, k=2, iterations=100, seed=42)

    curvatures = [G[u][v].get('ricciCurvature', 0) for u, v in G.edges()]
    curv_min, curv_max = min(curvatures), max(curvatures)
    norm = plt.Normalize(vmin=curv_min, vmax=curv_max)
    cmap = plt.cm.RdYlBu

    # Draw edges
    for (u, v, data) in G.edges(data=True):
        curv = data.get('ricciCurvature', 0)
        weight = data.get('weight', 0.5)

        x0, y0 = pos[u]
        x1, y1 = pos[v]

        lw = 2 + 8 * weight
        color = cmap(norm(curv))

        ax.plot([x0, x1], [y0, y1], color=color, linewidth=lw,
                alpha=0.7, solid_capstyle='round', zorder=1)

    # Draw nodes
    max_degree = max(dict(G.degree(weight='weight')).values())
    for node in G.nodes():
        x, y = pos[node]
        color = CELL_COLORS.get(node, '#666')
        degree = G.degree(node, weight='weight')
        size = 800 + 400 * (degree / max_degree)

        ax.scatter(x, y, s=size, c=[color], edgecolors='white',
                   linewidth=3, zorder=10)

        offset_x = 0.08 if x > 0 else -0.08
        ha = 'left' if x > 0 else 'right'
        ax.text(x + offset_x, y, _short_name(node), fontsize=11,
                fontweight='bold', ha=ha, va='center', color=color)

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')

    ax.set_title('Cell Communication Network\nColored by Ollivier-Ricci Curvature',
                 fontweight='bold', fontsize=16)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label('Ricci Curvature\n(negative=bottleneck, positive=redundant)',
                   fontweight='bold', fontsize=10)

    ax.text(0.02, 0.02, 'Edge width = interaction strength',
            transform=ax.transAxes, fontsize=9, color='#555')

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        fig.savefig(save_path.with_suffix('.pdf'))
        fig.savefig(save_path.with_suffix('.png'))

    if not show:
        plt.close()

    return fig


def plot_ricci_heatmap(
    df: pd.DataFrame,
    G: Optional[nx.Graph] = None,
    figsize: tuple = (10, 9),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> plt.Figure:
    """Heatmap of Ricci curvature between cell types.

    Parameters
    ----------
    df : pd.DataFrame
        LIANA interaction data.
    G : nx.Graph, optional
        Pre-computed graph with Ricci curvature.
    figsize : tuple
        Figure size.
    save_path : str or Path, optional
        Path to save figure.
    show : bool
        Whether to display the figure.

    Returns
    -------
    matplotlib.Figure
    """
    _apply_style()

    if G is None:
        G = compute_ricci_curvature(df)

    fig, ax = plt.subplots(figsize=figsize)

    cell_order = []
    for comp in ['Epithelial', 'Immune', 'Stromal']:
        for ct in COMPARTMENTS[comp]:
            if ct in G.nodes():
                cell_order.append(ct)

    n = len(cell_order)
    curv_matrix = np.zeros((n, n))

    for i, src in enumerate(cell_order):
        for j, tgt in enumerate(cell_order):
            if G.has_edge(src, tgt):
                curv_matrix[i, j] = G[src][tgt].get('ricciCurvature', 0)

    vmax = max(abs(curv_matrix.min()), abs(curv_matrix.max()))
    im = ax.imshow(curv_matrix, cmap='RdBu', vmin=-vmax, vmax=vmax, aspect='auto')

    short_labels = [_short_name(c) for c in cell_order]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=11)
    ax.set_yticklabels(short_labels, fontsize=11)

    # Compartment dividers
    pos = 0
    for comp in ['Epithelial', 'Immune']:
        ct_count = sum(1 for ct in COMPARTMENTS[comp] if ct in cell_order)
        pos += ct_count
        ax.axhline(pos - 0.5, color='black', linewidth=2)
        ax.axvline(pos - 0.5, color='black', linewidth=2)

    # Add values
    for i in range(n):
        for j in range(n):
            val = curv_matrix[i, j]
            if val != 0:
                color = 'white' if abs(val) > vmax * 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=9, color=color, fontweight='bold')

    ax.set_xlabel('Target Cell Type', fontweight='bold', fontsize=12)
    ax.set_ylabel('Source Cell Type', fontweight='bold', fontsize=12)
    ax.set_title('Ollivier-Ricci Curvature Between Cell Types\n'
                 '(negative=bottleneck, positive=redundant)',
                 fontweight='bold', fontsize=14)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Ricci Curvature', fontweight='bold')

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        fig.savefig(save_path.with_suffix('.pdf'))
        fig.savefig(save_path.with_suffix('.png'))

    if not show:
        plt.close()

    return fig


def plot_chord_diagram(
    df: pd.DataFrame,
    figsize: tuple = (12, 12),
    threshold_percentile: float = 50,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> plt.Figure:
    """Circular chord diagram of cell-cell communication.

    Parameters
    ----------
    df : pd.DataFrame
        LIANA interaction data.
    figsize : tuple
        Figure size.
    threshold_percentile : float
        Only show edges above this percentile (0-100).
    save_path : str or Path, optional
        Path to save figure.
    show : bool
        Whether to display the figure.

    Returns
    -------
    matplotlib.Figure
    """
    _apply_style()

    fig, ax = plt.subplots(figsize=figsize)

    cell_types = sorted(df['source'].unique())

    comm_matrix = df.groupby(['source', 'target'])['lrscore'].mean().unstack(fill_value=0)
    comm_matrix = comm_matrix.reindex(index=cell_types, columns=cell_types, fill_value=0)

    values = comm_matrix.values
    val_min, val_max = values.min(), values.max()

    node_totals = comm_matrix.sum(axis=0) + comm_matrix.sum(axis=1)
    node_totals = node_totals / node_totals.max()

    # Order by compartment
    ordered_cells = []
    for comp in ['Epithelial', 'Immune', 'Stromal']:
        for ct in COMPARTMENTS[comp]:
            if ct in cell_types:
                ordered_cells.append(ct)

    angles = np.linspace(0, 2 * np.pi, len(ordered_cells), endpoint=False) - np.pi/2
    radius = 0.36
    positions = {ct: (radius * np.cos(a), radius * np.sin(a))
                 for ct, a in zip(ordered_cells, angles)}

    # Compartment arcs
    comp_start = 0
    for comp in ['Epithelial', 'Immune', 'Stromal']:
        cts = [ct for ct in COMPARTMENTS[comp] if ct in ordered_cells]
        if not cts:
            continue
        n_comp = len(cts)

        start_angle = np.degrees(angles[comp_start] - np.pi/len(ordered_cells))
        end_angle = np.degrees(angles[comp_start + n_comp - 1] + np.pi/len(ordered_cells))

        wedge = Wedge((0, 0), 0.48, start_angle, end_angle, width=0.08,
                      facecolor=mcolors.to_rgba(COMP_COLORS[comp], 0.2),
                      edgecolor=COMP_COLORS[comp], linewidth=2)
        ax.add_patch(wedge)

        mid_angle = (start_angle + end_angle) / 2
        label_r = 0.54
        lx = label_r * np.cos(np.radians(mid_angle))
        ly = label_r * np.sin(np.radians(mid_angle))
        ax.text(lx, ly, comp, ha='center', va='center', fontsize=12,
                fontweight='bold', color=COMP_COLORS[comp])

        comp_start += n_comp

    # Edges
    threshold = np.percentile(values[values > 0], threshold_percentile)

    for i, source in enumerate(ordered_cells):
        for j, target in enumerate(ordered_cells):
            if i >= j:
                continue
            score = (comm_matrix.loc[source, target] + comm_matrix.loc[target, source]) / 2
            if score < threshold:
                continue

            sx, sy = positions[source]
            tx, ty = positions[target]

            norm_score = (score - threshold) / (val_max - threshold)

            c1 = np.array(mcolors.to_rgb(CELL_COLORS.get(source, '#666')))
            c2 = np.array(mcolors.to_rgb(CELL_COLORS.get(target, '#666')))
            blend = tuple((c1 + c2) / 2)

            alpha = 0.3 + 0.7 * norm_score
            width = 1 + 6 * norm_score

            ctrl_x = (sx + tx) / 5
            ctrl_y = (sy + ty) / 5

            verts = [(sx, sy), (ctrl_x, ctrl_y), (tx, ty)]
            codes = [MPath.MOVETO, MPath.CURVE3, MPath.CURVE3]
            path = MPath(verts, codes)

            patch = PathPatch(path, facecolor='none', edgecolor=blend,
                             alpha=alpha, linewidth=width, capstyle='round')
            ax.add_patch(patch)

    # Nodes
    for ct in ordered_cells:
        x, y = positions[ct]
        size = 0.04 + 0.025 * node_totals[ct]
        color = CELL_COLORS.get(ct, '#666')

        circle = Circle((x, y), size, facecolor=color, edgecolor='white',
                        linewidth=2.5, zorder=10)
        ax.add_patch(circle)

        angle = np.arctan2(y, x)
        label_r = radius + 0.1
        lx = label_r * np.cos(angle)
        ly = label_r * np.sin(angle)

        rot = np.degrees(angle)
        if rot > 90 or rot < -90:
            rot += 180
            ha = 'right'
        else:
            ha = 'left'

        ax.text(lx, ly, _short_name(ct), ha=ha, va='center', fontsize=11,
                fontweight='bold', color=color, rotation=rot, rotation_mode='anchor')

    ax.set_xlim(-0.7, 0.7)
    ax.set_ylim(-0.7, 0.7)
    ax.set_aspect('equal')
    ax.axis('off')

    ax.set_title('Cell-Cell Communication Network', fontweight='bold', fontsize=16, y=1.0)
    ax.text(0.5, -0.02, 'Edge width and opacity indicate interaction strength',
            ha='center', fontsize=10, color='#555', transform=ax.transAxes)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        fig.savefig(save_path.with_suffix('.pdf'))
        fig.savefig(save_path.with_suffix('.png'))

    if not show:
        plt.close()

    return fig


def plot_il1b_network(
    df: pd.DataFrame,
    figsize: tuple = (12, 8),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> plt.Figure:
    """IL1B proinflammatory signaling network.

    Parameters
    ----------
    df : pd.DataFrame
        LIANA interaction data.
    figsize : tuple
        Figure size.
    save_path : str or Path, optional
        Path to save figure.
    show : bool
        Whether to display the figure.

    Returns
    -------
    matplotlib.Figure
    """
    _apply_style()

    fig, ax = plt.subplots(figsize=figsize)

    il1b_df = df[df['ligand_complex'].str.contains('IL1B', na=False)].copy()
    il1b_agg = il1b_df.groupby(['source', 'target', 'receptor_complex'])['lrscore'].mean().reset_index()

    targets = sorted(il1b_agg['target'].unique())
    n_targets = len(targets)

    source_x, source_y = 0.12, 0.5

    target_positions = {}
    for i, t in enumerate(targets):
        y = 0.9 - (i / (n_targets - 1)) * 0.8 if n_targets > 1 else 0.5
        target_positions[t] = (0.75, y)

    max_score = il1b_agg['lrscore'].max()
    for _, row in il1b_agg.iterrows():
        target = row['target']
        receptor = row['receptor_complex']
        score = row['lrscore']

        tx, ty = target_positions[target]
        color = CELL_COLORS.get(target, '#666')

        norm_score = score / max_score
        edge_width = 2 + 6 * norm_score
        edge_alpha = 0.4 + 0.5 * norm_score

        arrow = FancyArrowPatch(
            (source_x + 0.07, source_y), (tx - 0.05, ty),
            connectionstyle="arc3,rad=0.15",
            arrowstyle='-|>',
            mutation_scale=15,
            color=color,
            alpha=edge_alpha,
            linewidth=edge_width,
            zorder=5
        )
        ax.add_patch(arrow)

        mid_x = (source_x + tx) / 2 + 0.08
        mid_y = (source_y + ty) / 2
        ax.text(mid_x, mid_y, receptor, fontsize=9, color='#333',
                ha='center', va='center', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                         edgecolor='none', alpha=0.8))

    source_circle = Circle((source_x, source_y), 0.065,
                           facecolor='#9D0208', edgecolor='white',
                           linewidth=3, zorder=10)
    ax.add_patch(source_circle)
    ax.text(source_x, source_y, 'IL1B+', ha='center', va='center',
            fontweight='bold', fontsize=13, color='white', zorder=11)
    ax.text(source_x, source_y - 0.11, 'Macrophages', ha='center', va='top',
            fontsize=10, color='#555', style='italic')

    for target, (tx, ty) in target_positions.items():
        color = CELL_COLORS.get(target, '#666')
        target_circle = Circle((tx, ty), 0.04,
                               facecolor=color, edgecolor='white',
                               linewidth=2, zorder=10)
        ax.add_patch(target_circle)
        ax.text(tx + 0.07, ty, _short_name(target),
                ha='left', va='center', fontsize=11,
                fontweight='bold', color=color)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')

    ax.set_title('IL1B Proinflammatory Signaling Network',
                 fontweight='bold', fontsize=16, y=0.98)
    ax.text(0.5, 0.94, 'Macrophage-derived IL1B signals via IL1R1/IL1R2/SIGIRR',
            ha='center', fontsize=11, color='#555', style='italic', transform=ax.transAxes)
    ax.text(0.02, 0.02, 'Line width = interaction strength',
            fontsize=9, color='#777', transform=ax.transAxes)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        fig.savefig(save_path.with_suffix('.pdf'))
        fig.savefig(save_path.with_suffix('.png'))

    if not show:
        plt.close()

    return fig


def plot_bottleneck_analysis(
    df: pd.DataFrame,
    G: Optional[nx.Graph] = None,
    top_n: int = 15,
    figsize: tuple = (10, 7),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> plt.Figure:
    """Bar chart of communication bottlenecks vs redundant pathways.

    Parameters
    ----------
    df : pd.DataFrame
        LIANA interaction data.
    G : nx.Graph, optional
        Pre-computed graph with Ricci curvature.
    top_n : int
        Number of edges to show on each side.
    figsize : tuple
        Figure size.
    save_path : str or Path, optional
        Path to save figure.
    show : bool
        Whether to display the figure.

    Returns
    -------
    matplotlib.Figure
    """
    _apply_style()

    if G is None:
        G = compute_ricci_curvature(df)

    fig, ax = plt.subplots(figsize=figsize)

    edge_data = []
    for u, v, data in G.edges(data=True):
        curv = data.get('ricciCurvature', 0)
        edge_data.append({
            'edge': f"{_short_name(u)} - {_short_name(v)}",
            'curvature': curv,
        })

    edge_df = pd.DataFrame(edge_data).sort_values('curvature')

    # Get both extremes - bottlenecks (low) and redundant (high)
    n_show = min(top_n, len(edge_df) // 2)
    bottlenecks = edge_df.head(n_show)
    redundant = edge_df.tail(n_show).iloc[::-1]  # Reverse for display

    # Combine with gap
    combined = pd.concat([bottlenecks, redundant])

    # Colors based on curvature - diverging
    curv_min = combined['curvature'].min()
    curv_max = combined['curvature'].max()
    norm = plt.Normalize(vmin=curv_min, vmax=curv_max)
    colors = plt.cm.RdYlBu(norm(combined['curvature']))

    y_pos = np.arange(len(combined))
    bars = ax.barh(y_pos, combined['curvature'], color=colors, edgecolor='white', linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(combined['edge'], fontsize=10)
    ax.invert_yaxis()

    ax.axvline(0, color='black', linewidth=1.5, linestyle='-', alpha=0.7)

    # Add section labels
    ax.axhline(n_show - 0.5, color='gray', linewidth=1, linestyle='--', alpha=0.5)
    ax.text(curv_min * 0.5, n_show / 2 - 0.5, 'BOTTLENECKS\n(unique pathways)',
            ha='center', va='center', fontsize=9, fontweight='bold', color='#9D0208')
    ax.text(curv_max * 0.5, n_show + n_show / 2 - 0.5, 'REDUNDANT\n(robust pathways)',
            ha='center', va='center', fontsize=9, fontweight='bold', color='#1E6091')

    ax.set_xlabel('Ollivier-Ricci Curvature', fontweight='bold', fontsize=12)
    ax.set_title('Communication Network Topology\nBottleneck vs Redundant Pathways',
                 fontweight='bold', fontsize=14)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        fig.savefig(save_path.with_suffix('.pdf'))
        fig.savefig(save_path.with_suffix('.png'))

    if not show:
        plt.close()

    return fig


def plot_communication_heatmap(
    df: pd.DataFrame,
    figsize: tuple = (11, 10),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> plt.Figure:
    """Cell-cell communication strength heatmap.

    Parameters
    ----------
    df : pd.DataFrame
        LIANA interaction data.
    figsize : tuple
        Figure size.
    save_path : str or Path, optional
        Path to save figure.
    show : bool
        Whether to display the figure.

    Returns
    -------
    matplotlib.Figure
    """
    _apply_style()

    fig, ax = plt.subplots(figsize=figsize)

    comm_matrix = df.groupby(['source', 'target'])['lrscore'].mean().unstack(fill_value=0)

    cell_order = []
    for comp in ['Epithelial', 'Immune', 'Stromal']:
        for ct in COMPARTMENTS[comp]:
            if ct in comm_matrix.index:
                cell_order.append(ct)

    comm_matrix = comm_matrix.reindex(index=cell_order, columns=cell_order, fill_value=0)

    values = comm_matrix.values
    values_vis = np.power(values, 0.5)

    im = ax.imshow(values_vis, cmap='viridis', aspect='auto')

    n = len(cell_order)
    short_labels = [_short_name(c) for c in cell_order]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=11)
    ax.set_yticklabels(short_labels, fontsize=11)

    pos = 0
    boundaries = []
    for comp in ['Epithelial', 'Immune', 'Stromal']:
        ct_count = sum(1 for ct in COMPARTMENTS[comp] if ct in cell_order)
        if ct_count > 0:
            boundaries.append((pos, pos + ct_count, comp))
            pos += ct_count

    for start, end, comp in boundaries[:-1]:
        ax.axhline(end - 0.5, color='white', linewidth=3)
        ax.axvline(end - 0.5, color='white', linewidth=3)

    for start, end, comp in boundaries:
        mid = (start + end - 1) / 2
        ax.text(-0.8, mid, comp, ha='right', va='center', fontweight='bold',
                fontsize=11, color=COMP_COLORS[comp], rotation=90)
        ax.text(mid, -0.8, comp, ha='center', va='top', fontweight='bold',
                fontsize=11, color=COMP_COLORS[comp])

    ax.set_xlabel('Target Cell Type', fontweight='bold', fontsize=12, labelpad=25)
    ax.set_ylabel('Source Cell Type', fontweight='bold', fontsize=12, labelpad=35)
    ax.set_title('Cell-Cell Communication Strength', fontweight='bold', fontsize=16)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('LIANA Score (sqrt scale)', fontweight='bold')

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        fig.savefig(save_path.with_suffix('.pdf'))
        fig.savefig(save_path.with_suffix('.png'))

    if not show:
        plt.close()

    return fig


def plot_compartment_flow(
    df: pd.DataFrame,
    figsize: tuple = (11, 7),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> plt.Figure:
    """Sankey-style flow diagram between cell compartments.

    Parameters
    ----------
    df : pd.DataFrame
        LIANA interaction data.
    figsize : tuple
        Figure size.
    save_path : str or Path, optional
        Path to save figure.
    show : bool
        Whether to display the figure.

    Returns
    -------
    matplotlib.Figure
    """
    _apply_style()

    fig, ax = plt.subplots(figsize=figsize)

    def get_compartment(ct):
        for comp, cts in COMPARTMENTS.items():
            if ct in cts:
                return comp
        return 'Other'

    df_comp = df.copy()
    df_comp['source_comp'] = df_comp['source'].apply(get_compartment)
    df_comp['target_comp'] = df_comp['target'].apply(get_compartment)

    flows = df_comp.groupby(['source_comp', 'target_comp'])['lrscore'].sum().reset_index()

    comps = ['Epithelial', 'Immune', 'Stromal']
    left_y = {c: 0.78 - i * 0.28 for i, c in enumerate(comps)}
    right_y = {c: 0.78 - i * 0.28 for i, c in enumerate(comps)}

    node_height = 0.2
    for comp in comps:
        color = COMP_COLORS[comp]

        rect = mpatches.FancyBboxPatch(
            (0.06, left_y[comp] - node_height/2), 0.14, node_height,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            facecolor=color, edgecolor='white', linewidth=3
        )
        ax.add_patch(rect)
        ax.text(0.13, left_y[comp], comp, ha='center', va='center',
                fontweight='bold', fontsize=12, color='white')

        rect = mpatches.FancyBboxPatch(
            (0.80, right_y[comp] - node_height/2), 0.14, node_height,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            facecolor=color, edgecolor='white', linewidth=3
        )
        ax.add_patch(rect)
        ax.text(0.87, right_y[comp], comp, ha='center', va='center',
                fontweight='bold', fontsize=12, color='white')

    max_flow = flows['lrscore'].max()
    min_flow = flows['lrscore'].min()

    for _, row in flows.iterrows():
        src, tgt, score = row['source_comp'], row['target_comp'], row['lrscore']

        x0, y0 = 0.20, left_y[src]
        x1, y1 = 0.80, right_y[tgt]

        norm_score = (score - min_flow) / (max_flow - min_flow)
        width = 0.02 + 0.12 * norm_score

        c1 = np.array(mcolors.to_rgb(COMP_COLORS[src]))
        c2 = np.array(mcolors.to_rgb(COMP_COLORS[tgt]))
        blend = tuple((c1 + c2) / 2)

        verts_upper = [(x0, y0 + width/2),
                       (0.5, (y0 + y1)/2 + width/2 + 0.01),
                       (x1, y1 + width/2)]
        verts_lower = [(x1, y1 - width/2),
                       (0.5, (y0 + y1)/2 - width/2 - 0.01),
                       (x0, y0 - width/2)]

        verts = verts_upper + verts_lower + [verts_upper[0]]
        codes = [MPath.MOVETO, MPath.CURVE3, MPath.CURVE3,
                 MPath.LINETO, MPath.CURVE3, MPath.CURVE3, MPath.CLOSEPOLY]

        path = MPath(verts, codes)
        patch = PathPatch(path, facecolor=blend, alpha=0.65, edgecolor='none')
        ax.add_patch(patch)

        mid_y = (y0 + y1) / 2
        ax.text(0.5, mid_y, f'{score:.0f}', ha='center', va='center',
                fontsize=10, fontweight='bold', color='#222')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    ax.text(0.13, 0.95, 'Source', ha='center', fontweight='bold', fontsize=14)
    ax.text(0.87, 0.95, 'Target', ha='center', fontweight='bold', fontsize=14)
    ax.set_title('Inter-Compartment Communication Flow', fontweight='bold', fontsize=16, y=1.02)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        fig.savefig(save_path.with_suffix('.pdf'))
        fig.savefig(save_path.with_suffix('.png'))

    if not show:
        plt.close()

    return fig


def plot_top_lr_pairs(
    df: pd.DataFrame,
    top_n: int = 20,
    figsize: tuple = (9, 7),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> plt.Figure:
    """Top ligand-receptor interaction pairs.

    Parameters
    ----------
    df : pd.DataFrame
        LIANA interaction data.
    top_n : int
        Number of top pairs to show.
    figsize : tuple
        Figure size.
    save_path : str or Path, optional
        Path to save figure.
    show : bool
        Whether to display the figure.

    Returns
    -------
    matplotlib.Figure
    """
    _apply_style()

    fig, ax = plt.subplots(figsize=figsize)

    top_pairs = df.groupby(['ligand_complex', 'receptor_complex'])['lrscore'].mean()
    top_pairs = top_pairs.sort_values(ascending=False).head(top_n)

    labels = [f"{l} -> {r}" for l, r in top_pairs.index]
    values = top_pairs.values

    colors = plt.cm.viridis(np.linspace(0.9, 0.3, len(values)))

    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, values, color=colors, edgecolor='white', linewidth=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('LIANA Interaction Score', fontweight='bold', fontsize=12)
    ax.set_title(f'Top {top_n} Ligand-Receptor Interactions', fontweight='bold', fontsize=16)

    for bar, val in zip(bars, values):
        ax.text(val + 0.008, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9, color='#333')

    ax.set_xlim(0, max(values) * 1.15)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        fig.savefig(save_path.with_suffix('.pdf'))
        fig.savefig(save_path.with_suffix('.png'))

    if not show:
        plt.close()

    return fig


def generate_all_figures(
    df: pd.DataFrame,
    output_dir: Union[str, Path],
    show: bool = False,
) -> Dict[str, plt.Figure]:
    """Generate all LIANA visualization figures.

    Parameters
    ----------
    df : pd.DataFrame
        LIANA interaction data.
    output_dir : str or Path
        Directory to save figures.
    show : bool
        Whether to display figures.

    Returns
    -------
    dict
        Dictionary mapping figure names to Figure objects.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    figures = {}

    print("Computing Ollivier-Ricci curvature...")
    G = compute_ricci_curvature(df)

    print("Generating figures...")

    figures['ricci_network'] = plot_ricci_network(
        df, G, save_path=output_dir / 'liana_ricci_network', show=show)
    print("  - liana_ricci_network")

    figures['ricci_heatmap'] = plot_ricci_heatmap(
        df, G, save_path=output_dir / 'liana_ricci_heatmap', show=show)
    print("  - liana_ricci_heatmap")

    figures['chord'] = plot_chord_diagram(
        df, save_path=output_dir / 'liana_chord_network', show=show)
    print("  - liana_chord_network")

    figures['il1b'] = plot_il1b_network(
        df, save_path=output_dir / 'liana_il1b_network', show=show)
    print("  - liana_il1b_network")

    figures['heatmap'] = plot_communication_heatmap(
        df, save_path=output_dir / 'liana_heatmap', show=show)
    print("  - liana_heatmap")

    figures['bottlenecks'] = plot_bottleneck_analysis(
        df, G, save_path=output_dir / 'liana_bottlenecks', show=show)
    print("  - liana_bottlenecks")

    figures['flow'] = plot_compartment_flow(
        df, save_path=output_dir / 'liana_compartment_flow', show=show)
    print("  - liana_compartment_flow")

    figures['top_lr'] = plot_top_lr_pairs(
        df, save_path=output_dir / 'liana_top_lr_pairs', show=show)
    print("  - liana_top_lr_pairs")

    print(f"\nAll figures saved to {output_dir}")

    return figures
