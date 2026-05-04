#!/usr/bin/env python3
"""Generate publication-quality LIANA cell-cell communication figures.

Includes Ollivier-Ricci curvature analysis for identifying communication bottlenecks.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Circle, Wedge, PathPatch
from matplotlib.path import Path as MPath
import matplotlib.colors as mcolors
from pathlib import Path
import networkx as nx
from GraphRicciCurvature.OllivierRicci import OllivierRicci


# Publication style
plt.rcParams.update({
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
})

# Rich, distinct color palette - more saturated
CELL_COLORS = {
    'AT2': '#0D3B66',           # Deep navy
    'Basal': '#1E6091',         # Ocean blue
    'Secretory': '#168AAD',     # Cerulean
    'Ciliated': '#34A0A4',      # Teal
    'Macrophages': '#9D0208',   # Deep red
    'T cell lineage': '#DC2F02',# Vermillion
    'Mast cells': '#F48C06',    # Orange
    'Fibroblast lineage': '#006D32', # Forest green
    'Capillary': '#6A0DAD',     # Purple
}

COMPARTMENTS = {
    'Epithelial': ['AT2', 'Basal', 'Secretory', 'Ciliated'],
    'Immune': ['Macrophages', 'T cell lineage', 'Mast cells'],
    'Stromal': ['Fibroblast lineage', 'Capillary'],
}

COMP_COLORS = {'Epithelial': '#1E6091', 'Immune': '#DC2F02', 'Stromal': '#006D32'}

OUTPUT_DIR = Path('/home/booka/projects/StageBridge/figures/publication')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data(path: str | Path | None = None):
    """Load LIANA interaction data."""
    if path is None:
        # Try LuCA version first, fall back to original
        luca_path = Path('/home/booka/projects/StageBridge/data/liana_interactions_luca.parquet')
        orig_path = Path('/home/booka/projects/StageBridge/data/liana_interactions.parquet')
        path = luca_path if luca_path.exists() else orig_path
    df = pd.read_parquet(path)
    return df


def short_name(ct):
    """Get short cell type name."""
    return ct.replace(' lineage', '').replace(' cells', '')


def compute_ricci_curvature(df):
    """Compute Ollivier-Ricci curvature on cell-cell communication graph."""
    G = nx.Graph()

    # Aggregate by cell type pairs
    agg = df.groupby(['source', 'target'])['lrscore'].mean().reset_index()

    for _, row in agg.iterrows():
        src, tgt, weight = row['source'], row['target'], row['lrscore']
        if src != tgt:
            if G.has_edge(src, tgt):
                G[src][tgt]['weight'] = max(G[src][tgt]['weight'], weight)
            else:
                G.add_edge(src, tgt, weight=weight)

    # Compute Ollivier-Ricci curvature
    orc = OllivierRicci(G, alpha=0.5, verbose="ERROR")
    orc.compute_ricci_curvature()

    return orc.G


# =============================================================================
# Figure 1: Ricci Network - Communication bottlenecks
# =============================================================================
def fig_ricci_network(df):
    """Network colored by Ollivier-Ricci curvature - the main novel figure."""
    fig, ax = plt.subplots(figsize=(12, 10))

    G = compute_ricci_curvature(df)

    pos = nx.spring_layout(G, k=2, iterations=100, seed=42)

    curvatures = [G[u][v].get('ricciCurvature', 0) for u, v in G.edges()]
    curv_min, curv_max = min(curvatures), max(curvatures)

    norm = plt.Normalize(vmin=curv_min, vmax=curv_max)
    cmap = plt.cm.RdYlBu

    for (u, v, data) in G.edges(data=True):
        curv = data.get('ricciCurvature', 0)
        weight = data.get('weight', 0.5)

        x0, y0 = pos[u]
        x1, y1 = pos[v]

        lw = 2 + 8 * weight
        color = cmap(norm(curv))

        ax.plot([x0, x1], [y0, y1], color=color, linewidth=lw,
                alpha=0.7, solid_capstyle='round', zorder=1)

    for node in G.nodes():
        x, y = pos[node]
        color = CELL_COLORS.get(node, '#666')

        degree = G.degree(node, weight='weight')
        size = 800 + 400 * (degree / max(dict(G.degree(weight='weight')).values()))

        ax.scatter(x, y, s=size, c=[color], edgecolors='white',
                   linewidth=3, zorder=10)

        offset_x = 0.08 if x > 0 else -0.08
        ha = 'left' if x > 0 else 'right'
        ax.text(x + offset_x, y, short_name(node), fontsize=11,
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
    fig.savefig(OUTPUT_DIR / 'liana_ricci_network.pdf')
    fig.savefig(OUTPUT_DIR / 'liana_ricci_network.png')
    plt.close()
    print("Saved: liana_ricci_network")

    return G


# =============================================================================
# Figure 2: Ricci Heatmap
# =============================================================================
def fig_ricci_heatmap(df, G):
    """Heatmap of Ricci curvature between cell types."""
    fig, ax = plt.subplots(figsize=(10, 9))

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

    short_labels = [short_name(c) for c in cell_order]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=11)
    ax.set_yticklabels(short_labels, fontsize=11)

    pos = 0
    for comp in ['Epithelial', 'Immune']:
        ct_count = sum(1 for ct in COMPARTMENTS[comp] if ct in cell_order)
        pos += ct_count
        ax.axhline(pos - 0.5, color='black', linewidth=2)
        ax.axvline(pos - 0.5, color='black', linewidth=2)

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
                 '(negative=bottleneck pathway, positive=redundant pathway)',
                 fontweight='bold', fontsize=14)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Ricci Curvature', fontweight='bold')

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'liana_ricci_heatmap.pdf')
    fig.savefig(OUTPUT_DIR / 'liana_ricci_heatmap.png')
    plt.close()
    print("Saved: liana_ricci_heatmap")


# =============================================================================
# Figure 3: Bottleneck Analysis
# =============================================================================
def fig_bottleneck_analysis(df, G):
    """Bar chart of most negative (bottleneck) curvature edges."""
    fig, ax = plt.subplots(figsize=(10, 6))

    edge_data = []
    for u, v, data in G.edges(data=True):
        curv = data.get('ricciCurvature', 0)
        weight = data.get('weight', 0)
        edge_data.append({
            'edge': f"{short_name(u)} - {short_name(v)}",
            'curvature': curv,
            'weight': weight,
            'source': u,
            'target': v
        })

    edge_df = pd.DataFrame(edge_data)
    edge_df = edge_df.sort_values('curvature').head(15)

    colors = plt.cm.RdYlBu(plt.Normalize(-0.5, 0.5)(edge_df['curvature']))

    y_pos = np.arange(len(edge_df))
    ax.barh(y_pos, edge_df['curvature'], color=colors, edgecolor='white')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(edge_df['edge'], fontsize=10)
    ax.invert_yaxis()

    ax.axvline(0, color='black', linewidth=1, linestyle='--', alpha=0.5)
    ax.set_xlabel('Ollivier-Ricci Curvature', fontweight='bold', fontsize=12)
    ax.set_title('Communication Bottlenecks\n(Most Negative Ricci Curvature)',
                 fontweight='bold', fontsize=14)

    ax.text(0.98, 0.02, 'Negative = unique/bottleneck pathway\n'
            'Positive = redundant pathway',
            ha='right', va='bottom', transform=ax.transAxes, fontsize=9,
            color='#555', style='italic')

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'liana_bottlenecks.pdf')
    fig.savefig(OUTPUT_DIR / 'liana_bottlenecks.png')
    plt.close()
    print("Saved: liana_bottlenecks")


# =============================================================================
# Figure 4: Chord Network
# =============================================================================
def fig_chord_network(df):
    """Improved chord diagram with stronger visual differentiation."""
    fig, ax = plt.subplots(figsize=(12, 12))

    cell_types = sorted(df['source'].unique())

    comm_matrix = df.groupby(['source', 'target'])['lrscore'].mean().unstack(fill_value=0)
    comm_matrix = comm_matrix.reindex(index=cell_types, columns=cell_types, fill_value=0)

    values = comm_matrix.values
    val_min, val_max = values.min(), values.max()

    node_totals = comm_matrix.sum(axis=0) + comm_matrix.sum(axis=1)
    node_totals = node_totals / node_totals.max()

    ordered_cells = []
    for comp in ['Epithelial', 'Immune', 'Stromal']:
        for ct in COMPARTMENTS[comp]:
            if ct in cell_types:
                ordered_cells.append(ct)

    angles = np.linspace(0, 2 * np.pi, len(ordered_cells), endpoint=False) - np.pi/2
    radius = 0.36
    positions = {ct: (radius * np.cos(a), radius * np.sin(a))
                 for ct, a in zip(ordered_cells, angles)}

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

    threshold = np.percentile(values[values > 0], 50)

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

        ax.text(lx, ly, short_name(ct), ha=ha, va='center', fontsize=11,
                fontweight='bold', color=color, rotation=rot, rotation_mode='anchor')

    ax.set_xlim(-0.7, 0.7)
    ax.set_ylim(-0.7, 0.7)
    ax.set_aspect('equal')
    ax.axis('off')

    ax.set_title('Cell-Cell Communication Network', fontweight='bold', fontsize=16, y=1.0)
    ax.text(0.5, -0.02, 'Edge width and opacity indicate interaction strength',
            ha='center', fontsize=10, color='#555', transform=ax.transAxes)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'liana_chord_network.pdf')
    fig.savefig(OUTPUT_DIR / 'liana_chord_network.png')
    plt.close()
    print("Saved: liana_chord_network")


# =============================================================================
# Figure 5: IL1B Network
# =============================================================================
def fig_il1b_network(df):
    """IL1B proinflammatory signaling network."""
    fig, ax = plt.subplots(figsize=(12, 8))

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

        ax.text(tx + 0.07, ty, short_name(target),
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
    fig.savefig(OUTPUT_DIR / 'liana_il1b_network.pdf')
    fig.savefig(OUTPUT_DIR / 'liana_il1b_network.png')
    plt.close()
    print("Saved: liana_il1b_network")


# =============================================================================
# Figure 6: Communication Heatmap
# =============================================================================
def fig_heatmap(df):
    """Cell-cell communication heatmap with compartment organization."""
    fig, ax = plt.subplots(figsize=(11, 10))

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
    short_labels = [short_name(c) for c in cell_order]
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
    fig.savefig(OUTPUT_DIR / 'liana_heatmap.pdf')
    fig.savefig(OUTPUT_DIR / 'liana_heatmap.png')
    plt.close()
    print("Saved: liana_heatmap")


# =============================================================================
# Figure 7: Compartment Flow
# =============================================================================
def fig_compartment_flow(df):
    """Sankey-style flow between compartments."""
    fig, ax = plt.subplots(figsize=(11, 7))

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
    fig.savefig(OUTPUT_DIR / 'liana_compartment_flow.pdf')
    fig.savefig(OUTPUT_DIR / 'liana_compartment_flow.png')
    plt.close()
    print("Saved: liana_compartment_flow")


# =============================================================================
# Figure 8: Source/Target Bar Charts
# =============================================================================
def fig_source_target_bars(df):
    """Source/target bar charts showing cell type communication roles."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    outgoing = df.groupby('source')['lrscore'].sum().sort_values(ascending=True)
    colors_out = [CELL_COLORS.get(ct, '#666') for ct in outgoing.index]
    short_labels = [short_name(c) for c in outgoing.index]

    axes[0].barh(range(len(outgoing)), outgoing.values, color=colors_out,
                 edgecolor='white', linewidth=1.5)
    axes[0].set_yticks(range(len(outgoing)))
    axes[0].set_yticklabels(short_labels, fontsize=11)
    axes[0].set_xlabel('Total Outgoing Signal', fontweight='bold', fontsize=12)
    axes[0].set_title('Ligand Production\n(Source Activity)', fontweight='bold', fontsize=14)

    for i, v in enumerate(outgoing.values):
        axes[0].text(v + 50, i, f'{v:.0f}', va='center', fontsize=9, color='#333')

    incoming = df.groupby('target')['lrscore'].sum().sort_values(ascending=True)
    colors_in = [CELL_COLORS.get(ct, '#666') for ct in incoming.index]
    short_labels = [short_name(c) for c in incoming.index]

    axes[1].barh(range(len(incoming)), incoming.values, color=colors_in,
                 edgecolor='white', linewidth=1.5)
    axes[1].set_yticks(range(len(incoming)))
    axes[1].set_yticklabels(short_labels, fontsize=11)
    axes[1].set_xlabel('Total Incoming Signal', fontweight='bold', fontsize=12)
    axes[1].set_title('Receptor Activity\n(Target Activity)', fontweight='bold', fontsize=14)

    for i, v in enumerate(incoming.values):
        axes[1].text(v + 50, i, f'{v:.0f}', va='center', fontsize=9, color='#333')

    fig.suptitle('Cell Type Communication Roles', fontweight='bold', fontsize=16, y=1.02)
    plt.tight_layout()

    fig.savefig(OUTPUT_DIR / 'liana_source_target_bars.pdf')
    fig.savefig(OUTPUT_DIR / 'liana_source_target_bars.png')
    plt.close()
    print("Saved: liana_source_target_bars")


# =============================================================================
# Figure 9: Top L-R Pairs
# =============================================================================
def fig_top_lr_pairs(df, top_n=20):
    """Top L-R interactions with gradient coloring."""
    fig, ax = plt.subplots(figsize=(9, 7))

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
    ax.set_title('Top 20 Ligand-Receptor Interactions', fontweight='bold', fontsize=16)

    for bar, val in zip(bars, values):
        ax.text(val + 0.008, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9, color='#333')

    ax.set_xlim(0, max(values) * 1.15)
    plt.tight_layout()

    fig.savefig(OUTPUT_DIR / 'liana_top_lr_pairs.pdf')
    fig.savefig(OUTPUT_DIR / 'liana_top_lr_pairs.png')
    plt.close()
    print("Saved: liana_top_lr_pairs")


# =============================================================================
# Figure 10: Receptor Dotplot
# =============================================================================
def fig_receptor_dotplot(df, top_n=15):
    """Dot plot of top receptors by cell type."""
    fig, ax = plt.subplots(figsize=(12, 8))

    top_receptors = df.groupby('receptor_complex')['lrscore'].mean().nlargest(top_n).index.tolist()

    df_filt = df[df['receptor_complex'].isin(top_receptors)]
    pivot = df_filt.groupby(['target', 'receptor_complex']).agg({
        'lrscore': 'mean',
        'lr_means': 'mean'
    }).reset_index()

    cell_order = []
    for comp in ['Epithelial', 'Immune', 'Stromal']:
        for ct in COMPARTMENTS[comp]:
            if ct in pivot['target'].unique():
                cell_order.append(ct)

    for i, receptor in enumerate(top_receptors):
        for j, cell in enumerate(cell_order):
            subset = pivot[(pivot['receptor_complex'] == receptor) & (pivot['target'] == cell)]
            if len(subset) > 0:
                score = subset['lrscore'].values[0]
                expr = subset['lr_means'].values[0]

                size = 20 + 200 * (expr / pivot['lr_means'].max())
                color = plt.cm.RdYlBu_r((score - 0.5) / 0.5)

                ax.scatter(j, i, s=size, c=[color], edgecolors='white', linewidth=0.5)

    ax.set_xticks(range(len(cell_order)))
    ax.set_yticks(range(len(top_receptors)))

    short_labels = [short_name(c) for c in cell_order]
    ax.set_xticklabels(short_labels, rotation=45, ha='right')
    ax.set_yticklabels(top_receptors)

    ax.set_xlabel('Target Cell Type', fontweight='bold')
    ax.set_ylabel('Receptor', fontweight='bold')
    ax.set_title('Receptor Expression by Cell Type', fontweight='bold', fontsize=14)

    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.3, linestyle='--')

    for size_val, label in [(50, 'Low'), (150, 'Med'), (250, 'High')]:
        ax.scatter([], [], s=size_val, c='gray', label=f'{label} expr', edgecolors='white')
    ax.legend(title='Expression', loc='upper left', bbox_to_anchor=(1.02, 1), frameon=False)

    sm = plt.cm.ScalarMappable(cmap='RdYlBu_r', norm=plt.Normalize(0.5, 1.0))
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, pad=0.15)
    cbar.set_label('Interaction Score', fontweight='bold')

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'liana_receptor_dotplot.pdf')
    fig.savefig(OUTPUT_DIR / 'liana_receptor_dotplot.png')
    plt.close()
    print("Saved: liana_receptor_dotplot")


# =============================================================================
# Figure 11: Pathway Focus
# =============================================================================
def fig_pathway_focus(df):
    """Focus panel on key inflammatory pathways."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    pathways = {
        'IL1 Signaling': ['IL1B', 'IL1A', 'IL1R1', 'IL1R2', 'IL1RAP'],
        'TNF Signaling': ['TNF', 'TNFRSF1A', 'TNFRSF1B', 'TNFSF10', 'TNFRSF10'],
        'TGF-beta Signaling': ['TGFB1', 'TGFB2', 'TGFB3', 'TGFBR1', 'TGFBR2'],
    }

    for ax, (pathway_name, genes) in zip(axes, pathways.items()):
        mask = df['ligand_complex'].str.contains('|'.join(genes), na=False) | \
               df['receptor_complex'].str.contains('|'.join(genes), na=False)
        pathway_df = df[mask]

        if len(pathway_df) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(pathway_name, fontweight='bold')
            continue

        agg = pathway_df.groupby(['source', 'target'])['lrscore'].mean().unstack(fill_value=0)

        cell_order = []
        for comp in ['Epithelial', 'Immune', 'Stromal']:
            for ct in COMPARTMENTS[comp]:
                if ct in agg.index:
                    cell_order.append(ct)

        if len(cell_order) == 0:
            ax.text(0.5, 0.5, 'No matching cells', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(pathway_name, fontweight='bold')
            continue

        agg = agg.reindex(index=cell_order, columns=cell_order, fill_value=0)

        im = ax.imshow(agg.values, cmap='Reds', aspect='auto')

        short_labels = [short_name(c)[:6] for c in cell_order]
        ax.set_xticks(range(len(cell_order)))
        ax.set_yticks(range(len(cell_order)))
        ax.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(short_labels, fontsize=8)

        ax.set_title(pathway_name, fontweight='bold', fontsize=12)

        plt.colorbar(im, ax=ax, shrink=0.6)

    fig.suptitle('Key Inflammatory Pathway Communication', fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()

    fig.savefig(OUTPUT_DIR / 'liana_pathway_focus.pdf')
    fig.savefig(OUTPUT_DIR / 'liana_pathway_focus.png')
    plt.close()
    print("Saved: liana_pathway_focus")


# =============================================================================
# Main
# =============================================================================
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Generate LIANA figures")
    parser.add_argument("--input", "-i", type=str, default=None,
                        help="Path to LIANA parquet (default: auto-detect)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output directory (default: figures/publication)")
    args = parser.parse_args()

    if args.output:
        OUTPUT_DIR = Path(args.output)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading LIANA data...")
    df = load_data(args.input)
    print(f"Loaded {len(df)} interactions")
    print(f"  Sources: {df['source'].nunique()} cell types")
    print(f"  Targets: {df['target'].nunique()} cell types")

    print("\nComputing Ollivier-Ricci curvature...")
    G = compute_ricci_curvature(df)
    print(f"Graph has {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    print("\nGenerating figures...")

    # Ricci curvature figures
    fig_ricci_network(df)
    fig_ricci_heatmap(df, G)
    fig_bottleneck_analysis(df, G)

    # Network figures
    fig_chord_network(df)
    fig_il1b_network(df)

    # Heatmaps
    fig_heatmap(df)
    fig_compartment_flow(df)

    # Bar charts
    fig_source_target_bars(df)
    fig_top_lr_pairs(df)

    # Additional panels
    fig_receptor_dotplot(df)
    fig_pathway_focus(df)

    print(f"\nAll figures saved to {OUTPUT_DIR}")
