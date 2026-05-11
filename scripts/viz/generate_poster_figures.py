#!/usr/bin/env python3
"""Generate poster figures from z_fused embeddings and cell metadata.

Uses pre-computed z_fused (40D) embeddings from cells.parquet.
Handles stage labels: Normal/Preinvasive/Invasive.

Generates:
  1. UMAP by stage (z_fused)
  2. UMAP by cell type (z_fused)
  3. UMAP colored by pathway scores (NFkB, etc.)
  4. Signature violin plots by stage
  5. Cell type composition by stage
  6. Fold change summary
  7. HLCA vs LuCA embedding comparison (if attention available)

Usage:
    python scripts/viz/generate_poster_figures.py \
        --data-dir /home/booka/projects/StageBridge/data \
        --output-dir ./figures/poster

    # Or with remote data:
    python scripts/viz/generate_poster_figures.py \
        --data-dir /data1/chaunzt1/stagebridge/processed/luad_evo/canonical \
        --output-dir /data1/chaunzt1/stagebridge/figures/poster
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from typing import Optional, List, Tuple
import warnings

warnings.filterwarnings("ignore")

# Stage colors - matches your data
STAGE_COLORS = {
    'Normal': '#228B22',      # forest green
    'Preinvasive': '#4169E1', # royal blue
    'Invasive': '#CB4154',    # brick red
}
STAGE_ORDER = ['Normal', 'Preinvasive', 'Invasive']

# Top cell types to show (avoid clutter)
TOP_CELL_TYPES = [
    'T cell lineage', 'Macrophages', 'Fibroblast lineage',
    'AT2', 'pulmonary alveolar type 2 cell', 'Basal',
    'malignant cell', 'Capillary', 'capillary endothelial cell',
    'plasma cell', 'alveolar macrophage', 'Secretory'
]


def load_data(data_dir: Path) -> pd.DataFrame:
    """Load cells.parquet with embeddings."""
    path = data_dir / 'cells.parquet'
    if not path.exists():
        raise FileNotFoundError(f"cells.parquet not found at {path}")

    df = pd.read_parquet(path)
    print(f"Loaded {len(df):,} cells with {len(df.columns)} columns")
    print(f"Stages: {df['stage'].value_counts().to_dict()}")
    return df


def extract_embeddings(df: pd.DataFrame, col: str = 'z_fused') -> np.ndarray:
    """Extract embedding array from DataFrame column."""
    if col not in df.columns:
        raise ValueError(f"Column {col} not found")

    # Stack arrays
    emb = np.stack(df[col].values)
    print(f"Extracted {col}: {emb.shape}")
    return emb


def compute_umap(embeddings: np.ndarray, n_neighbors: int = 30,
                 min_dist: float = 0.3, random_state: int = 42) -> np.ndarray:
    """Compute UMAP coordinates."""
    try:
        import umap
        print(f"Computing UMAP on {embeddings.shape}...")
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=2,
            random_state=random_state,
            n_jobs=-1
        )
        coords = reducer.fit_transform(embeddings)
        print(f"UMAP complete: {coords.shape}")
        return coords
    except ImportError:
        print("ERROR: umap-learn not installed")
        return None


def plot_umap_by_stage(umap_coords: np.ndarray, stages: pd.Series,
                       output_dir: Path, suffix: str = '', title: str = None):
    """UMAP colored by disease stage."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for stage in STAGE_ORDER:
        mask = stages == stage
        n = mask.sum()
        if n > 0:
            ax.scatter(
                umap_coords[mask, 0], umap_coords[mask, 1],
                c=STAGE_COLORS.get(stage, 'gray'),
                s=1, alpha=0.5, label=f'{stage} (n={n:,})',
                rasterized=True
            )

    ax.legend(markerscale=8, frameon=False, loc='upper right', fontsize=11)
    ax.set_xlabel('UMAP 1', fontsize=12)
    ax.set_ylabel('UMAP 2', fontsize=12)
    ax.set_title(title or 'Embeddings by Disease Stage', fontsize=14, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    fname = f'umap_stage{suffix}'
    fig.savefig(output_dir / f'{fname}.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / f'{fname}.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {fname}")


def plot_three_umaps_comparison(df: pd.DataFrame, output_dir: Path) -> dict:
    """Side-by-side UMAPs: z_hlca, z_luca, z_fused colored by stage.

    Returns dict of UMAP coordinates for reuse.
    """

    # Check all three exist
    for col in ['z_hlca', 'z_luca', 'z_fused']:
        if col not in df.columns:
            print(f"Skipping three-UMAP comparison: {col} not found")
            return {}

    try:
        import umap
    except ImportError:
        print("UMAP not available")
        return {}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    embeddings = {
        'z_hlca': ('HLCA Reference', extract_embeddings(df, 'z_hlca')),
        'z_luca': ('LuCA Reference', extract_embeddings(df, 'z_luca')),
        'z_fused': ('Fused (StageBridge)', extract_embeddings(df, 'z_fused')),
    }

    stages = df['stage']
    umap_coords_dict = {}

    for idx, (col, (title, emb)) in enumerate(embeddings.items()):
        ax = axes[idx]

        print(f"Computing UMAP for {col}...")
        reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, n_components=2,
                           random_state=42, n_jobs=-1)
        coords = reducer.fit_transform(emb)
        umap_coords_dict[col] = coords

        # Save individual coords
        np.save(output_dir / f'umap_{col}.npy', coords)

        for stage in STAGE_ORDER:
            mask = stages == stage
            n = mask.sum()
            if n > 0:
                ax.scatter(
                    coords[mask, 0], coords[mask, 1],
                    c=STAGE_COLORS.get(stage, 'gray'),
                    s=0.5, alpha=0.5, label=f'{stage}' if idx == 2 else '',
                    rasterized=True
                )

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Single legend on the right
    axes[2].legend(markerscale=10, frameon=False, loc='upper right', fontsize=11)

    plt.tight_layout()
    fig.savefig(output_dir / 'umap_comparison_stage.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'umap_comparison_stage.pdf', bbox_inches='tight')
    plt.close(fig)
    print("Saved: umap_comparison_stage")

    return umap_coords_dict


def plot_three_umaps_celltype(df: pd.DataFrame, umap_coords_dict: dict,
                               output_dir: Path, top_n: int = 10):
    """Side-by-side UMAPs colored by cell type."""

    if not umap_coords_dict:
        print("No UMAP coords available for cell type plot")
        return

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    cell_types = df['cell_type']
    top_types = cell_types.value_counts().head(top_n).index.tolist()

    cmap = plt.cm.get_cmap('tab10', len(top_types))
    colors = {ct: cmap(i) for i, ct in enumerate(top_types)}

    titles = {
        'z_hlca': 'HLCA Reference',
        'z_luca': 'LuCA Reference',
        'z_fused': 'Fused (StageBridge)'
    }

    for idx, (col, coords) in enumerate(umap_coords_dict.items()):
        ax = axes[idx]

        # Plot "other" first
        other_mask = ~cell_types.isin(top_types)
        if other_mask.sum() > 0:
            ax.scatter(coords[other_mask, 0], coords[other_mask, 1],
                      c='lightgray', s=0.3, alpha=0.2, rasterized=True)

        # Plot top cell types
        for ct in top_types:
            mask = cell_types == ct
            if mask.sum() > 0:
                label = f'{ct[:20]}' if idx == 2 else ''
                ax.scatter(coords[mask, 0], coords[mask, 1],
                          c=[colors[ct]], s=0.5, alpha=0.5,
                          label=label, rasterized=True)

        ax.set_title(titles.get(col, col), fontsize=14, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    axes[2].legend(markerscale=8, frameon=False, loc='upper right',
                   fontsize=8, ncol=1)

    plt.tight_layout()
    fig.savefig(output_dir / 'umap_comparison_celltype.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'umap_comparison_celltype.pdf', bbox_inches='tight')
    plt.close(fig)
    print("Saved: umap_comparison_celltype")


def plot_emt_panel(df: pd.DataFrame, umap_coords: np.ndarray, output_dir: Path):
    """EMT-focused panel: UMAP + violin + stage progression."""

    if 'emt_score' not in df.columns:
        print("No emt_score column found")
        return

    fig = plt.figure(figsize=(16, 5))

    # Panel A: UMAP colored by EMT
    ax1 = fig.add_subplot(131)
    vals = df['emt_score'].values
    vmin, vmax = np.nanpercentile(vals[~np.isnan(vals)], [2, 98])

    scatter = ax1.scatter(
        umap_coords[:, 0], umap_coords[:, 1],
        c=vals, cmap='RdYlBu_r', s=1, alpha=0.6,
        vmin=vmin, vmax=vmax, rasterized=True
    )
    cbar = plt.colorbar(scatter, ax=ax1, shrink=0.6, pad=0.02)
    cbar.set_label('EMT Score', fontsize=10)
    ax1.set_title('A. EMT Score on Fused UMAP', fontsize=12, fontweight='bold')
    ax1.set_xticks([])
    ax1.set_yticks([])
    for spine in ax1.spines.values():
        spine.set_visible(False)

    # Panel B: Violin by stage
    ax2 = fig.add_subplot(132)
    data_by_stage = []
    positions = []
    colors = []

    for i, stage in enumerate(STAGE_ORDER):
        mask = df['stage'] == stage
        vals_stage = df.loc[mask, 'emt_score'].dropna().values
        if len(vals_stage) > 0:
            data_by_stage.append(vals_stage)
            positions.append(i)
            colors.append(STAGE_COLORS[stage])

    all_vals = np.concatenate(data_by_stage)
    ymin, ymax = np.percentile(all_vals, [1, 99])
    margin = (ymax - ymin) * 0.15

    parts = ax2.violinplot(data_by_stage, positions=positions, showmeans=True)
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    parts['cmeans'].set_color('black')
    parts['cmeans'].set_linewidth(2)

    ax2.set_ylim(ymin - margin, ymax + margin)
    ax2.set_xticks(range(len(STAGE_ORDER)))
    ax2.set_xticklabels(STAGE_ORDER, fontsize=11)
    ax2.set_ylabel('EMT Score', fontsize=11)
    ax2.set_title('B. EMT by Disease Stage', fontsize=12, fontweight='bold')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Panel C: Mean EMT with error bars
    ax3 = fig.add_subplot(133)
    means = []
    stds = []
    for stage in STAGE_ORDER:
        mask = df['stage'] == stage
        vals_stage = df.loc[mask, 'emt_score'].dropna().values
        means.append(np.mean(vals_stage))
        stds.append(np.std(vals_stage) / np.sqrt(len(vals_stage)))  # SEM

    x = range(len(STAGE_ORDER))
    bars = ax3.bar(x, means, yerr=stds, capsize=5,
                   color=[STAGE_COLORS[s] for s in STAGE_ORDER], alpha=0.8)

    # Add fold change annotation
    if means[0] != 0:
        fc_inv = means[2] / means[0]
        fc_pre = means[1] / means[0]
        ax3.text(1, means[1] + stds[1] + 0.02, f'{fc_pre:.2f}x',
                ha='center', fontsize=10)
        ax3.text(2, means[2] + stds[2] + 0.02, f'{fc_inv:.2f}x',
                ha='center', fontsize=10, fontweight='bold')

    ax3.set_xticks(x)
    ax3.set_xticklabels(STAGE_ORDER, fontsize=11)
    ax3.set_ylabel('Mean EMT Score', fontsize=11)
    ax3.set_title('C. EMT Progression', fontsize=12, fontweight='bold')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_dir / 'emt_panel.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'emt_panel.pdf', bbox_inches='tight')
    plt.close(fig)
    print("Saved: emt_panel")


def plot_biological_validation_panel(df: pd.DataFrame, umap_coords: np.ndarray,
                                     output_dir: Path):
    """Multi-panel showing biological consistency: EMT, Hypoxia, Senescence, CytoTRACE."""

    scores = ['emt_score', 'pathway_Hypoxia', 'senescence_score', 'cytotrace']
    titles = ['EMT', 'Hypoxia', 'Senescence', 'CytoTRACE (Stemness)']
    cmaps = ['RdYlBu_r', 'YlOrRd', 'PuRd', 'viridis']

    # Filter to existing
    existing = [(s, t, c) for s, t, c in zip(scores, titles, cmaps) if s in df.columns]

    if not existing:
        print("No biological scores found")
        return

    n = len(existing)
    fig, axes = plt.subplots(2, n, figsize=(5*n, 9))
    if n == 1:
        axes = axes.reshape(2, 1)

    for idx, (score, title, cmap) in enumerate(existing):
        # Top row: UMAP
        ax_umap = axes[0, idx]
        vals = df[score].values
        vmin, vmax = np.nanpercentile(vals[~np.isnan(vals)], [2, 98])

        scatter = ax_umap.scatter(
            umap_coords[:, 0], umap_coords[:, 1],
            c=vals, cmap=cmap, s=0.5, alpha=0.5,
            vmin=vmin, vmax=vmax, rasterized=True
        )
        plt.colorbar(scatter, ax=ax_umap, shrink=0.6, pad=0.02)
        ax_umap.set_title(title, fontsize=12, fontweight='bold')
        ax_umap.set_xticks([])
        ax_umap.set_yticks([])
        for spine in ax_umap.spines.values():
            spine.set_visible(False)

        # Bottom row: Violin
        ax_vio = axes[1, idx]
        data_by_stage = []
        positions = []
        colors = []

        for i, stage in enumerate(STAGE_ORDER):
            mask = df['stage'] == stage
            vals_stage = df.loc[mask, score].dropna().values
            if len(vals_stage) > 0:
                data_by_stage.append(vals_stage)
                positions.append(i)
                colors.append(STAGE_COLORS[stage])

        if data_by_stage:
            all_vals = np.concatenate(data_by_stage)
            ymin, ymax = np.percentile(all_vals, [1, 99])
            margin = (ymax - ymin) * 0.15

            parts = ax_vio.violinplot(data_by_stage, positions=positions, showmeans=True)
            for pc, color in zip(parts['bodies'], colors):
                pc.set_facecolor(color)
                pc.set_alpha(0.7)
            parts['cmeans'].set_color('black')
            parts['cmeans'].set_linewidth(2)

            ax_vio.set_ylim(ymin - margin, ymax + margin)

        ax_vio.set_xticks(range(len(STAGE_ORDER)))
        ax_vio.set_xticklabels(STAGE_ORDER, fontsize=10)
        ax_vio.set_ylabel('Score', fontsize=10)
        ax_vio.spines['top'].set_visible(False)
        ax_vio.spines['right'].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_dir / 'biological_validation.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'biological_validation.pdf', bbox_inches='tight')
    plt.close(fig)
    print("Saved: biological_validation")


def plot_umap_by_celltype(umap_coords: np.ndarray, cell_types: pd.Series,
                          output_dir: Path, top_n: int = 12):
    """UMAP colored by cell type (top N only for clarity)."""
    fig, ax = plt.subplots(figsize=(10, 7))

    # Get top cell types by count
    top_types = cell_types.value_counts().head(top_n).index.tolist()

    # Color map
    cmap = plt.cm.get_cmap('tab20', len(top_types))
    colors = {ct: cmap(i) for i, ct in enumerate(top_types)}

    # Plot "other" first (background)
    other_mask = ~cell_types.isin(top_types)
    if other_mask.sum() > 0:
        ax.scatter(
            umap_coords[other_mask, 0], umap_coords[other_mask, 1],
            c='lightgray', s=0.5, alpha=0.3, label='Other', rasterized=True
        )

    # Plot each cell type
    for ct in top_types:
        mask = cell_types == ct
        n = mask.sum()
        if n > 0:
            ax.scatter(
                umap_coords[mask, 0], umap_coords[mask, 1],
                c=[colors[ct]], s=1, alpha=0.6,
                label=f'{ct[:25]} ({n:,})', rasterized=True
            )

    ax.legend(markerscale=6, frameon=False, loc='upper right',
              fontsize=8, ncol=1)
    ax.set_xlabel('UMAP 1', fontsize=12)
    ax.set_ylabel('UMAP 2', fontsize=12)
    ax.set_title('Fused Embeddings by Cell Type', fontsize=14, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.savefig(output_dir / 'umap_celltype.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'umap_celltype.pdf', bbox_inches='tight')
    plt.close(fig)
    print("Saved: umap_celltype")


def plot_umap_by_score(umap_coords: np.ndarray, scores: pd.Series,
                       output_dir: Path, name: str, cmap: str = 'viridis'):
    """UMAP colored by continuous score."""
    fig, ax = plt.subplots(figsize=(8, 6))

    vals = scores.values
    vmin, vmax = np.nanpercentile(vals[~np.isnan(vals)], [2, 98])

    scatter = ax.scatter(
        umap_coords[:, 0], umap_coords[:, 1],
        c=vals, cmap=cmap, s=1, alpha=0.6,
        vmin=vmin, vmax=vmax, rasterized=True
    )
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label(name, fontsize=11)

    ax.set_xlabel('UMAP 1', fontsize=12)
    ax.set_ylabel('UMAP 2', fontsize=12)
    ax.set_title(f'{name}', fontsize=14, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    fname = f'umap_{name.lower().replace(" ", "_").replace("-", "_")}'
    fig.savefig(output_dir / f'{fname}.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / f'{fname}.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {fname}")


def plot_stage_distribution(stages: pd.Series, output_dir: Path):
    """Bar chart of cells per stage."""
    fig, ax = plt.subplots(figsize=(5, 4))

    counts = stages.value_counts().reindex(STAGE_ORDER)
    colors = [STAGE_COLORS[s] for s in STAGE_ORDER]

    bars = ax.bar(range(len(STAGE_ORDER)), counts.values, color=colors)

    for bar, count in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{int(count):,}', ha='center', va='bottom', fontsize=10)

    ax.set_xticks(range(len(STAGE_ORDER)))
    ax.set_xticklabels(STAGE_ORDER, fontsize=11)
    ax.set_ylabel('Cell Count', fontsize=12)
    ax.set_title('Cells by Disease Stage', fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.savefig(output_dir / 'stage_distribution.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'stage_distribution.pdf', bbox_inches='tight')
    plt.close(fig)
    print("Saved: stage_distribution")


def plot_celltype_composition(df: pd.DataFrame, output_dir: Path, top_n: int = 15):
    """Heatmap of cell type composition by stage."""
    fig, ax = plt.subplots(figsize=(6, 7))

    # Cross-tabulation normalized by stage
    ct_stage = pd.crosstab(df['cell_type'], df['stage'], normalize='columns')
    ct_stage = ct_stage.reindex(columns=STAGE_ORDER, fill_value=0)

    # Keep top N by max proportion
    top_types = ct_stage.max(axis=1).sort_values(ascending=False).head(top_n).index
    ct_stage = ct_stage.loc[top_types]

    im = ax.imshow(ct_stage.values, cmap='YlOrRd', aspect='auto')
    cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label('Proportion', fontsize=10)

    ax.set_xticks(range(len(STAGE_ORDER)))
    ax.set_xticklabels(STAGE_ORDER, fontsize=11)
    ax.set_yticks(range(len(ct_stage.index)))
    ax.set_yticklabels(ct_stage.index, fontsize=9)
    ax.set_title('Cell Type Composition', fontsize=14, fontweight='bold')

    fig.savefig(output_dir / 'celltype_composition.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'celltype_composition.pdf', bbox_inches='tight')
    plt.close(fig)
    print("Saved: celltype_composition")


def plot_signature_violins(df: pd.DataFrame, output_dir: Path,
                           signatures: Optional[List[str]] = None):
    """Violin plots of signatures by stage with robust scaling."""

    # Default signatures - pathway scores + biological
    if signatures is None:
        signatures = [
            'pathway_NFkB', 'pathway_TGFb', 'pathway_Hypoxia',
            'pathway_MAPK', 'pathway_JAK-STAT', 'pathway_p53',
            'emt_score', 'senescence_score', 'cytotrace'
        ]

    # Filter to existing columns
    signatures = [s for s in signatures if s in df.columns]

    if not signatures:
        print("No signature columns found")
        return

    n_sigs = len(signatures)
    n_cols = 3
    n_rows = (n_sigs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3.5*n_rows))
    axes = np.atleast_2d(axes).flatten()

    for idx, sig in enumerate(signatures):
        ax = axes[idx]

        data_by_stage = []
        positions = []
        colors = []

        for i, stage in enumerate(STAGE_ORDER):
            mask = df['stage'] == stage
            vals = df.loc[mask, sig].dropna().values
            if len(vals) > 0:
                data_by_stage.append(vals)
                positions.append(i)
                colors.append(STAGE_COLORS[stage])

        if not data_by_stage:
            ax.text(0.5, 0.5, f'{sig}\n(no data)', ha='center', va='center',
                   transform=ax.transAxes)
            continue

        # Robust y-limits
        all_vals = np.concatenate(data_by_stage)
        ymin, ymax = np.percentile(all_vals, [1, 99])
        margin = (ymax - ymin) * 0.15

        parts = ax.violinplot(data_by_stage, positions=positions,
                              showmeans=True, showmedians=False)

        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)

        parts['cmeans'].set_color('black')
        parts['cmeans'].set_linewidth(2)
        for part in ['cbars', 'cmins', 'cmaxs']:
            if part in parts:
                parts[part].set_color('black')
                parts[part].set_linewidth(0.8)

        ax.set_ylim(ymin - margin, ymax + margin)
        ax.set_xticks(range(len(STAGE_ORDER)))
        ax.set_xticklabels(STAGE_ORDER, fontsize=10)
        ax.set_ylabel('Score', fontsize=10)

        # Clean up name for title
        title = sig.replace('pathway_', '').replace('_score', '').replace('_', ' ')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    for idx in range(len(signatures), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_dir / 'signature_violins.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'signature_violins.pdf', bbox_inches='tight')
    plt.close(fig)
    print("Saved: signature_violins")


def compute_fold_changes(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Compute fold changes between stages."""

    # Find score/pathway columns
    score_cols = [c for c in df.columns if 'pathway_' in c or '_score' in c]

    results = []

    # Invasive vs Normal
    for col in score_cols:
        mean_normal = df.loc[df['stage'] == 'Normal', col].mean()
        mean_preinv = df.loc[df['stage'] == 'Preinvasive', col].mean()
        mean_inv = df.loc[df['stage'] == 'Invasive', col].mean()

        # Invasive vs Normal
        if mean_normal != 0 and not np.isnan(mean_normal):
            fc_inv_vs_norm = mean_inv / mean_normal
            results.append({
                'signature': col.replace('pathway_', '').replace('_score', ''),
                'comparison': 'Invasive vs Normal',
                'mean_A': mean_normal,
                'mean_B': mean_inv,
                'fold_change': fc_inv_vs_norm,
                'log2FC': np.log2(fc_inv_vs_norm) if fc_inv_vs_norm > 0 else np.nan
            })

        # Preinvasive vs Normal
        if mean_normal != 0 and not np.isnan(mean_normal):
            fc_pre_vs_norm = mean_preinv / mean_normal
            results.append({
                'signature': col.replace('pathway_', '').replace('_score', ''),
                'comparison': 'Preinvasive vs Normal',
                'mean_A': mean_normal,
                'mean_B': mean_preinv,
                'fold_change': fc_pre_vs_norm,
                'log2FC': np.log2(fc_pre_vs_norm) if fc_pre_vs_norm > 0 else np.nan
            })

    fc_df = pd.DataFrame(results)
    fc_df.to_csv(output_dir / 'fold_changes.csv', index=False)
    print(f"Saved: fold_changes.csv ({len(fc_df)} rows)")

    # Print notable results
    print("\nNotable fold changes (Invasive vs Normal, |log2FC| > 0.3):")
    inv_vs_norm = fc_df[fc_df['comparison'] == 'Invasive vs Normal'].copy()
    notable = inv_vs_norm[abs(inv_vs_norm['log2FC']) > 0.3].sort_values('log2FC', ascending=False)
    for _, row in notable.head(10).iterrows():
        direction = 'UP' if row['log2FC'] > 0 else 'DOWN'
        print(f"  {row['signature']}: {row['fold_change']:.2f}x ({direction}, log2FC={row['log2FC']:.2f})")

    return fc_df


def plot_fold_change_bar(fc_df: pd.DataFrame, output_dir: Path):
    """Bar chart of fold changes (Invasive vs Normal)."""

    inv_vs_norm = fc_df[fc_df['comparison'] == 'Invasive vs Normal'].copy()
    inv_vs_norm = inv_vs_norm.sort_values('log2FC', ascending=True)

    # Take top/bottom
    n_show = min(15, len(inv_vs_norm))

    fig, ax = plt.subplots(figsize=(8, 6))

    y_pos = range(n_show)
    colors = ['#CB4154' if x > 0 else '#4169E1' for x in inv_vs_norm['log2FC'].head(n_show)]

    ax.barh(y_pos, inv_vs_norm['log2FC'].head(n_show), color=colors, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(inv_vs_norm['signature'].head(n_show), fontsize=9)
    ax.set_xlabel('log2 Fold Change (Invasive vs Normal)', fontsize=12)
    ax.set_title('Pathway/Signature Changes', fontsize=14, fontweight='bold')
    ax.axvline(0, color='black', linewidth=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.savefig(output_dir / 'fold_change_bar.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'fold_change_bar.pdf', bbox_inches='tight')
    plt.close(fig)
    print("Saved: fold_change_bar")


def main():
    parser = argparse.ArgumentParser(description='Generate poster figures')
    parser.add_argument('--data-dir', type=Path, required=True,
                       help='Directory containing cells.parquet')
    parser.add_argument('--output-dir', type=Path, required=True,
                       help='Output directory for figures')
    parser.add_argument('--skip-umap', action='store_true',
                       help='Skip UMAP computation (if already have coords)')
    parser.add_argument('--umap-file', type=Path, default=None,
                       help='Pre-computed UMAP coordinates (.npy)')

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("STAGEBRIDGE POSTER FIGURE GENERATOR")
    print("="*60)
    print(f"Data: {args.data_dir}")
    print(f"Output: {args.output_dir}")

    # Load data
    print("\n--- Loading data ---")
    df = load_data(args.data_dir)

    # Extract fused embeddings
    print("\n--- Extracting embeddings ---")
    z_fused = extract_embeddings(df, 'z_fused')

    # Compute or load UMAP
    print("\n--- UMAP ---")
    if args.umap_file and args.umap_file.exists():
        umap_coords = np.load(args.umap_file)
        print(f"Loaded UMAP from {args.umap_file}: {umap_coords.shape}")
    elif args.skip_umap:
        print("Skipping UMAP (--skip-umap)")
        umap_coords = None
    else:
        umap_coords = compute_umap(z_fused)
        if umap_coords is not None:
            np.save(args.output_dir / 'umap_coords.npy', umap_coords)
            print(f"Saved UMAP to {args.output_dir / 'umap_coords.npy'}")

    # Generate figures
    print("\n--- Generating figures ---")

    # Stage distribution (always)
    plot_stage_distribution(df['stage'], args.output_dir)

    # Cell type composition (always)
    plot_celltype_composition(df, args.output_dir)

    # Three-panel UMAP comparison: HLCA vs LuCA vs Fused
    print("\n--- Three-panel UMAP comparison ---")
    plot_three_umaps_comparison(df, args.output_dir)

    # UMAP plots (if available)
    if umap_coords is not None:
        plot_umap_by_stage(umap_coords, df['stage'], args.output_dir)
        plot_umap_by_celltype(umap_coords, df['cell_type'], args.output_dir)

        # Score UMAPs
        score_cols = ['pathway_NFkB', 'pathway_Hypoxia', 'emt_score', 'cytotrace']
        for col in score_cols:
            if col in df.columns:
                plot_umap_by_score(umap_coords, df[col], args.output_dir,
                                  col.replace('pathway_', '').replace('_score', ''),
                                  cmap='magma')

    # Signature violins
    plot_signature_violins(df, args.output_dir)

    # Fold changes
    fc_df = compute_fold_changes(df, args.output_dir)
    plot_fold_change_bar(fc_df, args.output_dir)

    print("\n" + "="*60)
    print(f"COMPLETE - {args.output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
