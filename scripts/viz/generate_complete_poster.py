#!/usr/bin/env python3
"""Generate complete poster figures for StageBridge.

Creates a comprehensive set of figures covering:
1. Embedding overview (UMAP by stage, by data type)
2. Biological validation (EMT, senescence, cytotrace)
3. Pathway activation (NFkB, Hypoxia, TGFb, etc.)
4. Niche composition (CAF/immune fractions)
5. Spatial visualization (example sections)
6. Cell-cell interactions (IL1B from LIANA)
7. Fold changes summary

Usage:
    python scripts/viz/generate_complete_poster.py \
        --data-dir ./data \
        --output-dir ./figures/poster
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from typing import Optional, List
import warnings

warnings.filterwarnings("ignore")

# Stage configuration
STAGE_COLORS = {
    'Normal': '#228B22',
    'Preinvasive': '#4169E1',
    'Invasive': '#CB4154',
}
STAGE_ORDER = ['Normal', 'Preinvasive', 'Invasive']

DATA_TYPE_COLORS = {
    'snrna': '#9467bd',
    'spatial': '#2ca02c',
}


def load_all_data(data_dir: Path) -> dict:
    """Load all relevant data files."""
    data = {}

    # Main cells file
    cells_path = data_dir / 'cells.parquet'
    if cells_path.exists():
        data['cells'] = pd.read_parquet(cells_path)
        print(f"Loaded cells: {len(data['cells']):,} rows")

    # Signature scores (KAC, IL1_axis, AP1, etc.)
    scores_path = data_dir / 'signatures' / 'caf_kac_scores.parquet'
    if scores_path.exists():
        data['scores'] = pd.read_parquet(scores_path)
        print(f"Loaded scores: {len(data['scores']):,} rows, {len(data['scores'].columns)} cols")
        # Merge into cells if possible
        if 'cells' in data and len(data['scores']) == len(data['cells']):
            for col in data['scores'].columns:
                if col not in data['cells'].columns:
                    data['cells'][col] = data['scores'][col].values
            print(f"  Merged {len(data['scores'].columns)} score columns into cells")

    # Pre-computed UMAP
    umap_path = data_dir / 'embeddings' / 'umap_embedding.parquet'
    if umap_path.exists():
        umap_df = pd.read_parquet(umap_path)
        if 'UMAP1' in umap_df.columns and 'UMAP2' in umap_df.columns:
            data['umap'] = umap_df[['UMAP1', 'UMAP2']].values
        elif 'umap_1' in umap_df.columns:
            data['umap'] = umap_df[['umap_1', 'umap_2']].values
        else:
            data['umap'] = umap_df.iloc[:, :2].values
        print(f"Loaded pre-computed UMAP: {data['umap'].shape}")

    # LIANA interactions
    liana_path = data_dir / 'liana_interactions.parquet'
    if liana_path.exists():
        data['liana'] = pd.read_parquet(liana_path)
        print(f"Loaded LIANA: {len(data['liana']):,} interactions")

    # IL1B-specific interactions
    il1b_path = data_dir / 'communication' / 'il1b_interactions.parquet'
    if il1b_path.exists():
        data['il1b'] = pd.read_parquet(il1b_path)
        print(f"Loaded IL1B interactions: {len(data['il1b']):,} rows")

    # TF activity (AP-1)
    tf_path = data_dir / 'activity' / 'tf_activity_collectri.parquet'
    if tf_path.exists():
        data['tf_activity'] = pd.read_parquet(tf_path)
        print(f"Loaded TF activity: {data['tf_activity'].shape}")

    # Attention weights
    attn_path = data_dir / 'attention_weights.npz'
    if attn_path.exists():
        data['attention'] = np.load(attn_path)
        print(f"Loaded attention: {list(data['attention'].keys())}")

    return data


def extract_embeddings(df: pd.DataFrame, col: str) -> np.ndarray:
    """Extract embedding array from column of arrays."""
    return np.stack(df[col].values)


def compute_umap(emb: np.ndarray, cache_path: Optional[Path] = None) -> np.ndarray:
    """Compute UMAP, optionally caching result."""
    if cache_path and cache_path.exists():
        print(f"Loading cached UMAP from {cache_path}")
        return np.load(cache_path)

    try:
        import umap
        print(f"Computing UMAP on {emb.shape}...")
        reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=42, n_jobs=-1)
        coords = reducer.fit_transform(emb)

        if cache_path:
            np.save(cache_path, coords)
            print(f"Saved UMAP to {cache_path}")

        return coords
    except ImportError:
        print("UMAP not available")
        return None


# =============================================================================
# FIGURE 1: EMBEDDING OVERVIEW
# =============================================================================

def fig1_embedding_overview(df: pd.DataFrame, output_dir: Path, data: dict):
    """2-panel: UMAP by stage + UMAP by data type."""
    print("\n=== Figure 1: Embedding Overview ===")

    # Get UMAP - prefer pre-computed, fallback to computing from z_fused
    if 'umap' in data and data['umap'] is not None:
        coords = data['umap']
        print(f"Using pre-computed UMAP: {coords.shape}")
    else:
        z_fused = extract_embeddings(df, 'z_fused')
        umap_cache = output_dir / 'umap_fused_cache.npy'
        coords = compute_umap(z_fused, umap_cache)

    if coords is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: By stage
    ax = axes[0]
    for stage in STAGE_ORDER:
        mask = df['stage'] == stage
        n = mask.sum()
        ax.scatter(coords[mask, 0], coords[mask, 1],
                  c=STAGE_COLORS[stage], s=0.5, alpha=0.4,
                  label=f'{stage} ({n:,})', rasterized=True)
    ax.legend(markerscale=10, frameon=False, fontsize=11)
    ax.set_title('A. Fused Embeddings by Disease Stage', fontsize=14, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Panel B: By data type
    ax = axes[1]
    for dtype in ['snrna', 'spatial']:
        mask = df['data_type'] == dtype
        n = mask.sum()
        label = f'snRNA ({n:,})' if dtype == 'snrna' else f'Spatial ({n:,})'
        ax.scatter(coords[mask, 0], coords[mask, 1],
                  c=DATA_TYPE_COLORS[dtype], s=0.5, alpha=0.4,
                  label=label, rasterized=True)
    ax.legend(markerscale=10, frameon=False, fontsize=11)
    ax.set_title('B. Integration of snRNA + Spatial', fontsize=14, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    fig.savefig(output_dir / 'fig1_embedding_overview.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'fig1_embedding_overview.pdf', bbox_inches='tight')
    plt.close(fig)
    print("Saved: fig1_embedding_overview")

    return coords  # Return for reuse


# =============================================================================
# FIGURE 2: BIOLOGICAL VALIDATION
# =============================================================================

def fig2_biological_validation(df: pd.DataFrame, coords: np.ndarray, output_dir: Path):
    """Key biological scores - UMAP + violin for each."""
    print("\n=== Figure 2: Biological Validation ===")

    scores = [
        ('emt_score', 'EMT Score', 'RdYlBu_r'),
        ('KAC_score', 'KAC (Progenitor)', 'Purples'),
        ('IL1_axis_score', 'IL1 Axis', 'Reds'),
        ('AP1_score', 'AP-1 Activity', 'Oranges'),
    ]

    # Filter to existing
    scores = [(s, t, c) for s, t, c in scores if s in df.columns]
    n = len(scores)

    if n == 0:
        print("No biological scores found")
        return

    fig, axes = plt.subplots(2, n, figsize=(5*n, 9))
    if n == 1:
        axes = axes.reshape(2, 1)

    for idx, (score, title, cmap) in enumerate(scores):
        vals = df[score].values
        valid = ~np.isnan(vals)
        vmin, vmax = np.nanpercentile(vals[valid], [2, 98])

        # Top: UMAP
        ax = axes[0, idx]
        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=vals,
                            cmap=cmap, s=0.3, alpha=0.5,
                            vmin=vmin, vmax=vmax, rasterized=True)
        plt.colorbar(scatter, ax=ax, shrink=0.6)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Bottom: Violin
        ax = axes[1, idx]
        data_by_stage = []
        colors = []
        for stage in STAGE_ORDER:
            mask = df['stage'] == stage
            v = df.loc[mask, score].dropna().values
            if len(v) > 0:
                data_by_stage.append(v)
                colors.append(STAGE_COLORS[stage])

        if data_by_stage:
            all_v = np.concatenate(data_by_stage)
            ymin, ymax = np.percentile(all_v, [1, 99])
            margin = (ymax - ymin) * 0.15

            parts = ax.violinplot(data_by_stage, showmeans=True)
            for pc, color in zip(parts['bodies'], colors):
                pc.set_facecolor(color)
                pc.set_alpha(0.7)
            parts['cmeans'].set_color('black')
            parts['cmeans'].set_linewidth(2)

            ax.set_ylim(ymin - margin, ymax + margin)

        ax.set_xticks(range(1, len(STAGE_ORDER)+1))
        ax.set_xticklabels(STAGE_ORDER, fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_dir / 'fig2_biological_validation.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'fig2_biological_validation.pdf', bbox_inches='tight')
    plt.close(fig)
    print("Saved: fig2_biological_validation")


# =============================================================================
# FIGURE 3: PATHWAY ACTIVATION
# =============================================================================

def fig3_pathway_activation(df: pd.DataFrame, output_dir: Path):
    """Pathway scores by stage - key ones only."""
    print("\n=== Figure 3: Pathway Activation ===")

    pathways = ['pathway_NFkB', 'pathway_Hypoxia', 'pathway_TGFb',
                'pathway_MAPK', 'pathway_JAK-STAT', 'pathway_p53']
    pathways = [p for p in pathways if p in df.columns]

    if not pathways:
        print("No pathway columns found")
        return

    n = len(pathways)
    n_cols = 3
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3.5*n_rows))
    axes = np.atleast_2d(axes).flatten()

    for idx, pathway in enumerate(pathways):
        ax = axes[idx]

        data_by_stage = []
        colors = []
        for stage in STAGE_ORDER:
            mask = df['stage'] == stage
            v = df.loc[mask, pathway].dropna().values
            if len(v) > 0:
                data_by_stage.append(v)
                colors.append(STAGE_COLORS[stage])

        if data_by_stage:
            all_v = np.concatenate(data_by_stage)
            ymin, ymax = np.percentile(all_v, [1, 99])
            margin = (ymax - ymin) * 0.15

            parts = ax.violinplot(data_by_stage, showmeans=True)
            for pc, color in zip(parts['bodies'], colors):
                pc.set_facecolor(color)
                pc.set_alpha(0.7)
            parts['cmeans'].set_color('black')
            parts['cmeans'].set_linewidth(2)

            ax.set_ylim(ymin - margin, ymax + margin)

        ax.set_xticks(range(1, len(STAGE_ORDER)+1))
        ax.set_xticklabels(STAGE_ORDER, fontsize=10)
        ax.set_title(pathway.replace('pathway_', ''), fontsize=11, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    for idx in range(len(pathways), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_dir / 'fig3_pathway_activation.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'fig3_pathway_activation.pdf', bbox_inches='tight')
    plt.close(fig)
    print("Saved: fig3_pathway_activation")


# =============================================================================
# FIGURE 4: NICHE COMPOSITION
# =============================================================================

def fig4_niche_composition(df: pd.DataFrame, output_dir: Path):
    """CAF and immune fractions by stage."""
    print("\n=== Figure 4: Niche Composition ===")

    fractions = [
        ('caf_fraction', 'CAF Fraction'),
        ('immune_fraction', 'Immune Fraction'),
    ]
    fractions = [(f, t) for f, t in fractions if f in df.columns]

    if not fractions:
        print("No fraction columns found")
        return

    fig, axes = plt.subplots(1, len(fractions), figsize=(5*len(fractions), 4))
    if len(fractions) == 1:
        axes = [axes]

    for idx, (frac, title) in enumerate(fractions):
        ax = axes[idx]

        data_by_stage = []
        colors = []
        means = []

        for stage in STAGE_ORDER:
            mask = df['stage'] == stage
            v = df.loc[mask, frac].dropna().values
            if len(v) > 0:
                data_by_stage.append(v)
                colors.append(STAGE_COLORS[stage])
                means.append(np.mean(v))

        if data_by_stage:
            all_v = np.concatenate(data_by_stage)
            ymin, ymax = np.percentile(all_v, [1, 99])
            margin = (ymax - ymin) * 0.15

            parts = ax.violinplot(data_by_stage, showmeans=True)
            for pc, color in zip(parts['bodies'], colors):
                pc.set_facecolor(color)
                pc.set_alpha(0.7)
            parts['cmeans'].set_color('black')
            parts['cmeans'].set_linewidth(2)

            ax.set_ylim(max(0, ymin - margin), ymax + margin)

            # Add fold change
            if means[0] > 0:
                fc = means[-1] / means[0]
                ax.text(0.95, 0.95, f'{fc:.2f}x', transform=ax.transAxes,
                       ha='right', va='top', fontsize=12, fontweight='bold')

        ax.set_xticks(range(1, len(STAGE_ORDER)+1))
        ax.set_xticklabels(STAGE_ORDER, fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_dir / 'fig4_niche_composition.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'fig4_niche_composition.pdf', bbox_inches='tight')
    plt.close(fig)
    print("Saved: fig4_niche_composition")


# =============================================================================
# FIGURE 5: SPATIAL VISUALIZATION
# =============================================================================

def fig5_spatial_example(df: pd.DataFrame, output_dir: Path, donor: str = None):
    """Example spatial section colored by cell type and EMT."""
    print("\n=== Figure 5: Spatial Example ===")

    spatial = df[df['data_type'] == 'spatial'].copy()

    if len(spatial) == 0:
        print("No spatial data")
        return

    # Pick donor with most spots if not specified
    if donor is None:
        donor = spatial['donor_id'].value_counts().index[0]

    section = spatial[spatial['donor_id'] == donor].copy()
    print(f"Donor {donor}: {len(section)} spots")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: By cell type
    ax = axes[0]
    cell_types = section['cell_type'].value_counts().head(8).index
    cmap = plt.cm.get_cmap('tab10', len(cell_types))
    colors = {ct: cmap(i) for i, ct in enumerate(cell_types)}

    for ct in cell_types:
        mask = section['cell_type'] == ct
        if mask.sum() > 0:
            ax.scatter(section.loc[mask, 'x'], section.loc[mask, 'y'],
                      c=[colors[ct]], s=3, alpha=0.7,
                      label=ct[:20], rasterized=True)

    ax.legend(markerscale=3, frameon=False, fontsize=8, loc='upper right')
    ax.set_title(f'A. Cell Types ({donor})', fontsize=12, fontweight='bold')
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.invert_yaxis()

    # Panel B: By EMT score
    ax = axes[1]
    if 'emt_score' in section.columns:
        vals = section['emt_score'].values
        valid = ~np.isnan(vals)
        vmin, vmax = np.nanpercentile(vals[valid], [2, 98])

        scatter = ax.scatter(section['x'], section['y'], c=vals,
                            cmap='RdYlBu_r', s=3, alpha=0.7,
                            vmin=vmin, vmax=vmax, rasterized=True)
        plt.colorbar(scatter, ax=ax, shrink=0.6, label='EMT Score')

    ax.set_title(f'B. EMT Score ({donor})', fontsize=12, fontweight='bold')
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.invert_yaxis()

    plt.tight_layout()
    fig.savefig(output_dir / 'fig5_spatial_example.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'fig5_spatial_example.pdf', bbox_inches='tight')
    plt.close(fig)
    print("Saved: fig5_spatial_example")


# =============================================================================
# FIGURE 6: IL1B INTERACTIONS (LIANA)
# =============================================================================

def fig6_il1b_interactions(liana: pd.DataFrame, output_dir: Path):
    """IL1B interactions from LIANA."""
    print("\n=== Figure 6: IL1B Interactions ===")

    # Filter to IL1B
    il1b = liana[liana['ligand_complex'].str.contains('IL1B', na=False)].copy()

    if len(il1b) == 0:
        print("No IL1B interactions found")
        return

    # Sort by score
    il1b = il1b.sort_values('lrscore', ascending=False)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Create interaction labels
    il1b['interaction'] = il1b['source'] + ' → ' + il1b['target']

    # Top interactions
    top = il1b.head(10)

    y_pos = range(len(top))
    ax.barh(y_pos, top['lrscore'], color='#CB4154', alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top['interaction'], fontsize=10)
    ax.set_xlabel('LIANA Score', fontsize=12)
    ax.set_title('IL1B Signaling Interactions', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add receptor info
    for i, (_, row) in enumerate(top.iterrows()):
        ax.text(row['lrscore'] + 0.01, i, row['receptor_complex'],
               va='center', fontsize=8, style='italic')

    plt.tight_layout()
    fig.savefig(output_dir / 'fig6_il1b_interactions.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'fig6_il1b_interactions.pdf', bbox_inches='tight')
    plt.close(fig)
    print("Saved: fig6_il1b_interactions")


# =============================================================================
# FIGURE 7: FOLD CHANGE SUMMARY
# =============================================================================

def fig7_fold_changes(df: pd.DataFrame, output_dir: Path):
    """Summary of key fold changes Invasive vs Normal."""
    print("\n=== Figure 7: Fold Changes ===")

    # Columns to analyze
    cols = [
        ('emt_score', 'EMT'),
        ('senescence_score', 'Senescence'),
        ('sasp_score', 'SASP'),
        ('caf_fraction', 'CAF Fraction'),
        ('immune_fraction', 'Immune Fraction'),
        ('pathway_NFkB', 'NFkB'),
        ('pathway_Hypoxia', 'Hypoxia'),
        ('pathway_TGFb', 'TGFb'),
        ('cytotrace', 'CytoTRACE'),
    ]
    cols = [(c, n) for c, n in cols if c in df.columns]

    results = []
    for col, name in cols:
        mean_norm = df.loc[df['stage'] == 'Normal', col].mean()
        mean_inv = df.loc[df['stage'] == 'Invasive', col].mean()

        if mean_norm != 0 and not np.isnan(mean_norm):
            fc = mean_inv / mean_norm
            log2fc = np.log2(fc) if fc > 0 else np.nan
            results.append({'name': name, 'fold_change': fc, 'log2fc': log2fc})

    if not results:
        print("No fold changes computed")
        return

    fc_df = pd.DataFrame(results).sort_values('log2fc')

    # Save to CSV
    fc_df.to_csv(output_dir / 'fold_changes.csv', index=False)

    fig, ax = plt.subplots(figsize=(8, 6))

    y_pos = range(len(fc_df))
    colors = ['#CB4154' if x > 0 else '#4169E1' for x in fc_df['log2fc']]

    ax.barh(y_pos, fc_df['log2fc'], color=colors, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(fc_df['name'], fontsize=11)
    ax.set_xlabel('log2 Fold Change (Invasive vs Normal)', fontsize=12)
    ax.set_title('Progression-Associated Changes', fontsize=14, fontweight='bold')
    ax.axvline(0, color='black', linewidth=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add FC values
    for i, (_, row) in enumerate(fc_df.iterrows()):
        xpos = row['log2fc'] + (0.05 if row['log2fc'] > 0 else -0.05)
        ha = 'left' if row['log2fc'] > 0 else 'right'
        ax.text(xpos, i, f"{row['fold_change']:.2f}x", va='center', ha=ha, fontsize=9)

    plt.tight_layout()
    fig.savefig(output_dir / 'fig7_fold_changes.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'fig7_fold_changes.pdf', bbox_inches='tight')
    plt.close(fig)
    print("Saved: fig7_fold_changes")

    # Print notable
    print("\nKey fold changes:")
    for _, row in fc_df.iterrows():
        print(f"  {row['name']}: {row['fold_change']:.2f}x")


# =============================================================================
# FIGURE 8: MODEL COMPARISON (ABLATIONS)
# =============================================================================

def fig8_model_comparison(data_dir: Path, output_dir: Path):
    """Compare full model vs ablations and baselines."""
    print("\n=== Figure 8: Model Comparison ===")

    import json
    from pathlib import Path

    # Look for metrics in various places
    metrics_dirs = [
        data_dir,  # If extracted here
        data_dir.parent / 'outputs' / 'v1.1',  # Standard location
    ]

    results = []

    for base in metrics_dirs:
        if not base.exists():
            continue

        # Full model
        for fold in range(5):
            for seed in [42, 43, 44]:
                path = base / 'full' / f'fold_{fold}' / f'seed_{seed}' / 'logs' / 'metrics.json'
                if path.exists():
                    with open(path) as f:
                        m = json.load(f)
                    results.append({
                        'model': 'StageBridge',
                        'fold': fold, 'seed': seed,
                        'val_loss': m.get('best_val_loss', m.get('val_loss', None)),
                        'val_acc': m.get('val_acc', m.get('best_val_acc', None)),
                    })

        # Ablations
        for ablation in ['no_niche', 'hlca_only', 'luca_only', 'no_distance']:
            for fold in range(5):
                for seed in [42, 43, 44]:
                    path = base / 'ablations' / ablation / f'fold_{fold}' / f'seed_{seed}' / 'logs' / 'metrics.json'
                    if path.exists():
                        with open(path) as f:
                            m = json.load(f)
                        results.append({
                            'model': ablation.replace('_', ' ').title(),
                            'fold': fold, 'seed': seed,
                            'val_loss': m.get('best_val_loss', m.get('val_loss', None)),
                            'val_acc': m.get('val_acc', m.get('best_val_acc', None)),
                        })

        # Baselines
        for baseline in ['pooling', 'deepsets', 'set_transformer', 'graphsage']:
            for fold in range(5):
                for seed in [42, 43, 44]:
                    path = base / 'baselines' / baseline / f'fold_{fold}' / f'seed_{seed}' / f'baseline_{baseline}.json'
                    if path.exists():
                        with open(path) as f:
                            m = json.load(f)
                        results.append({
                            'model': baseline.replace('_', ' ').title(),
                            'fold': fold, 'seed': seed,
                            'val_loss': m.get('val_loss', None),
                            'val_acc': m.get('val_acc', m.get('accuracy', None)),
                        })

    if not results:
        print("No metrics found")
        return

    df = pd.DataFrame(results)
    print(f"Loaded {len(df)} runs across {df['model'].nunique()} models")

    # Aggregate by model
    summary = df.groupby('model').agg({
        'val_loss': ['mean', 'std'],
        'val_acc': ['mean', 'std'],
    }).round(4)

    # Save summary
    summary.to_csv(output_dir / 'model_comparison.csv')
    print(f"Saved: model_comparison.csv")
    print(summary)

    # Plot
    model_order = ['StageBridge', 'No Niche', 'Hlca Only', 'Luca Only',
                   'Graphsage', 'Set Transformer', 'Deepsets', 'Pooling']
    model_order = [m for m in model_order if m in df['model'].unique()]

    if not model_order:
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    means = df.groupby('model')['val_loss'].mean().reindex(model_order)
    stds = df.groupby('model')['val_loss'].std().reindex(model_order)

    colors = ['#228B22' if m == 'StageBridge' else '#4169E1' if 'Only' in m or 'Niche' in m else '#888888'
              for m in model_order]

    x = range(len(model_order))
    ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(model_order, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Validation Loss', fontsize=12)
    ax.set_title('Model Comparison (5-fold CV, 3 seeds)', fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_dir / 'fig8_model_comparison.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'fig8_model_comparison.pdf', bbox_inches='tight')
    plt.close(fig)
    print("Saved: fig8_model_comparison")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate complete poster figures')
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--skip-umap', action='store_true', help='Skip UMAP computation')

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("STAGEBRIDGE COMPLETE POSTER GENERATOR")
    print("="*60)

    # Load data
    data = load_all_data(args.data_dir)

    if 'cells' not in data:
        print("ERROR: No cells.parquet found")
        return

    df = data['cells']

    # Generate figures
    coords = fig1_embedding_overview(df, args.output_dir, data)

    if coords is not None:
        fig2_biological_validation(df, coords, args.output_dir)

    fig3_pathway_activation(df, args.output_dir)
    fig4_niche_composition(df, args.output_dir)
    fig5_spatial_example(df, args.output_dir)

    if 'liana' in data:
        fig6_il1b_interactions(data['liana'], args.output_dir)

    fig7_fold_changes(df, args.output_dir)
    fig8_model_comparison(args.data_dir, args.output_dir)

    print("\n" + "="*60)
    print(f"COMPLETE - Figures saved to: {args.output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
