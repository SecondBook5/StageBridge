#!/usr/bin/env python3
"""
COMPLETE StageBridge Poster Figure Generator

Uses ALL available data:
- cells.parquet (embeddings, stage, cell_type)
- signatures/caf_kac_scores.parquet (KAC, IL1_axis, AP1, etc.)
- progression/progression_scores.parquet (cytotrace, pseudotime)
- trajectories/diffusion_pseudotime.parquet
- activity/tf_activity_collectri.parquet (AP-1, etc.)
- communication/il1b_interactions.parquet
- embeddings/umap_embedding.parquet
- Model outputs (metrics, attention, predictions)

Run on HPC with GPUs:
    python scripts/viz/generate_full_poster.py \
        --data-dir /data1/chaunzt1/stagebridge/processed/luad_evo/canonical \
        --model-dir /data1/chaunzt1/stagebridge/outputs/v1.1 \
        --output-dir /data1/chaunzt1/stagebridge/figures/poster
"""

import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from collections import defaultdict
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

STAGE_COLORS = {
    'Normal': '#228B22',
    'Preinvasive': '#4169E1',
    'Invasive': '#CB4154',
    # Also handle 5-stage
    'AAH': '#4682B4',
    'AIS': '#4169E1',
    'MIA': '#8B008B',
    'LUAD': '#CB4154',
}

STAGE_ORDER_3 = ['Normal', 'Preinvasive', 'Invasive']
STAGE_ORDER_5 = ['Normal', 'AAH', 'AIS', 'MIA', 'LUAD']

MODEL_COLORS = {
    'StageBridge': '#228B22',
    'No Niche': '#CB4154',
    'HLCA Only': '#4169E1',
    'LuCA Only': '#9467bd',
    'No Distance': '#ff7f0e',
    'GraphSAGE': '#888888',
    'Set Transformer': '#888888',
    'DeepSets': '#888888',
    'Pooling': '#888888',
}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_all_data(data_dir: Path, model_dir: Path) -> Dict:
    """Load all available data files."""
    data = {}

    print("="*60)
    print("LOADING DATA")
    print("="*60)

    # --- Core data ---
    cells_path = data_dir / 'cells.parquet'
    if cells_path.exists():
        data['cells'] = pd.read_parquet(cells_path)
        print(f"[+] cells.parquet: {len(data['cells']):,} cells")

    # --- Signatures (KAC, IL1_axis, AP1, etc.) ---
    sig_path = data_dir / 'signatures' / 'caf_kac_scores.parquet'
    if sig_path.exists():
        data['signatures'] = pd.read_parquet(sig_path)
        print(f"[+] caf_kac_scores.parquet: {data['signatures'].shape}")
        # List score columns
        score_cols = [c for c in data['signatures'].columns if 'score' in c.lower()]
        print(f"    Scores: {score_cols}")

    # --- Progression (cytotrace, pseudotime) ---
    prog_path = data_dir / 'progression' / 'progression_scores.parquet'
    if prog_path.exists():
        data['progression'] = pd.read_parquet(prog_path)
        print(f"[+] progression_scores.parquet: {data['progression'].shape}")
        print(f"    Columns: {list(data['progression'].columns)}")

    # --- Trajectories (diffusion pseudotime) ---
    traj_path = data_dir / 'trajectories' / 'diffusion_pseudotime.parquet'
    if traj_path.exists():
        data['trajectories'] = pd.read_parquet(traj_path)
        print(f"[+] diffusion_pseudotime.parquet: {data['trajectories'].shape}")

    # --- TF Activity ---
    tf_path = data_dir / 'activity' / 'tf_activity_collectri.parquet'
    if tf_path.exists():
        data['tf_activity'] = pd.read_parquet(tf_path)
        print(f"[+] tf_activity_collectri.parquet: {data['tf_activity'].shape}")

    # --- Pathway activity ---
    pw_path = data_dir / 'activity' / 'pathway_activity_progeny.parquet'
    if pw_path.exists():
        data['pathway_activity'] = pd.read_parquet(pw_path)
        print(f"[+] pathway_activity_progeny.parquet: {data['pathway_activity'].shape}")

    # --- IL1B interactions ---
    il1b_path = data_dir / 'communication' / 'il1b_interactions.parquet'
    if il1b_path.exists():
        data['il1b'] = pd.read_parquet(il1b_path)
        print(f"[+] il1b_interactions.parquet: {len(data['il1b'])} interactions")

    # --- Pre-computed UMAP ---
    umap_path = data_dir / 'embeddings' / 'umap_embedding.parquet'
    if umap_path.exists():
        umap_df = pd.read_parquet(umap_path)
        # Handle different column names
        if 'UMAP1' in umap_df.columns:
            data['umap'] = umap_df[['UMAP1', 'UMAP2']].values
        elif 'umap_1' in umap_df.columns:
            data['umap'] = umap_df[['umap_1', 'umap_2']].values
        else:
            data['umap'] = umap_df.iloc[:, :2].values
        print(f"[+] umap_embedding.parquet: {data['umap'].shape}")

    # --- LIANA interactions ---
    liana_path = data_dir / 'liana_interactions.parquet'
    if liana_path.exists():
        data['liana'] = pd.read_parquet(liana_path)
        print(f"[+] liana_interactions.parquet: {len(data['liana'])} interactions")

    # --- Rare cells ---
    rare_path = data_dir / 'rare_cells' / 'rare_cell_signatures.parquet'
    if rare_path.exists():
        data['rare_cells'] = pd.read_parquet(rare_path)
        print(f"[+] rare_cell_signatures.parquet: {data['rare_cells'].shape}")

    # --- OT dynamics (flux, curl, divergence) ---
    for ot_name in ['ot_dynamics.parquet', 'velocity_field.parquet', 'flow_analysis.parquet']:
        ot_path = data_dir / ot_name
        if ot_path.exists():
            data['ot_dynamics'] = pd.read_parquet(ot_path)
            print(f"[+] {ot_name}: {data['ot_dynamics'].shape}")
            break
    # Also check in progression folder
    ot_path = data_dir / 'progression' / 'ot_dynamics.parquet'
    if ot_path.exists() and 'ot_dynamics' not in data:
        data['ot_dynamics'] = pd.read_parquet(ot_path)
        print(f"[+] progression/ot_dynamics.parquet: {data['ot_dynamics'].shape}")

    # --- Model metrics ---
    if model_dir.exists():
        data['model_metrics'] = load_model_metrics(model_dir)
        print(f"[+] Model metrics: {len(data['model_metrics'])} runs")

    # --- Inference outputs ---
    inf_dir = model_dir / 'inference' / 'full'
    if inf_dir.exists():
        data['inference'] = load_inference_outputs(inf_dir)
        print(f"[+] Inference outputs loaded")

    return data


def extract_val_loss(m: dict) -> float:
    """Extract validation loss from metrics dict, handling various formats."""
    # Try different keys
    for key in ['best_val_loss', 'val_loss', 'best_loss', 'validation_loss']:
        if key in m and m[key] is not None:
            val = m[key]
            # Handle if it's a list (take last or min)
            if isinstance(val, list):
                return min(val) if val else None
            return val
    # Check nested structure
    if 'metrics' in m:
        return extract_val_loss(m['metrics'])
    if 'history' in m and 'val_loss' in m['history']:
        return min(m['history']['val_loss'])
    return None


def load_model_metrics(model_dir: Path) -> pd.DataFrame:
    """Load all model training metrics."""
    results = []

    # Full model
    for fold in range(5):
        for seed in [42, 43, 44]:
            path = model_dir / 'full' / f'fold_{fold}' / f'seed_{seed}' / 'logs' / 'metrics.json'
            if path.exists():
                with open(path) as f:
                    m = json.load(f)
                val_loss = extract_val_loss(m)
                if val_loss is not None:
                    results.append({
                        'model': 'StageBridge',
                        'fold': fold, 'seed': seed,
                        'val_loss': val_loss,
                    })
                else:
                    print(f"  Warning: No val_loss in {path}, keys: {list(m.keys())}")

    # Ablations
    ablation_names = {
        'no_niche': 'No Niche',
        'hlca_only': 'HLCA Only',
        'luca_only': 'LuCA Only',
        'no_distance': 'No Distance',
        'no_gate': 'No Gate',
    }

    for ablation, name in ablation_names.items():
        for fold in range(5):
            for seed in [42, 43, 44]:
                path = model_dir / 'ablations' / ablation / f'fold_{fold}' / f'seed_{seed}' / 'logs' / 'metrics.json'
                if path.exists():
                    with open(path) as f:
                        m = json.load(f)
                    val_loss = extract_val_loss(m)
                    if val_loss is not None:
                        results.append({
                            'model': name,
                            'fold': fold, 'seed': seed,
                            'val_loss': val_loss,
                        })

    # Baselines
    baseline_names = {
        'pooling': 'Pooling',
        'deepsets': 'DeepSets',
        'set_transformer': 'Set Transformer',
        'graphsage': 'GraphSAGE',
    }

    for baseline, name in baseline_names.items():
        for fold in range(5):
            for seed in [42, 43, 44]:
                path = model_dir / 'baselines' / baseline / f'fold_{fold}' / f'seed_{seed}' / f'baseline_{baseline}.json'
                if path.exists():
                    with open(path) as f:
                        m = json.load(f)
                    val_loss = extract_val_loss(m)
                    if val_loss is not None:
                        results.append({
                            'model': name,
                            'fold': fold, 'seed': seed,
                            'val_loss': val_loss,
                        })

    if results:
        print(f"  Loaded {len(results)} metric records")
    else:
        print("  Warning: No metrics found, checking first file structure...")
        # Debug: show what's in a metrics file
        sample_path = model_dir / 'full' / 'fold_0' / 'seed_42' / 'logs' / 'metrics.json'
        if sample_path.exists():
            with open(sample_path) as f:
                m = json.load(f)
            print(f"  Sample keys: {list(m.keys())}")
            for k, v in m.items():
                print(f"    {k}: {type(v).__name__} = {str(v)[:100]}")

    return pd.DataFrame(results)


def load_inference_outputs(inf_dir: Path) -> Dict:
    """Load inference outputs (predictions, attention, embeddings, displacements)."""
    outputs = {}

    # Load from fold_0/seed_42 as representative
    run_dir = inf_dir / 'fold_0' / 'seed_42'

    if (run_dir / 'predictions.parquet').exists():
        outputs['predictions'] = pd.read_parquet(run_dir / 'predictions.parquet')
        print(f"    predictions: {len(outputs['predictions'])} cells")

    if (run_dir / 'embeddings.parquet').exists():
        outputs['embeddings'] = pd.read_parquet(run_dir / 'embeddings.parquet')
        print(f"    embeddings: {outputs['embeddings'].shape}")

    if (run_dir / 'attention_weights.npz').exists():
        outputs['attention'] = np.load(run_dir / 'attention_weights.npz')
        attn = outputs['attention']['attention']
        print(f"    attention: {attn.shape}, range: [{attn.min():.4f}, {attn.max():.4f}]")

    if (run_dir / 'displacements.npy').exists():
        outputs['displacements'] = np.load(run_dir / 'displacements.npy')
        disp = outputs['displacements']
        print(f"    displacements: {disp.shape}, range: [{disp.min():.4f}, {disp.max():.4f}]")

    # Aggregate displacements from all runs for more coverage
    all_displacements = {}
    for fold_dir in sorted(inf_dir.glob('fold_*')):
        for seed_dir in sorted(fold_dir.glob('seed_*')):
            disp_path = seed_dir / 'displacements.npy'
            pred_path = seed_dir / 'predictions.parquet'
            if disp_path.exists() and pred_path.exists():
                disp = np.load(disp_path)
                pred_df = pd.read_parquet(pred_path)
                for i, cid in enumerate(pred_df['cell_id'].values):
                    if cid not in all_displacements:
                        all_displacements[cid] = []
                    all_displacements[cid].append(disp[i])

    if all_displacements:
        # Average across runs
        outputs['all_displacements'] = {
            cid: np.mean(vels, axis=0) for cid, vels in all_displacements.items()
        }
        print(f"    aggregated displacements: {len(outputs['all_displacements'])} cells")

    return outputs


def merge_all_scores(data: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Merge all scores into the cells dataframe.

    Returns:
        df_all: Full dataframe (all cells)
        df_snrna: snRNA-only dataframe with signatures merged
    """
    df = data['cells'].copy()
    n_cells = len(df)

    # Separate snRNA for signature analysis
    df_snrna = df[df['data_type'] == 'snrna'].copy().reset_index(drop=True)
    n_snrna = len(df_snrna)
    print(f"Total cells: {n_cells}, snRNA: {n_snrna}")

    # Signatures - likely snRNA only
    if 'signatures' in data:
        sig_len = len(data['signatures'])
        print(f"Signatures length: {sig_len}")

        if sig_len == n_cells:
            # Full match
            for col in data['signatures'].columns:
                if col not in df.columns:
                    df[col] = data['signatures'][col].values
            print(f"Merged signatures to all cells: {list(data['signatures'].columns)}")
        elif sig_len == n_snrna:
            # snRNA only - merge to snrna subset
            for col in data['signatures'].columns:
                if col not in df_snrna.columns:
                    df_snrna[col] = data['signatures'][col].values
            print(f"Merged signatures to snRNA only: {list(data['signatures'].columns)}")
        else:
            print(f"WARNING: Signatures length {sig_len} doesn't match cells ({n_cells}) or snRNA ({n_snrna})")

    # Progression - likely snRNA only
    if 'progression' in data:
        prog_len = len(data['progression'])
        print(f"Progression length: {prog_len}")

        if prog_len == n_cells:
            for col in data['progression'].columns:
                if col not in df.columns:
                    df[col] = data['progression'][col].values
            print(f"Merged progression to all cells")
        elif prog_len == n_snrna:
            for col in data['progression'].columns:
                if col not in df_snrna.columns:
                    df_snrna[col] = data['progression'][col].values
            print(f"Merged progression to snRNA only")

    # Trajectories
    if 'trajectories' in data:
        traj_len = len(data['trajectories'])
        print(f"Trajectories length: {traj_len}")

        if traj_len == n_cells:
            for col in data['trajectories'].columns:
                if col not in df.columns:
                    df[col] = data['trajectories'][col].values
            print(f"Merged trajectories to all cells")
        elif traj_len == n_snrna:
            for col in data['trajectories'].columns:
                if col not in df_snrna.columns:
                    df_snrna[col] = data['trajectories'][col].values
            print(f"Merged trajectories to snRNA only")

    return df, df_snrna


# =============================================================================
# FIGURE GENERATION
# =============================================================================

def get_stage_order(df: pd.DataFrame) -> Tuple[List[str], Dict[str, str]]:
    """Detect stage schema and return order + colors."""
    stages = df['stage'].unique()
    if 'Preinvasive' in stages or 'preinvasive' in stages.astype(str):
        return STAGE_ORDER_3, STAGE_COLORS
    elif 'AAH' in stages:
        return STAGE_ORDER_5, STAGE_COLORS
    else:
        # Default
        return STAGE_ORDER_3, STAGE_COLORS


def fig1_overview(df: pd.DataFrame, umap: np.ndarray, output_dir: Path):
    """Figure 1: Data and embedding overview."""
    print("\n=== Figure 1: Overview ===")

    stage_order, colors = get_stage_order(df)

    fig = plt.figure(figsize=(16, 5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])

    # A: UMAP by stage
    ax = fig.add_subplot(gs[0])
    for stage in stage_order:
        mask = df['stage'] == stage
        n = mask.sum()
        if n > 0:
            ax.scatter(umap[mask, 0], umap[mask, 1],
                      c=colors.get(stage, 'gray'), s=0.5, alpha=0.4,
                      label=f'{stage} ({n:,})', rasterized=True)
    ax.legend(markerscale=10, frameon=False, fontsize=10)
    ax.set_title('A. Embeddings by Stage', fontsize=12, fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values(): spine.set_visible(False)

    # B: UMAP by data type
    ax = fig.add_subplot(gs[1])
    for dtype, color in [('snrna', '#9467bd'), ('spatial', '#2ca02c')]:
        mask = df['data_type'] == dtype
        n = mask.sum()
        if n > 0:
            ax.scatter(umap[mask, 0], umap[mask, 1],
                      c=color, s=0.5, alpha=0.4,
                      label=f'{dtype} ({n:,})', rasterized=True)
    ax.legend(markerscale=10, frameon=False, fontsize=10)
    ax.set_title('B. snRNA + Spatial Integration', fontsize=12, fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values(): spine.set_visible(False)

    # C: Stage counts
    ax = fig.add_subplot(gs[2])
    counts = df['stage'].value_counts().reindex(stage_order).fillna(0)
    bars = ax.bar(range(len(stage_order)), counts.values,
                  color=[colors.get(s, 'gray') for s in stage_order])
    for bar, count in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{int(count):,}', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(range(len(stage_order)))
    ax.set_xticklabels(stage_order, fontsize=10)
    ax.set_ylabel('Cells')
    ax.set_title('C. Dataset Composition', fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_dir / 'fig1_overview.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'fig1_overview.pdf', bbox_inches='tight')
    plt.close(fig)
    print("Saved: fig1_overview")


def fig2_biological_scores(df: pd.DataFrame, umap: np.ndarray, output_dir: Path):
    """Figure 2: Key biological scores on UMAP and violin."""
    print("\n=== Figure 2: Biological Scores ===")

    stage_order, colors = get_stage_order(df)

    # Key scores to show
    scores = [
        ('emt_score', 'EMT', 'RdYlBu_r'),
        ('KAC_score', 'KAC Progenitor', 'Purples'),
        ('IL1_axis_score', 'IL1 Axis', 'Reds'),
        ('cytotrace', 'CytoTRACE', 'viridis'),
    ]

    # Filter to existing
    scores = [(col, name, cmap) for col, name, cmap in scores if col in df.columns]

    if not scores:
        print("No biological score columns found")
        return

    n = len(scores)
    fig, axes = plt.subplots(2, n, figsize=(4*n, 8))

    for idx, (col, name, cmap) in enumerate(scores):
        vals = df[col].values
        valid = ~np.isnan(vals)
        vmin, vmax = np.nanpercentile(vals[valid], [2, 98])

        # Top: UMAP
        ax = axes[0, idx]
        scatter = ax.scatter(umap[:, 0], umap[:, 1], c=vals,
                            cmap=cmap, s=0.3, alpha=0.5,
                            vmin=vmin, vmax=vmax, rasterized=True)
        plt.colorbar(scatter, ax=ax, shrink=0.5)
        ax.set_title(name, fontsize=11, fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values(): spine.set_visible(False)

        # Bottom: Violin
        ax = axes[1, idx]
        data_by_stage = []
        stage_colors = []
        for stage in stage_order:
            mask = df['stage'] == stage
            v = df.loc[mask, col].dropna().values
            if len(v) > 0:
                data_by_stage.append(v)
                stage_colors.append(colors.get(stage, 'gray'))

        if data_by_stage:
            all_v = np.concatenate(data_by_stage)
            ymin, ymax = np.percentile(all_v, [1, 99])
            margin = (ymax - ymin) * 0.15

            parts = ax.violinplot(data_by_stage, showmeans=True)
            for pc, c in zip(parts['bodies'], stage_colors):
                pc.set_facecolor(c)
                pc.set_alpha(0.7)
            parts['cmeans'].set_color('black')
            parts['cmeans'].set_linewidth(2)

            ax.set_ylim(ymin - margin, ymax + margin)

        ax.set_xticks(range(1, len(stage_order)+1))
        ax.set_xticklabels(stage_order, fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_dir / 'fig2_biological_scores.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'fig2_biological_scores.pdf', bbox_inches='tight')
    plt.close(fig)
    print("Saved: fig2_biological_scores")


def fig3_progression(df: pd.DataFrame, umap: np.ndarray, output_dir: Path):
    """Figure 3: Progression analysis (cytotrace, pseudotime, diffusion)."""
    print("\n=== Figure 3: Progression ===")

    stage_order, colors = get_stage_order(df)

    scores = [
        ('cytotrace', 'CytoTRACE (Differentiation)', 'viridis'),
        ('pseudotime', 'Pseudotime', 'plasma'),
        ('dpt_pseudotime', 'Diffusion Pseudotime', 'inferno'),
    ]

    scores = [(col, name, cmap) for col, name, cmap in scores if col in df.columns]

    if not scores:
        print("No progression columns found")
        return

    n = len(scores)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4))
    if n == 1:
        axes = [axes]

    for idx, (col, name, cmap) in enumerate(scores):
        ax = axes[idx]
        vals = df[col].values
        valid = ~np.isnan(vals)
        vmin, vmax = np.nanpercentile(vals[valid], [2, 98])

        scatter = ax.scatter(umap[:, 0], umap[:, 1], c=vals,
                            cmap=cmap, s=0.5, alpha=0.5,
                            vmin=vmin, vmax=vmax, rasterized=True)
        plt.colorbar(scatter, ax=ax, shrink=0.6)
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values(): spine.set_visible(False)

    plt.tight_layout()
    fig.savefig(output_dir / 'fig3_progression.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'fig3_progression.pdf', bbox_inches='tight')
    plt.close(fig)
    print("Saved: fig3_progression")


def fig4_model_performance(metrics: pd.DataFrame, output_dir: Path):
    """Figure 4: Model comparison (StageBridge vs ablations vs baselines)."""
    print("\n=== Figure 4: Model Performance ===")

    if metrics.empty:
        print("No model metrics available")
        return

    # Order models by mean val_loss
    model_means = metrics.groupby('model')['val_loss'].mean().sort_values()
    model_order = model_means.index.tolist()

    # Summary stats
    summary = metrics.groupby('model').agg({
        'val_loss': ['mean', 'std', 'count']
    }).round(4)
    summary.columns = ['mean', 'std', 'n_runs']
    summary = summary.reindex(model_order)
    summary.to_csv(output_dir / 'model_performance.csv')
    print("\nModel Performance:")
    print(summary)

    fig, ax = plt.subplots(figsize=(10, 5))

    means = metrics.groupby('model')['val_loss'].mean().reindex(model_order)
    stds = metrics.groupby('model')['val_loss'].std().reindex(model_order)

    bar_colors = [MODEL_COLORS.get(m, '#888888') for m in model_order]

    x = range(len(model_order))
    bars = ax.bar(x, means, yerr=stds, capsize=4, color=bar_colors, alpha=0.8, edgecolor='black')

    # Highlight best
    best_idx = means.argmin()
    bars[best_idx].set_edgecolor('gold')
    bars[best_idx].set_linewidth(3)

    ax.set_xticks(x)
    ax.set_xticklabels(model_order, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Validation Loss', fontsize=12)
    ax.set_title('Model Comparison (5-fold CV x 3 seeds)', fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add delta from best
    best_loss = means.min()
    for i, (m, loss) in enumerate(means.items()):
        if i != best_idx:
            delta = ((loss - best_loss) / best_loss) * 100
            ax.text(i, loss + stds.iloc[i] + 0.001, f'+{delta:.1f}%',
                   ha='center', fontsize=8, color='gray')

    plt.tight_layout()
    fig.savefig(output_dir / 'fig4_model_performance.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'fig4_model_performance.pdf', bbox_inches='tight')
    plt.close(fig)
    print("Saved: fig4_model_performance")


def fig5_il1b_niche(data: Dict, output_dir: Path):
    """Figure 5: IL1B-macrophage niche interactions."""
    print("\n=== Figure 5: IL1B Niche ===")

    il1b = data.get('il1b')
    if il1b is None or (hasattr(il1b, 'empty') and il1b.empty):
        il1b = data.get('liana')
    if il1b is None or (hasattr(il1b, 'empty') and il1b.empty):
        print("No interaction data available")
        return

    # Filter to IL1B if using full LIANA
    if 'ligand_complex' in il1b.columns:
        il1b_filt = il1b[il1b['ligand_complex'].str.contains('IL1B', na=False)].copy()
    else:
        il1b_filt = il1b.copy()

    if len(il1b_filt) == 0:
        print("No IL1B interactions found")
        return

    # Sort by score
    score_col = 'lrscore' if 'lrscore' in il1b_filt.columns else il1b_filt.columns[-1]
    il1b_filt = il1b_filt.sort_values(score_col, ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Create labels
    if 'source' in il1b_filt.columns and 'target' in il1b_filt.columns:
        labels = il1b_filt['source'] + ' → ' + il1b_filt['target']
    else:
        labels = il1b_filt.index.astype(str)

    y_pos = range(len(il1b_filt))
    ax.barh(y_pos, il1b_filt[score_col], color='#CB4154', alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Interaction Score', fontsize=12)
    ax.set_title('IL1B Signaling from Macrophages', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add receptor info if available
    if 'receptor_complex' in il1b_filt.columns:
        for i, (_, row) in enumerate(il1b_filt.iterrows()):
            ax.text(row[score_col] + 0.01, i, row['receptor_complex'],
                   va='center', fontsize=7, style='italic', color='gray')

    plt.tight_layout()
    fig.savefig(output_dir / 'fig5_il1b_niche.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'fig5_il1b_niche.pdf', bbox_inches='tight')
    plt.close(fig)
    print("Saved: fig5_il1b_niche")


def fig6_attention_analysis(inference: Dict, df: pd.DataFrame, output_dir: Path):
    """Figure 6: What does the model attend to?"""
    print("\n=== Figure 6: Attention Analysis ===")

    if 'attention' not in inference:
        print("No attention weights available")
        return

    attn_data = inference['attention']

    # Get attention array - handle different formats
    if isinstance(attn_data, np.lib.npyio.NpzFile):
        keys = list(attn_data.keys())
        print(f"NPZ keys: {keys}")
        for k in keys:
            print(f"  {k}: shape={attn_data[k].shape}, dtype={attn_data[k].dtype}")
        # Use first key or look for specific ones
        if 'attention' in keys:
            attn = attn_data['attention']
        elif 'attn' in keys:
            attn = attn_data['attn']
        elif 'weights' in keys:
            attn = attn_data['weights']
        else:
            attn = attn_data[keys[0]]
    else:
        attn = attn_data

    print(f"Attention shape: {attn.shape}")
    print(f"Attention range: [{attn.min():.4f}, {attn.max():.4f}]")
    print(f"Attention mean: {attn.mean():.4f}")

    if attn.max() == 0:
        print("WARNING: Neighbor attention weights are all zeros!")
        print("This means the model attends to the empty token (AMICI feature)")
        print("The model learned that receiver embedding alone is sufficient.")
        print("Skipping attention figure - use displacements for dynamics instead.")
        return

    # Token labels (9-token architecture)
    if attn.shape[-1] == 9:
        token_labels = ['Receiver', 'Ring1', 'Ring2', 'Ring3', 'Ring4',
                        'HLCA', 'LuCA', 'Pathway', 'Stats']
    else:
        token_labels = [f'Token{i}' for i in range(attn.shape[-1])]

    # Average over heads if needed
    if attn.ndim == 3:
        attn = attn.mean(axis=1)

    stage_order, colors = get_stage_order(df)

    # Match lengths
    n_attn = len(attn)
    n_cells = len(df)

    if n_attn != n_cells:
        print(f"Attention length ({n_attn}) != cells ({n_cells}), using min")
        n = min(n_attn, n_cells)
        attn = attn[:n]
        df = df.iloc[:n]

    fig, ax = plt.subplots(figsize=(12, 5))

    x = np.arange(len(token_labels))
    width = 0.8 / len(stage_order)

    for i, stage in enumerate(stage_order):
        mask = df['stage'].values == stage
        if mask.sum() > 0:
            mean_attn = attn[mask].mean(axis=0)
            offset = (i - len(stage_order)/2 + 0.5) * width
            ax.bar(x + offset, mean_attn, width,
                   color=colors.get(stage, 'gray'), label=stage, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(token_labels, rotation=45, ha='right', fontsize=11)
    ax.set_ylabel('Mean Attention Weight', fontsize=12)
    ax.set_title('Attention Distribution by Stage', fontsize=14, fontweight='bold')
    ax.legend(frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_dir / 'fig6_attention.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'fig6_attention.pdf', bbox_inches='tight')
    plt.close(fig)
    print("Saved: fig6_attention")


def fig7_fold_changes(df: pd.DataFrame, output_dir: Path):
    """Figure 7: Summary of progression-associated changes."""
    print("\n=== Figure 7: Fold Changes ===")

    stage_order, _ = get_stage_order(df)

    # Columns to analyze
    candidates = [
        'emt_score', 'KAC_score', 'IL1_axis_score', 'AP1_score',
        'cytotrace', 'senescence_score', 'sasp_score',
        'caf_fraction', 'immune_fraction',
        'pathway_NFkB', 'pathway_Hypoxia', 'pathway_TGFb',
    ]

    cols = [c for c in candidates if c in df.columns]

    results = []
    baseline_stage = stage_order[0]  # Normal
    compare_stage = stage_order[-1]  # Invasive

    for col in cols:
        mean_base = df.loc[df['stage'] == baseline_stage, col].mean()
        mean_comp = df.loc[df['stage'] == compare_stage, col].mean()

        if mean_base != 0 and not np.isnan(mean_base):
            fc = mean_comp / mean_base
            log2fc = np.log2(fc) if fc > 0 else np.nan

            # Clean name
            name = col.replace('_score', '').replace('pathway_', '').replace('_', ' ').title()

            results.append({
                'feature': name,
                'baseline': mean_base,
                'comparison': mean_comp,
                'fold_change': fc,
                'log2fc': log2fc,
            })

    fc_df = pd.DataFrame(results).sort_values('log2fc')
    fc_df.to_csv(output_dir / 'fold_changes.csv', index=False)

    print(f"\nFold Changes ({compare_stage} vs {baseline_stage}):")
    for _, row in fc_df.iterrows():
        direction = 'UP' if row['log2fc'] > 0 else 'DOWN'
        print(f"  {row['feature']}: {row['fold_change']:.2f}x ({direction})")

    fig, ax = plt.subplots(figsize=(8, 6))

    y_pos = range(len(fc_df))
    bar_colors = ['#CB4154' if x > 0 else '#4169E1' for x in fc_df['log2fc']]

    ax.barh(y_pos, fc_df['log2fc'], color=bar_colors, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(fc_df['feature'], fontsize=10)
    ax.set_xlabel(f'log2 Fold Change ({compare_stage} vs {baseline_stage})', fontsize=12)
    ax.set_title('Progression-Associated Changes', fontsize=14, fontweight='bold')
    ax.axvline(0, color='black', linewidth=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add FC values
    for i, (_, row) in enumerate(fc_df.iterrows()):
        xpos = row['log2fc'] + (0.1 if row['log2fc'] > 0 else -0.1)
        ha = 'left' if row['log2fc'] > 0 else 'right'
        ax.text(xpos, i, f"{row['fold_change']:.2f}x", va='center', ha=ha, fontsize=8)

    plt.tight_layout()
    fig.savefig(output_dir / 'fig7_fold_changes.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'fig7_fold_changes.pdf', bbox_inches='tight')
    plt.close(fig)
    print("Saved: fig7_fold_changes")


def fig8_cell_cycle(df: pd.DataFrame, umap: np.ndarray, output_dir: Path):
    """Figure 8: Cell cycle analysis."""
    print("\n=== Figure 8: Cell Cycle ===")

    stage_order, colors = get_stage_order(df)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # A: UMAP by S score
    ax = axes[0]
    if 'S_score' in df.columns:
        vals = df['S_score'].values
        vmin, vmax = np.nanpercentile(vals[~np.isnan(vals)], [2, 98])
        scatter = ax.scatter(umap[:, 0], umap[:, 1], c=vals,
                            cmap='YlOrRd', s=0.3, alpha=0.5,
                            vmin=vmin, vmax=vmax, rasterized=True)
        plt.colorbar(scatter, ax=ax, shrink=0.6)
    ax.set_title('A. S Phase Score', fontsize=12, fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values(): spine.set_visible(False)

    # B: UMAP by G2M score
    ax = axes[1]
    if 'G2M_score' in df.columns:
        vals = df['G2M_score'].values
        vmin, vmax = np.nanpercentile(vals[~np.isnan(vals)], [2, 98])
        scatter = ax.scatter(umap[:, 0], umap[:, 1], c=vals,
                            cmap='YlGnBu', s=0.3, alpha=0.5,
                            vmin=vmin, vmax=vmax, rasterized=True)
        plt.colorbar(scatter, ax=ax, shrink=0.6)
    ax.set_title('B. G2/M Phase Score', fontsize=12, fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values(): spine.set_visible(False)

    # C: Violin by stage
    ax = axes[2]
    if 'S_score' in df.columns and 'G2M_score' in df.columns:
        # Combined proliferation
        df['prolif_combined'] = df['S_score'] + df['G2M_score']

        data_by_stage = []
        stage_colors = []
        for stage in stage_order:
            mask = df['stage'] == stage
            v = df.loc[mask, 'prolif_combined'].dropna().values
            if len(v) > 0:
                data_by_stage.append(v)
                stage_colors.append(colors.get(stage, 'gray'))

        if data_by_stage:
            all_v = np.concatenate(data_by_stage)
            ymin, ymax = np.percentile(all_v, [1, 99])
            margin = (ymax - ymin) * 0.15

            parts = ax.violinplot(data_by_stage, showmeans=True)
            for pc, c in zip(parts['bodies'], stage_colors):
                pc.set_facecolor(c)
                pc.set_alpha(0.7)
            parts['cmeans'].set_color('black')
            parts['cmeans'].set_linewidth(2)

            ax.set_ylim(ymin - margin, ymax + margin)

        ax.set_xticks(range(1, len(stage_order)+1))
        ax.set_xticklabels(stage_order, fontsize=10)
        ax.set_ylabel('S + G2M Score')
    ax.set_title('C. Proliferation by Stage', fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_dir / 'fig8_cell_cycle.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'fig8_cell_cycle.pdf', bbox_inches='tight')
    plt.close(fig)
    print("Saved: fig8_cell_cycle")


def compute_ot_from_displacements(
    displacements: dict,
    df: pd.DataFrame,
    umap: np.ndarray,
    grid_size: int = 50
) -> pd.DataFrame:
    """Compute flux/curl/divergence from model displacement vectors."""
    from scipy.interpolate import griddata, RegularGridInterpolator
    from scipy.ndimage import sobel

    n_cells = len(df)
    vel_2d = np.zeros((n_cells, 2))
    has_vel = np.zeros(n_cells, dtype=bool)

    # Get cell IDs
    if 'cell_id' in df.columns:
        cell_ids = df['cell_id'].values
    else:
        cell_ids = df.index.values

    # Match displacements to cells - simple projection using first 2 dims
    for i, cid in enumerate(cell_ids):
        if cid in displacements:
            vel = displacements[cid]
            has_vel[i] = True
            # Use magnitude to scale a random direction in UMAP space
            # or just take first 2 dims as rough projection
            vel_mag = np.linalg.norm(vel)
            if vel_mag > 0:
                vel_2d[i] = vel[:2] * vel_mag / (np.linalg.norm(vel[:2]) + 1e-8)

    print(f"  Cells with velocity: {has_vel.sum()} / {n_cells}")
    if has_vel.sum() < 100:
        return None

    # Compute flux (magnitude)
    flux = np.linalg.norm(vel_2d, axis=1)

    # Interpolate to grid for divergence/curl
    umap_sub = umap[has_vel]
    vel_sub = vel_2d[has_vel]

    x_min, x_max = umap[:, 0].min(), umap[:, 0].max()
    y_min, y_max = umap[:, 1].min(), umap[:, 1].max()

    xi = np.linspace(x_min, x_max, grid_size)
    yi = np.linspace(y_min, y_max, grid_size)
    xi_grid, yi_grid = np.meshgrid(xi, yi)

    vx_grid = griddata(umap_sub, vel_sub[:, 0], (xi_grid, yi_grid), method='linear')
    vy_grid = griddata(umap_sub, vel_sub[:, 1], (xi_grid, yi_grid), method='linear')

    # Fill NaN with nearest
    vx_nearest = griddata(umap_sub, vel_sub[:, 0], (xi_grid, yi_grid), method='nearest')
    vy_nearest = griddata(umap_sub, vel_sub[:, 1], (xi_grid, yi_grid), method='nearest')
    vx_grid = np.where(np.isnan(vx_grid), vx_nearest, vx_grid)
    vy_grid = np.where(np.isnan(vy_grid), vy_nearest, vy_grid)

    dx = xi[1] - xi[0]
    dy = yi[1] - yi[0]

    # Divergence and curl via Sobel
    dvx_dx = sobel(vx_grid, axis=1) / (8 * dx)
    dvy_dy = sobel(vy_grid, axis=0) / (8 * dy)
    dvx_dy = sobel(vx_grid, axis=0) / (8 * dy)
    dvy_dx = sobel(vy_grid, axis=1) / (8 * dx)

    div_grid = dvx_dx + dvy_dy
    curl_grid = dvy_dx - dvx_dy

    # Interpolate back to cells
    interp_div = RegularGridInterpolator((yi, xi), div_grid, method='linear',
                                         bounds_error=False, fill_value=np.nan)
    interp_curl = RegularGridInterpolator((yi, xi), curl_grid, method='linear',
                                          bounds_error=False, fill_value=np.nan)

    divergence = interp_div(umap[:, ::-1])
    curl = interp_curl(umap[:, ::-1])

    return pd.DataFrame({
        'flux': flux,
        'divergence': divergence,
        'curl': curl,
        'velocity_x': vel_2d[:, 0],
        'velocity_y': vel_2d[:, 1],
    })


def fig9_ot_dynamics(data: Dict, df: pd.DataFrame, umap: np.ndarray, output_dir: Path):
    """Figure 9: OT dynamics - flux, curl, divergence."""
    print("\n=== Figure 9: OT Dynamics ===")

    # Check for OT data - could be in various places
    ot_cols = ['flux', 'curl', 'divergence', 'velocity_x', 'velocity_y',
               'flow_magnitude', 'irreversibility']

    found_cols = [c for c in ot_cols if c in df.columns]

    # Also check for separate OT file
    if 'ot_dynamics' in data:
        ot_df = data['ot_dynamics']
        for c in ot_cols:
            if c in ot_df.columns and c not in df.columns:
                if len(ot_df) == len(df):
                    df[c] = ot_df[c].values
                    found_cols.append(c)

    # Compute from model displacements if available
    if not found_cols and 'inference' in data:
        inference = data['inference']
        if 'all_displacements' in inference and len(inference['all_displacements']) > 0:
            print("Computing OT dynamics from model displacements...")
            ot_df = compute_ot_from_displacements(inference['all_displacements'], df, umap)
            if ot_df is not None:
                for c in ot_df.columns:
                    df[c] = ot_df[c].values
                found_cols = list(ot_df.columns)

    if not found_cols:
        print("No OT dynamics columns found")
        print("Expected columns like: flux, curl, divergence, velocity_x/y")
        print(f"Available columns: {[c for c in df.columns if 'vel' in c.lower() or 'flux' in c.lower() or 'div' in c.lower() or 'curl' in c.lower()]}")
        return

    n_cols = min(len(found_cols), 4)
    fig, axes = plt.subplots(1, n_cols, figsize=(5*n_cols, 4.5))
    if n_cols == 1:
        axes = [axes]

    cmaps = {'flux': 'coolwarm', 'curl': 'PiYG', 'divergence': 'RdBu_r',
             'flow_magnitude': 'viridis', 'irreversibility': 'magma'}

    for idx, col in enumerate(found_cols[:n_cols]):
        ax = axes[idx]
        vals = df[col].values
        valid = ~np.isnan(vals)

        if valid.sum() == 0:
            ax.text(0.5, 0.5, f'{col}\n(no data)', ha='center', va='center', transform=ax.transAxes)
            continue

        vmin, vmax = np.nanpercentile(vals[valid], [2, 98])

        # Symmetric colormap for curl/divergence
        if col in ['curl', 'divergence', 'flux']:
            vabs = max(abs(vmin), abs(vmax))
            vmin, vmax = -vabs, vabs

        cmap = cmaps.get(col, 'viridis')
        scatter = ax.scatter(umap[:, 0], umap[:, 1], c=vals,
                            cmap=cmap, s=0.3, alpha=0.5,
                            vmin=vmin, vmax=vmax, rasterized=True)
        plt.colorbar(scatter, ax=ax, shrink=0.6)

        ax.set_title(col.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values(): spine.set_visible(False)

    plt.tight_layout()
    fig.savefig(output_dir / 'fig9_ot_dynamics.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'fig9_ot_dynamics.pdf', bbox_inches='tight')
    plt.close(fig)
    print("Saved: fig9_ot_dynamics")


def fig10_spatial_showcase(df: pd.DataFrame, output_dir: Path, n_examples: int = 3):
    """Figure 8: Spatial data examples."""
    print("\n=== Figure 8: Spatial Showcase ===")

    spatial = df[df['data_type'] == 'spatial'].copy()
    if len(spatial) == 0:
        print("No spatial data")
        return

    # Pick top donors by spot count
    donors = spatial['donor_id'].value_counts().head(n_examples).index.tolist()

    fig, axes = plt.subplots(2, n_examples, figsize=(5*n_examples, 9))

    for idx, donor in enumerate(donors):
        section = spatial[spatial['donor_id'] == donor]

        # Top: by cell type
        ax = axes[0, idx]
        cell_types = section['cell_type'].value_counts().head(6).index
        cmap = plt.cm.get_cmap('tab10', len(cell_types))

        for i, ct in enumerate(cell_types):
            mask = section['cell_type'] == ct
            if mask.sum() > 0:
                ax.scatter(section.loc[mask, 'x'], section.loc[mask, 'y'],
                          c=[cmap(i)], s=1, alpha=0.7, label=ct[:15], rasterized=True)

        ax.legend(markerscale=5, frameon=False, fontsize=7, loc='upper right')
        ax.set_title(f'{donor} - Cell Types', fontsize=10, fontweight='bold')
        ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])
        ax.invert_yaxis()

        # Bottom: by EMT or other score
        ax = axes[1, idx]
        score_col = 'emt_score' if 'emt_score' in section.columns else None

        if score_col:
            vals = section[score_col].values
            valid = ~np.isnan(vals)
            vmin, vmax = np.nanpercentile(vals[valid], [5, 95]) if valid.sum() > 0 else (0, 1)

            scatter = ax.scatter(section['x'], section['y'], c=vals,
                                cmap='RdYlBu_r', s=1, alpha=0.7,
                                vmin=vmin, vmax=vmax, rasterized=True)
            plt.colorbar(scatter, ax=ax, shrink=0.5, label='EMT')

        ax.set_title(f'{donor} - EMT Score', fontsize=10, fontweight='bold')
        ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])
        ax.invert_yaxis()

    plt.tight_layout()
    fig.savefig(output_dir / 'fig8_spatial.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'fig8_spatial.pdf', bbox_inches='tight')
    plt.close(fig)
    print("Saved: fig8_spatial")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate complete poster figures')
    parser.add_argument('--data-dir', type=Path, required=True,
                       help='Canonical data directory')
    parser.add_argument('--model-dir', type=Path, required=True,
                       help='Model outputs directory (v1.1)')
    parser.add_argument('--output-dir', type=Path, required=True,
                       help='Output directory for figures')

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("STAGEBRIDGE FULL POSTER GENERATOR")
    print("="*60)
    print(f"Data: {args.data_dir}")
    print(f"Models: {args.model_dir}")
    print(f"Output: {args.output_dir}")

    # Load all data
    data = load_all_data(args.data_dir, args.model_dir)

    if 'cells' not in data:
        print("ERROR: cells.parquet not found")
        return

    # Merge all scores into one dataframe
    print("\n" + "="*60)
    print("MERGING DATA")
    print("="*60)
    df_all, df_snrna = merge_all_scores(data)
    print(f"All cells: {df_all.shape}")
    print(f"snRNA cells: {df_snrna.shape}")
    print(f"snRNA score columns: {[c for c in df_snrna.columns if 'score' in c.lower() or 'KAC' in c or 'IL1' in c or 'AP1' in c]}")

    # Get UMAP - this is likely snRNA only
    if 'umap' in data:
        umap = data['umap']
        print(f"Pre-computed UMAP: {umap.shape}")
    else:
        print("Computing UMAP from z_fused (snRNA)...")
        z_fused = np.stack(df_snrna['z_fused'].values)
        import umap as umap_lib
        reducer = umap_lib.UMAP(n_neighbors=30, min_dist=0.3, random_state=42, n_jobs=-1)
        umap = reducer.fit_transform(z_fused)
        np.save(args.output_dir / 'umap_computed.npy', umap)

    # Use snRNA dataframe for UMAP-based figures
    df = df_snrna
    if len(umap) != len(df):
        print(f"UMAP length ({len(umap)}) != snRNA ({len(df)})")
        # If UMAP matches snRNA, good. Otherwise truncate
        n = min(len(umap), len(df))
        umap = umap[:n]
        df = df.iloc[:n].reset_index(drop=True)

    # Generate figures
    print("\n" + "="*60)
    print("GENERATING FIGURES")
    print("="*60)

    fig1_overview(df, umap, args.output_dir)
    fig2_biological_scores(df, umap, args.output_dir)
    fig3_progression(df, umap, args.output_dir)

    if 'model_metrics' in data and not data['model_metrics'].empty:
        fig4_model_performance(data['model_metrics'], args.output_dir)

    fig5_il1b_niche(data, args.output_dir)

    if 'inference' in data:
        fig6_attention_analysis(data['inference'], df, args.output_dir)

    fig7_fold_changes(df, args.output_dir)
    fig8_cell_cycle(df, umap, args.output_dir)
    fig9_ot_dynamics(data, df, umap, args.output_dir)
    fig10_spatial_showcase(df_all, args.output_dir)  # Use all cells for spatial

    print("\n" + "="*60)
    print(f"COMPLETE - Figures saved to: {args.output_dir}")
    print("="*60)

    # List output files
    print("\nGenerated files:")
    for f in sorted(args.output_dir.glob('*.png')):
        print(f"  {f.name}")


if __name__ == '__main__':
    main()
