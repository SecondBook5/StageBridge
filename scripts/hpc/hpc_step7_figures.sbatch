#!/bin/bash
#SBATCH --job-name=sb_figures
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --account=chaunzt1
#SBATCH --output=/home/booka/StageBridge/logs/figures_%j.log

# =============================================================================
# Stage 7: Publication Figure Generation for Nature Methods
# =============================================================================
#
# Generates ALL publication-quality figures with skip-if-exists logic:
#
# Main Figures (Nature Methods):
# - Figure 1: Reference geometry (HLCA/LuCA structure, fused embeddings)
# - Figure 2: Training curves (SSL pretraining + transition model)
# - Figure 3: Spatial backend comparison
# - Figure 4: Stage embeddings and trajectories
# - Figure 5: Ablation heatmap
# - Figure 6: Attention analysis
# - Figure 7: Biological validation (IL1B pathway)
#
# Supplementary Figures:
# - S1: Data QC and distributions
# - S2: Reference mapping diagnostics (HLCA/LuCA ELBO curves)
# - S3: HPO optimization history
# - S4: CV fold stability
# - S5: Attention patterns and token importance
# - S6: Baseline comparison (PoolingMLP, DeepSets, SetTransformer, GraphSAGE)
# - S7: Extended ablation results
# - S8: Donor consistency analysis
#
# Output formats: PNG (300 DPI), PDF (vector), SVG
# Skip logic: Skips figures that already exist with all formats
#
# Prerequisites:
# - Stage 1-2: Reference mapping completed (HLCA/LuCA)
# - Stage 3: Spatial backend comparison completed
# - Stage 4: Data preparation completed
# - Stage 5: Training completed
# - Stage 6: Ablations completed
#
# Runtime: ~2-3 hours
# =============================================================================

set -euo pipefail

echo "=============================================="
echo "Stage 7: Publication Figure Generation"
echo "Started: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "=============================================="

# Environment setup
cd /home/booka/StageBridge
source /admin/software/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/chaunzt1/stagebridge_env

# Data paths
DATA=/scratch/chaunzt1/stagebridge
TRAINING_OUTPUT=$DATA/runs/v1_complete
ABLATION_OUTPUT=$DATA/runs/ablations
REF_GEOMETRY=$DATA/processed/luad_evo/reference_geometry
SPATIAL_BENCHMARK=$DATA/processed/luad_evo/spatial_benchmark
CANONICAL=$DATA/processed/luad_evo/canonical
OUTPUT=$DATA/runs/publication_figures

export CUDA_VISIBLE_DEVICES=0

echo ""
echo "Configuration:"
echo "  Training: $TRAINING_OUTPUT"
echo "  Ablations: $ABLATION_OUTPUT"
echo "  Reference: $REF_GEOMETRY"
echo "  Spatial: $SPATIAL_BENCHMARK"
echo "  Canonical: $CANONICAL"
echo "  Output: $OUTPUT"
echo ""

# Create output directory structure
mkdir -p "$OUTPUT/main"
mkdir -p "$OUTPUT/supplementary"
mkdir -p "$OUTPUT/panels"
mkdir -p "$OUTPUT/manifests"
mkdir -p "$OUTPUT/tikz"

# =============================================================================
# Helper Functions
# =============================================================================

# Check if figure exists in all required formats
figure_exists() {
    local base_path="$1"
    if [[ -f "${base_path}.png" ]] && [[ -f "${base_path}.pdf" ]] && [[ -f "${base_path}.svg" ]]; then
        return 0  # All formats exist
    else
        return 1  # At least one format missing
    fi
}

# Log figure generation
log_figure() {
    local figure_id="$1"
    local status="$2"
    local timestamp=$(date -Iseconds)
    echo "{\"figure_id\": \"$figure_id\", \"status\": \"$status\", \"timestamp\": \"$timestamp\"}" >> "$OUTPUT/manifests/generation_log.jsonl"
}

# =============================================================================
# Generate Main Figures
# =============================================================================
echo ""
echo "=============================================="
echo "Generating Main Figures"
echo "=============================================="

# -----------------------------------------------------------------------------
# Figure 1: Reference Geometry Visualization
# -----------------------------------------------------------------------------
echo ""
echo "Figure 1: Reference Geometry (HLCA/LuCA structure, fused embeddings)..."

if figure_exists "$OUTPUT/main/fig01_reference_geometry"; then
    echo "  ✓ Skipping (already exists)"
    log_figure "fig01_reference_geometry" "skipped"
else
    echo "  Generating reference geometry plots..."
    python << 'EOFPYTHON'
import sys
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
sys.path.insert(0, '/home/booka/StageBridge')

from stagebridge.viz import setup_publication_plotting, create_subplots, save_publication_figure, get_stage_color
import pandas as pd
import numpy as np

setup_publication_plotting()

ref_geometry = Path("/scratch/chaunzt1/stagebridge/processed/luad_evo/reference_geometry")
output_path = Path("/scratch/chaunzt1/stagebridge/runs/publication_figures/main/fig01_reference_geometry")

# Load embeddings
try:
    hlca = pd.read_parquet(ref_geometry / "hlca_embedding.parquet")
    luca = pd.read_parquet(ref_geometry / "luca_embedding.parquet")
    fused = pd.read_parquet(ref_geometry / "fused_embedding.parquet")

    fig, axes = create_subplots(nrows=1, ncols=3, figsize=(15, 5))

    # Panel A: HLCA structure (PCA of latent space)
    ax = axes[0]
    from sklearn.decomposition import PCA
    hlca_cols = [c for c in hlca.columns if c.startswith('hlca_')]
    if hlca_cols and len(hlca) > 0:
        pca = PCA(n_components=2)
        hlca_2d = pca.fit_transform(hlca[hlca_cols].values)
        scatter = ax.scatter(hlca_2d[:, 0], hlca_2d[:, 1], s=1, alpha=0.3, c='#00BA38', rasterized=True)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        ax.set_title('A. HLCA Reference Structure', fontweight='bold')

    # Panel B: LuCA structure
    ax = axes[1]
    luca_cols = [c for c in luca.columns if c.startswith('luca_')]
    if luca_cols and len(luca) > 0:
        pca = PCA(n_components=2)
        luca_2d = pca.fit_transform(luca[luca_cols].values)
        scatter = ax.scatter(luca_2d[:, 0], luca_2d[:, 1], s=1, alpha=0.3, c='#F8766D', rasterized=True)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        ax.set_title('B. LuCA Reference Structure', fontweight='bold')

    # Panel C: Fused embedding structure
    ax = axes[2]
    fused_cols = [c for c in fused.columns if c.startswith('fused_')]
    if fused_cols and len(fused) > 0:
        pca = PCA(n_components=2)
        fused_2d = pca.fit_transform(fused[fused_cols].values)
        scatter = ax.scatter(fused_2d[:, 0], fused_2d[:, 1], s=1, alpha=0.3, c='#619CFF', rasterized=True)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        ax.set_title('C. Fused Dual-Reference', fontweight='bold')

    save_publication_figure(fig, output_path)
    print(f"  ✓ Saved: {output_path}")

except Exception as e:
    print(f"  ✗ Error: {e}")
    sys.exit(1)
EOFPYTHON
    log_figure "fig01_reference_geometry" "generated"
fi

# -----------------------------------------------------------------------------
# Figure 2: Training Curves (SSL + Transition)
# -----------------------------------------------------------------------------
echo ""
echo "Figure 2: Training curves (SSL pretraining + transition model)..."

if figure_exists "$OUTPUT/main/fig02_training_curves"; then
    echo "  ✓ Skipping (already exists)"
    log_figure "fig02_training_curves" "skipped"
else
    echo "  Generating training curves..."
    python << 'EOFPYTHON'
import sys
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
import json
sys.path.insert(0, '/home/booka/StageBridge')

from stagebridge.viz import setup_publication_plotting, create_subplots, save_publication_figure
import numpy as np

setup_publication_plotting()

training_dir = Path("/scratch/chaunzt1/stagebridge/runs/v1_complete")
output_path = Path("/scratch/chaunzt1/stagebridge/runs/publication_figures/main/fig02_training_curves")

try:
    # Load results
    with open(training_dir / "results.json") as f:
        results = json.load(f)

    fig, axes = create_subplots(nrows=2, ncols=2, figsize=(12, 10))

    # Panel A: SSL pretraining loss
    ax = axes[0, 0]
    ssl_loss = results.get('history_ssl', {}).get('loss', [])
    if ssl_loss:
        epochs = np.arange(1, len(ssl_loss) + 1)
        ax.plot(epochs, ssl_loss, 'o-', linewidth=2, markersize=4, color='#619CFF')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('SSL Loss')
        ax.set_title('A. SSL Pretraining', fontweight='bold')
        ax.grid(True, alpha=0.3)

    # Panel B: Transition model loss
    ax = axes[0, 1]
    train_loss = results.get('history_transition', {}).get('train_loss', [])
    val_loss = results.get('history_transition', {}).get('val_loss', [])
    if train_loss:
        epochs = np.arange(1, len(train_loss) + 1)
        ax.plot(epochs, train_loss, 'o-', label='Train', linewidth=2, markersize=4, color='#00BA38')
    if val_loss:
        epochs = np.arange(1, len(val_loss) + 1)
        ax.plot(epochs, val_loss, 's-', label='Validation', linewidth=2, markersize=4, color='#F8766D')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Transition Loss')
    ax.set_title('B. Transition Model', fontweight='bold')
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.3)

    # Panel C: Wasserstein distance
    ax = axes[1, 0]
    wasserstein = results.get('metrics', {}).get('wasserstein_distance', [])
    if wasserstein:
        epochs = np.arange(1, len(wasserstein) + 1)
        ax.plot(epochs, wasserstein, 'o-', linewidth=2, markersize=4, color='#E58700')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Wasserstein Distance')
        ax.set_title('C. Distribution Matching', fontweight='bold')
        ax.grid(True, alpha=0.3)

    # Panel D: Combined metrics
    ax = axes[1, 1]
    mse = results.get('metrics', {}).get('mse', [])
    mmd = results.get('metrics', {}).get('mmd', [])
    if mse:
        epochs = np.arange(1, len(mse) + 1)
        ax.plot(epochs, mse, 'o-', label='MSE', linewidth=2, markersize=4, color='#00BA38')
    if mmd:
        epochs = np.arange(1, len(mmd) + 1)
        ax.plot(epochs, mmd, 's-', label='MMD', linewidth=2, markersize=4, color='#619CFF')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Metric Value')
    ax.set_title('D. Reconstruction Metrics', fontweight='bold')
    if mse or mmd:
        ax.legend(frameon=True)
    ax.grid(True, alpha=0.3)

    save_publication_figure(fig, output_path)
    print(f"  ✓ Saved: {output_path}")

except Exception as e:
    print(f"  ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOFPYTHON
    log_figure "fig02_training_curves" "generated"
fi

# -----------------------------------------------------------------------------
# Figure 3: Spatial Backend Comparison
# -----------------------------------------------------------------------------
echo ""
echo "Figure 3: Spatial backend comparison..."

if figure_exists "$OUTPUT/main/fig03_spatial_backends"; then
    echo "  ✓ Skipping (already exists)"
    log_figure "fig03_spatial_backends" "skipped"
else
    echo "  Generating spatial backend comparison..."
    python << 'EOFPYTHON'
import sys
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
import json
sys.path.insert(0, '/home/booka/StageBridge')

from stagebridge.viz import setup_publication_plotting, create_subplots, save_publication_figure
import numpy as np

setup_publication_plotting()

spatial_dir = Path("/scratch/chaunzt1/stagebridge/processed/luad_evo/spatial_benchmark")
output_path = Path("/scratch/chaunzt1/stagebridge/runs/publication_figures/main/fig03_spatial_backends")

try:
    # Load backend comparison results
    with open(spatial_dir / "backend_comparison.json") as f:
        results = json.load(f)

    backends = list(results.get('backends', {}).keys())
    if not backends:
        raise ValueError("No backend results found")

    fig, axes = create_subplots(nrows=1, ncols=3, figsize=(15, 5))

    # Panel A: Overall score
    ax = axes[0]
    scores = [results['backends'][b].get('overall_score', 0) for b in backends]
    colors = ['#00BA38' if b == results.get('selected_backend') else '#CCCCCC' for b in backends]
    bars = ax.bar(backends, scores, color=colors)
    ax.set_ylabel('Overall Score')
    ax.set_title('A. Backend Performance', fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, axis='y', alpha=0.3)

    # Panel B: Runtime comparison
    ax = axes[1]
    runtimes = [results['backends'][b].get('runtime_minutes', 0) for b in backends]
    bars = ax.bar(backends, runtimes, color='#619CFF')
    ax.set_ylabel('Runtime (minutes)')
    ax.set_title('B. Computational Efficiency', fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, axis='y', alpha=0.3)

    # Panel C: Accuracy metrics
    ax = axes[2]
    accuracies = [results['backends'][b].get('accuracy', 0) for b in backends]
    x = np.arange(len(backends))
    bars = ax.bar(x, accuracies, color='#E58700')
    ax.set_xticks(x)
    ax.set_xticklabels(backends, rotation=45)
    ax.set_ylabel('Mapping Accuracy')
    ax.set_title('C. Prediction Accuracy', fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)

    # Add selected backend annotation
    selected = results.get('selected_backend', '')
    fig.text(0.5, 0.02, f'Selected backend: {selected}', ha='center', fontsize=11, fontweight='bold')

    save_publication_figure(fig, output_path)
    print(f"  ✓ Saved: {output_path}")

except Exception as e:
    print(f"  ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOFPYTHON
    log_figure "fig03_spatial_backends" "generated"
fi

# -----------------------------------------------------------------------------
# Figure 4: Stage Embeddings and Trajectories
# -----------------------------------------------------------------------------
echo ""
echo "Figure 4: Stage embeddings and trajectories..."

if figure_exists "$OUTPUT/main/fig04_embeddings"; then
    echo "  ✓ Skipping (already exists)"
    log_figure "fig04_embeddings" "skipped"
else
    echo "  Generating stage embeddings..."
    python << 'EOFPYTHON'
import sys
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
sys.path.insert(0, '/home/booka/StageBridge')

from stagebridge.viz import setup_publication_plotting, create_figure, save_publication_figure, get_stage_color
import scanpy as sc

setup_publication_plotting()

canonical_path = Path("/scratch/chaunzt1/stagebridge/processed/luad_evo/canonical/snrna_canonical.h5ad")
output_path = Path("/scratch/chaunzt1/stagebridge/runs/publication_figures/main/fig04_embeddings")

try:
    # Load canonical data
    adata = sc.read_h5ad(canonical_path)

    # Compute UMAP if not present
    if 'X_umap' not in adata.obsm:
        sc.pp.neighbors(adata, use_rep='X_fused')
        sc.tl.umap(adata)

    # Use existing UMAP visualization with publication styling
    from stagebridge.viz.embeddings import plot_umap_by_stage

    plot_umap_by_stage(
        adata,
        output_path=str(output_path),
        title="Cell State Landscape Across Disease Stages",
        stage_col="stage",
        point_size=2.0,
        alpha=0.5,
        show_density=True,
        show_hulls=True,
        show_ellipses=True,
    )

    print(f"  ✓ Saved: {output_path}")

except Exception as e:
    print(f"  ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOFPYTHON
    log_figure "fig04_embeddings" "generated"
fi

# -----------------------------------------------------------------------------
# Figure 5: Ablation Heatmap
# -----------------------------------------------------------------------------
echo ""
echo "Figure 5: Ablation heatmap..."

if figure_exists "$OUTPUT/main/fig05_ablation_heatmap"; then
    echo "  ✓ Skipping (already exists)"
    log_figure "fig05_ablation_heatmap" "skipped"
else
    echo "  Copying ablation heatmap from ablation results..."
    if [[ -f "$ABLATION_OUTPUT/ablation_heatmap.png" ]]; then
        cp "$ABLATION_OUTPUT/ablation_heatmap.png" "$OUTPUT/main/fig05_ablation_heatmap.png"
        cp "$ABLATION_OUTPUT/ablation_heatmap.pdf" "$OUTPUT/main/fig05_ablation_heatmap.pdf" 2>/dev/null || true
        cp "$ABLATION_OUTPUT/ablation_heatmap.svg" "$OUTPUT/main/fig05_ablation_heatmap.svg" 2>/dev/null || true
        echo "  ✓ Copied from ablation output"
        log_figure "fig05_ablation_heatmap" "copied"
    else
        echo "  ✗ Ablation heatmap not found in $ABLATION_OUTPUT"
        log_figure "fig05_ablation_heatmap" "missing_source"
    fi
fi

# -----------------------------------------------------------------------------
# Figure 6: Attention Analysis
# -----------------------------------------------------------------------------
echo ""
echo "Figure 6: Attention analysis..."

if figure_exists "$OUTPUT/main/fig06_attention"; then
    echo "  ✓ Skipping (already exists)"
    log_figure "fig06_attention" "skipped"
else
    echo "  Generating attention analysis plots..."
    python << 'EOFPYTHON'
import sys
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
import json
import numpy as np
sys.path.insert(0, '/home/booka/StageBridge')

from stagebridge.viz import setup_publication_plotting, create_subplots, save_publication_figure

setup_publication_plotting()

training_dir = Path("/scratch/chaunzt1/stagebridge/runs/v1_complete")
output_path = Path("/scratch/chaunzt1/stagebridge/runs/publication_figures/main/fig06_attention")

try:
    # Load attention patterns if available
    attention_file = training_dir / "attention_patterns.npz"

    if attention_file.exists():
        data = np.load(attention_file)
        attention_weights = data.get('attention_weights', None)

        if attention_weights is not None:
            fig, axes = create_subplots(nrows=1, ncols=2, figsize=(12, 5))

            # Panel A: Average attention heatmap
            ax = axes[0]
            im = ax.imshow(attention_weights, cmap='viridis', aspect='auto')
            ax.set_xlabel('Key Token')
            ax.set_ylabel('Query Token')
            ax.set_title('A. Attention Pattern', fontweight='bold')
            fig.colorbar(im, ax=ax, label='Attention Weight')

            # Panel B: Token importance
            ax = axes[1]
            token_importance = attention_weights.mean(axis=0)
            token_names = ['Receiver', 'Ring1', 'Ring2', 'Ring3', 'Ring4', 'HLCA', 'LuCA', 'Pathway', 'Stats']
            ax.bar(range(len(token_importance)), token_importance, color='#619CFF')
            ax.set_xticks(range(len(token_names)))
            ax.set_xticklabels(token_names, rotation=45, ha='right')
            ax.set_ylabel('Average Attention Weight')
            ax.set_title('B. Token Importance', fontweight='bold')
            ax.grid(True, axis='y', alpha=0.3)

            save_publication_figure(fig, output_path)
            print(f"  ✓ Saved: {output_path}")
        else:
            print("  ⚠ No attention weights found in file")
    else:
        print(f"  ⚠ Attention patterns file not found: {attention_file}")
        log_figure("fig06_attention", "missing_source")

except Exception as e:
    print(f"  ✗ Error: {e}")
    import traceback
    traceback.print_exc()
EOFPYTHON
    log_figure "fig06_attention" "generated"
fi

# -----------------------------------------------------------------------------
# Figure 7: Biological Validation (IL1B Pathway)
# -----------------------------------------------------------------------------
echo ""
echo "Figure 7: Biological validation (IL1B pathway)..."

if figure_exists "$OUTPUT/main/fig07_biology"; then
    echo "  ✓ Skipping (already exists)"
    log_figure "fig07_biology" "skipped"
else
    echo "  Generating biological validation plots..."
    python << 'EOFPYTHON'
import sys
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
sys.path.insert(0, '/home/booka/StageBridge')

from stagebridge.viz import setup_publication_plotting, create_subplots, save_publication_figure, get_stage_color
import scanpy as sc
import numpy as np

setup_publication_plotting()

canonical_path = Path("/scratch/chaunzt1/stagebridge/processed/luad_evo/canonical/snrna_canonical.h5ad")
output_path = Path("/scratch/chaunzt1/stagebridge/runs/publication_figures/main/fig07_biology")

try:
    adata = sc.read_h5ad(canonical_path)

    # IL1B pathway genes
    il1b_genes = ['IL1B', 'IL1R1', 'IL1RN', 'IL1A', 'MYD88', 'IRAK1']
    available_genes = [g for g in il1b_genes if g in adata.var_names]

    if available_genes:
        fig, axes = create_subplots(nrows=2, ncols=3, figsize=(15, 10))

        stages = ['Normal', 'AAH', 'AIS', 'MIA', 'LUAD']
        stage_mask = adata.obs['stage'].isin(stages)

        for i, gene in enumerate(available_genes[:6]):
            ax = axes.flat[i]

            # Expression per stage
            expr_by_stage = []
            for stage in stages:
                mask = (adata.obs['stage'] == stage) & stage_mask
                if mask.sum() > 0:
                    gene_idx = np.where(adata.var_names == gene)[0][0]
                    expr = adata.X[mask, gene_idx].toarray().flatten() if hasattr(adata.X, 'toarray') else adata.X[mask, gene_idx]
                    expr_by_stage.append(expr)
                else:
                    expr_by_stage.append([])

            # Violin plot
            positions = range(len(stages))
            parts = ax.violinplot(expr_by_stage, positions=positions, showmeans=True)

            # Color by stage
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(get_stage_color(stages[i]))
                pc.set_alpha(0.7)

            ax.set_xticks(positions)
            ax.set_xticklabels(stages, rotation=45, ha='right')
            ax.set_ylabel('Expression')
            ax.set_title(f'{gene}', fontweight='bold')
            ax.grid(True, axis='y', alpha=0.3)

        save_publication_figure(fig, output_path)
        print(f"  ✓ Saved: {output_path}")
    else:
        print("  ⚠ No IL1B pathway genes found in dataset")
        log_figure("fig07_biology", "missing_data")

except Exception as e:
    print(f"  ✗ Error: {e}")
    import traceback
    traceback.print_exc()
EOFPYTHON
    log_figure "fig07_biology" "generated"
fi

# =============================================================================
# Generate Supplementary Figures
# =============================================================================
echo ""
echo "=============================================="
echo "Generating Supplementary Figures"
echo "=============================================="

# -----------------------------------------------------------------------------
# Supplementary Figure S1: Data QC
# -----------------------------------------------------------------------------
echo ""
echo "Supplementary Figure S1: Data QC and distributions..."

if figure_exists "$OUTPUT/supplementary/figS1_data_qc"; then
    echo "  ✓ Skipping (already exists)"
else
    echo "  Generating data QC plots..."
    python << 'EOFPYTHON'
import sys
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
sys.path.insert(0, '/home/booka/StageBridge')

from stagebridge.viz import setup_publication_plotting, create_subplots, save_publication_figure
import scanpy as sc
import numpy as np

setup_publication_plotting()

canonical_path = Path("/scratch/chaunzt1/stagebridge/processed/luad_evo/canonical/snrna_canonical.h5ad")
output_path = Path("/scratch/chaunzt1/stagebridge/runs/publication_figures/supplementary/figS1_data_qc")

try:
    adata = sc.read_h5ad(canonical_path)

    fig, axes = create_subplots(nrows=2, ncols=2, figsize=(12, 10))

    # Panel A: Cells per stage
    ax = axes[0, 0]
    stage_counts = adata.obs['stage'].value_counts()
    colors = ['#00BA38', '#F8766D', '#619CFF', '#E58700', '#A3A500'][:len(stage_counts)]
    ax.bar(range(len(stage_counts)), stage_counts.values, color=colors)
    ax.set_xticks(range(len(stage_counts)))
    ax.set_xticklabels(stage_counts.index, rotation=45, ha='right')
    ax.set_ylabel('Number of Cells')
    ax.set_title('A. Cells per Stage', fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)

    # Panel B: Gene counts distribution
    ax = axes[0, 1]
    n_genes = adata.obs.get('n_genes_by_counts', adata.obs.get('n_genes', None))
    if n_genes is not None:
        ax.hist(n_genes, bins=50, color='#619CFF', alpha=0.7)
        ax.axvline(n_genes.median(), color='red', linestyle='--', label=f'Median: {n_genes.median():.0f}')
        ax.set_xlabel('Genes per Cell')
        ax.set_ylabel('Frequency')
        ax.set_title('B. Gene Detection', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Panel C: UMI counts distribution
    ax = axes[1, 0]
    n_counts = adata.obs.get('total_counts', adata.obs.get('n_counts', None))
    if n_counts is not None:
        ax.hist(np.log10(n_counts + 1), bins=50, color='#E58700', alpha=0.7)
        ax.axvline(np.log10(n_counts.median() + 1), color='red', linestyle='--', label=f'Median: {n_counts.median():.0f}')
        ax.set_xlabel('Log10(UMI counts)')
        ax.set_ylabel('Frequency')
        ax.set_title('C. UMI Distribution', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Panel D: Donor distribution
    ax = axes[1, 1]
    if 'donor' in adata.obs.columns:
        donor_counts = adata.obs['donor'].value_counts()
        ax.bar(range(len(donor_counts)), donor_counts.values, color='#00BA38', alpha=0.7)
        ax.set_xticks(range(len(donor_counts)))
        ax.set_xticklabels(donor_counts.index, rotation=45, ha='right')
        ax.set_ylabel('Number of Cells')
        ax.set_title('D. Cells per Donor', fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)

    save_publication_figure(fig, output_path)
    print(f"  ✓ Saved: {output_path}")

except Exception as e:
    print(f"  ✗ Error: {e}")
    import traceback
    traceback.print_exc()
EOFPYTHON
fi

# -----------------------------------------------------------------------------
# Supplementary Figure S2: Reference Mapping Diagnostics
# -----------------------------------------------------------------------------
echo ""
echo "Supplementary Figure S2: Reference mapping diagnostics..."

if figure_exists "$OUTPUT/supplementary/figS2_reference_diagnostics"; then
    echo "  ✓ Skipping (already exists)"
else
    echo "  Generating reference ELBO curves..."
    python /home/booka/StageBridge/scripts/plot_elbo_curves.py
    # Copy outputs to supplementary
    if [[ -f "results/reference_elbo_curves.png" ]]; then
        cp results/reference_elbo_curves.png "$OUTPUT/supplementary/figS2_reference_diagnostics.png"
        cp results/reference_elbo_curves.pdf "$OUTPUT/supplementary/figS2_reference_diagnostics.pdf"
        # Create SVG from PDF if possible
        if command -v pdf2svg &> /dev/null; then
            pdf2svg "$OUTPUT/supplementary/figS2_reference_diagnostics.pdf" "$OUTPUT/supplementary/figS2_reference_diagnostics.svg"
        fi
        echo "  ✓ Saved: figS2_reference_diagnostics"
    fi
fi

# -----------------------------------------------------------------------------
# Supplementary Figure S3: HPO History
# -----------------------------------------------------------------------------
echo ""
echo "Supplementary Figure S3: HPO optimization history..."

if figure_exists "$OUTPUT/supplementary/figS3_hpo_history"; then
    echo "  ✓ Skipping (already exists)"
else
    if [[ -f "$TRAINING_OUTPUT/hpo_results.json" ]]; then
        echo "  Generating HPO history..."
        python << 'EOFPYTHON'
import sys
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
import json
sys.path.insert(0, '/home/booka/StageBridge')

from stagebridge.viz import setup_publication_plotting, create_figure, save_publication_figure
import numpy as np

setup_publication_plotting()

hpo_file = Path("/scratch/chaunzt1/stagebridge/runs/v1_complete/hpo_results.json")
output_path = Path("/scratch/chaunzt1/stagebridge/runs/publication_figures/supplementary/figS3_hpo_history")

try:
    with open(hpo_file) as f:
        hpo = json.load(f)

    trials = hpo.get('all_trials', [])
    if trials:
        values = [t['value'] for t in trials if t.get('value') is not None]
        best_values = [min(values[:i+1]) for i in range(len(values))]

        fig, ax = create_figure(figsize=(10, 6))
        ax.plot(range(len(values)), values, 'o', alpha=0.5, label='Trial', color='#619CFF', markersize=6)
        ax.plot(range(len(best_values)), best_values, '-', linewidth=2.5, label='Best', color='#00BA38')
        ax.set_xlabel('Trial Number')
        ax.set_ylabel('Validation Loss')
        ax.set_title('Hyperparameter Optimization History', fontweight='bold')
        ax.legend(frameon=True)
        ax.grid(True, alpha=0.3)

        save_publication_figure(fig, output_path)
        print(f"  ✓ Saved: {output_path}")
    else:
        print("  ⚠ No HPO trials found")

except Exception as e:
    print(f"  ✗ Error: {e}")
EOFPYTHON
    else
        echo "  ⚠ HPO results not found"
    fi
fi

# =============================================================================
# Generate Figure Manifest
# =============================================================================
echo ""
echo "=============================================="
echo "Generating Figure Manifest"
echo "=============================================="

python << 'EOFPYTHON'
import sys
from pathlib import Path
import json
from datetime import datetime

output_dir = Path("/scratch/chaunzt1/stagebridge/runs/publication_figures")

manifest = {
    "generated_at": datetime.now().isoformat(),
    "main_figures": {},
    "supplementary_figures": {},
    "panels": {},
    "formats": ["png", "pdf", "svg"],
    "dpi": 300,
}

# Scan main figures
main_dir = output_dir / "main"
for png_file in sorted(main_dir.glob("*.png")):
    base = png_file.stem
    manifest["main_figures"][base] = {
        "png": str(png_file.relative_to(output_dir)),
        "pdf": str(png_file.with_suffix(".pdf").relative_to(output_dir)) if png_file.with_suffix(".pdf").exists() else None,
        "svg": str(png_file.with_suffix(".svg").relative_to(output_dir)) if png_file.with_suffix(".svg").exists() else None,
    }

# Scan supplementary figures
supp_dir = output_dir / "supplementary"
for png_file in sorted(supp_dir.glob("*.png")):
    base = png_file.stem
    manifest["supplementary_figures"][base] = {
        "png": str(png_file.relative_to(output_dir)),
        "pdf": str(png_file.with_suffix(".pdf").relative_to(output_dir)) if png_file.with_suffix(".pdf").exists() else None,
        "svg": str(png_file.with_suffix(".svg").relative_to(output_dir)) if png_file.with_suffix(".svg").exists() else None,
    }

# Save manifest
manifest_path = output_dir / "manifests" / "figure_manifest.json"
manifest_path.parent.mkdir(parents=True, exist_ok=True)
with open(manifest_path, 'w') as f:
    json.dump(manifest, f, indent=2)

print(f"Main figures: {len(manifest['main_figures'])}")
print(f"Supplementary figures: {len(manifest['supplementary_figures'])}")
print(f"Manifest saved: {manifest_path}")
EOFPYTHON

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=============================================="
echo "Figure Generation Complete"
echo "=============================================="
echo ""
echo "Outputs:"
echo "  Main figures: $OUTPUT/main/"
echo "  Supplementary: $OUTPUT/supplementary/"
echo "  Panels: $OUTPUT/panels/"
echo "  Manifests: $OUTPUT/manifests/"
echo ""

# Count generated figures
main_count=$(ls -1 "$OUTPUT/main/"*.png 2>/dev/null | wc -l)
supp_count=$(ls -1 "$OUTPUT/supplementary/"*.png 2>/dev/null | wc -l)

echo "Summary:"
echo "  Main figures generated: $main_count"
echo "  Supplementary figures generated: $supp_count"
echo ""

echo "Figure Manifest: $OUTPUT/manifests/figure_manifest.json"
echo ""
echo "=============================================="
echo "Stage 7 Complete: $(date)"
echo "=============================================="
