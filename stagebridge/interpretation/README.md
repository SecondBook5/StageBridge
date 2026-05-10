# Interpretation

Post-training interpretability tools for StageBridge, covering attention analysis, trajectory dynamics, manifold visualization, and interaction networks.

## Modules Overview

| Module | Purpose |
|--------|---------|
| `ablation.py` | Token ablation analysis (ring, reference, pathway contribution) |
| `attention.py` | Attention pattern extraction and distance decay analysis |
| `networks.py` | Cell-cell interaction network inference |
| `dynamics.py` | OT-based trajectory analysis and fate probability |
| `trajectory_plots.py` | Publication-quality trajectory visualizations |
| `manifold_viz.py` | Multi-method manifold comparison (UMAP, PHATE, t-SNE, PCA) |
| `plotting.py` | General plotting utilities |

## Key Capabilities

### 1. Token Ablation Analysis (`ablation.py`)

Measure contribution of each input component by ablating and measuring reconstruction change.

```python
from stagebridge.interpretation import AblationModule, plot_ablation_importance

# Compute ablation importance for each token type
results = AblationModule.compute(model, dataloader, device)

# results contains:
# - ring_1, ring_2, ring_3, ring_4: Spatial ring contributions
# - hlca, luca: Reference atlas contributions
# - pathway, stats: Auxiliary token contributions

plot_ablation_importance(results, output_path="ablation.pdf")
```

### 2. Attention Pattern Extraction (`attention.py`)

Extract and analyze attention weights from the niche encoder.

```python
from stagebridge.interpretation import AttentionModule, extract_attention_patterns

patterns = extract_attention_patterns(model, batch)
# patterns.neighbor_attention: [B, K] attention to each neighbor
# patterns.empty_attention: [B] attention to empty token
# patterns.distance_decay: Attention vs distance relationship
```

### 3. Cell-Cell Interaction Networks (`networks.py`)

Build interaction networks from attention weights, identifying cell type communication patterns.

```python
from stagebridge.interpretation import InteractionNetwork, build_interaction_network

network = build_interaction_network(
    attention_weights,
    cell_types,
    threshold=0.1,
)
# Returns directed graph: sender -> receiver edges weighted by attention
```

### 4. Trajectory Dynamics (`dynamics.py`)

OT-based trajectory inference with fate probability estimation.

```python
from stagebridge.interpretation import TrajectoryAnalysis, FateProbability

# Compute cell fate probabilities
fate = FateProbability.compute(
    source_latent=z_normal,
    target_latent=z_invasive,
    reg=0.02,
)
# fate.probabilities: [n_source] probability of reaching target

# Identify dynamic driver genes along trajectory
drivers = TrajectoryAnalysis.find_drivers(
    expression=adata.X,
    pseudotime=fate.pseudotime,
    gene_names=adata.var_names,
)
```

### 5. Trajectory Visualization (`trajectory_plots.py`)

Publication-quality trajectory figures.

```python
from stagebridge.interpretation import (
    plot_temporal_evolution,
    plot_fate_probability,
    plot_single_cell_trajectories,
    plot_driver_heatmap,
    plot_gene_dynamics,
    create_trajectory_animation,
)

# Temporal density evolution
plot_temporal_evolution(embeddings, stages, times, output_path="temporal.pdf")

# Fate probability on embedding
plot_fate_probability(embeddings, fate_probs, output_path="fate.pdf")

# Single-cell trajectories with velocity arrows
plot_single_cell_trajectories(
    source_emb, target_emb, velocities,
    output_path="trajectories.pdf"
)

# Driver gene dynamics heatmap
plot_driver_heatmap(drivers, output_path="drivers.pdf")
```

### 6. Manifold Comparison (`manifold_viz.py`)

Compare expression vs latent manifolds using multiple embedding methods.

```python
from stagebridge.interpretation import (
    plot_manifold_comparison,
    plot_multi_method_comparison,
    plot_trajectory_straightness,
    plot_geodesic_comparison,
    plot_phase_map,
    compute_manifold_comparison,
)

# Side-by-side expression vs latent comparison
plot_manifold_comparison(
    expression=adata.X,
    latent=model_output.context,
    stages=adata.obs['stage'],
    method='umap',
    output_path="manifold_compare.pdf"
)

# Multi-method grid (PCA, UMAP, t-SNE, PHATE)
plot_multi_method_comparison(
    data=latent,
    stages=stages,
    methods=['pca', 'umap', 'tsne', 'phate'],
    output_path="methods_grid.pdf"
)

# Quantify trajectory straightening
result = compute_manifold_comparison(expression, latent, stages)
# result.expression_curvature vs result.latent_curvature

# Phase portrait with velocity field
plot_phase_map(embeddings, velocities, stages, output_path="phase.pdf")
```

## Full Interpretation Pipeline

The `run_interpretation.py` script in `scripts/analysis/` runs all analyses:

```bash
python scripts/analysis/run_interpretation.py \
    --checkpoint results/best_checkpoint.pt \
    --data-dir $DATA/canonical \
    --output-dir results/interpretation \
    --all  # Run ablation, attention, networks, trajectories, manifolds
```

## Attribution

The token ablation and attention analysis modules are adapted from AMICI (Hong et al., bioRxiv 2025) for StageBridge's architecture:
- Visium spot-level data (vs single-cell Xenium)
- Continuous K-nearest neighbors (vs discrete rings in original)
- Stage progression context (vs static snapshot)

AMICI reference: Hong J, Desai K, Nguyen TD, Nazaret A, Levy N, Ergen C, Plitas G, Azizi E. AMICI: Attention Mechanism Interpretation of Cell-cell Interactions. bioRxiv 2025. doi:10.1101/2025.09.22.677860. https://github.com/azizilab/amici. License: CC BY-NC-ND 4.0. Patent pending (U.S. Serial No. 63/884,704).
