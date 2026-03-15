# Architecture: Evaluation and Interpretation

**Scientific layer:** Evaluation
**Package location:** `stagebridge/evaluation/`

## Role in the System

This layer evaluates the complete StageBridge pipeline: assessing transition model quality, running ablations on Layer B+C, and computing metrics that support scientific claims about niche-gated transitions.

## Primary Evaluation: Transition Quality

### V1 Metrics

| Metric | Description |
|--------|-------------|
| Sinkhorn distance | OT distance between predicted and true target distributions |
| MMD-RBF | Maximum mean discrepancy with RBF kernel |
| Trajectory smoothness | Mean velocity magnitude along paths |
| Niche sensitivity | Change in predictions under context perturbation |
| Donor consistency | Within-donor trajectory agreement |

### Biological Validation

| Validation | Method |
|------------|--------|
| Pseudotime correlation | Compare learned trajectories to independent pseudotime methods |
| Gene program attribution | Which genes drive velocity at each transition? |
| Niche regime identification | Cluster niches by transition behavior |

## Secondary Evaluation: Layer B+C Ablations

The EA-MIST layers (B+C) are evaluated via auxiliary classification:

### Classification Metrics

| Metric | Description |
|--------|-------------|
| `macro_f1` | Mean per-class F1 |
| `balanced_accuracy` | Mean per-class recall |
| `displacement_spearman` | Rank correlation of ordinal predictions |
| `weighted_kappa` | Linear-weighted Cohen's κ (grouped labels) |

### Atlas Ablation Grid

Tests contribution of reference features:

| Mode | HLCA | LuCA | Tests |
|------|------|------|-------|
| `no_atlas` | Zeroed | Zeroed | Spatial-only baseline |
| `hlca_only` | Active | Zeroed | Healthy atlas contribution |
| `luca_only` | Zeroed | Active | Cancer atlas contribution |
| `hlca_luca` | Active | Active | Combined signal |

### Model Family Comparison

| Family | Description |
|--------|-------------|
| `pooled` | Mean-pool aggregation (no attention) |
| `deep_sets` | DeepSets φ→ρ MLP |
| `eamist` | Full set-transformer with prototypes |

## Negative Controls

### Atlas Label Shuffle

- Shuffle HLCA/LuCA features globally (breaking atlas ↔ stage correspondence)
- **Expected:** Performance drops, proving atlas features carry signal

### Niche Shuffle

- Randomly permute niche order within samples
- **Expected:** Minimal impact on pooled, larger impact on attention models

### Context Ablation

- Remove niche conditioning from Layer D
- **Expected:** Transition quality degrades if niche context matters

## Cross-Validation Protocol

Donor-held-out evaluation:
- No donor appears in both train and test
- 3-fold CV with stratified stage distribution
- Report mean ± std across folds and seeds

## Uncertainty Quantification

V1 must report uncertainty:
- Bootstrap confidence intervals on metrics
- Trajectory variance (if using stochastic inference)
- Per-prediction confidence scores

## Key Scientific Claims Supported

1. **Niche context improves transitions** — Conditioned model > unconditioned
2. **Both atlases contribute** — `hlca_luca` ≥ max(single atlas modes)
3. **Results are robust** — Consistent across spatial backends
4. **Transitions are biologically meaningful** — Gene programs align with known biology

## Relationship to Other Layers

- **Upstream:** All model layers produce predictions
- **Downstream:** Results tracking persists artifacts; visualization renders figures
