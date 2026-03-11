# Architecture: Rescue Ablation Design

**Purpose:** Document the grouped ordinal atlas ablation study that constitutes the primary publishable evaluation of EA-MIST.

## Motivation

The original 5-class classification benchmark (Normal → AAH → AIS → MIA → LUAD) suffered from:

1. **Insufficient per-class counts** — Only 56 lesions across 25 donors, with some stages having ≤ 5 examples per fold
2. **Empty test classes** — 3-fold CV produced folds with zero test examples for rare stages
3. **Metric instability** — Macro-F1 undefined when a class is absent from test set
4. **Dead pretrained checkpoint** — Embedding dimension mismatch made pretrained local encoder unusable

The rescue design addresses all four issues through label grouping, systematic atlas ablation, and robust ordinal metrics.

## Grouped Ordinal Labels

### Mapping

| Grouped label | Original stages | Biological rationale |
|--------------|----------------|---------------------|
| `early_like` (0) | Normal, AAH | Pre-neoplastic, intact alveolar architecture |
| `intermediate_like` (1) | AIS, MIA | In-situ / minimally invasive, early transformation |
| `invasive_like` (2) | LUAD | Fully invasive adenocarcinoma |

### Class Balance

| Class | Count | Proportion |
|-------|-------|-----------|
| `early_like` | 12 | 21% |
| `intermediate_like` | 18 | 32% |
| `invasive_like` | 26 | 46% |

This yields ≥ 4 examples per class per fold (3-fold CV), eliminating the empty-class problem.

### Displacement Targets

Ordinal regression targets are evenly spaced across the progression axis:

| Class | Target |
|-------|--------|
| `early_like` | 0.0 |
| `intermediate_like` | 0.5 |
| `invasive_like` | 1.0 |

## Ablation Grid

### Axes

**Model families (3):**

| Family | Architecture | Tests |
|--------|-------------|-------|
| `pooled` | Mean-pool aggregation | Baseline — no attention |
| `deep_sets` | φ→ρ MLP | Permutation invariance without attention overhead |
| `eamist` | Set transformer + prototypes | Full model with induced attention and prototype bottleneck |

**Reference feature modes (5):**

| Mode | Description | Tests |
|------|------------|-------|
| `no_atlas` | All atlas features zeroed | Spatial structure alone |
| `hlca_only` | Only HLCA (healthy reference) | Healthy atlas contribution |
| `luca_only` | Only LuCA (cancer reference) | Cancer atlas contribution |
| `hlca_luca` | Both atlases active | Combined atlas signal |
| `hlca_luca_contrast` | Both + explicit contrast token | Cross-atlas relationship modeling |

### Full Grid

3 models × 5 atlas conditions = **15 configurations**.

Each evaluated under:
- 3-fold donor-held-out cross-validation
- 50 Optuna HPO trials per fold
- 3 random seeds for the best hyperparameters

Total: 15 × 3 folds × 50 trials = **2,250 HPO trials** (phase 1), then 15 × 3 folds × 3 seeds = **135 final evaluations** (phase 2).

## Evaluation Protocol

### Phase 1: Hyperparameter Optimization

For each (model, mode, fold) triple:
1. Run 50 Optuna trials with TPE sampler + median pruning
2. Select the trial maximizing the grouped composite selection score on the validation set
3. Record best parameters and validation metrics

### Phase 2: Fixed-Parameter Evaluation

For each (model, mode):
1. Use the best hyperparameters from Phase 1 (per fold)
2. Train 3 independent seeds per fold
3. Report mean ± std across 3 folds × 3 seeds = 9 runs

### Primary Metrics

| Metric | Weight in composite | Role |
|--------|-------------------|------|
| Displacement Spearman ($\rho_s$) | 40% | Ordinal ranking fidelity |
| Weighted kappa ($\kappa_w$) | 30% | Classification agreement penalizing distant errors |
| Balanced accuracy | 20% | Per-class recall fairness |
| Macro F1 | 10% | Classification precision-recall balance |

### Composite Selection Score

$$\text{score} = 0.40 \cdot \max(\rho_s, 0) + 0.30 \cdot \max(\kappa_w, 0) + 0.20 \cdot \text{bal\_acc} + 0.10 \cdot F_1^{macro}$$

The 60/30 ordinal/classification split reflects the study's emphasis: correctly ordering lesions along the progression axis matters more than exact class identity.

## Negative Controls

Permutation-based controls run as a separate pass with `--with-controls`:

### Atlas Label Shuffle

- Deep copy all bags
- Globally shuffle HLCA and LuCA features across lesions (breaking atlas ↔ stage correspondence)
- Train and evaluate in `hlca_luca` mode
- **Expected result:** Performance drops to near-chance, proving atlas features carry stage-relevant signal

### Within-Lesion Niche Shuffle

- Deep copy all bags
- Randomly permute neighborhood order within each lesion
- Train and evaluate in `hlca_luca` mode
- **Expected result:** Minimal impact on pooled model (mean-pool is permutation-invariant), moderate impact on attention-based models if spatial ordering contains signal

## Key Scientific Claims This Design Supports

1. **Atlas features carry stage signal** — `hlca_luca` > `no_atlas`, confirmed by atlas shuffle control
2. **Both atlases contribute** — `hlca_luca` ≥ max(`hlca_only`, `luca_only`)
3. **Attention helps** — `eamist` or `deep_sets` > `pooled` under the same atlas mode
4. **Ordinal structure is preserved** — High displacement Spearman and weighted kappa indicate the model captures the biological ordering, not just class boundaries

## Config Reference

Key YAML parameters (`configs/context_model/eamist.yaml`):

```yaml
use_grouped_labels: true
model_families: [pooled, deep_sets, eamist]
reference_feature_modes: [no_atlas, hlca_only, luca_only, hlca_luca, hlca_luca_contrast]
use_atlas_contrast_token: false      # set true only for hlca_luca_contrast mode
pretrained_local_checkpoint: null     # disabled — train from scratch
n_hpo_trials: 50
n_seeds_final: 3
```

## Launch

```bash
# Phase 1: HPO ablation
bash scripts/run_rescue_ablation.sh

# Phase 2: Negative controls
bash scripts/run_rescue_ablation.sh --with-controls
```

Log output: `outputs/scratch/rescue_ablation_*.log`
