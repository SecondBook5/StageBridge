# Architecture: Evaluation and Interpretation

**Scientific layer:** 6 — Lesion-level evaluation, ablation, and negative controls
**Package location:** `stagebridge/evaluation/`

## Role in the System

This layer evaluates trained EA-MIST models: computing classification and ordinal metrics, running permutation-based negative controls, and assembling ablation tables that compare model families and atlas configurations. It converts raw predictions into the evidence needed to support claims about niche-stage relationships.

## Metrics

### Classification Metrics

Computed by `compute_stage_metrics` (canonical 5-class) and `compute_grouped_stage_metrics` (grouped 3-class):

| Metric | Formula | Scope |
|--------|---------|-------|
| `macro_f1` | Mean of per-class F1 | Both |
| `balanced_accuracy` | Mean of per-class recall | Both |
| `accuracy` | Fraction correct | Both |
| `central_recall` | Mean recall of intermediate classes (AAH, AIS, MIA) | Canonical only |
| `weighted_kappa` | Linear-weighted Cohen's κ | Grouped only |

**Linear-weighted kappa** penalizes disagreements proportional to the ordinal distance between predicted and true classes:

$$\kappa_w = 1 - \frac{\sum_{i,j} w_{ij} \cdot O_{ij}}{\sum_{i,j} w_{ij} \cdot E_{ij}} \quad \text{where } w_{ij} = \frac{|i - j|}{C - 1}$$

$O$ is the observed confusion matrix, $E$ is the expected matrix under chance.

### Displacement Metrics

Computed from the scalar displacement predictions against ordinal targets:

| Metric | Description |
|--------|------------|
| `displacement_mae` | Mean absolute error |
| `displacement_spearman` ($\rho_s$) | Spearman rank correlation of displacement predictions vs targets |
| `stage_monotonicity` | Fraction of stage pairs where mean predicted displacement preserves the correct ordering |

### Composite Selection Scores

Used by the HPO loop to select the best trial. The two score variants reflect different evaluation priorities:

**Canonical (5-class):**
$$\text{score} = F_1^{macro} + 0.25 \cdot \text{bal\_acc} + 0.10 \cdot \max(\rho_s, 0) + 0.05 \cdot \text{central\_recall}$$

**Grouped (3-class):**
$$\text{score} = 0.40 \cdot \max(\rho_s, 0) + 0.30 \cdot \max(\kappa_w, 0) + 0.20 \cdot \text{bal\_acc} + 0.10 \cdot F_1^{macro}$$

The grouped score prioritizes ordinal metrics: Spearman displacement correlation (40%) and weighted kappa (30%). This reflects the scientific goal — correctly ordering lesions along the progression continuum matters more than exact 3-class accuracy.

### Confusion Matrix and Support

`grouped_confusion_matrix_payload` and `grouped_support_payload` produce structured payloads for logging and reporting:

- Confusion matrix as a flat dictionary with keys like `pred_{i}_true_{j}`
- Per-class support counts for train/val/test splits

## Ablation Framework

### Atlas Ablation Grid

The benchmark evaluates each model family × reference feature mode combination:

| Model Family | Description |
|-------------|-------------|
| `pooled` | Mean-pool bag aggregation (no attention) |
| `deep_sets` | DeepSets φ→ρ MLP |
| `eamist` | Full set-transformer with prototypes |

| Reference Mode | Atlas Features |
|---------------|----------------|
| `no_atlas` | All atlas features zeroed |
| `hlca_only` | Only HLCA healthy atlas |
| `luca_only` | Only LuCA cancer atlas |
| `hlca_luca` | Both atlases |
| `hlca_luca_contrast` | Both + contrast token |

Full grid: 3 × 5 = 15 configurations, each evaluated under 3-fold donor-held-out CV with 50 HPO trials per fold.

### Cross-Validation

Donor-held-out 3-fold CV ensures no donor appears in both train and test:

- `split_donor_cv` groups lesions by donor/patient
- Each fold: ~37 train, ~9 val, ~10 test lesions
- Stratified by stage to maintain class proportions

### Negative Controls

Two permutation baselines verify that model performance depends on atlas feature content, not just feature dimensionality or bag structure:

| Control | Method | Preserves | Destroys |
|---------|--------|-----------|----------|
| `atlas_label_shuffle` | Shuffle HLCA/LuCA features across lesions globally | Spatial structure, feature statistics | Atlas ↔ stage alignment |
| `within_lesion_niche_shuffle` | Randomly permute neighborhood order within each lesion | Per-lesion bag statistics | Spatial structure |

Controls use deep copies of the original bags, run `hlca_luca` mode, and are evaluated with the same HPO budget. A valid model should perform **worse** under `atlas_label_shuffle` than the intact `hlca_luca` condition.

## Reporting

### Per-Configuration Output

Each configuration (model × mode × fold) produces:

- Best trial parameters and composite score
- Full metric dictionary (classification + displacement)
- Confusion matrix
- Per-fold support counts

### Benchmark Summary

The benchmark loop (`benchmark_full_atlas_ablation`) aggregates across folds and seeds:

- Mean ± std of all metrics per configuration
- Ranked comparison tables by composite score
- Delta columns showing lift/drop vs `no_atlas` baseline
- Statistical significance tests across seeds

## Key Design Principles

1. **Evaluation is non-optional.** All metrics, controls, and ablation tables are computed during the benchmark, not as a separate post-hoc step.
2. **Grouped labels are the primary evaluation axis.** The 3-class grouped ordinal scheme addresses the statistical weakness of 5-class classification with small cohorts.
3. **Negative controls are part of the evidence.** Performance drop under atlas shuffle is essential for claiming that atlas features carry stage-relevant signal.

## Relationship to Other Layers

- **Upstream:** Context model produces stage logits and displacement predictions; training pipeline runs HPO and fold loops
- **Downstream:** Results tracking persists metric tables; visualization renders ablation plots and confusion matrices
