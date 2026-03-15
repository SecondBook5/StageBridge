# Architecture: Layer B+C Ablation Design

**Purpose:** Document the systematic ablation study for Layers B+C (EA-MIST components) that validates the niche encoding architecture.

## Context in V1

The primary V1 evaluation focuses on **transition quality** from Layer D (flow matching). However, validating that Layers B+C properly encode niche information is essential — if the context vector doesn't carry stage-relevant signal, Layer D cannot learn meaningful niche-conditioned transitions.

This ablation study uses **auxiliary classification** as a probe for Layer B+C quality.

## Grouped Ordinal Labels

### Mapping

| Grouped label | Original stages | Biological rationale |
|--------------|----------------|---------------------|
| `early_like` (0) | Normal, AAH | Pre-neoplastic, intact alveolar architecture |
| `intermediate_like` (1) | AIS, MIA | In-situ / minimally invasive |
| `invasive_like` (2) | LUAD | Fully invasive adenocarcinoma |

### Why Grouping?

The original 5-class setup has insufficient per-class counts for reliable evaluation. Grouping to 3 classes ensures ≥4 examples per class per fold.

### Displacement Targets

| Class | Target |
|-------|--------|
| `early_like` | 0.0 |
| `intermediate_like` | 0.5 |
| `invasive_like` | 1.0 |

## Ablation Grid

### Axes

**Model variants (4):**

| Variant | Layer B | Layer C | Tests |
|---------|---------|---------|-------|
| `pooled` | Full encoder | Mean-pool | Baseline — no attention |
| `deep_sets` | Full encoder | DeepSets φ→ρ | Permutation invariance |
| `eamist_no_prototypes` | Full encoder | Set transformer | Attention without prototypes |
| `eamist` | Full encoder | Set transformer + prototypes | Full architecture |

**Reference feature modes (5):**

| Mode | HLCA | LuCA | Tests |
|------|------|------|-------|
| `no_atlas` | Zeroed | Zeroed | Spatial structure alone |
| `hlca_only` | Active | Zeroed | Healthy atlas contribution |
| `luca_only` | Zeroed | Active | Cancer atlas contribution |
| `hlca_luca` | Active | Active | Combined atlas signal |
| `hlca_luca_contrast` | Active | Active + contrast token | Cross-atlas modeling |

### Full Grid

4 variants × 5 atlas modes = **20 configurations**.

Each evaluated under:
- 3-fold donor-held-out cross-validation
- HPO to find best hyperparameters per configuration
- Multiple seeds for the final evaluation

## Evaluation Protocol

### Metrics

| Metric | Weight | Role |
|--------|--------|------|
| Displacement Spearman (ρ_s) | 40% | Ordinal ranking fidelity |
| Weighted kappa (κ_w) | 30% | Classification with ordinal penalty |
| Balanced accuracy | 20% | Per-class recall fairness |
| Macro F1 | 10% | Classification precision-recall |

### Composite Score

```
score = 0.40 * max(ρ_s, 0) + 0.30 * max(κ_w, 0) + 0.20 * bal_acc + 0.10 * macro_f1
```

The 70% ordinal weight reflects the goal: correctly ordering samples along the progression axis.

## Negative Controls

### Atlas Label Shuffle

- Globally shuffle HLCA/LuCA features (breaking atlas ↔ stage correspondence)
- **Expected:** Performance drops to near-chance
- **Validates:** Atlas features carry stage-relevant signal

### Within-Sample Niche Shuffle

- Randomly permute niche order within each sample
- **Expected:** Minimal impact on pooled, larger impact on attention models
- **Validates:** Attention mechanisms use niche relationships

## Scientific Claims Supported

1. **Atlas features carry signal** — `hlca_luca` > `no_atlas`, confirmed by atlas shuffle
2. **Both atlases contribute** — `hlca_luca` ≥ max(single atlas modes)
3. **Attention helps** — `eamist` > `pooled` under same atlas mode
4. **Context vector is informative** — High ablation scores indicate Layer D receives useful conditioning

## Relationship to V1 Evaluation

This ablation is **not the primary V1 evaluation**. It validates that:
- Layers B+C encode stage-relevant information
- The context vector passed to Layer D is meaningful
- The architectural choices in EA-MIST are justified

The primary V1 evaluation focuses on **transition quality** from Layer D.
