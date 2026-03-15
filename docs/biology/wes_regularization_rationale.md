# WES Regularization Rationale

## Why Evolutionary State Matters

Cancer progression is driven by the accumulation of somatic mutations, copy-number alterations, and other genomic changes. These evolutionary events:

- Enable or constrain which cell states are accessible
- Influence the rate and direction of phenotypic transitions
- Create patient-specific evolutionary contexts that modulate disease dynamics

Two patients at the same histological stage but with different mutational profiles (e.g., KRAS-mutant vs EGFR-mutant) may undergo different transition dynamics. Ignoring genomic state treats all patients as interchangeable, which they are not.

## V1 Approach: Regularization

In V1, WES features enter as a **regularizer on transitions**, not as direct input to the velocity network:

### Why Regularization Rather Than Conditioning?

1. **Limited sample size** — The number of donors is small relative to genomic feature dimensionality. Direct conditioning risks overfitting to donor-specific patterns.

2. **Separation of concerns** — The primary V1 question is about niche gating. WES regularization tests whether evolutionary state constrains transitions without confounding the niche-gating analysis.

3. **Testable hypothesis** — Regularization provides a clean ablation: compare transition quality with and without WES constraints.

### How It Works

- Per-donor features: mutation burden, driver mutation status (KRAS, EGFR, STK11, TP53), copy-number summary
- Auxiliary loss: penalizes transitions where donors with different evolutionary states produce identical dynamics
- Effect: model is encouraged to learn evolutionary-state-aware transitions
- Example: high-mutation-burden transitions should differ from low-mutation-burden transitions

## WES Features (V1)

| Feature | Description |
|---------|-------------|
| `total_variants` | Total number of somatic variants |
| `missense_count` | Count of missense mutations |
| `frameshift_count` | Count of frameshift mutations |
| `stop_gained_count` | Count of stop-gain mutations |
| `tmb` | Tumor mutation burden (variants/Mb) |
| `transition_transversion_ratio` | Ti/Tv ratio |
| `driver_mutations` | Binary flags for key drivers (KRAS, EGFR, etc.) |
| `cna_burden` | Copy number alteration burden (if available) |

## What This Enables

- Identification of transitions where evolutionary state matters most
- Comparison of niche-gated dynamics across evolutionary subgroups
- Foundation for V2 direct WES conditioning, informed by V1 results

## V2 Extension: Direct Conditioning

If V1 regularization shows evolutionary state matters:
- V2 can add WES features directly to the velocity network
- FiLM or gated conditioning (similar to evolution branch in EA-MIST)
- Enables evolutionary-trajectory-specific predictions

## Ablation Design

| Condition | WES Regularization | Tests |
|-----------|-------------------|-------|
| Baseline | Off | Pure niche-conditioned transitions |
| Regularized | On | Evolutionary constraint effect |

Compare: transition quality, niche regime consistency, per-donor trajectory variance
