# The Niche Gating Hypothesis

## Statement

Local epithelial-stromal-immune neighborhood structure modulates cell-state transitions between LUAD initiation stages. The composition and spatial arrangement of the tissue microenvironment around cells influences the probability, direction, and dynamics of progression to subsequent disease stages.

## What "Niche-Gated" Means

A transition is niche-gated if the learned dynamics depend on local tissue context — not just the intrinsic state of the transitioning cell:

- Two cells at the AAH stage with similar transcriptional profiles but different surrounding niches (one immune-rich, one fibroblast-rich) should have different predicted trajectories.
- Removing niche conditioning should measurably degrade transition model performance.
- The model should learn niche-type-specific contributions to transition dynamics.

## Why This Hypothesis Is Plausible

1. **Stromal remodeling** — Cancer-associated fibroblasts create permissive environments for invasion. The AIS-to-MIA transition likely involves stromal activation captured in spatial composition.

2. **Immune surveillance** — Immune cell composition changes across the initiation ladder. Immune-hot vs immune-cold niches may gate progression differently, particularly at the AAH-to-AIS boundary.

3. **Vascular remodeling** — Angiogenesis and vascular patterning change as tumors progress. Endothelial cell density may influence nutrient supply and progression rate.

4. **Spatial evidence** — The Peng cohort includes matched Visium spatial data, enabling direct measurement of spot-level cell-type composition at each stage.

## How StageBridge Tests This

### Primary Test: Context Ablation

Compare flow matching with vs without niche conditioning:
- **Conditioned:** Velocity field receives context vector from Layer C
- **Unconditioned:** Velocity field receives no niche information

If niche-gated hypothesis is correct: conditioned model should produce better transitions (lower Sinkhorn distance, better trajectory smoothness).

### Secondary Tests

1. **Niche perturbation** — Shuffle niche contexts; observe change in predicted trajectories
2. **Niche regime analysis** — Cluster niches by composition; compare transition dynamics across clusters
3. **Context sensitivity** — Measure gradient of velocity field with respect to context vector

### Ablation Framework (Layers B+C)

The Layer B+C ablation tests whether the context vector carries stage-relevant information:
- `no_atlas` vs `hlca_luca` mode comparison
- Atlas shuffle negative control
- Model family comparison (pooled vs attention)

## Expected Outcomes

If hypothesis is **supported**:
- Conditioned transitions > unconditioned transitions
- Niche perturbation changes predictions meaningfully
- Distinct niche regimes show distinct transition dynamics
- Atlas features improve context quality

If hypothesis is **not supported**:
- Conditioning doesn't improve transitions
- Niche perturbation has minimal effect
- Transitions are primarily cell-intrinsic

## What It Does Not Claim

- Niche composition is not claimed to be the **only** determinant of progression
- Niche gating may not be uniform across all transitions
- The model provides quantitative evidence, not definitive proof
