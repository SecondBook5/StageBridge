# The Niche Gating Hypothesis

## Statement

Local epithelial-stromal-immune neighborhood structure changes transition behavior between LUAD initiation stages. Specifically, the composition and spatial arrangement of the tissue microenvironment around premalignant epithelial cells influences whether and how those cells progress to the next disease stage.

## What "Niche-Gated" Means

A transition is niche-gated if the probability, speed, or trajectory of stage progression depends on the local tissue context — not just the intrinsic state of the transitioning cell. In concrete terms:

- Two epithelial cells at the AAH stage with similar transcriptional profiles but different surrounding niches (one immune-rich, one fibroblast-rich) should have different predicted transition dynamics.
- Shuffling niche compositions while holding cell state fixed should measurably change model predictions.
- The context model should learn niche-type-specific contributions to transition behavior.

## Why This Hypothesis Is Plausible

1. **Stromal remodeling** — Cancer-associated fibroblasts are known to create permissive environments for invasion. The transition from AIS (non-invasive) to MIA (minimally invasive) likely involves stromal activation that could be captured in spatial composition.

2. **Immune surveillance** — Immune cell composition changes across the initiation ladder. Immune-hot vs immune-cold niches may gate progression differently, particularly at the AAH-to-AIS boundary where immune escape mechanisms may first become relevant.

3. **Vascular remodeling** — Angiogenesis and vascular patterning change as tumors progress. Endothelial cell density and spatial organization in the niche may influence nutrient supply and thus progression rate.

4. **Spatial evidence** — The Peng cohort includes matched Visium spatial data, allowing direct measurement of spot-level cell-type composition at each stage. Tangram mapping connects single-cell identities to spatial positions.

## How StageBridge Tests This

1. The context model encodes niche composition as typed tokens (epithelial, stromal, immune, vascular).
2. The transition model is conditioned on this niche context.
3. The ablation framework compares niche-conditioned vs unconditioned transitions (set-only vs RNA-only).
4. The context sensitivity analysis (niche shuffling) directly tests whether the model uses niche information.
5. The niche regime analysis clusters niches and compares transition dynamics across clusters.

If the niche-gated hypothesis is correct, set-only should outperform RNA-only, niche shuffling should change predictions, and niche regime analysis should reveal composition-dependent transition differences.

## What It Does Not Claim

- It does not claim that niche composition is the only determinant of progression
- It does not claim that niche gating is uniform across all transitions
- It does not claim that the model will definitively prove or disprove the hypothesis — it provides a framework for quantitative testing
