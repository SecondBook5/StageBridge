# WES Regularization Rationale

## Why Evolutionary State Matters

Cancer progression is driven by the accumulation of somatic mutations, copy-number alterations, and other genomic changes. These evolutionary events:

- Enable or constrain which cell states are accessible
- Influence the rate and direction of phenotypic transitions
- Create patient-specific evolutionary contexts that modulate disease dynamics

Two patients at the same histological stage but with different mutational profiles (e.g., KRAS-mutant vs EGFR-mutant) may undergo different transition dynamics. Ignoring genomic state treats all patients at a given stage as interchangeable, which they are not.

## Why Regularization Rather Than Conditioning

In v1, WES features enter as a regularizer on transport, not as direct input to the drift network. This is a conservative design choice:

1. **Limited sample size** — The number of donors is small relative to the dimensionality of genomic features. Direct conditioning risks learning donor-specific associations that do not generalize.

2. **Separation of concerns** — The primary question is about niche gating. WES regularization tests whether evolutionary state constrains transport without confounding the niche-gating analysis. If WES features directly condition the drift network alongside niche context, disentangling their contributions is harder.

3. **Testable hypothesis** — Regularization provides a clean ablation: compare transport quality with and without WES constraints. If WES regularization improves held-out performance, evolutionary state is informatively constraining the model.

## How WES Regularization Works (Conceptually)

- Per-donor features: mutation burden, driver mutation status (KRAS, EGFR, STK11, TP53), copy-number summary
- Auxiliary loss: penalizes transport paths where donors with different evolutionary states produce identical transition dynamics
- Effect: the model is encouraged to learn evolutionary-state-aware transitions without being given direct genomic input
- Example: a high-mutation-burden donor's transitions should differ from a low-mutation-burden donor's, and the regularizer enforces this

## What This Enables

- Identification of transitions where evolutionary state matters most
- Comparison of niche-gated dynamics across evolutionary subgroups
- A principled path toward direct WES conditioning in v2, informed by v1 regularization results

## What This Does Not Claim

- WES regularization does not guarantee better predictions
- Negative results (regularization does not help) are informative
- The auxiliary loss formulation is a modeling choice that may need iteration
