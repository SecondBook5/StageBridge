# ADR 002 WES As Regularizer

Decision: WES enters v1 as a regularizer on admissible transport, not as unrestricted conditioning.

## Context

WES provides per-donor mutation burden, driver mutations, and CNV patterns. Two entry paths: (1) direct conditioning where the drift network reads WES features as input; (2) regularization via auxiliary loss constraining transport consistency with genomic state.

## Rationale

- Limited patient count makes direct conditioning prone to overfitting
- Regularization tests the hypothesis (evolutionary state constrains transport) without confounding niche-gating
- Ablation (with vs without WES) is cleaner when WES does not change drift architecture
- If regularization proves useful, direct conditioning can follow in v2

## Consequences

- WES must be a real regularizer with measurable effect, not metadata tagging
- Ablation framework compares regularized vs unregularized
- Direct WES conditioning is deferred, not abandoned
