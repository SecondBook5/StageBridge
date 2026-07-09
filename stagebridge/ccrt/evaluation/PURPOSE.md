# `evaluation/` — Standardized Metrics from Model Outputs

## Purpose

`evaluation/` computes standardized metrics from CCRT model outputs and exported
predictions. It is the layer that turns the operator's context-residual
decomposition — self-intrinsic transition, full context-conditioned transition,
context-residual drift and growth, regulatory mediator, sender-context
attribution, counterfactual context perturbation, and semantic transport stability
— together with the synthetic ground-truth labels, into reproducible, comparable
numbers. It scores mechanism recovery on `synthetic/` benchmarks, quantifies
matching and context-effect stability, and produces the prediction summaries that
`plotting/` later renders. By consuming only model outputs and standardized
artifacts, `evaluation/` stays decoupled from disease-specific ingestion and
remains fully reproducible across biological systems.

## Owns

- Metric definitions over model outputs and exported predictions.
- Mechanism-recovery scoring against `synthetic/` ground-truth labels.
- Stability diagnostics (matching stability, effective rank, context-effect stability).
- Standardized, serialized evaluation outputs consumed by `plotting/`.

## Does NOT

- Does NOT read raw disease data directly — it consumes model outputs and standardized artifacts only.
- Does NOT hard-code `LUAD`, `PanIN`, `macrophage`, `CAF`, `cancer`, `virus`, or a malignancy axis.
- Does NOT define architecture, run training, or perform sender-context/transport computation — it scores their results.
- Does NOT write outputs into the source tree; results land only in allowed output roots (see `../../../docs/ccrt/DIRECTORY_PURPOSES.md`).
