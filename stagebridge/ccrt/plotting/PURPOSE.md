# `plotting/` — Figures from Evaluation Outputs Only

## Purpose

`plotting/` produces figures from **evaluation outputs only**. It is a pure
downstream consumer at the very end of the CCRT pipeline: it reads the
standardized metrics and prediction summaries emitted by `evaluation/` and renders
them into figures, with no back-channel into biology-aware ingestion or
training-time code. Keeping figure generation strictly downstream guarantees that
what appears in a figure is exactly what evaluation measured — nothing is
recomputed, re-fit, or re-read from raw data at plot time. This makes every figure
traceable to a specific evaluation artifact and reproducible across biological
systems without `plotting/` knowing which system it is drawing.

## Owns

- Figure generation from evaluation outputs (metrics, decomposition summaries, stability diagnostics).
- Figure styling, layout, and export to allowed output roots.

## Does NOT

- Does NOT import `adapters/` (see `../../../docs/ccrt/AGENT_BOUNDARIES.md`).
- Does NOT import `training/`.
- Does NOT read raw disease data or re-run the model, operators, or transport — it plots evaluation outputs only.
- Does NOT hard-code `LUAD`, `PanIN`, `macrophage`, `CAF`, `cancer`, `virus`, or a malignancy axis.
- Does NOT write figures into the source tree; outputs land only in allowed roots (`$STAGEBRIDGE_RESULTS/`).
