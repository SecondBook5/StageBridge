# `scripts/ccrt/` — CCRT Workflow Entrypoints

This directory holds thin runnable scripts and entrypoints for CCRT workflows. Each
script calls exactly one workflow and contains **minimal logic** — argument parsing,
config loading, and dispatch, not business logic. Like `stagebridge/ccrt/cli/`,
scripts here stay thin: they wire together configured workflows rather than
reimplementing the substance, which lives in the packages the script invokes. This
keeps the runnable surface a transparent shell over the pipeline, so the same
architecture serves every biological system without duplicating logic in scripts.

## Owns

- Thin runnable scripts / entrypoints that launch one CCRT workflow apiece.
- Wiring between a configuration file (from `configs/ccrt/`) and the workflow it
  parameterizes.
- Minimal argument parsing and dispatch to the underlying package workflow.

## Does NOT

- Define architecture, operators, losses, or any model or training logic — those
  live in `stagebridge/ccrt/`.
- Reimplement business logic or hold nontrivial computation.
- Contain disease-specific biology; biology enters only at the `adapters/` step.
- Write generated artifacts inside the source tree — outputs go to
  `$STAGEBRIDGE_DATA/` or `$STAGEBRIDGE_RESULTS/`, never `stagebridge/`, `tests/`,
  or the repo `data/` directory.

See `../../docs/ccrt/ARCHITECTURE_LOCK.md` and `../../docs/ccrt/DIRECTORY_PURPOSES.md`.
