# Project Layout

This repository is notebook-first for grading and review.

- Primary entrypoint: `StageBridge.ipynb` (top-level)
- Supporting code: `stagebridge/` package + `scripts/` wrappers/utilities
- Generated artifacts: local-only under `outputs/` (not versioned except keepers)

## Top-Level Directories

- `stagebridge/`: reusable Python package (model, IO, preprocessing, training, viz)
  - `stagebridge/workflows/`: package-first orchestration functions
  - `stagebridge/notebook_api.py`: notebook facade (`compose_config`, `run_step`, `run_pipeline`)
  - `stagebridge/cli.py`: unified CLI (`python -m stagebridge.cli ...`)
- `scripts/`: operational wrappers called by notebook controls
  - `scripts/pipeline/`: canonical pipeline implementations
  - `scripts/train/`: canonical training implementations
  - `scripts/eval/`: canonical evaluation/audit implementations
  - `scripts/viz/`: canonical figure/QC implementations
- `configs/`: Hydra configs grouped by domain (`data`, `model`, `training`, etc.)
- `tests/`: automated regression and smoke tests
- `docs/`: project docs, references, and notes
- `outputs/`: generated run artifacts (local only)

## Notebook-First Workflow

1. Open top-level `StageBridge.ipynb`.
2. Use notebook controls to run preprocessing/mapping/training/eval.
3. Scripts provide versioned implementations for notebook actions.

## Directory Hygiene Rules

- Keep the repo root minimal; avoid dropping run notes or ad-hoc files at top-level.
- Place notes in `docs/notes/`.
- Keep generated files in `outputs/` or external data root; do not commit outputs.
- Keep machine-local settings/configs out of version control.
