# scripts/

These scripts are operational building blocks used by `StageBridge.ipynb`.

Root-level `scripts/*.py` files are compatibility wrappers. Canonical
implementations are organized in subdirectories:

## `scripts/pipeline/`

- data/mapping pipelines and orchestration
- includes: snRNA/spatial builds, HLCA, Tangram, feature-bank build, WES prep

## `scripts/train/`

- model training entrypoints

## `scripts/eval/`

- evaluation and environment/data audit entrypoints

## `scripts/viz/`

- figure and QC generation entrypoints

Notebook-first policy: keep invoking root-level `scripts/*.py` from notebook
cells; wrappers preserve the original command surface.

Implementation policy: business logic lives in the package workflow modules:
- `stagebridge/workflows/pipeline.py`
- `stagebridge/workflows/train.py`
- `stagebridge/workflows/eval.py`
