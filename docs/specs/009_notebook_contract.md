# 009 — Notebook Contract

`StageBridge.ipynb` is the only active top-level notebook. Core logic belongs in the package, not in notebook cells.

## What the Notebook Is

An orchestration interface over the `stagebridge` package. It calls pipeline functions, displays outputs, and provides narrative context.

## Allowed

- Load configuration and set experiment parameters
- Call pipeline functions from `stagebridge/pipelines/`
- Display figures, tables, metrics, and diagnostics
- Write structured results via the results system
- Promote milestones
- Provide explanatory markdown narrative

## Forbidden

- Define model classes (belongs in `context_model/` or `transition_model/`)
- Define training loops (belongs in `pipelines/` or training utilities)
- Define loss functions (belongs in the package)
- Implement data processing beyond simple loading calls
- Bypass the results system with ad-hoc file writes
- Become a hidden second codebase
- Export functions that other code depends on

## Why This Contract Exists

Notebooks with core logic are untestable (pytest cannot run cells), unreviewable (ipynb diffs are unreadable), duplicative (diverge from package logic), and fragile (cell execution order creates hidden state).

## The Test

If you delete all code cells and rewrite them as one-line function calls to `stagebridge/pipelines/`, the system should work identically. Any cell that cannot be replaced by a single pipeline call is a candidate for extraction into the package.
