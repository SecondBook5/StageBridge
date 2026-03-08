# ADR 004 Single Notebook Frontend

Decision: `StageBridge.ipynb` is the only active top-level notebook interface.

## Context

Scientific projects often accumulate multiple notebooks: one for exploration, one for training, one for evaluation, one for visualization. This creates:

- Duplicated logic across notebooks
- Inconsistent state (different notebooks assume different data paths)
- Difficulty knowing which notebook produces which result
- Core logic that lives in notebooks instead of the package

## Decision

One notebook: `StageBridge.ipynb`. It is an orchestration interface that calls into the `stagebridge` package. All other notebooks are eliminated or absorbed.

## Rationale

- Single point of entry reduces confusion
- Forces core logic into the testable package
- Makes the notebook reviewable (one file to check)
- Pipeline functions in `stagebridge/pipelines/` can be called from notebook or CLI identically

## Consequences

- Any cell that cannot be replaced by a one-line call to `stagebridge/pipelines/` is a candidate for extraction into the package
- No model definitions, training loops, or loss functions in notebook cells
- Exploration and ad-hoc analysis should still use the package functions, not notebook-local code
