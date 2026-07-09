# `cli/` — Thin Command-Line Entry Points

## Purpose

`cli/` provides thin command-line entry points to CCRT workflows. Each file in
`cli/` calls exactly one workflow and contains minimal logic — argument parsing and
dispatch, not business logic. The substance of every command lives in the packages
the CLI invokes (`adapters/`, `data/`, `training/`, `evaluation/`, `plotting/`, and
their siblings); the CLI merely wires configured inputs to the appropriate workflow
and returns. This thinness keeps the command surface easy to audit, ensures the
same workflow behaves identically whether launched from a CLI, a script in
`scripts/ccrt/`, or a test, and prevents logic from silently accumulating in
entry-point code where it would escape the architecture lock.

## Owns

- One entry point per workflow, each dispatching to a single workflow.
- Argument parsing and configuration wiring (from `configs/ccrt/`).
- Invocation and exit handling — nothing more.

## Does NOT

- Does NOT contain business logic, model definitions, or training/evaluation logic — it delegates to the owning packages.
- Does NOT bypass the standardized pipeline or open hidden side channels between stages.
- Does NOT hard-code `LUAD`, `PanIN`, `macrophage`, `CAF`, `cancer`, `virus`, or a malignancy axis.
- Does NOT write outputs into the source tree; workflows it launches emit only into allowed output roots (see `../../../docs/ccrt/DIRECTORY_PURPOSES.md`).
