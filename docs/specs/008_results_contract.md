# 008 — Results Contract

Mission 1 creates the target results tree and package stubs. Formal run writing, registry updates, and milestone promotion are Mission 2 work. This document defines the contract that implementation must satisfy.

## What Every Run Must Save

### Run Directory

Each run creates `$STAGEBRIDGE_DATA_ROOT/runs/<run_id>/` containing:

| File | Description |
|------|-------------|
| `config_resolved.yaml` | Fully resolved configuration |
| `metrics.json` | Quantitative evaluation results |
| `result_card.json` | Structured summary |
| `ablation_comparison.json` | Ablation results (if applicable) |
| `tables/` | Additional structured outputs |
| `figures/` | Run-specific visualizations |

The `run_id` encodes timestamp + config hash + short description.

### Result Card

Structured JSON summarizing what the run did and found:

- `run_id`, `timestamp`, `git_commit`, `config_hash`
- `mode`: one of `rna_only`, `set_only`, `graph_of_sets`
- `edges_evaluated`: list of disease edges
- `metrics_summary`: key metrics
- `ablation_delta`: improvement/regression vs baseline (if applicable)
- `notes`: free text

Result cards are machine-readable and aggregatable.

## When a Run Becomes Milestone-Worthy

- New best on a primary metric
- Meaningful ablation result (e.g., GoST first outperforms set-only)
- Qualitative discovery (e.g., first evidence of niche gating)
- Significant negative result (e.g., WES regularization does not help)
- Completion of a major pipeline stage for the first time

Not every run is a milestone.

## What Promotion Means

1. **Git tag** — Descriptive tag on the associated commit (e.g., `milestone/set-only-aah-ais-v1`)
2. **Registry entry** — Added to `results/milestones.json`
3. **Summary note** — Brief description of significance

Promotion is deliberate, not automatic.

## Why Results Must Be Tied to Git State

A result without a code reference is unreproducible. Git commit ties ensure exact code can be checked out, configuration drift is detectable, and rollback is possible.

## Registry

`results/milestones.json` — ordered list of milestone entries, each with run_id, git commit, and summary. Append-only in normal operation.
