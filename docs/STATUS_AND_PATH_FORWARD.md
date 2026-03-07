# StageBridge Status And Path Forward

Last updated: March 6, 2026

## Current Position

- Core package, notebook API, CLI, and workflows are functioning.
- Test baseline is green (`70 passed`).
- Core benchmark pipeline components are implemented:
  - GEO snRNA + spatial interim build
  - HLCA mapping
  - Tangram projection
  - Training/evaluation workflows
- Tier-3 model features exist in model/trainer code (LR, spatial niche, Dirichlet, multihop) and are now wired into workflow variant handling.

## What Was Fixed In This Iteration

- Training variant parsing now supports all declared `full_benchmark` Tier-3 variants and ablations.
- Unknown variants/ablations now fail fast (no silent dropping).
- Evaluation config reconstruction now restores advanced architecture flags from checkpoint config.
- Evaluation split selection now derives fold index from checkpoint name (`*_fold{k}_*.pt`) when present.
- Sinkhorn iterations now respect configured `sinkhorn_iters`.
- Gradient accumulation now flushes remainder steps at epoch end.
- Workflow regression tests added for variant parsing + eval model-config restoration.
- Sampler donor/stage pools were refactored from on-device tensors to CPU index pools with on-demand batch transfer.
- Transition sampler now validates input shapes/donor subsets and fails fast on malformed large-run inputs.
- Trainer profiling was added (`training.profile_train_steps`) and now writes per-fold `profile_*.json` artifacts with step runtime/memory stats.
- Shared StageBridge config construction is now centralized in `stagebridge.utils.config_helpers.build_stagebridge_config_from_cfg` and used by both train/eval workflows.
- Regression coverage was expanded for sampler failure modes, donor split guardrails, and profiling artifact generation.

## Primary Risks Still Open

- Full-run profiling still needs baseline thresholds and acceptance gates (currently metrics are emitted but not yet enforced).
- Path/config resolution remains partially fragmented outside the train/eval helper path.
- Legacy stub pipeline package (`stagebridge/pipeline/*`) still exists alongside the production workflow stack.
- Benchmark/eval reporting still needs stronger paper-grade result tracking, versioning, and experiment metadata discipline.

## Delivery Roadmap

1. **Scale And Stability (next)**
   - Add profiler acceptance thresholds to fail CI when runtime/memory regress beyond tolerance.
   - Add optional sampler caching strategy for repeated donor/stage pools on very large folds.
   - Expand edge-case tests to cover sparse input matrices and extremely small donor cohorts.

2. **Architecture Hardening**
   - Move remaining path/config utility logic into one canonical module.
   - Remove/archive legacy stub pipeline surfaces.
   - Centralize shared train/eval config construction utilities for all workflow entrypoints (including CLI wrappers).

3. **Paper-Grade Reproducibility**
   - Standardize experiment registry (`run_id`, seed, config hash, code hash, dataset hash).
   - Add deterministic benchmark manifests for all headline variants.
   - Expand evaluation outputs to publication tables and statistical summaries.

4. **Generalizable Tooling**
   - Add adapter points for non-lung cohorts with configurable ontology + transition graph.
   - Add plugin interface for alternate mapping backends (beyond HLCA/Tangram defaults).
   - Add template config packs for new datasets/sites.

## Working Agreement

- Every merged feature should include:
  - one test for correctness
  - one clear config path
  - one explicit output artifact contract
- No silent fallback for benchmark-defining options.
- Unknown experiment options should fail fast with actionable errors.
