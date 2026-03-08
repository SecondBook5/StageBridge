# CODEX_MASTER_EXECUTION.md

## Read this before touching the repo

You are not being asked to “improve the codebase.”
You are being asked to perform a controlled architectural rewrite of the StageBridge repository so that:

1. the repository structure matches the corrected scientific model exactly
2. the repository becomes easy for a human to navigate by scientific concept
3. stale exploratory structure is removed from the active tree
4. the results system becomes formal, reproducible, and rollback-safe
5. the active codebase supports the corrected v1 model only
6. future model development happens on top of a coherent method platform rather than on top of historical clutter

Do not creatively reinterpret the scope.
Do not preserve duplicate pathways out of caution.
Do not leave old wrappers in place “just in case.”
Do not keep dead files in the active repo because git history already preserves them.

If something no longer serves the active architecture, remove it from the active branch.

---

# 0. Active package and naming rules

## 0.1 Active Python package

The active Python package root is:

- `stagebridge/`

Do not create a second active package root under `src/`.
If `src/` is currently empty, stale, or redundant, remove it from the active tree.
Do not split package logic across both `stagebridge/` and `src/`.

## 0.2 Dataset naming

Use these names in the active repo:

- `luad_evo` for the Peng/Kadara LUAD evolution dataset
- `brainmets` for the secondary brain metastasis dataset

Do not use:
- `peng_kadara` as the primary active folder name
- `rossi_brainmets`
- `dataset_prep`
- `provider`

The biological program is more important than the publication nickname for active repo navigation.

## 0.3 Architectural naming

Use these names exactly:

- `data`
- `reference`
- `spatial_mapping`
- `context_model`
- `transition_model`
- `evaluation`
- `results`
- `pipelines`
- `viz`
- `utils`

Do not use vague section names like:
- `provider`
- `legacy`
- `workflow`
- `analysis` when the contents are actually evaluation
- `prep` when the contents are actually dataset-specific readers
- `topoflow` as an active architecture namespace

---

# 1. Non-negotiable v1 scientific definition

## 1.1 v1 problem

The repository exists to support this question:

**Which within-lung LUAD initiation stage transitions are niche-gated, and how is that gating modulated by evolutionary state?**

## 1.2 v1 stages

The active within-lung disease ladder is:

- Normal
- AAH
- AIS
- MIA
- LUAD

## 1.3 v1 modalities

The flagship v1 path uses:

- snRNA-seq
- Visium spatial transcriptomics
- WES

## 1.4 v1 exclusions

Do not implement or imply any of the following in the active v1 path:

- no full continuous Normal -> BrainMets progression claim
- no TCR conditioning
- no requirement that brainmets is part of the first complete model
- no claim of zero batch effects
- no assumption that graph propagation is automatically beneficial
- no unrestricted learned genomics conditioning in v1
- no giant “all modalities always required” design

## 1.5 v1 architectural statement

The active model is:

**a reference-anchored, spatially grounded, edge-wise stochastic transition framework with typed niche context and WES regularization**

It is:

- spatially grounded
- dynamical
- bridge-based
- tissue-interpretable
- not merely a latent interpolation model

The transition layer must be implemented as a **learned stochastic drift-diffusion operator**, not as a single opaque transition block.

State-dependent diffusion is a design target now, not a future wish-list item.

---

# 2. Scientific layers the repo must mirror

The repository must be organized around these scientific layers:

1. data ingestion
2. reference latent mapping
3. spatial mapping
4. typed niche context modeling
5. edge-wise stochastic transition modeling
6. tissue-level interpretation and evaluation
7. results tracking

These layers must be visible directly in the file structure.

Every concept gets exactly one home.

Examples:

- Tangram logic belongs in `stagebridge/spatial_mapping/tangram_mapper.py`
- TACCO logic belongs in `stagebridge/spatial_mapping/tacco_mapper.py`
- DestVI logic belongs in `stagebridge/spatial_mapping/destvi_mapper.py`
- HLCA logic belongs in `stagebridge/reference/hlca_mapper.py`
- Set Transformer logic belongs in `stagebridge/context_model/set_encoder.py`
- Graph Transformer logic belongs in `stagebridge/context_model/graph_encoder.py`
- Graph-of-Sets integration belongs in `stagebridge/context_model/graph_of_sets.py`
- Gaussian bridge initialization belongs in `stagebridge/transition_model/gaussian_init.py`
- drift network belongs in `stagebridge/transition_model/drift_network.py`
- diffusion network belongs in `stagebridge/transition_model/diffusion_network.py`
- Schrödinger bridge logic belongs in `stagebridge/transition_model/schrodinger_bridge.py`
- WES regularization belongs in `stagebridge/transition_model/wes_regularizer.py`

Notebook code may call these modules.
Notebook code may not duplicate them.

---

# 3. High-level repo surgery rules

## 3.1 Archive via git, not via folders

Before destructive cleanup:

1. create git tag `archive/pre_stagebridge_rebuild`
2. create git branch `archive/pre_stagebridge_rebuild`

Do not create a `legacy/` folder.
Do not preserve stale active code inside the live repo tree.
Git is the archive.

## 3.2 Keep exactly one active top-level notebook

Keep:

- `StageBridge.ipynb`

Delete from the active tree:

- `StageBridge_TopoFlow.ipynb`

Any other notebooks that are only historical, debug, or duplicated workflow surfaces should be removed from the active root.
If any notebook is retained for narrow utility, it must not behave like a second primary interface.

## 3.3 Keep exactly one orchestration namespace

Collapse all current orchestration duplication into:

- `stagebridge/pipelines/`

Do not leave active duplicates across:
- `stagebridge/pipeline/`
- `stagebridge/pipelines/`
- `stagebridge/workflows/`

After migration, only `stagebridge/pipelines/` should remain active.

## 3.4 Delete clutter aggressively

The active tree should not contain:

- stale log forests
- loose dated output directories
- loose `history_*.json` files
- duplicated script wrappers
- poster-only artifact dumps
- dead checkpoint graveyards
- topoflow-specific abandoned namespaces
- old notebook variants
- empty or misleading package roots

If an artifact is important, move it into the formal `results/` system.
If it is not important, delete it from the active branch.

---

# 4. Target repository structure

The active repository must converge to this structure.

```text
StageBridge/
├── README.md
├── AGENTS.md
├── CLAUDE.md
├── REPO_REBUILD_DIRECTIVE.md
├── CODEX_MASTER_EXECUTION.md
├── pyproject.toml
├── environment.yml
├── StageBridge.ipynb
├── .gitignore
│
├── configs/
│   ├── default.yaml
│   ├── data/
│   │   ├── luad_evo.yaml
│   │   ├── brainmets.yaml
│   │   └── local.yaml
│   ├── spatial_mapping/
│   │   ├── tangram.yaml
│   │   ├── tacco.yaml
│   │   └── destvi.yaml
│   ├── context_model/
│   │   ├── set_only.yaml
│   │   └── graph_of_sets.yaml
│   ├── transition_model/
│   │   ├── disease_edges.yaml
│   │   ├── gaussian_init.yaml
│   │   ├── stochastic_dynamics.yaml
│   │   ├── schrodinger_bridge.yaml
│   │   └── wes_regularizer.yaml
│   ├── splits/
│   │   └── donor_holdout.yaml
│   ├── train/
│   │   ├── smoke.yaml
│   │   └── full_v1.yaml
│   └── evaluation/
│       ├── baseline.yaml
│       └── ablation.yaml
│
├── docs/
│   ├── specs/
│   │   ├── 000_master_spec.md
│   │   ├── 001_repo_contract.md
│   │   ├── 002_data_contract.md
│   │   ├── 003_reference_contract.md
│   │   ├── 004_spatial_mapping_contract.md
│   │   ├── 005_context_model_contract.md
│   │   ├── 006_transition_model_contract.md
│   │   ├── 007_evaluation_contract.md
│   │   ├── 008_results_contract.md
│   │   └── 009_notebook_contract.md
│   ├── decisions/
│   │   ├── ADR_001_lung_only_v1.md
│   │   ├── ADR_002_wes_as_regularizer.md
│   │   ├── ADR_003_graph_must_earn_it.md
│   │   ├── ADR_004_single_notebook_frontend.md
│   │   └── ADR_005_git_is_rollback.md
│   ├── biology/
│   └── papers/
│
├── stagebridge/
│   ├── __init__.py
│   ├── config.py
│   ├── notebook_api.py
│   │
│   ├── data/
│   │   ├── common/
│   │   │   ├── manifests.py
│   │   │   ├── paths.py
│   │   │   ├── schema.py
│   │   │   ├── h5ad_atomic.py
│   │   │   └── harmonize.py
│   │   ├── luad_evo/
│   │   │   ├── snrna.py
│   │   │   ├── visium.py
│   │   │   ├── wes.py
│   │   │   ├── metadata.py
│   │   │   └── stages.py
│   │   └── brainmets/
│   │       ├── snrna.py
│   │       ├── spatial.py
│   │       ├── lpwgs.py
│   │       └── metadata.py
│   │
│   ├── reference/
│   │   ├── hlca_mapper.py
│   │   ├── diagnostics.py
│   │   ├── latent_store.py
│   │   └── label_transfer.py
│   │
│   ├── spatial_mapping/
│   │   ├── base.py
│   │   ├── tangram_mapper.py
│   │   ├── tacco_mapper.py
│   │   ├── destvi_mapper.py
│   │   ├── outputs.py
│   │   └── qc.py
│   │
│   ├── context_model/
│   │   ├── token_schema.py
│   │   ├── token_builder.py
│   │   ├── cell_to_spot_assignment.py
│   │   ├── set_encoder.py
│   │   ├── graph_builder.py
│   │   ├── graph_encoder.py
│   │   ├── graph_of_sets.py
│   │   └── context_outputs.py
│   │
│   ├── transition_model/
│   │   ├── disease_edges.py
│   │   ├── couplings.py
│   │   ├── gaussian_init.py
│   │   ├── drift_network.py
│   │   ├── diffusion_network.py
│   │   ├── stochastic_dynamics.py
│   │   ├── schrodinger_bridge.py
│   │   ├── losses.py
│   │   ├── wes_regularizer.py
│   │   ├── baselines.py
│   │   ├── train.py
│   │   └── infer.py
│   │
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── calibration.py
│   │   ├── ablations.py
│   │   ├── context_sensitivity.py
│   │   ├── trajectory_analysis.py
│   │   ├── fixed_points.py
│   │   ├── niche_regimes.py
│   │   ├── pseudotime_structure.py
│   │   ├── gene_attribution.py
│   │   └── reports.py
│   │
│   ├── results/
│   │   ├── registry.py
│   │   ├── run_writer.py
│   │   ├── milestone.py
│   │   ├── result_card.py
│   │   └── manifest.py
│   │
│   ├── pipelines/
│   │   ├── run_reference.py
│   │   ├── run_spatial_mapping.py
│   │   ├── run_context_model.py
│   │   ├── run_transition_model.py
│   │   ├── run_evaluation.py
│   │   └── run_full.py
│   │
│   ├── viz/
│   │   ├── curves.py
│   │   ├── embeddings.py
│   │   ├── flows.py
│   │   ├── spatial.py
│   │   └── summary_panels.py
│   │
│   └── utils/
│       ├── artifacts.py
│       ├── checks.py
│       ├── config_loader.py
│       ├── seeds.py
│       └── types.py
│
├── tests/
│
├── results/
│   ├── registry/
│   │   ├── results_registry.csv
│   │   ├── milestone_index.csv
│   │   └── promoted_results.yaml
│   ├── runs/
│   ├── milestones/
│   ├── summaries/
│   └── figures/
│
└── outputs/
    └── scratch/
````

---

# 5. Exact current-to-target migration map

Perform these moves, renames, and deletions explicitly.

## 5.1 Notebook migration

Keep:

* `StageBridge.ipynb`

Delete:

* `StageBridge_TopoFlow.ipynb`

Remove any other active notebook surface that duplicates the main workflow.

## 5.2 Config migration

Normalize current config sprawl into the target tree.

Current concepts that must be absorbed:

* `configs/topoflow/`
* `configs/hlca/`
* `configs/tangram/`
* `configs/model/`
* `configs/training/`
* `configs/eval.yaml`
* `configs/train.yaml`
* `configs/experiment/`

Required normalized targets:

* `configs/data/luad_evo.yaml`
* `configs/data/brainmets.yaml`
* `configs/spatial_mapping/tangram.yaml`
* `configs/spatial_mapping/tacco.yaml`
* `configs/spatial_mapping/destvi.yaml`
* `configs/context_model/set_only.yaml`
* `configs/context_model/graph_of_sets.yaml`
* `configs/transition_model/disease_edges.yaml`
* `configs/transition_model/gaussian_init.yaml`
* `configs/transition_model/stochastic_dynamics.yaml`
* `configs/transition_model/schrodinger_bridge.yaml`
* `configs/transition_model/wes_regularizer.yaml`
* `configs/splits/donor_holdout.yaml`
* `configs/train/smoke.yaml`
* `configs/train/full_v1.yaml`
* `configs/evaluation/baseline.yaml`
* `configs/evaluation/ablation.yaml`

Rules:

* no duplicate global train/eval YAMLs once scoped configs exist
* no topoflow-only config subtree in active v1
* no ambiguous root config files that duplicate more specific scoped configs

## 5.3 Data layer migration

Move dataset-specific readers out of `stagebridge/io/` into `stagebridge/data/...`

Examples:

* `stagebridge/io/geo_snrna.py` -> `stagebridge/data/luad_evo/snrna.py`
* `stagebridge/io/geo_spatial.py` -> `stagebridge/data/luad_evo/visium.py`
* `stagebridge/io/geo_wes.py` -> `stagebridge/data/luad_evo/wes.py`
* `stagebridge/io/geo_brainmets.py` -> split into:

  * `stagebridge/data/brainmets/snrna.py`
  * `stagebridge/data/brainmets/spatial.py`
  * `stagebridge/data/brainmets/lpwgs.py`
* `stagebridge/io/manifests.py` -> `stagebridge/data/common/manifests.py`
* `stagebridge/io/paths.py` -> `stagebridge/data/common/paths.py`
* `stagebridge/io/h5ad_atomic.py` -> `stagebridge/data/common/h5ad_atomic.py`

Also create or migrate:

* `stagebridge/data/common/schema.py`
* `stagebridge/data/common/harmonize.py`
* `stagebridge/data/luad_evo/metadata.py`
* `stagebridge/data/luad_evo/stages.py`
* `stagebridge/data/brainmets/metadata.py`

## 5.4 Reference layer migration

Move all HLCA and latent mapping logic under `stagebridge/reference/`

Examples:

* `stagebridge/io/hlca.py` -> `stagebridge/reference/hlca_mapper.py`

Create or migrate:

* `stagebridge/reference/diagnostics.py`
* `stagebridge/reference/latent_store.py`
* `stagebridge/reference/label_transfer.py`

## 5.5 Spatial mapping migration

Create a named spatial mapping layer.

Move:

* `stagebridge/io/tangram.py` -> `stagebridge/spatial_mapping/tangram_mapper.py`

Create:

* `stagebridge/spatial_mapping/base.py`
* `stagebridge/spatial_mapping/tacco_mapper.py`
* `stagebridge/spatial_mapping/destvi_mapper.py`
* `stagebridge/spatial_mapping/outputs.py`
* `stagebridge/spatial_mapping/qc.py`

Even if Tangram is the only fully implemented method at first, the interface for Tangram, TACCO, and DestVI must exist now.

## 5.6 Context model migration

Unify all niche token, spot assignment, set encoder, graph builder, graph encoder, and graph-of-sets logic under `stagebridge/context_model/`

Examples:

* `stagebridge/models/graph_of_sets.py` -> `stagebridge/context_model/graph_of_sets.py`
* `stagebridge/preprocessing/spatial_graph.py` -> `stagebridge/context_model/graph_builder.py`
* `stagebridge/preprocessing/spatial_niche.py` -> split into:

  * `stagebridge/context_model/token_builder.py`
  * `stagebridge/context_model/cell_to_spot_assignment.py`
* `stagebridge/io/niche_tokens.py` -> split into:

  * `token_schema.py`
  * `token_builder.py`
  * `context_outputs.py`

Create or migrate:

* `token_schema.py`
* `token_builder.py`
* `cell_to_spot_assignment.py`
* `set_encoder.py`
* `graph_builder.py`
* `graph_encoder.py`
* `graph_of_sets.py`
* `context_outputs.py`

## 5.7 Transition model migration

Unify all transition learning under `stagebridge/transition_model/`

Examples:

* `stagebridge/training/schrodinger_bridge.py` -> `stagebridge/transition_model/schrodinger_bridge.py`
* `stagebridge/training/losses.py` -> `stagebridge/transition_model/losses.py`
* `stagebridge/models/stagebridge.py` -> refactor into:

  * `drift_network.py`
  * `diffusion_network.py`
  * `stochastic_dynamics.py`
  * `train.py`
  * `infer.py`
* `stagebridge/models/baselines.py` -> `stagebridge/transition_model/baselines.py`
* `stagebridge/models/genomic_niche.py` -> `stagebridge/transition_model/wes_regularizer.py`

Create:

* `disease_edges.py`
* `couplings.py`
* `gaussian_init.py`
* `drift_network.py`
* `diffusion_network.py`
* `stochastic_dynamics.py`
* `schrodinger_bridge.py`
* `losses.py`
* `wes_regularizer.py`
* `baselines.py`
* `train.py`
* `infer.py`

## 5.8 Evaluation migration

Move all scientific interpretation of model outputs under `stagebridge/evaluation/`

Examples:

* `stagebridge/analysis/context_sensitivity.py` -> `stagebridge/evaluation/context_sensitivity.py`
* `stagebridge/analysis/gene_attribution.py` -> `stagebridge/evaluation/gene_attribution.py`
* `stagebridge/analysis/trajectory.py` -> `stagebridge/evaluation/trajectory_analysis.py`

Create:

* `metrics.py`
* `calibration.py`
* `ablations.py`
* `fixed_points.py`
* `niche_regimes.py`
* `pseudotime_structure.py`
* `reports.py`

## 5.9 Pipeline migration

Collapse all orchestration duplication into one namespace:

Keep only:

* `stagebridge/pipelines/run_reference.py`
* `stagebridge/pipelines/run_spatial_mapping.py`
* `stagebridge/pipelines/run_context_model.py`
* `stagebridge/pipelines/run_transition_model.py`
* `stagebridge/pipelines/run_evaluation.py`
* `stagebridge/pipelines/run_full.py`

Delete active duplication across:

* `stagebridge/pipeline/`
* `stagebridge/pipelines/`
* `stagebridge/workflows/`

## 5.10 Results migration

Current ad hoc artifacts spread across:

* `outputs/`
* loose `history_*.json`
* loose figures
* loose checkpoints
* side experiment subtrees

must be normalized into the formal `results/` system.

Keep only:

* `results/registry/`
* `results/runs/`
* `results/milestones/`
* `results/summaries/`
* `results/figures/`

Everything transient goes in:

* `outputs/scratch/`

---

# 6. Things to delete from the active repo

Delete from the active tree unless explicitly migrated into the new structure:

* `StageBridge_TopoFlow.ipynb`
* `configs/topoflow/`
* duplicated wrappers across `scripts/`, `scripts/pipeline/`, `scripts/eval/`, `scripts/train/`, `scripts/viz/`
* loose dated `outputs/2026-*` folders once important material is migrated or discarded
* loose `history_*.json` files not registered as formal runs
* loose figure dumps not attached to milestone-worthy runs
* poster-only artifact subtrees if not part of active v1 execution
* stale checkpoints not linked to registered run directories
* random note files outside specs, ADRs, biology docs, and papers
* empty or redundant `src/`
* `stagebridge.egg-info/`
* any duplicate implementation path for the same concept

Do not leave dead compatibility wrappers in the active tree.

---

# 7. Mission structure

You must complete the rewrite in three missions.
Do not do all three at once.
Stop and report after each mission.

---

# Mission 1: Repository surgery

## Objective

Restructure the repo so the file tree reflects the corrected scientific model and remove stale active pathways.

## Do this now

1. create git tag `archive/pre_stagebridge_rebuild`
2. create git branch `archive/pre_stagebridge_rebuild`
3. create the target directory structure
4. move and rename files according to the migration map
5. remove stale duplicated wrappers, notebooks, and dead paths
6. normalize the config tree
7. create stub files required by the target structure
8. make the active package importable after the move
9. ensure `StageBridge.ipynb` remains the only active top-level notebook
10. ensure `stagebridge/pipelines/` is the only active pipeline namespace

## Do not do yet

* do not implement new model behavior
* do not implement the results registry yet
* do not refactor training logic deeply beyond what is required for the new structure
* do not run large experiments
* do not start scientific benchmarking

## Mission 1 acceptance criteria

Mission 1 is complete only if:

* the repo tree matches the target structure closely
* every scientific concept has one obvious home
* there is no `legacy/` folder
* `StageBridge.ipynb` is the only active top-level notebook
* `stagebridge/pipelines/` is the only active orchestration namespace
* duplicate script wrappers are removed
* import paths are coherent enough for the package to import
* old topoflow-specific structure is gone from active architecture

## Required report at end of Mission 1

Return:

* a concise tree of the rebuilt repo
* a table of moved files
* a table of deleted files and folders
* unresolved issues if any
* explicit yes/no on Mission 1 acceptance criteria

Then stop.

---

# Mission 2: Results system and notebook contract

## Objective

Make the repository reproducible and rollback-safe before further model implementation.

## Build exactly these capabilities

### 2.1 Formal run writer

Implement:

* `stagebridge/results/run_writer.py`
* `stagebridge/results/manifest.py`
* `stagebridge/results/result_card.py`

Every run must create:

* `results/runs/<run_id>/resolved_config.yaml`
* `results/runs/<run_id>/run_metadata.json`
* `results/runs/<run_id>/stdout.log`
* `results/runs/<run_id>/artifacts/`
* `results/runs/<run_id>/metrics.json`
* `results/runs/<run_id>/result_card.md`

### 2.2 Registry

Implement:

* `stagebridge/results/registry.py`

It must maintain:

* `results/registry/results_registry.csv`
* `results/registry/milestone_index.csv`
* `results/registry/promoted_results.yaml`

### 2.3 Milestone promotion

Implement:

* `stagebridge/results/milestone.py`

Promoted milestones must create:

* `results/milestones/<milestone_id>/milestone_summary.md`
* links or copies to source artifacts
* metrics snapshot
* interpretation notes
* next-step recommendation

### 2.4 Notebook contract

Refactor `StageBridge.ipynb` so it only:

* loads config
* calls package pipeline functions
* displays outputs
* writes runs through the formal results system
* optionally promotes milestones

It may not:

* define model classes
* define training loops
* define hidden business logic
* bypass the results registry

### 2.5 Tests

Implement at minimum:

* config loading test
* run directory creation test
* registry update test
* milestone promotion test
* notebook contract test

## Mission 2 acceptance criteria

Mission 2 is complete only if:

* a smoke execution can write a formal run directory
* registry files are created and updated
* milestone promotion works
* `StageBridge.ipynb` calls package functions rather than inline implementation logic
* results-system tests pass

## Required report at end of Mission 2

Return:

* list of implemented results-system files
* one example smoke run directory tree
* contents or schema of registry files
* notebook contract summary
* test results
* explicit yes/no on Mission 2 acceptance criteria

Then stop.

---

# Mission 3: Build the corrected v1 model framework

## Objective

Implement the corrected v1 scientific architecture in executable form.

## Build in this order only

### 3.1 Data layer

Implement clean readers and contracts for:

* LUAD evolution snRNA
* LUAD evolution Visium
* LUAD evolution WES

Keep brainmets readers only as secondary utilities, not active v1 path.

Required files:

* `stagebridge/data/luad_evo/snrna.py`
* `stagebridge/data/luad_evo/visium.py`
* `stagebridge/data/luad_evo/wes.py`
* `stagebridge/data/luad_evo/metadata.py`
* `stagebridge/data/luad_evo/stages.py`

### 3.2 Reference layer

Implement:

* HLCA mapping wrapper
* diagnostics for latent quality
* residual leakage checks
* latent persistence

Required outputs:

* reference latent embeddings
* stage label preservation diagnostics
* diagnostic artifact bundle

### 3.3 Spatial mapping layer

Implement Tangram first.

Required:

* a base interface for spatial mapping methods
* Tangram execution wrapper
* standardized output contract for spot-level composition and state outputs
* interface-compatible placeholders for TACCO and DestVI

Do not overbuild TACCO or DestVI yet.
Their interface must exist cleanly now even if their internals remain partial.

### 3.4 Typed context model

Implement three levels clearly.

#### Level A: typed token schema

Each Visium spot becomes a typed biological set with typed tokens such as:

* epithelial state tokens
* stromal state tokens
* immune state tokens
* vascular or program tokens

#### Level B: set-only context

Implement:

* token builder
* Set Transformer encoder
* context output object

This is the first serious spatial baseline.

#### Level C: graph-of-sets context

Implement:

* spatial graph builder
* Graph Transformer encoder
* combined graph-of-sets representation

This must be switchable and ablatable.

### 3.5 Transition model

Implement the transition model as:

* disease edge definitions
* OT coupling helpers
* Gaussian bridge initialization
* learned drift network
* learned diffusion network
* stochastic dynamics wrapper
* Schrödinger bridge objective
* WES regularizer
* baseline modes

Required baseline modes:

* RNA-only
* set-only context
* graph-of-sets context
* graph-of-sets + WES

Required edge support:

* Normal -> AAH
* AAH -> AIS
* AIS -> MIA
* MIA -> LUAD

Required initial biological emphasis:

* AAH -> AIS
* AIS -> MIA

### 3.6 Evaluation layer

Implement:

* held-out performance metrics
* calibration
* context sensitivity and niche shuffling
* trajectory analysis
* fixed point summaries
* niche regime summaries
* pseudotime structure summaries
* attribution summaries
* report generation

### 3.7 Full pipeline glue

Implement and wire:

* `run_reference.py`
* `run_spatial_mapping.py`
* `run_context_model.py`
* `run_transition_model.py`
* `run_evaluation.py`
* `run_full.py`

`StageBridge.ipynb` must call them in this order.

## Mission 3 acceptance criteria

Mission 3 is complete only if:

* RNA-only mode runs
* set-only context mode runs
* graph-of-sets mode runs
* WES regularization is wired in as an explicit regularizer
* AAH -> AIS and AIS -> MIA can be run end to end
* every run writes formal results
* ablation comparison is possible
* notebook orchestration works without hidden core logic

## Required report at end of Mission 3

Return:

* implemented modules
* remaining stubs
* exact runnable modes
* exact unimplemented pieces
* first recommended smoke experiment
* explicit yes/no on Mission 3 acceptance criteria

Then stop.

---

# 8. Scientific constraints that must shape implementation

## 8.1 What the biology demands

Do not treat spatial context as decoration.
The purpose of the context model is to test whether local epithelial–stromal–immune niche structure changes the transition operator.

Do not treat WES as an annotation sidecar.
The purpose of WES in v1 is to constrain biologically admissible transport.

Do not overclaim continuity.
The goal is edge-wise transition modeling inside the within-lung initiation ladder.

## 8.2 What the graph branch must prove

Graph-of-sets is not automatically the main model.
It is an ablation candidate until it materially beats set-only.

The repo must make comparison easy across:

* RNA-only
* set-only
* graph-of-sets
* graph-of-sets + WES

## 8.3 What outputs matter

The system is not done when it trains.
It is only useful when it produces:

* reproducible run artifacts
* interpretable result cards
* ablations
* tissue-level summaries
* milestone-ready outputs

---

# 9. Definition of failure

You have failed if any of the following remain true:

* the repo still has multiple homes for the same scientific concept
* the notebook still hides core logic
* results still live in ad hoc output folders
* graph-of-sets cannot be ablated cleanly
* WES is still only metadata
* stale wrappers and duplicate scripts still clutter the active tree
* the active repo still reads like an exploratory student project instead of a methods platform

---

# 10. Final command

Do Mission 1 now.
Do not start Mission 2 until Mission 1 acceptance criteria are satisfied and reported.
