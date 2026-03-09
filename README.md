# StageBridge

**Dual-reference lesion-level modeling of early LUAD progression from spatial, single-cell, and evolution features**

StageBridge contains several research pipelines, but the current primary lesion model is **EA-MIST**:

**Evolution-Aware Multiple-Instance Set Transformer**

In its current form, EA-MIST is best described as:

**a dual-reference lesion-level Set Transformer for healthy-to-tumor niche displacement**

The prediction unit is a **lesion**, represented as a **set of local niche instances** derived from spatial neighborhoods and enriched with:

- receiver-centered niche summaries
- local ring composition features
- HLCA healthy-reference similarity features
- LuCA tumor-aware reference similarity features
- pathway/program summaries
- lesion-level evolution features from WES-derived tables

The primary supervised task is **5-stage lesion placement** across:

- `Normal`
- `AAH`
- `AIS`
- `MIA`
- `LUAD`

The secondary scalar head is a **weak stage-ordered displacement target**, not independent biological progression truth. Optional edge heads remain auxiliary and are only used where viability metadata supports them.

---

## Architecture

The repository contains multiple transformer systems. The active EA-MIST path is the lesion-level MIL pipeline below.

### EA-MIST V1

EA-MIST is a **multiple-instance learning** model over lesion bags:

1. **Parquet-first lesion bags**
   Each lesion is loaded from a prebuilt `eamist_bags.parquet` row rather than rebuilt implicitly during training.

2. **9-token local niche encoder**
   Every niche is encoded as:
   1. receiver token
   2. ring 1
   3. ring 2
   4. ring 3
   5. ring 4
   6. HLCA token
   7. LuCA token
   8. pathway token
   9. niche-stats token

3. **Optional prototype bottleneck**
   Compresses recurrent niche motifs into interpretable prototype structure.

4. **Lesion-level Set Transformer**
   Aggregates local niche embeddings across the lesion in a permutation-invariant way.

5. **Evolution branch**
   Fuses lesion-level evolution features into the lesion embedding when available.

6. **Multitask heads**
   - 5-way stage head
   - scalar weak displacement head
   - optional masked auxiliary edge heads

Supported lesion-level model families share the same bag contract:

- `pooled`
- `deep_sets`
- `lesion_set_transformer`
- `eamist_no_prototypes`
- `eamist`

### Dual Reference Design

EA-MIST uses two reference systems:

- **HLCA**: healthy lung anchor
- **LuCA extended atlas**: tumor-aware reference

Current LuCA usage is in a **shared token-composition space**, not direct latent-space lesion-to-atlas projection. After the `cell_type_tumor` fix, LuCA now provides finer malignant epithelial context than the previous broad `ann_fine` setup, but the README should be read honestly: this is **tumor-aware context**, not direct malignant-state ground truth.

### Other StageBridge Components

### Hierarchical Context Encoder

A separate three-level typed transformer that learns population context from spatial niches:

| Level | Component | Purpose |
|-------|-----------|---------|
| **L1** | Per-group Set Encoders (ISAB &rarr; SAB &rarr; PMA) | Summarize epithelial, stromal, immune, and vascular tokens independently |
| **L2** | Relation Token MLP | Encode pairwise inter-group interactions (6 relation tokens) |
| **L3** | Fusion Query Attention + dual SAB refinement | Cross-attend 8 learned queries over group summaries + relation tokens |

Key features:
- **Spatial relative position encoding** within ISAB for coordinate-aware attention
- **Confidence gating** via learned sigmoid gate on per-token mapping confidence
- **FiLM conditioning** on dataset and edge embeddings for multi-task flexibility
- **Residual context head** with two-layer MLP + skip connections
- **Token dropout** (3%) for regularization during training

Current config: `hidden_dim=192`, `num_heads=8`, `num_inducing_points=24`, `num_group_summary_tokens=4`, `num_fusion_queries=8`

### Transition Model (OT Flow Matching)

Population-context-conditioned Schr&ouml;dinger bridge for continuous cell-state trajectory prediction:
- Sinkhorn OT coupling for pseudo-pair construction
- FiLM-conditioned drift/diffusion networks `v_φ(x, t, c_s, s)`
- Euler integration for trajectory rollout

### Graph-of-Sets Transformer (GoST)

Optional inter-patient message passing via sparse graph attention:
- Edge types: stage-adjacent, same-patient cross-stage, same-stage cross-patient
- Per-edge-type learned bias with scatter-softmax

---

## Data

| Dataset | Modality | GEO Accession |
|---------|----------|---------------|
| snRNA-seq (early LUAD cohort) | Single-cell transcriptomics | GSE308103 |
| Visium spatial transcriptomics | 10x Visium | GSE307534 |
| Whole-exome sequencing | WES mutations | GSE307529 |
| Brain metastasis snRNA-seq | Single-cell (extension) | GSE223499 |

Spatial mapping via three providers (Tangram, TACCO, DestVI) with automated hybrid benchmark selection.

Reference assets used by the active EA-MIST pipeline:

- **HLCA** for healthy latent/reference summaries
- **LuCA extended atlas** for tumor-aware static reference summaries
- **WES-derived lesion evolution features** for lesion-level conditioning

Canonical real-data EA-MIST feature outputs live under the existing StageBridge data tree rooted at `STAGEBRIDGE_DATA_ROOT` or `/mnt/e/StageBridge_data`.

---

## Quick Start

### Canonical EA-MIST Flow

1. Build or refresh the canonical lesion bags
2. Train the lesion-level benchmark from those bags
3. Evaluate and report stage/displacement results

The trainer is **parquet-first** and expects a valid `eamist_bags.parquet` plus audit sidecar. It will reject stale bags with:

- the wrong ring schema
- stale LuCA selector metadata
- missing HLCA/LuCA typed tokens
- missing weak displacement target

### Python

```python
from stagebridge.notebook_api import compose_config
from stagebridge.pipelines import (
    run_train_lesion,
    run_evaluate_lesion,
    run_eamist_reporting,
)

cfg = compose_config(overrides=["context_model=eamist"])
train = run_train_lesion(cfg)
report = run_eamist_reporting(cfg)
```

### Build Bags

```bash
python -m stagebridge.data.luad_evo.build_eamist_bags \
  --niche-bank /mnt/e/StageBridge_data/processed/features/niche_token_bank.zarr \
  --niche-parquet /mnt/e/StageBridge_data/processed/features/niche_tokens_full.parquet \
  --hlca-features /mnt/e/StageBridge_data/processed/features/niche_hlca_features.parquet \
  --luca-features /mnt/e/StageBridge_data/processed/features/niche_luca_features.parquet \
  --evo-features /mnt/e/StageBridge_data/processed/features/lesion_evo_features.parquet \
  --out /mnt/e/StageBridge_data/processed/features/eamist_bags.parquet
```

### Train / Evaluate / Report

```bash
python -m stagebridge.pipelines step train_lesion -o context_model=eamist
python -m stagebridge.pipelines step evaluate_lesion -o context_model=eamist
python -m stagebridge.pipelines step eamist_report -o context_model=eamist
```

### Canonical Bag Contract

Each lesion row in `eamist_bags.parquet` contains:

- lesion metadata: `lesion_id`, `sample_id`, `donor_id`, `patient_id`, `stage_label`, `stage_index`
- niche structure: `niche_ids`, `receiver_features`, `ring_features`, `hlca_features`, `luca_features`, `pathway_features`, `niche_stats_features`
- lesion features: `evo_features`
- multitask targets: `displacement_target`, optional `edge_targets`, optional `edge_target_mask`

### Label Repair

Use the label-repair workflow before forcing a weak edge into donor-held-out benchmarking.

```bash
python -m stagebridge.pipelines step label_repair -o labels=repair
```

Current outputs are written under `reports/labels/` and include:

- refined lesion labels
- continuous progression-risk scores
- donor-held-out viability diagnostics
- target recommendation report
- figures and tables for label support

---

## Repository Structure

```text
stagebridge/
├── context_model/             # lesion MIL heads, set encoders, prototypes, legacy context models
│   ├── hierarchical_transformer.py   # TypedHierarchicalTransformerEncoder
│   ├── set_encoder.py                # ISAB, SAB, PMA, FeedForwardBlock
│   ├── graph_of_sets.py              # GraphOfSetsTransformer
│   ├── prototype_bottleneck.py       # prototype compression layer
│   └── local_niche_encoder.py        # receiver-centered niche transformer
├── models/                    # StageBridgeModel, drift/diffusion networks
├── data/luad_evo/             # lesion bags, neighborhood features, LuCA/HLCA/WES preprocessing
├── pipelines/                 # train_lesion, evaluate_lesion, reporting, label repair
├── evaluation/                # stage/displacement metrics, calibration, legacy metrics
├── transition_model/          # OT coupling, Schrödinger bridge, flow matching
├── reference/                 # HLCA alignment, latent construction
├── spatial_mapping/           # Tangram / TACCO / DestVI wrappers
├── viz/                       # research frontend, embedding, spatial, poster figures
│   ├── research_frontend.py          # notebook-facing multi-panel figures
│   ├── eamist_figures.py             # EA-MIST publication figures
│   └── summary_panels.py            # poster assembly
└── utils/                     # config, h5ad I/O, seeds, types
configs/
├── context_model/             # typed_hierarchical_transformer.yaml, eamist.yaml
├── train/                     # training profiles
└── evaluation/                # evaluation configs
StageBridge.ipynb              # research notebook (12-step protocol)
tests/                         # architecture and pipeline tests
```

---

## Visualization

The research frontend generates publication-quality multi-panel figures:

- **Multi-embedding views**: PCA (with explained variance %), UMAP, t-SNE, PHATE
- **Spatial transcriptomics**: Visium spot maps colored by stage, cell-type composition, gene expression
- **Transformer diagnostics**: fusion attention heatmaps, per-group token profiles, relation scores
- **WES landscape**: TMB by stage, oncoprint, mutation frequency, stage-stratified mutation profiles
- **Transition dynamics**: source/predicted/target PCA, training curves, macroflow heatmaps
- **Provider comparison**: side-by-side Tangram/TACCO/DestVI winner maps and QC profiles

---

## Evaluation

### EA-MIST

- Donor-held-out lesion folds
- Primary metrics:
  - stage macro-F1
  - stage balanced accuracy
  - confusion matrices and per-stage support
  - displacement MAE
  - displacement Spearman correlation
  - stage-wise displacement monotonicity
- Optional auxiliary edge AUROC/AUPRC only where viability masks allow them

### Important Interpretation Note

EA-MIST is currently a disciplined lesion-level stage/displacement model. It should **not** be described as proof of true biological progression mechanics.

- The stage head is the primary supervised task.
- The displacement target is **weak stage-ordered supervision** derived from stage order.
- `Normal` and `MIA` are low-support tail classes and should be interpreted cautiously.

### Other StageBridge Pipelines

- Sinkhorn distance, MMD-RBF, classifier AUC, direction cosine, calibration error
- Context sensitivity analysis (real vs. shuffled context delta)
- Biological readout: gene-context correlations, typed niche shift profiles

---

## Testing

```bash
# Full suite
pytest -q tests/

# EA-MIST focused
pytest -q tests/test_eamist_data.py tests/test_eamist_model.py tests/test_eamist_pipelines.py
```

---

## Environment

- Python 3.11, PyTorch, CUDA 12.x
- `micromamba env create -f environment.yml`
- External data: set `STAGEBRIDGE_DATA_ROOT` environment variable

## License

MIT
