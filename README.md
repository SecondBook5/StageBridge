# StageBridge

**Hierarchical Transformer-Based Modeling of Lung Adenocarcinoma Stage Progression from Spatial and Single-Cell Transcriptomics**

StageBridge learns niche-conditioned cell-state transitions across the LUAD initiation ladder (Normal &rarr; AAH &rarr; AIS &rarr; MIA &rarr; LUAD) using population-level spatial context from Visium and snRNA-seq, with optional WES genomic conditioning.

---

## Architecture

The framework is built around two complementary transformer systems:

### Hierarchical Context Encoder

A three-level typed transformer that learns population context from spatial niches:

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

### EA-MIST (Lesion-Level Set Transformer)

For direct lesion-level classification (`AAH→AIS`, `AIS→MIA`):

1. **Local niche encoder** &mdash; small transformer over receiver, ring, LR/pathway, and neighborhood-stats tokens
2. **Prototype bottleneck** &mdash; compress local niches into recurrent motifs
3. **Lesion-level Set Transformer** &mdash; model each lesion as a bag of niche prototypes with WES/evolution conditioning

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

HLCA reference atlas alignment for latent space construction.

---

## Quick Start

```python
from stagebridge.notebook_api import compose_config
from stagebridge.pipelines import (
    run_pretrain_local, run_train_lesion,
    run_evaluate_lesion, run_eamist_reporting,
)

cfg = compose_config(overrides=["context_model=eamist"])
pretrain = run_pretrain_local(cfg)
cfg.context_model.eamist.pretrained_local_checkpoint = pretrain["best_checkpoint"]
train = run_train_lesion(cfg)
report = run_eamist_reporting(cfg)
```

```bash
# CLI entry points
python -m stagebridge.pipelines step pretrain_local -o context_model=eamist
python -m stagebridge.pipelines step train_lesion -o context_model=eamist
python -m stagebridge.pipelines step evaluate_lesion -o context_model=eamist
python -m stagebridge.pipelines step eamist_report -o context_model=eamist
```

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
├── context_model/             # hierarchical transformer, set encoder, GoST, prototypes
│   ├── hierarchical_transformer.py   # TypedHierarchicalTransformerEncoder
│   ├── set_encoder.py                # ISAB, SAB, PMA, FeedForwardBlock
│   ├── graph_of_sets.py              # GraphOfSetsTransformer
│   ├── prototype_bottleneck.py       # prototype compression layer
│   └── local_niche_encoder.py        # receiver-centered niche transformer
├── models/                    # StageBridgeModel, drift/diffusion networks
├── data/luad_evo/             # lesion bags, neighborhood features, splits
├── pipelines/                 # pretrain_local, train_lesion, evaluate_lesion
├── evaluation/                # metrics, calibration, EA-MIST metrics
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

- Donor-held-out cross-validation across all stage transitions
- Metrics: Sinkhorn distance, MMD-RBF, classifier AUC, direction cosine, calibration error
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
