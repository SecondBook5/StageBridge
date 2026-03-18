# StageBridge Pipelines

This directory contains all pipeline scripts for the StageBridge V1 workflow.

## Pipeline Flow

```
                                    PREPARATION
                                    ===========

┌─────────────────────────────────────────────────────────────────────────────┐
│  download_references.py                                                      │
│  Download HLCA + LuCA reference atlases                                      │
│  Output: references/hlca/hlca_core.h5ad, references/luca/luca_luad.h5ad     │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  run_data_prep.py                                                            │
│  Step 0: QC and merge raw GEO data (snRNA, spatial, WES)                    │
│  Output: processed/luad_evo/snrna_qc_normalized.h5ad                        │
│          processed/luad_evo/spatial_qc_normalized.h5ad                      │
│          processed/luad_evo/wes_features.parquet                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
                                REFERENCE MAPPING
                                ================

┌─────────────────────────────────────────────────────────────────────────────┐
│  run_reference.py                                                            │
│  Map query cells to HLCA (healthy) + LuCA (cancer) reference spaces         │
│  Output: z_hlca, z_luca, z_fused embeddings per cell                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
                             SPATIAL DECONVOLUTION
                             ====================

┌─────────────────────────────────────────────────────────────────────────────┐
│  run_spatial_benchmark.py                                                    │
│  Compare Tangram vs DestVI vs TACCO, select canonical backend               │
│  Output: spatial_benchmark/comparison_report.json                           │
│          spatial_benchmark/{tangram,destvi,tacco}/cell_type_proportions.parquet │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  complete_data_prep.py                                                       │
│  Build canonical training format using spatial backend results              │
│  Output: canonical/cells.parquet                                            │
│          canonical/neighborhoods.parquet (9-token structure)                │
│          canonical/stage_edges.parquet                                      │
│          canonical/split_manifest.json                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
                                   TRAINING
                                   ========

┌─────────────────────────────────────────────────────────────────────────────┐
│  run_v1_complete.py        <-- RECOMMENDED: Runs BOTH datasets              │
│  Full pipeline: SSL pretraining + transition training                       │
│  Runs on: Semi-synthetic (with ground truth) + Real LUAD data              │
│  Output: runs/v1_complete/weights/, figures/, results.json                  │
└─────────────────────────────────────────────────────────────────────────────┘


                              ALTERNATIVE TRAINING
                              ====================

┌─────────────────────────────┐    ┌─────────────────────────────┐
│  run_v1_synthetic.py        │    │  run_v1_full.py             │
│  Synthetic data ONLY        │    │  Real data ONLY             │
│  For testing/validation     │    │  Production components      │
└─────────────────────────────┘    └─────────────────────────────┘

┌─────────────────────────────┐    ┌─────────────────────────────┐
│  run_full.py                │    │  pretrain_local.py          │
│  OmegaConf-based orchestrator│   │  Local SSL pretraining only │
│  Calls: run_reference →     │    │  For development/debugging  │
│         run_spatial_mapping →│    └─────────────────────────────┘
│         run_context_model → │
│         run_transition_model│
└─────────────────────────────┘
```

## Quick Reference

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `download_references.py` | Get HLCA/LuCA atlases | Once, before first run |
| `run_data_prep.py` | QC/merge raw data | After downloading GEO data |
| `run_reference.py` | HLCA/LuCA mapping | After data prep |
| `run_spatial_benchmark.py` | Compare spatial backends | After reference mapping |
| `run_spatial_mapping.py` | Run single spatial backend | If you know which backend |
| `complete_data_prep.py` | Build canonical format | After spatial benchmark |
| **`run_v1_complete.py`** | **Full training (both datasets)** | **Production runs** |
| `run_v1_full.py` | Training (real data only) | Real data only |
| `run_v1_synthetic.py` | Training (synthetic only) | Testing/validation |
| `run_full.py` | OmegaConf orchestrator | Config-driven runs |

## HPC Execution Order

```bash
# 1. Download references (once)
python -m stagebridge.pipelines.download_references --output_dir $DATA/references --all

# 2. Data preparation
python -m stagebridge.pipelines.run_data_prep --data-root $DATA

# 3. Reference mapping
python -m stagebridge.pipelines.run_reference --data-root $DATA

# 4. Spatial backend benchmark
python -m stagebridge.pipelines.run_spatial_benchmark \
    --snrna $DATA/processed/luad_evo/snrna_qc_normalized.h5ad \
    --spatial $DATA/processed/luad_evo/spatial_qc_normalized.h5ad \
    --output_dir $DATA/processed/luad_evo/spatial_benchmark

# 5. Complete data prep (canonical format)
python -m stagebridge.pipelines.complete_data_prep \
    --snrna $DATA/processed/luad_evo/snrna_qc_normalized.h5ad \
    --spatial $DATA/processed/luad_evo/spatial_qc_normalized.h5ad \
    --wes $DATA/processed/luad_evo/wes_features.parquet \
    --spatial_backend_dir $DATA/processed/luad_evo/spatial_benchmark \
    --output_dir $DATA/processed/luad_evo/canonical

# 6. Training (both semi-synthetic + real)
python -m stagebridge.pipelines.run_v1_complete \
    --data_dir $DATA/processed/luad_evo/canonical \
    --output_dir $DATA/runs/v1_complete
```

## Other Pipelines

| Script | Purpose |
|--------|---------|
| `run_evaluation.py` | Evaluate trained models |
| `run_ablations.py` | Run ablation experiments |
| `run_transition_model.py` | Train transition model only |
| `run_context_model.py` | Train context model only |
| `run_communication_benchmark.py` | Benchmark L/R communication |
| `run_label_repair.py` | Fix/harmonize stage labels |
| `evaluate_lesion.py` | Lesion-specific evaluation |
| `train_lesion.py` | Lesion-specific training |
| `run_story_reporting.py` | Generate narrative reports |
| `run_eamist_reporting.py` | EAMIST-specific reporting |

## Key Distinctions

### `run_data_prep.py` vs `complete_data_prep.py`
- `run_data_prep.py`: Raw data QC and merge (Step 0)
- `complete_data_prep.py`: Build canonical parquet format (requires spatial backend results)

### `run_spatial_benchmark.py` vs `run_spatial_mapping.py`
- `run_spatial_benchmark.py`: Compare ALL backends, select best
- `run_spatial_mapping.py`: Run a SINGLE chosen backend

### `run_v1_complete.py` vs `run_v1_full.py` vs `run_v1_synthetic.py`
- `run_v1_complete.py`: Both datasets, comprehensive (RECOMMENDED)
- `run_v1_full.py`: Real data only, production model
- `run_v1_synthetic.py`: Synthetic only, for testing

## Artifact Locations

```
$DATA/
├── raw/geo/                    # Raw GEO downloads
├── references/
│   ├── hlca/hlca_core.h5ad    # HLCA reference
│   └── luca/luca_luad.h5ad    # LuCA reference
├── processed/luad_evo/
│   ├── snrna_qc_normalized.h5ad
│   ├── spatial_qc_normalized.h5ad
│   ├── wes_features.parquet
│   ├── spatial_benchmark/      # Backend comparison results
│   │   ├── tangram/
│   │   ├── destvi/
│   │   └── tacco/
│   └── canonical/              # Training-ready format
│       ├── cells.parquet
│       ├── neighborhoods.parquet
│       ├── stage_edges.parquet
│       └── split_manifest.json
└── runs/
    └── v1_complete/            # Training outputs
        ├── weights/
        ├── figures/
        └── results.json
```
