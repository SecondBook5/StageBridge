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
│  - Dual-reference geometry: healthy anchor + disease anchor                 │
│  - Calibrated confidence (percentile rank, comparable across refs)          │
│  - L2-normalized latents before fusion                                      │
│  Output: reference_geometry/                                                │
│          ├── hlca_embedding.parquet                                         │
│          ├── luca_embedding.parquet                                         │
│          ├── fused_embedding.parquet                                        │
│          ├── reference_confidence.parquet                                   │
│          ├── reference_manifest.json                                        │
│          └── feature_overlap_report.json                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
                             SPATIAL DECONVOLUTION
                             ====================

┌─────────────────────────────────────────────────────────────────────────────┐
│  run_spatial_benchmark.py                                                    │
│  Compare Tangram vs DestVI vs TACCO vs Cell2Location, select canonical      │
│  Output: spatial_benchmark/comparison_report.json                           │
│          spatial_benchmark/{backend}/cell_type_proportions.parquet          │
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
│  run_v1_ddp.py             <-- RECOMMENDED: DDP training with biology heads │
│  Full pipeline: SSL pretraining + transition training                       │
│                                                                             │
│  Phase 1 - SSL Pretraining (100 epochs):                                   │
│    - Masked receiver reconstruction (70%)                                   │
│    - Pathway/Proliferation supervision (5% each)                           │
│    - IL1B head (Peng/Kadara hypothesis test)                               │
│    - KAC head (KRT8+ intermediate state, Han et al. 2024)                  │
│                                                                             │
│  Phase 2 - Transition Training (50 epochs):                                │
│    - OT-CFM flow matching with Sinkhorn coupling                           │
│    - Stage-aware OT pairing (adjacent stages only)                         │
│    - DestVI gamma integration (functional state)                           │
│                                                                             │
│  Output: runs/v1_complete/fold{N}_seed{S}/checkpoints/                     │
│          ssl_pretrained.pt, best_checkpoint.pt, final_checkpoint.pt        │
└─────────────────────────────────────────────────────────────────────────────┘


                              ALTERNATIVE TRAINING
                              ====================

┌─────────────────────────────┐    ┌─────────────────────────────┐
│  run_v1_full.py             │    │  pretrain_local.py          │
│  Real data ONLY             │    │  Local SSL pretraining only │
│  Production components      │    │  For development/debugging  │
└─────────────────────────────┘    └─────────────────────────────┘

┌─────────────────────────────┐
│  run_full.py                │
│  OmegaConf-based orchestrator│
│  Calls: run_reference →     │
│         run_spatial_mapping →│
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
| `run_full.py` | OmegaConf orchestrator | Config-driven runs |

## HPC Execution Order

```bash
# 0. Environment setup (CRITICAL)
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
# Must show True before proceeding

# 1. Download references (once)
python -m stagebridge.pipelines.download_references --output_dir $DATA/references --all

# 2. Data preparation
#    --spatial-merge-only: Skip spatial QC/norm (backends handle it internally)
python -m stagebridge.pipelines.run_data_prep --data-root $DATA --spatial-merge-only

# 3. Add ENSG IDs to query (required for model-based mapping)
python scripts/add_ensembl_ids.py \
    --query $DATA/processed/luad_evo/snrna_qc_normalized.h5ad \
    --hlca $DATA/references/hlca/hlca_reference.h5ad \
    --output $DATA/processed/luad_evo/snrna_qc_normalized_with_ensg.h5ad

# 4. Reference mapping (model-based scArches surgery)
#    Run HLCA first (works with current pandas)
python -m stagebridge.pipelines.run_reference \
    --data-root $DATA \
    --hpc \
    --hlca-only \
    --snrna $DATA/processed/luad_evo/snrna_qc_normalized_with_ensg.h5ad

#    Run LuCA separately (may need pandas 1.5.x environment)
python -m stagebridge.pipelines.run_reference \
    --data-root $DATA \
    --hpc \
    --luca-only \
    --snrna $DATA/processed/luad_evo/snrna_qc_normalized_with_ensg.h5ad \
    --luca $DATA/references/luca/luca_core_atlas.h5ad

# 5. Spatial backend benchmark (Tangram/DestVI/TACCO/Cell2location)
#    RECOMMENDED: Use Snakemake for per-sample job management
#    snakemake --profile workflow/slurm --jobs 20
#
#    Manual single-sample execution:
python -m stagebridge.pipelines.run_spatial_benchmark \
    --snrna $DATA/processed/luad_evo/snrna_with_celltypes.h5ad \
    --spatial $DATA/processed/luad_evo/spatial_merged.h5ad \
    --output_dir $DATA/runs/spatial_benchmark/hlca/tangram/samples/SAMPLE_ID \
    --backends tangram \
    --sample SAMPLE_ID \
    --label-source hlca

# 6. Complete data prep (canonical format)
python -m stagebridge.pipelines.complete_data_prep \
    --snrna $DATA/processed/luad_evo/snrna_qc_normalized.h5ad \
    --spatial $DATA/processed/luad_evo/spatial_merged.h5ad \
    --wes $DATA/processed/luad_evo/wes_features.parquet \
    --spatial_backend_dir $DATA/processed/luad_evo/spatial_benchmark \
    --output_dir $DATA/processed/luad_evo/canonical

# 7. Training (both semi-synthetic + real)
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

### `run_v1_complete.py` vs `run_v1_full.py`
- `run_v1_complete.py`: Both datasets, comprehensive (RECOMMENDED)
- `run_v1_full.py`: Real data only, production model

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
├── runs/
│   └── spatial_benchmark/      # Backend comparison results
│       ├── hlca/               # Using HLCA cell type labels
│       │   ├── tangram/samples/*/
│       │   ├── destvi/samples/*/
│       │   ├── tacco/samples/*/
│       │   └── cell2location/samples/*/
│       ├── luca/               # Using LuCA cell type labels (ablation)
│       │   ├── tangram/samples/*/
│       │   ├── destvi/samples/*/
│       │   ├── tacco/samples/*/
│       │   └── cell2location/samples/*/
│       ├── canonical/samples/*/ # Best backend on all samples
│       ├── backend_comparison.json
│       └── canonical_backend.json
│   └── canonical/              # Training-ready format
│       ├── cells.parquet
│       ├── neighborhoods.parquet
│       ├── stage_edges.parquet
│       └── split_manifest.json
├── processed/luad_evo/
│   └── reference_geometry/    # Dual-reference mapping outputs
│       ├── hlca_embedding.parquet
│       ├── luca_embedding.parquet
│       ├── fused_embedding.parquet
│       ├── reference_confidence.parquet
│       ├── reference_manifest.json
│       └── feature_overlap_report.json
└── runs/
    └── v1_complete/            # Training outputs
        ├── weights/
        ├── figures/
        └── results.json
```

## Reference Sources and Verification

### HLCA (Human Lung Cell Atlas)

| Property | Value |
|----------|-------|
| Source | CZI cellxgene via scvi-tools Hugging Face Hub |
| Repository | `scvi-tools/human-lung-cell-atlas-scanvi` |
| Cells | ~584K healthy lung cells |
| Latent key | `X_scanvi_emb` (30 dimensions) |
| Download | `python -m stagebridge.pipelines.download_references --download_hlca` |

### LuCA (Lung Cancer Atlas)

| Property | Value |
|----------|-------|
| Source | Zenodo / LungCancerAtlas GitHub |
| Cells | ~790K (core) or ~1.3M (extended) |
| Latent key | `X_scVI` (10 dimensions) |

**CRITICAL: Use LuCA Core, NOT Extended**

LuCA Extended has 31% NaN in latent embeddings. Always verify before mapping:

```bash
# Diagnose latent integrity
python -m stagebridge.reference.diagnose_reference /path/to/luca.h5ad \
    --latent-key X_scVI --diagnose-only

# Expected output for usable reference:
# Valid cells: 790,000 (100.0%)
# Recommendation: usable
```

If NaN detected, either:
1. Use LuCA Core instead
2. Clean the reference: `--output /path/to/luca_cleaned.h5ad`

### Reference Verification Checklist

Before running `run_reference.py`:

1. [ ] HLCA downloaded and model cached
2. [ ] LuCA is Core version (100% valid latents)
3. [ ] Run `diagnose_reference.py --diagnose-only` on both
4. [ ] Gene overlap expected >30% (symbols vs ENSG handled automatically)

## Dual-Reference Mapping Design

### Purpose

The dual-reference mapping provides a **comparative coordinate system** for progression-relevant cells:

- **HLCA** (Human Lung Cell Atlas): Healthy lung anchor - defines "normal" cell states
- **LuCA** (Lung Cancer Atlas): Disease-aware anchor - defines cancer/progression cell states
- **Fused**: Combined representation capturing both healthy deviation and disease similarity

### Why Two References?

Single-reference mapping loses information:
- HLCA-only: Can detect "abnormality" but not cancer-specific patterns
- LuCA-only: Can detect cancer similarity but not deviation from healthy

Dual-reference captures both signals simultaneously, enabling the model to learn:
- How far a cell has deviated from healthy (HLCA distance)
- How similar a cell is to known cancer states (LuCA distance)
- The trajectory through the healthy→cancer coordinate space

### Confidence Calibration

**Problem**: LuCA (1.3M cells) is denser than HLCA (584K cells). Raw k-NN distances would be systematically smaller for LuCA, causing the model to overtrust LuCA for the wrong reason.

**Solution**: Percentile rank calibration

```python
# Instead of: conf = 1 / (1 + dist)  # WRONG - density-biased
# We use:
ranks = rankdata(distances)
confidence = 1.0 - (ranks - 1) / (len(ranks) - 1)  # Percentile rank
```

This ensures:
- A cell at the 90th percentile of HLCA distances gets the same confidence as a cell at the 90th percentile of LuCA distances
- Confidence is comparable across references regardless of density
- The transformer sees balanced signals from both references

### Latent Normalization

Before fusion, each reference's latent space is L2-normalized:

```python
hlca_normalized = hlca_latent / ||hlca_latent||_2
luca_normalized = luca_latent / ||luca_latent||_2
fused = concat(hlca_normalized, luca_normalized)
```

This prevents one reference from dominating the fused representation due to different latent scales.

### Output Schema

```
reference_confidence.parquet columns:
├── cell_id, donor_id, sample_id, stage_id
├── hlca_confidence          # Calibrated [0,1], comparable
├── luca_confidence          # Calibrated [0,1], comparable
├── hlca_raw_distance        # Original k-NN distance (for debugging)
├── luca_raw_distance        # Original k-NN distance (for debugging)
├── hlca_confidence_method   # "percentile_rank"
├── luca_confidence_method   # "percentile_rank"
└── reference_mode_used      # "both", "hlca_only", or "luca_only"
```

### Three Modes

```bash
# Default: Both references (recommended)
python -m stagebridge.pipelines.run_reference --data-root $DATA

# HLCA-only: Healthy reference only
python -m stagebridge.pipelines.run_reference --data-root $DATA --hlca-only

# LuCA-only: Cancer reference only
python -m stagebridge.pipelines.run_reference --data-root $DATA --luca-only
```

### HPC Mode

For large datasets (>100K cells) or large references (>1M cells):

```bash
python -m stagebridge.pipelines.run_reference \
    --data-root $DATA \
    --luca /path/to/luca_extended.h5ad \
    --hpc \
    --chunk-size 50000
```

HPC mode features:
- Chunked/streaming processing (never loads full reference matrix)
- FAISS GPU acceleration for k-NN search
- Backed mode for memory-efficient h5ad loading
- Handles references >10GB without OOM

### Model-Based Projection (scArches Surgery)

The primary method for reference mapping uses **scArches surgery** - fine-tuning pretrained scANVI models on query data. This is more principled than k-NN projection because the latent space was learned to capture biological variation.

```bash
# Model-based is used automatically when pretrained models are available
python -m stagebridge.pipelines.run_reference \
    --data-root $DATA \
    --hpc \
    --snrna $DATA/processed/luad_evo/snrna_qc_normalized_with_ensg.h5ad
```

**How it works:**
1. Load pretrained scANVI model (HLCA from HuggingFace Hub, LuCA from local)
2. Match query genes to model genes (auto-converts symbols to ENSG if needed)
3. Run scArches surgery (200 epochs, fine-tunes batch correction layers)
4. Extract latent representations from the adapted model

**Gene ID Format:**
- HLCA model expects ENSG IDs (Ensembl gene identifiers)
- Query data typically uses gene symbols
- The pipeline auto-converts using the `ensembl_id` column in `adata.var`
- Prepare query with: `scripts/add_ensembl_ids.py`

**Fallback:**
If model-based fails (model not found, version incompatibility), the pipeline falls back to k-NN projection with PCA reduction.

### LuCA Model Compatibility

The LuCA scANVI model may fail with newer pandas versions:
```
Argument 'placement' has incorrect type (expected pandas._libs.internals.BlockPlacement, got slice)
```

**Solution:** Create a separate environment with pandas 1.5.x:
```bash
conda create -n luca_compat python=3.11 -y
conda activate luca_compat
pip install scvi-tools pandas==1.5.3 torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Or run HLCA-only first, then LuCA separately:
```bash
# HLCA only (works with current pandas)
python -m stagebridge.pipelines.run_reference --data-root $DATA --hpc --hlca-only

# LuCA in compat environment
conda activate luca_compat
python -m stagebridge.pipelines.run_reference --data-root $DATA --hpc --luca-only
```

### PyTorch CUDA Setup

Before running on HPC with GPUs:
```bash
# Set visible GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Verify PyTorch sees GPUs
python -c "import torch; print('CUDA:', torch.cuda.is_available(), 'Devices:', torch.cuda.device_count())"

# If False, reinstall PyTorch with CUDA 12.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 --force-reinstall
pip install torchmetrics
```

Use cu124 (CUDA 12.4) for best compatibility. Even if nvidia-smi shows CUDA 13.x, drivers are backward compatible.
