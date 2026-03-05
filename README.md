# StageBridge

Transformer-first benchmark for stage-to-stage lung progression modeling
with HLCA-aligned latent representations.

**Operational rule:** `StageBridge.ipynb` is the one-stop orchestration entrypoint.
All end-to-end runs (data build -> HLCA mapping -> evaluation/training) should be
triggered from notebook controls. Scripts under `scripts/` remain the underlying,
versioned implementations that the notebook calls.

**Submission/Grading target:** The top-level `StageBridge.ipynb` notebook is the
primary artefact for end-to-end execution (preprocessing -> training -> final
outputs). The Python package and scripts exist to support notebook execution.

Integrates:
- **snRNA-seq** (GSE308103 — custom dense-counts format)
- **10x Visium spatial** (GSE307534)
- **HLCA full reference atlas** (`.h5ad`, ~20.36 GB)

---

## Setup

### 1. Create the micromamba environment

**Hardware**: NVIDIA RTX 4000 Ada (20 GB VRAM), driver 581.42, CUDA ≤12.8.
The `environment.yml` targets **CUDA 12.6** (PyTorch `cu126`).

```bash
# Install micromamba if not already available
# https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html

# From the repo root — creates env named 'stagebridge'
micromamba create -f environment.yml

# Activate
micromamba activate stagebridge

# Install the stagebridge package itself in editable mode
pip install -e .
```

Verify GPU is visible after activation:
```bash
python -c "import torch; print(torch.cuda.get_device_name(0))"
# Expected: NVIDIA RTX 4000 Ada Generation
```

#### Optional pip extras

```bash
# scVI, scANVI, TOTALVI (GPU — already has PyTorch, just adds scvi-tools)
pip install -e ".[dl]"

# Tangram cell-type mapping + cell2location deconvolution
pip install -e ".[mapping]"

# SpatialData format (Xenium, MERSCOPE, new Visium HD)
pip install -e ".[spatialdata]"

# RNA velocity (scVelo) + trajectory inference (CellRank)
pip install -e ".[trajectory]"

# Doublet detection (Scrublet, DoubletDetection)
pip install -e ".[qc]"

# Everything at once
pip install -e ".[all]"
```

### 2. Validate the environment

Run the environment gate before anything else:

```bash
python scripts/check_env.py        # human-readable output
python scripts/check_env.py --json # machine-readable JSON → outputs/tables/env_check.json
python scripts/check_env.py --no-gpu  # skip CUDA check (CPU CI)
```

Run the test suite:

```bash
pytest -q
```

Both must exit 0 before proceeding.

### 3. Configure the external data root

The data lives **outside** the repo at a configurable path. Two options:

**Option A — environment variable (quick)**

```bash
export STAGEBRIDGE_DATA_ROOT=/mnt/e/StageBridge_data
```

Add this to your `~/.bashrc` or `~/.zshrc` to make it permanent.

**Option B — Hydra local config (recommended)**

```bash
cp configs/data/local.yaml.example configs/data/local.yaml
# Edit configs/data/local.yaml and set data_root to your path.
# Then pass data=local to any Hydra script:
python scripts/train_stagebridge.py data=local
```

`configs/data/local.yaml` is git-ignored — it never enters the repo.

> **Default**: If neither is set, a `ValueError` is raised with a clear message.

---

## Expected External Data Layout

```
$STAGEBRIDGE_DATA_ROOT/
  data/raw/geo/
    GSE308103_snrna/extracted/
      *.raw_counts.mtx.txt.gz          ← downloaded from GEO
    GSE307534_spatial/
      extracted/
        GSM*.tar.gz                    ← downloaded from GEO
      samples/                         ← created by run_spatial_pipeline.py
        <SAMPLE_ID>/
          filtered_feature_bc_matrix/
          spatial/
  interim/anndata/
    snrna/
      snrna_smoke.h5ad                 ← experiment=smoke
      snrna_full.h5ad                  ← experiment=full
    spatial/
      spatial_smoke.h5ad               ← experiment=smoke
      spatial_full.h5ad                ← experiment=full
  processed/anndata/
    snrna_hlca_latent_full.h5ad        ← full-cell HLCA latent embedding
  processed/hlca/
    snrna_full_hlca_labels.parquet     ← HLCA labels keyed by cell_id
    query_model_full/                  ← trained scArches query model
  data/reference/hlca/
    hlca_full_v1.h5ad                  ← HLCA full atlas
  runs/
    <run_id>/tables/
      run_manifest.json
      data_audit.json
      config_resolved.yaml
      hlca_mapping_report.json
      hlca_gene_id_report.json
      hlca_validation_report.json
```

HLCA full atlas download (persistent version URL):

```text
https://datasets.cellxgene.cziscience.com/dbb5ad81-1713-4aee-8257-396fbabe7c6e.h5ad
```

---

## Running the Conversion Pipelines

### snRNA-seq (GSE308103)

Build interim snRNA AnnData:

```bash
export STAGEBRIDGE_DATA_ROOT=/mnt/e/StageBridge_data

# Fast smoke artifact (small subset)
python scripts/run_snrna_pipeline.py data=local experiment=smoke
# -> /mnt/e/StageBridge_data/interim/anndata/snrna/snrna_smoke.h5ad

# Full artifact (all selected samples)
python scripts/run_snrna_pipeline.py data=local experiment=full
# -> /mnt/e/StageBridge_data/interim/anndata/snrna/snrna_full.h5ad
```

Notes:
- Pipelines are Hydra-driven (`data=local`, `experiment=smoke|full`).
- Run-scoped manifests/audits are written under `runs/<run_id>/tables/`.

To parse a **single file** for debugging:

```bash
python -m stagebridge.io.geo_snrna convert \
  /mnt/e/StageBridge_data/data/raw/geo/GSE308103_snrna/extracted/GSM9237901_P3_Normal.raw_counts.mtx.txt.gz \
  /mnt/e/StageBridge_data/interim/anndata/snrna/GSM9237901_P3_Normal.debug.h5ad
```

This prints: `n_cells`, `n_genes`, `nnz`.

### Spatial (GSE307534)

Build interim spatial AnnData:

```bash
export STAGEBRIDGE_DATA_ROOT=/mnt/e/StageBridge_data

python scripts/run_spatial_pipeline.py data=local experiment=smoke
# -> /mnt/e/StageBridge_data/interim/anndata/spatial/spatial_smoke.h5ad

python scripts/run_spatial_pipeline.py data=local experiment=full
# -> /mnt/e/StageBridge_data/interim/anndata/spatial/spatial_full.h5ad
```

To load a **single sample directory** for debugging:

```bash
python -m stagebridge.io.geo_spatial load \
  /mnt/e/StageBridge_data/data/raw/geo/GSE307534_spatial/samples/GSM9234567_P1_Normal \
  /mnt/e/StageBridge_data/interim/anndata/spatial/GSM9234567_P1_Normal.debug.h5ad
```

---

## Running the Notebook

`StageBridge.ipynb` is the primary operator interface.
It includes a control section (**0A — Pipeline Controls**) that can trigger:
- interim snRNA build (`scripts/run_snrna_pipeline.py`)
- interim spatial build (`scripts/run_spatial_pipeline.py`)
- HLCA mapping (`scripts/run_hlca_mapping.py`)
- HLCA validation (`scripts/eval_hlca_mapping.py`)
- Tangram mapping (`scripts/run_tangram_mapping.py`)
- Tangram map plotting (`scripts/plot_tangram_maps.py`)
- niche token feature build (`scripts/build_niche_tokens.py`)
- niche token bank build (`scripts/build_niche_token_bank.py`)
- niche token QC maps (`scripts/qc_niche_tokens.py`)
- post-Tangram AIS→MIA training + ablation smoke (`scripts/train_stagebridge.py`)
- one-command post-Tangram acceptance (`scripts/run_full_step_after_tangram.py`)

Launch the notebook:

```bash
export STAGEBRIDGE_DATA_ROOT=/mnt/e/StageBridge_data

# Option A: Jupyter Lab
jupyter lab StageBridge.ipynb

# Option B: Classic Jupyter
jupyter notebook StageBridge.ipynb

# Option C: Run headless (no display)
jupyter nbconvert --to notebook --execute StageBridge.ipynb --output StageBridge_executed.ipynb
```

Recommended flow:
1. For a single-shot full refresh, set `RUN_PIPELINE_ALL = True`.
2. Otherwise set individual 0A toggles (`RUN_BUILD_SNRNA`, `RUN_BUILD_SPATIAL`, `RUN_HLCA_MAPPING`, `RUN_HLCA_EVAL`, `RUN_TANGRAM_MAPPING`, `RUN_TANGRAM_PLOTS`, `RUN_BUILD_NICHE_TOKENS`, `RUN_BUILD_NICHE_TOKEN_BANK`, `RUN_QC_NICHE_TOKENS`, `RUN_TRAIN_POST_TANGRAM`).
3. Run the 0A runner cell (this calls the scripts, records run IDs).
4. Watch progress in-cell:
   - HLCA training/inference prints progress (including tqdm chunk bars in mapping stages).
   - Tangram profile aggregation prints tqdm chunk bars.
   - Post-Tangram niche-token and training scripts stream live stdout in the same runner cell.
5. Continue through downstream notebook analysis/training cells.

The notebook then:
1. Loads interim artifacts (`snrna_smoke|full.h5ad`, `spatial_smoke|full.h5ad`).
2. Optionally attaches HLCA latent/labels if available.
3. Optionally runs Tangram projection onto spatial spots.
4. Runs harmonization and latent/training analyses.
5. Saves figures to `./outputs/figures/`:
   - `pca_scatter.png`
   - `stage_distribution.png`
   - `spatial_tissue_<SAMPLE_ID>.png`

---

## HLCA Mapping and Validation (CLI, underlying implementation)

Full-scale HLCA mapping:

```bash
python scripts/run_hlca_mapping.py data=local experiment=full
```

Expected primary outputs:
- `/mnt/e/StageBridge_data/processed/anndata/snrna_hlca_latent_full.h5ad`
- `/mnt/e/StageBridge_data/processed/hlca/snrna_full_hlca_labels.parquet`
- `/mnt/e/StageBridge_data/runs/<run_id>/tables/hlca_mapping_report.json`
- `/mnt/e/StageBridge_data/runs/<run_id>/tables/hlca_gene_id_report.json`

Quantitative validation:

```bash
python scripts/eval_hlca_mapping.py data=local
```

Validation report:
- `/mnt/e/StageBridge_data/runs/<run_id>/tables/hlca_validation_report.json`

The notebook can invoke both commands from section 0A; use CLI directly for
batch/server execution.

---

## Tangram Mapping (CLI, underlying implementation)

Project HLCA-labeled snRNA profiles onto spatial spots:

```bash
python scripts/run_tangram_mapping.py data=local experiment=full
```

Expected primary outputs:
- `/mnt/e/StageBridge_data/processed/tangram/tangram_map_full.h5ad`
- `/mnt/e/StageBridge_data/processed/tangram/spatial_tangram_full.h5ad`
- `/mnt/e/StageBridge_data/processed/tangram/spatial_tangram_celltype_scores.parquet`
- `/mnt/e/StageBridge_data/runs/<run_id>/tables/tangram_report.json`

Plot Tangram per-celltype spatial maps and winner-label map:

```bash
python scripts/plot_tangram_maps.py data=local
```

Outputs:
- `./outputs/figures/tangram_celltype_maps_<sample_id>.png`
- `./outputs/figures/tangram_winner_map_<sample_id>.png`
- `/mnt/e/StageBridge_data/runs/<run_id>/tables/tangram_plot_report.json`

---

## Niche Tokens and Token Bank (post-Tangram)

Build niche-token features from Tangram scores:

```bash
python scripts/build_niche_tokens.py \
  --spatial_h5ad /mnt/e/StageBridge_data/processed/tangram/spatial_tangram_full.h5ad \
  --scores_parquet /mnt/e/StageBridge_data/processed/tangram/spatial_tangram_celltype_scores.parquet \
  --out_parquet /mnt/e/StageBridge_data/processed/features/niche_tokens_full.parquet \
  --json
```

Build a per-sample Zarr token bank for fast training-time sampling:

```bash
python scripts/build_niche_token_bank.py \
  --niche_tokens_parquet /mnt/e/StageBridge_data/processed/features/niche_tokens_full.parquet \
  --out_zarr /mnt/e/StageBridge_data/processed/features/niche_token_bank.zarr \
  --json
```

Generate quick QC maps:

```bash
python scripts/qc_niche_tokens.py \
  --niche_tokens_parquet /mnt/e/StageBridge_data/processed/features/niche_tokens_full.parquet \
  --spatial_h5ad /mnt/e/StageBridge_data/processed/tangram/spatial_tangram_full.h5ad \
  --out_dir outputs/figures/niche_tokens \
  --json
```

One-command acceptance chain (tokens -> bank -> qc -> AIS→MIA smoke training):

```bash
python scripts/run_full_step_after_tangram.py --data_config=local --json
```

---

## Training the Transformer Benchmark

Validate environment and data contracts first:

```bash
python scripts/check_env.py          # exits 0 = all required deps present
python scripts/audit_data.py         # exits 0 = all h5ads valid + have required columns
```

Train cross-validated donor-held-out benchmark (StageBridge + baselines):

```bash
# Using environment variable
python scripts/train_stagebridge.py experiment=full_benchmark

# Using local config file
python scripts/train_stagebridge.py data=local experiment=full_benchmark

# Smoke run (tiny model, 1 epoch, CPU)
python scripts/train_stagebridge.py model=smoke training=smoke experiment=smoke
```

Evaluate a checkpoint:

```bash
python scripts/eval_stagebridge.py checkpoint=/path/to/checkpoint.pt
```

Generate poster assets from metrics:

```bash
python scripts/make_poster_assets.py outputs/tables/metrics_stagebridge_full_benchmark.json
```

---

## Expected Outputs

After running notebook-driven build/mapping/evaluation:

| File | Description |
|------|-------------|
| `$DATA_ROOT/interim/anndata/snrna/snrna_smoke.h5ad` | snRNA smoke artifact |
| `$DATA_ROOT/interim/anndata/snrna/snrna_full.h5ad` | snRNA full artifact |
| `$DATA_ROOT/interim/anndata/spatial/spatial_smoke.h5ad` | Spatial smoke artifact |
| `$DATA_ROOT/interim/anndata/spatial/spatial_full.h5ad` | Spatial full artifact |
| `$DATA_ROOT/processed/anndata/snrna_hlca_latent_full.h5ad` | HLCA latent embedding (all query cells, 30D) |
| `$DATA_ROOT/processed/hlca/snrna_full_hlca_labels.parquet` | HLCA label table keyed by cell_id |
| `$DATA_ROOT/processed/tangram/tangram_map_full.h5ad` | Tangram learned mapping object |
| `$DATA_ROOT/processed/tangram/spatial_tangram_full.h5ad` | Spatial AnnData with projected cell-type scores |
| `$DATA_ROOT/processed/tangram/spatial_tangram_celltype_scores.parquet` | Spot x HLCA-label score table |
| `$DATA_ROOT/runs/<run_id>/tables/hlca_mapping_report.json` | HLCA runtime + optimization telemetry |
| `$DATA_ROOT/runs/<run_id>/tables/hlca_gene_id_report.json` | Gene-ID overlap/mapping diagnostics |
| `$DATA_ROOT/runs/<run_id>/tables/hlca_validation_report.json` | Post-mapping quality checks |
| `$DATA_ROOT/runs/<run_id>/tables/tangram_report.json` | Tangram run telemetry and shape summary |
| `$DATA_ROOT/runs/<run_id>/tables/config_resolved.yaml` | Resolved Hydra config for reproducibility |
| `$DATA_ROOT/data/reference/hlca/hlca_full_v1.h5ad` | HLCA full reference atlas |
| `./outputs/figures/pca_scatter.png` | PCA scatter plot |
| `./outputs/figures/stage_distribution.png` | Stage distribution bar chart |
| `./outputs/figures/spatial_tissue_*.png` | Tissue spot map |
| `./outputs/tables/metrics_<RUN>.json` | Cross-validation benchmark summary |
| `./outputs/tables/run_manifest_<RUN>.json` | Reproducibility manifest (runs/variants/folds, config hash, timestamp) |
| `./outputs/tables/env_check.json` | Environment validation report |
| `./outputs/tables/data_audit.json` | Dataset contract audit report |
| `./outputs/figures/poster_panel_*.png` | Poster-ready figure panels |

---

## snRNA File Format Note

The `*.raw_counts.mtx.txt.gz` files are **not** standard Matrix Market.

```
barcode_1  barcode_2  barcode_3  ...   ← line 1: space-delimited barcodes
GENE1  0  5  0  ...                    ← subsequent lines: GENE count_per_cell
GENE2  3  0  1  ...
```

The parser in [stagebridge/io/geo_snrna.py](stagebridge/io/geo_snrna.py) streams
line-by-line and builds a sparse CSR matrix — the full dense matrix is never loaded.

---

## Project Structure

```
StageBridge/                   ← repo root (code only, no data)
├── StageBridge.ipynb          ← main notebook entry point
├── README.md
├── pyproject.toml
├── environment.yml
├── .gitignore
├── configs/                   ← Hydra config tree
│   ├── config.yaml            ← master defaults list (entry point)
│   ├── train.yaml             ← training run defaults
│   ├── eval.yaml              ← eval run defaults
│   ├── data/
│   │   ├── default.yaml       ← portable paths (env-var based)
│   │   └── local.yaml.example ← copy → local.yaml (git-ignored)
│   ├── model/stagebridge.yaml ← full model / model/smoke.yaml
│   ├── training/default.yaml  ← trainer hparams / training/smoke.yaml
│   ├── hlca/default.yaml      ← HLCA mapping defaults
│   ├── splits/donor_holdout.yaml
│   └── experiment/            ← full_benchmark.yaml / smoke.yaml
├── scripts/                   ← thin CLI wrappers (Hydra entry points)
│   ├── check_env.py           ← environment gate
│   ├── audit_data.py          ← data contract gate
│   ├── run_snrna_pipeline.py  ← GEO snRNA → snrna_smoke/full.h5ad
│   ├── run_spatial_pipeline.py← GEO spatial → spatial_smoke/full.h5ad
│   ├── run_hlca_mapping.py    ← full-scale HLCA mapping (scArches query)
│   ├── eval_hlca_mapping.py   ← HLCA quantitative validation report
│   ├── run_tangram_mapping.py ← HLCA-labeled snRNA → spatial projection
│   ├── plot_tangram_maps.py   ← Tangram per-celltype and winner-map figures
│   ├── build_niche_tokens.py  ← Tangram scores → niche token features parquet
│   ├── build_niche_token_bank.py ← niche tokens parquet → Zarr sample bank
│   ├── qc_niche_tokens.py     ← entropy / token spatial QC maps
│   ├── run_full_step_after_tangram.py ← one-command post-Tangram smoke chain
│   ├── train_stagebridge.py   ← full benchmark training
│   ├── eval_stagebridge.py    ← evaluation
│   ├── make_poster_assets.py  ← poster panel generation
│   └── ...
├── outputs/                   ← generated artefacts (git-ignored except .gitkeep)
│   ├── figures/               ← poster panels, UMAPs, metrics plots
│   ├── tables/                ← metrics JSON, manifests, env/audit reports
│   └── checkpoints/           ← model checkpoints (*.pt)
└── stagebridge/               ← Python package (`pip install -e .`)
    ├── io/
    │   ├── paths.py           ← single path resolver (StageBridgePaths)
    │   ├── geo_snrna.py       ← custom dense snRNA parser
    │   ├── geo_spatial.py     ← Visium loader
    │   ├── interim_build.py   ← module-level snRNA/spatial build orchestration
    │   ├── hlca.py            ← HLCA mapping + validation core logic
    │   ├── tangram.py         ← Tangram mapping core logic
    │   ├── niche_tokens.py    ← niche-token features + Zarr token-bank core
    │   └── manifests.py
    ├── pipeline/              ← high-level orchestration
    │   ├── steps.py           ← stub step functions (audit → poster)
    │   └── run.py             ← run_smoke() integration runner
    ├── preprocessing/
    │   ├── stage_ontology.py  ← canonical stage order + normalization
    │   ├── latent.py          ← HLCA / PCA latent builders
    │   ├── harmonize.py       ← gene intersection, normalisation, HVG, PCA
    │   └── normalize.py / qc.py
    ├── models/
    │   ├── layers.py          ← SAB, ISAB, PMA, FiLM, SinusoidalTimeEmbedding
    │   ├── stagebridge.py     ← StageBridgeModel (Set Transformer + flow matching)
    │   └── baselines.py       ← DeepSets, NoContext, Linear baselines
    ├── training/
    │   ├── losses.py          ← Sinkhorn coupling, flow matching loss
    │   ├── trainer.py         ← StageBridgeTrainer + donor-holdout CV
    │   └── eval.py            ← Sinkhorn dist, MMD, AUC, JSD metrics
    ├── analysis/              ← biological analysis modules
    │   ├── context_sensitivity.py
    │   ├── gene_attribution.py
    │   └── trajectory.py
    ├── viz/                   ← visualization
    │   ├── embeddings.py      ← UMAP plots
    │   ├── poster.py          ← 4-panel poster assembly
    │   ├── spatial_plots.py   ← Visium spot plots
    │   └── curves.py
    └── utils/                 ← types, seeds, checks

$STAGEBRIDGE_DATA_ROOT/        ← external data (NEVER in repo)
    data/raw/geo/              ← GEO downloads
    interim/anndata/           ← snrna_smoke/full + spatial_smoke/full
    processed/anndata/         ← snrna_hlca_latent_full.h5ad
    processed/hlca/            ← labels parquet + query model/cache
    runs/                      ← run-scoped manifests/reports
    data/reference/hlca/       ← HLCA atlas (~20 GB)
```
