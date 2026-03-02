# StageBridge

Transformer-first benchmark for stage-to-stage lung progression modeling
with HLCA-aligned latent representations.

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
      <SAMPLE_ID>.h5ad                 ← created by run_snrna_pipeline.py
      manifest.csv
    spatial/
      <SAMPLE_ID>.h5ad                 ← created by run_spatial_pipeline.py
      manifest.csv
  processed/anndata/
    snrna_merged.h5ad                  ← merged snRNA
    spatial_merged.h5ad                ← merged spatial
  data/reference/hlca/
    hlca_full_v1.h5ad                  ← HLCA full atlas
```

HLCA full atlas download (persistent version URL):

```text
https://datasets.cellxgene.cziscience.com/dbb5ad81-1713-4aee-8257-396fbabe7c6e.h5ad
```

---

## Running the Conversion Pipelines

### snRNA-seq (GSE308103)

Convert all samples and build the merged h5ad:

```bash
export STAGEBRIDGE_DATA_ROOT=/mnt/e/StageBridge_data

# Full pipeline (manifest + per-sample conversion + merge)
python scripts/run_snrna_pipeline.py

# Dry run (shows what would be done)
python scripts/run_snrna_pipeline.py --dry-run
```

To convert a **single file** for testing:

```bash
python -m stagebridge.io.geo_snrna convert \
  /mnt/e/StageBridge_data/data/raw/geo/GSE308103_snrna/extracted/GSM9237901_P3_Normal.raw_counts.mtx.txt.gz \
  /mnt/e/StageBridge_data/interim/anndata/snrna/GSM9237901_P3_Normal.h5ad
```

This prints: `n_cells`, `n_genes`, `nnz`.

### Spatial (GSE307534)

Expand tarballs, convert samples, build manifest, merge:

```bash
export STAGEBRIDGE_DATA_ROOT=/mnt/e/StageBridge_data

python scripts/run_spatial_pipeline.py

# Dry run
python scripts/run_spatial_pipeline.py --dry-run
```

To load a **single sample** for testing:

```bash
python -m stagebridge.io.geo_spatial load \
  /mnt/e/StageBridge_data/data/raw/geo/GSE307534_spatial/samples/GSM9234567_P1_Normal \
  /mnt/e/StageBridge_data/interim/anndata/spatial/GSM9234567_P1_Normal.h5ad
```

---

## Running the Notebook

Once both processed h5ad files exist:

```bash
export STAGEBRIDGE_DATA_ROOT=/mnt/e/StageBridge_data

# Option A: Jupyter Lab
jupyter lab StageBridge.ipynb

# Option B: Classic Jupyter
jupyter notebook StageBridge.ipynb

# Option C: Run headless (no display)
jupyter nbconvert --to notebook --execute StageBridge.ipynb --output StageBridge_executed.ipynb
```

The notebook:
1. Loads `snrna_merged.h5ad` and `spatial_merged.h5ad`
2. Prints dataset summary (patients, stages, shapes)
3. Computes gene intersection + log1p normalisation + HVG selection + PCA
4. Saves figures to `./outputs/figures/`:
   - `pca_scatter.png`
   - `stage_distribution.png`
   - `spatial_tissue_<SAMPLE_ID>.png`

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

After running both pipelines + the notebook:

| File | Description |
|------|-------------|
| `$DATA_ROOT/interim/anndata/snrna/manifest.csv` | snRNA sample manifest |
| `$DATA_ROOT/interim/anndata/snrna/<SAMPLE_ID>.h5ad` | Per-sample snRNA AnnData |
| `$DATA_ROOT/processed/anndata/snrna_merged.h5ad` | Merged snRNA (all samples) |
| `$DATA_ROOT/data/raw/geo/GSE307534_spatial/samples/<SAMPLE_ID>/` | Extracted Visium dirs |
| `$DATA_ROOT/interim/anndata/spatial/manifest.csv` | Spatial sample manifest |
| `$DATA_ROOT/interim/anndata/spatial/<SAMPLE_ID>.h5ad` | Per-sample spatial AnnData |
| `$DATA_ROOT/processed/anndata/spatial_merged.h5ad` | Merged spatial (all samples) |
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
│   ├── splits/donor_holdout.yaml
│   └── experiment/            ← full_benchmark.yaml / smoke.yaml
├── scripts/                   ← thin CLI wrappers (Hydra entry points)
│   ├── check_env.py           ← environment gate
│   ├── audit_data.py          ← data contract gate
│   ├── train_stagebridge.py   ← full benchmark training
│   ├── eval_stagebridge.py    ← evaluation
│   ├── make_poster_assets.py  ← poster panel generation
│   ├── run_snrna_pipeline.py  ← GEO snRNA conversion
│   └── run_spatial_pipeline.py← GEO spatial conversion
├── outputs/                   ← generated artefacts (git-ignored except .gitkeep)
│   ├── figures/               ← poster panels, UMAPs, metrics plots
│   ├── tables/                ← metrics JSON, manifests, env/audit reports
│   └── checkpoints/           ← model checkpoints (*.pt)
└── stagebridge/               ← Python package (`pip install -e .`)
    ├── io/
    │   ├── paths.py           ← single path resolver (StageBridgePaths)
    │   ├── geo_snrna.py       ← custom dense snRNA parser
    │   ├── geo_spatial.py     ← Visium loader
    │   ├── hlca.py            ← HLCA reference atlas I/O
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
    interim/anndata/           ← per-sample h5ads
    processed/anndata/         ← merged h5ads
    data/reference/hlca/       ← HLCA atlas (~20 GB)
```
