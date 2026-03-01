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

### 2. Configure the external data root

The data lives **outside** the repo. Point StageBridge at it:

```bash
export STAGEBRIDGE_DATA_ROOT=/mnt/e/StageBridge_data
```

Add this to your `~/.bashrc` or `~/.zshrc` to make it permanent.

> **Default fallback**: If the variable is unset, `/mnt/e/StageBridge_data` is used.
> If that path doesn't exist either, a clear `ValueError` is raised.

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
python scripts/check_env.py
python scripts/audit_data.py
```

Train cross-validated donor-held-out benchmark (StageBridge + baselines):

```bash
python scripts/train_stagebridge.py experiment=full_benchmark
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
StageBridge/
├── StageBridge.ipynb          ← main notebook entry point
├── README.md
├── pyproject.toml
├── .gitignore
├── scripts/
│   ├── run_snrna_pipeline.py
│   ├── run_spatial_pipeline.py
│   ├── check_env.py
│   ├── audit_data.py
│   ├── train_stagebridge.py
│   ├── eval_stagebridge.py
│   └── make_poster_assets.py
├── outputs/
│   ├── figures/               ← generated plots (git-ignored)
│   └── tables/                ← generated tables (git-ignored)
└── stagebridge/               ← Python package
    ├── __init__.py
    ├── config.py
    ├── logging_utils.py
    ├── io/
    │   ├── geo_snrna.py       ← snRNA parser + manifest + merge
    │   ├── geo_spatial.py     ← Visium loader + manifest + merge
    │   └── manifests.py       ← shared manifest utilities
    ├── preprocessing/
    │   ├── harmonize.py       ← gene intersection, normalisation, HVG, PCA
    │   ├── normalize.py       ← additional normalisation
    │   └── qc.py              ← QC metrics and filtering
    ├── models/                ← Set Transformer + baselines
    ├── training/              ← OT losses, trainer, evaluation
    ├── viz/                   ← plotting utilities for benchmarks/poster panels
    └── utils/                 ← checks, seeds, type aliases
```
