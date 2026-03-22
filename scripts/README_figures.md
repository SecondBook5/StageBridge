# Publication Figure Generation

This directory contains the HPC batch script for generating all publication-quality figures for the Nature Methods manuscript.

## Quick Start

```bash
# Submit figure generation job
sbatch scripts/hpc_step7_figures.sbatch

# Check job status
squeue -u $USER

# Monitor progress
tail -f /home/booka/StageBridge/logs/figures_<job_id>.log
```

## Overview

**Script:** `hpc_step7_figures.sbatch`

Generates ALL publication figures with:
- Multi-format export (PNG 300 DPI, PDF vector, SVG)
- Skip-if-exists logic (only regenerates missing figures)
- Automatic manifest generation
- Full error reporting

**Runtime:** ~2-3 hours on HPC with GPU

## Output Structure

```
$DATA/runs/publication_figures/
├── main/                          # Main figures (1-7)
│   ├── fig01_reference_geometry.{png,pdf,svg}
│   ├── fig02_training_curves.{png,pdf,svg}
│   ├── fig03_spatial_backends.{png,pdf,svg}
│   ├── fig04_embeddings.{png,pdf,svg}
│   ├── fig05_ablation_heatmap.{png,pdf,svg}
│   ├── fig06_attention.{png,pdf,svg}
│   └── fig07_biology.{png,pdf,svg}
├── supplementary/                 # Supplementary figures (S1-S8)
│   ├── figS1_data_qc.{png,pdf,svg}
│   ├── figS2_reference_diagnostics.{png,pdf,svg}
│   └── figS3_hpo_history.{png,pdf,svg}
├── panels/                        # Individual panels
├── manifests/                     # Metadata
│   ├── figure_manifest.json       # All figures with paths
│   └── generation_log.jsonl       # Generation log
└── tikz/                          # TikZ source files
```

## Figure Groups

### Main Figures

1. **Reference Geometry** (`fig01_reference_geometry`)
   - Panel A: HLCA reference structure
   - Panel B: LuCA reference structure
   - Panel C: Fused dual-reference embedding

2. **Training Curves** (`fig02_training_curves`)
   - Panel A: SSL pretraining loss
   - Panel B: Transition model (train/val)
   - Panel C: Wasserstein distance
   - Panel D: Reconstruction metrics (MSE/MMD)

3. **Spatial Backend Comparison** (`fig03_spatial_backends`)
   - Panel A: Overall performance score
   - Panel B: Runtime comparison
   - Panel C: Prediction accuracy

4. **Stage Embeddings** (`fig04_embeddings`)
   - UMAP visualization with density contours
   - Convex hulls per stage
   - 95% confidence ellipses

5. **Ablation Heatmap** (`fig05_ablation_heatmap`)
   - Component ablation results
   - Copied from ablation output

6. **Attention Analysis** (`fig06_attention`)
   - Panel A: Attention pattern heatmap
   - Panel B: Token importance

7. **Biological Validation** (`fig07_biology`)
   - IL1B pathway genes (6 panels)
   - Expression per stage (violin plots)

### Supplementary Figures

- **S1: Data QC** - Cell counts, gene detection, UMI, donor distribution
- **S2: Reference Diagnostics** - HLCA/LuCA ELBO training curves
- **S3: HPO History** - Hyperparameter optimization trials
- **S4-S8:** Extended analysis (planned)

## Prerequisites

All previous pipeline stages must be complete:

```bash
# Required inputs
$DATA/processed/luad_evo/reference_geometry/   # Stage 1-2: Reference mapping
$DATA/processed/luad_evo/spatial_benchmark/    # Stage 3: Spatial backends
$DATA/processed/luad_evo/canonical/            # Stage 4: Data prep
$DATA/runs/v1_complete/                        # Stage 5: Training
$DATA/runs/ablations/                          # Stage 6: Ablations
```

## Skip-If-Exists Logic

The script checks for **all three formats** before skipping:

```bash
# Figure is considered complete only if ALL formats exist:
fig01_reference_geometry.png
fig01_reference_geometry.pdf
fig01_reference_geometry.svg
```

To force regeneration:
```bash
# Delete ALL formats for a figure
rm $DATA/runs/publication_figures/main/fig01_reference_geometry.*
```

## Manifest Format

**File:** `manifests/figure_manifest.json`

```json
{
  "generated_at": "2026-03-22T15:30:00",
  "main_figures": {
    "fig01_reference_geometry": {
      "png": "main/fig01_reference_geometry.png",
      "pdf": "main/fig01_reference_geometry.pdf",
      "svg": "main/fig01_reference_geometry.svg"
    }
  },
  "supplementary_figures": { ... },
  "formats": ["png", "pdf", "svg"],
  "dpi": 300
}
```

## Generation Log

**File:** `manifests/generation_log.jsonl`

Each figure generation attempt is logged:

```json
{"figure_id": "fig01_reference_geometry", "status": "generated", "timestamp": "2026-03-22T15:30:00"}
{"figure_id": "fig02_training_curves", "status": "skipped", "timestamp": "2026-03-22T15:31:00"}
{"figure_id": "fig03_spatial_backends", "status": "missing_source", "timestamp": "2026-03-22T15:32:00"}
```

## Troubleshooting

### Missing Source Data

**Error:** `FileNotFoundError: results.json not found`

**Solution:**
```bash
# Check prerequisite stages completed
ls $DATA/runs/v1_complete/results.json
ls $DATA/runs/ablations/ablation_heatmap.png
ls $DATA/processed/luad_evo/reference_geometry/fused_embedding.parquet
```

### Import Errors

**Error:** `ModuleNotFoundError: No module named 'stagebridge'`

**Solution:** Verify conda environment activation in sbatch script:
```bash
source /admin/software/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/chaunzt1/stagebridge_env
```

### Incomplete Figures

**Error:** Figure exists but missing PDF or SVG

**Solution:**
```bash
# Delete all formats and regenerate
rm $DATA/runs/publication_figures/main/fig01_reference_geometry.*
sbatch scripts/hpc_step7_figures.sbatch
```

### Memory Issues

**Error:** Job killed due to OOM

**Solution:**
- Current memory: 128G (should be sufficient)
- If needed, increase: `#SBATCH --mem=256G`
- Check for large datasets being loaded into memory

## Publication Theme

All figures use the centralized publication theme system:

```python
from stagebridge.viz import setup_publication_plotting
setup_publication_plotting()
```

**Style standards:**
- Pure white backgrounds (#FFFFFF)
- 300 DPI for raster formats
- Colorblind-friendly stage palette
- Top/right spines removed
- Consistent fonts (10-14pt)
- Multi-format export

## For Notebook Assembly

When building the master notebook (Section 7):

1. **Load pre-generated figures** (don't regenerate):
   ```python
   from IPython.display import Image
   Image(filename='$DATA/runs/publication_figures/main/fig01_reference_geometry.png')
   ```

2. **Use figure manifest** to locate all outputs:
   ```python
   import json
   with open('$DATA/runs/publication_figures/manifests/figure_manifest.json') as f:
       manifest = json.load(f)
   ```

3. **Display with captions** as needed for paper

## Related Documentation

- `stagebridge/viz/PUBLICATION_PLOTTING.md` - Full visualization guide
- `stagebridge/viz/publication_theme.py` - Theme system implementation
- `.claude/agent-memory/publication-plot/MEMORY.md` - Agent memory

## Contact

For issues with figure generation:
1. Check the job log: `/home/booka/StageBridge/logs/figures_<job_id>.log`
2. Review the generation log: `manifests/generation_log.jsonl`
3. Verify prerequisites are complete
4. Check publication plot agent memory for known issues
