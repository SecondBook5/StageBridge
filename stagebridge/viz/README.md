# StageBridge Visualization Modules

## viz/ vs visualization/

This project has **two** visualization directories with different purposes:

### `stagebridge/viz/` (this directory)
**Purpose:** Core plotting utilities and publication-quality figures

- `publication_theme.py` - Publication styling and rcParams
- `advanced_plots.py` - Radar charts, parallel coordinates, ridge plots
- `embeddings.py` - UMAP, PCA, t-SNE with density contours
- `flows.py` - Sankey diagrams for transitions
- `curves.py` - Training curves with confidence bands
- `spatial.py` - Spatial heatmaps
- `lungpca_style.py` - Canonical color palettes (LungPCA)

**Usage:**
```python
from stagebridge.viz import setup_publication_plotting, save_publication_figure
from stagebridge.viz.lungpca_style import STAGE_COLORS
```

### `stagebridge/visualization/`
**Purpose:** High-level figure generation for the paper/notebook

- `figure_generation.py` - Multi-panel publication figures
- `professional_figures.py` - Pre-composed figure templates
- `plot_cache.py` - Cached dimensionality reduction computations

**Usage:**
```python
from stagebridge.visualization import generate_dr_figure, generate_attention_figure
```

## Summary

| Module | Level | Purpose |
|--------|-------|---------|
| `viz/` | Low-level | Individual plot components |
| `visualization/` | High-level | Assembled multi-panel figures |
