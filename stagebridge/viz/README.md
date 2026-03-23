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
from stagebridge.viz import generate_figure2_dimensionality_reduction, generate_fig2_pro
```

## Summary

All visualization code is now unified in `stagebridge/viz/`. The former `visualization/` module has been merged here.

| Submodule | Purpose |
|-----------|---------|
| Individual plots | `plot_umap`, `plot_confusion_matrix`, etc. |
| Multi-panel figures | `generate_figure1_architecture`, etc. |
| Professional figures | `generate_fig2_pro`, `generate_fig4_pro` |
| Plot cache | Cached DR computations (`get_cache`) |
