# Publication Plotting Infrastructure

This document describes the publication-quality plotting utilities available in StageBridge for the notebook-assembly agent and downstream users.

## Quick Start

```python
from stagebridge.viz import setup_publication_plotting, create_figure, save_publication_figure

# One-line setup (call once at start of script/notebook)
setup_publication_plotting()

# Create figure with clean publication styling
fig, ax = create_figure(figsize=(10, 8))
ax.plot([1, 2, 3], [1, 4, 9])
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_title("My Publication Figure")

# Save in multiple formats (PNG 300 DPI, PDF, SVG)
save_publication_figure(fig, "output/figure1")
```

## Core Modules

### 1. `publication_theme.py` - Publication Style System

**Key Functions:**
- `setup_publication_plotting()` - One-line setup for publication figures
- `configure_publication_style()` - Configure matplotlib rcParams
- `create_figure(figsize, dpi)` - Create single-panel figure
- `create_subplots(nrows, ncols)` - Create multi-panel figure
- `save_publication_figure(fig, path, formats)` - Multi-format export
- `get_stage_color(stage)` - Colorblind-friendly stage colors
- `apply_clean_spines(ax)` - Remove top/right spines
- `add_clean_legend(ax, title)` - Publication-quality legend

**Style Settings:**
- Pure white background (`#FFFFFF`)
- 300 DPI for saved figures
- Font sizes: 10pt (body), 12pt (labels), 14pt (titles)
- Bold axis labels and titles
- Top/right spines removed
- Colorblind-friendly stage palette

**Stage Colors (Colorblind-Safe):**
```python
from stagebridge.viz import PUBLICATION_PALETTE

PUBLICATION_PALETTE = {
    "Normal": "#00BA38",  # green
    "AAH": "#F8766D",     # coral
    "AIS": "#619CFF",     # blue
    "MIA": "#E58700",     # orange
    "LUAD": "#A3A500",    # olive
    "Unknown": "#999999", # gray
}
```

### 2. `research_frontend.py` - Notebook Styling

**Enhanced Function:**
```python
from stagebridge.viz import configure_research_style

# Notebook-friendly style (off-white background)
configure_research_style(publication=False)

# Publication style (pure white background, 300 DPI)
configure_research_style(publication=True)
```

**Features:**
- Dual-mode: notebook vs. publication
- Configurable font scaling
- Proper DPI settings (300 for publication)
- Consistent legend styling

### 3. `embeddings.py` - Advanced UMAP Visualization

**Available Functions:**
- `plot_umap_by_stage()` - UMAP scatter with stage colors
  - Density contours ✓
  - Convex hulls ✓
  - Confidence ellipses ✓
  - 300 DPI export ✓

- `plot_umap_with_trajectories()` - UMAP with trajectory arrows
  - Density contours ✓
  - Stage-colored background ✓
  - Trajectory arrows with proper styling ✓

- `plot_context_vector_umap()` - Context embedding visualization
  - Density contours ✓
  - Convex hulls ✓
  - Confidence ellipses ✓
  - Statistical annotations ✓

**All embedding functions:**
- Save at 300 DPI
- Export PNG + PDF automatically
- Use pure white backgrounds
- Apply colorblind-friendly palette

### 4. `advanced_plots.py` - Specialized Visualizations

**Available Functions:**
- `plot_radar_chart()` - Multi-metric radar plots
- `plot_parallel_coordinates()` - High-dimensional comparisons
- `plot_ridge_distributions()` - Distribution overlays (joyplots)
- `plot_correlation_matrix()` - Correlation heatmaps
- `plot_3d_embedding()` - 3D scatter plots

**All advanced plots:**
- Save at 300 DPI ✓
- Export PNG + PDF ✓
- Use white backgrounds ✓
- Proper font sizes ✓

### 5. `flows.py` - Sankey Diagrams

**Functions:**
- `compute_macroflow_matrix()` - Cluster-level flow computation
- `plot_macroflow_sankey()` - Stage transition Sankey diagrams

**Enhanced Features:**
- Plotly-based Sankey (primary)
- Publication-quality matplotlib fallback
- 300 DPI export ✓
- PDF export ✓
- Pure white backgrounds ✓

## Usage Examples

### Example 1: Simple Publication Figure

```python
from stagebridge.viz import (
    setup_publication_plotting,
    create_figure,
    save_publication_figure,
    get_stage_color,
)

# Setup
setup_publication_plotting()

# Create figure
fig, ax = create_figure(figsize=(8, 6))

# Plot data with stage colors
stages = ["Normal", "AAH", "AIS", "MIA", "LUAD"]
for stage in stages:
    x = [1, 2, 3]
    y = [i for i in range(3)]
    ax.plot(x, y, label=stage, color=get_stage_color(stage), linewidth=2)

ax.set_xlabel("Time")
ax.set_ylabel("Expression")
ax.set_title("Gene Expression Across Stages")
ax.legend(title="Stage")
ax.grid(True, alpha=0.3)

# Save in all formats
save_publication_figure(fig, "output/expression_plot")
# Creates: expression_plot.png (300 DPI), expression_plot.pdf, expression_plot.svg
```

### Example 2: Multi-Panel Figure

```python
from stagebridge.viz import create_subplots, save_publication_figure, add_clean_legend

# Create 2x2 grid
fig, axes = create_subplots(nrows=2, ncols=2, figsize=(12, 10))

# Panel A
axes[0, 0].plot([1, 2, 3], [1, 4, 9])
axes[0, 0].set_title("Panel A: Training Loss")
add_clean_legend(axes[0, 0])

# Panel B
axes[0, 1].scatter([1, 2, 3], [3, 2, 1])
axes[0, 1].set_title("Panel B: Validation Accuracy")

# Panel C & D
# ... add more panels

# Add panel labels
for i, ax in enumerate(axes.flat):
    ax.text(-0.1, 1.05, chr(65 + i), transform=ax.transAxes,
            fontsize=16, fontweight="bold")

save_publication_figure(fig, "output/figure_multiPanel")
```

### Example 3: UMAP with All Features

```python
from stagebridge.viz import plot_umap_by_stage

# Full-featured UMAP
plot_umap_by_stage(
    adata,
    output_path="output/umap_stage.png",
    title="Cell State Landscape",
    stage_col="stage",
    point_size=3.0,
    alpha=0.6,
    show_density=True,      # Density contours
    show_hulls=True,        # Convex hulls
    show_ellipses=True,     # 95% confidence ellipses
)
# Automatically saves both PNG (300 DPI) and PDF
```

### Example 4: Radar Chart Comparison

```python
import pandas as pd
from stagebridge.viz import plot_radar_chart

# Prepare metrics
df = pd.DataFrame({
    "label": ["Model A", "Model B", "Model C"],
    "accuracy": [0.85, 0.90, 0.88],
    "precision": [0.82, 0.88, 0.87],
    "recall": [0.87, 0.91, 0.86],
    "f1_score": [0.84, 0.89, 0.87],
})

fig = plot_radar_chart(
    df,
    metrics=["accuracy", "precision", "recall", "f1_score"],
    labels_col="label",
    output_path="output/model_comparison_radar.png",
    title="Model Performance Comparison",
    normalize=True,
)
```

### Example 5: Sankey Flow Diagram

```python
from stagebridge.viz import compute_macroflow_matrix, plot_macroflow_sankey

# Compute cluster-level flow
flow_matrix, src_labels, tgt_labels = compute_macroflow_matrix(
    x_src=source_embeddings,
    x_tgt_pred=predicted_embeddings,
    n_clusters=9,
)

# Plot Sankey
plot_macroflow_sankey(
    flow_matrix,
    src_labels,
    tgt_labels,
    output_path="output/stage_transitions.png",
    title="Stage Transition Flow",
)
```

## Testing

All utilities are tested in `tests/viz/test_publication_theme.py`:

```bash
python -m pytest tests/viz/test_publication_theme.py -v
```

**Test Coverage:**
- Style configuration ✓
- Multi-format export ✓
- Color palette validation ✓
- Figure creation utilities ✓
- Complete pipeline test ✓

## Checklist for Notebook-Assembly Agent

When upgrading notebook figures, ensure:

- [ ] `setup_publication_plotting()` called at notebook start
- [ ] All figures use `create_figure()` or `create_subplots()`
- [ ] Stage colors retrieved via `get_stage_color()`
- [ ] All figures saved via `save_publication_figure()`
- [ ] Multi-format export enabled (PNG, PDF, SVG)
- [ ] 300 DPI verified for PNG outputs
- [ ] Pure white backgrounds (`#FFFFFF`)
- [ ] Top/right spines removed via `apply_clean_spines()`
- [ ] Legends styled via `add_clean_legend()`
- [ ] Font sizes appropriate (10-14pt)
- [ ] Grid enabled where appropriate (with `alpha=0.3`)

## Key Files

```
stagebridge/viz/
├── publication_theme.py      # Core publication style system (NEW)
├── research_frontend.py      # Enhanced with publication mode
├── embeddings.py             # Advanced UMAP (already publication-ready)
├── advanced_plots.py         # Specialized plots (already publication-ready)
├── flows.py                  # Enhanced Sankey diagrams
├── __init__.py               # Updated exports
└── PUBLICATION_PLOTTING.md   # This file

tests/viz/
└── test_publication_theme.py # Comprehensive tests (NEW)
```

## Migration Notes

**For existing code:**
```python
# OLD (notebook style)
from stagebridge.viz import configure_research_style
configure_research_style()

# NEW (publication style)
from stagebridge.viz import setup_publication_plotting
setup_publication_plotting()
```

**For existing figures:**
```python
# OLD (manual saving)
fig.savefig("output.png", dpi=300, bbox_inches="tight")

# NEW (multi-format export)
from stagebridge.viz import save_publication_figure
save_publication_figure(fig, "output")  # Creates .png, .pdf, .svg
```

## Support

For questions or issues:
1. Check this documentation first
2. Review test examples in `tests/viz/test_publication_theme.py`
3. Check inline docstrings in `publication_theme.py`
4. Consult the publication plot agent memory

---

**Last Updated:** 2026-03-21
**Agent:** Publication Plot Agent
**Status:** Ready for notebook upgrade
