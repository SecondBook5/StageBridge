# StageBridge Visualization Enhancements

## Summary of Improvements

All visualization modules have been enhanced with advanced plotting features, publication-quality styling, and better visual communication. Below is a detailed breakdown of the improvements.

---

## 1. Enhanced Embeddings Module (`embeddings.py`)

### New Features:
- **Color-blind friendly palette**: Switched to scientifically validated color schemes
- **Density contours**: Added KDE-based density visualization for overall distributions
- **Convex hulls**: Optional overlay showing cluster boundaries
- **Confidence ellipses**: 95% confidence interval ellipses for stage clusters
- **Statistical annotations**: Cell counts and summary statistics
- **Improved legends**: Enhanced with stage counts and better styling
- **Higher DPI**: Default 300 DPI for publication quality

### Enhanced Functions:
- `plot_umap_by_stage()`: Now includes density contours, convex hulls, and confidence ellipses
- `plot_umap_with_trajectories()`: Better arrow rendering with gradient effects, dual legends
- `plot_context_vector_umap()`: Filled density contours and statistical annotations

### Key Visual Improvements:
- White point edges for better visibility
- Subtle background grid with dotted lines
- Removed unnecessary spines (top and right)
- Increased font sizes and weights for titles/labels
- Better aspect ratios and figure sizes

---

## 2. Enhanced Benchmark & Curves Module (`curves.py`)

### New Features:
- **Statistical significance markers**: Automatic highlighting of best models
- **Value annotations**: Display metric values on bars
- **Smoothed training curves**: Optional smoothing for noisy data
- **Best validation marker**: Gold star highlighting best epoch
- **Logarithmic scale**: Auto-detection for multi-order-of-magnitude ranges
- **Violin plots**: New `plot_metric_violin()` function for distribution comparisons

### Enhanced Functions:
- `plot_benchmark_bars()`: Added value annotations, best-model highlighting, gradient effects
- `plot_training_curves()`: Smoothing, best-val markers, better color schemes
- New `plot_metric_violin()`: Seaborn-based violin plots with swarm overlays

### Key Visual Improvements:
- Stronger error bars (linewidth 2)
- Reference lines for best performance
- Shadow effects on legends
- Better color coding by model type
- Improved grid styling

---

## 3. Enhanced Spatial Visualization Module (`spatial.py`)

### New Features:
- **Hierarchical clustering**: Dendrogram for metric heatmaps
- **Improved heatmaps**: Better colormaps (RdBu_r), value annotations, white grid lines
- **Scale bars**: Physical distance indicators on spatial plots
- **Contour overlays**: Density contours on context score maps
- **Summary statistics**: Annotation boxes with mean/median/range
- **Stage counts**: Legend includes sample counts per stage

### Enhanced Functions:
- `plot_metric_heatmap()`: Hierarchical clustering, improved colormaps, better annotations
- `plot_spatial_stage_map()`: Scale bars, stage counts in legend, better point styling
- `plot_spatial_context_score()`: Contours, statistics box, enhanced colorbars

### Key Visual Improvements:
- Color-blind friendly stage palette
- Enhanced colorbars with bold labels
- White point edges for spatial plots
- Better grid styling (dotted lines)
- Physical units (μm) in axis labels

---

## 4. New Advanced Plotting Module (`advanced_plots.py`)

### New Visualization Types:

1. **Radar/Spider Charts** (`plot_radar_chart`):
   - Multi-metric comparison across models
   - Normalized metrics with fill areas
   - Publication-quality polar plots

2. **Parallel Coordinates** (`plot_parallel_coordinates`):
   - High-dimensional metric visualization
   - Color-coded model trajectories
   - Normalized scales for comparison

3. **Correlation Matrices** (`plot_correlation_matrix`):
   - Pearson, Spearman, Kendall correlations
   - RdBu_r colormap with value annotations
   - White grid lines for clarity

4. **3D Embeddings** (`plot_3d_embedding`):
   - 3D scatter plots with rotation
   - Color-coded by labels
   - Publication-quality 3D rendering

5. **Ridge Plots** (`plot_ridge_distributions`):
   - Distribution comparison (joyplots)
   - KDE overlays on histograms
   - Stacked layout for multiple groups

### Use Cases:
- Multi-metric model comparison
- Distribution analysis
- High-dimensional data exploration
- Alternative embedding visualizations

---

## 5. Enhanced Summary Panels Module (`summary_panels.py`)

### Key Enhancements:

1. **Panel B (Benchmark)**:
   - Value annotations with rounded boxes
   - Best performance reference line
   - Enhanced legend with shadows
   - Stronger error bars
   - Gradient bar effects

2. **Panel C (Context Sensitivity)**:
   - Color gradient by sensitivity score
   - Star markers for peak values
   - Biological significance annotations
   - Colorbar showing sensitivity scale
   - Enhanced styling for maximum impact

### Key Visual Improvements:
- Higher DPI (150-300)
- Thicker spines (linewidth 2)
- Fancybox legends with shadows
- Better annotation boxes
- Gradient effects on bars

---

## Global Improvements Across All Modules

### Styling Standards:
- **DPI**: Increased from 220 to 300 for ALL plots
- **Figure sizes**: Larger (9x7.5" to 12x6.5") for better readability
- **Fonts**: Larger, bolder titles and labels
- **Spines**: Removed top/right, strengthened left/bottom (2px)
- **Grid**: Dotted lines (:) with reduced alpha
- **Colors**: Color-blind friendly palettes throughout

### File Output:
- All functions now save both PNG (300 DPI) and PDF
- White backgrounds for all figures
- Tight bounding boxes for clean exports
- Better path handling with mkdir parents

### Code Quality:
- Enhanced docstrings with parameter descriptions
- Optional features with boolean flags
- Better error handling and logging
- Type hints where applicable

---

## How to Use the Enhanced Visualizations

### Basic Example - UMAP with all features:
```python
from stagebridge.viz import plot_umap_by_stage

plot_umap_by_stage(
    adata,
    output_path="figures/umap_enhanced.png",
    show_density=True,      # KDE contours
    show_hulls=True,        # Convex hulls
    show_ellipses=True,     # Confidence ellipses
    point_size=3.0,
    alpha=0.6
)
```

### Advanced Example - Radar chart for metrics:
```python
from stagebridge.viz import plot_radar_chart

plot_radar_chart(
    df=metrics_df,
    metrics=["sinkhorn_mean", "mmd_rbf_mean", "classifier_auc_mean"],
    labels_col="model_name",
    output_path="figures/radar_comparison.png",
    normalize=True
)
```

### Spatial Example - Enhanced heatmap:
```python
from stagebridge.viz import plot_metric_heatmap

plot_metric_heatmap(
    metrics_df,
    output_path="figures/metrics_heatmap.png",
    cluster_rows=True,      # Hierarchical clustering
    show_values=True,       # Annotate cells
    figsize=(12, 8)
)
```

---

## Color Palettes

### Stage Colors (Color-blind safe):
- Normal: `#00BA38` (green)
- AAH: `#F8766D` (coral)
- AIS: `#619CFF` (blue)
- MIA: `#E58700` (orange)
- LUAD: `#A3A500` (olive)

### Model Type Colors:
- StageBridge: `#0E7490` (teal)
- Baselines: `#64748B` (medium gray)
- Ablations: `#94A3B8` (light gray)
- Accent/Highlight: `#F59E0B` or `#D97706` (amber)

---

## Performance Considerations

- **Rasterization**: Large scatter plots are rasterized for smaller file sizes
- **Caching**: KDE computations are wrapped in try/except for robustness
- **Optional features**: Heavy computations (contours, clustering) can be disabled
- **DPI settings**: High DPI only used for final outputs, lower for interactive use

---

## Requirements

All visualizations work with existing dependencies:
- matplotlib >= 3.5
- numpy >= 1.20
- pandas >= 1.3
- scipy >= 1.7

Optional (for enhanced features):
- seaborn >= 0.11 (violin plots)
- scikit-learn (dimensionality reduction)
- umap-learn (UMAP embeddings)

---

## Next Steps

1. **Test the new visualizations** with your existing data pipelines
2. **Adjust parameters** (DPI, sizes, colors) to match journal requirements
3. **Combine plots** into multi-panel figures using GridSpec
4. **Export to vector formats** (PDF, SVG) for maximum quality

All enhancements maintain backward compatibility - existing code will work with improved default styling!
