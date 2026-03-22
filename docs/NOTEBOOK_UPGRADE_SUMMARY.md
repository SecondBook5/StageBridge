# StageBridge V1 Notebook Publication Figure Upgrade

## Status: Phase 2 Complete (High-Priority Cells Upgraded)

### Completed Upgrades

#### 1. Cell 2: Visualization Setup ✓
**Changes Made:**
- Replaced manual matplotlib configuration with official `configure_research_style()`
- Imported canonical `STAGE_COLORS` from `stagebridge.viz.embeddings`
- Added imports for all advanced visualization functions:
  - `plot_radar_chart`, `plot_parallel_coordinates`, `plot_ridge_distributions`
  - `plot_macroflow_sankey` from viz/flows
  - `plot_training_curves`, `plot_metric_violin` from viz/curves
- Configured pure white background (journal standard)
- Set proper DPI (150 display, 300 save)
- Added utility functions: `save_figure()`, `add_panel_label()`, etc.

**Key Improvements:**
- Uses official StageBridge viz modules (no duplication)
- Canonical colorblind-safe stage colors
- Consistent with project-wide visualization standards
- Publication-ready configuration (300 DPI, white background)

#### 2. Cell 15: Cell Type Distribution ✓
**Upgraded Features:**
- Panel A: Horizontal bar chart for cell type abundance (sorted)
- Panel B: Stacked bar chart showing cell type composition by stage (top 10 types)
- Panel C: Cell type diversity by stage (unique type count per stage)
- Separate ridge plot showing log-scale abundance distribution by stage
- Uses `plot_ridge_distributions()` from official viz module
- Pure white background, 300 DPI, canonical STAGE_COLORS

#### 3. Cell 17: Stage Centroids with Advanced UMAP ✓
**Upgraded Features:**
- Panel A: UMAP with density contours, convex hulls, confidence ellipses, and centroid markers
- Panel B: Inter-stage centroid distance matrix (heatmap with annotations)
- Panel C: Within-stage variance showing stage compactness
- Implements scipy.stats.gaussian_kde for density estimation
- scipy.spatial.ConvexHull for boundary visualization
- matplotlib.patches.Ellipse for 95% confidence regions
- Publication-quality styling throughout

#### 4. Cell 18: Niche Influence Vectors ✓
**Upgraded Features:**
- Primary: Radar chart using `plot_radar_chart()` from official viz module
- Supplementary: Heatmap showing influence dimensions across cell types
- Highlights influential cell types with gold star markers
- Normalized display for cross-cell-type comparison
- Dual output format (PNG + PDF)

#### 5. Cell 19: Stage Transitions ✓
**Upgraded Features:**
- Primary: Sankey diagram using `plot_macroflow_sankey()` from official viz module
- Supplementary: Transition probability matrix heatmap with annotations
- Computed from drift field ground truth
- Shows dominant transitions (>10% probability)
- Falls back to heatmap if Plotly unavailable

#### 6. Cell 30: Training Curves ✓
**Upgraded Features:**
- Uses `plot_training_curves()` from official viz module
- Adds smoothed overlay for noisy curves
- Marks best validation loss with gold star
- Optional supplementary panels for fold distributions
- Automatic log-scale for wide loss ranges
- Includes training effectiveness metrics and overfitting detection

### Remaining Medium-Priority Upgrades

#### Cell 14: Donor Structure Clustered Heatmap
**Current:** Has heatmap, needs better clustering
**Enhancement:** Use seaborn.clustermap() with dendrograms
```python
import seaborn as sns

# Create stage x donor matrix
stage_donor = cells_df.groupby(['stage', 'donor_id']).size().unstack(fill_value=0)

# Clustered heatmap
g = sns.clustermap(
    stage_donor,
    cmap='viridis',
    figsize=(12, 8),
    dendrogram_ratio=0.15,
    cbar_pos=(0.02, 0.8, 0.03, 0.15),
    row_colors=[STAGE_COLORS[s] for s in stage_donor.index],
    linewidths=0.5,
    linecolor='white',
    method='ward',
    metric='correlation'
)

g.ax_heatmap.set_xlabel('Donor ID', fontsize=12, fontweight='bold')
g.ax_heatmap.set_ylabel('Stage', fontsize=12, fontweight='bold')
```

#### Cell 27: HPO Results - Parallel Coordinates
Use `plot_parallel_coordinates()` for hyperparameter visualization

#### Cell 36: Embeddings Multi-Panel
Enhance with consistent styling across all DR methods

#### Cell 38: Attention Analysis
Add clustered heatmap + radar chart for token importance

## Implementation Notes

### Common Patterns
1. **Always use canonical STAGE_COLORS** (imported from viz/embeddings)
2. **Always save dual-format** (PNG @ 300 DPI + PDF vector)
3. **Use official viz functions** instead of reimplementing
4. **White background** for all figures (journal standard)
5. **Proper legends** outside plot area when possible

### API Signatures Used

**plot_radar_chart:**
```python
plot_radar_chart(
    df: pd.DataFrame,
    metrics: list[str],
    labels_col: str = "label",
    output_path: Path | None = None,
    title: str = "Multi-Metric Comparison",
    normalize: bool = True,
) -> plt.Figure
```

**plot_ridge_distributions:**
```python
plot_ridge_distributions(
    data_dict: dict[str, np.ndarray],
    output_path: Path | None = None,
    title: str = "Distribution Comparison",
    colors: list[str] | None = None,
) -> plt.Figure
```

**plot_macroflow_sankey:**
```python
plot_macroflow_sankey(
    flow_matrix: np.ndarray,
    source_labels: list[str],
    target_labels: list[str],
    output_path: Path,
    title: str = "Macro Coupling Sankey",
) -> None
```

**plot_training_curves:**
```python
plot_training_curves(
    history_payloads: list[dict[str, object]],
    output_path: Path,
    show_smoothed: bool = True,
) -> None
```

### Testing Checklist
For each upgraded cell:
- [x] Figure renders without errors
- [x] Colors match canonical palette
- [x] Saved at 300 DPI
- [x] Both PNG and PDF exported
- [x] Legends readable and well-positioned
- [x] No overlapping elements
- [x] Professional appearance

## Next Steps

1. ~~**Phase 2:** Upgrade high-priority cells (18, 19, 30, 15, 17)~~ ✓ COMPLETE
2. **Phase 3:** Enhance medium-priority cells (14, 27, 36, 38)
3. **Phase 4:** Full notebook smoke test
4. **Phase 5:** Visual QA and final polish

## Files Modified
- `/home/booka/projects/StageBridge/StageBridge_V1.ipynb`
  - Cell 2: Visualization setup ✓
  - Cell 15: Cell type distribution ✓
  - Cell 17: Stage centroids (advanced UMAP) ✓
  - Cell 18: Niche influence (radar chart) ✓
  - Cell 19: Transitions (Sankey diagram) ✓
  - Cell 30: Training curves ✓

## Files Referenced
- `stagebridge/viz/research_frontend.py` - Style configuration
- `stagebridge/viz/embeddings.py` - STAGE_COLORS, UMAP utilities
- `stagebridge/viz/advanced_plots.py` - Radar, parallel coords, ridge plots
- `stagebridge/viz/flows.py` - Sankey diagrams
- `stagebridge/viz/curves.py` - Training curves, violins
- `stagebridge/viz/spatial.py` - Spatial heatmaps

## Notebook Cell Structure (43 cells total)
- Cell 2: Visualization setup ✓ UPGRADED
- Cell 13: Stage distribution
- Cell 14: Donor structure (needs clustered heatmap)
- Cell 15: Cell type distribution ✓ UPGRADED
- Cell 17: Ground truth centroids ✓ UPGRADED
- Cell 18: Niche influence ✓ UPGRADED
- Cell 19: Transitions ✓ UPGRADED
- Cell 20: Spatial structure
- Cell 27: HPO results
- Cell 30: Training curves ✓ UPGRADED
- Cell 36: Embeddings (needs enhancement)
- Cell 37: PHATE trajectory
- Cell 38: Attention analysis (needs major upgrade)
