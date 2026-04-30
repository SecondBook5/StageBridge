"""Visualization module for publication-quality figures."""

from stagebridge.viz.theme import (
    configure_publication_style,
    STAGE_COLORS,
    CELLTYPE_COLORS,
    save_figure,
)
from stagebridge.viz.features import (
    plot_feature_distributions,
    plot_spatial_features,
    plot_umap_features,
    plot_progression_panel,
)

__all__ = [
    "configure_publication_style",
    "STAGE_COLORS",
    "CELLTYPE_COLORS",
    "save_figure",
    "plot_feature_distributions",
    "plot_spatial_features",
    "plot_umap_features",
    "plot_progression_panel",
]
