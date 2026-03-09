"""Visualisation utilities for StageBridge."""

from .curves import build_metrics_dataframe, plot_benchmark_bars, plot_training_curves
from .embeddings import (
    plot_context_vector_umap,
    plot_umap_by_stage,
    plot_umap_with_trajectories,
)
from .flows import compute_macroflow_matrix, plot_macroflow_sankey
from .research_frontend import (
    configure_research_style,
    plot_context_frontend,
    plot_reference_frontend,
    plot_spatial_mapping_frontend,
    plot_transition_frontend,
)
from .summary_panels import (
    make_full_poster,
    make_panel_a_model_diagram,
    make_panel_b_benchmark,
    make_panel_c_context_sensitivity,
    make_panel_d_gene_context_heatmap,
)
from .spatial import (
    plot_method_schematic,
    plot_metric_heatmap,
    plot_spatial_context_score,
    plot_spatial_stage_map,
    plot_tangram_celltype_maps,
    plot_tangram_winner_map,
    plot_transition_trajectory,
)
from .story_figures import (
    plot_communication_metric_panels,
    plot_context_shuffle_deltas,
    plot_label_balance,
    plot_transition_vs_communication,
)

__all__ = [
    # curves
    "build_metrics_dataframe",
    "plot_benchmark_bars",
    "plot_training_curves",
    # embeddings
    "plot_umap_by_stage",
    "plot_umap_with_trajectories",
    "plot_context_vector_umap",
    # macro flow
    "compute_macroflow_matrix",
    "plot_macroflow_sankey",
    "configure_research_style",
    "plot_reference_frontend",
    "plot_spatial_mapping_frontend",
    "plot_context_frontend",
    "plot_transition_frontend",
    # spatial plots (Visium)
    "plot_spatial_stage_map",
    "plot_spatial_context_score",
    "plot_tangram_celltype_maps",
    "plot_tangram_winner_map",
    "plot_transition_vs_communication",
    "plot_communication_metric_panels",
    "plot_context_shuffle_deltas",
    "plot_label_balance",
    # poster panels (backward-compat re-exports)
    "plot_method_schematic",
    "plot_transition_trajectory",
    "plot_metric_heatmap",
    # poster assembly
    "make_panel_a_model_diagram",
    "make_panel_b_benchmark",
    "make_panel_c_context_sensitivity",
    "make_panel_d_gene_context_heatmap",
    "make_full_poster",
]
