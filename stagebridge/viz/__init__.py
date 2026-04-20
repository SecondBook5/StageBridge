"""Visualisation utilities for StageBridge."""

from .curves import (
    build_metrics_dataframe,
    plot_benchmark_bars,
    plot_training_curves,
    plot_metric_violin,
)
from .embeddings import (
    plot_context_vector_umap,
    plot_umap_by_stage,
    plot_umap_with_trajectories,
)
from .eamist_figures import (
    save_ablation_figure,
    save_benchmark_comparison_figure,
    save_embedding_diagnostics_figure,
    save_method_overview_figure,
    save_prototype_interpretation_figure,
)
from .flows import compute_macroflow_matrix, plot_macroflow_sankey
from .research_frontend import (
    configure_research_style,
    plot_context_frontend,
    plot_multi_embedding_frontend,
    plot_reference_frontend,
    plot_spatial_mapping_frontend,
    plot_transformer_attention_frontend,
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
from .advanced_plots import (
    plot_radar_chart,
    plot_parallel_coordinates,
    plot_correlation_matrix,
    plot_3d_embedding,
    plot_ridge_distributions,
)
from .niche_influence import (
    plot_niche_beeswarm,
    plot_niche_importance_bar,
    plot_niche_stage_heatmap,
    plot_niche_influence_combined,
    from_interpretability,
)
from .transition_flow import (
    compute_transition_flows,
    plot_sankey_flow,
    plot_alluvial as plot_transition_alluvial,
)
from .spatial_niche import (
    compute_niche_scores,
    get_spatial_coords,
    plot_spatial_risk_map,
    plot_niche_composition_map,
    plot_spatial_niche_combined,
)
from .dual_reference import (
    compute_reduced_embedding,
    plot_3d_trajectory,
    plot_reference_contribution,
    plot_flow_field,
    plot_dual_reference_combined,
)
from .ablation_waterfall import (
    compute_degradation,
    plot_ablation_waterfall,
    plot_multi_metric_waterfall,
    plot_component_importance,
    plot_ablation_summary,
    ABLATION_LABELS,
    COMPONENT_GROUPS,
)
from .publication_theme import (
    configure_publication_style,
    save_publication_figure,
    get_stage_color,
    apply_clean_spines,
    add_clean_legend,
    create_figure,
    create_subplots,
    setup_publication_plotting,
    PUBLICATION_PALETTE,
)
from .lungpca_style import (
    # Color palettes
    STAGE_COLORS,
    STAGE_COLORS_ALT,
    EPITHELIAL_COLORS,
    STROMAL_COLORS,
    MAJOR_CELLTYPE_COLORS,
    MP_COLORS,
    CLONE_COLORS,
    LINEAGE_COLORS,
    STAGE_ORDER,
    # Configuration
    configure_lungpca_style,
    create_lungpca_figure,
    save_lungpca_figure,
    # Color getters
    get_celltype_color,
    get_mp_color,
    get_stage_colors_list,
    get_stage_cmap,
    # Colormaps
    get_magma_white,
    get_rdbu_diverging,
    get_turbo_truncated,
    get_correlation_cmap,
    EXPRESSION_CMAP_COLORS,
    # Plot types
    plot_violin_boxplot,
    plot_stacked_bar,
    plot_pie_chart,
    plot_boxplot_jitter,
    plot_heatmap,
    plot_spatial_hexbin,
    plot_spatial_categorical,
    plot_sankey_diagram,
    plot_alluvial,
)
from .lungpca_figures import (
    # Figure recreations
    figure_1b_sankey,
    figure_1c_umap,
    figure_3a_celltype_umap,
    figure_3h_stage_boxplot,
    figure_3l_alluvial,
    figure_4b_correlation,
    figure_5d_neighborhood,
    figure_5e_violin,
    generate_stagebridge_figure_panel,
)

# Merged from visualization/ module
from .figure_generation import (
    generate_figure1_architecture,
    generate_figure2_dimensionality_reduction,
    generate_figure3_niche_influence_biology,
    generate_figure4_model_performance,
    generate_figure5_attention_patterns,
)
from .individual_plots import (
    plot_confusion_matrix,
    plot_loss_curve,
    plot_pca_with_variance,
    plot_tsne,
    plot_umap,
)
from .plot_cache import clear_cache, get_cache
from .professional_figures import (
    generate_figure2_dimensionality_reduction as generate_fig2_pro,
    generate_figure4_model_performance as generate_fig4_pro,
)

__all__ = [
    # curves
    "build_metrics_dataframe",
    "plot_benchmark_bars",
    "plot_training_curves",
    "save_ablation_figure",
    "save_benchmark_comparison_figure",
    "save_embedding_diagnostics_figure",
    "save_method_overview_figure",
    "save_prototype_interpretation_figure",
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
    "plot_multi_embedding_frontend",
    "plot_transformer_attention_frontend",
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
    # publication theme
    "configure_publication_style",
    "save_publication_figure",
    "get_stage_color",
    "apply_clean_spines",
    "add_clean_legend",
    "create_figure",
    "create_subplots",
    "setup_publication_plotting",
    "PUBLICATION_PALETTE",
    # advanced plots
    "plot_radar_chart",
    "plot_parallel_coordinates",
    "plot_correlation_matrix",
    "plot_3d_embedding",
    "plot_ridge_distributions",
    # LungPCA style (publication-quality matching original paper)
    "STAGE_COLORS",
    "STAGE_COLORS_ALT",
    "EPITHELIAL_COLORS",
    "STROMAL_COLORS",
    "MAJOR_CELLTYPE_COLORS",
    "MP_COLORS",
    "CLONE_COLORS",
    "LINEAGE_COLORS",
    "STAGE_ORDER",
    "configure_lungpca_style",
    "create_lungpca_figure",
    "save_lungpca_figure",
    "get_celltype_color",
    "get_mp_color",
    "get_stage_colors_list",
    "get_stage_cmap",
    "get_magma_white",
    "get_rdbu_diverging",
    "get_turbo_truncated",
    "get_correlation_cmap",
    "EXPRESSION_CMAP_COLORS",
    "plot_violin_boxplot",
    "plot_stacked_bar",
    "plot_pie_chart",
    "plot_boxplot_jitter",
    "plot_heatmap",
    "plot_spatial_hexbin",
    "plot_spatial_categorical",
    "plot_sankey_diagram",
    "plot_alluvial",
    # LungPCA figure recreations
    "figure_1b_sankey",
    "figure_1c_umap",
    "figure_3a_celltype_umap",
    "figure_3h_stage_boxplot",
    "figure_3l_alluvial",
    "figure_4b_correlation",
    "figure_5d_neighborhood",
    "figure_5e_violin",
    "generate_stagebridge_figure_panel",
    # From merged visualization/ module
    "generate_figure1_architecture",
    "generate_figure2_dimensionality_reduction",
    "generate_figure3_niche_influence_biology",
    "generate_figure4_model_performance",
    "generate_figure5_attention_patterns",
    "plot_confusion_matrix",
    "plot_loss_curve",
    "plot_pca_with_variance",
    "plot_tsne",
    "plot_umap",
    "clear_cache",
    "get_cache",
    "generate_fig2_pro",
    "generate_fig4_pro",
    # Niche influence (SHAP-style)
    "plot_niche_beeswarm",
    "plot_niche_importance_bar",
    "plot_niche_stage_heatmap",
    "plot_niche_influence_combined",
    "from_interpretability",
    # Transition flow
    "compute_transition_flows",
    "plot_sankey_flow",
    "plot_transition_alluvial",
    # Spatial niche
    "compute_niche_scores",
    "get_spatial_coords",
    "plot_spatial_risk_map",
    "plot_niche_composition_map",
    "plot_spatial_niche_combined",
    # Dual reference
    "compute_reduced_embedding",
    "plot_3d_trajectory",
    "plot_reference_contribution",
    "plot_flow_field",
    "plot_dual_reference_combined",
    # Ablation waterfall
    "compute_degradation",
    "plot_ablation_waterfall",
    "plot_multi_metric_waterfall",
    "plot_component_importance",
    "plot_ablation_summary",
    "ABLATION_LABELS",
    "COMPONENT_GROUPS",
]
