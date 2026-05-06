"""Pipeline scripts for StageBridge.

Available pipelines:
    - generate_figures: Publication figure generation
    - run_biological_validation: Biological validation analysis
"""

from stagebridge.pipelines.generate_figures import (
    generate_all_figures,
    generate_flux_figures,
    generate_gw_figures,
    generate_training_figures,
    generate_ablation_figures,
    generate_architecture_figure,
    generate_embedding_flow_figure,
    generate_biological_figure,
    generate_phase_portrait_figure,
    generate_trajectories_figure,
    generate_spatial_attention_figure,
    generate_novel_biology_figure,
)

__all__ = [
    "generate_all_figures",
    "generate_flux_figures",
    "generate_gw_figures",
    "generate_training_figures",
    "generate_ablation_figures",
    "generate_architecture_figure",
    "generate_embedding_flow_figure",
    "generate_biological_figure",
    "generate_phase_portrait_figure",
    "generate_trajectories_figure",
    "generate_spatial_attention_figure",
    "generate_novel_biology_figure",
]
