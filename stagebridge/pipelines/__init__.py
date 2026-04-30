"""Pipeline scripts for StageBridge.

Available pipelines:
    - generate_figures: Publication figure generation
    - run_biological_validation: Biological validation analysis
"""

from stagebridge.pipelines.generate_figures import (
    generate_architecture,
    generate_training,
    generate_ablations,
    generate_embedding_flow,
    generate_biological,
    generate_phase_portrait,
    generate_trajectories,
    generate_spatial_attention,
    generate_novel_biology,
)

__all__ = [
    "generate_architecture",
    "generate_training",
    "generate_ablations",
    "generate_embedding_flow",
    "generate_biological",
    "generate_phase_portrait",
    "generate_trajectories",
    "generate_spatial_attention",
    "generate_novel_biology",
]
