"""Interpretation module for StageBridge.

Adapted from AMICI (Hong et al., bioRxiv 2025) for:
- Visium spot-level data (not single-cell Xenium)
- Ring-based niche structure (not raw k-NN neighbors)
- Stage progression context (not static snapshot)

Key modules:
- ablation: Token ablation analysis (ring, reference, pathway ablation)
- attention: Attention pattern extraction and visualization
- networks: Cell-cell interaction network inference
- dynamics: Trajectory analysis with OT-based fate probability
- trajectory_plots: Publication-quality trajectory visualizations
"""

from stagebridge.interpretation.ablation import (
    AblationModule,
    compute_token_ablation,
)
from stagebridge.interpretation.attention import (
    AttentionModule,
    extract_attention_patterns,
)
from stagebridge.interpretation.networks import (
    InteractionNetwork,
    build_interaction_network,
)
from stagebridge.interpretation.plotting import (
    plot_interaction_network,
    plot_interaction_heatmap,
    plot_ring_attention_decay,
    plot_ablation_importance,
    plot_reference_balance,
    plot_stage_network_comparison,
)
from stagebridge.interpretation.dynamics import (
    TrajectoryAnalysis,
    FateProbability,
    DynamicDriverResult,
    cluster_driver_genes,
)
from stagebridge.interpretation.trajectory_plots import (
    plot_temporal_evolution,
    plot_fate_probability,
    plot_single_cell_trajectories,
    plot_driver_heatmap,
    plot_gene_dynamics,
    create_trajectory_animation,
)
from stagebridge.interpretation.manifold_viz import (
    plot_manifold_comparison,
    plot_multi_method_comparison,
    plot_trajectory_straightness,
    plot_geodesic_comparison,
    plot_phase_map,
    plot_phase_portrait_grid,
    compute_manifold_comparison,
    ManifoldComparisonResult,
)

__all__ = [
    # Ablation
    "AblationModule",
    "compute_token_ablation",
    # Attention
    "AttentionModule",
    "extract_attention_patterns",
    # Networks
    "InteractionNetwork",
    "build_interaction_network",
    # Plotting (AMICI-style)
    "plot_interaction_network",
    "plot_interaction_heatmap",
    "plot_ring_attention_decay",
    "plot_ablation_importance",
    "plot_reference_balance",
    "plot_stage_network_comparison",
    # Dynamics (trajectory analysis)
    "TrajectoryAnalysis",
    "FateProbability",
    "DynamicDriverResult",
    "cluster_driver_genes",
    # Trajectory plots
    "plot_temporal_evolution",
    "plot_fate_probability",
    "plot_single_cell_trajectories",
    "plot_driver_heatmap",
    "plot_gene_dynamics",
    "create_trajectory_animation",
    # Manifold visualization
    "plot_manifold_comparison",
    "plot_multi_method_comparison",
    "plot_trajectory_straightness",
    "plot_geodesic_comparison",
    "plot_phase_map",
    "plot_phase_portrait_grid",
    "compute_manifold_comparison",
    "ManifoldComparisonResult",
]
