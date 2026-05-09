"""Transition modeling for StageBridge.

The transition module implements:
- OT-CFM (Optimal Transport Conditional Flow Matching) for deterministic transitions
- Schrödinger Bridge for stochastic transitions with branching/reversibility

OT-CFM learns the mean velocity field - good for identifying drift direction.
Schrödinger Bridge learns the full distribution of paths - good for EMT/MET
branching and transition probability estimation.
"""

from stagebridge.transition.drift import (
    CrossAttentionDrift,
    FiLMConditioner,
    EdgeConditionedDriftMLP,
)
from stagebridge.transition.losses import (
    pairwise_squared_euclidean,
    build_sinkhorn_coupling,
    sinkhorn_distance,
    sample_coupling_pairs,
    random_pair_indices,
    flow_matching_loss,
    multihop_consistency_loss,
)
from stagebridge.transition.couplings import (
    build_cost_matrix,
    build_sinkhorn_coupling_from_cost,
    build_ot_coupling,
)
from stagebridge.transition.schrodinger_bridge import (
    SchrodingerBridge,
    SchrodingerBridgeConfig,
    SchrodingerBridgeWrapper,
    schrodinger_bridge_loss,
    sb_ot_coupled_loss,
    get_dynamics_module,
)

__all__ = [
    # Drift networks
    "CrossAttentionDrift",
    "FiLMConditioner",
    "EdgeConditionedDriftMLP",
    # Losses
    "pairwise_squared_euclidean",
    "build_sinkhorn_coupling",
    "sinkhorn_distance",
    "sample_coupling_pairs",
    "random_pair_indices",
    "flow_matching_loss",
    "multihop_consistency_loss",
    # Couplings
    "build_cost_matrix",
    "build_sinkhorn_coupling_from_cost",
    "build_ot_coupling",
    # Schrödinger Bridge
    "SchrodingerBridge",
    "SchrodingerBridgeConfig",
    "SchrodingerBridgeWrapper",
    "schrodinger_bridge_loss",
    "sb_ot_coupled_loss",
    "get_dynamics_module",
]
