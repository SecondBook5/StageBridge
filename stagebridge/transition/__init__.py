"""Transition modeling for StageBridge.

The transition module implements OT-CFM (Optimal Transport Conditional
Flow Matching) for learning stage transitions conditioned on niche context.
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
]
