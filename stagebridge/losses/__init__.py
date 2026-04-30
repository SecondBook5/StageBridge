"""Loss functions for StageBridge.

Includes:
- manifold: Geodesic-preserving losses (isometry, CVL, velocity consistency)
"""

from stagebridge.losses.manifold import (
    LocalIsometryLoss,
    ConstantVelocityLinearLoss,
    VelocityConsistencyLoss,
    ManifoldLoss,
    compute_mmd,
    sinkhorn_ot_plan,
)

__all__ = [
    "LocalIsometryLoss",
    "ConstantVelocityLinearLoss",
    "VelocityConsistencyLoss",
    "ManifoldLoss",
    "compute_mmd",
    "sinkhorn_ot_plan",
]
