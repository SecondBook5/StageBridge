"""StageBridge model components."""

from stagebridge.models.heads import (
    AuxiliaryHead,
    PathwayHead,
    ProliferationHead,
)
from stagebridge.models.stagebridge import (
    StageBridge,
    StageBridgeConfig,
    StageBridgeOutput,
)

__all__ = [
    "StageBridge",
    "StageBridgeConfig",
    "StageBridgeOutput",
    # Auxiliary heads (only orthogonal signals - see design_conditioning_vs_encoding.md)
    "AuxiliaryHead",
    "PathwayHead",
    "ProliferationHead",
]
