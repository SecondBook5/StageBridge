"""StageBridge: Receiver-centered niche modeling for cancer progression."""

from stagebridge.contracts import (
    STAGES_3,
    STAGES_4,
    STAGES_5,
    STAGES,
    N_STAGES,
    LATENT_DIM,
    HLCA_DIM,
    LUCA_DIM,
    N_TOKENS,
    TOKEN_NAMES,
    WES_COLS,
    WES_DIM,
)
from stagebridge.models import StageBridge, StageBridgeConfig, StageBridgeOutput

__version__ = "1.0.0"
__all__ = [
    # Model
    "StageBridge",
    "StageBridgeConfig",
    "StageBridgeOutput",
    # Constants
    "STAGES_3",
    "STAGES_4",
    "STAGES_5",
    "STAGES",
    "N_STAGES",
    "LATENT_DIM",
    "HLCA_DIM",
    "LUCA_DIM",
    "N_TOKENS",
    "TOKEN_NAMES",
    "WES_COLS",
    "WES_DIM",
]
