"""StageBridge configuration module."""

from stagebridge.config.paths import PathConfig, get_paths, detect_environment, print_status

# =============================================================================
# Reference Geometry Dimensions (Fixed by Upstream Models)
# =============================================================================
# These are NOT hyperparameters - they are determined by the pretrained
# reference atlas models and cannot be changed without retraining those models.
#
# HLCA (Human Lung Cell Atlas):
#   - Model: scANVI from HuggingFace (LungRef)
#   - Latent key: "X_scanvi_emb"
#   - Dimension: 30 (fixed by pretrained model)
#
# LuCA (Lung Cancer Atlas):
#   - Model: scVI from original LuCA paper
#   - Latent key: "X_scANVI"
#   - Dimension: 10 (fixed by pretrained model)
#
# Fused embedding is simple concatenation: [HLCA | LuCA]
# This preserves both healthy (HLCA) and disease-aware (LuCA) geometry.
# =============================================================================

HLCA_LATENT_DIM: int = 30  # From HLCA scANVI model (fixed)
LUCA_LATENT_DIM: int = 10  # From LuCA scVI model (fixed)
FUSED_LATENT_DIM: int = HLCA_LATENT_DIM + LUCA_LATENT_DIM  # 40 (derived)

# Number of niche tokens in the receiver-centered architecture
# Token layout: [receiver, ring1, ring2, ring3, ring4, hlca_ref, luca_ref, pathway, stats]
N_NICHE_TOKENS: int = 9

__all__ = [
    "PathConfig",
    "get_paths",
    "detect_environment",
    "print_status",
    # Model constants
    "HLCA_LATENT_DIM",
    "LUCA_LATENT_DIM",
    "FUSED_LATENT_DIM",
    "N_NICHE_TOKENS",
]
