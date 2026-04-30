"""Reference geometry module for StageBridge.

Provides dual-reference mapping (HLCA + LuCA) and fusion for single-cell data.

The dual-reference approach:
1. HLCA (30d): Healthy Lung Cell Atlas - captures normal tissue structure
2. LuCA (10d): Lung Cancer Atlas - captures disease-specific states
3. Fused (40d): Concatenation of HLCA + LuCA embeddings

This allows cells to be positioned relative to both healthy and disease references.
"""

from stagebridge.reference.mapper import (
    ReferenceMapper,
    MappingResult,
    map_to_hlca,
    map_to_luca,
)
from stagebridge.reference.fusion import (
    FusionMethod,
    fuse_embeddings,
    concat_fusion,
    weighted_fusion,
    gated_fusion,
    GatedFusion,
    FiLMFusion,
    ConcatFusion,
    get_fusion_module,
)
from stagebridge.reference.gw_fusion import (
    GromovWassersteinFusion,
    GWFusionConfig,
    GWFusionLoss,
    entropic_gromov_wasserstein,
    sinkhorn,
)
from stagebridge.reference.confidence import (
    compute_confidence,
    ConfidenceResult,
)

__all__ = [
    # Mapper
    "ReferenceMapper",
    "MappingResult",
    "map_to_hlca",
    "map_to_luca",
    # Fusion
    "FusionMethod",
    "fuse_embeddings",
    "concat_fusion",
    "weighted_fusion",
    "gated_fusion",
    "GatedFusion",
    "FiLMFusion",
    "ConcatFusion",
    "get_fusion_module",
    # Gromov-Wasserstein fusion
    "GromovWassersteinFusion",
    "GWFusionConfig",
    "GWFusionLoss",
    "entropic_gromov_wasserstein",
    "sinkhorn",
    # Confidence
    "compute_confidence",
    "ConfidenceResult",
]
