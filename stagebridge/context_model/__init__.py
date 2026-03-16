"""Context model exports for StageBridge.

This module provides:
- Receiver-centered niche encoder (doctrine-compliant, preferred)
- Legacy local niche encoders (for backward compatibility)
- Bag-level baselines (computational containers)
- Evolution branch for transition modeling
"""

# Receiver-centered niche encoder (PREFERRED - per doctrine)
from .receiver_niche_encoder import (
    ReceiverCenteredNicheEncoder,
    ReceiverNicheEncoderWithDualReference,
    ReceiverCenteredAttention,
    ReceiverNicheOutput,
    DistanceEncoding,
    SparsityType,
)

# Legacy local niche encoders (for backward compatibility)
from .local_niche_encoder import (
    LocalNicheMLPEncoder,
    LocalNicheTokenizer,
    LocalNicheTransformerEncoder,
)

# Bag-level aggregation (computational containers, not scientific center)
from .baselines_lesion import (
    DeepSetsLesionBaseline,
    LesionSetTransformerBaseline,
    PooledLesionBaseline,
)
from .heads import LesionMultitaskHeads, LesionTaskHeadOutput
from .lesion_set_transformer import EAMISTModel, EAMISTOutput, LesionSetTransformerBackbone

# Other components
from .evolution_branch import EvolutionBranch
from .prototype_bottleneck import PrototypeBottleneck, PrototypeBottleneckOutput

__all__ = [
    # Receiver-centered niche encoder (PREFERRED)
    "ReceiverCenteredNicheEncoder",
    "ReceiverNicheEncoderWithDualReference",
    "ReceiverCenteredAttention",
    "ReceiverNicheOutput",
    "DistanceEncoding",
    "SparsityType",
    # Legacy niche encoders
    "LocalNicheMLPEncoder",
    "LocalNicheTokenizer",
    "LocalNicheTransformerEncoder",
    # Bag-level baselines
    "DeepSetsLesionBaseline",
    "LesionSetTransformerBaseline",
    "PooledLesionBaseline",
    # Heads and outputs
    "LesionMultitaskHeads",
    "LesionTaskHeadOutput",
    "EAMISTModel",
    "EAMISTOutput",
    "LesionSetTransformerBackbone",
    # Other
    "EvolutionBranch",
    "PrototypeBottleneck",
    "PrototypeBottleneckOutput",
]
