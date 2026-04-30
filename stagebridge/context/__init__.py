"""Context encoding for StageBridge.

The context module encodes the 9-token niche structure into a context vector
that captures what the receiver cell "receives" from its neighborhood.
"""

from stagebridge.context.encoder import (
    ReceiverCenteredNicheEncoder,
    SelfAttentionNicheEncoder,
    ReceiverNicheOutput,
    DistanceEncoding,
    SparsityType,
)
from stagebridge.context.layers import (
    SAB,
    ISAB,
    PMA,
    RingPooler,
    FeedForwardBlock,
    SpatialRPE,
    SinusoidalTimeEmbedding,
)
from stagebridge.context.tokenizer import (
    NicheTokenizer,
    NicheTokenizerConfig,
)

__all__ = [
    "ReceiverCenteredNicheEncoder",
    "SelfAttentionNicheEncoder",
    "ReceiverNicheOutput",
    "DistanceEncoding",
    "SparsityType",
    "SAB",
    "ISAB",
    "PMA",
    "RingPooler",
    "FeedForwardBlock",
    "SpatialRPE",
    "SinusoidalTimeEmbedding",
    "NicheTokenizer",
    "NicheTokenizerConfig",
]
