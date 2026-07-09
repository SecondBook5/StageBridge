"""CCRT sender_context — the AMICI-inspired local-influence layer.

Receiver-centered typed sender-context attention with continuous distance
modulation, uncertainty downweighting, an empty sender token, sparsity losses,
and signed sender effects. System-agnostic: it consumes grammar-typed ids and
continuous distances, never disease vocabulary, rings, or radial bins. Depends
only on torch and the standard library.
"""

from __future__ import annotations

from .attention import (
    TypedSenderContextAttention,
    TypedSenderContextAttentionConfig,
    TypedSenderContextAttentionOutput,
)
from .distance_kernels import (
    ContinuousDistanceTransform,
    DistanceTransformConfig,
    validate_distance_tensor,
)
from .empty_sender import append_empty_sender_context
from .sender_effects import (
    SignedSenderEffects,
    SignedSenderEffectsConfig,
    SignedSenderEffectsOutput,
)
from .sparsity import attention_entropy_loss, value_l1_sparsity_loss

__all__ = [
    # distance kernels
    "DistanceTransformConfig",
    "ContinuousDistanceTransform",
    "validate_distance_tensor",
    # empty sender token
    "append_empty_sender_context",
    # sparsity losses
    "attention_entropy_loss",
    "value_l1_sparsity_loss",
    # attention
    "TypedSenderContextAttentionConfig",
    "TypedSenderContextAttention",
    "TypedSenderContextAttentionOutput",
    # signed sender effects
    "SignedSenderEffectsConfig",
    "SignedSenderEffects",
    "SignedSenderEffectsOutput",
]
