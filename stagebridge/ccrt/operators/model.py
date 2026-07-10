"""The full Context-Residual Transport (CCRT) operator.

Composes the Milestone-3 typed sender-context attention and signed sender
effects with the Milestone-4 regulatory bottleneck, drift head, and growth head
into one operator that exposes the *entire* decomposition:

    typed sender-context attention
      -> signed sender effects
      -> regulatory bottleneck (r)
      -> self drift / regulatory drift (B_e r) / residual drift -> full drift
      -> self growth / regulatory growth (a_e^T r) / residual growth -> full growth

The operator returns every intermediate component — it never collapses to just
``full_drift`` / ``full_growth`` — because the decomposition (what a receiver
does intrinsically vs. what context makes it do) is the whole point of CCRT.

System-agnostic: it consumes grammar-typed ids and continuous distances only. It
implements no losses, no training, no semantic transport, and no disease
vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..sender_context.attention import (
    TypedSenderContextAttention,
    TypedSenderContextAttentionConfig,
)
from ..sender_context.sender_effects import (
    SignedSenderEffects,
    SignedSenderEffectsConfig,
)
from .drift import DriftHead, DriftHeadConfig
from .growth import GrowthHead, GrowthHeadConfig
from .regulatory_bottleneck import (
    RegulatoryBottleneck,
    RegulatoryBottleneckConfig,
)

__all__ = [
    "ContextResidualTransportConfig",
    "ContextResidualTransportOutput",
    "ContextResidualTransportOperator",
]


@dataclass(frozen=True)
class ContextResidualTransportConfig:
    """Configuration for :class:`ContextResidualTransportOperator`."""

    receiver_dim: int
    sender_dim: int
    hidden_dim: int
    num_heads: int
    num_sender_context_types: int
    empty_sender_context_type_id: int
    regulatory_dim: int
    drift_dim: int
    growth_dim: int = 1
    num_transition_edges: int | None = None
    sender_effect_dim: int | None = None
    distance_transform: str = "log1p"
    use_uncertainty: bool = True
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.receiver_dim <= 0:
            raise ValueError("receiver_dim must be > 0")
        if self.sender_dim <= 0:
            raise ValueError("sender_dim must be > 0")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be > 0")
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
        if self.num_sender_context_types <= 0:
            raise ValueError("num_sender_context_types must be > 0")
        if not (0 <= self.empty_sender_context_type_id < self.num_sender_context_types):
            raise ValueError(
                "empty_sender_context_type_id must be in "
                f"[0, {self.num_sender_context_types})"
            )
        if self.regulatory_dim <= 0:
            raise ValueError("regulatory_dim must be > 0")
        if self.drift_dim <= 0:
            raise ValueError("drift_dim must be > 0")
        if self.growth_dim <= 0:
            raise ValueError("growth_dim must be > 0")
        if self.num_transition_edges is not None and self.num_transition_edges <= 0:
            raise ValueError("num_transition_edges must be > 0 when provided")
        if self.sender_effect_dim is not None and self.sender_effect_dim <= 0:
            raise ValueError("sender_effect_dim must be > 0 when provided")
        if not (0.0 <= self.dropout <= 1.0):
            raise ValueError("dropout must be in [0, 1]")

    @property
    def head_dim(self) -> int:
        return self.hidden_dim // self.num_heads

    @property
    def resolved_sender_effect_dim(self) -> int:
        if self.sender_effect_dim is not None:
            return self.sender_effect_dim
        return self.head_dim


@dataclass(frozen=True)
class ContextResidualTransportOutput:
    """Full CCRT operator output — attention diagnostics + full decomposition."""

    # attention / context
    context: torch.Tensor
    per_head_context: torch.Tensor
    attention_weights: torch.Tensor
    attention_logits: torch.Tensor
    sender_value_vectors: torch.Tensor
    # signed sender effects
    sender_effects: torch.Tensor
    aggregated_sender_effect: torch.Tensor
    # regulatory state
    regulatory_state: torch.Tensor
    # drift decomposition
    self_drift: torch.Tensor
    regulatory_drift: torch.Tensor
    residual_drift: torch.Tensor
    context_delta_drift: torch.Tensor
    full_drift: torch.Tensor
    # growth decomposition
    self_growth: torch.Tensor
    regulatory_growth: torch.Tensor
    residual_growth: torch.Tensor
    context_delta_growth: torch.Tensor
    full_growth: torch.Tensor
    # sender-axis diagnostics (with empty token)
    sender_mask_with_empty: torch.Tensor
    distance_with_empty: torch.Tensor
    sender_context_type_ids_with_empty: torch.Tensor
    uncertainty_with_empty: torch.Tensor | None


class ContextResidualTransportOperator(nn.Module):
    """The full CCRT operator."""

    def __init__(self, config: ContextResidualTransportConfig) -> None:
        super().__init__()
        self.config = config

        self.attention = TypedSenderContextAttention(
            TypedSenderContextAttentionConfig(
                receiver_dim=config.receiver_dim,
                sender_dim=config.sender_dim,
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                num_sender_context_types=config.num_sender_context_types,
                empty_sender_context_type_id=config.empty_sender_context_type_id,
                num_transition_edges=config.num_transition_edges,
                distance_transform=config.distance_transform,
                use_uncertainty=config.use_uncertainty,
                dropout=config.dropout,
            )
        )

        self.sender_effects = SignedSenderEffects(
            SignedSenderEffectsConfig(
                head_dim=config.head_dim,
                effect_dim=config.resolved_sender_effect_dim,
                num_heads=config.num_heads,
            )
        )

        # The attention context vector has dimension hidden_dim; it is the
        # context input to the bottleneck and both behavior heads.
        context_dim = config.hidden_dim

        self.regulatory_bottleneck = RegulatoryBottleneck(
            RegulatoryBottleneckConfig(
                receiver_dim=config.receiver_dim,
                context_dim=context_dim,
                regulatory_dim=config.regulatory_dim,
                hidden_dim=config.hidden_dim,
                num_transition_edges=config.num_transition_edges,
                dropout=config.dropout,
            )
        )

        self.drift_head = DriftHead(
            DriftHeadConfig(
                receiver_dim=config.receiver_dim,
                context_dim=context_dim,
                regulatory_dim=config.regulatory_dim,
                drift_dim=config.drift_dim,
                hidden_dim=config.hidden_dim,
                num_transition_edges=config.num_transition_edges,
                dropout=config.dropout,
            )
        )

        self.growth_head = GrowthHead(
            GrowthHeadConfig(
                receiver_dim=config.receiver_dim,
                context_dim=context_dim,
                regulatory_dim=config.regulatory_dim,
                growth_dim=config.growth_dim,
                hidden_dim=config.hidden_dim,
                num_transition_edges=config.num_transition_edges,
                dropout=config.dropout,
            )
        )

    def forward(
        self,
        *,
        receiver_features: torch.Tensor,
        sender_features: torch.Tensor,
        sender_mask: torch.Tensor,
        distance_to_receiver: torch.Tensor,
        sender_context_type_ids: torch.Tensor,
        transition_edge_index: torch.Tensor | None = None,
        uncertainty: torch.Tensor | None = None,
    ) -> ContextResidualTransportOutput:
        # 1) typed sender-context attention
        attn = self.attention(
            receiver_features=receiver_features,
            sender_features=sender_features,
            sender_mask=sender_mask,
            distance_to_receiver=distance_to_receiver,
            sender_context_type_ids=sender_context_type_ids,
            transition_edge_index=transition_edge_index,
            uncertainty=uncertainty,
        )

        # 2) signed sender effects (over the K+1 axis, using the empty-aware mask)
        effects = self.sender_effects(
            sender_value_vectors=attn.sender_value_vectors,
            attention_weights=attn.attention_weights,
            sender_mask=attn.sender_mask_with_empty,
        )

        context = attn.context

        # 3) regulatory bottleneck
        reg = self.regulatory_bottleneck(
            receiver_features=receiver_features,
            context=context,
            transition_edge_index=transition_edge_index,
        )

        # 4) drift decomposition
        drift = self.drift_head(
            receiver_features=receiver_features,
            context=context,
            regulatory_state=reg.regulatory_state,
            transition_edge_index=transition_edge_index,
        )

        # 5) growth decomposition
        growth = self.growth_head(
            receiver_features=receiver_features,
            context=context,
            regulatory_state=reg.regulatory_state,
            transition_edge_index=transition_edge_index,
        )

        return ContextResidualTransportOutput(
            context=context,
            per_head_context=attn.per_head_context,
            attention_weights=attn.attention_weights,
            attention_logits=attn.attention_logits,
            sender_value_vectors=attn.sender_value_vectors,
            sender_effects=effects.sender_effects,
            aggregated_sender_effect=effects.aggregated_effect,
            regulatory_state=reg.regulatory_state,
            self_drift=drift.self_drift,
            regulatory_drift=drift.regulatory_drift,
            residual_drift=drift.residual_drift,
            context_delta_drift=drift.context_delta_drift,
            full_drift=drift.full_drift,
            self_growth=growth.self_growth,
            regulatory_growth=growth.regulatory_growth,
            residual_growth=growth.residual_growth,
            context_delta_growth=growth.context_delta_growth,
            full_growth=growth.full_growth,
            sender_mask_with_empty=attn.sender_mask_with_empty,
            distance_with_empty=attn.distance_with_empty,
            sender_context_type_ids_with_empty=attn.sender_context_type_ids_with_empty,
            uncertainty_with_empty=attn.uncertainty_with_empty,
        )
