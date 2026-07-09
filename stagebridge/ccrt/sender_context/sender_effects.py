"""Signed sender-context effects.

Converts per-sender value vectors and attention weights into **signed**
per-sender contributions in a generic effect space. Effects may be positive or
negative (no ReLU): a sender-context element can push the receiver's behavior in
either direction. This module is deliberately generic — it does not know about
drift, growth, or any disease-specific behavior; those layers consume its output
later.

The per-sender effect is ``attention_weight * project(value_vector)``, and the
aggregated effect is the sum over the sender axis.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

__all__ = [
    "SignedSenderEffectsConfig",
    "SignedSenderEffectsOutput",
    "SignedSenderEffects",
]


@dataclass(frozen=True)
class SignedSenderEffectsConfig:
    """Configuration for :class:`SignedSenderEffects`."""

    head_dim: int
    effect_dim: int
    num_heads: int

    def __post_init__(self) -> None:
        if self.head_dim <= 0:
            raise ValueError("head_dim must be > 0")
        if self.effect_dim <= 0:
            raise ValueError("effect_dim must be > 0")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be > 0")


@dataclass(frozen=True)
class SignedSenderEffectsOutput:
    """Per-sender and aggregated signed effects."""

    sender_effects: torch.Tensor       # [B, H, K, effect_dim]
    aggregated_effect: torch.Tensor    # [B, H, effect_dim]


class SignedSenderEffects(nn.Module):
    """Project attention-weighted sender values into a signed effect space."""

    def __init__(self, config: SignedSenderEffectsConfig) -> None:
        super().__init__()
        self.config = config
        # Signed projection: no activation, bias-free so a zero value maps to a
        # zero effect (padded senders zeroed by attention/mask stay zero).
        self.effect_proj = nn.Linear(config.head_dim, config.effect_dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.effect_proj.weight)

    def forward(
        self,
        *,
        sender_value_vectors: torch.Tensor,
        attention_weights: torch.Tensor,
        sender_mask: torch.Tensor | None = None,
    ) -> SignedSenderEffectsOutput:
        cfg = self.config
        if sender_value_vectors.dim() != 4:
            raise ValueError(
                "sender_value_vectors must be [B, H, K, D_head], got "
                f"{tuple(sender_value_vectors.shape)}"
            )
        b, h, k, dh = sender_value_vectors.shape
        if dh != cfg.head_dim:
            raise ValueError(
                f"sender_value_vectors head dim {dh} != config.head_dim "
                f"{cfg.head_dim}"
            )
        if h != cfg.num_heads:
            raise ValueError(
                f"sender_value_vectors heads {h} != config.num_heads "
                f"{cfg.num_heads}"
            )
        if tuple(attention_weights.shape) != (b, h, k):
            raise ValueError(
                f"attention_weights must be {(b, h, k)}, got "
                f"{tuple(attention_weights.shape)}"
            )

        # Signed projection of each value vector -> [B, H, K, effect_dim].
        projected = self.effect_proj(sender_value_vectors)

        # Scale by attention weight per sender.
        effects = projected * attention_weights.unsqueeze(-1)

        # Zero out masked (padded) sender positions.
        if sender_mask is not None:
            if sender_mask.shape[0] != b or sender_mask.shape[-1] != k:
                raise ValueError(
                    f"sender_mask shape {tuple(sender_mask.shape)} incompatible "
                    f"with [B={b}, K={k}]"
                )
            mask = sender_mask.bool().to(effects.dtype).view(b, 1, k, 1)
            effects = effects * mask

        aggregated = effects.sum(dim=2)  # [B, H, effect_dim]
        return SignedSenderEffectsOutput(
            sender_effects=effects, aggregated_effect=aggregated
        )
