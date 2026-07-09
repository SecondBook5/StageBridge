"""Typed sender-context attention — the AMICI-inspired local-influence layer.

Receiver-centered attention: the receiver is the *query*; the typed local
sender-context elements are *keys/values*. Attention logits combine

* phenotype similarity (scaled dot product),
* a per-(sender-type, head) additive **type bias**,
* a **continuous distance penalty** ``-lambda * phi(distance)`` with a strictly
  positive ``lambda`` (per head, per sender-type, and optionally per transition
  edge) — never a bin or ring,
* optional **uncertainty downweighting** ``-gamma * uncertainty`` with positive
  ``gamma`` per head,
* a **sender mask** that drives padded positions to ~zero weight.

A learnable **empty sender token** is appended before attention so every
receiver always has at least one valid key/value (K -> K+1). Distance and
uncertainty penalties are *subtracted*, so a nearer / more certain sender is
never penalized more than a farther / less certain one.

This layer produces a context vector and rich diagnostics; it does NOT model
drift, growth, transport, or any disease-specific behavior.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .distance_kernels import (
    ContinuousDistanceTransform,
    DistanceTransformConfig,
    validate_distance_tensor,
)
from .empty_sender import append_empty_sender_context

__all__ = [
    "TypedSenderContextAttentionConfig",
    "TypedSenderContextAttentionOutput",
    "TypedSenderContextAttention",
]


@dataclass(frozen=True)
class TypedSenderContextAttentionConfig:
    """Configuration for :class:`TypedSenderContextAttention`."""

    receiver_dim: int
    sender_dim: int
    hidden_dim: int
    num_heads: int
    num_sender_context_types: int
    empty_sender_context_type_id: int
    num_transition_edges: int | None = None
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
        if not (0.0 <= self.dropout <= 1.0):
            raise ValueError("dropout must be in [0, 1]")
        if self.num_transition_edges is not None and self.num_transition_edges <= 0:
            raise ValueError("num_transition_edges must be > 0 when provided")

    @property
    def head_dim(self) -> int:
        return self.hidden_dim // self.num_heads


@dataclass(frozen=True)
class TypedSenderContextAttentionOutput:
    """Outputs and diagnostics from typed sender-context attention.

    All sender-axis tensors are reported with the empty token included (K+1).
    """

    context: torch.Tensor
    per_head_context: torch.Tensor
    attention_weights: torch.Tensor
    attention_logits: torch.Tensor
    sender_value_vectors: torch.Tensor
    sender_mask_with_empty: torch.Tensor
    distance_with_empty: torch.Tensor
    sender_context_type_ids_with_empty: torch.Tensor
    uncertainty_with_empty: torch.Tensor | None


class TypedSenderContextAttention(nn.Module):
    """Receiver-as-query attention over typed sender-context keys/values."""

    def __init__(self, config: TypedSenderContextAttentionConfig) -> None:
        super().__init__()
        self.config = config
        h = config.num_heads
        dh = config.head_dim

        # Receiver query and sender key/value projections.
        self.query_proj = nn.Linear(config.receiver_dim, config.hidden_dim)
        self.key_proj = nn.Linear(config.sender_dim, config.hidden_dim)
        self.value_proj = nn.Linear(config.sender_dim, config.hidden_dim)
        self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim)

        # Learnable empty sender feature (in raw sender-feature space).
        self.empty_sender_feature = nn.Parameter(torch.zeros(config.sender_dim))

        # Per-(sender-type, head) additive bias on attention logits.
        self.type_bias = nn.Embedding(config.num_sender_context_types, h)

        # Positive distance coefficient lambda. Stored as a raw parameter passed
        # through softplus so lambda > 0 always (distance can only *penalize*).
        if config.num_transition_edges is None:
            # per (sender-type, head)
            self.distance_lambda_raw = nn.Parameter(
                torch.zeros(config.num_sender_context_types, h)
            )
        else:
            # per (edge, sender-type, head)
            self.distance_lambda_raw = nn.Parameter(
                torch.zeros(config.num_transition_edges, config.num_sender_context_types, h)
            )

        # Positive per-head uncertainty coefficient gamma (softplus).
        self.uncertainty_gamma_raw = nn.Parameter(torch.zeros(h))

        self.distance_transform = ContinuousDistanceTransform(
            DistanceTransformConfig(transform=config.distance_transform)
        )
        self.dropout = nn.Dropout(config.dropout)
        self._scale = 1.0 / math.sqrt(dh)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for lin in (self.query_proj, self.key_proj, self.value_proj, self.out_proj):
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
        nn.init.zeros_(self.type_bias.weight)
        nn.init.zeros_(self.empty_sender_feature)

    # -- validation --------------------------------------------------------

    def _validate_inputs(
        self,
        receiver_features: torch.Tensor,
        sender_features: torch.Tensor,
        sender_mask: torch.Tensor,
        distance_to_receiver: torch.Tensor,
        sender_context_type_ids: torch.Tensor,
        transition_edge_index: torch.Tensor | None,
        uncertainty: torch.Tensor | None,
    ) -> tuple[int, int]:
        for name, t in (
            ("receiver_features", receiver_features),
            ("sender_features", sender_features),
            ("sender_mask", sender_mask),
            ("distance_to_receiver", distance_to_receiver),
            ("sender_context_type_ids", sender_context_type_ids),
        ):
            if not isinstance(t, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")

        if receiver_features.dim() != 2:
            raise ValueError(
                f"receiver_features must be [B, D_R], got "
                f"{tuple(receiver_features.shape)}"
            )
        if sender_features.dim() != 3:
            raise ValueError(
                f"sender_features must be [B, K, D_S], got "
                f"{tuple(sender_features.shape)}"
            )
        b, k, d_s = sender_features.shape
        if receiver_features.shape[0] != b:
            raise ValueError("receiver_features and sender_features batch mismatch")
        if receiver_features.shape[1] != self.config.receiver_dim:
            raise ValueError(
                f"receiver_features last dim {receiver_features.shape[1]} != "
                f"config.receiver_dim {self.config.receiver_dim}"
            )
        if d_s != self.config.sender_dim:
            raise ValueError(
                f"sender_features last dim {d_s} != config.sender_dim "
                f"{self.config.sender_dim}"
            )
        for name, t in (
            ("sender_mask", sender_mask),
            ("distance_to_receiver", distance_to_receiver),
            ("sender_context_type_ids", sender_context_type_ids),
        ):
            if tuple(t.shape) != (b, k):
                raise ValueError(
                    f"{name} must have shape {(b, k)}, got {tuple(t.shape)}"
                )

        # distance: non-negative float [B, K]
        validate_distance_tensor(distance_to_receiver)

        # type ids: integer + in range
        if sender_context_type_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("sender_context_type_ids must be an integer tensor")
        if k > 0:
            max_id = int(sender_context_type_ids.max())
            min_id = int(sender_context_type_ids.min())
            if min_id < 0 or max_id >= self.config.num_sender_context_types:
                raise ValueError(
                    "sender_context_type_ids out of range "
                    f"[0, {self.config.num_sender_context_types}); got "
                    f"[{min_id}, {max_id}]"
                )

        # transition edge index
        if self.config.num_transition_edges is not None:
            if transition_edge_index is None:
                raise ValueError(
                    "transition_edge_index is required when "
                    "config.num_transition_edges is set"
                )
            if tuple(transition_edge_index.shape) != (b,):
                raise ValueError(
                    f"transition_edge_index must have shape {(b,)}, got "
                    f"{tuple(transition_edge_index.shape)}"
                )
            if transition_edge_index.dtype not in (torch.int32, torch.int64):
                raise ValueError("transition_edge_index must be an integer tensor")
            emin = int(transition_edge_index.min())
            emax = int(transition_edge_index.max())
            if emin < 0 or emax >= self.config.num_transition_edges:
                raise ValueError(
                    "transition_edge_index out of range "
                    f"[0, {self.config.num_transition_edges}); got [{emin}, {emax}]"
                )

        # uncertainty
        if uncertainty is not None:
            if not torch.is_floating_point(uncertainty):
                raise ValueError("uncertainty must be a floating tensor")
            if tuple(uncertainty.shape) != (b, k):
                raise ValueError(
                    f"uncertainty must have shape {(b, k)}, got "
                    f"{tuple(uncertainty.shape)}"
                )

        return b, k

    # -- forward -----------------------------------------------------------

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
    ) -> TypedSenderContextAttentionOutput:
        cfg = self.config
        b, k = self._validate_inputs(
            receiver_features,
            sender_features,
            sender_mask,
            distance_to_receiver,
            sender_context_type_ids,
            transition_edge_index,
            uncertainty,
        )
        h, dh = cfg.num_heads, cfg.head_dim

        # 1) Append the learnable empty sender token (K -> K+1).
        appended = append_empty_sender_context(
            sender_features=sender_features,
            sender_mask=sender_mask,
            distance_to_receiver=distance_to_receiver,
            sender_context_type_ids=sender_context_type_ids.long(),
            empty_sender_feature=self.empty_sender_feature,
            empty_sender_context_type_id=cfg.empty_sender_context_type_id,
            uncertainty=uncertainty,
        )
        s_feats = appended["sender_features"]              # [B, K+1, D_S]
        s_mask = appended["sender_mask"]                   # [B, K+1]
        s_dist = appended["distance_to_receiver"]          # [B, K+1]
        s_types = appended["sender_context_type_ids"]      # [B, K+1]
        s_unc = appended.get("uncertainty")                # [B, K+1] or None
        kp1 = k + 1

        # 2) Project queries / keys / values into heads.
        q = self.query_proj(receiver_features).view(b, h, dh)        # [B, H, Dh]
        k_proj = self.key_proj(s_feats).view(b, kp1, h, dh).permute(0, 2, 1, 3)
        v_proj = self.value_proj(s_feats).view(b, kp1, h, dh).permute(0, 2, 1, 3)
        # k_proj, v_proj: [B, H, K+1, Dh]

        # 3) Phenotype similarity: scaled dot product. [B, H, K+1]
        logits = torch.einsum("bhd,bhkd->bhk", q, k_proj) * self._scale

        # 4) Type bias per (sender-type, head): [B, K+1, H] -> [B, H, K+1]
        type_bias = self.type_bias(s_types).permute(0, 2, 1)
        logits = logits + type_bias

        # 5) Continuous distance penalty: -softplus(lambda) * phi(distance).
        phi = self.distance_transform(s_dist)                        # [B, K+1]
        if cfg.num_transition_edges is None:
            # lambda by (type, head): gather per token -> [B, K+1, H]
            lam = F.softplus(self.distance_lambda_raw)               # [T, H]
            lam_tok = lam[s_types]                                   # [B, K+1, H]
        else:
            lam = F.softplus(self.distance_lambda_raw)               # [E, T, H]
            edge = transition_edge_index.long()                      # [B]
            lam_edge = lam[edge]                                     # [B, T, H]
            # gather sender-type slice per token: [B, K+1, H]
            lam_tok = torch.gather(
                lam_edge,
                dim=1,
                index=s_types.unsqueeze(-1).expand(b, kp1, h),
            )
        lam_tok = lam_tok.permute(0, 2, 1)                           # [B, H, K+1]
        logits = logits - lam_tok * phi.unsqueeze(1)

        # 6) Uncertainty downweighting: -softplus(gamma_head) * uncertainty.
        if cfg.use_uncertainty and s_unc is not None:
            gamma = F.softplus(self.uncertainty_gamma_raw)           # [H]
            logits = logits - gamma.view(1, h, 1) * s_unc.unsqueeze(1)

        # 7) Mask padded senders with a large negative logit. The empty token is
        #    unmasked by construction, so at least one key is always valid.
        mask_bool = s_mask.bool().unsqueeze(1)                       # [B, 1, K+1]
        neg_inf = torch.finfo(logits.dtype).min
        logits = torch.where(mask_bool, logits, torch.full_like(logits, neg_inf))

        # 8) Softmax over the sender axis (K+1).
        attn = torch.softmax(logits, dim=-1)                         # [B, H, K+1]
        attn = self.dropout(attn)

        # 9) Per-head context = weighted sum of values.
        per_head = torch.einsum("bhk,bhkd->bhd", attn, v_proj)       # [B, H, Dh]

        # 10) Concatenate heads and project out.
        context = self.out_proj(per_head.reshape(b, h * dh))         # [B, hidden]

        return TypedSenderContextAttentionOutput(
            context=context,
            per_head_context=per_head,
            attention_weights=attn,
            attention_logits=logits,
            sender_value_vectors=v_proj,
            sender_mask_with_empty=s_mask,
            distance_with_empty=s_dist,
            sender_context_type_ids_with_empty=s_types,
            uncertainty_with_empty=s_unc,
        )
