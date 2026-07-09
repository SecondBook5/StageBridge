"""Sparsity losses for sender-context attention.

Two regularizers encourage the receiver to attend to *few* sender-context
elements and to keep sender contributions small unless they matter:

* ``attention_entropy_loss`` — entropy of the attention distribution over the
  sender axis (lower entropy = sparser attention).
* ``value_l1_sparsity_loss`` — mean absolute magnitude of per-sender effects
  over the real (unmasked) sender positions.

Both honor an optional ``sender_mask`` so padded positions never contribute, and
both return a scalar tensor (mean over batch/head).
"""

from __future__ import annotations

import torch

__all__ = ["attention_entropy_loss", "value_l1_sparsity_loss"]


def _expand_mask_to(attn_like: torch.Tensor, sender_mask: torch.Tensor) -> torch.Tensor:
    """Broadcast a ``[B, K]`` mask to the leading dims of ``attn_like``.

    ``attn_like`` is ``[B, H, K]``; the returned bool mask is ``[B, 1, K]`` so it
    broadcasts across heads.
    """
    if sender_mask.shape[0] != attn_like.shape[0]:
        raise ValueError(
            f"sender_mask batch {sender_mask.shape[0]} != attention batch "
            f"{attn_like.shape[0]}"
        )
    if sender_mask.shape[-1] != attn_like.shape[-1]:
        raise ValueError(
            f"sender_mask K {sender_mask.shape[-1]} != attention K "
            f"{attn_like.shape[-1]}"
        )
    return sender_mask.bool().unsqueeze(1)  # [B, 1, K]


def attention_entropy_loss(
    attention_weights: torch.Tensor,
    *,
    sender_mask: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Mean attention entropy over the sender axis.

    Args:
        attention_weights: ``[B, H, K]`` non-negative weights (typically summing
            to 1 over K per (B, H)).
        sender_mask: optional ``[B, K]``; masked positions do not contribute to
            the entropy sum.
        eps: numerical floor inside the log.

    Returns a scalar tensor (mean over batch and heads). Lower = sparser.
    """
    if attention_weights.dim() != 3:
        raise ValueError(
            f"attention_weights must be rank 3 [B, H, K], got "
            f"{tuple(attention_weights.shape)}"
        )
    weights = attention_weights
    if sender_mask is not None:
        mask = _expand_mask_to(weights, sender_mask)
        weights = weights * mask
    # H(p) = -sum_k p_k log p_k. Masked/zero weights contribute 0 because the
    # multiplication by (masked-out) zero weight zeroes the term.
    entropy = -(weights * torch.log(weights + eps)).sum(dim=-1)  # [B, H]
    return entropy.mean()


def value_l1_sparsity_loss(
    sender_effects: torch.Tensor,
    *,
    sender_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean absolute magnitude of sender effects over real sender positions.

    Args:
        sender_effects: ``[B, H, K, D]`` or ``[B, H, K]``.
        sender_mask: optional ``[B, K]``; masked positions are excluded from
            both the numerator and the denominator (mean over real positions).

    Returns a scalar tensor.
    """
    if sender_effects.dim() not in (3, 4):
        raise ValueError(
            "sender_effects must be rank 3 [B, H, K] or rank 4 [B, H, K, D], "
            f"got {tuple(sender_effects.shape)}"
        )
    abs_effects = sender_effects.abs()

    if sender_mask is None:
        return abs_effects.mean()

    b, h, k = abs_effects.shape[0], abs_effects.shape[1], abs_effects.shape[2]
    if sender_mask.shape[0] != b or sender_mask.shape[-1] != k:
        raise ValueError(
            f"sender_mask shape {tuple(sender_mask.shape)} incompatible with "
            f"sender_effects [B={b}, H={h}, K={k}, ...]"
        )
    mask = sender_mask.bool().to(abs_effects.dtype)  # [B, K]

    if abs_effects.dim() == 4:
        d = abs_effects.shape[3]
        mask_b = mask.view(b, 1, k, 1)  # broadcast over H and D
        total = (abs_effects * mask_b).sum()
        # denominator = number of real (B,H,K,D) entries
        denom = mask.sum() * h * d
    else:
        mask_b = mask.view(b, 1, k)  # broadcast over H
        total = (abs_effects * mask_b).sum()
        denom = mask.sum() * h

    denom = denom.clamp_min(1.0)
    return total / denom
