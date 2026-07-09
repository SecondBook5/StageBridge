"""The empty sender-context token.

Every receiver must be able to attend to "no informative sender-context
element." CCRT models this by appending a single reserved **empty sender token**
to the sender axis (K -> K+1). This is an escape hatch per receiver — it is NOT
a global/world summary token and carries no aggregated context. A receiver with
no meaningful local context attends to its empty token and yields a near-zero
context residual rather than a fabricated one.

The empty token is always unmasked (mask value 1), sits at distance 0, and (when
uncertainty is tracked) carries uncertainty 0. Inputs are never mutated.
"""

from __future__ import annotations

import torch

__all__ = ["append_empty_sender_context"]


def append_empty_sender_context(
    *,
    sender_features: torch.Tensor,
    sender_mask: torch.Tensor,
    distance_to_receiver: torch.Tensor,
    sender_context_type_ids: torch.Tensor,
    empty_sender_feature: torch.Tensor,
    empty_sender_context_type_id: int,
    uncertainty: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Append an empty sender token to the sender axis.

    Args:
        sender_features: ``[B, K, D_S]``.
        sender_mask: ``[B, K]``.
        distance_to_receiver: ``[B, K]``.
        sender_context_type_ids: ``[B, K]`` (integer).
        empty_sender_feature: ``[D_S]`` — the (typically learnable) feature for
            the empty token, broadcast to ``[B, 1, D_S]``.
        empty_sender_context_type_id: reserved ontology id for the empty token.
        uncertainty: optional ``[B, K]``.

    Returns a dict with the same keys extended to ``K+1`` along the sender axis
    (``uncertainty`` present only if it was supplied). Inputs are not mutated.
    """
    if sender_features.dim() != 3:
        raise ValueError(
            f"sender_features must be rank 3 [B, K, D_S], got "
            f"{tuple(sender_features.shape)}"
        )
    b, k, d_s = sender_features.shape

    for name, tensor, expected in (
        ("sender_mask", sender_mask, (b, k)),
        ("distance_to_receiver", distance_to_receiver, (b, k)),
        ("sender_context_type_ids", sender_context_type_ids, (b, k)),
    ):
        if tuple(tensor.shape) != expected:
            raise ValueError(
                f"{name} must have shape {expected}, got {tuple(tensor.shape)}"
            )
    if uncertainty is not None and tuple(uncertainty.shape) != (b, k):
        raise ValueError(
            f"uncertainty must have shape {(b, k)}, got {tuple(uncertainty.shape)}"
        )
    if empty_sender_feature.dim() != 1 or empty_sender_feature.shape[0] != d_s:
        raise ValueError(
            f"empty_sender_feature must have shape [D_S={d_s}], got "
            f"{tuple(empty_sender_feature.shape)}"
        )

    # --- features: broadcast the empty feature to [B, 1, D_S] and concat ---
    empty_feat = empty_sender_feature.to(
        dtype=sender_features.dtype, device=sender_features.device
    ).view(1, 1, d_s).expand(b, 1, d_s)
    out_features = torch.cat([sender_features, empty_feat], dim=1)

    # --- mask: empty token is always valid (1) ---
    empty_mask = torch.ones(
        (b, 1), dtype=sender_mask.dtype, device=sender_mask.device
    )
    out_mask = torch.cat([sender_mask, empty_mask], dim=1)

    # --- distance: empty token at distance 0 ---
    empty_dist = torch.zeros(
        (b, 1),
        dtype=distance_to_receiver.dtype,
        device=distance_to_receiver.device,
    )
    out_dist = torch.cat([distance_to_receiver, empty_dist], dim=1)

    # --- type ids: reserved empty type id ---
    empty_type = torch.full(
        (b, 1),
        int(empty_sender_context_type_id),
        dtype=sender_context_type_ids.dtype,
        device=sender_context_type_ids.device,
    )
    out_types = torch.cat([sender_context_type_ids, empty_type], dim=1)

    result: dict[str, torch.Tensor] = {
        "sender_features": out_features,
        "sender_mask": out_mask,
        "distance_to_receiver": out_dist,
        "sender_context_type_ids": out_types,
    }

    if uncertainty is not None:
        empty_unc = torch.zeros(
            (b, 1), dtype=uncertainty.dtype, device=uncertainty.device
        )
        result["uncertainty"] = torch.cat([uncertainty, empty_unc], dim=1)

    return result
