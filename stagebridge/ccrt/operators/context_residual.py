"""Shared context-residual arithmetic.

Every CCRT behavior head (drift, growth, and any future behavior) decomposes the
same way:

    context_delta  = regulatory_component + neural_residual
    full_component = self_component + context_delta

Centralizing this here guarantees all heads use identical arithmetic — the
decomposition ``full = self + (regulatory + residual)`` is the load-bearing
claim of CCRT, so it must be defined exactly once. No activation, clipping, or
positivity constraint is applied: every component may be signed.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = ["ContextResidualComponents", "compose_context_residual"]


@dataclass(frozen=True)
class ContextResidualComponents:
    """The five tensors of a context-residual decomposition (all same shape)."""

    self_component: torch.Tensor
    regulatory_component: torch.Tensor
    neural_residual: torch.Tensor
    context_delta: torch.Tensor
    full_component: torch.Tensor


def compose_context_residual(
    *,
    self_component: torch.Tensor,
    regulatory_component: torch.Tensor,
    neural_residual: torch.Tensor,
) -> ContextResidualComponents:
    """Compose a context-residual decomposition from its three inputs.

    Returns a :class:`ContextResidualComponents` with

        context_delta  = regulatory_component + neural_residual
        full_component = self_component + context_delta

    All three inputs must share the same shape.
    """
    shape = tuple(self_component.shape)
    if tuple(regulatory_component.shape) != shape:
        raise ValueError(
            f"regulatory_component shape {tuple(regulatory_component.shape)} != "
            f"self_component shape {shape}"
        )
    if tuple(neural_residual.shape) != shape:
        raise ValueError(
            f"neural_residual shape {tuple(neural_residual.shape)} != "
            f"self_component shape {shape}"
        )

    context_delta = regulatory_component + neural_residual
    full_component = self_component + context_delta
    return ContextResidualComponents(
        self_component=self_component,
        regulatory_component=regulatory_component,
        neural_residual=neural_residual,
        context_delta=context_delta,
        full_component=full_component,
    )
