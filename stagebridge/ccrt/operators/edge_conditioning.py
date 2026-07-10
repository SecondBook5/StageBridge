"""Transition-edge-conditioned linear maps.

Provides ``EdgeLinear``, a linear map whose parameters may be shared across all
transition edges or specialized per edge. This realizes edge-conditioned terms
such as the regulatory-to-drift map ``B_e r`` and the regulatory-to-growth map
``a_e^T r``: when ``num_transition_edges`` is set, each edge ``e`` gets its own
weight/bias; otherwise a single shared map is used.

Generic and system-agnostic — no biological vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

__all__ = ["EdgeLinearConfig", "EdgeLinear"]


@dataclass(frozen=True)
class EdgeLinearConfig:
    """Configuration for :class:`EdgeLinear`."""

    input_dim: int
    output_dim: int
    num_transition_edges: int | None = None
    bias: bool = True

    def __post_init__(self) -> None:
        if self.input_dim <= 0:
            raise ValueError("input_dim must be > 0")
        if self.output_dim <= 0:
            raise ValueError("output_dim must be > 0")
        if self.num_transition_edges is not None and self.num_transition_edges <= 0:
            raise ValueError("num_transition_edges must be > 0 when provided")


class EdgeLinear(nn.Module):
    """A linear map that is either shared or transition-edge-specific.

    * ``num_transition_edges is None`` -> one shared ``nn.Linear``.
    * otherwise -> per-edge weight ``[E, input_dim, output_dim]`` and (optional)
      bias ``[E, output_dim]``, selected by ``transition_edge_index``.
    """

    def __init__(self, config: EdgeLinearConfig) -> None:
        super().__init__()
        self.config = config
        if config.num_transition_edges is None:
            self.shared = nn.Linear(
                config.input_dim, config.output_dim, bias=config.bias
            )
            self.weight = None
            self.bias = None
        else:
            self.shared = None
            self.weight = nn.Parameter(
                torch.empty(
                    config.num_transition_edges, config.input_dim, config.output_dim
                )
            )
            if config.bias:
                self.bias = nn.Parameter(
                    torch.zeros(config.num_transition_edges, config.output_dim)
                )
            else:
                self.bias = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.shared is not None:
            nn.init.xavier_uniform_(self.shared.weight)
            if self.shared.bias is not None:
                nn.init.zeros_(self.shared.bias)
        else:
            # Initialize each edge slice independently so distinct edges start
            # from distinct maps (a shared init would make edges indistinguishable).
            assert self.weight is not None
            for e in range(self.weight.shape[0]):
                nn.init.xavier_uniform_(self.weight[e])
            if self.bias is not None:
                nn.init.zeros_(self.bias)

    def forward(
        self,
        x: torch.Tensor,
        transition_edge_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(f"x must be [B, input_dim], got {tuple(x.shape)}")
        b = x.shape[0]
        if x.shape[1] != self.config.input_dim:
            raise ValueError(
                f"x last dim {x.shape[1]} != input_dim {self.config.input_dim}"
            )

        if self.shared is not None:
            return self.shared(x)

        # Edge-specific path.
        if transition_edge_index is None:
            raise ValueError(
                "transition_edge_index is required for an edge-specific EdgeLinear"
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
        num_edges = self.config.num_transition_edges
        assert num_edges is not None
        if emin < 0 or emax >= num_edges:
            raise ValueError(
                f"transition_edge_index out of range [0, {num_edges}); "
                f"got [{emin}, {emax}]"
            )

        edge = transition_edge_index.long()
        assert self.weight is not None
        w = self.weight[edge]  # [B, input_dim, output_dim]
        y = torch.einsum("bi,bio->bo", x, w)
        if self.bias is not None:
            y = y + self.bias[edge]
        return y
