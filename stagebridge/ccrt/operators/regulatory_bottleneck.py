"""Regulatory bottleneck.

Learns a compact regulatory state ``r`` from the receiver's own features and its
sender-context summary (and, optionally, a learned transition-edge embedding).
``r`` is the mediator through which context acts on behavior in the CCRT
decomposition (``B_e r`` for drift, ``a_e^T r`` for growth). It is signed — no
sigmoid/softmax — so it can represent both up- and down-regulation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

__all__ = [
    "RegulatoryBottleneckConfig",
    "RegulatoryBottleneckOutput",
    "RegulatoryBottleneck",
]


@dataclass(frozen=True)
class RegulatoryBottleneckConfig:
    """Configuration for :class:`RegulatoryBottleneck`."""

    receiver_dim: int
    context_dim: int
    regulatory_dim: int
    hidden_dim: int
    num_transition_edges: int | None = None
    edge_embedding_dim: int | None = None
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.receiver_dim <= 0:
            raise ValueError("receiver_dim must be > 0")
        if self.context_dim <= 0:
            raise ValueError("context_dim must be > 0")
        if self.regulatory_dim <= 0:
            raise ValueError("regulatory_dim must be > 0")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0")
        if not (0.0 <= self.dropout <= 1.0):
            raise ValueError("dropout must be in [0, 1]")
        if self.num_transition_edges is not None and self.num_transition_edges <= 0:
            raise ValueError("num_transition_edges must be > 0 when provided")
        if self.edge_embedding_dim is not None and self.edge_embedding_dim <= 0:
            raise ValueError("edge_embedding_dim must be > 0 when provided")

    @property
    def resolved_edge_embedding_dim(self) -> int | None:
        """The edge embedding width actually used (None if edges are unused)."""
        if self.num_transition_edges is None:
            return None
        if self.edge_embedding_dim is not None:
            return self.edge_embedding_dim
        return min(self.hidden_dim, 16)

    @property
    def bottleneck_input_dim(self) -> int:
        dim = self.receiver_dim + self.context_dim
        edge_dim = self.resolved_edge_embedding_dim
        if edge_dim is not None:
            dim += edge_dim
        return dim


@dataclass(frozen=True)
class RegulatoryBottleneckOutput:
    """Outputs of the regulatory bottleneck."""

    regulatory_state: torch.Tensor   # [B, regulatory_dim]
    bottleneck_input: torch.Tensor   # [B, bottleneck_input_dim]


class RegulatoryBottleneck(nn.Module):
    """MLP bottleneck mapping (receiver, context[, edge]) -> regulatory state."""

    def __init__(self, config: RegulatoryBottleneckConfig) -> None:
        super().__init__()
        self.config = config

        edge_dim = config.resolved_edge_embedding_dim
        if config.num_transition_edges is not None:
            assert edge_dim is not None
            self.edge_embedding = nn.Embedding(
                config.num_transition_edges, edge_dim
            )
        else:
            self.edge_embedding = None

        self.mlp = nn.Sequential(
            nn.Linear(config.bottleneck_input_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.regulatory_dim),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        if self.edge_embedding is not None:
            nn.init.normal_(self.edge_embedding.weight, std=0.02)

    def forward(
        self,
        *,
        receiver_features: torch.Tensor,
        context: torch.Tensor,
        transition_edge_index: torch.Tensor | None = None,
    ) -> RegulatoryBottleneckOutput:
        cfg = self.config
        if receiver_features.dim() != 2 or receiver_features.shape[1] != cfg.receiver_dim:
            raise ValueError(
                f"receiver_features must be [B, {cfg.receiver_dim}], got "
                f"{tuple(receiver_features.shape)}"
            )
        if context.dim() != 2 or context.shape[1] != cfg.context_dim:
            raise ValueError(
                f"context must be [B, {cfg.context_dim}], got {tuple(context.shape)}"
            )
        b = receiver_features.shape[0]
        if context.shape[0] != b:
            raise ValueError("receiver_features and context batch mismatch")

        parts = [receiver_features, context]
        if self.edge_embedding is not None:
            if transition_edge_index is None:
                raise ValueError(
                    "transition_edge_index is required when num_transition_edges "
                    "is set"
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
            assert cfg.num_transition_edges is not None
            if emin < 0 or emax >= cfg.num_transition_edges:
                raise ValueError(
                    "transition_edge_index out of range "
                    f"[0, {cfg.num_transition_edges}); got [{emin}, {emax}]"
                )
            parts.append(self.edge_embedding(transition_edge_index.long()))

        bottleneck_input = torch.cat(parts, dim=-1)
        regulatory_state = self.mlp(bottleneck_input)
        return RegulatoryBottleneckOutput(
            regulatory_state=regulatory_state,
            bottleneck_input=bottleneck_input,
        )
