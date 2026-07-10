"""Growth head — the growth / mass / tissue-rate decomposition.

Produces the context-residual decomposition of the receiver's growth effect
``g``:

    self_growth            g_self(receiver[, edge])
    regulatory_growth      a_e^T r          (edge-conditioned regulatory -> growth)
    residual_growth        rho_theta(receiver, context, r[, edge])  (neural residual)
    context_delta_growth   regulatory_growth + residual_growth
    full_growth            self_growth + context_delta_growth

Growth is a continuous **signed** effect in this milestone: no sigmoid, no
softplus, no positivity constraint. The residual arithmetic is delegated to
``compose_context_residual`` so it is identical to the drift head.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .context_residual import compose_context_residual
from .edge_conditioning import EdgeLinear, EdgeLinearConfig

__all__ = ["GrowthHeadConfig", "GrowthOutput", "GrowthHead"]


@dataclass(frozen=True)
class GrowthHeadConfig:
    """Configuration for :class:`GrowthHead`."""

    receiver_dim: int
    context_dim: int
    regulatory_dim: int
    growth_dim: int = 1
    hidden_dim: int = 64
    num_transition_edges: int | None = None
    edge_embedding_dim: int | None = None
    dropout: float = 0.0

    def __post_init__(self) -> None:
        for name in ("receiver_dim", "context_dim", "regulatory_dim", "growth_dim", "hidden_dim"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0")
        if not (0.0 <= self.dropout <= 1.0):
            raise ValueError("dropout must be in [0, 1]")
        if self.num_transition_edges is not None and self.num_transition_edges <= 0:
            raise ValueError("num_transition_edges must be > 0 when provided")
        if self.edge_embedding_dim is not None and self.edge_embedding_dim <= 0:
            raise ValueError("edge_embedding_dim must be > 0 when provided")

    @property
    def resolved_edge_embedding_dim(self) -> int | None:
        if self.num_transition_edges is None:
            return None
        if self.edge_embedding_dim is not None:
            return self.edge_embedding_dim
        return min(self.hidden_dim, 16)


@dataclass(frozen=True)
class GrowthOutput:
    """The growth decomposition (all tensors [B, growth_dim])."""

    self_growth: torch.Tensor
    regulatory_growth: torch.Tensor
    residual_growth: torch.Tensor
    context_delta_growth: torch.Tensor
    full_growth: torch.Tensor


class GrowthHead(nn.Module):
    """Context-residual growth head (signed growth/mass/tissue-rate effect)."""

    def __init__(self, config: GrowthHeadConfig) -> None:
        super().__init__()
        self.config = config
        edge_dim = config.resolved_edge_embedding_dim

        if config.num_transition_edges is not None:
            assert edge_dim is not None
            self.edge_embedding = nn.Embedding(config.num_transition_edges, edge_dim)
            extra = edge_dim
        else:
            self.edge_embedding = None
            extra = 0

        self.self_mlp = nn.Sequential(
            nn.Linear(config.receiver_dim + extra, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.growth_dim),
        )

        # a_e^T r
        self.regulatory_map = EdgeLinear(
            EdgeLinearConfig(
                input_dim=config.regulatory_dim,
                output_dim=config.growth_dim,
                num_transition_edges=config.num_transition_edges,
                bias=False,
            )
        )

        self.residual_mlp = nn.Sequential(
            nn.Linear(
                config.receiver_dim + config.context_dim + config.regulatory_dim + extra,
                config.hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.growth_dim),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for seq in (self.self_mlp, self.residual_mlp):
            for module in seq:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        if self.edge_embedding is not None:
            nn.init.normal_(self.edge_embedding.weight, std=0.02)

    def _edge_embed(
        self, b: int, transition_edge_index: torch.Tensor | None
    ) -> torch.Tensor | None:
        if self.edge_embedding is None:
            return None
        if transition_edge_index is None:
            raise ValueError(
                "transition_edge_index is required when num_transition_edges is set"
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
        assert self.config.num_transition_edges is not None
        if emin < 0 or emax >= self.config.num_transition_edges:
            raise ValueError(
                "transition_edge_index out of range "
                f"[0, {self.config.num_transition_edges}); got [{emin}, {emax}]"
            )
        return self.edge_embedding(transition_edge_index.long())

    def forward(
        self,
        *,
        receiver_features: torch.Tensor,
        context: torch.Tensor,
        regulatory_state: torch.Tensor,
        transition_edge_index: torch.Tensor | None = None,
    ) -> GrowthOutput:
        b = receiver_features.shape[0]
        edge_emb = self._edge_embed(b, transition_edge_index)

        self_in = receiver_features
        if edge_emb is not None:
            self_in = torch.cat([receiver_features, edge_emb], dim=-1)
        self_growth = self.self_mlp(self_in)

        regulatory_growth = self.regulatory_map(regulatory_state, transition_edge_index)

        residual_in = [receiver_features, context, regulatory_state]
        if edge_emb is not None:
            residual_in.append(edge_emb)
        residual_growth = self.residual_mlp(torch.cat(residual_in, dim=-1))

        composed = compose_context_residual(
            self_component=self_growth,
            regulatory_component=regulatory_growth,
            neural_residual=residual_growth,
        )
        return GrowthOutput(
            self_growth=composed.self_component,
            regulatory_growth=composed.regulatory_component,
            residual_growth=composed.neural_residual,
            context_delta_growth=composed.context_delta,
            full_growth=composed.full_component,
        )
