"""State-dependent diffusion networks for stochastic StageBridge dynamics."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from stagebridge.context_model.set_encoder import SinusoidalTimeEmbedding


@dataclass(slots=True, frozen=True)
class DiffusionConfig:
    """Configuration for the diagonal diffusion head."""

    state_dependent: bool = True
    min_scale: float = 1e-3


class StateDependentDiffusionNetwork(nn.Module):
    """Predict diagonal diffusion scales conditioned on state, time, edge, and context."""

    def __init__(
        self,
        input_dim: int,
        context_dim: int,
        *,
        hidden_dim: int = 128,
        time_dim: int = 32,
        edge_dim: int = 16,
        num_edges: int = 4,
        dropout: float = 0.1,
        min_scale: float = 1e-3,
        state_dependent: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.context_dim = int(context_dim)
        self.state_dependent = bool(state_dependent)
        self.min_scale = float(min_scale)
        self.time_embedding = SinusoidalTimeEmbedding(int(time_dim))
        self.edge_embedding = nn.Embedding(int(num_edges), int(edge_dim))
        effective_input_dim = self.context_dim + int(time_dim) + int(edge_dim)
        if self.state_dependent:
            effective_input_dim += self.input_dim

        self.network = nn.Sequential(
            nn.Linear(effective_input_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), self.input_dim),
        )

    def forward(
        self,
        x_t: Tensor,
        t: Tensor,
        context: Tensor,
        edge_ids: Tensor,
    ) -> Tensor:
        if x_t.ndim != 2:
            raise ValueError(f"x_t must be 2D, got shape {tuple(x_t.shape)}.")

        if context.ndim == 1:
            context = context.unsqueeze(0)
        if context.shape[0] == 1 and x_t.shape[0] > 1:
            context = context.expand(x_t.shape[0], -1)
        if context.shape != (x_t.shape[0], self.context_dim):
            raise ValueError(
                "context must align to the batch dimension and configured context dim: "
                f"got {tuple(context.shape)}, expected ({x_t.shape[0]}, {self.context_dim})."
            )

        if t.ndim == 0:
            t = t.repeat(x_t.shape[0])
        if t.ndim != 1 or t.shape[0] != x_t.shape[0]:
            raise ValueError(f"t must have shape ({x_t.shape[0]},), got {tuple(t.shape)}.")

        if edge_ids.ndim == 0:
            edge_ids = edge_ids.repeat(x_t.shape[0])
        if edge_ids.ndim != 1 or edge_ids.shape[0] != x_t.shape[0]:
            raise ValueError(
                f"edge_ids must have shape ({x_t.shape[0]},), got {tuple(edge_ids.shape)}."
            )

        features = [self.time_embedding(t), context, self.edge_embedding(edge_ids.long())]
        if self.state_dependent:
            features.insert(0, x_t)
        raw = self.network(torch.cat(features, dim=-1))
        return torch.nn.functional.softplus(raw) + self.min_scale
