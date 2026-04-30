"""Pooling-based baselines for StageBridge.

These baselines aggregate cell populations without spatial structure,
testing whether niche context provides signal beyond simple pooling.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from stagebridge.contracts import LATENT_DIM, N_STAGES


@dataclass(slots=True, frozen=True)
class PoolingBaselineConfig:
    """Configuration for pooling baselines."""

    input_dim: int = LATENT_DIM
    hidden_dim: int = 128
    output_dim: int = LATENT_DIM
    num_stages: int = N_STAGES
    time_dim: int = 32
    stage_dim: int = 32
    dropout: float = 0.1
    pool_mode: str = "mean"

    @property
    def num_edges(self) -> int:
        return self.num_stages * self.num_stages


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time embedding for flow matching."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0, device=t.device)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t.unsqueeze(-1) * emb.unsqueeze(0)
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class PoolingMLP(nn.Module):
    """Mean/Max pooling baseline - no structure at all.

    This is the simplest baseline: pool all cells in a sample and use MLP.
    Tests whether ANY structure (spatial or set) matters.

    Args:
        config: Model configuration
    """

    def __init__(self, config: PoolingBaselineConfig) -> None:
        super().__init__()
        self.config = config

        self.time_embedding = SinusoidalTimeEmbedding(config.time_dim)
        self.stage_embedding = nn.Embedding(config.num_edges, config.stage_dim)

        mlp_input = config.input_dim + config.hidden_dim + config.time_dim + config.stage_dim
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input, config.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.output_dim),
        )

        self.context_proj = nn.Linear(config.input_dim, config.hidden_dim)

    def encode_context(
        self,
        receiver: Tensor,
        neighbors: Tensor,
        neighbor_mask: Tensor | None = None,
        **kwargs: object,
    ) -> Tensor:
        """Pool receiver and neighbors into context.

        Args:
            receiver: [B, D] receiver embeddings
            neighbors: [B, K, D] neighbor embeddings
            neighbor_mask: [B, K] valid neighbor mask

        Returns:
            [B, H] context embedding
        """
        all_tokens = torch.cat([receiver.unsqueeze(1), neighbors], dim=1)

        if neighbor_mask is not None:
            full_mask = torch.cat(
                [torch.ones_like(neighbor_mask[:, :1]), neighbor_mask], dim=1
            )
            full_mask = full_mask.unsqueeze(-1)

            if self.config.pool_mode == "mean":
                pooled = (all_tokens * full_mask).sum(dim=1) / full_mask.sum(dim=1).clamp(min=1)
            else:
                all_tokens = all_tokens.masked_fill(~full_mask.bool(), float("-inf"))
                pooled = all_tokens.max(dim=1).values
        else:
            if self.config.pool_mode == "mean":
                pooled = all_tokens.mean(dim=1)
            else:
                pooled = all_tokens.max(dim=1).values

        return self.context_proj(pooled)

    def forward_vector_field(
        self,
        x_t: Tensor,
        t: Tensor,
        context: Tensor,
        stage_pair_id: Tensor,
        **kwargs: object,
    ) -> Tensor:
        """Predict drift velocity.

        Args:
            x_t: [B, D] current state
            t: [B] time
            context: [B, H] context from encode_context
            stage_pair_id: [B] stage transition indices

        Returns:
            [B, D] drift velocity
        """
        time_emb = self.time_embedding(t)
        stage_emb = self.stage_embedding(stage_pair_id)

        inp = torch.cat([x_t, context, time_emb, stage_emb], dim=-1)
        return self.mlp(inp)

    def forward(
        self,
        receiver: Tensor,
        neighbors: Tensor,
        distances: Tensor,
        x_t: Tensor,
        t: Tensor,
        stage_pair_id: Tensor,
        neighbor_mask: Tensor | None = None,
        **kwargs: object,
    ) -> Tensor:
        """Full forward pass.

        Args:
            receiver: [B, D] receiver embeddings
            neighbors: [B, K, D] neighbor embeddings
            distances: [B, K] distances (ignored)
            x_t: [B, D] current state
            t: [B] time
            stage_pair_id: [B] stage indices
            neighbor_mask: [B, K] valid neighbor mask

        Returns:
            [B, D] drift velocity prediction
        """
        context = self.encode_context(receiver, neighbors, neighbor_mask)
        return self.forward_vector_field(x_t, t, context, stage_pair_id)

    def integrate_euler(
        self,
        x0: Tensor,
        context: Tensor,
        stage_pair_id: Tensor,
        num_steps: int = 8,
        **kwargs: object,
    ) -> Tensor:
        """Euler integration from t=0 to t=1."""
        x = x0
        dt = 1.0 / float(num_steps)
        for k in range(num_steps):
            t = torch.full((x.shape[0],), (k + 0.5) * dt, device=x.device, dtype=x.dtype)
            v = self.forward_vector_field(x, t, context, stage_pair_id)
            x = x + dt * v
        return x


class MaxPoolMLP(PoolingMLP):
    """Max pooling variant."""

    def __init__(self, config: PoolingBaselineConfig | None = None) -> None:
        if config is None:
            config = PoolingBaselineConfig(pool_mode="max")
        else:
            config = PoolingBaselineConfig(
                input_dim=config.input_dim,
                hidden_dim=config.hidden_dim,
                output_dim=config.output_dim,
                num_stages=config.num_stages,
                time_dim=config.time_dim,
                stage_dim=config.stage_dim,
                dropout=config.dropout,
                pool_mode="max",
            )
        super().__init__(config)
