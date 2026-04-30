"""GraphSAGE baseline for StageBridge.

Tests spatial graph structure without receiver-centering.
Reference: Hamilton et al. "Inductive Representation Learning on Large Graphs" (NeurIPS 2017)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from stagebridge.contracts import LATENT_DIM, N_STAGES


@dataclass(slots=True, frozen=True)
class GraphSAGEConfig:
    """Configuration for GraphSAGE baseline."""

    input_dim: int = LATENT_DIM
    hidden_dim: int = 128
    output_dim: int = LATENT_DIM
    num_stages: int = N_STAGES
    num_layers: int = 2
    time_dim: int = 32
    stage_dim: int = 32
    dropout: float = 0.1
    aggregator: str = "mean"

    @property
    def num_edges(self) -> int:
        return self.num_stages * self.num_stages


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time embedding."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0, device=t.device)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t.unsqueeze(-1) * emb.unsqueeze(0)
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class SAGEConv(nn.Module):
    """Single GraphSAGE convolution layer.

    Aggregates neighbor features then concatenates with self features.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        aggregator: str = "mean",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.aggregator = aggregator

        self.self_transform = nn.Linear(in_dim, out_dim)
        self.neighbor_transform = nn.Linear(in_dim, out_dim)

        self.combine = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(
        self,
        node: Tensor,
        neighbors: Tensor,
        edge_weights: Tensor | None = None,
        neighbor_mask: Tensor | None = None,
    ) -> Tensor:
        """Apply SAGE convolution.

        Args:
            node: [B, D] node features
            neighbors: [B, K, D] neighbor features
            edge_weights: [B, K] edge weights (from distance)
            neighbor_mask: [B, K] valid neighbor mask

        Returns:
            [B, out_dim] updated node features
        """
        self_feat = self.self_transform(node)

        if neighbor_mask is not None:
            mask = neighbor_mask.unsqueeze(-1)
            neighbors = neighbors * mask

        if edge_weights is not None:
            weights = edge_weights.unsqueeze(-1)
            if neighbor_mask is not None:
                weights = weights * mask
            neighbors = neighbors * weights

        if self.aggregator == "mean":
            if neighbor_mask is not None:
                denom = neighbor_mask.float().sum(dim=1, keepdim=True).clamp(min=1)
                agg = neighbors.sum(dim=1) / denom
            else:
                agg = neighbors.mean(dim=1)
        elif self.aggregator == "max":
            if neighbor_mask is not None:
                neighbors = neighbors.masked_fill(~mask.bool(), float("-inf"))
            agg = neighbors.max(dim=1).values
        elif self.aggregator == "sum":
            agg = neighbors.sum(dim=1)
        else:
            raise ValueError(f"Unknown aggregator: {self.aggregator}")

        neigh_feat = self.neighbor_transform(agg)

        combined = torch.cat([self_feat, neigh_feat], dim=-1)
        return self.norm(self.combine(combined))


class GraphSAGE(nn.Module):
    """GraphSAGE baseline: spatial graph structure without receiver-centering.

    Architecture:
        - Stack of SAGE convolution layers
        - Distance-based edge weighting (inverse distance)
        - MLP drift head

    This tests whether spatial graph structure (without receiver-centering)
    captures progression signal better than flat attention.

    Key difference from StageBridge:
        - GraphSAGE: node aggregates neighbors symmetrically
        - StageBridge: receiver as query, neighbors as keys/values (asymmetric)

    Args:
        config: Model configuration
    """

    def __init__(self, config: GraphSAGEConfig) -> None:
        super().__init__()
        self.config = config

        self.input_proj = nn.Linear(config.input_dim, config.hidden_dim)

        self.convs = nn.ModuleList()
        for _ in range(config.num_layers):
            self.convs.append(
                SAGEConv(config.hidden_dim, config.hidden_dim, config.aggregator, config.dropout)
            )

        self.time_embedding = SinusoidalTimeEmbedding(config.time_dim)
        self.stage_embedding = nn.Embedding(config.num_edges, config.stage_dim)

        mlp_input = config.input_dim + config.hidden_dim + config.time_dim + config.stage_dim
        self.drift_mlp = nn.Sequential(
            nn.Linear(mlp_input, config.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.output_dim),
        )

    def _distance_to_weight(self, distances: Tensor) -> Tensor:
        """Convert distances to edge weights (inverse distance)."""
        return 1.0 / (distances + 1.0)

    def encode_context(
        self,
        receiver: Tensor,
        neighbors: Tensor,
        distances: Tensor | None = None,
        neighbor_mask: Tensor | None = None,
        **kwargs: object,
    ) -> Tensor:
        """Encode via GraphSAGE message passing.

        Args:
            receiver: [B, D] receiver (central node) embeddings
            neighbors: [B, K, D] neighbor embeddings
            distances: [B, K] distances for edge weighting
            neighbor_mask: [B, K] valid neighbor mask

        Returns:
            [B, H] context embedding
        """
        node = self.input_proj(receiver)
        neigh = self.input_proj(neighbors)

        edge_weights = None
        if distances is not None:
            edge_weights = self._distance_to_weight(distances)

        for conv in self.convs:
            node = conv(node, neigh, edge_weights, neighbor_mask)

        return node

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
        return self.drift_mlp(inp)

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
            distances: [B, K] distances (used for edge weighting)
            x_t: [B, D] current state
            t: [B] time
            stage_pair_id: [B] stage indices
            neighbor_mask: [B, K] valid neighbor mask

        Returns:
            [B, D] drift velocity prediction
        """
        context = self.encode_context(receiver, neighbors, distances, neighbor_mask)
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
