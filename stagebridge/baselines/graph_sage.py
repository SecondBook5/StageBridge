"""GraphSAGE baseline for spatial cell-cell interactions.

Tests whether explicit spatial graph structure helps vs. flat attention.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class GraphSAGELayer(nn.Module):
    """Single GraphSAGE layer with mean aggregation."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.self_linear = nn.Linear(in_dim, out_dim)
        self.neighbor_linear = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: Tensor, adj: Tensor, mask: Tensor | None = None) -> Tensor:
        """
        Args:
            x: Node features [batch, n_nodes, in_dim]
            adj: Adjacency matrix [batch, n_nodes, n_nodes] (binary or weighted)
            mask: Node mask [batch, n_nodes]

        Returns:
            Updated node features [batch, n_nodes, out_dim]
        """
        # Aggregate neighbors (mean)
        if mask is not None:
            # Mask invalid nodes in adjacency
            adj_masked = adj * mask.unsqueeze(1).float()
            degree = adj_masked.sum(dim=-1, keepdim=True).clamp_min(1.0)
        else:
            degree = adj.sum(dim=-1, keepdim=True).clamp_min(1.0)
            adj_masked = adj

        neighbor_feats = torch.bmm(adj_masked, x) / degree  # [batch, n_nodes, in_dim]

        # Combine self + neighbor
        self_out = self.self_linear(x)
        neighbor_out = self.neighbor_linear(neighbor_feats)
        h = F.gelu(self_out + neighbor_out)
        h = self.norm(h)
        h = self.dropout(h)

        return h


class GraphSAGEBaseline(nn.Module):
    """GraphSAGE baseline for progression inference from spatial cell graphs.

    Constructs spatial graph from coordinates, applies GraphSAGE layers,
    pools to lesion/world level, predicts stage/transition.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_stage_classes: int = 5,
        dropout: float = 0.1,
        spatial_radius: float = 500.0,  # um
    ):
        super().__init__()
        self.spatial_radius = spatial_radius

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # GraphSAGE layers
        self.layers = nn.ModuleList([
            GraphSAGELayer(hidden_dim, hidden_dim, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Readout
        self.pool = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # mean + max
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        # Task heads
        self.stage_head = nn.Linear(hidden_dim, num_stage_classes)
        self.displacement_head = nn.Linear(hidden_dim, input_dim)

    def forward(
        self,
        x: Tensor,
        coords: Tensor,
        mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """
        Args:
            x: Cell features [batch, n_cells, input_dim]
            coords: Spatial coordinates [batch, n_cells, 2] (x, y in um)
            mask: Cell mask [batch, n_cells]

        Returns:
            Dictionary with stage_logits, displacement, embedding
        """
        # Build spatial adjacency matrix
        adj = self._build_spatial_adjacency(coords, mask)  # [batch, n_cells, n_cells]

        # Project input
        h = self.input_proj(x)

        # Apply GraphSAGE layers
        for layer in self.layers:
            h = layer(h, adj, mask)

        # Global pooling (mean + max)
        if mask is not None:
            mask_float = mask.unsqueeze(-1).float()
            h_masked = h * mask_float
            denom = mask.sum(dim=1, keepdim=True).clamp_min(1).unsqueeze(-1)
            h_mean = h_masked.sum(dim=1) / denom.squeeze(-1)

            neg_inf = torch.full_like(h, -1e9)
            h_masked_max = torch.where(mask.unsqueeze(-1), h, neg_inf)
            h_max = h_masked_max.max(dim=1).values
        else:
            h_mean = h.mean(dim=1)
            h_max = h.max(dim=1).values

        lesion_emb = self.pool(torch.cat([h_mean, h_max], dim=-1))

        # Task predictions
        stage_logits = self.stage_head(lesion_emb)
        displacement = self.displacement_head(lesion_emb)

        return {
            "lesion_embedding": lesion_emb,
            "stage_logits": stage_logits,
            "displacement": displacement,
            "node_embeddings": h,
        }

    def _build_spatial_adjacency(self, coords: Tensor, mask: Tensor | None = None) -> Tensor:
        """Build binary adjacency matrix based on spatial distance threshold.

        Args:
            coords: [batch, n_nodes, 2]
            mask: [batch, n_nodes]

        Returns:
            Adjacency matrix [batch, n_nodes, n_nodes]
        """
        # Compute pairwise distances
        # coords: [B, N, 2]
        diff = coords.unsqueeze(2) - coords.unsqueeze(1)  # [B, N, N, 2]
        dist = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-8)  # [B, N, N]

        # Binary adjacency (within radius)
        adj = (dist <= self.spatial_radius).float()

        # Remove self-loops
        n_nodes = coords.shape[1]
        eye = torch.eye(n_nodes, device=coords.device, dtype=torch.bool)
        adj = adj.masked_fill(eye.unsqueeze(0), 0.0)

        # Mask invalid nodes
        if mask is not None:
            valid = mask.unsqueeze(1) & mask.unsqueeze(2)  # [B, N, N]
            adj = adj * valid.float()

        return adj
