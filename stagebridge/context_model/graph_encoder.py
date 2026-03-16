"""Graph encoders for graph-of-sets niche context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch import Tensor, nn

from stagebridge.context_model.graph_of_sets import GraphOfSetsTransformer


@dataclass(slots=True, frozen=True)
class GraphEncodingSummary:
    """Graph-encoder output bundle for downstream transition conditioning."""

    pooled_context: Tensor
    node_contexts: Tensor
    num_nodes: int
    num_edges: int
    status: str = "complete"


class GraphOfSetsContextEncoder(nn.Module):
    """Encode typed spot features with spatial graph propagation."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        *,
        num_graph_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.node_projection = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.GELU(),
            nn.LayerNorm(int(hidden_dim)),
        )
        self.graph_transformer = GraphOfSetsTransformer(
            dim=int(hidden_dim),
            num_graph_layers=int(num_graph_layers),
            num_heads=int(num_heads),
            dropout=float(dropout),
        )
        self.output_norm = nn.LayerNorm(int(hidden_dim))

    def forward(self, node_features: Tensor, graph: Any) -> GraphEncodingSummary:
        if node_features.ndim != 2:
            raise ValueError(f"node_features must be 2D, got shape {tuple(node_features.shape)}.")

        projected = self.node_projection(node_features).unsqueeze(1)
        enriched = self.graph_transformer(
            node_summaries=projected,
            graph=graph,
        ).squeeze(1)
        pooled = self.output_norm(enriched.mean(dim=0))
        return GraphEncodingSummary(
            pooled_context=pooled,
            node_contexts=enriched,
            num_nodes=int(node_features.shape[0]),
            num_edges=int(graph.edge_index.shape[1]),
            status="complete",
        )
