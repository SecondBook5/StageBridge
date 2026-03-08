"""Drift-network components for the transition model."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from stagebridge.context_model.set_encoder import FeedForwardBlock, SinusoidalTimeEmbedding


class CrossAttentionDrift(nn.Module):
    """Cross-attention drift head over context tokens and stage tokens."""

    def __init__(
        self,
        input_dim: int,
        context_dim: int,
        time_dim: int,
        stage_dim: int,
        num_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        d_model = context_dim
        self.query_proj = nn.Linear(input_dim + time_dim, d_model)
        self.kv_proj = nn.Linear(context_dim, d_model)
        self.stage_proj = nn.Linear(stage_dim, d_model)
        self.mha = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = FeedForwardBlock(dim=d_model, dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, input_dim)

    def forward(self, x_t: Tensor, time_emb: Tensor, context_tokens: Tensor, stage_emb: Tensor) -> Tensor:
        q = self.query_proj(torch.cat([x_t, time_emb], dim=-1)).unsqueeze(1)
        kv_ctx = self.kv_proj(context_tokens)
        stage_tok = self.stage_proj(stage_emb).unsqueeze(1)
        kv = torch.cat([kv_ctx, stage_tok], dim=1)
        attn_out, _ = self.mha(query=q, key=kv, value=kv, need_weights=False)
        h = self.ln1(q + attn_out)
        h = self.ln2(h + self.ff(h))
        return self.out_proj(h.squeeze(1))


class FiLMConditioner(nn.Module):
    """Feature-wise linear modulation conditioned on context."""

    def __init__(self, feature_dim: int, condition_dim: int) -> None:
        super().__init__()
        self.to_gamma_beta = nn.Linear(condition_dim, feature_dim * 2)

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        gamma_beta = self.to_gamma_beta(condition)
        gamma, beta = torch.chunk(gamma_beta, chunks=2, dim=-1)
        gamma = 1.0 + 0.1 * torch.tanh(gamma)
        return gamma * x + beta


class EdgeConditionedDriftMLP(nn.Module):
    """Diagonal drift network conditioned on time, edge identity, and niche context."""

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
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.context_dim = int(context_dim)
        self.time_embedding = SinusoidalTimeEmbedding(int(time_dim))
        self.edge_embedding = nn.Embedding(int(num_edges), int(edge_dim))
        self.network = nn.Sequential(
            nn.Linear(self.input_dim + self.context_dim + int(time_dim) + int(edge_dim), int(hidden_dim)),
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

        time_emb = self.time_embedding(t)
        edge_emb = self.edge_embedding(edge_ids.long())
        return self.network(torch.cat([x_t, time_emb, context, edge_emb], dim=-1))
