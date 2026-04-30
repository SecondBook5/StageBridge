"""Drift network components for OT-CFM transition modeling.

This module provides drift heads that predict the velocity field v(x_t, t)
conditioned on niche context, stage embeddings, and optional features.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from stagebridge.context.layers import FeedForwardBlock, SinusoidalTimeEmbedding


class CrossAttentionDrift(nn.Module):
    """Cross-attention drift head over context tokens and stage tokens.

    The drift network queries the niche context via cross-attention to
    produce a context-informed velocity, then gates it with a latent-only
    baseline to allow the model to learn when context matters.

    Args:
        input_dim: Latent state dimension
        context_dim: Context token dimension
        time_dim: Time embedding dimension
        stage_dim: Stage embedding dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
    """

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
        self.context_out_proj = nn.Linear(d_model, input_dim)

        self.latent_only = nn.Sequential(
            nn.Linear(input_dim + time_dim + stage_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, input_dim),
        )
        self.context_gate = nn.Sequential(
            nn.Linear(d_model * 2 + stage_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

        self.last_context_gate_mean: float = 0.0
        self.last_context_attention_entropy: float = 0.0

    def forward(
        self,
        x_t: Tensor,
        time_emb: Tensor,
        context_tokens: Tensor,
        stage_emb: Tensor,
    ) -> Tensor:
        """Compute drift velocity via gated cross-attention.

        Args:
            x_t: [B, D] current state
            time_emb: [B, T] time embedding
            context_tokens: [B, N, C] niche context tokens
            stage_emb: [B, S] stage embedding

        Returns:
            [B, D] drift velocity
        """
        q = self.query_proj(torch.cat([x_t, time_emb], dim=-1)).unsqueeze(1)
        kv_ctx = self.kv_proj(context_tokens)
        stage_tok = self.stage_proj(stage_emb).unsqueeze(1)
        kv = torch.cat([kv_ctx, stage_tok], dim=1)

        attn_out, attn_weights = self.mha(
            query=q, key=kv, value=kv, need_weights=True, average_attn_weights=False
        )
        h = self.ln1(q + attn_out)
        h = self.ln2(h + self.ff(h))
        context_only = self.context_out_proj(h.squeeze(1))

        latent_only = self.latent_only(torch.cat([x_t, time_emb, stage_emb], dim=-1))
        gate = self.context_gate(torch.cat([q.squeeze(1), h.squeeze(1), stage_emb], dim=-1))

        if attn_weights is not None:
            probs = attn_weights.mean(dim=1).squeeze(1).clamp_min(1e-8)
            entropy = -(probs * probs.log()).sum(dim=-1)
            self.last_context_attention_entropy = float(entropy.mean().detach().item())
        self.last_context_gate_mean = float(gate.mean().detach().item())

        return gate * context_only + (1.0 - gate) * latent_only


class FiLMConditioner(nn.Module):
    """Feature-wise linear modulation conditioned on context.

    Applies affine transformation gamma * x + beta where gamma and beta
    are predicted from the conditioning input.
    """

    def __init__(self, feature_dim: int, condition_dim: int) -> None:
        super().__init__()
        self.to_gamma_beta = nn.Linear(condition_dim, feature_dim * 2)

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        gamma_beta = self.to_gamma_beta(condition)
        gamma, beta = torch.chunk(gamma_beta, chunks=2, dim=-1)
        gamma = 1.0 + 0.5 * torch.tanh(gamma)
        return gamma * x + beta


class EdgeConditionedDriftMLP(nn.Module):
    """Simple MLP drift network conditioned on time, edge, and niche context.

    Provides a baseline architecture without cross-attention complexity.

    Args:
        input_dim: Latent state dimension
        context_dim: Context vector dimension
        hidden_dim: MLP hidden dimension
        time_dim: Time embedding dimension
        edge_dim: Edge embedding dimension
        num_edges: Number of stage transition edges
        dropout: Dropout rate
    """

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
            nn.Linear(
                self.input_dim + self.context_dim + int(time_dim) + int(edge_dim),
                int(hidden_dim),
            ),
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
        """Compute drift velocity.

        Args:
            x_t: [B, D] current state
            t: [B] time values in [0, 1]
            context: [B, C] context vector
            edge_ids: [B] edge indices

        Returns:
            [B, D] drift velocity
        """
        if x_t.ndim != 2:
            raise ValueError(f"x_t must be 2D, got shape {tuple(x_t.shape)}")

        if context.ndim == 1:
            context = context.unsqueeze(0)
        if context.shape[0] == 1 and x_t.shape[0] > 1:
            context = context.expand(x_t.shape[0], -1)
        if context.shape != (x_t.shape[0], self.context_dim):
            raise ValueError(
                f"context shape {tuple(context.shape)} incompatible with "
                f"expected ({x_t.shape[0]}, {self.context_dim})"
            )

        if t.ndim == 0:
            t = t.repeat(x_t.shape[0])
        if t.ndim != 1 or t.shape[0] != x_t.shape[0]:
            raise ValueError(f"t must have shape ({x_t.shape[0]},), got {tuple(t.shape)}")

        if edge_ids.ndim == 0:
            edge_ids = edge_ids.repeat(x_t.shape[0])
        if edge_ids.ndim != 1 or edge_ids.shape[0] != x_t.shape[0]:
            raise ValueError(
                f"edge_ids must have shape ({x_t.shape[0]},), got {tuple(edge_ids.shape)}"
            )

        time_emb = self.time_embedding(t)
        edge_emb = self.edge_embedding(edge_ids.long())
        return self.network(torch.cat([x_t, time_emb, context, edge_emb], dim=-1))


class BiologicalBaselineDrift(nn.Module):
    """Per-edge diagonal linear drift with time gating.

    Implements a simple mechanistic prior: v = gate(t) * (scale * x_t + bias).
    This captures average stage-transition dynamics without transformer context.
    In gated baseline mode, the transformer correction is blended via a learned gate.

    Args:
        input_dim: Latent state dimension
        num_edges: Number of stage transition edges
        time_dim: Time embedding dimension
    """

    def __init__(
        self,
        input_dim: int,
        num_edges: int,
        time_dim: int = 32,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.edge_scale = nn.Embedding(int(num_edges), int(input_dim))
        self.edge_bias = nn.Embedding(int(num_edges), int(input_dim))
        self.time_embedding = SinusoidalTimeEmbedding(int(time_dim))
        self.time_gate = nn.Sequential(
            nn.Linear(int(time_dim), int(input_dim)),
            nn.Sigmoid(),
        )
        nn.init.normal_(self.edge_scale.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.edge_bias.weight)

    def forward(self, x_t: Tensor, t: Tensor, edge_ids: Tensor) -> Tensor:
        """Compute baseline drift velocity.

        Args:
            x_t: [B, D] current state
            t: [B] time values
            edge_ids: [B] edge indices

        Returns:
            [B, D] drift velocity
        """
        if t.ndim == 0:
            t = t.repeat(x_t.shape[0])
        if edge_ids.ndim == 0:
            edge_ids = edge_ids.repeat(x_t.shape[0])

        scale = self.edge_scale(edge_ids.long())
        bias = self.edge_bias(edge_ids.long())
        gate = self.time_gate(self.time_embedding(t))
        return gate * (scale * x_t + bias)


class BaselineGate(nn.Module):
    """Per-edge learnable gate for baseline/learned drift mixing.

    gate = sigmoid(logit[edge_id])

    After training, learned gate values reveal which stage transitions
    rely most on transformer-encoded niche context vs baseline dynamics.

    Args:
        num_edges: Number of stage transition edges
        init_logit: Initial logit value (0.0 gives gate=0.5)
    """

    def __init__(self, num_edges: int, init_logit: float = 0.0) -> None:
        super().__init__()
        self.gate_logits = nn.Parameter(torch.full((int(num_edges),), float(init_logit)))

    def forward(self, edge_ids: Tensor) -> Tensor:
        """Return gate values for given edges.

        Args:
            edge_ids: [B] or scalar edge indices

        Returns:
            [B, 1] gate values in (0, 1)
        """
        if edge_ids.ndim == 0:
            edge_ids = edge_ids.unsqueeze(0)
        return torch.sigmoid(self.gate_logits[edge_ids.long()]).unsqueeze(-1)
