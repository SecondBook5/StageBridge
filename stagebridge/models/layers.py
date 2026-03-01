"""Neural network layer building blocks for StageBridge."""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class FeedForwardBlock(nn.Module):
    """Transformer-style two-layer feed-forward network."""

    def __init__(self, dim: int, hidden_dim: int | None = None, dropout: float = 0.1) -> None:
        super().__init__()
        inner = hidden_dim or dim * 4
        self.net = nn.Sequential(
            nn.Linear(dim, inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class SAB(nn.Module):
    """Self-Attention Block (permutation equivariant)."""

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ln1 = nn.LayerNorm(dim)
        self.ff = FeedForwardBlock(dim=dim, dropout=dropout)
        self.ln2 = nn.LayerNorm(dim)

    @staticmethod
    def _key_padding_mask(mask: Tensor | None) -> Tensor | None:
        if mask is None:
            return None
        # MultiheadAttention expects True for padding positions.
        return ~mask.bool()

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        key_padding_mask = self._key_padding_mask(mask)
        attn_out, _ = self.mha(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = self.ln1(x + attn_out)
        x = self.ln2(x + self.ff(x))
        return x


class ISAB(nn.Module):
    """Induced Set Attention Block for efficient set processing."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_inducing_points: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.inducing_points = nn.Parameter(torch.randn(1, num_inducing_points, dim) * 0.02)

        self.mha_1 = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.mha_2 = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.ln_h1 = nn.LayerNorm(dim)
        self.ln_h2 = nn.LayerNorm(dim)
        self.ln_x1 = nn.LayerNorm(dim)
        self.ln_x2 = nn.LayerNorm(dim)
        self.ff_h = FeedForwardBlock(dim=dim, dropout=dropout)
        self.ff_x = FeedForwardBlock(dim=dim, dropout=dropout)

    @staticmethod
    def _key_padding_mask(mask: Tensor | None) -> Tensor | None:
        if mask is None:
            return None
        return ~mask.bool()

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        batch_size = x.shape[0]
        inducing = self.inducing_points.expand(batch_size, -1, -1)

        key_padding_mask = self._key_padding_mask(mask)
        h, _ = self.mha_1(
            query=inducing,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        h = self.ln_h1(inducing + h)
        h = self.ln_h2(h + self.ff_h(h))

        x_attn, _ = self.mha_2(query=x, key=h, value=h, need_weights=False)
        x = self.ln_x1(x + x_attn)
        x = self.ln_x2(x + self.ff_x(x))
        return x


class PMA(nn.Module):
    """Pooling by Multihead Attention (permutation invariant)."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_seed_vectors: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.seed_vectors = nn.Parameter(torch.randn(1, num_seed_vectors, dim) * 0.02)
        self.mha = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ln1 = nn.LayerNorm(dim)
        self.ff = FeedForwardBlock(dim=dim, dropout=dropout)
        self.ln2 = nn.LayerNorm(dim)

    @staticmethod
    def _key_padding_mask(mask: Tensor | None) -> Tensor | None:
        if mask is None:
            return None
        return ~mask.bool()

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        batch_size = x.shape[0]
        seeds = self.seed_vectors.expand(batch_size, -1, -1)
        key_padding_mask = self._key_padding_mask(mask)
        pooled, _ = self.mha(
            query=seeds,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        pooled = self.ln1(seeds + pooled)
        pooled = self.ln2(pooled + self.ff(pooled))
        return pooled


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal embedding of continuous time scalar t in [0, 1]."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        if t.ndim == 0:
            t = t[None]
        if t.ndim == 2 and t.shape[1] == 1:
            t = t[:, 0]

        half = self.dim // 2
        if half == 0:
            return t[:, None]

        device = t.device
        dtype = t.dtype
        freq = torch.exp(
            torch.arange(half, device=device, dtype=dtype)
            * (-math.log(10_000.0) / max(half - 1, 1))
        )
        phase = t[:, None] * freq[None, :]
        emb = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)

        if emb.shape[1] < self.dim:
            pad = torch.zeros((emb.shape[0], self.dim - emb.shape[1]), device=device, dtype=dtype)
            emb = torch.cat([emb, pad], dim=-1)
        return emb


class FiLMConditioner(nn.Module):
    """Feature-wise linear modulation conditioned on context."""

    def __init__(self, feature_dim: int, condition_dim: int) -> None:
        super().__init__()
        self.to_gamma_beta = nn.Linear(condition_dim, feature_dim * 2)

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        gamma_beta = self.to_gamma_beta(condition)
        gamma, beta = torch.chunk(gamma_beta, chunks=2, dim=-1)
        # Keep gamma around identity to avoid unstable scaling.
        gamma = 1.0 + 0.1 * torch.tanh(gamma)
        return gamma * x + beta
