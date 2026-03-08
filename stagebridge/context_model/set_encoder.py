"""Set Transformer components used by StageBridge context encoding."""
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
    """Self-attention block."""

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.mha = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.ff = FeedForwardBlock(dim=dim, dropout=dropout)
        self.ln2 = nn.LayerNorm(dim)

    @staticmethod
    def _key_padding_mask(mask: Tensor | None) -> Tensor | None:
        return None if mask is None else ~mask.bool()

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        attn_out, _ = self.mha(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self._key_padding_mask(mask),
            need_weights=False,
        )
        x = self.ln1(x + attn_out)
        return self.ln2(x + self.ff(x))


class SpatialRPE(nn.Module):
    """Additive spatial relative position encoding."""

    def __init__(self, num_heads: int, hidden: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.mlp = nn.Sequential(nn.Linear(1, hidden), nn.GELU(), nn.Linear(hidden, num_heads))

    def forward(self, coords: Tensor, n_src: int) -> Tensor:
        bsz, total, _ = coords.shape
        niche_coords = coords[:, n_src:, :]
        centroid = niche_coords.mean(dim=1, keepdim=True)
        dists = (coords - centroid).norm(dim=-1, keepdim=True)
        niche_mask = torch.zeros_like(dists)
        niche_mask[:, n_src:, :] = 1.0
        bias = self.mlp(dists * niche_mask)
        return bias.permute(0, 2, 1).unsqueeze(2).reshape(bsz * self.num_heads, 1, total)


class ISAB(nn.Module):
    """Induced set attention block."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_inducing_points: int = 16,
        dropout: float = 0.1,
        use_spatial_rpe: bool = False,
        rpe_hidden: int = 16,
    ) -> None:
        super().__init__()
        self.inducing_points = nn.Parameter(torch.randn(1, num_inducing_points, dim) * 0.02)
        self.mha_1 = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.mha_2 = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ln_h1 = nn.LayerNorm(dim)
        self.ln_h2 = nn.LayerNorm(dim)
        self.ln_x1 = nn.LayerNorm(dim)
        self.ln_x2 = nn.LayerNorm(dim)
        self.ff_h = FeedForwardBlock(dim=dim, dropout=dropout)
        self.ff_x = FeedForwardBlock(dim=dim, dropout=dropout)
        self.rpe = SpatialRPE(num_heads=num_heads, hidden=rpe_hidden) if use_spatial_rpe else None

    @staticmethod
    def _key_padding_mask(mask: Tensor | None) -> Tensor | None:
        return None if mask is None else ~mask.bool()

    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
        coords: Tensor | None = None,
        n_src: int = 0,
    ) -> Tensor:
        batch_size = x.shape[0]
        inducing = self.inducing_points.expand(batch_size, -1, -1)
        attn_mask = None
        if self.rpe is not None and coords is not None:
            rpe_bias = self.rpe(coords, n_src=n_src)
            attn_mask = rpe_bias.expand(-1, inducing.shape[1], -1).contiguous()

        h, _ = self.mha_1(
            query=inducing,
            key=x,
            value=x,
            key_padding_mask=self._key_padding_mask(mask),
            attn_mask=attn_mask,
            need_weights=False,
        )
        h = self.ln_h1(inducing + h)
        h = self.ln_h2(h + self.ff_h(h))
        x_attn, _ = self.mha_2(query=x, key=h, value=h, need_weights=False)
        x = self.ln_x1(x + x_attn)
        return self.ln_x2(x + self.ff_x(x))


class PMA(nn.Module):
    """Pooling by multihead attention."""

    def __init__(self, dim: int, num_heads: int = 8, num_seed_vectors: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        self.seed_vectors = nn.Parameter(torch.randn(1, num_seed_vectors, dim) * 0.02)
        self.mha = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.ff = FeedForwardBlock(dim=dim, dropout=dropout)
        self.ln2 = nn.LayerNorm(dim)

    @staticmethod
    def _key_padding_mask(mask: Tensor | None) -> Tensor | None:
        return None if mask is None else ~mask.bool()

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        batch_size = x.shape[0]
        seeds = self.seed_vectors.expand(batch_size, -1, -1)
        pooled, _ = self.mha(
            query=seeds,
            key=x,
            value=x,
            key_padding_mask=self._key_padding_mask(mask),
            need_weights=False,
        )
        pooled = self.ln1(seeds + pooled)
        return self.ln2(pooled + self.ff(pooled))


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal embedding of continuous time."""

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
            torch.arange(half, device=device, dtype=dtype) * (-math.log(10_000.0) / max(half - 1, 1))
        )
        phase = t[:, None] * freq[None, :]
        emb = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)
        if emb.shape[1] < self.dim:
            pad = torch.zeros((emb.shape[0], self.dim - emb.shape[1]), device=device, dtype=dtype)
            emb = torch.cat([emb, pad], dim=-1)
        return emb
