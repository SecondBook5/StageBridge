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


class SpatialRPE(nn.Module):
    """Additive spatial relative position encoding for niche token attention.

    Computes a per-head scalar bias ``bias_j = MLP(||coord_j - centroid||₂)``
    for each token position j.  Source-cell positions (no coordinates) receive
    zero bias.  The bias is added to the ISAB mha_1 attention logits
    (inducing-points → token set) before softmax, giving the Set Transformer
    explicit spatial awareness of the niche microenvironment layout.
    """

    def __init__(self, num_heads: int, hidden: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_heads),
        )

    def forward(self, coords: Tensor, n_src: int) -> Tensor:
        """Return additive attn bias for mha_1.

        Parameters
        ----------
        coords : (B, N, 2)
            Full coordinate tensor — zeros for the first ``n_src`` source-cell
            positions, real (x, y) for the remaining niche token positions.
        n_src : int
            Number of source-cell positions (no spatial info).

        Returns
        -------
        Tensor : (B * num_heads, 1, N)
            Bias ready to be expanded to (B*H, num_inducing, N) and passed as
            ``attn_mask`` to ``nn.MultiheadAttention``.
        """
        B, N, _ = coords.shape
        # Centroid from niche positions only (exclude src cells)
        niche_coords = coords[:, n_src:, :]           # (B, m_niche, 2)
        centroid = niche_coords.mean(dim=1, keepdim=True)  # (B, 1, 2)
        dists = (coords - centroid).norm(dim=-1, keepdim=True)  # (B, N, 1)
        # Zero bias for source-cell positions (no spatial coordinate)
        niche_mask = torch.zeros_like(dists)
        niche_mask[:, n_src:, :] = 1.0
        dists = dists * niche_mask
        bias = self.mlp(dists)                         # (B, N, H)
        bias = bias.permute(0, 2, 1).unsqueeze(2)      # (B, H, 1, N)
        return bias.reshape(B * self.num_heads, 1, N)  # (B*H, 1, N)


class ISAB(nn.Module):
    """Induced Set Attention Block for efficient set processing."""

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
        self.rpe: SpatialRPE | None = SpatialRPE(num_heads=num_heads, hidden=rpe_hidden) if use_spatial_rpe else None

    @staticmethod
    def _key_padding_mask(mask: Tensor | None) -> Tensor | None:
        if mask is None:
            return None
        return ~mask.bool()

    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
        coords: Tensor | None = None,
        n_src: int = 0,
    ) -> Tensor:
        """Forward pass with optional Spatial RPE.

        Parameters
        ----------
        x : (B, N, dim)
        mask : (1, N) or (B, N) boolean validity mask, optional.
        coords : (B, N, 2) spatial coordinates, zeros for src-cell positions, optional.
            Required when ``self.rpe is not None``.
        n_src : int
            Number of source-cell positions at the start of the N axis.
            Used by SpatialRPE to zero-out src positions.
        """
        batch_size = x.shape[0]
        inducing = self.inducing_points.expand(batch_size, -1, -1)
        num_inducing = inducing.shape[1]

        key_padding_mask = self._key_padding_mask(mask)

        # Compute Spatial RPE bias for mha_1 if enabled and coords provided.
        attn_mask: Tensor | None = None
        if self.rpe is not None and coords is not None:
            # rpe_bias: (B*H, 1, N) → expand to (B*H, num_inducing, N)
            rpe_bias = self.rpe(coords, n_src=n_src)
            attn_mask = rpe_bias.expand(-1, num_inducing, -1).contiguous()

        h, _ = self.mha_1(
            query=inducing,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
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


class CrossAttentionDrift(nn.Module):
    """Cross-attention drift transformer for niche-conditioned flow matching.

    Instead of concatenating a single pooled context vector with x_t, the drift
    network treats x_t as a query that attends over a *sequence* of context
    tokens (the full PMA output).  This forces the transformer to be
    functionally central: every drift prediction requires attending to niche
    tokens.  Architecture:

        query  = Linear(x_t ++ time_emb)      → (B, 1, d)
        keys   = Linear(context_tokens)        → (B, k, d)
        values = same projection
        stage_token = Linear(stage_emb)        → (B, 1, d)
        KV     = cat([keys, stage_token], dim=1)  (B, k+1, d)
        attn   = MultiheadAttention(query, KV, KV)
        out    = Linear(attn_out + FF)         → (B, input_dim)
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
        d = context_dim
        self.query_proj = nn.Linear(input_dim + time_dim, d)
        self.kv_proj = nn.Linear(context_dim, d)
        self.stage_proj = nn.Linear(stage_dim, d)
        self.mha = nn.MultiheadAttention(
            embed_dim=d,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ln1 = nn.LayerNorm(d)
        self.ff = FeedForwardBlock(dim=d, dropout=dropout)
        self.ln2 = nn.LayerNorm(d)
        self.out_proj = nn.Linear(d, input_dim)

    def forward(
        self,
        x_t: Tensor,
        time_emb: Tensor,
        context_tokens: Tensor,
        stage_emb: Tensor,
    ) -> Tensor:
        """Return drift vector v ∈ ℝ^input_dim.

        Parameters
        ----------
        x_t:             (B, input_dim)
        time_emb:        (B, time_dim)
        context_tokens:  (B, k, context_dim) — k niche context tokens from PMA
        stage_emb:       (B, stage_dim)
        """
        # Query: x_t conditioned on time
        q = self.query_proj(torch.cat([x_t, time_emb], dim=-1)).unsqueeze(1)  # (B, 1, d)

        # Keys/values: niche context tokens + stage token
        kv_ctx = self.kv_proj(context_tokens)                    # (B, k, d)
        stage_tok = self.stage_proj(stage_emb).unsqueeze(1)      # (B, 1, d)
        kv = torch.cat([kv_ctx, stage_tok], dim=1)               # (B, k+1, d)

        # Cross-attention: each cell query attends over niche + stage tokens
        attn_out, _ = self.mha(query=q, key=kv, value=kv, need_weights=False)
        h = self.ln1(q + attn_out)     # (B, 1, d)
        h = self.ln2(h + self.ff(h))   # (B, 1, d)
        return self.out_proj(h.squeeze(1))  # (B, input_dim)


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
