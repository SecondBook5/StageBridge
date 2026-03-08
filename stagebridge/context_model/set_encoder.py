"""Set Transformer components used by StageBridge context encoding."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

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

    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
        *,
        return_attention: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        attn_out, attn_weights = self.mha(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self._key_padding_mask(mask),
            need_weights=bool(return_attention),
            average_attn_weights=False,
        )
        x = self.ln1(x + attn_out)
        out = self.ln2(x + self.ff(x))
        if return_attention:
            return out, attn_weights
        return out


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
        *,
        return_attention: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        batch_size = x.shape[0]
        inducing = self.inducing_points.expand(batch_size, -1, -1)
        attn_mask = None
        if self.rpe is not None and coords is not None:
            rpe_bias = self.rpe(coords, n_src=n_src)
            attn_mask = rpe_bias.expand(-1, inducing.shape[1], -1).contiguous()

        h, attn_1 = self.mha_1(
            query=inducing,
            key=x,
            value=x,
            key_padding_mask=self._key_padding_mask(mask),
            attn_mask=attn_mask,
            need_weights=bool(return_attention),
            average_attn_weights=False,
        )
        h = self.ln_h1(inducing + h)
        h = self.ln_h2(h + self.ff_h(h))
        x_attn, attn_2 = self.mha_2(
            query=x,
            key=h,
            value=h,
            need_weights=bool(return_attention),
            average_attn_weights=False,
        )
        x = self.ln_x1(x + x_attn)
        out = self.ln_x2(x + self.ff_x(x))
        if return_attention:
            return out, {
                "inducing_to_tokens": attn_1,
                "tokens_to_inducing": attn_2,
            }
        return out


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

    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
        *,
        return_attention: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        batch_size = x.shape[0]
        seeds = self.seed_vectors.expand(batch_size, -1, -1)
        pooled, attn_weights = self.mha(
            query=seeds,
            key=x,
            value=x,
            key_padding_mask=self._key_padding_mask(mask),
            need_weights=bool(return_attention),
            average_attn_weights=False,
        )
        pooled = self.ln1(seeds + pooled)
        out = self.ln2(pooled + self.ff(pooled))
        if return_attention:
            return out, attn_weights
        return out


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


@dataclass(slots=True, frozen=True)
class SetContextSummary:
    """Output summary from the typed set encoder."""

    pooled_context: Tensor
    token_embeddings: Tensor
    attention_maps: dict[str, Tensor] = field(default_factory=dict)
    token_type_ids: Tensor | None = None


class DeepSetsContextEncoder(nn.Module):
    """Permutation-invariant Deep Sets baseline for typed niche tokens."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.GELU(),
            nn.LayerNorm(int(hidden_dim)),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.LayerNorm(int(hidden_dim)),
        )
        self.rho = nn.Sequential(
            nn.Linear(int(hidden_dim) * 2, int(hidden_dim)),
            nn.GELU(),
            nn.LayerNorm(int(hidden_dim)),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.LayerNorm(int(hidden_dim)),
        )

    def forward(self, tokens: Tensor) -> SetContextSummary:
        if tokens.ndim != 2:
            raise ValueError(f"tokens must be 2D, got shape {tuple(tokens.shape)}.")
        embeddings = self.phi(tokens)
        pooled_mean = embeddings.mean(dim=0)
        pooled_max = embeddings.max(dim=0).values
        pooled = torch.cat([pooled_mean, pooled_max], dim=0)
        context = self.rho(pooled.unsqueeze(0))[0]
        return SetContextSummary(pooled_context=context, token_embeddings=embeddings)


class TypedSetContextEncoder(nn.Module):
    """Encode a donor-stage set of typed spatial tokens into one context vector."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_inducing_points: int = 16,
        dropout: float = 0.1,
        *,
        num_token_types: int = 4,
        use_token_type_embeddings: bool = True,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.num_token_types = int(num_token_types)
        self.token_type_embedding = (
            nn.Embedding(self.num_token_types, hidden_dim) if use_token_type_embeddings else None
        )
        self.isab = ISAB(
            dim=hidden_dim,
            num_heads=num_heads,
            num_inducing_points=num_inducing_points,
            dropout=dropout,
        )
        self.sab = SAB(dim=hidden_dim, num_heads=num_heads, dropout=dropout)
        self.pma = PMA(dim=hidden_dim, num_heads=num_heads, num_seed_vectors=1, dropout=dropout)
        self.context_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

    def _infer_token_type_ids(self, tokens: Tensor) -> Tensor:
        if tokens.shape[-1] >= self.num_token_types:
            return tokens[..., : self.num_token_types].argmax(dim=-1)
        return torch.zeros(tokens.shape[:-1], dtype=torch.long, device=tokens.device)

    def forward(
        self,
        tokens: Tensor,
        mask: Tensor | None = None,
        *,
        token_type_ids: Tensor | None = None,
        return_attention: bool = False,
    ) -> SetContextSummary:
        squeeze = False
        if tokens.ndim == 2:
            tokens = tokens.unsqueeze(0)
            squeeze = True
        if token_type_ids is None:
            token_type_ids = self._infer_token_type_ids(tokens)
        if token_type_ids.ndim == 1:
            token_type_ids = token_type_ids.unsqueeze(0)
        token_type_ids = token_type_ids.long()
        h = self.input_projection(tokens)
        if self.token_type_embedding is not None:
            h = h + self.token_type_embedding(token_type_ids)

        attention_maps: dict[str, Tensor] = {}
        if return_attention:
            h, isab_attention = self.isab(h, mask=mask, return_attention=True)
            h, sab_attention = self.sab(h, mask=mask, return_attention=True)
            pooled_tokens, pma_attention = self.pma(h, mask=mask, return_attention=True)
            attention_maps = {
                "isab_inducing_to_tokens": isab_attention["inducing_to_tokens"],
                "isab_tokens_to_inducing": isab_attention["tokens_to_inducing"],
                "sab_self_attention": sab_attention,
                "pma_seed_attention": pma_attention,
            }
        else:
            h = self.isab(h, mask=mask)
            h = self.sab(h, mask=mask)
            pooled_tokens = self.pma(h, mask=mask)
        pooled = pooled_tokens[:, 0, :]
        context = self.context_head(pooled)
        if squeeze:
            return SetContextSummary(
                pooled_context=context[0],
                token_embeddings=h[0],
                attention_maps={key: value[0] for key, value in attention_maps.items()},
                token_type_ids=token_type_ids[0],
            )
        return SetContextSummary(
            pooled_context=context,
            token_embeddings=h,
            attention_maps=attention_maps,
            token_type_ids=token_type_ids,
        )


class PooledContextEncoder(nn.Module):
    """Simple pooled baseline over typed token sets."""

    def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.summary_mlp = nn.Sequential(
            nn.Linear(int(input_dim) * 3, int(hidden_dim)),
            nn.GELU(),
            nn.LayerNorm(int(hidden_dim)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.LayerNorm(int(hidden_dim)),
        )

    def forward(self, tokens: Tensor) -> SetContextSummary:
        if tokens.ndim != 2:
            raise ValueError(f"tokens must be 2D, got shape {tuple(tokens.shape)}.")
        token_mean = tokens.mean(dim=0)
        token_std = tokens.std(dim=0, unbiased=False)
        token_max = tokens.max(dim=0).values
        pooled = torch.cat([token_mean, token_std, token_max], dim=0)
        context = self.summary_mlp(pooled.unsqueeze(0))[0]
        return SetContextSummary(pooled_context=context, token_embeddings=tokens)
