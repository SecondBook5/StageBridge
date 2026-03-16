"""Set Transformer components used by StageBridge context encoding."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

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

    def __init__(
        self, dim: int, num_heads: int = 8, num_seed_vectors: int = 1, dropout: float = 0.1
    ) -> None:
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
            torch.arange(half, device=device, dtype=dtype)
            * (-math.log(10_000.0) / max(half - 1, 1))
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
    context_tokens: Tensor | None = None
    group_summary_tokens: Tensor | None = None
    relation_tokens: Tensor | None = None
    attention_maps: dict[str, Tensor] = field(default_factory=dict)
    token_type_ids: Tensor | None = None
    token_confidence: Tensor | None = None
    token_coords: Tensor | None = None
    diagnostics: dict[str, float] = field(default_factory=dict)


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
        use_spatial_rpe: bool = True,
        token_dropout_rate: float = 0.05,
        use_confidence_gate: bool = True,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.num_token_types = int(num_token_types)
        self.token_dropout_rate = float(token_dropout_rate)
        self.token_type_embedding = (
            nn.Embedding(self.num_token_types, hidden_dim) if use_token_type_embeddings else None
        )
        self.coord_projection = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.confidence_gate = (
            nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Sigmoid(),
            )
            if use_confidence_gate
            else None
        )
        self.isab = ISAB(
            dim=hidden_dim,
            num_heads=num_heads,
            num_inducing_points=num_inducing_points,
            dropout=dropout,
            use_spatial_rpe=use_spatial_rpe,
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

    def _normalize_by_group(self, tokens: Tensor, token_type_ids: Tensor) -> Tensor:
        normalized = tokens.clone()
        for group_idx in range(self.num_token_types):
            group_mask = token_type_ids == group_idx
            if not torch.any(group_mask):
                continue
            group_values = normalized[group_mask]
            group_mean = group_values.mean(dim=0, keepdim=True)
            group_std = group_values.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-4)
            normalized[group_mask] = (group_values - group_mean) / group_std
        return normalized

    @staticmethod
    def _normalize_coords(coords: Tensor | None) -> Tensor | None:
        if coords is None:
            return None
        centered = coords - coords.mean(dim=-2, keepdim=True)
        scale = centered.norm(dim=-1).quantile(0.95).clamp_min(1e-4)
        return centered / scale

    def _apply_training_dropout(
        self,
        tokens: Tensor,
        token_type_ids: Tensor,
        token_confidence: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        if not self.training or self.token_dropout_rate <= 0.0 or tokens.shape[-2] <= 1:
            return tokens, token_type_ids, token_confidence
        keep_prob = 1.0 - self.token_dropout_rate
        keep_mask = torch.rand(tokens.shape[:-1], device=tokens.device) < keep_prob
        if keep_mask.ndim == 1:
            keep_mask[0] = True
        else:
            keep_mask[..., 0] = True
        dropped = tokens * keep_mask.unsqueeze(-1).to(tokens.dtype)
        if token_confidence is not None:
            token_confidence = token_confidence * keep_mask.to(token_confidence.dtype)
        return dropped, token_type_ids, token_confidence

    def forward(
        self,
        tokens: Tensor,
        mask: Tensor | None = None,
        *,
        token_type_ids: Tensor | None = None,
        token_coords: Tensor | None = None,
        token_confidence: Tensor | None = None,
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
        if token_confidence is not None and token_confidence.ndim == 1:
            token_confidence = token_confidence.unsqueeze(0)
        if token_coords is not None and token_coords.ndim == 2:
            token_coords = token_coords.unsqueeze(0)

        tokens, token_type_ids, token_confidence = self._apply_training_dropout(
            tokens,
            token_type_ids,
            token_confidence,
        )
        normalized_tokens = torch.stack(
            [
                self._normalize_by_group(batch_tokens, batch_type_ids)
                for batch_tokens, batch_type_ids in zip(tokens, token_type_ids, strict=False)
            ],
            dim=0,
        )
        h = self.input_projection(normalized_tokens)
        if self.token_type_embedding is not None:
            h = h + self.token_type_embedding(token_type_ids)
        normalized_coords = self._normalize_coords(token_coords)
        if normalized_coords is not None:
            h = h + self.coord_projection(normalized_coords)
        confidence_gate_mean = 1.0
        if token_confidence is not None and self.confidence_gate is not None:
            gate = self.confidence_gate(token_confidence.unsqueeze(-1))
            confidence_gate_mean = float(gate.detach().mean().item())
            h = h * gate

        attention_maps: dict[str, Tensor] = {}
        if return_attention:
            h, isab_attention = self.isab(
                h,
                mask=mask,
                coords=normalized_coords,
                n_src=0,
                return_attention=True,
            )
            h, sab_attention = self.sab(h, mask=mask, return_attention=True)
            pooled_tokens, pma_attention = self.pma(h, mask=mask, return_attention=True)
            attention_maps = {
                "isab_inducing_to_tokens": isab_attention["inducing_to_tokens"],
                "isab_tokens_to_inducing": isab_attention["tokens_to_inducing"],
                "sab_self_attention": sab_attention,
                "pma_seed_attention": pma_attention,
            }
        else:
            h = self.isab(h, mask=mask, coords=normalized_coords, n_src=0)
            h = self.sab(h, mask=mask)
            pooled_tokens = self.pma(h, mask=mask)
        pooled = pooled_tokens[:, 0, :]
        context = self.context_head(pooled)
        diagnostics = {
            "confidence_gate_mean": confidence_gate_mean,
            "mean_token_confidence": float(token_confidence.detach().mean().item())
            if token_confidence is not None
            else 1.0,
            "mean_token_radius": float(normalized_coords.detach().norm(dim=-1).mean().item())
            if normalized_coords is not None
            else 0.0,
        }
        if squeeze:
            return SetContextSummary(
                pooled_context=context[0],
                token_embeddings=h[0],
                context_tokens=pooled_tokens[0],
                attention_maps={key: value[0] for key, value in attention_maps.items()},
                token_type_ids=token_type_ids[0],
                token_confidence=None if token_confidence is None else token_confidence[0],
                token_coords=None if normalized_coords is None else normalized_coords[0],
                diagnostics=diagnostics,
            )
        return SetContextSummary(
            pooled_context=context,
            token_embeddings=h,
            context_tokens=pooled_tokens,
            attention_maps=attention_maps,
            token_type_ids=token_type_ids,
            token_confidence=token_confidence,
            token_coords=normalized_coords,
            diagnostics=diagnostics,
        )


class DeepSetsTransformerHybridEncoder(nn.Module):
    """Deep Sets backbone with gated transformer refinement.

    The baseline permutation-invariant path is always available. The transformer
    only adds a residual refinement and exposes a token bank for downstream
    cross-attention drift.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_inducing_points: int = 16,
        num_seed_vectors: int = 2,
        dropout: float = 0.1,
        *,
        num_token_types: int = 4,
        use_token_type_embeddings: bool = True,
        use_spatial_rpe: bool = True,
        token_dropout_rate: float = 0.05,
        use_confidence_gate: bool = True,
    ) -> None:
        super().__init__()
        self.num_token_types = int(num_token_types)
        self.token_dropout_rate = float(token_dropout_rate)

        self.input_projection = nn.Linear(int(input_dim), int(hidden_dim))
        self.token_type_embedding = (
            nn.Embedding(self.num_token_types, int(hidden_dim))
            if use_token_type_embeddings
            else None
        )
        self.coord_projection = nn.Sequential(
            nn.Linear(2, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
        )
        self.confidence_gate = (
            nn.Sequential(
                nn.Linear(1, int(hidden_dim)),
                nn.GELU(),
                nn.Linear(int(hidden_dim), int(hidden_dim)),
                nn.Sigmoid(),
            )
            if use_confidence_gate
            else None
        )
        self.phi = nn.Sequential(
            nn.Linear(int(hidden_dim), int(hidden_dim)),
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
        self.isab = ISAB(
            dim=int(hidden_dim),
            num_heads=int(num_heads),
            num_inducing_points=int(num_inducing_points),
            dropout=float(dropout),
            use_spatial_rpe=bool(use_spatial_rpe),
        )
        self.sab = SAB(dim=int(hidden_dim), num_heads=int(num_heads), dropout=float(dropout))
        self.pma = PMA(
            dim=int(hidden_dim),
            num_heads=int(num_heads),
            num_seed_vectors=int(num_seed_vectors),
            dropout=float(dropout),
        )
        self.transformer_head = nn.Sequential(
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.LayerNorm(int(hidden_dim)),
        )
        self.hybrid_gate = nn.Sequential(
            nn.Linear(int(hidden_dim) * 2, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.Sigmoid(),
        )

    def _infer_token_type_ids(self, tokens: Tensor) -> Tensor:
        usable = min(int(tokens.shape[-1]), self.num_token_types)
        return tokens[..., :usable].argmax(dim=-1).long()

    def _normalize_by_group(self, tokens: Tensor, token_type_ids: Tensor) -> Tensor:
        normalized = tokens.clone()
        for group_idx in range(self.num_token_types):
            group_mask = token_type_ids == group_idx
            if not torch.any(group_mask):
                continue
            group_values = normalized[group_mask]
            group_mean = group_values.mean(dim=0, keepdim=True)
            group_std = group_values.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-4)
            normalized[group_mask] = (group_values - group_mean) / group_std
        return normalized

    @staticmethod
    def _normalize_coords(coords: Tensor | None) -> Tensor | None:
        if coords is None:
            return None
        centered = coords - coords.mean(dim=-2, keepdim=True)
        scale = centered.norm(dim=-1).quantile(0.95).clamp_min(1e-4)
        return centered / scale

    def _apply_training_dropout(
        self,
        tokens: Tensor,
        token_type_ids: Tensor,
        token_confidence: Tensor | None,
        token_coords: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
        if not self.training or self.token_dropout_rate <= 0.0 or tokens.shape[-2] <= 1:
            return tokens, token_type_ids, token_confidence, token_coords
        keep_prob = 1.0 - self.token_dropout_rate
        keep_mask = torch.rand(tokens.shape[:-1], device=tokens.device) < keep_prob
        keep_mask[..., 0] = True
        dropped_tokens = tokens * keep_mask.unsqueeze(-1).to(tokens.dtype)
        dropped_confidence = (
            None
            if token_confidence is None
            else token_confidence * keep_mask.to(token_confidence.dtype)
        )
        dropped_coords = (
            None
            if token_coords is None
            else token_coords * keep_mask.unsqueeze(-1).to(token_coords.dtype)
        )
        return dropped_tokens, token_type_ids, dropped_confidence, dropped_coords

    def forward(
        self,
        tokens: Tensor,
        mask: Tensor | None = None,
        *,
        token_type_ids: Tensor | None = None,
        token_coords: Tensor | None = None,
        token_confidence: Tensor | None = None,
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
        if token_confidence is not None and token_confidence.ndim == 1:
            token_confidence = token_confidence.unsqueeze(0)
        if token_coords is not None and token_coords.ndim == 2:
            token_coords = token_coords.unsqueeze(0)

        tokens, token_type_ids, token_confidence, token_coords = self._apply_training_dropout(
            tokens,
            token_type_ids,
            token_confidence,
            token_coords,
        )
        normalized_tokens = torch.stack(
            [
                self._normalize_by_group(batch_tokens, batch_type_ids)
                for batch_tokens, batch_type_ids in zip(tokens, token_type_ids, strict=False)
            ],
            dim=0,
        )
        normalized_coords = self._normalize_coords(token_coords)
        h = self.input_projection(normalized_tokens)
        if self.token_type_embedding is not None:
            h = h + self.token_type_embedding(token_type_ids)
        if normalized_coords is not None:
            h = h + self.coord_projection(normalized_coords)

        confidence_gate_mean = 1.0
        if token_confidence is not None and self.confidence_gate is not None:
            gate = self.confidence_gate(token_confidence.unsqueeze(-1))
            confidence_gate_mean = float(gate.detach().mean().item())
            h = h * gate

        deep_embeddings = self.phi(h)
        pooled_mean = deep_embeddings.mean(dim=1)
        pooled_max = deep_embeddings.max(dim=1).values
        deep_context = self.rho(torch.cat([pooled_mean, pooled_max], dim=-1))

        attention_maps: dict[str, Tensor] = {}
        if return_attention:
            transformer_h, isab_attention = self.isab(
                deep_embeddings,
                mask=mask,
                coords=normalized_coords,
                n_src=0,
                return_attention=True,
            )
            transformer_h, sab_attention = self.sab(
                transformer_h, mask=mask, return_attention=True
            )
            pooled_tokens, pma_attention = self.pma(
                transformer_h, mask=mask, return_attention=True
            )
            attention_maps = {
                "hybrid_isab_inducing_to_tokens": isab_attention["inducing_to_tokens"],
                "hybrid_isab_tokens_to_inducing": isab_attention["tokens_to_inducing"],
                "hybrid_sab_attention": sab_attention,
                "hybrid_pma_seed_attention": pma_attention,
                "pma_seed_attention": pma_attention,
            }
        else:
            transformer_h = self.isab(
                deep_embeddings, mask=mask, coords=normalized_coords, n_src=0
            )
            transformer_h = self.sab(transformer_h, mask=mask)
            pooled_tokens = self.pma(transformer_h, mask=mask)

        transformer_summary = self.transformer_head(pooled_tokens.mean(dim=1))
        gate = self.hybrid_gate(torch.cat([deep_context, transformer_summary], dim=-1))
        fused_context = deep_context + gate * transformer_summary
        baseline_token = deep_context.unsqueeze(1)
        drift_tokens = torch.cat([baseline_token, pooled_tokens], dim=1)
        diagnostics = {
            "confidence_gate_mean": confidence_gate_mean,
            "mean_token_confidence": float(token_confidence.detach().mean().item())
            if token_confidence is not None
            else 1.0,
            "mean_token_radius": float(normalized_coords.detach().norm(dim=-1).mean().item())
            if normalized_coords is not None
            else 0.0,
            "hybrid_gate_mean": float(gate.detach().mean().item()),
            "deep_sets_context_norm": float(deep_context.detach().norm(dim=-1).mean().item()),
            "transformer_refinement_norm": float(
                transformer_summary.detach().norm(dim=-1).mean().item()
            ),
        }
        if squeeze:
            return SetContextSummary(
                pooled_context=fused_context[0],
                token_embeddings=deep_embeddings[0],
                context_tokens=drift_tokens[0],
                attention_maps={key: value[0] for key, value in attention_maps.items()},
                token_type_ids=token_type_ids[0],
                token_confidence=None if token_confidence is None else token_confidence[0],
                token_coords=None if normalized_coords is None else normalized_coords[0],
                diagnostics=diagnostics,
            )
        return SetContextSummary(
            pooled_context=fused_context,
            token_embeddings=deep_embeddings,
            context_tokens=drift_tokens,
            attention_maps=attention_maps,
            token_type_ids=token_type_ids,
            token_confidence=token_confidence,
            token_coords=normalized_coords,
            diagnostics=diagnostics,
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


class SetTransformer(nn.Module):
    """
    Standard Set Transformer for hierarchical set aggregation.

    Combines ISAB (induced set attention blocks) with PMA (pooling by multihead attention)
    for efficient permutation-invariant processing of variable-size sets.

    Args:
        dim_input: Input feature dimension
        dim_hidden: Hidden dimension (used throughout)
        dim_output: Output dimension
        num_heads: Number of attention heads
        num_inds: Number of inducing points for ISAB
        ln: Use layer normalization
    """

    def __init__(
        self,
        dim_input: int,
        dim_hidden: int = 128,
        dim_output: int = 128,
        num_heads: int = 4,
        num_inds: int = 16,
        ln: bool = True,
    ):
        super().__init__()

        # Input projection
        self.input_proj = nn.Linear(dim_input, dim_hidden)

        # ISAB layers for hierarchical processing
        self.isab1 = ISAB(dim_hidden, num_heads, num_inds)
        self.isab2 = ISAB(dim_hidden, num_heads, num_inds)

        # PMA for pooling to single vector
        self.pma = PMA(dim_hidden, num_heads, num_seed_vectors=1)

        # Output projection
        self.output_proj = nn.Linear(dim_hidden, dim_output)

        # Optional layer norm
        self.ln = nn.LayerNorm(dim_output) if ln else nn.Identity()

    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        """
        Forward pass through Set Transformer.

        Args:
            x: Input tensor (batch_size, num_elements, dim_input)
            mask: Optional mask (batch_size, num_elements)

        Returns:
            Pooled output (batch_size, dim_output)
        """
        # Project input
        x = self.input_proj(x)

        # ISAB layers
        x = self.isab1(x, mask=mask)
        x = self.isab2(x, mask=mask)

        # PMA pooling
        x = self.pma(x, mask=mask)  # (batch_size, 1, dim_hidden)
        x = x.squeeze(1)  # (batch_size, dim_hidden)

        # Output projection
        x = self.output_proj(x)
        x = self.ln(x)

        return x
