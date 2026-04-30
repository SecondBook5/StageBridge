"""Set Transformer baseline for StageBridge.

Tests flat attention without receiver-centering or spatial structure.
Reference: Lee et al. "Set Transformer" (ICML 2019)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from stagebridge.contracts import LATENT_DIM, N_STAGES


@dataclass(slots=True, frozen=True)
class SetTransformerConfig:
    """Configuration for Set Transformer baseline."""

    input_dim: int = LATENT_DIM
    hidden_dim: int = 128
    output_dim: int = LATENT_DIM
    num_stages: int = N_STAGES
    num_heads: int = 4
    num_layers: int = 2
    num_inducing: int = 8
    time_dim: int = 32
    stage_dim: int = 32
    dropout: float = 0.1

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


class MAB(nn.Module):
    """Multihead Attention Block."""

    def __init__(
        self,
        dim_q: int,
        dim_kv: int,
        dim_out: int,
        num_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=dim_q,
            num_heads=num_heads,
            dropout=dropout,
            kdim=dim_kv,
            vdim=dim_kv,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(dim_q)
        self.norm2 = nn.LayerNorm(dim_out)
        self.ffn = nn.Sequential(
            nn.Linear(dim_q, dim_out * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_out * 2, dim_out),
            nn.Dropout(dropout),
        )
        self.proj = nn.Linear(dim_q, dim_out) if dim_q != dim_out else nn.Identity()

    def forward(
        self,
        q: Tensor,
        kv: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        attn_mask = None
        if mask is not None:
            attn_mask = ~mask

        attn_out, _ = self.attention(q, kv, kv, key_padding_mask=attn_mask)
        x = self.norm1(q + attn_out)
        x = self.proj(x)
        return self.norm2(x + self.ffn(x))


class SAB(nn.Module):
    """Self-Attention Block."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.mab = MAB(dim, dim, dim, num_heads, dropout)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        return self.mab(x, x, mask)


class ISAB(nn.Module):
    """Induced Set Attention Block.

    Uses inducing points for O(n*m) complexity instead of O(n^2).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_inducing: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.inducing = nn.Parameter(torch.randn(1, num_inducing, dim))
        self.mab1 = MAB(dim, dim, dim, num_heads, dropout)
        self.mab2 = MAB(dim, dim, dim, num_heads, dropout)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        b = x.shape[0]
        inducing = self.inducing.expand(b, -1, -1)
        h = self.mab1(inducing, x, mask)
        return self.mab2(x, h)


class PMA(nn.Module):
    """Pooling by Multihead Attention.

    Learns a query to pool the set into fixed-size output.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_seeds: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.seeds = nn.Parameter(torch.randn(1, num_seeds, dim))
        self.mab = MAB(dim, dim, dim, num_heads, dropout)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        b = x.shape[0]
        seeds = self.seeds.expand(b, -1, -1)
        return self.mab(seeds, x, mask)


class SetTransformer(nn.Module):
    """Set Transformer baseline: flat attention without spatial structure.

    Architecture:
        - Input projection
        - Stack of ISAB layers (efficient self-attention)
        - PMA for pooling
        - Output projection

    This tests whether flat attention (without receiver-centering or
    spatial distances) captures progression signal.

    Args:
        config: Model configuration
    """

    def __init__(self, config: SetTransformerConfig) -> None:
        super().__init__()
        self.config = config

        self.input_proj = nn.Linear(config.input_dim, config.hidden_dim)

        self.encoder = nn.ModuleList([
            ISAB(config.hidden_dim, config.num_heads, config.num_inducing, config.dropout)
            for _ in range(config.num_layers)
        ])

        self.pooling = PMA(config.hidden_dim, config.num_heads, num_seeds=1, dropout=config.dropout)

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

    def encode_context(
        self,
        receiver: Tensor,
        neighbors: Tensor,
        neighbor_mask: Tensor | None = None,
        **kwargs: object,
    ) -> Tensor:
        """Encode set via Set Transformer.

        Args:
            receiver: [B, D] receiver embeddings
            neighbors: [B, K, D] neighbor embeddings
            neighbor_mask: [B, K] valid neighbor mask

        Returns:
            [B, H] context embedding
        """
        all_tokens = torch.cat([receiver.unsqueeze(1), neighbors], dim=1)

        if neighbor_mask is not None:
            full_mask = torch.cat(
                [torch.ones_like(neighbor_mask[:, :1]), neighbor_mask], dim=1
            )
        else:
            full_mask = None

        x = self.input_proj(all_tokens)

        for layer in self.encoder:
            x = layer(x, full_mask)

        pooled = self.pooling(x, full_mask)

        return pooled.squeeze(1)

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
            distances: [B, K] distances (ignored - no spatial structure)
            x_t: [B, D] current state
            t: [B] time
            stage_pair_id: [B] stage indices
            neighbor_mask: [B, K] valid neighbor mask

        Returns:
            [B, D] drift velocity prediction
        """
        context = self.encode_context(receiver, neighbors, neighbor_mask)
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
