"""Local niche tokenization and encoding for EA-MIST."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from stagebridge.context_model.set_encoder import PMA, SAB


@dataclass(slots=True, frozen=True)
class LocalNicheEncoderOutput:
    """Structured output from the local niche encoder."""

    neighborhood_embedding: Tensor
    token_embeddings: Tensor
    attention_weights: Tensor | None = None


class LocalNicheTokenizer(nn.Module):
    """Convert structured local niche features into compact transformer tokens.

    Args:
        receiver_dim: Dimension of the receiver embedding.
        sender_feature_dim: Number of composition features per ring token.
        lr_summary_dim: Dimension of the compact LR/pathway summary vector.
        stats_dim: Dimension of the compact neighborhood stats vector.
        model_dim: Token embedding dimension.
        num_receiver_states: Size of the receiver state vocabulary.
        num_rings: Number of distance-ring tokens.
        dropout: Dropout rate applied after token projections.
    """

    def __init__(
        self,
        *,
        receiver_dim: int,
        sender_feature_dim: int,
        lr_summary_dim: int,
        stats_dim: int,
        model_dim: int = 128,
        num_receiver_states: int = 32,
        num_rings: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if receiver_dim <= 0 or sender_feature_dim <= 0 or lr_summary_dim <= 0 or stats_dim <= 0:
            raise ValueError("All local niche tokenizer input dimensions must be positive.")
        self.receiver_proj = nn.Linear(int(receiver_dim), int(model_dim))
        self.ring_proj = nn.Linear(int(sender_feature_dim), int(model_dim))
        self.lr_proj = nn.Linear(int(lr_summary_dim), int(model_dim))
        self.stats_proj = nn.Linear(int(stats_dim), int(model_dim))
        self.receiver_state_embedding = nn.Embedding(int(num_receiver_states), int(model_dim))
        self.token_type_embedding = nn.Embedding(4, int(model_dim))
        self.ring_embedding = nn.Embedding(int(num_rings), int(model_dim))
        self.dropout = nn.Dropout(float(dropout))

    def forward(
        self,
        *,
        receiver_embeddings: Tensor,
        receiver_state_ids: Tensor,
        ring_compositions: Tensor,
        lr_pathway_summary: Tensor,
        neighborhood_stats: Tensor,
    ) -> Tensor:
        """Return tokenized local neighborhoods with shape ``(B, T_local, D)``."""
        if receiver_embeddings.ndim != 2:
            raise ValueError(f"receiver_embeddings must be 2D, got shape={tuple(receiver_embeddings.shape)}")
        if receiver_state_ids.ndim != 1:
            raise ValueError(f"receiver_state_ids must be 1D, got shape={tuple(receiver_state_ids.shape)}")
        if ring_compositions.ndim != 3:
            raise ValueError(f"ring_compositions must be 3D, got shape={tuple(ring_compositions.shape)}")
        if lr_pathway_summary.ndim != 2 or neighborhood_stats.ndim != 2:
            raise ValueError("lr_pathway_summary and neighborhood_stats must be 2D tensors.")

        batch_size = receiver_embeddings.shape[0]
        if receiver_state_ids.shape[0] != batch_size or ring_compositions.shape[0] != batch_size:
            raise ValueError("All local niche tokenizer inputs must share the same batch dimension.")

        receiver_token = self.receiver_proj(receiver_embeddings)
        receiver_token = receiver_token + self.receiver_state_embedding(receiver_state_ids.clamp_min(0))
        receiver_token = receiver_token + self.token_type_embedding(torch.zeros(batch_size, dtype=torch.long, device=receiver_embeddings.device))

        num_rings = ring_compositions.shape[1]
        ring_tokens = self.ring_proj(ring_compositions)
        ring_type_ids = torch.ones((batch_size, num_rings), dtype=torch.long, device=ring_compositions.device)
        ring_ids = torch.arange(num_rings, device=ring_compositions.device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
        ring_tokens = ring_tokens + self.token_type_embedding(ring_type_ids) + self.ring_embedding(ring_ids)

        lr_token = self.lr_proj(lr_pathway_summary)
        lr_token = lr_token + self.token_type_embedding(torch.full((batch_size,), 2, dtype=torch.long, device=lr_pathway_summary.device))

        stats_token = self.stats_proj(neighborhood_stats)
        stats_token = stats_token + self.token_type_embedding(torch.full((batch_size,), 3, dtype=torch.long, device=neighborhood_stats.device))

        tokens = torch.cat(
            [
                receiver_token.unsqueeze(1),
                ring_tokens,
                lr_token.unsqueeze(1),
                stats_token.unsqueeze(1),
            ],
            dim=1,
        )
        return self.dropout(tokens)


class LocalNicheTransformerEncoder(nn.Module):
    """Compact transformer encoder over one local niche token set."""

    def __init__(
        self,
        *,
        receiver_dim: int,
        sender_feature_dim: int,
        lr_summary_dim: int,
        stats_dim: int,
        model_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        num_receiver_states: int = 32,
        num_rings: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.tokenizer = LocalNicheTokenizer(
            receiver_dim=receiver_dim,
            sender_feature_dim=sender_feature_dim,
            lr_summary_dim=lr_summary_dim,
            stats_dim=stats_dim,
            model_dim=model_dim,
            num_receiver_states=num_receiver_states,
            num_rings=num_rings,
            dropout=dropout,
        )
        self.blocks = nn.ModuleList(
            [SAB(dim=int(model_dim), num_heads=int(num_heads), dropout=float(dropout)) for _ in range(int(num_layers))]
        )
        self.pool = PMA(dim=int(model_dim), num_heads=int(num_heads), num_seed_vectors=1, dropout=float(dropout))
        self.norm = nn.LayerNorm(int(model_dim))

    def forward(
        self,
        *,
        receiver_embeddings: Tensor,
        receiver_state_ids: Tensor,
        ring_compositions: Tensor,
        lr_pathway_summary: Tensor,
        neighborhood_stats: Tensor,
        return_attention: bool = False,
    ) -> LocalNicheEncoderOutput:
        """Encode structured neighborhoods into one embedding each."""
        tokens = self.tokenizer(
            receiver_embeddings=receiver_embeddings,
            receiver_state_ids=receiver_state_ids,
            ring_compositions=ring_compositions,
            lr_pathway_summary=lr_pathway_summary,
            neighborhood_stats=neighborhood_stats,
        )
        attention = None
        hidden = tokens
        for layer_idx, block in enumerate(self.blocks):
            if return_attention and layer_idx == len(self.blocks) - 1:
                hidden, attention = block(hidden, return_attention=True)
            else:
                hidden = block(hidden)
        pooled = self.pool(hidden)
        pooled = self.norm(pooled[:, 0, :])
        return LocalNicheEncoderOutput(
            neighborhood_embedding=pooled,
            token_embeddings=hidden,
            attention_weights=attention,
        )


class LocalNicheMLPEncoder(nn.Module):
    """MLP fallback encoder over flattened neighborhood features."""

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}.")
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.GELU(),
            nn.LayerNorm(int(hidden_dim)),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.LayerNorm(int(hidden_dim)),
        )

    def forward(self, flat_features: Tensor) -> LocalNicheEncoderOutput:
        """Encode flattened neighborhood features."""
        if flat_features.ndim != 2:
            raise ValueError(f"flat_features must be 2D, got shape={tuple(flat_features.shape)}")
        hidden = self.net(flat_features)
        return LocalNicheEncoderOutput(neighborhood_embedding=hidden, token_embeddings=hidden.unsqueeze(1), attention_weights=None)
