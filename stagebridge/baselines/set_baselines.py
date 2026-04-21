"""Set-based baselines for niche classification.

These baselines properly test H2: "Cross-sectional progression becomes more
identifiable when conditioned on receiver-centered local niche context."

Each baseline takes a SET of tokens (receiver + neighbors) and outputs
a single classification for the receiver's stage.

Input format: [batch, K, D] where K=9 tokens, D=40 dim
Output format: [batch, num_classes] logits

Baseline ladder (increasing structural bias):
1. PoolingMLP - Mean pool, no structure
2. DeepSets - Permutation invariant, no spatial
3. SetTransformer - Self-attention, no receiver privilege
4. GraphSAGE - Spatial structure via aggregation
5. ReceiverCentered - Receiver as query (our model)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class BaselineOutput:
    """Standardized output for all baselines."""
    logits: Tensor  # [B, num_classes]
    embedding: Tensor  # [B, hidden_dim] - for analysis


class PoolingMLPBaseline(nn.Module):
    """Baseline 1: Mean pooling + MLP.

    No set structure - just averages all tokens.
    This is the weakest baseline (bag-of-cells).

    Tests: Does ANY structure help?
    """

    def __init__(
        self,
        input_dim: int = 40,
        hidden_dim: int = 128,
        num_classes: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, tokens: Tensor, mask: Tensor | None = None) -> BaselineOutput:
        """
        Args:
            tokens: [B, K, D] token embeddings
            mask: [B, K] valid token mask (True = valid)

        Returns:
            BaselineOutput with logits and embedding
        """
        # Mean pool over tokens
        if mask is not None:
            tokens = tokens * mask.unsqueeze(-1).float()
            pooled = tokens.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        else:
            pooled = tokens.mean(dim=1)  # [B, D]

        embedding = self.encoder(pooled)  # [B, hidden]
        logits = self.classifier(embedding)  # [B, num_classes]

        return BaselineOutput(logits=logits, embedding=embedding)


class DeepSetsBaseline(nn.Module):
    """Baseline 2: DeepSets architecture.

    Permutation invariant: phi(each token) -> sum -> rho -> classify

    Tests: Does permutation invariance help beyond pooling?
    """

    def __init__(
        self,
        input_dim: int = 40,
        hidden_dim: int = 128,
        num_classes: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Per-element encoder
        self.phi = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        # Set-level decoder
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, tokens: Tensor, mask: Tensor | None = None) -> BaselineOutput:
        """
        Args:
            tokens: [B, K, D] token embeddings
            mask: [B, K] valid token mask

        Returns:
            BaselineOutput with logits and embedding
        """
        # Apply phi to each token
        h = self.phi(tokens)  # [B, K, hidden]

        # Sum pooling (permutation invariant)
        if mask is not None:
            h = h * mask.unsqueeze(-1).float()
        pooled = h.sum(dim=1)  # [B, hidden]

        # Apply rho
        embedding = self.rho(pooled)  # [B, hidden]
        logits = self.classifier(embedding)  # [B, num_classes]

        return BaselineOutput(logits=logits, embedding=embedding)


class SetTransformerBaseline(nn.Module):
    """Baseline 3: Set Transformer with self-attention.

    All tokens attend to all tokens equally (no receiver privilege).

    Tests: Does attention help? (But receiver not privileged)
    """

    def __init__(
        self,
        input_dim: int = 40,
        hidden_dim: int = 128,
        num_classes: int = 5,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Standard transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Pool and classify
        self.pool_proj = nn.Linear(hidden_dim, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, tokens: Tensor, mask: Tensor | None = None) -> BaselineOutput:
        """
        Args:
            tokens: [B, K, D] token embeddings
            mask: [B, K] valid token mask

        Returns:
            BaselineOutput with logits and embedding
        """
        # Project to hidden dim
        h = self.input_proj(tokens)  # [B, K, hidden]

        # Transformer encoder
        # Note: TransformerEncoder mask is True = IGNORE
        src_key_padding_mask = None
        if mask is not None:
            src_key_padding_mask = ~mask  # Invert: True = ignore
            # Handle edge case: if all tokens are masked in a sample, unmask at least one
            all_masked = src_key_padding_mask.all(dim=1)
            if all_masked.any():
                src_key_padding_mask[all_masked, 0] = False

        h = self.transformer(h, src_key_padding_mask=src_key_padding_mask)  # [B, K, hidden]

        # Mean pool
        if mask is not None:
            h = h * mask.unsqueeze(-1).float()
            pooled = h.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        else:
            pooled = h.mean(dim=1)  # [B, hidden]

        embedding = self.pool_proj(pooled)
        logits = self.classifier(embedding)

        return BaselineOutput(logits=logits, embedding=embedding)


class GraphSAGEBaseline(nn.Module):
    """Baseline 4: GraphSAGE-style spatial aggregation.

    Receiver aggregates information from neighbors.
    Assumes token 0 is receiver, tokens 1-8 are neighbors.

    Tests: Does spatial structure help? (But no cross-attention)
    """

    def __init__(
        self,
        input_dim: int = 40,
        hidden_dim: int = 128,
        num_classes: int = 5,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # GraphSAGE layers: aggregate neighbors then combine with self
        self.sage_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.sage_layers.append(nn.ModuleDict({
                'neighbor_proj': nn.Linear(hidden_dim, hidden_dim),
                'self_proj': nn.Linear(hidden_dim, hidden_dim),
                'combine': nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.GELU(),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout),
                ),
            }))

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, tokens: Tensor, mask: Tensor | None = None) -> BaselineOutput:
        """
        Args:
            tokens: [B, K, D] token embeddings (token 0 = receiver)
            mask: [B, K] valid token mask

        Returns:
            BaselineOutput with logits and embedding
        """
        # Project to hidden
        h = self.input_proj(tokens)  # [B, K, hidden]

        # Split receiver and neighbors
        h_receiver = h[:, 0, :]  # [B, hidden]
        h_neighbors = h[:, 1:, :]  # [B, K-1, hidden]

        # Neighbor mask (if provided)
        neighbor_mask = None
        if mask is not None:
            neighbor_mask = mask[:, 1:]  # [B, K-1]

        # GraphSAGE aggregation
        for layer in self.sage_layers:
            # Aggregate neighbors (mean)
            if neighbor_mask is not None:
                h_neighbors_masked = h_neighbors * neighbor_mask.unsqueeze(-1).float()
                h_agg = h_neighbors_masked.sum(dim=1) / neighbor_mask.sum(dim=1, keepdim=True).clamp(min=1)
            else:
                h_agg = h_neighbors.mean(dim=1)  # [B, hidden]

            # Project
            h_neigh_proj = layer['neighbor_proj'](h_agg)
            h_self_proj = layer['self_proj'](h_receiver)

            # Combine
            h_receiver = layer['combine'](torch.cat([h_self_proj, h_neigh_proj], dim=-1))

        embedding = h_receiver  # [B, hidden]
        logits = self.classifier(embedding)

        return BaselineOutput(logits=logits, embedding=embedding)


class ReceiverCenteredBaseline(nn.Module):
    """Baseline 5: Receiver-centered cross-attention.

    Receiver is query, neighbors are key/value.
    This is essentially our model (simplified version).

    Tests: Does receiver-centrality help?
    """

    def __init__(
        self,
        input_dim: int = 40,
        hidden_dim: int = 128,
        num_classes: int = 5,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Cross-attention layers
        self.cross_attn_layers = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norms1 = nn.ModuleList()
        self.norms2 = nn.ModuleList()

        for _ in range(num_layers):
            self.cross_attn_layers.append(
                nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
            )
            self.ffn_layers.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.Dropout(dropout),
            ))
            self.norms1.append(nn.LayerNorm(hidden_dim))
            self.norms2.append(nn.LayerNorm(hidden_dim))

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, tokens: Tensor, mask: Tensor | None = None) -> BaselineOutput:
        """
        Args:
            tokens: [B, K, D] token embeddings (token 0 = receiver)
            mask: [B, K] valid token mask

        Returns:
            BaselineOutput with logits and embedding
        """
        # Project to hidden
        h = self.input_proj(tokens)  # [B, K, hidden]

        # Split receiver and neighbors
        h_receiver = h[:, 0:1, :]  # [B, 1, hidden]
        h_neighbors = h[:, 1:, :]  # [B, K-1, hidden]

        # Neighbor mask for attention
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = ~mask[:, 1:]  # [B, K-1], True = ignore

        # Cross-attention layers
        for attn, ffn, norm1, norm2 in zip(
            self.cross_attn_layers, self.ffn_layers, self.norms1, self.norms2
        ):
            # Cross-attention: receiver queries neighbors
            attn_out, _ = attn(
                query=h_receiver,
                key=h_neighbors,
                value=h_neighbors,
                key_padding_mask=key_padding_mask,
            )
            h_receiver = norm1(h_receiver + attn_out)
            h_receiver = norm2(h_receiver + ffn(h_receiver))

        embedding = h_receiver.squeeze(1)  # [B, hidden]
        logits = self.classifier(embedding)

        return BaselineOutput(logits=logits, embedding=embedding)


# Registry for easy access
BASELINE_REGISTRY = {
    "pooling_mlp": PoolingMLPBaseline,
    "deep_sets": DeepSetsBaseline,
    "set_transformer": SetTransformerBaseline,
    "graph_sage": GraphSAGEBaseline,
    "receiver_centered": ReceiverCenteredBaseline,
}


def create_baseline(name: str, **kwargs) -> nn.Module:
    """Create a baseline by name.

    Args:
        name: Baseline name (pooling_mlp, deep_sets, set_transformer, graph_sage, receiver_centered)
        **kwargs: Arguments passed to baseline constructor

    Returns:
        Baseline model
    """
    if name not in BASELINE_REGISTRY:
        raise ValueError(f"Unknown baseline: {name}. Available: {list(BASELINE_REGISTRY.keys())}")
    return BASELINE_REGISTRY[name](**kwargs)
