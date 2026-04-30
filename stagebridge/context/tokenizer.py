"""Niche tokenizer with learned ring pooling.

Converts raw cell data (variable cells per ring) into the 9-token structure
using learned Set Transformer pooling instead of mean pooling.

Architecture per ring: ISAB → PMA → ring_token
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from stagebridge.context.layers import RingPooler


class NicheTokenizer(nn.Module):
    """Convert raw neighborhood data to 9-token structure with learned pooling.

    Takes individual cells per ring and pools them using learned attention,
    then assembles the 9-token sequence:
    [receiver, ring_1, ring_2, ring_3, ring_4, hlca, luca, pathway, stats]

    Args:
        input_dim: Raw cell embedding dimension
        hidden_dim: Output token dimension
        num_rings: Number of spatial rings (default 4)
        num_heads: Attention heads for ring pooling
        num_inducing: ISAB inducing points per ring
        dropout: Dropout rate
    """

    NUM_RINGS = 4
    NUM_TOKENS = 9  # receiver + 4 rings + hlca + luca + pathway + stats

    def __init__(
        self,
        input_dim: int = 40,
        hidden_dim: int = 128,
        num_rings: int = 4,
        num_heads: int = 4,
        num_inducing: int = 4,
        dropout: float = 0.1,
        stats_dim: int = 7,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_rings = num_rings
        self.stats_dim = stats_dim

        # Projection for non-pooled tokens (receiver, hlca, luca, pathway)
        self.token_proj = nn.Linear(input_dim, hidden_dim)

        # Separate projection for stats (different dimension)
        self.stats_proj = nn.Linear(stats_dim, hidden_dim)

        # Learned pooling for each ring
        self.ring_poolers = nn.ModuleList([
            RingPooler(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_inducing=num_inducing,
                dropout=dropout,
            )
            for _ in range(num_rings)
        ])

        # Reconstruction head: project back to input_dim for SSL
        self.reconstruction_head = nn.Linear(hidden_dim, input_dim)

    def forward(
        self,
        receiver: Tensor,
        ring_cells: list[Tensor],
        ring_masks: list[Tensor] | None,
        hlca: Tensor,
        luca: Tensor,
        pathway: Tensor | None = None,
        stats: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, list[Tensor]]:
        """Tokenize neighborhood with learned ring pooling.

        Args:
            receiver: [B, D] receiver cell embedding
            ring_cells: List of 4 tensors, each [B, max_cells, D] cells in that ring
            ring_masks: List of 4 tensors, each [B, max_cells] boolean masks
            hlca: [B, D] HLCA reference embedding
            luca: [B, D] LuCA reference embedding
            pathway: [B, D] pathway features (optional, zeros if None)
            stats: [B, D] stats features (optional, zeros if None)

        Returns:
            tokens: [B, 9, hidden_dim] the 9-token sequence
            receiver_reconstruction: [B, input_dim] reconstructed receiver (for SSL)
            ring_attention: List of attention weights from ring pooling
        """
        B = receiver.shape[0]
        device = receiver.device

        # Project fixed tokens
        receiver_token = self.token_proj(receiver)
        hlca_token = self.token_proj(hlca)
        luca_token = self.token_proj(luca)

        if pathway is not None:
            pathway_token = self.token_proj(pathway)
        else:
            pathway_token = torch.zeros(B, self.hidden_dim, device=device)

        if stats is not None:
            stats_token = self.stats_proj(stats)
        else:
            stats_token = torch.zeros(B, self.hidden_dim, device=device)

        # Pool each ring with learned attention
        ring_tokens = []
        ring_attention = []
        for i, pooler in enumerate(self.ring_poolers):
            cells = ring_cells[i]  # [B, max_cells, D]
            mask = ring_masks[i] if ring_masks else None

            ring_token = pooler(cells, mask=mask)  # [B, hidden_dim]
            ring_tokens.append(ring_token)

        # Assemble 9-token sequence
        # Order: [receiver, ring_1, ring_2, ring_3, ring_4, hlca, luca, pathway, stats]
        tokens = torch.stack([
            receiver_token,
            ring_tokens[0],
            ring_tokens[1],
            ring_tokens[2],
            ring_tokens[3],
            hlca_token,
            luca_token,
            pathway_token,
            stats_token,
        ], dim=1)  # [B, 9, hidden_dim]

        # Reconstruction: project receiver_token back to input_dim for SSL
        receiver_reconstruction = self.reconstruction_head(receiver_token)

        return tokens, receiver_reconstruction, ring_attention


class NicheTokenizerConfig:
    """Configuration for NicheTokenizer."""

    input_dim: int = 40
    hidden_dim: int = 128
    num_rings: int = 4
    num_heads: int = 4
    num_inducing: int = 4
    dropout: float = 0.1
    max_cells_per_ring: int = 50
