"""Niche tokenizer with learned ring pooling.

Converts raw cell data (variable cells per ring) into the 9-token structure
using learned Set Transformer pooling instead of mean pooling.

Architecture per ring: ISAB → PMA → ring_token
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from stagebridge.context.layers import RingPooler, PMA


class NicheTokenizer(nn.Module):
    """Convert raw neighborhood data to 9-token structure with learned pooling.

    Takes individual cells per ring and pools them using learned attention,
    then assembles the 9-token sequence:
    [receiver, ring_1, ring_2, ring_3, ring_4, hlca, luca, pathway, stats]

    When use_fused_reference=True (for GW fusion), uses 8 tokens:
    [receiver, ring_1, ring_2, ring_3, ring_4, fused_ref, pathway, stats]

    Args:
        input_dim: Raw cell embedding dimension
        hidden_dim: Output token dimension
        num_rings: Number of spatial rings (default 4)
        num_heads: Attention heads for ring pooling
        num_inducing: ISAB inducing points per ring
        dropout: Dropout rate
        use_fused_reference: Use single fused reference token (for GW fusion)
        fused_ref_dim: Dimension of fused reference (if use_fused_reference)
    """

    NUM_RINGS = 4
    NUM_TOKENS = 9  # receiver + 4 rings + hlca + luca + pathway + stats
    NUM_TOKENS_FUSED = 8  # receiver + 4 rings + fused_ref + pathway + stats

    # Fixed dimensions from contracts
    HLCA_DIM = 30
    LUCA_DIM = 10

    def __init__(
        self,
        input_dim: int = 40,
        hidden_dim: int = 128,
        num_rings: int = 4,
        num_heads: int = 4,
        num_inducing: int = 4,
        dropout: float = 0.1,
        stats_dim: int = 5,  # caf_fraction, immune_fraction, diversity, S_score, G2M_score
        pathway_dim: int | None = None,  # Auto-detect from data, or use input_dim if None
        use_fused_reference: bool = False,
        fused_ref_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_rings = num_rings
        self.stats_dim = stats_dim
        self.pathway_dim = pathway_dim if pathway_dim is not None else input_dim
        self.use_fused_reference = use_fused_reference
        self.fused_ref_dim = fused_ref_dim or input_dim

        # Projection for receiver (40d = HLCA+LuCA concat)
        self.token_proj = nn.Linear(input_dim, hidden_dim)

        # Separate projection for pathway features (detected from data or fallback to input_dim)
        self.pathway_proj = nn.Linear(self.pathway_dim, hidden_dim)

        # Separate projections for HLCA (30d) and LuCA (10d) reference tokens
        self.hlca_proj = nn.Linear(self.HLCA_DIM, hidden_dim)
        self.luca_proj = nn.Linear(self.LUCA_DIM, hidden_dim)

        # Projection for fused reference (when GW fusion enabled)
        if use_fused_reference:
            self.fused_ref_proj = nn.Linear(self.fused_ref_dim, hidden_dim)

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

        # Learned pooling for SSL reconstruction (pool context tokens, not receiver)
        self.ssl_pooler = PMA(
            dim=hidden_dim,
            num_heads=num_heads,
            num_seed_vectors=1,
            dropout=dropout,
        )

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
        fused_ref: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, list[Tensor]]:
        """Tokenize neighborhood with learned ring pooling.

        Args:
            receiver: [B, D] receiver cell embedding
            ring_cells: List of 4 tensors, each [B, max_cells, D] cells in that ring
            ring_masks: List of 4 tensors, each [B, max_cells] boolean masks
            hlca: [B, D] HLCA reference embedding (ignored if fused_ref provided)
            luca: [B, D] LuCA reference embedding (ignored if fused_ref provided)
            pathway: [B, D] pathway features (optional, zeros if None)
            stats: [B, D] stats features (optional, zeros if None)
            fused_ref: [B, D_fused] GW-fused reference (if use_fused_reference)

        Returns:
            tokens: [B, 9, hidden_dim] or [B, 8, hidden_dim] token sequence
            receiver_reconstruction: [B, input_dim] reconstructed receiver (for SSL)
            ring_attention: List of attention weights from ring pooling
        """
        B = receiver.shape[0]
        device = receiver.device

        # Project fixed tokens
        receiver_token = self.token_proj(receiver)

        if pathway is not None:
            pathway_token = self.pathway_proj(pathway)
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
            mask = ring_masks[i] if ring_masks is not None else None

            ring_token = pooler(cells, mask=mask)  # [B, hidden_dim]
            ring_tokens.append(ring_token)

        # Assemble token sequence
        if self.use_fused_reference and fused_ref is not None:
            # 8-token sequence with single fused reference
            fused_ref_token = self.fused_ref_proj(fused_ref)
            tokens = torch.stack([
                receiver_token,
                ring_tokens[0],
                ring_tokens[1],
                ring_tokens[2],
                ring_tokens[3],
                fused_ref_token,
                pathway_token,
                stats_token,
            ], dim=1)  # [B, 8, hidden_dim]

            # Context tokens WITHOUT receiver for SSL reconstruction
            context_only_tokens = torch.stack([
                ring_tokens[0],
                ring_tokens[1],
                ring_tokens[2],
                ring_tokens[3],
                fused_ref_token,
                pathway_token,
                stats_token,
            ], dim=1)  # [B, 7, hidden_dim]
        else:
            # Standard 9-token sequence with separate HLCA/LuCA projections
            hlca_token = self.hlca_proj(hlca)  # 30d -> hidden_dim
            luca_token = self.luca_proj(luca)  # 10d -> hidden_dim
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

            # Context tokens WITHOUT receiver for SSL reconstruction
            context_only_tokens = torch.stack([
                ring_tokens[0],
                ring_tokens[1],
                ring_tokens[2],
                ring_tokens[3],
                hlca_token,
                luca_token,
                pathway_token,
                stats_token,
            ], dim=1)  # [B, 8, hidden_dim]

        # SSL Reconstruction: predict receiver from CONTEXT ONLY (no receiver leakage)
        # Use learned PMA pooling over context tokens (more expressive than mean)
        context_pooled = self.ssl_pooler(context_only_tokens)  # [B, 1, hidden_dim]
        context_pooled = context_pooled.squeeze(1)  # [B, hidden_dim]
        receiver_reconstruction = self.reconstruction_head(context_pooled)

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
