"""Evolution-aware conditioning for WES feature fusion.

Fuses lesion-level evolution features (WES) into sample embeddings
via gated or FiLM conditioning.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class EvolutionBranch(nn.Module):
    """Project and fuse lesion-level evolution features.

    Args:
        evolution_dim: Input evolution feature dimension (WES)
        model_dim: Sample representation dimension
        mode: Conditioning mode ("gated" or "film")
        dropout: Dropout rate
    """

    def __init__(
        self,
        evolution_dim: int,
        model_dim: int,
        *,
        mode: str = "gated",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if evolution_dim <= 0 or model_dim <= 0:
            raise ValueError("Requires positive evolution_dim and model_dim")
        if mode not in {"gated", "film"}:
            raise ValueError(f"Unsupported mode '{mode}'. Use 'gated' or 'film'.")

        self.mode = str(mode)
        self.proj = nn.Sequential(
            nn.Linear(int(evolution_dim), int(model_dim)),
            nn.GELU(),
            nn.LayerNorm(int(model_dim)),
            nn.Dropout(float(dropout)),
        )

        if self.mode == "gated":
            self.gate = nn.Sequential(
                nn.Linear(int(model_dim) * 2, int(model_dim)),
                nn.GELU(),
                nn.Linear(int(model_dim), int(model_dim)),
                nn.Sigmoid(),
            )
        else:
            self.gamma = nn.Linear(int(model_dim), int(model_dim))
            self.beta = nn.Linear(int(model_dim), int(model_dim))

    def forward(
        self,
        lesion_embedding: Tensor,
        evolution_features: Tensor | None,
    ) -> tuple[Tensor, Tensor | None]:
        """Fuse lesion embedding with evolution features.

        Args:
            lesion_embedding: [B, D] sample/lesion embedding
            evolution_features: [B, E] WES features or None

        Returns:
            (fused_embedding, evolution_embedding)
        """
        if evolution_features is None:
            return lesion_embedding, None

        if lesion_embedding.ndim != 2 or evolution_features.ndim != 2:
            raise ValueError("Expected 2D tensors")
        if lesion_embedding.shape[0] != evolution_features.shape[0]:
            raise ValueError("Batch sizes must match")

        evo = self.proj(evolution_features)

        if self.mode == "gated":
            gate = self.gate(torch.cat([lesion_embedding, evo], dim=-1))
            fused = gate * lesion_embedding + (1.0 - gate) * evo
            return fused, evo

        gamma = self.gamma(evo)
        beta = self.beta(evo)
        return lesion_embedding * (1.0 + gamma) + beta, evo
