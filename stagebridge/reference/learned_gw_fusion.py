"""Learned Gromov-Wasserstein Fusion for HLCA-LuCA alignment.

The key insight: GW finds structure-preserving alignment between heterogeneous
spaces. We learn projections such that structure is preserved, then fuse per-cell.

Approach:
1. Learn projection heads for HLCA and LuCA to shared metric space
2. Add GW-style auxiliary loss: encourage distance structure preservation
3. Fuse per-cell via concat + fusion head (no cross-cell mixing)

The GW principle guides TRAINING (via structure loss), not INFERENCE.
This ensures fusion works identically regardless of batch size.

References:
- Peyré et al. (2016) "Gromov-Wasserstein Averaging"
- Bunne et al. (2019) "Learning Generative Models across Incomparable Spaces"
"""

from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class LearnedGWConfig:
    """Configuration for learned GW fusion."""
    hlca_dim: int = 30
    luca_dim: int = 10
    metric_dim: int = 32  # Shared metric space dimension
    output_dim: int = 40  # Fused output dimension

    # Architecture
    num_layers: int = 2
    dropout: float = 0.1

    # GW structure loss weight (auxiliary loss during training)
    structure_loss_weight: float = 0.1


class ProjectionHead(nn.Module):
    """Project to shared metric space."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        layers = []
        dim = input_dim
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(dim, output_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            dim = output_dim
        layers.append(nn.Linear(dim, output_dim))
        layers.append(nn.LayerNorm(output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class LearnedGWFusion(nn.Module):
    """Learned GW fusion with per-cell operation and structure-preserving loss.

    Key design:
    - Forward pass is PER-CELL (no batch dependence, works at batch_size=1)
    - GW principle is in the TRAINING LOSS, not inference
    - Structure loss encourages: if cells similar in HLCA, similar in LuCA projection

    Usage:
        fusion = LearnedGWFusion(config)

        # Forward (per-cell, any batch size)
        fused = fusion(hlca, luca)

        # Training: add structure loss
        main_loss = ...
        structure_loss = fusion.compute_structure_loss(hlca, luca)
        total_loss = main_loss + config.structure_loss_weight * structure_loss
    """

    def __init__(self, config: LearnedGWConfig):
        super().__init__()
        self.config = config

        # Project both spaces to shared metric space
        self.hlca_proj = ProjectionHead(
            config.hlca_dim,
            config.metric_dim,
            config.num_layers,
            config.dropout,
        )
        self.luca_proj = ProjectionHead(
            config.luca_dim,
            config.metric_dim,
            config.num_layers,
            config.dropout,
        )

        # Fusion head: combine projected representations
        self.fusion_head = nn.Sequential(
            nn.Linear(config.metric_dim * 2, config.output_dim),
            nn.GELU(),
            nn.Linear(config.output_dim, config.output_dim),
            nn.LayerNorm(config.output_dim),
        )

        # Learned interpolation weight
        self.alpha = nn.Parameter(torch.tensor(0.0))  # sigmoid(0) = 0.5

    def forward(
        self,
        hlca: Tensor,
        luca: Tensor,
        return_projections: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor, Tensor]:
        """Per-cell fusion of HLCA and LuCA.

        Args:
            hlca: [B, 30] HLCA embeddings
            luca: [B, 10] LuCA embeddings
            return_projections: Also return projected representations

        Returns:
            fused: [B, output_dim] fused representation
            h_proj, l_proj: [B, metric_dim] if return_projections
        """
        # Project to shared metric space
        h_proj = self.hlca_proj(hlca)  # [B, metric_dim]
        l_proj = self.luca_proj(luca)  # [B, metric_dim]

        # Learned weighted combination before fusion
        alpha = torch.sigmoid(self.alpha)
        combined = torch.cat([
            alpha * h_proj + (1 - alpha) * l_proj,  # Blended
            h_proj - l_proj,  # Difference (healthy vs cancer)
        ], dim=-1)

        # Fuse
        fused = self.fusion_head(combined)

        if return_projections:
            return fused, h_proj, l_proj
        return fused

    def compute_structure_loss(self, hlca: Tensor, luca: Tensor) -> Tensor:
        """GW-style structure preservation loss.

        Encourages: if two cells are close in HLCA space, their LuCA
        projections should also be close (and vice versa).

        This is the GW principle - structure should be preserved across spaces.

        Args:
            hlca: [B, 30] HLCA embeddings
            luca: [B, 10] LuCA embeddings

        Returns:
            Scalar structure loss
        """
        if hlca.shape[0] < 2:
            # Need at least 2 samples for pairwise distances
            return torch.tensor(0.0, device=hlca.device)

        # Project to metric space
        h_proj = self.hlca_proj(hlca)  # [B, metric_dim]
        l_proj = self.luca_proj(luca)  # [B, metric_dim]

        # Pairwise distances in projected spaces
        D_h = torch.cdist(h_proj, h_proj)  # [B, B]
        D_l = torch.cdist(l_proj, l_proj)  # [B, B]

        # Normalize distances to [0, 1] for stable loss
        D_h = D_h / (D_h.max() + 1e-8)
        D_l = D_l / (D_l.max() + 1e-8)

        # Structure loss: distance matrices should match
        # This is the GW objective in squared form
        structure_loss = F.mse_loss(D_h, D_l)

        return structure_loss

    def get_gw_loss(self, hlca: Tensor, luca: Tensor) -> Tensor:
        """Alias for compute_structure_loss (backward compatibility)."""
        return self.compute_structure_loss(hlca, luca)


# Backward compatibility
def create_learned_gw_fusion(config: LearnedGWConfig) -> LearnedGWFusion:
    """Factory function for learned GW fusion."""
    return LearnedGWFusion(config)
