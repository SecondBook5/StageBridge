"""Auxiliary prediction heads for StageBridge.

These provide biological regularization during training:
- PathwayHead: PROGENy pathway activity prediction (14 + cGAS-STING)
- ProliferationHead: Ki67 proliferation classification

DESIGN DECISION (see memory/design_conditioning_vs_encoding.md):
Only these two heads are used as prediction TARGETS because they are
truly orthogonal to our scientific claims about niche-progression.

Other biological signals (cell cycle, clonal, CAF, plasticity, LIANA)
should be INPUTS (in stats token) for conditioning, NOT targets.
This prevents circular validation while preserving ability to identify
rare cell types and control for confounders.

Literature precedent:
- PathwayHead inspired by SpatialFusion (pathway-aware latent structure)
- ProliferationHead inspired by OSDR (tissue dynamics signal)

NOTE: IL1BHead, KACHead, CellCycleHead, SenescenceHead, CAFHead,
PlasticityHead, and LIANAHead were explicitly REMOVED to prevent
circular validation (predicting signals that overlap with our hypothesis).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class AuxiliaryHead(nn.Module, ABC):
    """Abstract base class for auxiliary prediction heads.

    All auxiliary heads share:
    - Input: context embedding from niche encoder
    - Output: prediction tensor for auxiliary supervision
    - Loss: computed externally (MSE, BCE, etc.)
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [B, input_dim] context embedding

        Returns:
            [B, output_dim] predictions
        """
        pass


class PathwayHead(AuxiliaryHead):
    """Predict PROGENy pathway scores from context.

    Encourages pathway-aware latent structure by predicting
    14 canonical PROGENy pathways + optional cGAS-STING.

    Args:
        input_dim: Input feature dimension
        n_pathways: Number of pathways (14=PROGENy, 15=extended with cGAS-STING)
        include_cgas_sting: Add cGAS-STING pathway (sets n_pathways=15)
    """

    def __init__(
        self,
        input_dim: int,
        n_pathways: int = 14,
        include_cgas_sting: bool = False,
    ):
        if include_cgas_sting:
            n_pathways = 15
        super().__init__(input_dim=input_dim, output_dim=n_pathways)
        self.head = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_pathways),
        )
        self.include_cgas_sting = include_cgas_sting

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class ProliferationHead(AuxiliaryHead):
    """Predict Ki67 proliferation from context.

    Anchors model to broad tissue-dynamics signal.
    Single output for binary proliferation classification.

    Args:
        input_dim: Input feature dimension
    """

    def __init__(self, input_dim: int):
        super().__init__(input_dim=input_dim, output_dim=1)
        self.head = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)
