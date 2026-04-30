"""Dual-reference fusion methods.

Combines HLCA (30d) and LuCA (10d) embeddings into fused representations.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

import numpy as np
import torch
from torch import Tensor, nn

from stagebridge.contracts import HLCA_DIM, LUCA_DIM, LATENT_DIM


class FusionMethod(str, Enum):
    """Available fusion methods for dual-reference embeddings."""

    CONCAT = "concat"
    WEIGHTED = "weighted"
    GATED = "gated"
    FILM = "film"
    GROMOV_WASSERSTEIN = "gromov_wasserstein"


def fuse_embeddings(
    hlca: np.ndarray,
    luca: np.ndarray,
    method: FusionMethod | str = FusionMethod.CONCAT,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Fuse HLCA and LuCA embeddings.

    Args:
        hlca: [N, 30] HLCA embeddings
        luca: [N, 10] LuCA embeddings
        method: Fusion method (concat, weighted, gated, film)
        weights: Optional weights for weighted fusion

    Returns:
        Fused embeddings (shape depends on method)
    """
    method = FusionMethod(method)

    if method == FusionMethod.CONCAT:
        return concat_fusion(hlca, luca)
    elif method == FusionMethod.WEIGHTED:
        return weighted_fusion(hlca, luca, weights)
    elif method == FusionMethod.GATED:
        return gated_fusion(hlca, luca)
    elif method == FusionMethod.FILM:
        return film_fusion(hlca, luca)
    else:
        raise ValueError(f"Unknown fusion method: {method}")


def concat_fusion(hlca: np.ndarray, luca: np.ndarray) -> np.ndarray:
    """Simple concatenation: [HLCA; LuCA] -> 40d.

    This is the default fusion method. Simple, interpretable, and
    preserves all information from both references.

    Args:
        hlca: [N, 30] HLCA embeddings
        luca: [N, 10] LuCA embeddings

    Returns:
        [N, 40] concatenated embeddings
    """
    hlca = np.asarray(hlca, dtype=np.float32)
    luca = np.asarray(luca, dtype=np.float32)

    if hlca.shape[1] != HLCA_DIM:
        raise ValueError(f"HLCA should be {HLCA_DIM}d, got {hlca.shape[1]}d")
    if luca.shape[1] != LUCA_DIM:
        raise ValueError(f"LuCA should be {LUCA_DIM}d, got {luca.shape[1]}d")

    return np.concatenate([hlca, luca], axis=1)


def weighted_fusion(
    hlca: np.ndarray,
    luca: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Weighted combination in shared space.

    Projects LuCA to HLCA dimension then combines:
        fused = w * hlca + (1-w) * project(luca)

    Args:
        hlca: [N, 30] HLCA embeddings
        luca: [N, 10] LuCA embeddings
        weights: [N] or scalar weight for HLCA (0=luca, 1=hlca)

    Returns:
        [N, 30] weighted fused embeddings
    """
    hlca = np.asarray(hlca, dtype=np.float32)
    luca = np.asarray(luca, dtype=np.float32)

    luca_proj = np.zeros((luca.shape[0], HLCA_DIM), dtype=np.float32)
    luca_proj[:, :LUCA_DIM] = luca

    if weights is None:
        weights = 0.5
    weights = np.asarray(weights, dtype=np.float32)
    if weights.ndim == 0:
        weights = np.full(hlca.shape[0], weights)
    weights = weights.reshape(-1, 1)

    return weights * hlca + (1 - weights) * luca_proj


def gated_fusion(hlca: np.ndarray, luca: np.ndarray) -> np.ndarray:
    """Gated fusion with learned/heuristic gating.

    For numpy arrays, uses a simple heuristic gate based on
    relative magnitudes. For learned gating, use GatedFusion module.

    Args:
        hlca: [N, 30] HLCA embeddings
        luca: [N, 10] LuCA embeddings

    Returns:
        [N, 30] gated fused embeddings
    """
    hlca = np.asarray(hlca, dtype=np.float32)
    luca = np.asarray(luca, dtype=np.float32)

    luca_proj = np.zeros((luca.shape[0], HLCA_DIM), dtype=np.float32)
    luca_proj[:, :LUCA_DIM] = luca

    hlca_norm = np.linalg.norm(hlca, axis=1, keepdims=True) + 1e-6
    luca_norm = np.linalg.norm(luca_proj, axis=1, keepdims=True) + 1e-6
    gate = hlca_norm / (hlca_norm + luca_norm)

    return gate * hlca + (1 - gate) * luca_proj


def film_fusion(hlca: np.ndarray, luca: np.ndarray) -> np.ndarray:
    """FiLM-style modulation: LuCA modulates HLCA.

    fused = gamma * hlca + beta
    where gamma, beta are derived from LuCA.

    For numpy, uses simple linear projection. For learned FiLM,
    use FiLMFusion module.

    Args:
        hlca: [N, 30] HLCA embeddings
        luca: [N, 10] LuCA embeddings

    Returns:
        [N, 30] FiLM-modulated embeddings
    """
    hlca = np.asarray(hlca, dtype=np.float32)
    luca = np.asarray(luca, dtype=np.float32)

    luca_expanded = np.concatenate([
        luca,
        np.tile(luca[:, :HLCA_DIM - LUCA_DIM], (1, 1)) if HLCA_DIM > LUCA_DIM else np.empty((luca.shape[0], 0))
    ], axis=1)

    if luca_expanded.shape[1] < HLCA_DIM:
        pad = np.zeros((luca.shape[0], HLCA_DIM - luca_expanded.shape[1]), dtype=np.float32)
        luca_expanded = np.concatenate([luca_expanded, pad], axis=1)

    gamma = 1.0 + 0.1 * luca_expanded[:, :HLCA_DIM]
    beta = 0.1 * luca_expanded[:, :HLCA_DIM]

    return gamma * hlca + beta


class GatedFusion(nn.Module):
    """Learned gated fusion module.

    Learns to combine HLCA and LuCA based on content.
    """

    def __init__(
        self,
        hlca_dim: int = HLCA_DIM,
        luca_dim: int = LUCA_DIM,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.hlca_dim = hlca_dim
        self.luca_dim = luca_dim

        self.luca_proj = nn.Linear(luca_dim, hlca_dim)

        self.gate_net = nn.Sequential(
            nn.Linear(hlca_dim + hlca_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hlca_dim),
            nn.Sigmoid(),
        )

    def forward(self, hlca: Tensor, luca: Tensor) -> Tensor:
        """Forward pass.

        Args:
            hlca: [B, 30] HLCA embeddings
            luca: [B, 10] LuCA embeddings

        Returns:
            [B, 30] gated fused embeddings
        """
        luca_proj = self.luca_proj(luca)

        gate_input = torch.cat([hlca, luca_proj], dim=-1)
        gate = self.gate_net(gate_input)

        return gate * hlca + (1 - gate) * luca_proj


class FiLMFusion(nn.Module):
    """Learned FiLM fusion module.

    LuCA features modulate HLCA via learned affine transformation.
    """

    def __init__(
        self,
        hlca_dim: int = HLCA_DIM,
        luca_dim: int = LUCA_DIM,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.hlca_dim = hlca_dim
        self.luca_dim = luca_dim

        self.gamma_net = nn.Sequential(
            nn.Linear(luca_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hlca_dim),
        )

        self.beta_net = nn.Sequential(
            nn.Linear(luca_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hlca_dim),
        )

    def forward(self, hlca: Tensor, luca: Tensor) -> Tensor:
        """Forward pass.

        Args:
            hlca: [B, 30] HLCA embeddings
            luca: [B, 10] LuCA embeddings

        Returns:
            [B, 30] FiLM-modulated embeddings
        """
        gamma = 1.0 + self.gamma_net(luca)
        beta = self.beta_net(luca)

        return gamma * hlca + beta


class ConcatFusion(nn.Module):
    """Simple concatenation fusion (as a module for consistency)."""

    def __init__(
        self,
        hlca_dim: int = HLCA_DIM,
        luca_dim: int = LUCA_DIM,
    ):
        super().__init__()
        self.hlca_dim = hlca_dim
        self.luca_dim = luca_dim
        self.output_dim = hlca_dim + luca_dim

    def forward(self, hlca: Tensor, luca: Tensor) -> Tensor:
        """Forward pass.

        Args:
            hlca: [B, 30] HLCA embeddings
            luca: [B, 10] LuCA embeddings

        Returns:
            [B, 40] concatenated embeddings
        """
        return torch.cat([hlca, luca], dim=-1)


def get_fusion_module(
    method: FusionMethod | str,
    hlca_dim: int = HLCA_DIM,
    luca_dim: int = LUCA_DIM,
    **kwargs,
) -> nn.Module:
    """Get a fusion module by method name.

    Args:
        method: Fusion method
        hlca_dim: HLCA dimension
        luca_dim: LuCA dimension
        **kwargs: Additional arguments for the module

    Returns:
        Fusion module
    """
    method = FusionMethod(method)

    if method == FusionMethod.CONCAT:
        return ConcatFusion(hlca_dim, luca_dim)
    elif method == FusionMethod.GATED:
        return GatedFusion(hlca_dim, luca_dim, **kwargs)
    elif method == FusionMethod.FILM:
        return FiLMFusion(hlca_dim, luca_dim, **kwargs)
    elif method == FusionMethod.GROMOV_WASSERSTEIN:
        from stagebridge.reference.gw_fusion import GromovWassersteinFusion, GWFusionConfig
        output_dim = kwargs.pop('output_dim', hlca_dim + luca_dim)
        config = GWFusionConfig(
            hlca_dim=hlca_dim,
            luca_dim=luca_dim,
            output_dim=output_dim,
            **kwargs,
        )
        return GromovWassersteinFusion(config)
    else:
        raise ValueError(f"No module for fusion method: {method}")
