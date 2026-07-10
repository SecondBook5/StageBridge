"""Semantic transport loss.

Composes the native explicit coupling, its barycentric target, and up to three
loss terms into one differentiable objective for aligning predicted drift with
OT-derived semantic transitions:

* displacement: MSE(predicted_destination, barycentric_target)
* direction:    1 - cosine(predicted_displacement, target_displacement)
* distribution: Sinkhorn divergence(predicted_destination, target)  [native or geomloss]

where predicted_destination = prepared_source + delta_tau * predicted_drift.

The explicit coupling is intentionally fixed to the native backend (barycentric
targets require the transparent differentiable coupling). The distribution term
may use native or the optional GeomLoss backend, selected explicitly with no
fallback. No growth, no unbalanced OT, no operator import.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..contracts.errors import CCRTShapeError, CCRTValidationError
from ..representations.semantic import SemanticGeometryConfig, prepare_semantic_features
from .backends import DIVERGENCE_BACKENDS, NATIVE_BACKEND, compute_sinkhorn_divergence
from .barycentric import build_barycentric_transport_target
from .costs import build_transport_cost
from .geomloss_backend import GeomLossDivergenceConfig
from .native_sinkhorn import (
    SinkhornConfig,
    normalize_measure_weights,
    sinkhorn_coupling_native,
)

__all__ = [
    "SemanticTransportLossConfig",
    "SemanticTransportLossOutput",
    "SemanticTransportLoss",
]


@dataclass(frozen=True)
class SemanticTransportLossConfig:
    """Weights and options for the semantic transport loss."""

    delta_tau: float = 1.0
    displacement_weight: float = 1.0
    direction_weight: float = 0.0
    distribution_weight: float = 1.0
    distribution_backend: str = "native"
    eps: float = 1e-8

    def __post_init__(self) -> None:
        if self.delta_tau <= 0.0:
            raise ValueError("delta_tau must be > 0")
        weights = (
            self.displacement_weight,
            self.direction_weight,
            self.distribution_weight,
        )
        if any(w < 0 for w in weights):
            raise ValueError("all weights must be >= 0")
        if not any(w > 0 for w in weights):
            raise ValueError("at least one weight must be > 0")
        if self.distribution_backend not in DIVERGENCE_BACKENDS:
            raise ValueError(
                f"distribution_backend '{self.distribution_backend}' invalid; "
                f"allowed: {sorted(DIVERGENCE_BACKENDS)}"
            )
        if self.eps <= 0.0:
            raise ValueError("eps must be > 0")


@dataclass(frozen=True)
class SemanticTransportLossOutput:
    """All loss terms plus the transport signals that produced them."""

    total_loss: torch.Tensor
    displacement_loss: torch.Tensor
    direction_loss: torch.Tensor
    distribution_loss: torch.Tensor
    predicted_destination: torch.Tensor
    barycentric_target: torch.Tensor
    target_displacement: torch.Tensor
    coupling: torch.Tensor
    transport_cost: torch.Tensor
    marginal_error: torch.Tensor
    coupling_backend: str
    distribution_backend: str


class SemanticTransportLoss(nn.Module):
    """Differentiable semantic transport objective (native coupling)."""

    def __init__(
        self,
        *,
        geometry: SemanticGeometryConfig,
        native_sinkhorn: SinkhornConfig,
        loss: SemanticTransportLossConfig,
        geomloss: GeomLossDivergenceConfig | None = None,
    ) -> None:
        super().__init__()
        self.geometry = geometry
        self.native_sinkhorn = native_sinkhorn
        self.loss_config = loss
        self.geomloss_config = geomloss

    def _weighted_row_mean(
        self, per_row: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        # per_row: [N]; weights: [N] normalized to sum 1.
        return (per_row * weights).sum()

    def forward(
        self,
        *,
        source_semantic_features: torch.Tensor,
        target_semantic_features: torch.Tensor,
        predicted_drift: torch.Tensor,
        source_weights: torch.Tensor | None = None,
        target_weights: torch.Tensor | None = None,
    ) -> SemanticTransportLossOutput:
        cfg = self.loss_config

        # -- validation --
        for name, t in (
            ("source_semantic_features", source_semantic_features),
            ("target_semantic_features", target_semantic_features),
            ("predicted_drift", predicted_drift),
        ):
            if not isinstance(t, torch.Tensor) or not torch.is_floating_point(t):
                raise CCRTValidationError(f"{name} must be a floating tensor")
            if t.dim() != 2:
                raise CCRTShapeError(f"{name} must be rank 2 [., D]")
            if not bool(torch.isfinite(t).all()):
                raise CCRTValidationError(f"{name} contains non-finite values")

        n, d = source_semantic_features.shape
        if predicted_drift.shape != (n, d):
            raise CCRTShapeError(
                f"predicted_drift shape {tuple(predicted_drift.shape)} != "
                f"source shape {(n, d)}"
            )
        if target_semantic_features.shape[1] != d:
            raise CCRTShapeError(
                f"target dim {target_semantic_features.shape[1]} != source dim {d}"
            )

        # 1-2) prepare features + cost (native, canonical convention)
        cost_out = build_transport_cost(
            source_features=source_semantic_features,
            target_features=target_semantic_features,
            geometry=self.geometry,
        )
        prepared_source = cost_out.source_features
        prepared_target = cost_out.target_features

        # 3) native explicit coupling
        coupling_out = sinkhorn_coupling_native(
            cost_matrix=cost_out.cost_matrix,
            config=self.native_sinkhorn,
            source_weights=source_weights,
            target_weights=target_weights,
        )

        # 4) barycentric target from the native coupling
        bary = build_barycentric_transport_target(
            source_features=prepared_source,
            target_features=prepared_target,
            coupling=coupling_out.coupling,
            coupling_backend=NATIVE_BACKEND,
            eps=cfg.eps,
        )

        # 5) predicted destination
        predicted_destination = prepared_source + cfg.delta_tau * predicted_drift

        # normalized source weights for row means
        src_w = normalize_measure_weights(
            source_weights,
            size=n,
            dtype=prepared_source.dtype,
            device=prepared_source.device,
            name="source_weights",
        )

        # 6) displacement loss (source-weighted per-row MSE)
        per_row_sq = ((predicted_destination - bary.barycentric_target) ** 2).mean(dim=1)
        displacement_loss = self._weighted_row_mean(per_row_sq, src_w)

        # 7) direction loss (1 - cosine), zero where target displacement ~ 0
        predicted_displacement = cfg.delta_tau * predicted_drift
        target_displacement = bary.target_displacement
        cos = F.cosine_similarity(
            predicted_displacement, target_displacement, dim=-1, eps=cfg.eps
        )
        target_norm = target_displacement.norm(dim=-1)
        active = (target_norm > cfg.eps).to(cos.dtype)
        per_row_dir = (1.0 - cos) * active
        direction_loss = self._weighted_row_mean(per_row_dir, src_w)

        # 8) distribution loss (explicit backend, no fallback)
        div_out = compute_sinkhorn_divergence(
            backend=cfg.distribution_backend,
            source_features=predicted_destination,
            target_features=prepared_target,
            geometry=self.geometry,
            source_weights=source_weights,
            target_weights=target_weights,
            native_config=self.native_sinkhorn,
            geomloss_config=self.geomloss_config,
        )
        distribution_loss = div_out.divergence

        # 9) total
        total_loss = (
            cfg.displacement_weight * displacement_loss
            + cfg.direction_weight * direction_loss
            + cfg.distribution_weight * distribution_loss
        )

        return SemanticTransportLossOutput(
            total_loss=total_loss,
            displacement_loss=displacement_loss,
            direction_loss=direction_loss,
            distribution_loss=distribution_loss,
            predicted_destination=predicted_destination,
            barycentric_target=bary.barycentric_target,
            target_displacement=target_displacement,
            coupling=coupling_out.coupling,
            transport_cost=coupling_out.transport_cost,
            marginal_error=coupling_out.marginal_error,
            coupling_backend=NATIVE_BACKEND,
            distribution_backend=cfg.distribution_backend,
        )
