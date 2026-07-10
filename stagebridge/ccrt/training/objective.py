"""Composite CCRT training objective.

Runs the operator, computes the semantic transport loss on ``full_drift``, and
adds optional regularizers (attention entropy, sender-effect L1, regulatory L1,
residual drift/growth L2) and optional supervised growth loss. Every component is
returned individually — the objective never collapses CCRT to a single scalar and
never hides the decomposition. All weights default to zero except the semantic
weight.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..contracts.errors import CCRTShapeError, CCRTValidationError
from ..operators.model import (
    ContextResidualTransportOperator,
    ContextResidualTransportOutput,
)
from ..sender_context.sparsity import attention_entropy_loss, value_l1_sparsity_loss
from ..transport.semantic_loss import SemanticTransportLoss, SemanticTransportLossOutput
from .batch import CCRTTrainingBatch

__all__ = [
    "CompositeCCRTObjectiveConfig",
    "CompositeCCRTObjectiveOutput",
    "CompositeCCRTObjective",
]


@dataclass(frozen=True)
class CompositeCCRTObjectiveConfig:
    """Weights and options for the composite objective."""

    semantic_weight: float = 1.0
    attention_entropy_weight: float = 0.0
    sender_effect_l1_weight: float = 0.0
    regulatory_l1_weight: float = 0.0
    residual_drift_l2_weight: float = 0.0
    residual_growth_l2_weight: float = 0.0
    growth_supervision_weight: float = 0.0
    growth_loss: str = "mse"
    growth_huber_delta: float = 1.0

    def __post_init__(self) -> None:
        weights = (
            self.semantic_weight,
            self.attention_entropy_weight,
            self.sender_effect_l1_weight,
            self.regulatory_l1_weight,
            self.residual_drift_l2_weight,
            self.residual_growth_l2_weight,
            self.growth_supervision_weight,
        )
        if any(w < 0 for w in weights):
            raise ValueError("all weights must be >= 0")
        if self.semantic_weight <= 0:
            raise ValueError("semantic_weight must be > 0 for Milestone 6")
        if self.growth_loss not in ("mse", "huber"):
            raise ValueError("growth_loss must be 'mse' or 'huber'")
        if self.growth_huber_delta <= 0:
            raise ValueError("growth_huber_delta must be > 0")


@dataclass(frozen=True)
class CompositeCCRTObjectiveOutput:
    """Every loss component plus the full model and semantic outputs."""

    total_loss: torch.Tensor
    semantic_loss: torch.Tensor
    attention_entropy_loss: torch.Tensor
    sender_effect_l1_loss: torch.Tensor
    regulatory_l1_loss: torch.Tensor
    residual_drift_l2_loss: torch.Tensor
    residual_growth_l2_loss: torch.Tensor
    growth_supervision_loss: torch.Tensor
    model_output: ContextResidualTransportOutput
    semantic_output: SemanticTransportLossOutput


class CompositeCCRTObjective(nn.Module):
    """Composite CCRT objective as an nn.Module."""

    def __init__(
        self,
        *,
        semantic_transport_loss: SemanticTransportLoss,
        config: CompositeCCRTObjectiveConfig,
    ) -> None:
        super().__init__()
        self.semantic_transport_loss = semantic_transport_loss
        self.config = config

    def _zero(self, ref: torch.Tensor) -> torch.Tensor:
        return torch.zeros((), dtype=ref.dtype, device=ref.device)

    def forward(
        self,
        *,
        model: ContextResidualTransportOperator,
        batch: CCRTTrainingBatch,
    ) -> CompositeCCRTObjectiveOutput:
        cfg = self.config
        batch.validate()

        # 1) run the operator
        model_output = model(
            receiver_features=batch.receiver_features,
            sender_features=batch.sender_features,
            sender_mask=batch.sender_mask,
            distance_to_receiver=batch.distance_to_receiver,
            sender_context_type_ids=batch.sender_context_type_ids,
            transition_edge_index=batch.transition_edge_index,
            uncertainty=batch.uncertainty,
        )

        # 2) drift dimension must match the semantic space
        if model_output.full_drift.shape != batch.source_semantic_features.shape:
            raise CCRTShapeError(
                f"full_drift shape {tuple(model_output.full_drift.shape)} != "
                f"source_semantic_features shape "
                f"{tuple(batch.source_semantic_features.shape)}"
            )

        # 3) semantic transport loss on full_drift
        semantic_output = self.semantic_transport_loss(
            source_semantic_features=batch.source_semantic_features,
            target_semantic_features=batch.target_semantic_features,
            predicted_drift=model_output.full_drift,
            source_weights=batch.source_weights,
            target_weights=batch.target_weights,
        )
        semantic_loss = semantic_output.total_loss

        # reference tensor for zero-valued inactive components
        ref = semantic_loss

        # 4) attention entropy (empty sender is a valid attention option)
        if cfg.attention_entropy_weight > 0:
            attention_entropy = attention_entropy_loss(
                model_output.attention_weights,
                sender_mask=model_output.sender_mask_with_empty,
            )
        else:
            attention_entropy = self._zero(ref)

        # 5) sender-effect L1
        if cfg.sender_effect_l1_weight > 0:
            sender_effect_l1 = value_l1_sparsity_loss(
                model_output.sender_effects,
                sender_mask=model_output.sender_mask_with_empty,
            )
        else:
            sender_effect_l1 = self._zero(ref)

        # 6) regulatory L1
        if cfg.regulatory_l1_weight > 0:
            regulatory_l1 = model_output.regulatory_state.abs().mean()
        else:
            regulatory_l1 = self._zero(ref)

        # 7) residual drift L2
        if cfg.residual_drift_l2_weight > 0:
            residual_drift_l2 = (model_output.residual_drift ** 2).mean()
        else:
            residual_drift_l2 = self._zero(ref)

        # 8) residual growth L2
        if cfg.residual_growth_l2_weight > 0:
            residual_growth_l2 = (model_output.residual_growth ** 2).mean()
        else:
            residual_growth_l2 = self._zero(ref)

        # 9) optional growth supervision
        if cfg.growth_supervision_weight > 0:
            growth_supervision = self._growth_supervision_loss(model_output, batch)
        else:
            growth_supervision = self._zero(ref)

        total_loss = (
            cfg.semantic_weight * semantic_loss
            + cfg.attention_entropy_weight * attention_entropy
            + cfg.sender_effect_l1_weight * sender_effect_l1
            + cfg.regulatory_l1_weight * regulatory_l1
            + cfg.residual_drift_l2_weight * residual_drift_l2
            + cfg.residual_growth_l2_weight * residual_growth_l2
            + cfg.growth_supervision_weight * growth_supervision
        )

        return CompositeCCRTObjectiveOutput(
            total_loss=total_loss,
            semantic_loss=semantic_loss,
            attention_entropy_loss=attention_entropy,
            sender_effect_l1_loss=sender_effect_l1,
            regulatory_l1_loss=regulatory_l1,
            residual_drift_l2_loss=residual_drift_l2,
            residual_growth_l2_loss=residual_growth_l2,
            growth_supervision_loss=growth_supervision,
            model_output=model_output,
            semantic_output=semantic_output,
        )

    def _growth_supervision_loss(
        self,
        model_output: ContextResidualTransportOutput,
        batch: CCRTTrainingBatch,
    ) -> torch.Tensor:
        cfg = self.config
        if batch.growth_targets is None:
            raise CCRTValidationError(
                "growth_supervision_weight > 0 requires batch.growth_targets"
            )
        predicted = model_output.full_growth
        target = batch.growth_targets
        if target.shape[-1] != predicted.shape[-1]:
            raise CCRTShapeError(
                f"growth_targets last dim {target.shape[-1]} != full_growth last "
                f"dim {predicted.shape[-1]}"
            )
        if target.shape[0] != predicted.shape[0]:
            raise CCRTShapeError("growth_targets batch dim != full_growth batch dim")

        if cfg.growth_loss == "mse":
            per_elem = (predicted - target) ** 2
        else:  # huber
            per_elem = F.huber_loss(
                predicted, target, delta=cfg.growth_huber_delta, reduction="none"
            )

        if batch.growth_mask is not None:
            mask = batch.growth_mask.to(per_elem.dtype)
            denom = mask.sum()
            if float(denom) <= 0:
                raise CCRTValidationError(
                    "growth_mask has no valid (nonzero) entries"
                )
            return (per_elem * mask).sum() / denom
        return per_elem.mean()
