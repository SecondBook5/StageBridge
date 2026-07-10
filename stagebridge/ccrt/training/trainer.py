"""Reproducible CCRT trainer.

A generic, system-agnostic trainer that wires the operator + composite objective
+ optimizer into deterministic train/eval steps and an epoch loop. It performs no
shuffling, no early stopping, no automatic checkpointing, and knows no biology —
patient/donor-aware ordering is the caller's responsibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from ..contracts.errors import CCRTValidationError
from ..operators.model import ContextResidualTransportOperator
from .batch import CCRTTrainingBatch
from .objective import CompositeCCRTObjective, CompositeCCRTObjectiveOutput
from .reproducibility import set_reproducible_seed

__all__ = [
    "TrainerConfig",
    "TrainingStepMetrics",
    "EpochMetrics",
    "CCRTTrainer",
]

_DTYPES = {"float32": torch.float32, "float64": torch.float64}


@dataclass(frozen=True)
class TrainerConfig:
    """Trainer configuration."""

    epochs: int = 1
    device: str = "cpu"
    dtype: str = "float32"
    gradient_clip_norm: float | None = 1.0
    seed: int = 0
    deterministic_algorithms: bool = True
    require_finite_loss: bool = True

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be > 0")
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device must be a non-empty string")
        if self.dtype not in _DTYPES:
            raise ValueError("dtype must be 'float32' or 'float64'")
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be None or > 0")
        if self.seed < 0:
            raise ValueError("seed must be >= 0")

    @property
    def torch_dtype(self) -> torch.dtype:
        return _DTYPES[self.dtype]


@dataclass(frozen=True)
class TrainingStepMetrics:
    """Detached scalar metrics for a single step."""

    total_loss: float
    semantic_loss: float
    attention_entropy_loss: float
    sender_effect_l1_loss: float
    regulatory_l1_loss: float
    residual_drift_l2_loss: float
    residual_growth_l2_loss: float
    growth_supervision_loss: float
    gradient_norm: float | None
    learning_rate: float


@dataclass(frozen=True)
class EpochMetrics:
    """Aggregated metrics for one epoch."""

    epoch: int
    train: Mapping[str, float]
    validation: Mapping[str, float] | None


_METRIC_FIELDS = (
    "total_loss",
    "semantic_loss",
    "attention_entropy_loss",
    "sender_effect_l1_loss",
    "regulatory_l1_loss",
    "residual_drift_l2_loss",
    "residual_growth_l2_loss",
    "growth_supervision_loss",
)


class CCRTTrainer:
    """Deterministic trainer over pre-ordered batches."""

    def __init__(
        self,
        *,
        model: ContextResidualTransportOperator,
        objective: CompositeCCRTObjective,
        optimizer: torch.optim.Optimizer,
        config: TrainerConfig,
        scheduler: Any | None = None,
    ) -> None:
        self.config = config
        self.reproducibility = set_reproducible_seed(
            config.seed, deterministic_algorithms=config.deterministic_algorithms
        )
        self.device = torch.device(config.device)
        self.dtype = config.torch_dtype

        self.model = model.to(device=self.device, dtype=self.dtype)
        self.objective = objective.to(device=self.device, dtype=self.dtype)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.global_step = 0

    # -- helpers --

    def _current_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def _metrics_from_output(
        self, out: CompositeCCRTObjectiveOutput, *, gradient_norm: float | None
    ) -> TrainingStepMetrics:
        return TrainingStepMetrics(
            total_loss=float(out.total_loss.detach()),
            semantic_loss=float(out.semantic_loss.detach()),
            attention_entropy_loss=float(out.attention_entropy_loss.detach()),
            sender_effect_l1_loss=float(out.sender_effect_l1_loss.detach()),
            regulatory_l1_loss=float(out.regulatory_l1_loss.detach()),
            residual_drift_l2_loss=float(out.residual_drift_l2_loss.detach()),
            residual_growth_l2_loss=float(out.residual_growth_l2_loss.detach()),
            growth_supervision_loss=float(out.growth_supervision_loss.detach()),
            gradient_norm=gradient_norm,
            learning_rate=self._current_lr(),
        )

    # -- steps --

    def train_step(self, batch: CCRTTrainingBatch) -> TrainingStepMetrics:
        self.model.train()
        self.objective.train()
        batch = batch.to(self.device, dtype=self.dtype)

        self.optimizer.zero_grad(set_to_none=True)
        out = self.objective(model=self.model, batch=batch)

        if self.config.require_finite_loss and not bool(
            torch.isfinite(out.total_loss)
        ):
            raise CCRTValidationError(
                f"non-finite total loss encountered: {out.total_loss}"
            )

        out.total_loss.backward()

        params = [p for p in self.model.parameters() if p.grad is not None]
        if self.config.gradient_clip_norm is not None:
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    params, self.config.gradient_clip_norm
                )
            )
        else:
            grad_norm = float(
                torch.norm(
                    torch.stack([p.grad.detach().norm() for p in params])
                )
            ) if params else 0.0

        self.optimizer.step()
        self.global_step += 1
        return self._metrics_from_output(out, gradient_norm=grad_norm)

    def evaluate_step(self, batch: CCRTTrainingBatch) -> TrainingStepMetrics:
        self.model.eval()
        self.objective.eval()
        batch = batch.to(self.device, dtype=self.dtype)
        with torch.no_grad():
            out = self.objective(model=self.model, batch=batch)
            if self.config.require_finite_loss and not bool(
                torch.isfinite(out.total_loss)
            ):
                raise CCRTValidationError(
                    f"non-finite total loss encountered: {out.total_loss}"
                )
        return self._metrics_from_output(out, gradient_norm=None)

    # -- epoch loop --

    def _aggregate(
        self, step_metrics: Sequence[TrainingStepMetrics]
    ) -> dict[str, float]:
        n = len(step_metrics)
        agg = {
            field: sum(getattr(m, field) for m in step_metrics) / n
            for field in _METRIC_FIELDS
        }
        agg["learning_rate"] = sum(m.learning_rate for m in step_metrics) / n
        grad_norms = [m.gradient_norm for m in step_metrics if m.gradient_norm is not None]
        if grad_norms:
            agg["gradient_norm"] = sum(grad_norms) / len(grad_norms)
        return agg

    def fit(
        self,
        *,
        train_batches: Sequence[CCRTTrainingBatch],
        validation_batches: Sequence[CCRTTrainingBatch] | None = None,
        epochs: int | None = None,
    ) -> tuple[EpochMetrics, ...]:
        if not train_batches:
            raise CCRTValidationError("train_batches must be non-empty")
        if validation_batches is not None and not validation_batches:
            raise CCRTValidationError(
                "validation_batches must be non-empty when provided"
            )
        n_epochs = self.config.epochs if epochs is None else epochs
        if n_epochs <= 0:
            raise CCRTValidationError("epochs must be > 0")

        history: list[EpochMetrics] = []
        for epoch in range(n_epochs):
            train_steps = [self.train_step(b) for b in train_batches]
            train_agg = self._aggregate(train_steps)

            val_agg: dict[str, float] | None = None
            if validation_batches is not None:
                val_steps = [self.evaluate_step(b) for b in validation_batches]
                val_agg = self._aggregate(val_steps)

            if self.scheduler is not None:
                self.scheduler.step()

            history.append(
                EpochMetrics(epoch=epoch, train=train_agg, validation=val_agg)
            )

        return tuple(history)
