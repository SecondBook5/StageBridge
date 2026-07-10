"""Configuration for the synthetic CCRT benchmark.

Two frozen configs: ``SyntheticSystemConfig`` (the toy biological system + data
generation knobs) and ``SyntheticBenchmarkConfig`` (the student training setup).
Both validate strictly and never silently coerce values.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["SyntheticSystemConfig", "SyntheticBenchmarkConfig"]


@dataclass(frozen=True)
class SyntheticSystemConfig:
    """The synthetic system geometry and data-generation configuration."""

    biological_system_id: str = "synthetic_ccrt_system"
    seed: int = 0
    receiver_dim: int = 4
    sender_dim: int = 4
    context_dim: int = 3
    semantic_dim: int = 3
    regulatory_dim: int = 2
    growth_dim: int = 1
    num_sender_context_types: int = 3
    num_transition_edges: int = 2
    senders_per_receiver: int = 4
    batch_size: int = 8
    train_batches: int = 6
    validation_batches: int = 2
    test_batches: int = 2
    max_distance: float = 3.0
    delta_tau: float = 1.0
    source_semantic_scale: float = 2.0
    self_drift_strength: float = 0.15
    self_growth_strength: float = 0.10
    context_strength: float = 0.30
    target_noise_std: float = 0.01
    growth_noise_std: float = 0.01
    sender_mask_probability: float = 0.10
    include_null_context_training_pairs: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.biological_system_id, str) or not self.biological_system_id.strip():
            raise ValueError("biological_system_id must be non-empty")
        if self.seed < 0:
            raise ValueError("seed must be >= 0")
        for name in (
            "receiver_dim", "sender_dim", "context_dim", "semantic_dim",
            "regulatory_dim", "growth_dim",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0")
        if self.num_sender_context_types < 3:
            raise ValueError("num_sender_context_types must be >= 3")
        if self.num_transition_edges < 2:
            raise ValueError("num_transition_edges must be >= 2")
        if self.senders_per_receiver <= 0:
            raise ValueError("senders_per_receiver must be > 0")
        if self.batch_size <= 1:
            raise ValueError("batch_size must be > 1")
        for name in ("train_batches", "validation_batches", "test_batches"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0")
        if self.max_distance <= 0:
            raise ValueError("max_distance must be > 0")
        if self.delta_tau <= 0:
            raise ValueError("delta_tau must be > 0")
        if self.source_semantic_scale <= 0:
            raise ValueError("source_semantic_scale must be > 0")
        for name in ("self_drift_strength", "self_growth_strength", "context_strength"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        for name in ("target_noise_std", "growth_noise_std"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        if not (0.0 <= self.sender_mask_probability < 1.0):
            raise ValueError("sender_mask_probability must be in [0, 1)")


@dataclass(frozen=True)
class SyntheticBenchmarkConfig:
    """The student training / benchmark configuration."""

    seed: int = 0
    epochs: int = 20
    learning_rate: float = 3e-3
    weight_decay: float = 1e-5
    hidden_dim: int = 16
    num_heads: int = 2
    attention_entropy_weight: float = 1e-4
    sender_effect_l1_weight: float = 1e-4
    regulatory_l1_weight: float = 1e-4
    residual_drift_l2_weight: float = 1e-3
    residual_growth_l2_weight: float = 1e-3
    growth_supervision_weight: float = 1.0
    displacement_weight: float = 1.0
    direction_weight: float = 0.10
    distribution_weight: float = 0.10
    sinkhorn_epsilon: float = 0.20
    sinkhorn_iterations: int = 80
    gradient_clip_norm: float = 5.0
    device: str = "cpu"
    dtype: str = "float32"

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be >= 0")
        if self.epochs <= 0:
            raise ValueError("epochs must be > 0")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be >= 0")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be > 0")
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        for name in (
            "attention_entropy_weight", "sender_effect_l1_weight",
            "regulatory_l1_weight", "residual_drift_l2_weight",
            "residual_growth_l2_weight", "displacement_weight",
            "direction_weight", "distribution_weight",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        if self.growth_supervision_weight <= 0:
            raise ValueError("growth_supervision_weight must be > 0 for the benchmark")
        if self.sinkhorn_epsilon <= 0:
            raise ValueError("sinkhorn_epsilon must be > 0")
        if self.sinkhorn_iterations <= 0:
            raise ValueError("sinkhorn_iterations must be > 0")
        if self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be > 0")
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device must be non-empty")
        if self.dtype not in ("float32", "float64"):
            raise ValueError("dtype must be 'float32' or 'float64'")
