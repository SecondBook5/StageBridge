"""Continuous distance transforms for sender-context attention.

Distance enters CCRT attention *only* as a continuous, monotonic transform of
the receiver<-sender distance. There is no discretization: no bins, no rings, no
radius buckets. A transform ``phi`` maps a non-negative distance to a
non-negative penalty magnitude that a (positive) learned coefficient later
scales inside the attention logits.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = [
    "DistanceTransformConfig",
    "ContinuousDistanceTransform",
    "validate_distance_tensor",
    "SUPPORTED_DISTANCE_TRANSFORMS",
]

#: The supported continuous transforms. All are monotonically non-decreasing in
#: distance so that a larger distance never *reduces* the penalty magnitude.
SUPPORTED_DISTANCE_TRANSFORMS = ("identity", "log1p", "sqrt")


@dataclass(frozen=True)
class DistanceTransformConfig:
    """Configuration for a continuous distance transform."""

    transform: str = "log1p"
    eps: float = 1e-8

    def __post_init__(self) -> None:
        if self.transform not in SUPPORTED_DISTANCE_TRANSFORMS:
            raise ValueError(
                f"unsupported distance transform '{self.transform}'; "
                f"supported: {SUPPORTED_DISTANCE_TRANSFORMS}"
            )
        if self.eps <= 0.0:
            raise ValueError(f"eps must be positive, got {self.eps}")


def validate_distance_tensor(distance_to_receiver: torch.Tensor) -> None:
    """Validate a distance tensor: floating, rank-2 ``[B, K]``, non-negative.

    Raises ``ValueError`` (negative distances are a hard failure by default).
    """
    if not isinstance(distance_to_receiver, torch.Tensor):
        raise TypeError(
            "distance_to_receiver must be a torch.Tensor, got "
            f"{type(distance_to_receiver).__name__}"
        )
    if not torch.is_floating_point(distance_to_receiver):
        raise ValueError("distance_to_receiver must be a floating-point tensor")
    if distance_to_receiver.dim() != 2:
        raise ValueError(
            "distance_to_receiver must be rank 2 [B, K], got shape "
            f"{tuple(distance_to_receiver.shape)}"
        )
    # Guard against NaNs before the comparison (NaN < 0 is False).
    if torch.isnan(distance_to_receiver).any():
        raise ValueError("distance_to_receiver contains NaN values")
    if bool((distance_to_receiver < 0).any()):
        raise ValueError("distance_to_receiver must be non-negative")


class ContinuousDistanceTransform:
    """Applies a continuous, monotonic distance transform ``phi(d)``.

    Not an ``nn.Module`` — it holds no parameters. It is a small stateless
    callable configured by :class:`DistanceTransformConfig`.
    """

    def __init__(self, config: DistanceTransformConfig | None = None) -> None:
        self.config = config or DistanceTransformConfig()

    def __call__(self, distance_to_receiver: torch.Tensor) -> torch.Tensor:
        validate_distance_tensor(distance_to_receiver)
        d = distance_to_receiver
        transform = self.config.transform
        if transform == "identity":
            out = d
        elif transform == "log1p":
            # d is already validated non-negative; clamp defends against tiny
            # negative round-off without changing valid values.
            out = torch.log1p(d.clamp_min(0.0))
        elif transform == "sqrt":
            out = torch.sqrt(d.clamp_min(0.0) + self.config.eps)
        else:  # pragma: no cover - guarded by config validation
            raise ValueError(f"unsupported distance transform '{transform}'")
        return out
