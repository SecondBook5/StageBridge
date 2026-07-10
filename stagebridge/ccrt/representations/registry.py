"""Feature-space registry.

Every semantic / reconstruction / regulatory space CCRT uses must be *registered*
here with an explicit role, dimension, metric, and normalization. This is the
guard that prevents an arbitrary latent matrix from silently becoming transport
geometry: a tensor is only accepted as a given feature space if its shape and
dtype match a registered ``FeatureSpaceSpec``.

System-agnostic: registry ids and feature ids are opaque strings (never
lowercased or altered); no biological vocabulary lives here.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..contracts.errors import CCRTShapeError, CCRTValidationError

__all__ = [
    "ALLOWED_REPRESENTATION_ROLES",
    "ALLOWED_SEMANTIC_METRICS",
    "ALLOWED_NORMALIZATIONS",
    "FeatureSpaceSpec",
    "FeatureSpaceRegistry",
]

ALLOWED_REPRESENTATION_ROLES = frozenset({"semantic", "reconstruction", "regulatory"})
ALLOWED_SEMANTIC_METRICS = frozenset({"squared_euclidean", "cosine"})
ALLOWED_NORMALIZATIONS = frozenset({"none", "l2"})


@dataclass(frozen=True)
class FeatureSpaceSpec:
    """Declares one registered feature space."""

    feature_space_id: str
    role: str
    dimension: int
    feature_ids: tuple[str, ...] = ()
    metric: str | None = None
    normalization: str = "none"
    version: str = "1"
    description: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.feature_space_id, str) or not self.feature_space_id.strip():
            raise CCRTValidationError(
                "FeatureSpaceSpec.feature_space_id must be a non-empty string"
            )
        if self.role not in ALLOWED_REPRESENTATION_ROLES:
            raise CCRTValidationError(
                f"FeatureSpaceSpec.role '{self.role}' invalid; "
                f"allowed: {sorted(ALLOWED_REPRESENTATION_ROLES)}"
            )
        if not isinstance(self.dimension, int) or self.dimension <= 0:
            raise CCRTValidationError("FeatureSpaceSpec.dimension must be an int > 0")
        if self.normalization not in ALLOWED_NORMALIZATIONS:
            raise CCRTValidationError(
                f"FeatureSpaceSpec.normalization '{self.normalization}' invalid; "
                f"allowed: {sorted(ALLOWED_NORMALIZATIONS)}"
            )
        if not isinstance(self.version, str) or not self.version.strip():
            raise CCRTValidationError("FeatureSpaceSpec.version must be non-empty")

        # feature ids (coerce to tuple; validate when supplied)
        object.__setattr__(self, "feature_ids", tuple(self.feature_ids))
        if self.feature_ids:
            for fid in self.feature_ids:
                if not isinstance(fid, str) or not fid.strip():
                    raise CCRTValidationError(
                        "FeatureSpaceSpec.feature_ids entries must be non-empty strings"
                    )
            if len(set(self.feature_ids)) != len(self.feature_ids):
                raise CCRTValidationError(
                    "FeatureSpaceSpec.feature_ids must be unique"
                )
            if len(self.feature_ids) != self.dimension:
                raise CCRTValidationError(
                    f"FeatureSpaceSpec.feature_ids length {len(self.feature_ids)} "
                    f"!= dimension {self.dimension}"
                )

        # metric rules depend on role
        if self.role == "semantic":
            if self.metric is None:
                raise CCRTValidationError(
                    "FeatureSpaceSpec.metric is required for role 'semantic'"
                )
            if self.metric not in ALLOWED_SEMANTIC_METRICS:
                raise CCRTValidationError(
                    f"FeatureSpaceSpec.metric '{self.metric}' invalid; "
                    f"allowed: {sorted(ALLOWED_SEMANTIC_METRICS)}"
                )
        else:
            if self.metric is not None and self.metric not in ALLOWED_SEMANTIC_METRICS:
                raise CCRTValidationError(
                    f"FeatureSpaceSpec.metric '{self.metric}' invalid; "
                    f"allowed: {sorted(ALLOWED_SEMANTIC_METRICS)}"
                )


class FeatureSpaceRegistry:
    """An ordered registry of :class:`FeatureSpaceSpec` keyed by id."""

    def __init__(self) -> None:
        self._specs: dict[str, FeatureSpaceSpec] = {}

    def register(self, spec: FeatureSpaceSpec) -> None:
        if not isinstance(spec, FeatureSpaceSpec):
            raise CCRTValidationError("register expects a FeatureSpaceSpec")
        if spec.feature_space_id in self._specs:
            raise CCRTValidationError(
                f"duplicate feature_space_id '{spec.feature_space_id}'"
            )
        self._specs[spec.feature_space_id] = spec

    def get(self, feature_space_id: str) -> FeatureSpaceSpec:
        if feature_space_id not in self._specs:
            raise CCRTValidationError(
                f"unknown feature_space_id '{feature_space_id}' "
                f"(known: {list(self._specs.keys())})"
            )
        return self._specs[feature_space_id]

    def contains(self, feature_space_id: str) -> bool:
        return feature_space_id in self._specs

    def ids(self) -> tuple[str, ...]:
        """Registered ids in insertion order."""
        return tuple(self._specs.keys())

    def validate_tensor(
        self,
        feature_space_id: str,
        tensor: torch.Tensor,
        *,
        expected_role: str | None = None,
    ) -> None:
        """Validate that ``tensor`` conforms to a registered space."""
        spec = self.get(feature_space_id)
        if expected_role is not None and spec.role != expected_role:
            raise CCRTValidationError(
                f"feature_space '{feature_space_id}' has role '{spec.role}', "
                f"expected '{expected_role}'"
            )
        if not isinstance(tensor, torch.Tensor):
            raise CCRTValidationError(
                f"feature_space '{feature_space_id}': expected a torch.Tensor"
            )
        if not torch.is_floating_point(tensor):
            raise CCRTValidationError(
                f"feature_space '{feature_space_id}': tensor must be floating point"
            )
        if tensor.dim() != 2:
            raise CCRTShapeError(
                f"feature_space '{feature_space_id}': tensor must be rank 2 [N, D], "
                f"got shape {tuple(tensor.shape)}"
            )
        n, d = tensor.shape
        if n <= 0:
            raise CCRTShapeError(
                f"feature_space '{feature_space_id}': N must be > 0"
            )
        if d != spec.dimension:
            raise CCRTShapeError(
                f"feature_space '{feature_space_id}': dimension {d} != registered "
                f"dimension {spec.dimension}"
            )
        if not bool(torch.isfinite(tensor).all()):
            raise CCRTValidationError(
                f"feature_space '{feature_space_id}': tensor contains non-finite values"
            )
