"""PyTorch training batch and tensorization from a validated ``CCRTBatch``.

``CCRTTrainingBatch`` is the tensor container the operator + objective consume:
receiver/sender features, mask, distance, integer sender-context type indices,
integer transition-edge indices, source/target semantic populations, and
optional uncertainty / weights / growth supervision.

``build_training_batch`` converts a CPU ``CCRTBatch`` (with string grammar ids)
into a ``CCRTTrainingBatch`` by mapping grammar ids through a
``CCRTIndexRegistry``. Masked padding positions receive integer index 0 (they are
excluded by ``sender_mask``) — never the reserved empty-sender index, and never
a fabricated biological type.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch

from ..contracts.errors import CCRTShapeError, CCRTValidationError
from ..contracts.naming import (
    assert_no_forbidden_mechanism_fields,
    assert_no_model_input_leakage_fields,
)
from ..data.batch import CCRTBatch
from ..data.indexing import CCRTIndexRegistry

__all__ = ["CCRTTrainingBatch", "build_training_batch"]


def _is_int_tensor(t: torch.Tensor) -> bool:
    return t.dtype in (torch.int16, torch.int32, torch.int64)


@dataclass(frozen=True)
class CCRTTrainingBatch:
    """A validated, device-placed PyTorch training batch."""

    receiver_features: torch.Tensor
    sender_features: torch.Tensor
    sender_mask: torch.Tensor
    distance_to_receiver: torch.Tensor
    sender_context_type_ids: torch.Tensor
    transition_edge_index: torch.Tensor | None
    source_semantic_features: torch.Tensor
    target_semantic_features: torch.Tensor
    uncertainty: torch.Tensor | None = None
    source_weights: torch.Tensor | None = None
    target_weights: torch.Tensor | None = None
    growth_targets: torch.Tensor | None = None
    growth_mask: torch.Tensor | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    # -- sizes --

    def batch_size(self) -> int:
        return int(self.receiver_features.shape[0])

    def target_size(self) -> int:
        return int(self.target_semantic_features.shape[0])

    def max_sender_context(self) -> int:
        return int(self.sender_features.shape[1])

    def semantic_dim(self) -> int:
        return int(self.source_semantic_features.shape[1])

    # -- validation --

    def validate(self) -> None:
        rf, sf, sm, dist = (
            self.receiver_features,
            self.sender_features,
            self.sender_mask,
            self.distance_to_receiver,
        )
        for name, t in (
            ("receiver_features", rf),
            ("sender_features", sf),
            ("distance_to_receiver", dist),
            ("source_semantic_features", self.source_semantic_features),
            ("target_semantic_features", self.target_semantic_features),
        ):
            if not isinstance(t, torch.Tensor) or not torch.is_floating_point(t):
                raise CCRTValidationError(f"{name} must be a floating tensor")
            if not bool(torch.isfinite(t).all()):
                raise CCRTValidationError(f"{name} contains non-finite values")

        if rf.dim() != 2:
            raise CCRTShapeError("receiver_features must be [B, D_R]")
        b = rf.shape[0]
        if sf.dim() != 3 or sf.shape[0] != b:
            raise CCRTShapeError("sender_features must be [B, K, D_S]")
        k = sf.shape[1]

        if tuple(sm.shape) != (b, k):
            raise CCRTShapeError("sender_mask must be [B, K]")
        if tuple(dist.shape) != (b, k):
            raise CCRTShapeError("distance_to_receiver must be [B, K]")
        if bool((dist < 0).any()):
            raise CCRTValidationError("distance_to_receiver must be non-negative")

        st = self.sender_context_type_ids
        if not isinstance(st, torch.Tensor) or not _is_int_tensor(st):
            raise CCRTValidationError("sender_context_type_ids must be an integer tensor")
        if tuple(st.shape) != (b, k):
            raise CCRTShapeError("sender_context_type_ids must be [B, K]")
        if bool((st < 0).any()):
            raise CCRTValidationError("sender_context_type_ids must be non-negative")

        if self.transition_edge_index is not None:
            tei = self.transition_edge_index
            if not isinstance(tei, torch.Tensor) or not _is_int_tensor(tei):
                raise CCRTValidationError("transition_edge_index must be integer")
            if tuple(tei.shape) != (b,):
                raise CCRTShapeError("transition_edge_index must be [B]")
            if bool((tei < 0).any()):
                raise CCRTValidationError("transition_edge_index must be non-negative")

        src_sem = self.source_semantic_features
        tgt_sem = self.target_semantic_features
        if src_sem.dim() != 2 or src_sem.shape[0] != b:
            raise CCRTShapeError("source_semantic_features must be [B, D_Z]")
        if tgt_sem.dim() != 2:
            raise CCRTShapeError("target_semantic_features must be [M, D_Z]")
        if src_sem.shape[1] != tgt_sem.shape[1]:
            raise CCRTShapeError(
                "source and target semantic dimensions must match"
            )
        m = tgt_sem.shape[0]

        if self.uncertainty is not None:
            u = self.uncertainty
            if not torch.is_floating_point(u):
                raise CCRTValidationError("uncertainty must be floating")
            if tuple(u.shape) != (b, k):
                raise CCRTShapeError("uncertainty must be [B, K]")
            if not bool(torch.isfinite(u).all()) or bool((u < 0).any()):
                raise CCRTValidationError("uncertainty must be finite and non-negative")

        for name, w, size in (
            ("source_weights", self.source_weights, b),
            ("target_weights", self.target_weights, m),
        ):
            if w is not None:
                if not torch.is_floating_point(w):
                    raise CCRTValidationError(f"{name} must be floating")
                if tuple(w.shape) != (size,):
                    raise CCRTShapeError(f"{name} must be [{size}]")
                if not bool(torch.isfinite(w).all()):
                    raise CCRTValidationError(f"{name} contains non-finite values")
                if bool((w <= 0).any()):
                    raise CCRTValidationError(f"{name} must be strictly positive")

        if self.growth_targets is not None:
            gt = self.growth_targets
            if not torch.is_floating_point(gt):
                raise CCRTValidationError("growth_targets must be floating")
            if gt.dim() != 2 or gt.shape[0] != b:
                raise CCRTShapeError("growth_targets must be [B, G]")
            if not bool(torch.isfinite(gt).all()):
                raise CCRTValidationError("growth_targets contains non-finite values")
            g = gt.shape[1]
            if self.growth_mask is not None:
                gm = self.growth_mask
                if tuple(gm.shape) != (b, g):
                    raise CCRTShapeError("growth_mask must match growth_targets [B, G]")
        elif self.growth_mask is not None:
            raise CCRTValidationError("growth_mask requires growth_targets")

        meta_keys = list(self.metadata.keys())
        assert_no_forbidden_mechanism_fields(meta_keys)
        assert_no_model_input_leakage_fields(meta_keys)

    # -- device / dtype movement --

    def to(
        self,
        device: torch.device | str,
        *,
        dtype: torch.dtype | None = None,
    ) -> "CCRTTrainingBatch":
        """Return a new batch on ``device``; floats optionally recast to ``dtype``."""

        def move_float(t: torch.Tensor | None) -> torch.Tensor | None:
            if t is None:
                return None
            if dtype is not None and torch.is_floating_point(t):
                return t.to(device=device, dtype=dtype)
            return t.to(device=device)

        def move_int(t: torch.Tensor | None) -> torch.Tensor | None:
            return None if t is None else t.to(device=device)

        return CCRTTrainingBatch(
            receiver_features=move_float(self.receiver_features),
            sender_features=move_float(self.sender_features),
            sender_mask=move_int(self.sender_mask),  # bool/int preserved
            distance_to_receiver=move_float(self.distance_to_receiver),
            sender_context_type_ids=move_int(self.sender_context_type_ids),
            transition_edge_index=move_int(self.transition_edge_index),
            source_semantic_features=move_float(self.source_semantic_features),
            target_semantic_features=move_float(self.target_semantic_features),
            uncertainty=move_float(self.uncertainty),
            source_weights=move_float(self.source_weights),
            target_weights=move_float(self.target_weights),
            growth_targets=move_float(self.growth_targets),
            growth_mask=move_int(self.growth_mask),
            metadata=dict(self.metadata),  # stays on CPU, unchanged
        )


# ---------------------------------------------------------------------------
# Tensorization from a validated CCRTBatch
# ---------------------------------------------------------------------------


def _as_float_tensor(value: Any, dtype: torch.dtype, name: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        t = value.to(dtype=dtype)
    else:
        try:
            t = torch.as_tensor(value, dtype=dtype)
        except Exception as exc:  # pragma: no cover - defensive
            raise CCRTValidationError(f"{name}: could not convert to tensor: {exc}") from exc
    return t


def _row_system(batch: CCRTBatch, idx: int) -> str:
    sid = batch.biological_system_id
    if isinstance(sid, str):
        return sid
    return sid[idx]


def _row_edge(batch: CCRTBatch, idx: int) -> str:
    edge = batch.transition_edge_id
    if isinstance(edge, str):
        return edge
    return edge[idx]


def build_training_batch(
    *,
    source_batch: CCRTBatch,
    target_semantic_features: torch.Tensor | Sequence,
    index_registry: CCRTIndexRegistry,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    source_weights: torch.Tensor | Sequence | None = None,
    target_weights: torch.Tensor | Sequence | None = None,
    growth_targets: torch.Tensor | Sequence | None = None,
    growth_mask: torch.Tensor | Sequence | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> CCRTTrainingBatch:
    """Build a ``CCRTTrainingBatch`` from a validated source ``CCRTBatch``."""
    source_batch.validate()

    if source_batch.semantic_features is None:
        raise CCRTValidationError(
            "build_training_batch requires source_batch.semantic_features"
        )
    if source_batch.sender_context_type_ids is None:
        raise CCRTValidationError(
            "build_training_batch requires source_batch.sender_context_type_ids "
            "(run collation with the sender_context_type_id column)"
        )

    # -- float feature tensors --
    receiver_features = _as_float_tensor(
        source_batch.receiver_features, dtype, "receiver_features"
    )
    sender_features = _as_float_tensor(
        source_batch.sender_features, dtype, "sender_features"
    )
    distance = _as_float_tensor(
        source_batch.distance_to_receiver, dtype, "distance_to_receiver"
    )
    source_sem = _as_float_tensor(
        source_batch.semantic_features, dtype, "source_semantic_features"
    )
    target_sem = _as_float_tensor(
        target_semantic_features, dtype, "target_semantic_features"
    )

    b, k = sender_features.shape[0], sender_features.shape[1]

    # -- sender mask (bool) --
    mask = torch.as_tensor(source_batch.sender_mask)
    sender_mask = mask.to(dtype=torch.bool)
    if tuple(sender_mask.shape) != (b, k):
        raise CCRTShapeError("sender_mask shape does not match sender_features")

    # -- sender-context type indices --
    raw_types = source_batch.sender_context_type_ids
    type_index_rows: list[list[int]] = []
    for i in range(b):
        system = _row_system(source_batch, i)
        row: list[int] = []
        for j in range(k):
            token = raw_types[i][j]
            if token is None:
                # masked padding: harmless index 0 (excluded by sender_mask).
                if bool(sender_mask[i, j]):
                    raise CCRTValidationError(
                        f"sender_context_type_ids[{i}][{j}] is None but the "
                        "position is unmasked (real senders need a grammar id)"
                    )
                row.append(0)
            else:
                row.append(
                    index_registry.encode_sender_context_type(system, token)
                )
        type_index_rows.append(row)
    sender_type_index = torch.tensor(type_index_rows, dtype=torch.int64)

    # -- transition edge indices --
    edge_rows: list[int] = []
    for i in range(b):
        system = _row_system(source_batch, i)
        edge = _row_edge(source_batch, i)
        if edge is None or (isinstance(edge, str) and not edge.strip()):
            raise CCRTValidationError(
                f"row {i} is missing a transition_edge_id required for indexing"
            )
        edge_rows.append(index_registry.encode_transition_edge(system, edge))
    transition_edge_index = torch.tensor(edge_rows, dtype=torch.int64)

    # -- optional tensors --
    uncertainty = (
        None
        if source_batch.uncertainty is None
        else _as_float_tensor(source_batch.uncertainty, dtype, "uncertainty")
    )
    src_w = (
        None if source_weights is None
        else _as_float_tensor(source_weights, dtype, "source_weights")
    )
    tgt_w = (
        None if target_weights is None
        else _as_float_tensor(target_weights, dtype, "target_weights")
    )
    g_targets = (
        None if growth_targets is None
        else _as_float_tensor(growth_targets, dtype, "growth_targets")
    )
    g_mask = None
    if growth_mask is not None:
        g_mask = torch.as_tensor(growth_mask).to(dtype=torch.bool)

    if metadata is not None:
        meta_keys = list(metadata.keys())
        assert_no_forbidden_mechanism_fields(meta_keys)
        assert_no_model_input_leakage_fields(meta_keys)

    training_batch = CCRTTrainingBatch(
        receiver_features=receiver_features,
        sender_features=sender_features,
        sender_mask=sender_mask,
        distance_to_receiver=distance,
        sender_context_type_ids=sender_type_index,
        transition_edge_index=transition_edge_index,
        source_semantic_features=source_sem,
        target_semantic_features=target_sem,
        uncertainty=uncertainty,
        source_weights=src_w,
        target_weights=tgt_w,
        growth_targets=g_targets,
        growth_mask=g_mask,
        metadata=dict(metadata) if metadata is not None else {},
    )
    training_batch.validate()
    return training_batch.to(device, dtype=dtype)
