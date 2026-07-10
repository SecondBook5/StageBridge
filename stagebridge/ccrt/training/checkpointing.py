"""Safe checkpoint contract.

Checkpoints store only state dictionaries and plain metadata — never full model /
optimizer / trainer objects or arbitrary callables. Saves are atomic (temp
sibling + ``os.replace``). Loads validate the schema and prefer
``weights_only=True`` where the installed PyTorch supports it, so loading never
executes arbitrary pickle code.
"""

from __future__ import annotations

import inspect
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn

from ..contracts.errors import CCRTValidationError
from ..contracts.naming import (
    assert_no_forbidden_mechanism_fields,
    assert_no_model_input_leakage_fields,
)

__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "CheckpointMetadata",
    "build_checkpoint_state",
    "save_checkpoint",
    "load_checkpoint",
    "restore_checkpoint",
]

CHECKPOINT_SCHEMA_VERSION = "ccrt-checkpoint-1"

_REQUIRED_KEYS = (
    "schema_version",
    "metadata",
    "model_state_dict",
    "optimizer_state_dict",
    "scheduler_state_dict",
)


@dataclass(frozen=True)
class CheckpointMetadata:
    """Plain, serializable checkpoint metadata."""

    schema_version: str
    epoch: int
    global_step: int
    model_class: str
    optimizer_class: str
    scheduler_class: str | None
    extra: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.schema_version != CHECKPOINT_SCHEMA_VERSION:
            raise CCRTValidationError(
                f"schema_version must be '{CHECKPOINT_SCHEMA_VERSION}'"
            )
        if not isinstance(self.epoch, int) or self.epoch < 0:
            raise CCRTValidationError("epoch must be an int >= 0")
        if not isinstance(self.global_step, int) or self.global_step < 0:
            raise CCRTValidationError("global_step must be an int >= 0")
        if not self.model_class or not self.optimizer_class:
            raise CCRTValidationError("class names must be non-empty")
        extra_keys = list(self.extra.keys())
        assert_no_forbidden_mechanism_fields(extra_keys)
        assert_no_model_input_leakage_fields(extra_keys)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_class": self.model_class,
            "optimizer_class": self.optimizer_class,
            "scheduler_class": self.scheduler_class,
            "extra": dict(self.extra),
        }


def _qualified_class_name(obj: Any) -> str:
    cls = type(obj)
    return f"{cls.__module__}.{cls.__qualname__}"


def build_checkpoint_state(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    scheduler: Any | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a state dict containing only state dicts + plain metadata."""
    metadata = CheckpointMetadata(
        schema_version=CHECKPOINT_SCHEMA_VERSION,
        epoch=epoch,
        global_step=global_step,
        model_class=_qualified_class_name(model),
        optimizer_class=_qualified_class_name(optimizer),
        scheduler_class=None if scheduler is None else _qualified_class_name(scheduler),
        extra=dict(extra) if extra is not None else {},
    )
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "metadata": metadata.as_dict(),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": (
            None if scheduler is None else scheduler.state_dict()
        ),
    }


def save_checkpoint(path: str | Path, state: Mapping[str, Any]) -> Path:
    """Atomically save a checkpoint state to ``path`` (.pt/.pth)."""
    path = Path(path)
    if path.suffix not in (".pt", ".pth"):
        raise CCRTValidationError("checkpoint path must end in .pt or .pth")
    if not path.parent.exists():
        raise CCRTValidationError(
            f"checkpoint parent directory does not exist: {path.parent}"
        )
    if "schema_version" not in state or state["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise CCRTValidationError("state is missing a valid schema_version")

    tmp = path.with_name(path.name + ".tmp")
    try:
        torch.save(dict(state), tmp)
        os.replace(tmp, path)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise
    return path


def _supports_weights_only() -> bool:
    return "weights_only" in inspect.signature(torch.load).parameters


def load_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> Mapping[str, Any]:
    """Load and validate a checkpoint (weights_only when supported)."""
    path = Path(path)
    if not path.is_file():
        raise CCRTValidationError(f"checkpoint file not found: {path}")

    # Prefer weights_only=True so loading never executes arbitrary pickle code.
    # The checkpoint contains only tensors + a primitive metadata dict, which is
    # safe under the restricted unpickler.
    if _supports_weights_only():
        state = torch.load(path, map_location=map_location, weights_only=True)
    else:  # pragma: no cover - very old torch
        state = torch.load(path, map_location=map_location)

    if not isinstance(state, Mapping):
        raise CCRTValidationError("checkpoint content must be a mapping")
    if state.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise CCRTValidationError(
            f"checkpoint schema_version mismatch: expected "
            f"'{CHECKPOINT_SCHEMA_VERSION}', got {state.get('schema_version')!r}"
        )
    for key in _REQUIRED_KEYS:
        if key not in state:
            raise CCRTValidationError(f"checkpoint missing required key '{key}'")
    return state


def restore_checkpoint(
    *,
    state: Mapping[str, Any],
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
) -> CheckpointMetadata:
    """Restore model/optimizer/scheduler state and return the metadata."""
    if state.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise CCRTValidationError("state schema_version mismatch")

    model.load_state_dict(state["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer_state_dict"])

    if scheduler is not None:
        sched_state = state.get("scheduler_state_dict")
        if sched_state is None:
            raise CCRTValidationError(
                "scheduler restoration requested but checkpoint has no scheduler state"
            )
        scheduler.load_state_dict(sched_state)

    meta = state["metadata"]
    return CheckpointMetadata(
        schema_version=meta["schema_version"],
        epoch=meta["epoch"],
        global_step=meta["global_step"],
        model_class=meta["model_class"],
        optimizer_class=meta["optimizer_class"],
        scheduler_class=meta.get("scheduler_class"),
        extra=meta.get("extra", {}),
    )
