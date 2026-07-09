"""CCRT grammar entities.

Frozen dataclasses for the biological-meaning vocabulary a system declares.
Each entity validates only *itself* in ``__post_init__`` (non-empty IDs, tuple
coercion of collection fields). Cross-object reference checks (does this edge's
source_state exist? does this counterfactual target a real sender-context type?)
belong to ``BiologicalSystemSpec``, not here.

These objects carry no biology of their own: their *IDs* are system-specific
vocabulary supplied by each system's spec, but the classes are identical across
all systems. That is the "same grammar, different vocabulary" contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..contracts.errors import CCRTGrammarError

__all__ = [
    "ReceiverState",
    "TransitionEdge",
    "SenderContextType",
    "SignalProgram",
    "ReceiverBehavior",
    "RegulatoryMediator",
    "CounterfactualPerturbation",
]


def _require_nonempty_id(value: Any, field_name: str, cls_name: str) -> None:
    """Raise ``CCRTGrammarError`` unless ``value`` is a non-empty (stripped) str."""
    if not isinstance(value, str):
        raise CCRTGrammarError(
            f"{cls_name}.{field_name} must be a str, got {type(value).__name__}"
        )
    if not value.strip():
        raise CCRTGrammarError(
            f"{cls_name}.{field_name} must be a non-empty identifier"
        )


def _as_str_tuple(value: Any, field_name: str, cls_name: str) -> tuple[str, ...]:
    """Coerce a sequence of strings to a tuple; reject strings and non-str items."""
    if isinstance(value, (str, bytes)):
        raise CCRTGrammarError(
            f"{cls_name}.{field_name} must be a sequence of ids, not a bare string"
        )
    try:
        items = tuple(value)
    except TypeError as exc:
        raise CCRTGrammarError(
            f"{cls_name}.{field_name} must be an iterable of ids"
        ) from exc
    for item in items:
        if not isinstance(item, str) or not item.strip():
            raise CCRTGrammarError(
                f"{cls_name}.{field_name} entries must be non-empty strings; "
                f"got {item!r}"
            )
    return items


@dataclass(frozen=True)
class ReceiverState:
    """A receiver semantic state in a system's ReceiverStateOntology."""

    receiver_state_id: str
    label: str | None = None
    order: float | None = None
    description: str | None = None

    def __post_init__(self) -> None:
        _require_nonempty_id(
            self.receiver_state_id, "receiver_state_id", "ReceiverState"
        )


@dataclass(frozen=True)
class TransitionEdge:
    """A directed edge between two receiver states in the TransitionGraph."""

    transition_edge_id: str
    source_state_id: str
    target_state_id: str
    label: str | None = None
    order: float | None = None
    edge_type: str = "directed"

    def __post_init__(self) -> None:
        _require_nonempty_id(
            self.transition_edge_id, "transition_edge_id", "TransitionEdge"
        )
        _require_nonempty_id(self.source_state_id, "source_state_id", "TransitionEdge")
        _require_nonempty_id(self.target_state_id, "target_state_id", "TransitionEdge")
        _require_nonempty_id(self.edge_type, "edge_type", "TransitionEdge")


@dataclass(frozen=True)
class SenderContextType:
    """A typed sender-context category in a system's SenderContextOntology."""

    sender_context_type_id: str
    label: str | None = None
    description: str | None = None
    signal_program_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_nonempty_id(
            self.sender_context_type_id,
            "sender_context_type_id",
            "SenderContextType",
        )
        object.__setattr__(
            self,
            "signal_program_ids",
            _as_str_tuple(
                self.signal_program_ids, "signal_program_ids", "SenderContextType"
            ),
        )


@dataclass(frozen=True)
class SignalProgram:
    """A signal/program in a system's SignalProgramRegistry."""

    signal_program_id: str
    label: str | None = None
    feature_ids: tuple[str, ...] = ()
    description: str | None = None

    def __post_init__(self) -> None:
        _require_nonempty_id(
            self.signal_program_id, "signal_program_id", "SignalProgram"
        )
        object.__setattr__(
            self,
            "feature_ids",
            _as_str_tuple(self.feature_ids, "feature_ids", "SignalProgram"),
        )


@dataclass(frozen=True)
class ReceiverBehavior:
    """A modeled receiver behavior in a system's ReceiverBehaviorRegistry."""

    receiver_behavior_id: str
    label: str | None = None
    description: str | None = None

    def __post_init__(self) -> None:
        _require_nonempty_id(
            self.receiver_behavior_id, "receiver_behavior_id", "ReceiverBehavior"
        )


@dataclass(frozen=True)
class RegulatoryMediator:
    """A regulatory mediator in a system's RegulatoryMediatorRegistry."""

    regulatory_mediator_id: str
    label: str | None = None
    signal_program_ids: tuple[str, ...] = ()
    description: str | None = None

    def __post_init__(self) -> None:
        _require_nonempty_id(
            self.regulatory_mediator_id,
            "regulatory_mediator_id",
            "RegulatoryMediator",
        )
        object.__setattr__(
            self,
            "signal_program_ids",
            _as_str_tuple(
                self.signal_program_ids, "signal_program_ids", "RegulatoryMediator"
            ),
        )


@dataclass(frozen=True)
class CounterfactualPerturbation:
    """A named sender-context counterfactual perturbation.

    ``perturbation_kind`` names the class of intervention (e.g.
    ``remove_sender_context``, ``silence_signal_program``). Target ID tuples say
    *which* sender-context types / signal programs the perturbation acts on.
    """

    counterfactual_perturbation_id: str
    label: str | None = None
    perturbation_kind: str = "remove_sender_context"
    target_sender_context_type_ids: tuple[str, ...] = ()
    target_signal_program_ids: tuple[str, ...] = ()
    description: str | None = None

    def __post_init__(self) -> None:
        _require_nonempty_id(
            self.counterfactual_perturbation_id,
            "counterfactual_perturbation_id",
            "CounterfactualPerturbation",
        )
        _require_nonempty_id(
            self.perturbation_kind,
            "perturbation_kind",
            "CounterfactualPerturbation",
        )
        object.__setattr__(
            self,
            "target_sender_context_type_ids",
            _as_str_tuple(
                self.target_sender_context_type_ids,
                "target_sender_context_type_ids",
                "CounterfactualPerturbation",
            ),
        )
        object.__setattr__(
            self,
            "target_signal_program_ids",
            _as_str_tuple(
                self.target_signal_program_ids,
                "target_signal_program_ids",
                "CounterfactualPerturbation",
            ),
        )
