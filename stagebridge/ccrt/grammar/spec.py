"""``BiologicalSystemSpec`` — one system's full grammar vocabulary.

A spec declares the receiver states, transition edges, sender-context types,
signal programs, receiver behaviors, regulatory mediators, and counterfactual
perturbations for a single biological system, plus a one-line hypothesis in
``metadata``. It is the single place system-specific vocabulary lives; the model
core consumes it only through the grammar API.

The standard receiver-behavior IDs below are provided as *constants*, not
*requirements*: a system may use ``semantic_drift`` and ``growth_mass``, or
declare its own behaviors (e.g. ``infection_state_transition`` for a viral
system). CCRT never requires two systems to share any vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from ..contracts.errors import CCRTGrammarError
from ..contracts.naming import (
    assert_no_forbidden_mechanism_fields,
    assert_no_model_input_leakage_fields,
)
from .entities import (
    CounterfactualPerturbation,
    ReceiverBehavior,
    ReceiverState,
    RegulatoryMediator,
    SenderContextType,
    SignalProgram,
    TransitionEdge,
)
from .validation import require_ids_exist, require_unique_ids

__all__ = [
    "BiologicalSystemSpec",
    # standard (non-mandatory) receiver behavior ids
    "SEMANTIC_DRIFT",
    "GROWTH_MASS",
    "DEATH_REMOVAL",
    "MOTILITY_INVASION",
    "INFECTION_STATE_TRANSITION",
    "REPAIR_RECOVERY",
]

# Standard receiver behavior IDs — constants offered for reuse, never required.
SEMANTIC_DRIFT = "semantic_drift"
GROWTH_MASS = "growth_mass"
DEATH_REMOVAL = "death_removal"
MOTILITY_INVASION = "motility_invasion"
INFECTION_STATE_TRANSITION = "infection_state_transition"
REPAIR_RECOVERY = "repair_recovery"


@dataclass(frozen=True)
class BiologicalSystemSpec:
    """The full grammar declaration for one biological system."""

    biological_system_id: str
    display_name: str | None = None
    receiver_states: tuple[ReceiverState, ...] = ()
    transition_edges: tuple[TransitionEdge, ...] = ()
    sender_context_types: tuple[SenderContextType, ...] = ()
    signal_programs: tuple[SignalProgram, ...] = ()
    receiver_behaviors: tuple[ReceiverBehavior, ...] = ()
    regulatory_mediators: tuple[RegulatoryMediator, ...] = ()
    counterfactual_perturbations: tuple[CounterfactualPerturbation, ...] = ()
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Coerce the registry sequences to tuples so the spec is hashable and
        # order-stable even if constructed from lists.
        for attr in (
            "receiver_states",
            "transition_edges",
            "sender_context_types",
            "signal_programs",
            "receiver_behaviors",
            "regulatory_mediators",
            "counterfactual_perturbations",
        ):
            object.__setattr__(self, attr, tuple(getattr(self, attr)))
        object.__setattr__(self, "metadata", dict(self.metadata))

    # -- ID accessors ------------------------------------------------------

    @property
    def receiver_state_ids(self) -> frozenset[str]:
        return frozenset(s.receiver_state_id for s in self.receiver_states)

    @property
    def transition_edge_ids(self) -> frozenset[str]:
        return frozenset(e.transition_edge_id for e in self.transition_edges)

    @property
    def sender_context_type_ids(self) -> frozenset[str]:
        return frozenset(c.sender_context_type_id for c in self.sender_context_types)

    @property
    def signal_program_ids(self) -> frozenset[str]:
        return frozenset(p.signal_program_id for p in self.signal_programs)

    @property
    def receiver_behavior_ids(self) -> frozenset[str]:
        return frozenset(b.receiver_behavior_id for b in self.receiver_behaviors)

    @property
    def regulatory_mediator_ids(self) -> frozenset[str]:
        return frozenset(
            m.regulatory_mediator_id for m in self.regulatory_mediators
        )

    @property
    def counterfactual_perturbation_ids(self) -> frozenset[str]:
        return frozenset(
            p.counterfactual_perturbation_id
            for p in self.counterfactual_perturbations
        )

    # -- lookups -----------------------------------------------------------

    def get_transition_edge(self, edge_id: str) -> TransitionEdge:
        """Return the edge with ``edge_id`` or raise ``CCRTGrammarError``."""
        for edge in self.transition_edges:
            if edge.transition_edge_id == edge_id:
                return edge
        raise CCRTGrammarError(
            f"biological_system_id='{self.biological_system_id}': "
            f"unknown transition_edge_id '{edge_id}' "
            f"(known: {sorted(self.transition_edge_ids)})"
        )

    def has_receiver_behavior(self, behavior_id: str) -> bool:
        """True if ``behavior_id`` is a registered receiver behavior."""
        return behavior_id in self.receiver_behavior_ids

    # -- validation --------------------------------------------------------

    def validate(self) -> None:
        """Validate the full spec. Raises ``CCRTGrammarError`` on any violation.

        This is the single authoritative validation routine (no recursion into
        the free-function wrapper). It enforces presence minimums, uniqueness,
        referential integrity, and forbidden-term hygiene.
        """
        sid = self.biological_system_id
        if not isinstance(sid, str) or not sid.strip():
            raise CCRTGrammarError(
                "BiologicalSystemSpec.biological_system_id must be a non-empty str"
            )
        ctx = f"biological_system_id='{sid}'"

        # -- presence minimums --
        if len(self.receiver_states) < 2:
            raise CCRTGrammarError(
                f"{ctx}: need at least two receiver states, got "
                f"{len(self.receiver_states)}"
            )
        if len(self.transition_edges) < 1:
            raise CCRTGrammarError(f"{ctx}: need at least one transition edge")
        if len(self.sender_context_types) < 1:
            raise CCRTGrammarError(f"{ctx}: need at least one sender context type")
        if len(self.receiver_behaviors) < 1:
            raise CCRTGrammarError(f"{ctx}: need at least one receiver behavior")

        # -- uniqueness within each registry --
        require_unique_ids(
            self.receiver_states, lambda s: s.receiver_state_id,
            f"{ctx} receiver_states",
        )
        require_unique_ids(
            self.transition_edges, lambda e: e.transition_edge_id,
            f"{ctx} transition_edges",
        )
        require_unique_ids(
            self.sender_context_types, lambda c: c.sender_context_type_id,
            f"{ctx} sender_context_types",
        )
        require_unique_ids(
            self.signal_programs, lambda p: p.signal_program_id,
            f"{ctx} signal_programs",
        )
        require_unique_ids(
            self.receiver_behaviors, lambda b: b.receiver_behavior_id,
            f"{ctx} receiver_behaviors",
        )
        require_unique_ids(
            self.regulatory_mediators, lambda m: m.regulatory_mediator_id,
            f"{ctx} regulatory_mediators",
        )
        require_unique_ids(
            self.counterfactual_perturbations,
            lambda p: p.counterfactual_perturbation_id,
            f"{ctx} counterfactual_perturbations",
        )

        # -- referential integrity --
        receiver_state_ids = self.receiver_state_ids
        signal_program_ids = self.signal_program_ids
        sender_context_type_ids = self.sender_context_type_ids

        for edge in self.transition_edges:
            require_ids_exist(
                (edge.source_state_id, edge.target_state_id),
                receiver_state_ids,
                f"{ctx} transition_edge '{edge.transition_edge_id}'",
            )

        for sctype in self.sender_context_types:
            require_ids_exist(
                sctype.signal_program_ids,
                signal_program_ids,
                f"{ctx} sender_context_type '{sctype.sender_context_type_id}'"
                " signal_program_ids",
            )

        for mediator in self.regulatory_mediators:
            require_ids_exist(
                mediator.signal_program_ids,
                signal_program_ids,
                f"{ctx} regulatory_mediator '{mediator.regulatory_mediator_id}'"
                " signal_program_ids",
            )

        for pert in self.counterfactual_perturbations:
            require_ids_exist(
                pert.target_sender_context_type_ids,
                sender_context_type_ids,
                f"{ctx} counterfactual '{pert.counterfactual_perturbation_id}'"
                " target_sender_context_type_ids",
            )
            require_ids_exist(
                pert.target_signal_program_ids,
                signal_program_ids,
                f"{ctx} counterfactual '{pert.counterfactual_perturbation_id}'"
                " target_signal_program_ids",
            )

        # -- forbidden-term hygiene on registry IDs --
        all_ids = [
            *self.receiver_state_ids,
            *self.transition_edge_ids,
            *self.sender_context_type_ids,
            *self.signal_program_ids,
            *self.receiver_behavior_ids,
            *self.regulatory_mediator_ids,
            *self.counterfactual_perturbation_ids,
        ]
        assert_no_forbidden_mechanism_fields(all_ids)

        # -- forbidden-term hygiene on metadata keys --
        metadata_keys = list(self.metadata.keys())
        assert_no_forbidden_mechanism_fields(metadata_keys)
        assert_no_model_input_leakage_fields(metadata_keys)
