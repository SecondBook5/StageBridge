"""Cross-system categorical index registry.

CCRT is unified at the grammar level, so the same local vocabulary name may
appear in two biological systems without meaning the same thing. Neural indices
must therefore be **system-qualified**: a local id ``local`` in system ``S`` is
globally identified as ``S::local``. This registry builds deterministic,
order-preserving integer indices for sender-context types and transition edges
across one or more ``BiologicalSystemSpec`` objects.

The empty sender is a model-reserved global element (not a biological ontology
entry). It receives one reserved index immediately after all real sender-context
types, so ``num_sender_context_types`` includes it.

No hashing, no silent fallback, no ontology merging (identical local names in
different systems stay distinct).
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

from ..contracts.errors import CCRTValidationError
from ..contracts.naming import assert_no_forbidden_mechanism_fields
from ..grammar import BiologicalSystemSpec

__all__ = [
    "QUALIFIED_ID_SEPARATOR",
    "qualify_grammar_id",
    "CCRTIndexRegistry",
]

QUALIFIED_ID_SEPARATOR = "::"


def qualify_grammar_id(biological_system_id: str, local_id: str) -> str:
    """Return the system-qualified id ``biological_system_id::local_id``."""
    for name, value in (
        ("biological_system_id", biological_system_id),
        ("local_id", local_id),
    ):
        if not isinstance(value, str) or not value.strip():
            raise CCRTValidationError(f"{name} must be a non-empty string")
        if QUALIFIED_ID_SEPARATOR in value:
            raise CCRTValidationError(
                f"{name} '{value}' must not contain the separator "
                f"'{QUALIFIED_ID_SEPARATOR}'"
            )
    # Forbidden mechanism ids must never enter the index space.
    assert_no_forbidden_mechanism_fields([biological_system_id, local_id])
    return f"{biological_system_id}{QUALIFIED_ID_SEPARATOR}{local_id}"


@dataclass(frozen=True)
class CCRTIndexRegistry:
    """Deterministic categorical indices across one or more systems."""

    biological_system_to_index: Mapping[str, int]
    sender_context_type_to_index: Mapping[str, int]
    transition_edge_to_index: Mapping[str, int]
    empty_sender_context_type_index: int

    # -- counts --

    @property
    def num_biological_systems(self) -> int:
        return len(self.biological_system_to_index)

    @property
    def num_real_sender_context_types(self) -> int:
        return len(self.sender_context_type_to_index)

    @property
    def num_sender_context_types(self) -> int:
        # includes the reserved empty sender
        return self.num_real_sender_context_types + 1

    @property
    def num_transition_edges(self) -> int:
        return len(self.transition_edge_to_index)

    # -- construction --

    @classmethod
    def from_system_specs(
        cls, specs: Sequence[BiologicalSystemSpec]
    ) -> "CCRTIndexRegistry":
        if not specs:
            raise CCRTValidationError("at least one BiologicalSystemSpec is required")

        systems: dict[str, int] = {}
        senders: dict[str, int] = {}
        edges: dict[str, int] = {}

        for spec in specs:
            sid = spec.biological_system_id
            if sid in systems:
                raise CCRTValidationError(
                    f"duplicate biological_system_id '{sid}'"
                )
            systems[sid] = len(systems)

            # Preserve ontology order within each spec.
            for sctype in spec.sender_context_types:
                qualified = qualify_grammar_id(sid, sctype.sender_context_type_id)
                if qualified not in senders:
                    senders[qualified] = len(senders)
            for edge in spec.transition_edges:
                qualified = qualify_grammar_id(sid, edge.transition_edge_id)
                if qualified not in edges:
                    edges[qualified] = len(edges)

        empty_index = len(senders)  # reserved index after all real sender types

        return cls(
            biological_system_to_index=MappingProxyType(dict(systems)),
            sender_context_type_to_index=MappingProxyType(dict(senders)),
            transition_edge_to_index=MappingProxyType(dict(edges)),
            empty_sender_context_type_index=empty_index,
        )

    # -- encode --

    def encode_biological_system(self, biological_system_id: str) -> int:
        if biological_system_id not in self.biological_system_to_index:
            raise CCRTValidationError(
                f"unknown biological_system_id '{biological_system_id}'"
            )
        return self.biological_system_to_index[biological_system_id]

    def encode_sender_context_type(
        self, biological_system_id: str, sender_context_type_id: str
    ) -> int:
        qualified = qualify_grammar_id(biological_system_id, sender_context_type_id)
        if qualified not in self.sender_context_type_to_index:
            raise CCRTValidationError(
                f"unknown sender_context_type '{qualified}'"
            )
        return self.sender_context_type_to_index[qualified]

    def encode_transition_edge(
        self, biological_system_id: str, transition_edge_id: str
    ) -> int:
        qualified = qualify_grammar_id(biological_system_id, transition_edge_id)
        if qualified not in self.transition_edge_to_index:
            raise CCRTValidationError(
                f"unknown transition_edge '{qualified}'"
            )
        return self.transition_edge_to_index[qualified]

    # -- decode --

    def _split_qualified(self, qualified: str) -> tuple[str, str]:
        system, _, local = qualified.partition(QUALIFIED_ID_SEPARATOR)
        return system, local

    def decode_biological_system(self, index: int) -> str:
        for name, idx in self.biological_system_to_index.items():
            if idx == index:
                return name
        raise CCRTValidationError(f"invalid biological-system index {index}")

    def decode_sender_context_type(self, index: int) -> tuple[str, str]:
        if index == self.empty_sender_context_type_index:
            raise CCRTValidationError(
                f"index {index} is the reserved empty-sender index and is not a "
                "biological sender-context type; it is a model-reserved element"
            )
        for qualified, idx in self.sender_context_type_to_index.items():
            if idx == index:
                return self._split_qualified(qualified)
        raise CCRTValidationError(f"invalid sender-context-type index {index}")

    def decode_transition_edge(self, index: int) -> tuple[str, str]:
        for qualified, idx in self.transition_edge_to_index.items():
            if idx == index:
                return self._split_qualified(qualified)
        raise CCRTValidationError(f"invalid transition-edge index {index}")
