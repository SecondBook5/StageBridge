"""PanIN ontology construction.

Maps source-backed PanIN annotations to canonical CCRT grammar ids and builds a
``BiologicalSystemSpec`` that passes the existing grammar validation. Unknown
labels fail in strict mode and are never silently bucketed into a catch-all id.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from ...contracts.errors import CCRTGrammarError, CCRTValidationError
from ...grammar import (
    GROWTH_MASS,
    SEMANTIC_DRIFT,
    BiologicalSystemSpec,
    ReceiverBehavior,
    ReceiverState,
    SenderContextType,
    SignalProgram,
    TransitionEdge,
)
from .config import PanINAdapterConfig

__all__ = [
    "PanINOntologyEntry",
    "PanINOntology",
    "build_panin_ontology",
]


@dataclass(frozen=True)
class PanINOntologyEntry:
    source_label: str
    canonical_id: str
    role: str
    included: bool
    evidence: str
    exclusion_reason: str | None = None


@dataclass(frozen=True)
class PanINOntology:
    biological_system_spec: BiologicalSystemSpec
    receiver_entries: tuple[PanINOntologyEntry, ...]
    sender_context_entries: tuple[PanINOntologyEntry, ...]
    stage_entries: tuple[PanINOntologyEntry, ...]
    transition_edge_records: tuple[tuple[str, str, str], ...]
    receiver_map: Mapping[str, str]
    sender_map: Mapping[str, str]
    stage_map: Mapping[str, str]
    strict: bool = True
    excluded: frozenset[str] = frozenset()

    def _map(self, label: str, table: Mapping[str, str], kind: str) -> str | None:
        if label in table:
            return table[label]
        if label in self.excluded:
            return None
        if self.strict:
            raise CCRTValidationError(
                f"unknown {kind} label '{label}' (strict mode; not in map and not "
                "explicitly excluded)"
            )
        return None

    def map_receiver_annotation(self, label: str) -> str | None:
        return self._map(label, self.receiver_map, "receiver")

    def map_sender_context_annotation(self, label: str) -> str | None:
        return self._map(label, self.sender_map, "sender-context")

    def map_stage(self, label: str) -> str | None:
        return self._map(label, self.stage_map, "stage")


def build_panin_ontology(config: PanINAdapterConfig) -> PanINOntology:
    """Build the PanIN ontology + validated BiologicalSystemSpec."""
    # canonical receiver states (dedup, preserve first-seen order)
    receiver_states: list[str] = []
    for canonical in config.receiver_annotation_map.values():
        if canonical not in receiver_states:
            receiver_states.append(canonical)

    sender_types: list[str] = []
    for canonical in config.sender_context_annotation_map.values():
        if canonical not in sender_types:
            sender_types.append(canonical)

    # entries (provenance)
    receiver_entries = tuple(
        PanINOntologyEntry(
            source_label=src, canonical_id=canon, role="receiver_state",
            included=True, evidence="config.receiver_annotation_map",
        )
        for src, canon in config.receiver_annotation_map.items()
    )
    sender_entries = tuple(
        PanINOntologyEntry(
            source_label=src, canonical_id=canon, role="sender_context_type",
            included=True, evidence="config.sender_context_annotation_map",
        )
        for src, canon in config.sender_context_annotation_map.items()
    )
    stage_entries = tuple(
        PanINOntologyEntry(
            source_label=src, canonical_id=canon, role="stage",
            included=True, evidence="config.stage_map",
        )
        for src, canon in config.stage_map.items()
    )
    excluded_entries = tuple(
        PanINOntologyEntry(
            source_label=lbl, canonical_id="", role="excluded", included=False,
            evidence="config.excluded_annotations", exclusion_reason="source-excluded",
        )
        for lbl in config.excluded_annotations
    )

    # grammar objects
    grammar_states = tuple(
        ReceiverState(s, order=float(i)) for i, s in enumerate(receiver_states)
    )
    grammar_edges = tuple(
        TransitionEdge(eid, src, tgt) for eid, src, tgt in config.transition_edges
    )
    grammar_senders = tuple(
        SenderContextType(t, signal_program_ids=("panin_context_signal",))
        for t in sender_types
    )
    signal_programs = (SignalProgram("panin_context_signal"),)
    behaviors = (ReceiverBehavior(SEMANTIC_DRIFT), ReceiverBehavior(GROWTH_MASS))

    spec = BiologicalSystemSpec(
        biological_system_id=config.biological_system_id,
        receiver_states=grammar_states,
        transition_edges=grammar_edges,
        sender_context_types=grammar_senders,
        signal_programs=signal_programs,
        receiver_behaviors=behaviors,
        metadata={"hypothesis": "CAF/stromal context alters PanIN epithelial transitions"},
    )
    try:
        spec.validate()
    except CCRTGrammarError as exc:
        raise CCRTValidationError(f"PanIN BiologicalSystemSpec invalid: {exc}") from exc

    return PanINOntology(
        biological_system_spec=spec,
        receiver_entries=receiver_entries + excluded_entries,
        sender_context_entries=sender_entries,
        stage_entries=stage_entries,
        transition_edge_records=config.transition_edges,
        receiver_map=dict(config.receiver_annotation_map),
        sender_map=dict(config.sender_context_annotation_map),
        stage_map=dict(config.stage_map),
        strict=config.strict_unknown_annotations,
        excluded=frozenset(config.excluded_annotations),
    )
