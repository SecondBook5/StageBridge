"""Canonical disease edges for the v1 LUAD ladder."""
from __future__ import annotations

from dataclasses import dataclass

from stagebridge.data.luad_evo.stages import CANONICAL_STAGE_ORDER, ordered_transitions


@dataclass(slots=True, frozen=True)
class DiseaseEdge:
    """One directed transition in the v1 disease ladder."""

    stage_src: str
    stage_tgt: str


V1_DISEASE_EDGES = tuple(DiseaseEdge(src, tgt) for src, tgt in ordered_transitions(CANONICAL_STAGE_ORDER))
