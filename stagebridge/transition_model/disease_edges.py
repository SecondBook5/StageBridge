"""Canonical disease edges for the v1 LUAD ladder."""

from __future__ import annotations

from dataclasses import dataclass

from stagebridge.data.luad_evo.stages import CANONICAL_STAGE_ORDER, ordered_transitions


@dataclass(slots=True, frozen=True)
class DiseaseEdge:
    """One directed transition in the v1 disease ladder."""

    stage_src: str
    stage_tgt: str


V1_DISEASE_EDGES = tuple(
    DiseaseEdge(src, tgt) for src, tgt in ordered_transitions(CANONICAL_STAGE_ORDER)
)


def edge_label(edge: DiseaseEdge | tuple[str, str]) -> str:
    if isinstance(edge, DiseaseEdge):
        return f"{edge.stage_src}->{edge.stage_tgt}"
    src, tgt = edge
    return f"{src}->{tgt}"


def edge_id_map() -> dict[str, int]:
    return {edge_label(edge): idx for idx, edge in enumerate(V1_DISEASE_EDGES)}


def resolve_disease_edge(edge: str | tuple[str, str] | list[str] | None = None) -> DiseaseEdge:
    """Resolve an active v1 disease edge with a safe default."""
    if edge is None:
        return DiseaseEdge("AAH", "AIS")
    if isinstance(edge, str):
        if "->" not in edge:
            raise ValueError(f"Edge string must have 'src->tgt' form, got '{edge}'.")
        src, tgt = [part.strip() for part in edge.split("->", 1)]
        return DiseaseEdge(src, tgt)
    if len(edge) != 2:
        raise ValueError(f"Edge sequence must have length 2, got {edge}.")
    return DiseaseEdge(str(edge[0]), str(edge[1]))
