"""Synthetic mechanism specifications.

Each canonical scenario is described by a ``SyntheticMechanismSpec`` — a pure
data object of strengths, scales, and active/negative-control type ids. It
carries no model code and no biology. ``build_synthetic_mechanism_spec`` maps a
scenario id + system config to its exact intended isolation (see the milestone
scenario matrix).
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import SyntheticSystemConfig

__all__ = [
    "NULL_CONTEXT",
    "DRIFT_ONLY",
    "GROWTH_ONLY",
    "MIXED_DRIFT_GROWTH",
    "REGULATORY_MEDIATED",
    "DISTANCE_DEPENDENT",
    "SENDER_TYPE_SPECIFIC",
    "TRANSITION_EDGE_SPECIFIC",
    "WRONG_CONTEXT_NEGATIVE_CONTROL",
    "SYNTHETIC_SCENARIO_IDS",
    "SyntheticMechanismSpec",
    "build_synthetic_mechanism_spec",
]

NULL_CONTEXT = "null_context"
DRIFT_ONLY = "drift_only"
GROWTH_ONLY = "growth_only"
MIXED_DRIFT_GROWTH = "mixed_drift_growth"
REGULATORY_MEDIATED = "regulatory_mediated"
DISTANCE_DEPENDENT = "distance_dependent"
SENDER_TYPE_SPECIFIC = "sender_type_specific"
TRANSITION_EDGE_SPECIFIC = "transition_edge_specific"
WRONG_CONTEXT_NEGATIVE_CONTROL = "wrong_context_negative_control"

SYNTHETIC_SCENARIO_IDS = (
    NULL_CONTEXT,
    DRIFT_ONLY,
    GROWTH_ONLY,
    MIXED_DRIFT_GROWTH,
    REGULATORY_MEDIATED,
    DISTANCE_DEPENDENT,
    SENDER_TYPE_SPECIFIC,
    TRANSITION_EDGE_SPECIFIC,
    WRONG_CONTEXT_NEGATIVE_CONTROL,
)

#: Stable nonzero default scale for transition edges beyond the first two.
_DEFAULT_EXTRA_EDGE_SCALE = 0.5


@dataclass(frozen=True)
class SyntheticMechanismSpec:
    """A pure specification of one synthetic mechanism (no model code)."""

    scenario_id: str
    direct_drift_strength: float
    direct_growth_strength: float
    regulatory_drift_strength: float
    regulatory_growth_strength: float
    context_to_regulatory_strength: float
    distance_dependent: bool
    sender_type_effect_scales: tuple[float, ...]
    transition_edge_effect_scales: tuple[float, ...]
    active_sender_context_type_ids: tuple[int, ...]
    negative_control_sender_context_type_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.scenario_id not in SYNTHETIC_SCENARIO_IDS:
            raise ValueError(f"unsupported scenario_id '{self.scenario_id}'")
        for name in (
            "direct_drift_strength", "direct_growth_strength",
            "regulatory_drift_strength", "regulatory_growth_strength",
            "context_to_regulatory_strength",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        object.__setattr__(
            self, "sender_type_effect_scales", tuple(self.sender_type_effect_scales)
        )
        object.__setattr__(
            self, "transition_edge_effect_scales",
            tuple(self.transition_edge_effect_scales),
        )
        object.__setattr__(
            self, "active_sender_context_type_ids",
            tuple(self.active_sender_context_type_ids),
        )
        object.__setattr__(
            self, "negative_control_sender_context_type_ids",
            tuple(self.negative_control_sender_context_type_ids),
        )
        active = set(self.active_sender_context_type_ids)
        negctl = set(self.negative_control_sender_context_type_ids)
        if active & negctl:
            raise ValueError("active and negative-control type sets must be disjoint")

    def validate_against_system(self, system: SyntheticSystemConfig) -> None:
        """Validate scale counts and id ranges against the system config."""
        n_types = system.num_sender_context_types
        n_edges = system.num_transition_edges
        if len(self.sender_type_effect_scales) != n_types:
            raise ValueError(
                f"sender_type_effect_scales has {len(self.sender_type_effect_scales)} "
                f"entries, expected {n_types}"
            )
        if len(self.transition_edge_effect_scales) != n_edges:
            raise ValueError(
                f"transition_edge_effect_scales has "
                f"{len(self.transition_edge_effect_scales)} entries, expected {n_edges}"
            )
        for tid in (*self.active_sender_context_type_ids,
                    *self.negative_control_sender_context_type_ids):
            if not (0 <= tid < n_types):
                raise ValueError(f"sender-context type id {tid} out of range")


def _default_sender_scales(system: SyntheticSystemConfig, first: tuple[float, ...]) -> tuple[float, ...]:
    """Extend a leading scale tuple with zeros to the full type count."""
    scales = list(first)
    while len(scales) < system.num_sender_context_types:
        scales.append(0.0)
    return tuple(scales[: system.num_sender_context_types])


def _default_edge_scales(system: SyntheticSystemConfig, first: tuple[float, ...]) -> tuple[float, ...]:
    """Extend a leading edge-scale tuple with a stable nonzero default."""
    scales = list(first)
    while len(scales) < system.num_transition_edges:
        scales.append(_DEFAULT_EXTRA_EDGE_SCALE)
    return tuple(scales[: system.num_transition_edges])


def build_synthetic_mechanism_spec(
    scenario_id: str, *, system: SyntheticSystemConfig
) -> SyntheticMechanismSpec:
    """Build the canonical mechanism spec for a scenario id."""
    if scenario_id not in SYNTHETIC_SCENARIO_IDS:
        raise ValueError(f"unsupported scenario_id '{scenario_id}'")

    cs = system.context_strength
    # Defaults shared by most scenarios: all sender types active (scale 1), all
    # edges active (scale 1), no distance decay, no regulatory pathway.
    all_ones_types = _default_sender_scales(system, (1.0,) * system.num_sender_context_types)
    all_ones_edges = _default_edge_scales(system, (1.0,) * system.num_transition_edges)

    def spec(**kw) -> SyntheticMechanismSpec:
        base = dict(
            scenario_id=scenario_id,
            direct_drift_strength=0.0,
            direct_growth_strength=0.0,
            regulatory_drift_strength=0.0,
            regulatory_growth_strength=0.0,
            context_to_regulatory_strength=0.0,
            distance_dependent=False,
            sender_type_effect_scales=all_ones_types,
            transition_edge_effect_scales=all_ones_edges,
            active_sender_context_type_ids=(),
            negative_control_sender_context_type_ids=(),
        )
        base.update(kw)
        s = SyntheticMechanismSpec(**base)
        s.validate_against_system(system)
        return s

    if scenario_id == NULL_CONTEXT:
        return spec()  # all context paths zero
    if scenario_id == DRIFT_ONLY:
        return spec(direct_drift_strength=cs)
    if scenario_id == GROWTH_ONLY:
        return spec(direct_growth_strength=cs)
    if scenario_id == MIXED_DRIFT_GROWTH:
        return spec(direct_drift_strength=cs, direct_growth_strength=cs)
    if scenario_id == REGULATORY_MEDIATED:
        return spec(
            regulatory_drift_strength=cs,
            regulatory_growth_strength=cs,
            context_to_regulatory_strength=1.0,
        )
    if scenario_id == DISTANCE_DEPENDENT:
        return spec(direct_drift_strength=cs, distance_dependent=True)
    if scenario_id == SENDER_TYPE_SPECIFIC:
        return spec(
            direct_drift_strength=cs,
            direct_growth_strength=cs,
            sender_type_effect_scales=_default_sender_scales(system, (1.0, -0.60, 0.0)),
        )
    if scenario_id == TRANSITION_EDGE_SPECIFIC:
        return spec(
            direct_drift_strength=cs,
            direct_growth_strength=cs,
            transition_edge_effect_scales=_default_edge_scales(system, (1.0, -0.75)),
        )
    if scenario_id == WRONG_CONTEXT_NEGATIVE_CONTROL:
        return spec(
            direct_drift_strength=cs,
            sender_type_effect_scales=_default_sender_scales(system, (1.0, 0.0, 0.0)),
            active_sender_context_type_ids=(0,),
            negative_control_sender_context_type_ids=(2,),
        )
    raise ValueError(f"unsupported scenario_id '{scenario_id}'")  # pragma: no cover
