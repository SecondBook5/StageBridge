"""Tests for synthetic mechanism specifications (exact scenario isolation)."""

from __future__ import annotations

import pytest

from stagebridge.ccrt.synthetic import (
    SYNTHETIC_SCENARIO_IDS,
    SyntheticSystemConfig,
    build_synthetic_mechanism_spec,
)
from stagebridge.ccrt.synthetic.mechanisms import (
    DRIFT_ONLY,
    GROWTH_ONLY,
    MIXED_DRIFT_GROWTH,
    NULL_CONTEXT,
    REGULATORY_MEDIATED,
    DISTANCE_DEPENDENT,
    SENDER_TYPE_SPECIFIC,
    TRANSITION_EDGE_SPECIFIC,
    WRONG_CONTEXT_NEGATIVE_CONTROL,
)

SYS = SyntheticSystemConfig()


def spec(sid):
    return build_synthetic_mechanism_spec(sid, system=SYS)


def test_scenario_ids_order():
    assert SYNTHETIC_SCENARIO_IDS == (
        "null_context", "drift_only", "growth_only", "mixed_drift_growth",
        "regulatory_mediated", "distance_dependent", "sender_type_specific",
        "transition_edge_specific", "wrong_context_negative_control",
    )


def test_null_all_context_paths_zero():
    m = spec(NULL_CONTEXT)
    assert m.direct_drift_strength == 0.0
    assert m.direct_growth_strength == 0.0
    assert m.regulatory_drift_strength == 0.0
    assert m.regulatory_growth_strength == 0.0
    assert m.context_to_regulatory_strength == 0.0


def test_drift_only():
    m = spec(DRIFT_ONLY)
    assert m.direct_drift_strength == SYS.context_strength
    assert m.direct_growth_strength == 0.0
    assert m.regulatory_drift_strength == 0.0
    assert m.regulatory_growth_strength == 0.0


def test_growth_only():
    m = spec(GROWTH_ONLY)
    assert m.direct_drift_strength == 0.0
    assert m.direct_growth_strength == SYS.context_strength
    assert m.regulatory_drift_strength == 0.0


def test_mixed():
    m = spec(MIXED_DRIFT_GROWTH)
    assert m.direct_drift_strength == SYS.context_strength
    assert m.direct_growth_strength == SYS.context_strength


def test_regulatory_direct_zero_regulatory_active():
    m = spec(REGULATORY_MEDIATED)
    assert m.direct_drift_strength == 0.0
    assert m.direct_growth_strength == 0.0
    assert m.regulatory_drift_strength == SYS.context_strength
    assert m.regulatory_growth_strength == SYS.context_strength
    assert m.context_to_regulatory_strength == 1.0


def test_distance_enabled():
    m = spec(DISTANCE_DEPENDENT)
    assert m.distance_dependent is True
    assert m.direct_drift_strength == SYS.context_strength
    assert m.direct_growth_strength == 0.0


def test_sender_type_scales_exact():
    m = spec(SENDER_TYPE_SPECIFIC)
    assert m.sender_type_effect_scales[:3] == (1.0, -0.60, 0.0)


def test_edge_scales_exact():
    m = spec(TRANSITION_EDGE_SPECIFIC)
    assert m.transition_edge_effect_scales[:2] == (1.0, -0.75)


def test_wrong_context_ids():
    m = spec(WRONG_CONTEXT_NEGATIVE_CONTROL)
    assert m.active_sender_context_type_ids == (0,)
    assert m.negative_control_sender_context_type_ids == (2,)
    assert m.sender_type_effect_scales[:3] == (1.0, 0.0, 0.0)


def test_scale_counts_match_system():
    for sid in SYNTHETIC_SCENARIO_IDS:
        m = spec(sid)
        assert len(m.sender_type_effect_scales) == SYS.num_sender_context_types
        assert len(m.transition_edge_effect_scales) == SYS.num_transition_edges


def test_unsupported_scenario_fails():
    with pytest.raises(ValueError):
        build_synthetic_mechanism_spec("nonsense", system=SYS)


def test_active_negative_disjoint_enforced():
    from stagebridge.ccrt.synthetic.mechanisms import SyntheticMechanismSpec
    with pytest.raises(ValueError):
        SyntheticMechanismSpec(
            scenario_id=WRONG_CONTEXT_NEGATIVE_CONTROL,
            direct_drift_strength=0.1, direct_growth_strength=0.0,
            regulatory_drift_strength=0.0, regulatory_growth_strength=0.0,
            context_to_regulatory_strength=0.0, distance_dependent=False,
            sender_type_effect_scales=(1.0, 0.0, 0.0),
            transition_edge_effect_scales=(1.0, 1.0),
            active_sender_context_type_ids=(0,),
            negative_control_sender_context_type_ids=(0,),  # collides with active
        )
