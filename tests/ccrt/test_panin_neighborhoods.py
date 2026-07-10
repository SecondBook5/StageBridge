"""Tests for continuous PanIN spatial neighborhoods."""

from __future__ import annotations

import pytest

from stagebridge.ccrt.adapters.panin import (
    PanINNeighborhoodConfig,
    PanINSpatialObservation,
    build_continuous_sender_neighborhoods,
)


def recv(oid, sample, section, x, y, platform="xenium"):
    return PanINSpatialObservation(
        observation_id=oid, donor_id="d", sample_id=sample, section_id=section,
        platform=platform, observation_unit="cell", x_microns=x, y_microns=y,
        canonical_receiver_state_id="normal_duct",
    )


def sender(oid, sample, section, x, y, ctx="caf", platform="xenium"):
    return PanINSpatialObservation(
        observation_id=oid, donor_id="d", sample_id=sample, section_id=section,
        platform=platform, observation_unit="cell", x_microns=x, y_microns=y,
        canonical_context_type_id=ctx,
    )


CFG = PanINNeighborhoodConfig(
    max_neighbors=5, max_distance=None, distance_units="microns", coordinate_scale_to_microns=1.0
)


def test_exact_known_distances():
    r = recv("r0", "s1", "sec1", 0.0, 0.0)
    s = sender("s0", "s1", "sec1", 3.0, 4.0)  # distance 5
    out = build_continuous_sender_neighborhoods(receivers=[r], candidate_senders=[s], config=CFG)
    assert len(out) == 1
    assert out[0].distance_to_receiver == pytest.approx(5.0)
    assert out[0].sender_context_type_id == "caf"


def test_same_section_restriction():
    r = recv("r0", "s1", "sec1", 0.0, 0.0)
    s = sender("s0", "s1", "sec2", 1.0, 0.0)  # different section
    out = build_continuous_sender_neighborhoods(receivers=[r], candidate_senders=[s], config=CFG)
    assert out == ()


def test_same_platform_restriction():
    r = recv("r0", "s1", "sec1", 0.0, 0.0)
    s = sender("s0", "s1", "sec1", 1.0, 0.0, platform="visium")
    out = build_continuous_sender_neighborhoods(receivers=[r], candidate_senders=[s], config=CFG)
    assert out == ()


def test_self_exclusion():
    r = recv("r0", "s1", "sec1", 0.0, 0.0)
    s = sender("r0", "s1", "sec1", 1.0, 0.0)  # same id as receiver
    out = build_continuous_sender_neighborhoods(receivers=[r], candidate_senders=[s], config=CFG)
    assert out == ()


def test_deterministic_tie_ordering():
    r = recv("r0", "s1", "sec1", 0.0, 0.0)
    # two senders equidistant -> ordered by sender_id
    s_b = sender("sB", "s1", "sec1", 1.0, 0.0)
    s_a = sender("sA", "s1", "sec1", 0.0, 1.0)
    out = build_continuous_sender_neighborhoods(receivers=[r], candidate_senders=[s_b, s_a], config=CFG)
    assert [n.sender_id for n in out] == ["sA", "sB"]


def test_max_neighbor_truncation():
    r = recv("r0", "s1", "sec1", 0.0, 0.0)
    senders = [sender(f"s{i}", "s1", "sec1", float(i + 1), 0.0) for i in range(10)]
    cfg = PanINNeighborhoodConfig(max_neighbors=3, max_distance=None, distance_units="microns", coordinate_scale_to_microns=1.0)
    out = build_continuous_sender_neighborhoods(receivers=[r], candidate_senders=senders, config=cfg)
    assert len(out) == 3
    assert [n.sender_id for n in out] == ["s0", "s1", "s2"]  # nearest 3


def test_max_distance_truncation():
    r = recv("r0", "s1", "sec1", 0.0, 0.0)
    senders = [sender("near", "s1", "sec1", 1.0, 0.0), sender("far", "s1", "sec1", 100.0, 0.0)]
    cfg = PanINNeighborhoodConfig(max_neighbors=5, max_distance=10.0, distance_units="microns", coordinate_scale_to_microns=1.0)
    out = build_continuous_sender_neighborhoods(receivers=[r], candidate_senders=senders, config=cfg)
    assert [n.sender_id for n in out] == ["near"]


def test_zero_neighbor_receiver_supported():
    r = recv("r0", "s1", "sec1", 0.0, 0.0)
    out = build_continuous_sender_neighborhoods(receivers=[r], candidate_senders=[], config=CFG)
    assert out == ()  # no fabricated sender


def test_continuous_values_preserved():
    r = recv("r0", "s1", "sec1", 0.0, 0.0)
    s = sender("s0", "s1", "sec1", 1.3, 2.7)
    out = build_continuous_sender_neighborhoods(receivers=[r], candidate_senders=[s], config=CFG)
    import math
    assert out[0].distance_to_receiver == pytest.approx(math.hypot(1.3, 2.7))
