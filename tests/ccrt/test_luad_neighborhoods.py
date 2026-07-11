"""Tests for continuous LUAD spatial context neighborhoods."""

from __future__ import annotations

import math

import pytest

from stagebridge.ccrt.adapters.luad import (
    LUADContextComponent,
    LUADNeighborhoodConfig,
    LUADSpatialSpot,
    build_luad_context_neighborhoods,
)


def spot(spot_id, x, y, state="normal", donor="P3", sample="s", section="sec", platform="visium"):
    return LUADSpatialSpot(
        spot_id=spot_id, donor_id=donor, sample_id=sample, section_id=section,
        platform=platform, x_microns=x, y_microns=y, niche_id=f"n_{spot_id}",
        canonical_receiver_state_id=state,
    )


def component(spot_id, ctype="at2", backend="tangram", abundance=0.5, uncertainty=0.0):
    return LUADContextComponent(
        component_id=f"{backend}::{spot_id}::{ctype}", backend_id=backend, spot_id=spot_id,
        sender_context_type_id=ctype, source_sender_label=ctype.upper(),
        abundance=abundance, uncertainty=uncertainty, uncertainty_source="not_provided",
        feature_vector=(0.1, 0.2),
    )


CFG = LUADNeighborhoodConfig(
    max_neighbors=8, max_distance=None, distance_units="microns", coordinate_scale_to_microns=1.0
)


def test_same_spot_component_distance_zero():
    r = spot("s0", 0.0, 0.0)
    comps = [component("s0")]
    out = build_luad_context_neighborhoods(
        receivers=[r], context_components=comps, spots_by_id={"s0": r}, config=CFG
    )
    assert len(out) == 1
    assert out[0].distance_to_receiver == 0.0
    assert out[0].sender_context_type_id == "at2"


def test_exact_euclidean_microns():
    r = spot("s0", 0.0, 0.0)
    host = spot("s1", 3.0, 4.0, state=None)  # host spot at distance 5
    comps = [component("s1")]
    out = build_luad_context_neighborhoods(
        receivers=[r], context_components=comps, spots_by_id={"s0": r, "s1": host}, config=CFG
    )
    assert out[0].distance_to_receiver == pytest.approx(5.0)


def test_continuous_value_preserved():
    r = spot("s0", 0.0, 0.0)
    host = spot("s1", 1.3, 2.7, state=None)
    comps = [component("s1")]
    out = build_luad_context_neighborhoods(
        receivers=[r], context_components=comps, spots_by_id={"s0": r, "s1": host}, config=CFG
    )
    assert out[0].distance_to_receiver == pytest.approx(math.hypot(1.3, 2.7))


def test_same_section_restriction():
    r = spot("s0", 0.0, 0.0, section="secA")
    host = spot("s1", 1.0, 0.0, state=None, section="secB")
    comps = [component("s1")]
    out = build_luad_context_neighborhoods(
        receivers=[r], context_components=comps, spots_by_id={"s0": r, "s1": host}, config=CFG
    )
    assert out == ()


def test_same_donor_restriction():
    r = spot("s0", 0.0, 0.0, donor="P3")
    host = spot("s1", 1.0, 0.0, state=None, donor="P4")
    comps = [component("s1")]
    out = build_luad_context_neighborhoods(
        receivers=[r], context_components=comps, spots_by_id={"s0": r, "s1": host}, config=CFG
    )
    assert out == ()


def test_same_platform_restriction():
    r = spot("s0", 0.0, 0.0, platform="visium")
    host = spot("s1", 1.0, 0.0, state=None, platform="snrna")
    comps = [component("s1")]
    out = build_luad_context_neighborhoods(
        receivers=[r], context_components=comps, spots_by_id={"s0": r, "s1": host}, config=CFG
    )
    assert out == ()


def test_backend_local_only():
    # each partition is a single backend; distances are computed per backend
    r = spot("s0", 0.0, 0.0)
    comps = [component("s0", backend="tangram"), component("s0", ctype="macrophage", backend="tangram")]
    out = build_luad_context_neighborhoods(
        receivers=[r], context_components=comps, spots_by_id={"s0": r}, config=CFG
    )
    assert {c.backend_id for c in out} == {"tangram"}


def test_deterministic_sort_order():
    r = spot("s0", 0.0, 0.0)
    # two same-spot components -> tie on distance 0, ordered by (spot_id, type, comp_id)
    comps = [component("s0", ctype="macrophage"), component("s0", ctype="at2")]
    out = build_luad_context_neighborhoods(
        receivers=[r], context_components=comps, spots_by_id={"s0": r}, config=CFG
    )
    assert [c.sender_context_type_id for c in out] == ["at2", "macrophage"]


def test_max_distance_truncation():
    r = spot("s0", 0.0, 0.0)
    near = spot("s1", 1.0, 0.0, state=None)
    far = spot("s2", 100.0, 0.0, state=None)
    comps = [component("s1"), component("s2")]
    cfg = LUADNeighborhoodConfig(
        max_neighbors=8, max_distance=10.0, distance_units="microns", coordinate_scale_to_microns=1.0
    )
    out = build_luad_context_neighborhoods(
        receivers=[r], context_components=comps,
        spots_by_id={"s0": r, "s1": near, "s2": far}, config=cfg
    )
    assert [c.sender_spot_id for c in out] == ["s1"]


def test_max_neighbor_truncation():
    r = spot("s0", 0.0, 0.0)
    hosts = {f"h{i}": spot(f"h{i}", float(i + 1), 0.0, state=None) for i in range(6)}
    hosts["s0"] = r
    comps = [component(f"h{i}") for i in range(6)]
    cfg = LUADNeighborhoodConfig(
        max_neighbors=3, max_distance=None, distance_units="microns", coordinate_scale_to_microns=1.0
    )
    out = build_luad_context_neighborhoods(
        receivers=[r], context_components=comps, spots_by_id=hosts, config=cfg
    )
    assert len(out) == 3
    assert [c.sender_spot_id for c in out] == ["h0", "h1", "h2"]


def test_zero_context_receiver_supported():
    r = spot("s0", 0.0, 0.0)
    out = build_luad_context_neighborhoods(
        receivers=[r], context_components=[], spots_by_id={"s0": r}, config=CFG
    )
    assert out == ()  # empty-sender handled downstream; nothing fabricated
