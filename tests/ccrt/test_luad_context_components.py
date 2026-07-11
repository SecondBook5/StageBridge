"""Tests for LUAD deconvolved typed-context components."""

from __future__ import annotations

import pytest

from stagebridge.ccrt.adapters.luad import (
    LUADContextBackendConfig,
    LUADContextComponent,
    load_luad_context_components,
    validate_luad_context_components,
)
from stagebridge.ccrt.contracts import CCRTValidationError

_SENDER_MAP = {"AT2": "at2", "Macrophages": "macrophage"}


def _backend(path, uncertainty_key=None):
    return LUADContextBackendConfig(
        backend_id="tangram", source_path=path.name, spot_id_key="spot_id",
        sender_context_type_key="sender_context_type", abundance_key="abundance",
        feature_ids=("f1", "f2"), uncertainty_key=uncertainty_key,
    )


def _write(path, header, rows):
    lines = [",".join(header)]
    for row in rows:
        lines.append(",".join(str(v) for v in row))
    path.write_text("\n".join(lines), encoding="utf-8")


def test_one_backend_spot_type_is_one_component(tmp_path):
    p = tmp_path / "c.csv"
    _write(
        p,
        ["spot_id", "sender_context_type", "abundance", "f1", "f2"],
        [
            ["s0", "AT2", "0.4", "0.1", "0.2"],
            ["s0", "Macrophages", "0.6", "0.3", "0.4"],
        ],
    )
    comps = load_luad_context_components(
        _backend(p), source_root=tmp_path, sender_context_annotation_map=_SENDER_MAP
    )
    assert len(comps) == 2
    ids = {c.component_id for c in comps}
    assert ids == {"tangram::s0::at2", "tangram::s0::macrophage"}


def test_no_uncertainty_uses_zero_with_provenance(tmp_path):
    p = tmp_path / "c.csv"
    _write(
        p, ["spot_id", "sender_context_type", "abundance", "f1", "f2"],
        [["s0", "AT2", "0.4", "0.1", "0.2"]],
    )
    comps = load_luad_context_components(
        _backend(p), source_root=tmp_path, sender_context_annotation_map=_SENDER_MAP
    )
    assert comps[0].uncertainty == 0.0
    assert comps[0].uncertainty_source == "not_provided"
    # abundance is NOT reused as uncertainty
    assert comps[0].abundance == pytest.approx(0.4)


def test_provided_uncertainty_used(tmp_path):
    p = tmp_path / "c.csv"
    _write(
        p, ["spot_id", "sender_context_type", "abundance", "unc", "f1", "f2"],
        [["s0", "AT2", "0.4", "0.05", "0.1", "0.2"]],
    )
    comps = load_luad_context_components(
        _backend(p, uncertainty_key="unc"), source_root=tmp_path,
        sender_context_annotation_map=_SENDER_MAP,
    )
    assert comps[0].uncertainty == pytest.approx(0.05)
    assert comps[0].uncertainty_source == "provided"


def test_unknown_label_fails_strict(tmp_path):
    p = tmp_path / "c.csv"
    _write(
        p, ["spot_id", "sender_context_type", "abundance", "f1", "f2"],
        [["s0", "Mystery", "0.4", "0.1", "0.2"]],
    )
    with pytest.raises(CCRTValidationError):
        load_luad_context_components(
            _backend(p), source_root=tmp_path, sender_context_annotation_map=_SENDER_MAP,
            strict_unknown_annotations=True,
        )


def test_unknown_label_skipped_when_not_strict(tmp_path):
    p = tmp_path / "c.csv"
    _write(
        p, ["spot_id", "sender_context_type", "abundance", "f1", "f2"],
        [["s0", "Mystery", "0.4", "0.1", "0.2"], ["s0", "AT2", "0.6", "0.3", "0.4"]],
    )
    comps = load_luad_context_components(
        _backend(p), source_root=tmp_path, sender_context_annotation_map=_SENDER_MAP,
        strict_unknown_annotations=False,
    )
    assert len(comps) == 1
    assert comps[0].sender_context_type_id == "at2"


def test_negative_abundance_rejected(tmp_path):
    p = tmp_path / "c.csv"
    _write(
        p, ["spot_id", "sender_context_type", "abundance", "f1", "f2"],
        [["s0", "AT2", "-0.4", "0.1", "0.2"]],
    )
    with pytest.raises(CCRTValidationError):
        load_luad_context_components(
            _backend(p), source_root=tmp_path, sender_context_annotation_map=_SENDER_MAP
        )


def test_validate_rejects_duplicate_components():
    comp = LUADContextComponent(
        component_id="tangram::s0::at2", backend_id="tangram", spot_id="s0",
        sender_context_type_id="at2", source_sender_label="AT2", abundance=0.4,
        uncertainty=0.0, uncertainty_source="not_provided", feature_vector=(0.1, 0.2),
    )
    with pytest.raises(CCRTValidationError):
        validate_luad_context_components([comp, comp])


def test_validate_rejects_nonzero_uncertainty_without_source():
    comp = LUADContextComponent(
        component_id="tangram::s0::at2", backend_id="tangram", spot_id="s0",
        sender_context_type_id="at2", source_sender_label="AT2", abundance=0.4,
        uncertainty=0.3, uncertainty_source="not_provided", feature_vector=(0.1, 0.2),
    )
    with pytest.raises(CCRTValidationError):
        validate_luad_context_components([comp])


def test_component_is_not_a_cell(tmp_path):
    # sanity: abundance never inflates the component into multiple pseudo-cells
    p = tmp_path / "c.csv"
    _write(
        p, ["spot_id", "sender_context_type", "abundance", "f1", "f2"],
        [["s0", "AT2", "5.0", "0.1", "0.2"]],
    )
    comps = load_luad_context_components(
        _backend(p), source_root=tmp_path, sender_context_annotation_map=_SENDER_MAP
    )
    assert len(comps) == 1  # one component regardless of abundance magnitude
