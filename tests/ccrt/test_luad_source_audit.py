"""Tests for LUAD source auditing."""

from __future__ import annotations

import dataclasses

import pytest

from stagebridge.ccrt.adapters.luad import (
    audit_luad_source,
    validate_reference_source_audit,
)
from stagebridge.ccrt.contracts import CCRTValidationError

from _luad_fixtures import build_luad_source_fixture


def test_audit_missing_root_fails(tmp_path):
    with pytest.raises(CCRTValidationError):
        audit_luad_source(tmp_path / "does_not_exist")


def test_audit_without_manifest_reports_missing(tmp_path):
    (tmp_path / "README.md").write_text("no manifest", encoding="utf-8")
    audit = audit_luad_source(tmp_path)
    assert audit.platforms == ()
    assert audit.missing_requirements  # non-empty blocker


def test_audit_with_manifest_reports_platforms(tmp_path):
    build_luad_source_fixture(tmp_path)
    audit = audit_luad_source(tmp_path)
    assert "visium" in audit.platforms
    assert "snrna" in audit.platforms
    assert audit.observation_units["visium"] == "spot"
    assert audit.observation_units["snrna"] == "cell"
    assert audit.coordinate_units["visium"] == "microns"
    assert audit.context_backends == ("tangram",)
    assert audit.dataset_commit == "fixturecommit"


def test_audit_records_files_with_metadata(tmp_path):
    build_luad_source_fixture(tmp_path)
    audit = audit_luad_source(tmp_path)
    assert any(f.sha256 is not None for f in audit.files)
    assert all(f.size_bytes >= 0 for f in audit.files)


def test_audit_flags_smoke_check_artifacts(tmp_path):
    build_luad_source_fixture(tmp_path)
    (tmp_path / "processed").mkdir(exist_ok=True)
    (tmp_path / "processed" / "tangram_smoke_check").mkdir(parents=True, exist_ok=True)
    (tmp_path / "processed" / "tangram_smoke_check" / "scores_smoke.parquet").write_text(
        "x", encoding="utf-8"
    )
    audit = audit_luad_source(tmp_path)
    assert audit.smoke_check_artifacts  # detected
    assert any("smoke-check" in w for w in audit.warnings)


def test_validate_audit_passes_for_fixture(tmp_path):
    cfg = build_luad_source_fixture(tmp_path)
    audit = audit_luad_source(tmp_path)
    validate_reference_source_audit(audit, cfg)  # should not raise


def test_validate_audit_fails_when_visium_not_spot(tmp_path):
    cfg = build_luad_source_fixture(tmp_path)
    audit = audit_luad_source(tmp_path)
    bad = dataclasses.replace(audit, observation_units={"visium": "cell", "snrna": "cell"})
    with pytest.raises(CCRTValidationError):
        validate_reference_source_audit(bad, cfg)


def test_validate_audit_fails_when_units_not_microns(tmp_path):
    cfg = build_luad_source_fixture(tmp_path)
    audit = audit_luad_source(tmp_path)
    bad = dataclasses.replace(audit, coordinate_units={"visium": "pixels"})
    with pytest.raises(CCRTValidationError):
        validate_reference_source_audit(bad, cfg)


def test_validate_audit_fails_when_backend_absent(tmp_path):
    cfg = build_luad_source_fixture(tmp_path)
    audit = audit_luad_source(tmp_path)
    bad = dataclasses.replace(audit, context_backends=())
    with pytest.raises(CCRTValidationError):
        validate_reference_source_audit(bad, cfg)
