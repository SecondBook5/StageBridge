"""Tests for PanIN source auditing."""

from __future__ import annotations

import pytest

from stagebridge.ccrt.adapters.panin import (
    audit_panin_source,
    build_reference_panin_adapter_config,
    validate_reference_source_audit,
)
from stagebridge.ccrt.contracts import CCRTValidationError

from _panin_fixtures import build_panin_source_fixture


def test_audit_missing_root_fails(tmp_path):
    with pytest.raises(CCRTValidationError):
        audit_panin_source(tmp_path / "does_not_exist")


def test_audit_code_only_reports_missing(tmp_path):
    # a tree with no manifest = code-only reality
    (tmp_path / "README.md").write_text("code only", encoding="utf-8")
    audit = audit_panin_source(tmp_path)
    assert audit.platforms == ()
    assert audit.missing_requirements  # non-empty blocker


def test_audit_with_manifest_reports_platforms(tmp_path):
    build_panin_source_fixture(tmp_path)
    audit = audit_panin_source(tmp_path)
    assert "xenium" in audit.platforms
    assert audit.observation_units["xenium"] == "cell"
    assert audit.coordinate_units["xenium"] == "microns"
    assert audit.repository_commit == "fixturecommit"
    assert "PanIN_carcinogeneisis" not in "".join(audit.missing_requirements)  # sanity


def test_audit_records_files_with_metadata(tmp_path):
    build_panin_source_fixture(tmp_path)
    audit = audit_panin_source(tmp_path)
    # small CSV/JSON fixtures get a sha256
    assert any(f.sha256 is not None for f in audit.files)
    assert all(f.size_bytes >= 0 for f in audit.files)


def test_validate_audit_passes_for_fixture(tmp_path):
    cfg = build_panin_source_fixture(tmp_path)
    audit = audit_panin_source(tmp_path)
    validate_reference_source_audit(audit, cfg)  # should not raise


def test_validate_audit_fails_when_platform_not_cell(tmp_path):
    cfg = build_panin_source_fixture(tmp_path)
    audit = audit_panin_source(tmp_path)
    import dataclasses
    bad = dataclasses.replace(audit, observation_units={"xenium": "spot"})
    with pytest.raises(CCRTValidationError):
        validate_reference_source_audit(bad, cfg)


def test_validate_audit_fails_when_units_not_microns(tmp_path):
    cfg = build_panin_source_fixture(tmp_path)
    audit = audit_panin_source(tmp_path)
    import dataclasses
    bad = dataclasses.replace(audit, coordinate_units={"xenium": "pixels"})
    with pytest.raises(CCRTValidationError):
        validate_reference_source_audit(bad, cfg)
