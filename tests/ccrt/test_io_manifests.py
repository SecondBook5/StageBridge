"""Tests for the in-memory provenance manifest utility (io/manifests.py)."""

from __future__ import annotations

import pytest

from stagebridge.ccrt.contracts import (
    CCRTForbiddenFieldError,
    CCRTLeakageError,
    CCRTValidationError,
)
from stagebridge.ccrt.io import build_manifest, validate_manifest


def test_build_manifest_creates_valid_manifest():
    manifest = build_manifest(
        biological_system_id="sys_a",
        source_dataset="unit_test_dataset",
        table_names=["receivers", "sender_context"],
    )
    assert manifest["biological_system_id"] == "sys_a"
    assert manifest["source_dataset"] == "unit_test_dataset"
    assert manifest["table_names"] == ["receivers", "sender_context"]
    assert manifest["schema_version"] == "ccrt-0.1"
    assert manifest["created_by"] == "stagebridge.ccrt"


def test_validate_manifest_accepts_built_manifest():
    manifest = build_manifest(
        biological_system_id="sys_a",
        source_dataset="ds",
        table_names=["samples"],
        notes="a note",
        extra={"run_label": "milestone2"},
    )
    validate_manifest(manifest)
    assert manifest["notes"] == "a note"
    assert manifest["run_label"] == "milestone2"


def test_missing_biological_system_id_fails():
    with pytest.raises(CCRTValidationError):
        build_manifest(
            biological_system_id="",
            source_dataset="ds",
            table_names=["receivers"],
        )


def test_empty_table_names_fails():
    with pytest.raises(CCRTValidationError):
        build_manifest(
            biological_system_id="sys_a",
            source_dataset="ds",
            table_names=[],
        )


def test_unknown_table_name_fails():
    with pytest.raises(CCRTValidationError):
        build_manifest(
            biological_system_id="sys_a",
            source_dataset="ds",
            table_names=["receivers", "not_a_real_table"],
        )


def test_future_expression_key_fails():
    manifest = build_manifest(
        biological_system_id="sys_a",
        source_dataset="ds",
        table_names=["receivers"],
    )
    manifest["future_expression"] = "leak"
    with pytest.raises(CCRTLeakageError):
        validate_manifest(manifest)


def test_ring_id_key_fails():
    manifest = build_manifest(
        biological_system_id="sys_a",
        source_dataset="ds",
        table_names=["receivers"],
    )
    manifest["ring_id"] = 1
    with pytest.raises(CCRTForbiddenFieldError):
        validate_manifest(manifest)


def test_extra_forbidden_key_at_build_fails():
    with pytest.raises(CCRTForbiddenFieldError):
        build_manifest(
            biological_system_id="sys_a",
            source_dataset="ds",
            table_names=["receivers"],
            extra={"world_token": 1},
        )
