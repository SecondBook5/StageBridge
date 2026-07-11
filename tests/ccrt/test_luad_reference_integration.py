"""Optional real-source LUAD integration test.

Runs against the actual local LUAD data tree when ``CCRT_LUAD_SOURCE_ROOT`` is
set. If the variable is unset, the test is skipped. If it IS set but the required
context/coordinate artifacts are unavailable, the test FAILS with an exact
missing-data blocker (an explicit integration request is never silently converted
to a skip).

The test is DESCRIPTIVE / STRUCTURAL only: it audits + structurally validates
what exists. It never asserts biological correctness — the dataset accuracy is
explicitly unverified (see docs/ccrt/luad/SOURCE_AUDIT.md).
"""

from __future__ import annotations

import os

import pytest

from stagebridge.ccrt.adapters.luad import (
    audit_luad_source,
    build_reference_luad_adapter_config,
    validate_reference_source_audit,
)
from stagebridge.ccrt.contracts import CCRTValidationError

_SOURCE_ENV = "CCRT_LUAD_SOURCE_ROOT"


def test_reference_luad_source_integration():
    source_root = os.environ.get(_SOURCE_ENV)
    if not source_root:
        pytest.skip(f"{_SOURCE_ENV} not set; skipping real-source integration")

    # explicit integration request: audit the real data tree (structure only)
    audit = audit_luad_source(source_root)
    assert audit is not None

    config = build_reference_luad_adapter_config(source_root)

    # If required context/coordinate artifacts are unavailable, structural
    # validation MUST fail with an exact blocker — not skip.
    try:
        validate_reference_source_audit(audit, config)
    except CCRTValidationError as exc:
        pytest.fail(
            "LUAD real-source integration blocked: required context/coordinate "
            "artifacts are not available under the supplied CCRT_LUAD_SOURCE_ROOT "
            "(e.g. resolved Visium micron coordinates, the tangram context backend, "
            "or the HLCA/evo feature parquets). Blocker detail:\n" + str(exc)
        )

    # Reaching here means the source structurally supports the adapter. Absent a
    # concrete spatial loader in this environment, the adapter reports clearly
    # rather than fabricating observations.
    from stagebridge.ccrt.adapters.luad import adapt_reference_luad  # local import

    with pytest.raises(CCRTValidationError):
        adapt_reference_luad(config, spatial_loader=None)
