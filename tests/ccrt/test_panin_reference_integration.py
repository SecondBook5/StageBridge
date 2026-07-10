"""Optional real-source PanIN integration test.

Runs against the actual reference repository when ``CCRT_PANIN_SOURCE_ROOT`` is
set. If the variable is unset, the test is skipped. If it IS set but the required
biological data are unavailable, the test FAILS with an exact missing-data
blocker (an explicit integration request is never silently converted to a skip).
"""

from __future__ import annotations

import os

import pytest

from stagebridge.ccrt.adapters.panin import (
    audit_panin_source,
    build_reference_panin_adapter_config,
    validate_reference_source_audit,
)
from stagebridge.ccrt.contracts import CCRTValidationError

_SOURCE_ENV = "CCRT_PANIN_SOURCE_ROOT"


def test_reference_panin_source_integration():
    source_root = os.environ.get(_SOURCE_ENV)
    if not source_root:
        pytest.skip(f"{_SOURCE_ENV} not set; skipping real-source integration")

    # explicit integration request: audit the real repository
    audit = audit_panin_source(source_root)
    assert audit is not None

    config = build_reference_panin_adapter_config(source_root)

    # If the biological data are unavailable (the reference repo is code-only and
    # data live on GEO), validation MUST fail with an exact blocker — not skip.
    try:
        validate_reference_source_audit(audit, config)
    except CCRTValidationError as exc:
        pytest.fail(
            "PanIN real-source integration blocked: required data/columns are not "
            "available under the supplied CCRT_PANIN_SOURCE_ROOT. This is expected "
            "when Xenium data (GEO GSE267680) are not downloaded into "
            f"./data/xenium/. Blocker detail:\n{exc}"
        )

    # If we reach here, the source truly contains a usable Xenium layout: run the
    # adapter as far as locally available data permit.
    from stagebridge.ccrt.adapters.panin import adapt_reference_panin  # local import

    # A concrete spatial loader would be required here; absent one, report clearly.
    with pytest.raises(CCRTValidationError):
        adapt_reference_panin(config, spatial_loader=None)
