"""Pytest configuration for StageBridge tests.

Contract tests are critical - gradient flow issues break training silently.
"""

from __future__ import annotations

import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "contract: marks test as a contract test (enforces invariants)",
    )
    config.addinivalue_line(
        "markers",
        "slow: marks test as slow (deselect with '-m \"not slow\"')",
    )


def pytest_collection_modifyitems(config, items):
    """Auto-mark gradient contract tests."""
    for item in items:
        if "gradient" in item.nodeid.lower() or "contract" in item.nodeid.lower():
            item.add_marker(pytest.mark.contract)
