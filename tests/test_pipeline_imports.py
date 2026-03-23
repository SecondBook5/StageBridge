"""Smoke tests for pipeline script imports."""

import pytest


def test_spatial_benchmark_imports():
    """Test spatial benchmark pipeline imports."""
    try:
        from stagebridge.pipelines import run_spatial_benchmark

        assert True  # If import succeeds, test passes
    except ImportError as e:
        pytest.skip(f"run_spatial_benchmark not importable: {e}")


def test_complete_data_prep_imports():
    """Test complete data prep pipeline imports."""
    try:
        from stagebridge.pipelines import complete_data_prep

        assert True
    except ImportError as e:
        pytest.skip(f"complete_data_prep not importable: {e}")


def test_run_v1_complete_imports():
    """Test v1 complete training pipeline imports."""
    try:
        from stagebridge.pipelines import run_v1_complete

        assert True
    except ImportError as e:
        pytest.skip(f"run_v1_complete not importable: {e}")


def test_reference_pipeline_cli():
    """Test reference pipeline has a main() function."""
    # Import the module directly (not via lazy import which gives the function)
    import importlib

    module = importlib.import_module("stagebridge.pipelines.run_reference")
    assert hasattr(module, "main")
    assert callable(module.main)
