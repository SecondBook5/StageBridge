"""Tests for synthetic configuration validation."""

from __future__ import annotations

import pytest

from stagebridge.ccrt.synthetic import SyntheticBenchmarkConfig, SyntheticSystemConfig


def test_valid_default_configs():
    SyntheticSystemConfig()
    SyntheticBenchmarkConfig()


def test_invalid_dimensions_fail():
    with pytest.raises(ValueError):
        SyntheticSystemConfig(receiver_dim=0)
    with pytest.raises(ValueError):
        SyntheticSystemConfig(semantic_dim=0)


def test_invalid_counts_fail():
    with pytest.raises(ValueError):
        SyntheticSystemConfig(num_sender_context_types=2)
    with pytest.raises(ValueError):
        SyntheticSystemConfig(num_transition_edges=1)
    with pytest.raises(ValueError):
        SyntheticSystemConfig(batch_size=1)


def test_invalid_probability_fails():
    with pytest.raises(ValueError):
        SyntheticSystemConfig(sender_mask_probability=1.0)
    with pytest.raises(ValueError):
        SyntheticSystemConfig(sender_mask_probability=-0.1)


def test_invalid_strengths_and_noise_fail():
    with pytest.raises(ValueError):
        SyntheticSystemConfig(context_strength=-0.1)
    with pytest.raises(ValueError):
        SyntheticSystemConfig(target_noise_std=-0.1)
    with pytest.raises(ValueError):
        SyntheticSystemConfig(max_distance=0.0)


def test_invalid_benchmark_settings_fail():
    with pytest.raises(ValueError):
        SyntheticBenchmarkConfig(epochs=0)
    with pytest.raises(ValueError):
        SyntheticBenchmarkConfig(hidden_dim=9, num_heads=2)  # not divisible
    with pytest.raises(ValueError):
        SyntheticBenchmarkConfig(growth_supervision_weight=0.0)
    with pytest.raises(ValueError):
        SyntheticBenchmarkConfig(dtype="float16")
    with pytest.raises(ValueError):
        SyntheticBenchmarkConfig(sinkhorn_epsilon=0.0)
