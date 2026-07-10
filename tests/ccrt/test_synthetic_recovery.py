"""Deterministic mechanism-recovery tests — the scientific validation of CCRT.

The teacher knows the mechanism; the student sees only observable synthetic data;
the hidden mechanism is used only after training to evaluate recovery. Thresholds
are grounded in measured recovery (see Milestone-7 tuning) and must not be
weakened without diagnosing the cause.

Benchmark runs are cached at module scope so each scenario trains once.
"""

from __future__ import annotations

import functools

import pytest

from stagebridge.ccrt.synthetic import (
    SyntheticBenchmarkConfig,
    SyntheticSystemConfig,
    run_synthetic_scenario_benchmark,
)

# Light config: sufficient for drift/growth/distance/edge recovery (~5s each).
LIGHT_SYS = SyntheticSystemConfig(seed=0)
LIGHT_BENCH = SyntheticBenchmarkConfig(seed=0, epochs=20, dtype="float64")

# Stronger config: sender-type ranking + negative control need more training
# signal to become selective (~30s each). This is a training-design requirement,
# not a threshold relaxation.
STRONG_SYS = SyntheticSystemConfig(
    seed=0, train_batches=12, batch_size=16, senders_per_receiver=6
)
STRONG_BENCH = SyntheticBenchmarkConfig(
    seed=0, epochs=60, dtype="float64", learning_rate=5e-3
)


@functools.lru_cache(maxsize=None)
def light(scenario_id):
    return run_synthetic_scenario_benchmark(
        system=LIGHT_SYS, scenario_id=scenario_id, benchmark=LIGHT_BENCH
    ).result


@functools.lru_cache(maxsize=None)
def strong(scenario_id):
    return run_synthetic_scenario_benchmark(
        system=STRONG_SYS, scenario_id=scenario_id, benchmark=STRONG_BENCH
    ).result


# 1. Mixed scenario: training improves and drift is recovered.
def test_mixed_scenario_recovers():
    r = light("mixed_drift_growth")
    assert r.best_training_loss < r.initial_test_loss
    assert r.drift_cosine > 0.0
    # recovery beats a zero-vector baseline: predicted context effect is nonzero
    # and cosine-aligned with truth.
    assert r.predicted_context_drift_norm > 0.0
    assert r.all_metrics_finite


# 2. Null vs active context: null predicted context effect materially smaller.
def test_null_vs_active_context():
    r_null = light("null_context")
    r_drift = light("drift_only")
    assert r_null.predicted_context_drift_norm < 0.60 * r_drift.predicted_context_drift_norm


# 3. Drift-only selectivity: drift recovered without a large false growth effect.
def test_drift_only_selectivity():
    r = light("drift_only")
    assert r.true_context_drift_norm > 0.0
    assert r.true_context_growth_norm == pytest.approx(0.0, abs=1e-6)
    # predicted drift effect exceeds predicted growth effect
    assert r.predicted_context_drift_norm > r.predicted_context_growth_norm


# 4. Growth-only selectivity: growth recovered; false drift stays modest.
def test_growth_only_selectivity():
    r = light("growth_only")
    r_drift = light("drift_only")
    assert r.true_context_growth_norm > 0.0
    assert r.true_context_drift_norm == pytest.approx(0.0, abs=1e-6)
    # false predicted context drift must be smaller than genuine active drift
    assert r.predicted_context_drift_norm < r_drift.predicted_context_drift_norm


# 5. Distance dependence: predicted vs true distance-response curve correlated.
def test_distance_dependence_recovered():
    r = light("distance_dependent")
    assert r.distance_response_correlation is not None
    assert r.distance_response_correlation >= 0.50


# 6. Sender-type ranking: top active type recovered.
def test_sender_type_ranking_recovered():
    r = strong("sender_type_specific")
    assert r.sender_type_top_effect_correct is True
    assert r.sender_type_rank_recovery is not None
    assert r.sender_type_rank_recovery > 0.0


# 7. Transition-edge specificity: edge contrast recovered.
def test_transition_edge_specificity_recovered():
    r = light("transition_edge_specific")
    assert r.transition_edge_contrast_cosine is not None
    assert r.transition_edge_contrast_cosine > 0.0


# 8. Wrong-context negative control: negative type assigned less effect.
def test_wrong_context_negative_control_rejected():
    r = strong("wrong_context_negative_control")
    assert r.negative_control_effect_ratio is not None
    assert r.negative_control_effect_ratio < 1.0
