from __future__ import annotations

import matplotlib
import optuna
import pandas as pd
import pytest
from omegaconf import OmegaConf

# Skip entire module if scvi is not available (transitive dependency)
pytest.importorskip("scvi", reason="scvi required for transformer tuning tests")

from stagebridge.evaluation.transformer_tuning import (
    TrialMetrics,
    build_optuna_trial_table,
    make_set_only_objective,
    run_transformer_core_benchmark,
    run_set_only_optuna_study,
    summarize_transformer_vs_deep_sets,
)

matplotlib.use("Agg")


def _cfg() -> object:
    return OmegaConf.create(
        {
            "seed": 42,
            "train": {"profile": "medium", "seed": 42},
            "profiles": {"train": "medium"},
            "context_model": {"mode": "set_only", "graph_enabled": False},
            "transition_model": {
                "wes_regularizer": {"enabled": False},
                "schrodinger_bridge": {"sigma": 0.0},
                "stochastic_dynamics": {"dropout": 0.1},
            },
        }
    )


def test_set_only_objective_is_deterministic_for_fixed_trial(monkeypatch) -> None:
    def _fake_evaluate(cfg, *, params, edges=None, seeds=None, deep_sets_reference=None):
        objective = float(params["hidden_dim"]) / 1000.0 + float(params["dropout"])
        return TrialMetrics(
            sinkhorn_mean=10.0,
            calibration_mean=1.0,
            instability_penalty=0.1,
            deep_sets_penalty=0.0,
            objective=objective,
            edge_rows=[
                {
                    "edge": "AAH->AIS",
                    "sinkhorn_mean": 10.0,
                    "calibration_mean": 1.0,
                    "sinkhorn_ratio_vs_deep_sets": 0.99,
                    "calibration_ratio_vs_deep_sets": 0.98,
                    "deep_sets_penalty": 0.0,
                }
            ],
        )

    monkeypatch.setattr(
        "stagebridge.evaluation.transformer_tuning.evaluate_set_only_candidate", _fake_evaluate
    )

    objective = make_set_only_objective(
        _cfg(),
        edges=["AAH->AIS"],
        seeds=[7, 13],
        deep_sets_reference={"AAH->AIS": {"sinkhorn_mean": 10.0, "calibration_mean": 1.0}},
    )
    fixed = optuna.trial.FixedTrial(
        {
            "hidden_dim": 96,
            "num_heads": 4,
            "num_inducing_points": 12,
            "num_seed_vectors": 2,
            "dropout": 0.1,
            "token_dropout_rate": 0.05,
            "auxiliary_context_shuffle_weight": 0.1,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "max_context_spots": 64,
            "batch_cells": 32,
            "steps_per_epoch": 8,
            "max_epochs": 6,
        }
    )

    value_1 = objective(fixed)
    value_2 = objective(fixed)

    assert value_1 == value_2


def test_run_set_only_optuna_study_emits_trial_table_and_confirmation(monkeypatch) -> None:
    monkeypatch.setattr(
        "stagebridge.evaluation.transformer_tuning.run_mode_baseline_summary",
        lambda cfg, modes=None, edges=None, seeds=None: pd.DataFrame(
            [
                {
                    "edge": "AAH->AIS",
                    "mode": "deep_sets",
                    "sinkhorn_mean": 10.0,
                    "calibration_mean": 1.0,
                },
                {
                    "edge": "AIS->MIA",
                    "mode": "deep_sets",
                    "sinkhorn_mean": 11.0,
                    "calibration_mean": 1.2,
                },
                {
                    "edge": "AAH->AIS",
                    "mode": "rna_only",
                    "sinkhorn_mean": 10.4,
                    "calibration_mean": 1.2,
                },
                {
                    "edge": "AIS->MIA",
                    "mode": "rna_only",
                    "sinkhorn_mean": 11.5,
                    "calibration_mean": 1.3,
                },
            ]
        ),
    )

    def _fake_evaluate(cfg, *, params, edges=None, seeds=None, deep_sets_reference=None):
        hidden_dim = float(params["hidden_dim"])
        dropout = float(params["dropout"])
        objective = abs(hidden_dim - 96.0) / 1000.0 + dropout
        return TrialMetrics(
            sinkhorn_mean=9.8 + objective,
            calibration_mean=0.9 + dropout,
            instability_penalty=0.05,
            deep_sets_penalty=max(0.0, objective - 0.05),
            objective=objective,
            edge_rows=[
                {
                    "edge": "AAH->AIS",
                    "sinkhorn_mean": 9.8 + objective,
                    "calibration_mean": 0.9 + dropout,
                    "sinkhorn_ratio_vs_deep_sets": 0.98 + objective,
                    "calibration_ratio_vs_deep_sets": 0.95 + dropout,
                    "deep_sets_penalty": max(0.0, objective - 0.05),
                },
                {
                    "edge": "AIS->MIA",
                    "sinkhorn_mean": 10.7 + objective,
                    "calibration_mean": 1.0 + dropout,
                    "sinkhorn_ratio_vs_deep_sets": 0.97 + objective,
                    "calibration_ratio_vs_deep_sets": 0.93 + dropout,
                    "deep_sets_penalty": max(0.0, objective - 0.05),
                },
            ],
        )

    monkeypatch.setattr(
        "stagebridge.evaluation.transformer_tuning.evaluate_set_only_candidate", _fake_evaluate
    )

    output = run_set_only_optuna_study(
        _cfg(),
        n_trials=2,
        edges=["AAH->AIS", "AIS->MIA"],
        search_seeds=[7, 13],
        confirm_seeds=[7, 13, 29],
        study_name="test_stagebridge_set_only_optuna",
    )

    assert not output["trial_table"].empty
    assert {"hidden_dim", "dropout"} <= set(output["best_params"])
    assert not output["confirmed_table"].empty
    assert output["recommendation"] in {"keep", "keep_as_optional"}
    assert build_optuna_trial_table(output["study"]).shape[0] >= 2


def test_summarize_transformer_vs_deep_sets_reports_decision() -> None:
    benchmark = pd.DataFrame(
        [
            {
                "edge": "AAH->AIS",
                "mode": "deep_sets",
                "sinkhorn_mean": 10.0,
                "calibration_mean": 1.0,
            },
            {
                "edge": "AAH->AIS",
                "mode": "typed_hierarchical_transformer",
                "sinkhorn_mean": 9.8,
                "calibration_mean": 1.1,
            },
            {
                "edge": "AIS->MIA",
                "mode": "deep_sets",
                "sinkhorn_mean": 11.0,
                "calibration_mean": 1.2,
            },
            {
                "edge": "AIS->MIA",
                "mode": "typed_hierarchical_transformer",
                "sinkhorn_mean": 10.9,
                "calibration_mean": 1.3,
            },
        ]
    )

    payload = summarize_transformer_vs_deep_sets(benchmark)

    assert payload["status"] == "pass"
    assert payload["decision"] == "keep"
    assert len(payload["edge_results"]) == 2


def test_run_transformer_core_benchmark_uses_fixed_modes(monkeypatch) -> None:
    monkeypatch.setattr(
        "stagebridge.evaluation.transformer_tuning.run_mode_baseline_summary",
        lambda cfg, modes=None, edges=None, seeds=None: pd.DataFrame(
            [
                {
                    "edge": "AAH->AIS",
                    "mode": "deep_sets",
                    "sinkhorn_mean": 10.0,
                    "calibration_mean": 1.0,
                },
                {
                    "edge": "AAH->AIS",
                    "mode": "typed_hierarchical_transformer",
                    "sinkhorn_mean": 10.2,
                    "calibration_mean": 0.9,
                },
                {
                    "edge": "AIS->MIA",
                    "mode": "deep_sets",
                    "sinkhorn_mean": 11.0,
                    "calibration_mean": 1.2,
                },
                {
                    "edge": "AIS->MIA",
                    "mode": "typed_hierarchical_transformer",
                    "sinkhorn_mean": 11.3,
                    "calibration_mean": 1.1,
                },
            ]
        ),
    )

    payload = run_transformer_core_benchmark(_cfg())

    assert not payload["benchmark_table"].empty
    assert payload["transformer_decision"]["status"] in {"pass", "weak_pass", "fail"}
