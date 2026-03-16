"""Optuna-based tuning for the StageBridge Set Transformer branch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np
import optuna
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from stagebridge.pipelines.run_context_model import run_context_model
from stagebridge.pipelines.run_evaluation import run_evaluation
from stagebridge.pipelines.run_reference import run_reference
from stagebridge.pipelines.run_spatial_mapping import run_spatial_mapping
from stagebridge.pipelines.run_transition_model import run_transition_model


EDGE_LIST = ["AAH->AIS", "AIS->MIA"]
BASELINE_MODES = ["rna_only", "pooled", "deep_sets", "graph_of_sets"]
TRANSFORMER_CORE_MODES = [
    "deep_sets",
    "set_only",
    "graph_of_sets",
    "typed_hierarchical_transformer",
    "deep_sets_transformer_hybrid",
]


@dataclass(slots=True, frozen=True)
class TrialMetrics:
    sinkhorn_mean: float
    calibration_mean: float
    instability_penalty: float
    deep_sets_penalty: float
    objective: float
    edge_rows: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sinkhorn_mean": self.sinkhorn_mean,
            "calibration_mean": self.calibration_mean,
            "instability_penalty": self.instability_penalty,
            "deep_sets_penalty": self.deep_sets_penalty,
            "objective": self.objective,
            "edge_rows": self.edge_rows,
        }


def _clone_cfg(cfg: DictConfig) -> DictConfig:
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))


def prepare_transformer_tuning_config(cfg: DictConfig) -> DictConfig:
    tuned = _clone_cfg(cfg)
    tuned.context_model.mode = "set_only"
    tuned.context_model.graph_enabled = False
    tuned.transition_model.wes_regularizer.enabled = False
    tuned.transition_model.schrodinger_bridge.sigma = 0.0
    tuned.train.profile = "medium"
    if hasattr(tuned, "profiles") and tuned.profiles is not None:
        tuned.profiles.train = "medium"
    return tuned


def suggest_set_only_hyperparameters(trial: optuna.trial.Trial) -> dict[str, Any]:
    return {
        "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 96, 128, 160]),
        "num_heads": trial.suggest_categorical("num_heads", [2, 4, 8]),
        "num_inducing_points": trial.suggest_categorical("num_inducing_points", [8, 12, 16, 24]),
        "num_seed_vectors": trial.suggest_categorical("num_seed_vectors", [2, 4]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.25),
        "token_dropout_rate": trial.suggest_float("token_dropout_rate", 0.0, 0.2),
        "auxiliary_context_shuffle_weight": trial.suggest_float(
            "auxiliary_context_shuffle_weight", 0.1, 0.3
        ),
        "learning_rate": trial.suggest_float("learning_rate", 3e-4, 3e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "max_context_spots": trial.suggest_categorical("max_context_spots", [32, 64, 96, 128]),
        "batch_cells": trial.suggest_categorical("batch_cells", [32, 48, 64]),
        "steps_per_epoch": trial.suggest_categorical("steps_per_epoch", [8, 12, 16, 24]),
        "max_epochs": trial.suggest_categorical("max_epochs", [6, 8, 12, 16]),
        "use_ude": True,
        "use_cross_attention_drift": True,
    }


def apply_transformer_hyperparameters(cfg: DictConfig, params: Mapping[str, Any]) -> DictConfig:
    tuned = _clone_cfg(cfg)
    tuned.context_model.hidden_dim = int(params["hidden_dim"])
    tuned.context_model.num_heads = int(params["num_heads"])
    tuned.context_model.num_inducing_points = int(params["num_inducing_points"])
    tuned.context_model.max_context_spots = int(params["max_context_spots"])
    tuned.context_model.token_dropout_rate = float(params["token_dropout_rate"])
    tuned.context_model.auxiliary_context_shuffle_weight = float(
        params["auxiliary_context_shuffle_weight"]
    )
    if "num_seed_vectors" in params:
        tuned.context_model.num_seed_vectors = int(params["num_seed_vectors"])
    if "use_cross_attention_drift" in params:
        tuned.context_model.use_cross_attention_drift = bool(params["use_cross_attention_drift"])
    if "use_ude" in params:
        tuned.transition_model.stochastic_dynamics.use_ude = bool(params["use_ude"])
    tuned.transition_model.stochastic_dynamics.dropout = float(params["dropout"])
    tuned.train.learning_rate = float(params["learning_rate"])
    tuned.train.weight_decay = float(params["weight_decay"])
    tuned.train.batch_cells = int(params["batch_cells"])
    tuned.train.steps_per_epoch = int(params["steps_per_epoch"])
    tuned.train.max_epochs = int(params["max_epochs"])
    return tuned


def _run_edge_mode(
    cfg: DictConfig,
    *,
    edge: str,
    mode: str,
    seed: int,
    reference_output: dict[str, Any] | None = None,
    spatial_output: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg_run = _clone_cfg(cfg)
    src, tgt = [part.strip() for part in edge.split("->", 1)]
    cfg_run.seed = int(seed)
    cfg_run.train.seed = int(seed)
    cfg_run.context_model.mode = str(mode)
    cfg_run.context_model.graph_enabled = bool(mode == "graph_of_sets")
    cfg_run.transition_model.active_edge = [src, tgt]

    reference = reference_output or run_reference(cfg_run)
    spatial = spatial_output or run_spatial_mapping(cfg_run, reference_output=reference)
    context = run_context_model(cfg_run, spatial_output=spatial)
    transition = run_transition_model(
        cfg_run,
        reference_output=reference,
        spatial_output=spatial,
        context_output=context,
    )
    evaluation = run_evaluation(
        cfg_run,
        transition_output=transition,
        context_output=context,
    )
    return {
        "reference": reference,
        "spatial_mapping": spatial,
        "context_model": context,
        "transition_model": transition,
        "evaluation": evaluation,
    }


def run_mode_baseline_summary(
    cfg: DictConfig,
    *,
    modes: list[str] | None = None,
    edges: list[str] | None = None,
    seeds: list[int] | None = None,
) -> pd.DataFrame:
    """Run matched baseline summaries for context modes used in transformer comparisons."""
    modes = modes or BASELINE_MODES
    edges = edges or EDGE_LIST
    seeds = seeds or [7, 13, 29]

    cfg_base = prepare_transformer_tuning_config(cfg)
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        reference = run_reference(_clone_cfg(cfg_base))
        spatial = run_spatial_mapping(_clone_cfg(cfg_base), reference_output=reference)
        for edge in edges:
            for mode in modes:
                result = _run_edge_mode(
                    cfg_base,
                    edge=edge,
                    mode=mode,
                    seed=int(seed),
                    reference_output=reference,
                    spatial_output=spatial,
                )
                evaluation = result["evaluation"]
                biology = evaluation.get("biology_summary") or {}
                rows.append(
                    {
                        "seed": int(seed),
                        "edge": edge,
                        "mode": mode,
                        "sinkhorn": float(evaluation["heldout_metrics"]["sinkhorn"]),
                        "calibration_error": float(
                            evaluation["calibration"]["mean_abs_shift_error"]
                        ),
                        "dominant_increase_group": biology.get("dominant_increase_group"),
                        "dominant_decrease_group": biology.get("dominant_decrease_group"),
                    }
                )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.groupby(["edge", "mode"], as_index=False).agg(
        sinkhorn_mean=("sinkhorn", "mean"),
        sinkhorn_std=("sinkhorn", "std"),
        calibration_mean=("calibration_error", "mean"),
        calibration_std=("calibration_error", "std"),
        dominant_increase_group=(
            "dominant_increase_group",
            lambda s: s.mode().iloc[0] if not s.mode().empty else None,
        ),
        dominant_decrease_group=(
            "dominant_decrease_group",
            lambda s: s.mode().iloc[0] if not s.mode().empty else None,
        ),
    )


def summarize_transformer_vs_deep_sets(
    benchmark_table: pd.DataFrame,
    *,
    transformer_mode: str = "typed_hierarchical_transformer",
) -> dict[str, Any]:
    """Build a strict decision summary against Deep Sets for the flagship transformer."""
    if benchmark_table.empty:
        return {
            "status": "inconclusive",
            "decision": "needs_more_data",
            "interpretation": "No benchmark rows were available.",
            "edge_results": [],
        }
    deep_sets = benchmark_table.loc[benchmark_table["mode"] == "deep_sets"].set_index("edge")
    transformer = benchmark_table.loc[benchmark_table["mode"] == transformer_mode].set_index(
        "edge"
    )
    rows: list[dict[str, Any]] = []
    wins_sinkhorn = []
    calibration_ok = []
    for edge in sorted(set(deep_sets.index).intersection(set(transformer.index))):
        deep_row = deep_sets.loc[edge]
        trans_row = transformer.loc[edge]
        sinkhorn_delta = float(trans_row["sinkhorn_mean"] - deep_row["sinkhorn_mean"])
        calibration_delta = float(trans_row["calibration_mean"] - deep_row["calibration_mean"])
        rows.append(
            {
                "edge": str(edge),
                "transformer_sinkhorn_mean": float(trans_row["sinkhorn_mean"]),
                "deep_sets_sinkhorn_mean": float(deep_row["sinkhorn_mean"]),
                "sinkhorn_delta": sinkhorn_delta,
                "transformer_calibration_mean": float(trans_row["calibration_mean"]),
                "deep_sets_calibration_mean": float(deep_row["calibration_mean"]),
                "calibration_delta": calibration_delta,
            }
        )
        wins_sinkhorn.append(sinkhorn_delta <= 0.0)
        calibration_ok.append(calibration_delta <= 0.25)

    if rows and all(wins_sinkhorn) and all(calibration_ok):
        status = "pass"
        decision = "keep"
        interpretation = f"{transformer_mode} beat Deep Sets on Sinkhorn for all benchmarked edges without a material calibration collapse."
    elif rows and any(wins_sinkhorn):
        status = "weak_pass"
        decision = "keep_as_optional"
        interpretation = f"{transformer_mode} improved at least one edge but did not clear a strict all-edge win over Deep Sets."
    else:
        status = "fail"
        decision = "demote"
        interpretation = (
            f"{transformer_mode} did not beat Deep Sets under the current fixed benchmark."
        )
    return {
        "status": status,
        "decision": decision,
        "interpretation": interpretation,
        "edge_results": rows,
    }


def run_transformer_core_benchmark(
    cfg: DictConfig,
    *,
    modes: list[str] | None = None,
    edges: list[str] | None = None,
    seeds: list[int] | None = None,
) -> dict[str, Any]:
    """Run the fixed transformer-core benchmark against Deep Sets and baselines."""
    modes = modes or TRANSFORMER_CORE_MODES
    edges = edges or EDGE_LIST
    seeds = seeds or [7, 13, 29]
    cfg_base = prepare_transformer_tuning_config(cfg)
    summary = run_mode_baseline_summary(
        cfg_base,
        modes=modes,
        edges=edges,
        seeds=seeds,
    )
    decision = summarize_transformer_vs_deep_sets(summary)
    return {
        "benchmark_table": summary,
        "transformer_decision": decision,
    }


def _deep_sets_reference_map(baseline_summary: pd.DataFrame) -> dict[str, dict[str, float]]:
    reference: dict[str, dict[str, float]] = {}
    if baseline_summary.empty:
        return reference
    deep_sets = baseline_summary.loc[baseline_summary["mode"] == "deep_sets"]
    for row in deep_sets.itertuples(index=False):
        reference[str(row.edge)] = {
            "sinkhorn_mean": float(row.sinkhorn_mean),
            "calibration_mean": float(row.calibration_mean),
        }
    return reference


def evaluate_set_only_candidate(
    cfg: DictConfig,
    *,
    params: Mapping[str, Any],
    edges: list[str] | None = None,
    seeds: list[int] | None = None,
    deep_sets_reference: Mapping[str, Mapping[str, float]] | None = None,
) -> TrialMetrics:
    """Evaluate one set-only parameterization over the requested edges and seeds."""
    edges = edges or EDGE_LIST
    seeds = seeds or [7, 13]
    deep_sets_reference = deep_sets_reference or {}

    cfg_trial = apply_transformer_hyperparameters(prepare_transformer_tuning_config(cfg), params)
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        reference = run_reference(_clone_cfg(cfg_trial))
        spatial = run_spatial_mapping(_clone_cfg(cfg_trial), reference_output=reference)
        for edge in edges:
            result = _run_edge_mode(
                cfg_trial,
                edge=edge,
                mode="set_only",
                seed=int(seed),
                reference_output=reference,
                spatial_output=spatial,
            )
            evaluation = result["evaluation"]
            biology = evaluation.get("biology_summary") or {}
            rows.append(
                {
                    "seed": int(seed),
                    "edge": edge,
                    "sinkhorn": float(evaluation["heldout_metrics"]["sinkhorn"]),
                    "calibration_error": float(evaluation["calibration"]["mean_abs_shift_error"]),
                    "dominant_increase_group": biology.get("dominant_increase_group"),
                    "dominant_decrease_group": biology.get("dominant_decrease_group"),
                }
            )
    frame = pd.DataFrame(rows)
    sinkhorn_mean = float(frame["sinkhorn"].mean())
    calibration_mean = float(frame["calibration_error"].mean())
    instability_penalty = float(frame.groupby("edge")["sinkhorn"].std(ddof=0).fillna(0.0).mean())

    deep_sets_penalties: list[float] = []
    edge_rows: list[dict[str, Any]] = []
    for edge, group in frame.groupby("edge"):
        edge_sinkhorn = float(group["sinkhorn"].mean())
        edge_calibration = float(group["calibration_error"].mean())
        baseline = deep_sets_reference.get(str(edge), {})
        baseline_sinkhorn = float(baseline.get("sinkhorn_mean", edge_sinkhorn))
        baseline_calibration = float(baseline.get("calibration_mean", edge_calibration))
        sinkhorn_ratio = edge_sinkhorn / max(baseline_sinkhorn, 1e-8)
        calibration_ratio = edge_calibration / max(baseline_calibration, 1e-8)
        penalty = max(0.0, sinkhorn_ratio - 1.0)
        deep_sets_penalties.append(penalty)
        edge_rows.append(
            {
                "edge": str(edge),
                "sinkhorn_mean": edge_sinkhorn,
                "calibration_mean": edge_calibration,
                "sinkhorn_ratio_vs_deep_sets": sinkhorn_ratio,
                "calibration_ratio_vs_deep_sets": calibration_ratio,
                "deep_sets_penalty": penalty,
            }
        )

    deep_sets_penalty = float(np.mean(deep_sets_penalties)) if deep_sets_penalties else 0.0
    objective = (
        0.50 * float(np.mean([row["sinkhorn_ratio_vs_deep_sets"] for row in edge_rows]))
        + 0.25 * float(np.mean([row["calibration_ratio_vs_deep_sets"] for row in edge_rows]))
        + 0.15 * instability_penalty
        + 0.10 * deep_sets_penalty
    )
    return TrialMetrics(
        sinkhorn_mean=sinkhorn_mean,
        calibration_mean=calibration_mean,
        instability_penalty=instability_penalty,
        deep_sets_penalty=deep_sets_penalty,
        objective=objective,
        edge_rows=edge_rows,
    )


def make_set_only_objective(
    cfg: DictConfig,
    *,
    edges: list[str] | None = None,
    seeds: list[int] | None = None,
    deep_sets_reference: Mapping[str, Mapping[str, float]] | None = None,
) -> Callable[[optuna.trial.Trial], float]:
    """Create a deterministic Optuna objective for the Set Transformer branch."""
    edges = edges or EDGE_LIST
    seeds = seeds or [7, 13]
    deep_sets_reference = deep_sets_reference or {}

    def objective(trial: optuna.trial.Trial) -> float:
        params = suggest_set_only_hyperparameters(trial)
        metrics = evaluate_set_only_candidate(
            cfg,
            params=params,
            edges=edges,
            seeds=seeds,
            deep_sets_reference=deep_sets_reference,
        )
        for idx, row in enumerate(metrics.edge_rows, start=1):
            trial.report(float(row["sinkhorn_mean"]), step=idx)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("metrics", metrics.to_dict())
        return float(metrics.objective)

    return objective


def build_optuna_trial_table(study: optuna.study.Study) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for trial in study.trials:
        metrics = trial.user_attrs.get("metrics", {})
        row = {
            "trial_number": int(trial.number),
            "state": str(trial.state),
            "objective": None if trial.value is None else float(trial.value),
            **trial.params,
        }
        if isinstance(metrics, Mapping):
            row.update(
                {
                    "sinkhorn_mean": metrics.get("sinkhorn_mean"),
                    "calibration_mean": metrics.get("calibration_mean"),
                    "instability_penalty": metrics.get("instability_penalty"),
                    "deep_sets_penalty": metrics.get("deep_sets_penalty"),
                }
            )
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["trial_number", "state", "objective"])
    return (
        pd.DataFrame(rows)
        .sort_values(["state", "objective"], na_position="last")
        .reset_index(drop=True)
    )


def build_optuna_figure_bundle(study: optuna.study.Study) -> dict[str, Any]:
    """Generate the standard Optuna matplotlib figures for notebook display."""
    from optuna.visualization import matplotlib as ovm

    figures: dict[str, Any] = {}
    try:
        figures["optimization_history"] = ovm.plot_optimization_history(study).figure
    except Exception:
        pass
    try:
        figures["param_importances"] = ovm.plot_param_importances(study).figure
    except Exception:
        pass
    try:
        figures["slice_plot"] = ovm.plot_slice(study).figure
    except Exception:
        pass
    try:
        figures["parallel_coordinate"] = ovm.plot_parallel_coordinate(study).figure
    except Exception:
        pass
    return figures


def run_set_only_optuna_study(
    cfg: DictConfig,
    *,
    n_trials: int = 12,
    edges: list[str] | None = None,
    search_seeds: list[int] | None = None,
    confirm_seeds: list[int] | None = None,
    study_name: str = "stagebridge_set_only_optuna",
) -> dict[str, Any]:
    """Run the full Set Transformer tuning workflow after provider selection."""
    edges = edges or EDGE_LIST
    search_seeds = search_seeds or [7, 13]
    confirm_seeds = confirm_seeds or [7, 13, 29]

    cfg_base = prepare_transformer_tuning_config(cfg)
    baseline_summary = run_mode_baseline_summary(
        cfg_base,
        modes=BASELINE_MODES,
        edges=edges,
        seeds=confirm_seeds,
    )
    deep_sets_reference = _deep_sets_reference_map(baseline_summary)
    objective = make_set_only_objective(
        cfg_base,
        edges=edges,
        seeds=search_seeds,
        deep_sets_reference=deep_sets_reference,
    )
    sampler = optuna.samplers.TPESampler(seed=int(cfg_base.seed))
    pruner = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1)
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
    )
    study.optimize(objective, n_trials=int(n_trials))

    best_params = study.best_trial.params
    confirmed_metrics = evaluate_set_only_candidate(
        cfg_base,
        params=best_params,
        edges=edges,
        seeds=confirm_seeds,
        deep_sets_reference=deep_sets_reference,
    )
    confirmation_rows = []
    for row in confirmed_metrics.edge_rows:
        confirmation_rows.append({"edge": row["edge"], "mode": "set_only_tuned", **row})
    tuned_confirmation = pd.DataFrame(confirmation_rows)

    keep_rule = all(
        float(row["sinkhorn_ratio_vs_deep_sets"]) <= 1.0 for row in confirmed_metrics.edge_rows
    )
    weak_rule = all(
        float(row["sinkhorn_ratio_vs_deep_sets"]) <= 1.05 for row in confirmed_metrics.edge_rows
    )
    if keep_rule:
        recommendation = "keep"
        interpretation = (
            "The tuned Set Transformer beat or matched Deep Sets on both prioritized edges."
        )
    elif weak_rule:
        recommendation = "keep_as_optional"
        interpretation = "The tuned Set Transformer remained close to Deep Sets, but did not separate decisively enough for a flagship claim."
    else:
        recommendation = "keep_as_optional"
        interpretation = "The tuned Set Transformer still did not beat Deep Sets cleanly, so it should remain optional."

    return {
        "study": study,
        "baseline_summary": baseline_summary,
        "trial_table": build_optuna_trial_table(study),
        "best_params": dict(best_params),
        "confirmed_metrics": confirmed_metrics.to_dict(),
        "confirmed_table": tuned_confirmation,
        "recommendation": recommendation,
        "interpretation": interpretation,
        "figure_bundle": build_optuna_figure_bundle(study),
    }
