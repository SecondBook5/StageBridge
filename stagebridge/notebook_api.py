"""Notebook-facing orchestration API for the rebuilt StageBridge layout."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable
from collections import Counter

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from stagebridge.pipelines.run_context_model import run_context_model
from stagebridge.pipelines.run_evaluation import run_evaluation
from stagebridge.pipelines.run_full import run_full
from stagebridge.pipelines.run_reference import run_reference
from stagebridge.pipelines.run_spatial_mapping import run_spatial_mapping
from stagebridge.pipelines.run_transition_model import run_transition_model
from stagebridge.utils.config_loader import load_yaml_config

_CONFIG_DIR = (Path(__file__).resolve().parent.parent / "configs").resolve()
_DEFAULT_CONFIG = _CONFIG_DIR / "default.yaml"
_COMPONENT_DIRS = {
    "data": _CONFIG_DIR / "data",
    "spatial_mapping": _CONFIG_DIR / "spatial_mapping",
    "context_model": _CONFIG_DIR / "context_model",
    "splits": _CONFIG_DIR / "splits",
    "train": _CONFIG_DIR / "train",
    "evaluation": _CONFIG_DIR / "evaluation",
}
_TRANSITION_COMPONENTS = [
    _CONFIG_DIR / "transition_model" / "disease_edges.yaml",
    _CONFIG_DIR / "transition_model" / "gaussian_init.yaml",
    _CONFIG_DIR / "transition_model" / "stochastic_dynamics.yaml",
    _CONFIG_DIR / "transition_model" / "schrodinger_bridge.yaml",
    _CONFIG_DIR / "transition_model" / "wes_regularizer.yaml",
]
_ENTRYPOINT_ALIASES = {
    "config": "default",
    "train": "default",
    "eval": "default",
}

StepFn = Callable[..., dict[str, Any]]


def _load_component(path: Path) -> DictConfig:
    if not path.exists():
        raise FileNotFoundError(f"Missing config component: {path}")
    return OmegaConf.create(load_yaml_config(path))


def _selected_profiles(base_cfg: DictConfig) -> dict[str, str]:
    defaults = {
        "data": "luad_evo",
        "spatial_mapping": "tangram",
        "context_model": "set_only",
        "splits": "donor_holdout",
        "train": "full_v1",
        "evaluation": "baseline",
    }
    profiles = OmegaConf.to_container(base_cfg.get("profiles", {}), resolve=False) or {}
    return {key: str(profiles.get(key, value)) for key, value in defaults.items()}


def compose_config(entrypoint: str = "default", overrides: list[str] | None = None) -> DictConfig:
    """Compose a config tree from the normalized repo layout."""
    entrypoint = _ENTRYPOINT_ALIASES.get(entrypoint, entrypoint)
    if entrypoint != "default":
        raise ValueError(f"Unsupported config entrypoint '{entrypoint}'. Use 'default'.")

    overrides = list(overrides or [])
    cfg = _load_component(_DEFAULT_CONFIG)
    selected = _selected_profiles(cfg)

    if entrypoint == "default":
        pass

    if "train=smoke" not in overrides and "evaluation=baseline" not in overrides:
        if entrypoint == "train":
            selected["train"] = "full_v1"
        if entrypoint == "eval":
            selected["evaluation"] = "baseline"

    dotlist_overrides: list[str] = []
    for override in overrides:
        if "=" not in override:
            dotlist_overrides.append(override)
            continue
        key, value = override.split("=", 1)
        if "." not in key and key in _COMPONENT_DIRS:
            selected[key] = value
            continue
        dotlist_overrides.append(override)

    for group, profile in selected.items():
        cfg = OmegaConf.merge(cfg, _load_component(_COMPONENT_DIRS[group] / f"{profile}.yaml"))
    for component_path in _TRANSITION_COMPONENTS:
        cfg = OmegaConf.merge(cfg, _load_component(component_path))
    if dotlist_overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(dotlist_overrides))
    return cfg


def clone_config(cfg: DictConfig) -> DictConfig:
    """Create a mutable clone of a composed config."""
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))


_STEP_REGISTRY: dict[str, StepFn] = {
    "reference": run_reference,
    "spatial_mapping": run_spatial_mapping,
    "context_model": run_context_model,
    "transition_model": run_transition_model,
    "evaluation": run_evaluation,
    "full": run_full,
    # compatibility aliases retained at the API boundary only
    "build_snrna": run_reference,
    "build_spatial": run_reference,
    "map_hlca": run_reference,
    "run_tangram": run_spatial_mapping,
    "train": run_transition_model,
    "evaluate": run_evaluation,
}


def run_step(step: str, cfg: DictConfig) -> dict[str, Any]:
    """Run one pipeline step from the rebuilt pipeline namespace."""
    fn = _STEP_REGISTRY.get(step)
    if fn is None:
        valid = ", ".join(sorted(_STEP_REGISTRY))
        raise ValueError(f"Unknown step '{step}'. Valid steps: {valid}")
    return fn(cfg)


def run_pipeline(cfg: DictConfig, steps: list[str] | None = None) -> dict[str, dict[str, Any]]:
    """Run the default StageBridge v1 orchestration order."""
    ordered_steps = steps or [
        "reference",
        "spatial_mapping",
        "context_model",
        "transition_model",
        "evaluation",
    ]
    outputs: dict[str, dict[str, Any]] = {}
    for step in ordered_steps:
        if step == "reference":
            outputs[step] = run_reference(cfg)
        elif step == "spatial_mapping":
            outputs[step] = run_spatial_mapping(cfg, reference_output=outputs.get("reference"))
        elif step == "context_model":
            outputs[step] = run_context_model(cfg, spatial_output=outputs.get("spatial_mapping"))
        elif step == "transition_model":
            outputs[step] = run_transition_model(
                cfg,
                reference_output=outputs.get("reference"),
                spatial_output=outputs.get("spatial_mapping"),
                context_output=outputs.get("context_model"),
            )
        elif step == "evaluation":
            outputs[step] = run_evaluation(
                cfg,
                transition_output=outputs.get("transition_model"),
                context_output=outputs.get("context_model"),
            )
        else:
            outputs[step] = run_step(step, cfg)
    return outputs


def build_step_status_table(pipeline_output: dict[str, Any]) -> pd.DataFrame:
    """Build a compact step-status table for notebook display."""
    rows: list[dict[str, Any]] = []
    for step_name, payload in pipeline_output.get("steps", pipeline_output).items():
        rows.append(
            {
                "step": step_name,
                "ok": bool(payload.get("ok", False)),
                "status": payload.get("status", "n/a"),
                "pipeline": payload.get("pipeline", step_name),
            }
        )
    return pd.DataFrame(rows)


def build_reference_summary_table(reference_output: dict[str, Any]) -> pd.DataFrame:
    """Summarize the active reference-latent branch for notebook display."""
    reference = reference_output.get("reference", {})
    diagnostics = reference.get("diagnostics", {})
    donor = diagnostics.get("donor_leakage", {})
    label_transfer = reference.get("label_transfer", {})
    shape = tuple(reference.get("latent_shape", [0, 0]))
    rows = [
        {"metric": "latent_backend", "value": reference.get("backend_name", "n/a")},
        {"metric": "provenance_mode", "value": reference.get("provenance", {}).get("mode", "n/a")},
        {"metric": "reference_source", "value": reference.get("source_path", "n/a")},
        {"metric": "latent_n_cells", "value": int(shape[0]) if len(shape) > 0 else 0},
        {"metric": "latent_dim", "value": int(shape[1]) if len(shape) > 1 else 0},
        {"metric": "stage_count", "value": diagnostics.get("stage_preservation", {}).get("n_stages", 0)},
        {"metric": "donor_leakage_accuracy", "value": donor.get("logreg_accuracy", float("nan"))},
        {"metric": "donor_chance_accuracy", "value": donor.get("chance_accuracy", float("nan"))},
        {"metric": "label_coverage", "value": label_transfer.get("coverage", 0.0)},
    ]
    return pd.DataFrame(rows)


def build_reference_label_table(reference_output: dict[str, Any]) -> pd.DataFrame:
    """Expose the top transferred reference labels."""
    label_transfer = reference_output.get("reference", {}).get("label_transfer", {})
    top_labels = label_transfer.get("top_labels", [])
    if not top_labels:
        return pd.DataFrame(columns=["label", "count"])
    return pd.DataFrame(top_labels, columns=["label", "count"])


def build_spatial_summary_table(spatial_output: dict[str, Any]) -> pd.DataFrame:
    """Summarize the active spatial mapping provider for notebook display."""
    summary = spatial_output.get("spatial_mapping", {})
    qc = summary.get("qc", {})
    rows = [
        {"metric": "method", "value": summary.get("method", "n/a")},
        {"metric": "status", "value": summary.get("status", "n/a")},
        {"metric": "n_spots", "value": summary.get("n_spots", 0)},
        {"metric": "n_features", "value": summary.get("n_features", 0)},
        {"metric": "mean_row_sum", "value": qc.get("mean_row_sum", float("nan"))},
        {"metric": "mean_max_assignment", "value": qc.get("mean_max_assignment", float("nan"))},
        {"metric": "source_path", "value": summary.get("source_path", "n/a")},
    ]
    return pd.DataFrame(rows)


def build_context_summary_table(context_output: dict[str, Any]) -> pd.DataFrame:
    """Summarize the typed context branch for notebook display."""
    summary = context_output.get("context_model", {})
    token_summary = summary.get("typed_token_summary", {})
    rows = [
        {"metric": "mode", "value": summary.get("mode", "n/a")},
        {"metric": "spatial_mapping_method", "value": summary.get("spatial_mapping_method", "n/a")},
        {"metric": "n_token_rows", "value": token_summary.get("n_tokens", 0)},
        {"metric": "token_dim", "value": token_summary.get("token_dim", 0)},
    ]
    if "example_context_norm" in summary:
        rows.append({"metric": "context_norm", "value": summary.get("example_context_norm", float("nan"))})
        rows.append({"metric": "context_dim", "value": summary.get("example_context_dim", 0)})
    if "graph_context_norm" in summary:
        rows.append({"metric": "graph_context_norm", "value": summary.get("graph_context_norm", float("nan"))})
        rows.append({"metric": "graph_num_nodes", "value": summary.get("graph_num_nodes", 0)})
        rows.append({"metric": "graph_num_edges", "value": summary.get("graph_num_edges", 0)})
    return pd.DataFrame(rows)


def build_transition_summary_table(
    transition_output: dict[str, Any],
    evaluation_output: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Summarize one transition run and its evaluation diagnostics."""
    split = transition_output.get("split_summary", {})
    wes = transition_output.get("wes_diagnostics", {})
    calibration = {} if evaluation_output is None else evaluation_output.get("calibration", {})
    heldout = {} if evaluation_output is None else evaluation_output.get("heldout_metrics", {})
    rows = [
        {"metric": "edge", "value": transition_output.get("edge", "n/a")},
        {"metric": "mode", "value": transition_output.get("mode", "n/a")},
        {"metric": "sigma", "value": transition_output.get("sigma", float("nan"))},
        {"metric": "diffusion_weight", "value": transition_output.get("diffusion_weight", float("nan"))},
        {"metric": "split_strategy", "value": split.get("split_strategy", "n/a")},
        {"metric": "same_donor_overlap", "value": len(split.get("overlap_donors", []))},
        {"metric": "wes_enabled", "value": wes.get("enabled", False)},
        {"metric": "wes_penalty_mean", "value": wes.get("regularizer_mean_penalty", float("nan"))},
        {"metric": "heldout_sinkhorn", "value": heldout.get("sinkhorn", float("nan"))},
        {"metric": "heldout_auc", "value": heldout.get("classifier_auc", float("nan"))},
        {"metric": "calibration_error", "value": calibration.get("mean_abs_shift_error", float("nan"))},
    ]
    return pd.DataFrame(rows)


def build_biology_summary_table(evaluation_output: dict[str, Any]) -> pd.DataFrame:
    """Summarize edge-level biological insight signals for notebook display."""
    biology = evaluation_output.get("biology_summary") or {}
    if not biology:
        return pd.DataFrame(columns=["metric", "value"])
    rows = [
        {"metric": "edge", "value": biology.get("edge", "n/a")},
        {"metric": "dominant_increase_group", "value": biology.get("dominant_increase_group", "n/a")},
        {"metric": "dominant_decrease_group", "value": biology.get("dominant_decrease_group", "n/a")},
        {"metric": "split_strategy", "value": biology.get("split_strategy", "n/a")},
        {"metric": "n_overlap_donors", "value": len(biology.get("overlap_donors", []))},
        {"metric": "context_sensitivity_delta", "value": biology.get("context_sensitivity_delta", float("nan"))},
    ]
    return pd.DataFrame(rows)


def build_gate_ready_table(evaluation_output: dict[str, Any]) -> pd.DataFrame:
    """Expose the current evaluation outputs that feed scientific gates."""
    rows = [
        {"signal": "sinkhorn", "value": evaluation_output.get("heldout_metrics", {}).get("sinkhorn", float("nan"))},
        {
            "signal": "context_sensitivity_delta",
            "value": (evaluation_output.get("context_sensitivity", {}) or {}).get("context_sensitivity_delta", float("nan")),
        },
        {
            "signal": "mean_diffusion_scale",
            "value": evaluation_output.get("diffusion_diagnostics", {}).get("mean_diffusion_scale", float("nan")),
        },
        {
            "signal": "pseudotime_alignment",
            "value": evaluation_output.get("pseudotime_structure", {}).get("pseudotime_correlation", float("nan")),
        },
    ]
    return pd.DataFrame(rows)


def run_mode_ladder(
    cfg: DictConfig,
    *,
    modes: list[str] | None = None,
    edge: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Run matched context-mode comparisons while reusing upstream steps."""
    modes = modes or ["rna_only", "pooled", "deep_sets", "set_only", "graph_of_sets"]
    cfg_base = clone_config(cfg)
    if edge is not None:
        src, tgt = [part.strip() for part in str(edge).split("->", 1)]
        cfg_base.transition_model.active_edge = [src, tgt]

    reference = run_reference(cfg_base)
    spatial = run_spatial_mapping(cfg_base, reference_output=reference)
    results: dict[str, dict[str, Any]] = {}
    for mode in modes:
        cfg_mode = clone_config(cfg_base)
        cfg_mode.context_model.mode = mode
        cfg_mode.context_model.graph_enabled = bool(mode == "graph_of_sets")
        context = run_context_model(cfg_mode, spatial_output=spatial)
        transition = run_transition_model(
            cfg_mode,
            reference_output=reference,
            spatial_output=spatial,
            context_output=context,
        )
        evaluation = run_evaluation(cfg_mode, transition_output=transition, context_output=context)
        results[mode] = {
            "context_model": context,
            "transition_model": transition,
            "evaluation": evaluation,
        }
    return results


def build_mode_comparison_table(mode_results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Convert a matched mode ladder run into a compact comparison table."""
    rows: list[dict[str, Any]] = []
    for mode, payload in mode_results.items():
        evaluation = payload["evaluation"]
        rows.append(
            {
                "mode": mode,
                "sinkhorn": evaluation["heldout_metrics"]["sinkhorn"],
                "sinkhorn_delta": evaluation["heldout_metrics"]["sinkhorn_delta"],
                "classifier_auc": evaluation["heldout_metrics"]["classifier_auc"],
                "calibration_error": evaluation["calibration"]["mean_abs_shift_error"],
                "context_sensitivity_delta": (evaluation.get("context_sensitivity") or {}).get("context_sensitivity_delta"),
                "dominant_increase_group": (evaluation.get("biology_summary") or {}).get("dominant_increase_group"),
                "dominant_decrease_group": (evaluation.get("biology_summary") or {}).get("dominant_decrease_group"),
                "split_strategy": payload["transition_model"]["split_summary"]["split_strategy"],
            }
        )
    return pd.DataFrame(rows).sort_values("sinkhorn").reset_index(drop=True)


def run_seeded_mode_ladder(
    cfg: DictConfig,
    *,
    seeds: list[int],
    modes: list[str] | None = None,
    edge: str | None = None,
) -> dict[int, dict[str, dict[str, Any]]]:
    """Run a matched mode ladder over a deterministic seed grid."""
    seeded_results: dict[int, dict[str, dict[str, Any]]] = {}
    for seed in seeds:
        cfg_seed = clone_config(cfg)
        cfg_seed.seed = int(seed)
        cfg_seed.train.seed = int(seed)
        seeded_results[int(seed)] = run_mode_ladder(cfg_seed, modes=modes, edge=edge)
    return seeded_results


def build_seeded_mode_summary_table(
    seeded_results: dict[int, dict[str, dict[str, Any]]],
) -> pd.DataFrame:
    """Aggregate matched seed-grid mode runs into one summary table."""
    rows: list[dict[str, Any]] = []
    all_modes = sorted({mode for payload in seeded_results.values() for mode in payload})
    for mode in all_modes:
        sinkhorn: list[float] = []
        calibration: list[float] = []
        context_delta: list[float] = []
        increases: list[str] = []
        decreases: list[str] = []
        split_strategies: list[str] = []
        for _, payload in seeded_results.items():
            if mode not in payload:
                continue
            evaluation = payload[mode]["evaluation"]
            transition = payload[mode]["transition_model"]
            biology = evaluation.get("biology_summary") or {}
            sinkhorn.append(float(evaluation["heldout_metrics"]["sinkhorn"]))
            calibration.append(float(evaluation["calibration"]["mean_abs_shift_error"]))
            delta = (evaluation.get("context_sensitivity") or {}).get("context_sensitivity_delta")
            if delta is not None:
                context_delta.append(float(delta))
            if biology.get("dominant_increase_group") is not None:
                increases.append(str(biology["dominant_increase_group"]))
            if biology.get("dominant_decrease_group") is not None:
                decreases.append(str(biology["dominant_decrease_group"]))
            split_strategies.append(str(transition["split_summary"]["split_strategy"]))
        if not sinkhorn:
            continue
        increase_mode = Counter(increases).most_common(1)[0][0] if increases else None
        decrease_mode = Counter(decreases).most_common(1)[0][0] if decreases else None
        split_mode = Counter(split_strategies).most_common(1)[0][0] if split_strategies else None
        rows.append(
            {
                "mode": mode,
                "n_seeds": len(sinkhorn),
                "sinkhorn_mean": sum(sinkhorn) / len(sinkhorn),
                "sinkhorn_std": pd.Series(sinkhorn).std(ddof=0),
                "calibration_mean": sum(calibration) / len(calibration),
                "calibration_std": pd.Series(calibration).std(ddof=0),
                "context_delta_mean": None if not context_delta else sum(context_delta) / len(context_delta),
                "context_delta_std": None if not context_delta else pd.Series(context_delta).std(ddof=0),
                "dominant_increase_group": increase_mode,
                "dominant_decrease_group": decrease_mode,
                "split_strategy": split_mode,
            }
        )
    return pd.DataFrame(rows).sort_values(["sinkhorn_mean", "calibration_mean"]).reset_index(drop=True)


def run_latent_backend_compare(
    cfg: DictConfig,
    *,
    backends: list[str] | None = None,
    edge: str | None = None,
    mode: str = "set_only",
) -> dict[str, dict[str, Any]]:
    """Run matched latent-backend comparisons for one edge/mode pair."""
    backends = backends or ["hlca", "pca"]
    cfg_base = clone_config(cfg)
    cfg_base.context_model.mode = mode
    cfg_base.context_model.graph_enabled = bool(mode == "graph_of_sets")
    if edge is not None:
        src, tgt = [part.strip() for part in str(edge).split("->", 1)]
        cfg_base.transition_model.active_edge = [src, tgt]

    results: dict[str, dict[str, Any]] = {}
    for backend in backends:
        cfg_backend = clone_config(cfg_base)
        cfg_backend.reference.latent_backend = backend
        if backend == "pca":
            cfg_backend.reference.n_components = int(getattr(cfg_backend.reference, "n_components", 32))
        pipeline = run_full(cfg_backend)
        results[backend] = pipeline["steps"]
    return results


def build_latent_comparison_table(backend_results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Convert matched latent-backend runs into a compact comparison table."""
    rows: list[dict[str, Any]] = []
    for backend, steps in backend_results.items():
        reference = steps["reference"]["reference"]
        evaluation = steps["evaluation"]
        rows.append(
            {
                "backend": backend,
                "sinkhorn": evaluation["heldout_metrics"]["sinkhorn"],
                "calibration_error": evaluation["calibration"]["mean_abs_shift_error"],
                "dominant_increase_group": (evaluation.get("biology_summary") or {}).get("dominant_increase_group"),
                "dominant_decrease_group": (evaluation.get("biology_summary") or {}).get("dominant_decrease_group"),
                "latent_dim": reference["latent_shape"][1],
                "provenance_mode": reference.get("provenance", {}).get("mode"),
                "source_path": reference.get("source_path"),
            }
        )
    return pd.DataFrame(rows).sort_values("sinkhorn").reset_index(drop=True)


def load_run(run_dir: str | Path) -> dict[str, Any]:
    """Load JSON/YAML artifacts from a run-like directory for notebook inspection."""
    run_dir = Path(run_dir)
    payload: dict[str, Any] = {"run_dir": str(run_dir), "files": {}}
    if not run_dir.exists():
        payload["error"] = "missing_run_dir"
        return payload

    for path in sorted(run_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix not in {".json", ".yaml", ".yml", ".md"}:
            continue
        rel = path.relative_to(run_dir).as_posix()
        if path.suffix == ".json":
            try:
                payload["files"][rel] = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                payload["files"][rel] = {"_error": "failed_to_parse_json"}
        else:
            payload["files"][rel] = path.read_text(encoding="utf-8")
    return payload


def available_steps() -> list[str]:
    return sorted(_STEP_REGISTRY)


__all__ = [
    "available_steps",
    "build_biology_summary_table",
    "build_context_summary_table",
    "build_gate_ready_table",
    "build_latent_comparison_table",
    "build_mode_comparison_table",
    "build_seeded_mode_summary_table",
    "build_reference_label_table",
    "build_reference_summary_table",
    "build_spatial_summary_table",
    "build_step_status_table",
    "build_transition_summary_table",
    "clone_config",
    "compose_config",
    "load_run",
    "run_latent_backend_compare",
    "run_mode_ladder",
    "run_seeded_mode_ladder",
    "run_pipeline",
    "run_step",
]
