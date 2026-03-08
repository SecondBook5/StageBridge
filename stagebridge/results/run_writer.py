"""Scratch-run writer for the lightweight StageBridge results system."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from typing import Any

from omegaconf import DictConfig, OmegaConf
import yaml

from stagebridge.pipelines.run_full import run_full
from stagebridge.results.manifest import (
    RunMetrics,
    build_run_metadata,
    build_smoke_metrics,
    repo_root,
    scratch_current_dir,
    validate_status,
)
from stagebridge.results.registry import upsert_results_registry_row
from stagebridge.results.result_card import build_result_card


@dataclass(slots=True, frozen=True)
class ScratchRunPaths:
    """Paths for the reusable scratch workspace."""

    current_dir: Path
    resolved_config: Path
    run_metadata: Path
    metrics: Path
    result_card: Path
    stdout_log: Path
    artifacts_dir: Path


def scratch_run_paths(base_dir: str | Path | None = None) -> ScratchRunPaths:
    """Resolve the current scratch workspace paths."""
    current = scratch_current_dir(base_dir)
    return ScratchRunPaths(
        current_dir=current,
        resolved_config=current / "resolved_config.yaml",
        run_metadata=current / "run_metadata.json",
        metrics=current / "metrics.json",
        result_card=current / "result_card.md",
        stdout_log=current / "stdout.log",
        artifacts_dir=current / "artifacts",
    )


def _stage_pipeline_output(pipeline_output: Mapping[str, Any] | None) -> tuple[list[str], list[str]]:
    worked: list[str] = []
    failed: list[str] = []
    if not isinstance(pipeline_output, Mapping):
        return worked, failed

    steps = pipeline_output.get("steps", pipeline_output)
    if not isinstance(steps, Mapping):
        return worked, failed

    for name, payload in steps.items():
        if isinstance(payload, Mapping) and payload.get("ok") is True:
            worked.append(str(name))
        else:
            failed.append(str(name))
    return worked, failed


def _write_artifacts(
    artifacts_dir: Path,
    artifact_sources: Mapping[str, Any] | None,
) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    if not artifact_sources:
        return

    for relative_name, value in artifact_sources.items():
        destination = artifacts_dir / relative_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(value, (str, bytes)) and not isinstance(value, Path):
            if isinstance(value, bytes):
                destination.write_bytes(value)
            else:
                destination.write_text(value, encoding="utf-8")
            continue
        if isinstance(value, (Mapping, list, tuple, set, bool, int, float)) or value is None:
            destination.write_text(json.dumps(value, indent=2), encoding="utf-8")
            continue
        source_path = Path(value)
        if source_path.is_dir():
            shutil.copytree(source_path, destination, dirs_exist_ok=True)
            continue
        if source_path.is_file():
            shutil.copy2(source_path, destination)
            continue
        destination.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _prepare_staging_dir(paths: ScratchRunPaths) -> Path:
    scratch_root = paths.current_dir.parent
    scratch_root.mkdir(parents=True, exist_ok=True)
    staging_dir = scratch_root / ".staging-current"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)
    return staging_dir


def _finalize_staging(staging_dir: Path, current_dir: Path) -> None:
    if current_dir.exists():
        shutil.rmtree(current_dir)
    staging_dir.rename(current_dir)


def write_scratch_run(
    cfg: DictConfig | Mapping[str, Any] | None,
    pipeline_output: Mapping[str, Any] | None,
    *,
    experiment_name: str | None = None,
    mode: str | None = None,
    stage_edges: list[str] | None = None,
    status: str = "complete",
    notebook_source: str = "StageBridge.ipynb",
    metrics: RunMetrics | Mapping[str, Any] | None = None,
    worked: list[str] | None = None,
    failed: list[str] | None = None,
    milestone_candidate: bool = False,
    next_recommended_step: str = "Review scratch outputs and promote only if the run is worth keeping.",
    stdout_text: str | None = None,
    artifact_sources: Mapping[str, Any] | None = None,
    base_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Write the current scratch workspace and append a registry row."""
    validate_status(status)
    metadata = build_run_metadata(
        cfg,
        experiment_name=experiment_name,
        mode=mode,
        stage_edges=stage_edges,
        notebook_source=notebook_source,
        status=status,
        base_dir=base_dir,
    )
    if metrics is None:
        inferred_worked, inferred_failed = _stage_pipeline_output(pipeline_output)
        metrics_obj = build_smoke_metrics(
            completed_steps=len(inferred_worked),
            failed_steps=len(inferred_failed),
        )
    elif isinstance(metrics, RunMetrics):
        metrics_obj = metrics
        inferred_worked, inferred_failed = _stage_pipeline_output(pipeline_output)
    else:
        metrics_obj = RunMetrics.from_dict(metrics)
        inferred_worked, inferred_failed = _stage_pipeline_output(pipeline_output)

    worked = list(inferred_worked if worked is None else worked)
    failed = list(inferred_failed if failed is None else failed)
    result_card = build_result_card(
        metadata,
        metrics_obj,
        worked=worked,
        failed=failed,
        milestone_candidate=milestone_candidate,
        next_recommended_step=next_recommended_step,
    )

    paths = scratch_run_paths(base_dir)
    staging_dir = _prepare_staging_dir(paths)
    staging_paths = ScratchRunPaths(
        current_dir=staging_dir,
        resolved_config=staging_dir / "resolved_config.yaml",
        run_metadata=staging_dir / "run_metadata.json",
        metrics=staging_dir / "metrics.json",
        result_card=staging_dir / "result_card.md",
        stdout_log=staging_dir / "stdout.log",
        artifacts_dir=staging_dir / "artifacts",
    )

    if isinstance(cfg, DictConfig):
        config_payload = OmegaConf.to_container(cfg, resolve=True)
    elif isinstance(cfg, Mapping):
        config_payload = dict(cfg)
    else:
        config_payload = {}

    with staging_paths.resolved_config.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config_payload, handle, sort_keys=False)
    staging_paths.run_metadata.write_text(
        json.dumps(metadata.to_dict(), indent=2),
        encoding="utf-8",
    )
    staging_paths.metrics.write_text(
        json.dumps(metrics_obj.to_dict(), indent=2),
        encoding="utf-8",
    )
    staging_paths.result_card.write_text(result_card, encoding="utf-8")
    if stdout_text is not None:
        staging_paths.stdout_log.write_text(stdout_text, encoding="utf-8")
    _write_artifacts(staging_paths.artifacts_dir, artifact_sources)

    _finalize_staging(staging_dir, paths.current_dir)
    row = upsert_results_registry_row(
        metadata,
        metrics_obj,
        scratch_path=str(paths.current_dir.relative_to(repo_root(base_dir))),
        promoted=False,
        milestone_id=None,
        base_dir=base_dir,
    )
    return {
        "ok": True,
        "scratch_dir": str(paths.current_dir),
        "resolved_config": str(paths.resolved_config),
        "run_metadata": metadata.to_dict(),
        "metrics": metrics_obj.to_dict(),
        "result_card_path": str(paths.result_card),
        "registry_row": row,
    }


def load_current_scratch_run(base_dir: str | Path | None = None) -> dict[str, Any]:
    """Load the current scratch run payload."""
    paths = scratch_run_paths(base_dir)
    payload: dict[str, Any] = {"scratch_dir": str(paths.current_dir)}
    if not paths.current_dir.exists():
        payload["error"] = "missing_scratch_run"
        return payload

    payload["resolved_config"] = yaml.safe_load(paths.resolved_config.read_text(encoding="utf-8"))
    payload["run_metadata"] = json.loads(paths.run_metadata.read_text(encoding="utf-8"))
    payload["metrics"] = json.loads(paths.metrics.read_text(encoding="utf-8"))
    payload["result_card"] = paths.result_card.read_text(encoding="utf-8")
    payload["artifacts"] = sorted(
        str(path.relative_to(paths.current_dir))
        for path in paths.artifacts_dir.rglob("*")
        if path.is_file()
    )
    return payload


def write_pipeline_scratch_run(
    cfg: DictConfig | Mapping[str, Any] | None,
    pipeline_output: Mapping[str, Any],
    *,
    notebook_source: str = "StageBridge.ipynb",
    experiment_name: str | None = None,
    extra_artifact_sources: Mapping[str, Any] | None = None,
    base_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Write a biological pipeline run into the scratch workspace with inferred metrics."""
    steps = pipeline_output.get("steps", {}) if isinstance(pipeline_output, Mapping) else {}
    transition = steps.get("transition_model", {}) if isinstance(steps, Mapping) else {}
    evaluation = steps.get("evaluation", {}) if isinstance(steps, Mapping) else {}
    status = "complete" if all(bool(steps.get(name, {}).get("ok")) for name in steps) else "partial"
    edge = transition.get("edge")
    mode = transition.get("mode")
    heldout = evaluation.get("heldout_metrics", {})
    calibration = evaluation.get("calibration", {})
    metric_notes = []
    if transition.get("reference"):
        metric_notes.append(f"reference={transition['reference'].get('source_path')}")
    if transition.get("spatial_mapping"):
        metric_notes.append(f"spatial={transition['spatial_mapping'].get('method')}")
    if transition.get("context_diagnostics"):
        metric_notes.append(f"context={transition['context_diagnostics'].get('mode')}")
    metrics = RunMetrics(
        primary_metric=heldout.get("sinkhorn"),
        secondary_metrics=heldout,
        calibration=calibration,
        ablation_label=str(mode) if mode is not None else None,
        notes=" | ".join(metric_notes) if metric_notes else "Notebook pipeline execution.",
    )
    artifact_sources = {}
    if isinstance(evaluation, Mapping):
        artifact_sources.update(evaluation.get("artifact_sources", {}))
    if isinstance(transition, Mapping):
        artifact_sources["transition_summary.json"] = {
            "edge": transition.get("edge"),
            "mode": transition.get("mode"),
            "reference": transition.get("reference"),
            "spatial_mapping": transition.get("spatial_mapping"),
            "context_model": transition.get("context_model"),
            "split_summary": transition.get("split_summary"),
            "context_diagnostics": transition.get("context_diagnostics"),
            "wes_diagnostics": transition.get("wes_diagnostics"),
        }
    if extra_artifact_sources:
        artifact_sources.update(extra_artifact_sources)
    return write_scratch_run(
        cfg,
        pipeline_output,
        experiment_name=experiment_name or str(
            transition.get("edge", "stagebridge_pipeline_run")
        ).replace("->", "_to_"),
        mode=str(mode) if mode is not None else None,
        stage_edges=[str(edge)] if edge is not None else None,
        status=status,
        notebook_source=notebook_source,
        metrics=metrics,
        milestone_candidate=False,
        next_recommended_step="Inspect held-out metrics and gate outputs before any promotion decision.",
        artifact_sources=artifact_sources,
        base_dir=base_dir,
    )


def run_smoke_execution(
    cfg: DictConfig | Mapping[str, Any],
    *,
    notebook_source: str = "StageBridge.ipynb",
    base_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run the minimal package smoke path and record it in scratch/registry."""
    pipeline_output = run_full(cfg)  # current pipeline entrypoints accept the composed config object
    return write_scratch_run(
        cfg,
        pipeline_output,
        experiment_name=str(
            cfg.get("run_name", "stagebridge_smoke") if isinstance(cfg, Mapping) else getattr(cfg, "run_name", "stagebridge_smoke")
        ),
        mode="smoke_infrastructure",
        status="complete",
        notebook_source=notebook_source,
        milestone_candidate=False,
        next_recommended_step="Promote only if you need to keep the infrastructure proof artifact.",
        base_dir=base_dir,
    )
