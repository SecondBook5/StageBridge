"""Durable milestone helpers for promoted and archived runs."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any, Sequence

from stagebridge.results.manifest import (
    PROMOTED_RESULT_KEYS,
    RunMetadata,
    RunMetrics,
    milestones_dir,
    repo_root,
)
from stagebridge.results.registry import (
    attach_milestone_id,
    find_results_registry_row,
    mark_run_promoted,
    update_promoted_results,
    upsert_milestone_index_row,
)
from stagebridge.results.run_writer import load_current_scratch_run, scratch_run_paths


@dataclass(slots=True, frozen=True)
class MilestonePromotionResult:
    """Summary of one scratch-to-milestone promotion."""

    milestone_id: str
    milestone_path: Path
    source_timestamp: str


@dataclass(slots=True, frozen=True)
class MilestoneArchiveResult:
    """Summary of one scratch-to-milestone archive operation."""

    milestone_id: str
    milestone_path: Path
    source_timestamp: str


def _infer_promoted_slots(metadata: RunMetadata) -> list[str]:
    slots: list[str] = []
    if metadata.mode == "rna_only":
        slots.append("best_rna_only")
    if metadata.context_model_mode == "deep_sets" and not metadata.wes_regularizer_enabled:
        slots.append("best_deep_sets")
    if metadata.context_model_mode == "deep_sets_transformer_hybrid" and not metadata.wes_regularizer_enabled:
        slots.append("best_deep_sets_transformer_hybrid")
    if metadata.context_model_mode == "set_only" and not metadata.wes_regularizer_enabled:
        slots.append("best_set_only")
    if metadata.context_model_mode == "typed_hierarchical_transformer" and not metadata.wes_regularizer_enabled:
        slots.append("best_typed_hierarchical_transformer")
    if metadata.context_model_mode == "graph_of_sets" and not metadata.wes_regularizer_enabled:
        slots.append("best_graph_of_sets")
    if metadata.context_model_mode == "graph_of_sets" and metadata.wes_regularizer_enabled:
        slots.append("best_graph_of_sets_wes")
    if "AAH->AIS" in metadata.stage_edges:
        slots.append("best_aah_to_ais")
    if "AIS->MIA" in metadata.stage_edges:
        slots.append("best_ais_to_mia")
    if metadata.mode != "smoke_infrastructure":
        slots.append("best_full_v1_candidate")
    return [slot for slot in slots if slot in PROMOTED_RESULT_KEYS]


def _milestone_summary_text(
    metadata: RunMetadata,
    metrics: RunMetrics,
    *,
    summary: str,
    importance_level: str,
    archive_only: bool,
    interpretation_notes: str | None,
    next_step_recommendation: str | None,
    source_path: Path,
) -> str:
    notes = interpretation_notes or "No interpretation notes were supplied."
    next_step = next_step_recommendation or "Review the milestone artifacts before further use."
    return "\n".join(
        [
            "# Milestone Summary",
            "",
            f"- Milestone summary: {summary}",
            f"- Importance level: {importance_level}",
            f"- Archive only: {'yes' if archive_only else 'no'}",
            f"- Source run timestamp: {metadata.timestamp}",
            f"- Source scratch path: {source_path}",
            f"- Experiment: {metadata.experiment_name}",
            f"- Mode: {metadata.mode}",
            f"- Stage edge(s): {', '.join(metadata.stage_edges) if metadata.stage_edges else 'none specified'}",
            f"- Primary metric: {metrics.primary_metric if metrics.primary_metric is not None else 'n/a'}",
            f"- Secondary metrics: {metrics.secondary_metrics}",
            "",
            "## Interpretation Notes",
            notes,
            "",
            "## Next Step Recommendation",
            next_step,
            "",
        ]
    )


def _copy_current_scratch_run(
    *,
    milestone_id: str,
    summary: str,
    importance_level: str,
    git_tag: str,
    interpretation_notes: str | None,
    next_step_recommendation: str | None,
    archive_only: bool,
    base_dir: str | Path | None = None,
) -> tuple[RunMetadata, RunMetrics, Path]:
    paths = scratch_run_paths(base_dir)
    required_paths = [paths.run_metadata, paths.resolved_config, paths.metrics, paths.result_card]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Cannot export scratch run; missing files: {missing}")

    scratch_payload = load_current_scratch_run(base_dir)
    metadata = RunMetadata.from_dict(scratch_payload["run_metadata"])
    metrics = RunMetrics.from_dict(scratch_payload["metrics"])

    registry_row = find_results_registry_row(metadata.timestamp, base_dir=base_dir)
    if registry_row is None:
        raise FileNotFoundError(
            f"No results registry row exists for scratch run timestamp '{metadata.timestamp}'."
        )

    milestone_root = milestones_dir(base_dir)
    milestone_root.mkdir(parents=True, exist_ok=True)
    milestone_path = milestone_root / milestone_id
    if milestone_path.exists():
        raise FileExistsError(f"Milestone path already exists: {milestone_path}")
    milestone_path.mkdir(parents=True, exist_ok=False)

    shutil.copy2(paths.resolved_config, milestone_path / "resolved_config.yaml")
    shutil.copy2(paths.run_metadata, milestone_path / "run_metadata.json")
    shutil.copy2(paths.metrics, milestone_path / "metrics_snapshot.json")
    shutil.copy2(paths.result_card, milestone_path / "result_card.md")

    artifacts_target = milestone_path / "artifacts"
    if paths.artifacts_dir.exists():
        shutil.copytree(paths.artifacts_dir, artifacts_target, dirs_exist_ok=True)
    else:
        artifacts_target.mkdir(parents=True, exist_ok=True)

    source_run_text = "\n".join(
        [
            f"source_timestamp: {metadata.timestamp}",
            f"scratch_path: {paths.current_dir}",
            f"registry_path: {repo_root(base_dir) / 'results' / 'registry' / 'results_registry.csv'}",
            "",
        ]
    )
    (milestone_path / "source_run.txt").write_text(source_run_text, encoding="utf-8")

    milestone_summary = _milestone_summary_text(
        metadata,
        metrics,
        summary=summary,
        importance_level=importance_level,
        archive_only=archive_only,
        interpretation_notes=interpretation_notes,
        next_step_recommendation=next_step_recommendation,
        source_path=paths.current_dir,
    )
    (milestone_path / "milestone_summary.md").write_text(milestone_summary, encoding="utf-8")

    upsert_milestone_index_row(
        {
            "milestone_id": milestone_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source_timestamp": metadata.timestamp,
            "git_commit": metadata.git_commit,
            "git_short_hash": metadata.git_short_hash,
            "git_tag": git_tag,
            "summary": summary,
            "importance_level": importance_level,
            "milestone_path": str(milestone_path.relative_to(repo_root(base_dir))),
        },
        base_dir=base_dir,
    )
    return metadata, metrics, milestone_path


def promote_current_scratch_run(
    *,
    milestone_id: str,
    summary: str,
    importance_level: str = "candidate",
    git_tag: str = "",
    promotion_slots: Sequence[str] | None = None,
    interpretation_notes: str | None = None,
    next_step_recommendation: str | None = None,
    base_dir: str | Path | None = None,
) -> MilestonePromotionResult:
    """Promote the current complete scratch run into a durable milestone."""
    paths = scratch_run_paths(base_dir)
    metadata, metrics, milestone_path = _copy_current_scratch_run(
        milestone_id=milestone_id,
        summary=summary,
        importance_level=importance_level,
        git_tag=git_tag,
        interpretation_notes=interpretation_notes,
        next_step_recommendation=next_step_recommendation,
        archive_only=False,
        base_dir=base_dir,
    )
    if metadata.status != "complete":
        raise ValueError("Only scratch runs with status 'complete' may be promoted.")

    slots = list(promotion_slots or _infer_promoted_slots(metadata))
    promoted_entry = {
        "milestone_id": milestone_id,
        "timestamp": metadata.timestamp,
        "summary": summary,
        "mode": metadata.mode,
        "stage_edges": metadata.stage_edges,
        "primary_metric": metrics.primary_metric,
        "milestone_path": str(milestone_path.relative_to(repo_root(base_dir))),
    }
    promoted_updates: dict[str, Any] = {"latest_promoted": promoted_entry}
    for slot in slots:
        promoted_updates[slot] = promoted_entry
    update_promoted_results(promoted_updates, base_dir=base_dir)

    mark_run_promoted(timestamp=metadata.timestamp, milestone_id=milestone_id, base_dir=base_dir)
    metadata.status = "promoted"
    paths.run_metadata.write_text(json.dumps(metadata.to_dict(), indent=2), encoding="utf-8")

    return MilestonePromotionResult(
        milestone_id=milestone_id,
        milestone_path=milestone_path,
        source_timestamp=metadata.timestamp,
    )


def archive_current_scratch_run(
    *,
    milestone_id: str,
    summary: str,
    importance_level: str = "archive",
    git_tag: str = "",
    interpretation_notes: str | None = None,
    next_step_recommendation: str | None = None,
    base_dir: str | Path | None = None,
) -> MilestoneArchiveResult:
    """Archive the current scratch run as a durable record without promoting it."""
    metadata, _, milestone_path = _copy_current_scratch_run(
        milestone_id=milestone_id,
        summary=summary,
        importance_level=importance_level,
        git_tag=git_tag,
        interpretation_notes=interpretation_notes,
        next_step_recommendation=next_step_recommendation,
        archive_only=True,
        base_dir=base_dir,
    )
    attach_milestone_id(
        timestamp=metadata.timestamp,
        milestone_id=milestone_id,
        base_dir=base_dir,
    )
    return MilestoneArchiveResult(
        milestone_id=milestone_id,
        milestone_path=milestone_path,
        source_timestamp=metadata.timestamp,
    )
