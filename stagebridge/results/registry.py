"""Lightweight durable registries for StageBridge attempts and milestones."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping

import yaml

from stagebridge.results.manifest import (
    DEFAULT_PROMOTED_RESULTS,
    MILESTONE_INDEX_COLUMNS,
    RESULTS_REGISTRY_COLUMNS,
    SCRATCH_CURRENT_RELATIVE,
    RunMetadata,
    RunMetrics,
    registry_dir,
    stage_edges_label,
)


def _results_registry_path(base_dir: str | Path | None = None) -> Path:
    return registry_dir(base_dir) / "results_registry.csv"


def _milestone_index_path(base_dir: str | Path | None = None) -> Path:
    return registry_dir(base_dir) / "milestone_index.csv"


def _promoted_results_path(base_dir: str | Path | None = None) -> Path:
    return registry_dir(base_dir) / "promoted_results.yaml"


def _read_csv_rows(path: Path, columns: list[str]) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != columns:
            return []
        return [dict(row) for row in reader]


def _write_csv_rows(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})


def _normalize_promoted_results(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    normalized = dict(DEFAULT_PROMOTED_RESULTS)
    if not isinstance(payload, Mapping):
        return normalized
    for key in DEFAULT_PROMOTED_RESULTS:
        if key in payload:
            normalized[key] = payload[key]
    return normalized


def ensure_registry_files(base_dir: str | Path | None = None) -> None:
    """Create or normalize the three durable registry files."""
    reg_dir = registry_dir(base_dir)
    reg_dir.mkdir(parents=True, exist_ok=True)

    results_path = _results_registry_path(base_dir)
    existing_rows = _read_csv_rows(results_path, RESULTS_REGISTRY_COLUMNS)
    _write_csv_rows(results_path, RESULTS_REGISTRY_COLUMNS, existing_rows)

    milestone_path = _milestone_index_path(base_dir)
    existing_milestones = _read_csv_rows(milestone_path, MILESTONE_INDEX_COLUMNS)
    _write_csv_rows(milestone_path, MILESTONE_INDEX_COLUMNS, existing_milestones)

    promoted_path = _promoted_results_path(base_dir)
    promoted = read_promoted_results(base_dir) if promoted_path.exists() else {}
    normalized = _normalize_promoted_results(promoted)
    with promoted_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(normalized, handle, sort_keys=True)


def read_results_registry(base_dir: str | Path | None = None) -> list[dict[str, str]]:
    """Return the results registry rows."""
    ensure_registry_files(base_dir)
    return _read_csv_rows(_results_registry_path(base_dir), RESULTS_REGISTRY_COLUMNS)


def read_milestone_index(base_dir: str | Path | None = None) -> list[dict[str, str]]:
    """Return the milestone index rows."""
    ensure_registry_files(base_dir)
    return _read_csv_rows(_milestone_index_path(base_dir), MILESTONE_INDEX_COLUMNS)


def read_promoted_results(base_dir: str | Path | None = None) -> dict[str, Any]:
    """Load the promoted-results YAML."""
    promoted_path = _promoted_results_path(base_dir)
    if not promoted_path.exists():
        return dict(DEFAULT_PROMOTED_RESULTS)
    with promoted_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return _normalize_promoted_results(payload)


def _results_registry_row(
    metadata: RunMetadata,
    metrics: RunMetrics,
    *,
    scratch_path: str,
    promoted: bool,
    milestone_id: str | None,
) -> dict[str, Any]:
    return {
        "timestamp": metadata.timestamp,
        "git_commit": metadata.git_commit,
        "git_short_hash": metadata.git_short_hash,
        "git_branch": metadata.git_branch,
        "experiment_name": metadata.experiment_name,
        "mode": metadata.mode,
        "stage_edges": stage_edges_label(metadata.stage_edges),
        "split_name": metadata.split_name,
        "wes_regularizer_enabled": "yes" if metadata.wes_regularizer_enabled else "no",
        "spatial_mapping_method": metadata.spatial_mapping_method,
        "context_model_mode": metadata.context_model_mode,
        "status": metadata.status,
        "primary_metric": "" if metrics.primary_metric is None else metrics.primary_metric,
        "promoted": "yes" if promoted else "no",
        "scratch_path": scratch_path,
        "milestone_id": milestone_id or "",
    }


def upsert_results_registry_row(
    metadata: RunMetadata,
    metrics: RunMetrics,
    *,
    scratch_path: str | Path | None = None,
    promoted: bool = False,
    milestone_id: str | None = None,
    base_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Insert or update one results-registry row keyed by timestamp."""
    ensure_registry_files(base_dir)
    rows = read_results_registry(base_dir)
    row = _results_registry_row(
        metadata,
        metrics,
        scratch_path=str(scratch_path or SCRATCH_CURRENT_RELATIVE.as_posix()),
        promoted=promoted,
        milestone_id=milestone_id,
    )
    updated = False
    for idx, existing in enumerate(rows):
        if existing.get("timestamp") == metadata.timestamp:
            rows[idx] = {key: str(value) for key, value in row.items()}
            updated = True
            break
    if not updated:
        rows.append({key: str(value) for key, value in row.items()})
    _write_csv_rows(_results_registry_path(base_dir), RESULTS_REGISTRY_COLUMNS, rows)
    return row


def find_results_registry_row(timestamp: str, base_dir: str | Path | None = None) -> dict[str, str] | None:
    """Find one results-registry row by timestamp."""
    for row in read_results_registry(base_dir):
        if row.get("timestamp") == timestamp:
            return row
    return None


def mark_run_promoted(
    *,
    timestamp: str,
    milestone_id: str,
    base_dir: str | Path | None = None,
) -> None:
    """Mark an existing results row as promoted."""
    rows = read_results_registry(base_dir)
    for row in rows:
        if row.get("timestamp") == timestamp:
            row["promoted"] = "yes"
            row["status"] = "promoted"
            row["milestone_id"] = milestone_id
            _write_csv_rows(_results_registry_path(base_dir), RESULTS_REGISTRY_COLUMNS, rows)
            return
    raise KeyError(f"No results registry row found for timestamp '{timestamp}'.")


def attach_milestone_id(
    *,
    timestamp: str,
    milestone_id: str,
    base_dir: str | Path | None = None,
) -> None:
    """Attach a durable milestone id to an existing run without promoting it."""
    rows = read_results_registry(base_dir)
    for row in rows:
        if row.get("timestamp") == timestamp:
            row["milestone_id"] = milestone_id
            _write_csv_rows(_results_registry_path(base_dir), RESULTS_REGISTRY_COLUMNS, rows)
            return
    raise KeyError(f"No results registry row found for timestamp '{timestamp}'.")


def upsert_milestone_index_row(
    row: Mapping[str, Any],
    *,
    base_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Insert or update one milestone row keyed by milestone id."""
    ensure_registry_files(base_dir)
    rows = read_milestone_index(base_dir)
    normalized = {key: str(row.get(key, "")) for key in MILESTONE_INDEX_COLUMNS}
    updated = False
    for idx, existing in enumerate(rows):
        if existing.get("milestone_id") == normalized["milestone_id"]:
            rows[idx] = normalized
            updated = True
            break
    if not updated:
        rows.append(normalized)
    _write_csv_rows(_milestone_index_path(base_dir), MILESTONE_INDEX_COLUMNS, rows)
    return normalized


def update_promoted_results(
    updates: Mapping[str, Any],
    *,
    base_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Merge updates into the promoted-results YAML."""
    ensure_registry_files(base_dir)
    promoted = read_promoted_results(base_dir)
    for key, value in updates.items():
        if key in DEFAULT_PROMOTED_RESULTS:
            promoted[key] = value
    with _promoted_results_path(base_dir).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(promoted, handle, sort_keys=True)
    return promoted
