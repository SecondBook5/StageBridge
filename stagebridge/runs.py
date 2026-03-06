"""Run-context helpers for notebook/CLI orchestration."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from omegaconf import DictConfig

from stagebridge.io.paths import resolve_run_paths


@dataclass(slots=True)
class RunContext:
    """Filesystem context for one workflow execution."""

    run_id: str
    run_dir: Path
    tables_dir: Path
    figures_dir: Path
    logs_dir: Path


@dataclass(slots=True)
class OutputContext:
    """Output context for training/evaluation artifacts in `output_dir`."""

    output_dir: Path
    tables_dir: Path
    figures_dir: Path
    checkpoints_dir: Path


def build_pipeline_run_context(cfg: DictConfig, run_id: str | None = None) -> RunContext:
    """Resolve run-scoped context under external data root `runs/` layout."""
    run_paths = resolve_run_paths(cfg, run_id=run_id)
    figures_dir = run_paths.run_dir / "figures"
    logs_dir = run_paths.run_dir / "logs"
    figures_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    return RunContext(
        run_id=run_paths.run_id,
        run_dir=run_paths.run_dir,
        tables_dir=run_paths.tables_dir,
        figures_dir=figures_dir,
        logs_dir=logs_dir,
    )


def build_output_context(output_dir: str | Path) -> OutputContext:
    """Resolve output context under in-repo (or user-provided) output directory."""
    out = Path(output_dir)
    tables = out / "tables"
    figures = out / "figures"
    checkpoints = out / "checkpoints"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
    return OutputContext(
        output_dir=out,
        tables_dir=tables,
        figures_dir=figures,
        checkpoints_dir=checkpoints,
    )


def default_run_id(prefix: str) -> str:
    """Create a UTC timestamped run identifier."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{ts}"
