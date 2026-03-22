"""Run lifecycle management for StageBridge pipeline orchestration.

This module provides the core run context and manager for tracking pipeline
execution state, directories, metadata, and status.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from importlib.metadata import version as pkg_version, PackageNotFoundError
from pathlib import Path
from typing import Any

import yaml

from stagebridge.results.manifest import utc_timestamp


class RunStatus(StrEnum):
    """Status values for a pipeline run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class StageStatus(StrEnum):
    """Status values for an individual stage."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


# Standard subdirectories for every run
RUN_SUBDIRS = [
    "config",
    "splits",
    "data",
    "qc",
    "references",
    "spatial_backends",
    "baselines",
    "full_model",
    "ablations",
    "biology",
    "figures",
    "notebook_cache",
    "logs",
    "manifests",
    "checkpoints",
    "metrics",
]


@dataclass
class StageInfo:
    """Information about a single pipeline stage."""

    name: str
    status: StageStatus = StageStatus.PENDING
    start_time: str | None = None
    end_time: str | None = None
    duration_seconds: float | None = None
    output_dir: Path | None = None
    log_file: Path | None = None
    error_message: str | None = None
    artifacts: list[str] = field(default_factory=list)


@dataclass
class RunContext:
    """Context object holding all state for a single pipeline run.

    This is passed between stages and contains paths, config, and status.
    """

    run_id: str
    run_dir: Path
    config: dict[str, Any]
    status: RunStatus = RunStatus.PENDING
    current_stage: str | None = None
    start_time: str | None = None
    end_time: str | None = None
    seed: int = 42
    device: str = "cpu"
    resume_if_possible: bool = True
    force_rerun: bool = False
    stages: dict[str, StageInfo] = field(default_factory=dict)
    git_commit: str = "unknown"
    git_dirty: bool = False
    python_version: str = ""
    environment: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize derived attributes."""
        if not self.python_version:
            self.python_version = sys.version

    @property
    def config_dir(self) -> Path:
        """Return the config subdirectory."""
        return self.run_dir / "config"

    @property
    def logs_dir(self) -> Path:
        """Return the logs subdirectory."""
        return self.run_dir / "logs"

    @property
    def manifests_dir(self) -> Path:
        """Return the manifests subdirectory."""
        return self.run_dir / "manifests"

    @property
    def metadata_path(self) -> Path:
        """Return the path to run_metadata.yaml."""
        return self.config_dir / "run_metadata.yaml"

    @property
    def master_manifest_path(self) -> Path:
        """Return the path to master_manifest.json."""
        return self.manifests_dir / "master_manifest.json"

    def stage_dir(self, stage_name: str) -> Path:
        """Return the output directory for a stage."""
        # Map stage names to subdirectories
        stage_to_subdir = {
            "data_qc": "qc",
            "reference": "references",
            "spatial_backend": "spatial_backends",
            "baselines": "baselines",
            "full_model": "full_model",
            "ablations": "ablations",
            "biology": "biology",
            "figures": "figures",
        }
        subdir = stage_to_subdir.get(stage_name, stage_name)
        return self.run_dir / subdir

    def stage_log(self, stage_name: str) -> Path:
        """Return the log file path for a stage."""
        return self.logs_dir / f"{stage_name}.log"


def _get_git_info(repo_path: Path | None = None) -> tuple[str, bool]:
    """Get git commit hash and dirty status."""
    cwd = repo_path or Path.cwd()
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        commit = "unknown"

    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = len(status) > 0
    except Exception:
        dirty = False

    return commit, dirty


def _get_environment_info() -> dict[str, Any]:
    """Collect environment information."""
    env_info: dict[str, Any] = {
        "python_version": sys.version,
        "platform": sys.platform,
    }

    # Check for CUDA
    try:
        import torch

        env_info["torch_version"] = str(torch.__version__)
        env_info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env_info["cuda_version"] = str(torch.version.cuda) if torch.version.cuda else None
            env_info["cuda_device_count"] = torch.cuda.device_count()
    except ImportError:
        env_info["torch_version"] = None
        env_info["cuda_available"] = False

    # Check key package versions
    for pkg in ["numpy", "pandas", "anndata", "scanpy", "tqdm"]:
        try:
            version = pkg_version(pkg)
            env_info[f"{pkg}_version"] = str(version)
        except PackageNotFoundError:
            env_info[f"{pkg}_version"] = None

    return env_info


def _generate_run_id() -> str:
    """Generate a unique run ID based on timestamp."""
    now = datetime.now(timezone.utc)
    return now.strftime("run_%Y%m%d_%H%M%S")


class RunManager:
    """Manager for run lifecycle, directories, and metadata.

    This class handles:
    - Run initialization and directory creation
    - Metadata persistence
    - Status tracking
    - Run finalization
    """

    def __init__(
        self,
        artifacts_root: Path | str = "artifacts/runs",
        repo_root: Path | str | None = None,
    ) -> None:
        """Initialize the run manager.

        Parameters
        ----------
        artifacts_root : Path or str
            Root directory for run artifacts (default: artifacts/runs)
        repo_root : Path or str, optional
            Repository root for git info (default: auto-detect)
        """
        if repo_root is None:
            # Try to find repo root
            repo_root = Path(__file__).resolve().parents[2]
        self.repo_root = Path(repo_root)
        self.artifacts_root = self.repo_root / artifacts_root
        self._logger = logging.getLogger(__name__)

    def initialize_run(
        self,
        config: dict[str, Any],
        *,
        run_id: str | None = None,
        resume_if_possible: bool = True,
        force_rerun: bool = False,
    ) -> RunContext:
        """Initialize a new run or resume an existing one.

        Parameters
        ----------
        config : dict
            The resolved configuration for the run
        run_id : str, optional
            Explicit run ID (auto-generated if not provided)
        resume_if_possible : bool
            Whether to resume if run directory exists (default: True)
        force_rerun : bool
            Whether to force rerun all stages (default: False)

        Returns
        -------
        RunContext
            The initialized run context
        """
        # Generate or use provided run_id
        if run_id is None:
            run_id = config.get("run_id") or _generate_run_id()

        run_dir = self.artifacts_root / run_id

        # Check for existing run
        metadata_path = run_dir / "config" / "run_metadata.yaml"
        if metadata_path.exists() and resume_if_possible and not force_rerun:
            self._logger.info(f"Resuming existing run: {run_id}")
            return self._resume_run(run_dir, config, force_rerun)

        # Create new run
        self._logger.info(f"Initializing new run: {run_id}")
        return self._create_new_run(run_id, run_dir, config, resume_if_possible, force_rerun)

    def _create_new_run(
        self,
        run_id: str,
        run_dir: Path,
        config: dict[str, Any],
        resume_if_possible: bool,
        force_rerun: bool,
    ) -> RunContext:
        """Create a new run with fresh directories."""
        # Create directory structure
        run_dir.mkdir(parents=True, exist_ok=True)
        for subdir in RUN_SUBDIRS:
            (run_dir / subdir).mkdir(exist_ok=True)

        # Get environment info
        git_commit, git_dirty = _get_git_info(self.repo_root)
        env_info = _get_environment_info()

        # Extract config values
        seed = config.get("seed", 42)
        device = config.get("device", "cpu")

        # Create context
        ctx = RunContext(
            run_id=run_id,
            run_dir=run_dir,
            config=config,
            status=RunStatus.RUNNING,
            start_time=utc_timestamp(),
            seed=seed,
            device=device,
            resume_if_possible=resume_if_possible,
            force_rerun=force_rerun,
            git_commit=git_commit,
            git_dirty=git_dirty,
            environment=env_info,
        )

        # Save initial metadata
        self._save_metadata(ctx)

        # Save resolved config
        config_path = run_dir / "config" / "resolved_config.yaml"
        with config_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False)

        return ctx

    def _resume_run(
        self,
        run_dir: Path,
        config: dict[str, Any],
        force_rerun: bool,
    ) -> RunContext:
        """Resume an existing run from metadata."""
        metadata_path = run_dir / "config" / "run_metadata.yaml"

        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = yaml.safe_load(f)

        # Get fresh environment info
        git_commit, git_dirty = _get_git_info(self.repo_root)
        env_info = _get_environment_info()

        # Restore stage info
        stages: dict[str, StageInfo] = {}
        for stage_name, stage_data in metadata.get("stages", {}).items():
            stages[stage_name] = StageInfo(
                name=stage_name,
                status=StageStatus(stage_data.get("status", "pending")),
                start_time=stage_data.get("start_time"),
                end_time=stage_data.get("end_time"),
                duration_seconds=stage_data.get("duration_seconds"),
                output_dir=Path(stage_data["output_dir"])
                if stage_data.get("output_dir")
                else None,
                log_file=Path(stage_data["log_file"]) if stage_data.get("log_file") else None,
                error_message=stage_data.get("error_message"),
                artifacts=stage_data.get("artifacts", []),
            )

        ctx = RunContext(
            run_id=metadata["run_id"],
            run_dir=run_dir,
            config=config,  # Use new config (may have updates)
            status=RunStatus(metadata.get("status", "running")),
            current_stage=metadata.get("current_stage"),
            start_time=metadata.get("start_time"),
            seed=metadata.get("seed", 42),
            device=metadata.get("device", "cpu"),
            resume_if_possible=True,
            force_rerun=force_rerun,
            stages=stages,
            git_commit=git_commit,
            git_dirty=git_dirty,
            environment=env_info,
        )

        # Update status to running
        ctx.status = RunStatus.RUNNING
        self._save_metadata(ctx)

        return ctx

    def update_status(
        self,
        ctx: RunContext,
        status: RunStatus | None = None,
        current_stage: str | None = None,
    ) -> None:
        """Update run status and save metadata.

        Parameters
        ----------
        ctx : RunContext
            The run context to update
        status : RunStatus, optional
            New status (if changing)
        current_stage : str, optional
            Current stage name (if changing)
        """
        if status is not None:
            ctx.status = status
        if current_stage is not None:
            ctx.current_stage = current_stage

        self._save_metadata(ctx)

    def start_stage(self, ctx: RunContext, stage_name: str) -> StageInfo:
        """Mark a stage as starting.

        Parameters
        ----------
        ctx : RunContext
            The run context
        stage_name : str
            Name of the stage

        Returns
        -------
        StageInfo
            The stage info object
        """
        stage_info = StageInfo(
            name=stage_name,
            status=StageStatus.RUNNING,
            start_time=utc_timestamp(),
            output_dir=ctx.stage_dir(stage_name),
            log_file=ctx.stage_log(stage_name),
        )
        ctx.stages[stage_name] = stage_info
        ctx.current_stage = stage_name

        # Ensure output directory exists
        stage_info.output_dir.mkdir(parents=True, exist_ok=True)

        self._save_metadata(ctx)
        return stage_info

    def complete_stage(
        self,
        ctx: RunContext,
        stage_name: str,
        artifacts: list[str] | None = None,
    ) -> None:
        """Mark a stage as completed.

        Parameters
        ----------
        ctx : RunContext
            The run context
        stage_name : str
            Name of the stage
        artifacts : list of str, optional
            List of artifact paths produced
        """
        if stage_name not in ctx.stages:
            ctx.stages[stage_name] = StageInfo(name=stage_name)

        stage_info = ctx.stages[stage_name]
        stage_info.status = StageStatus.COMPLETED
        stage_info.end_time = utc_timestamp()

        if stage_info.start_time:
            start = datetime.fromisoformat(stage_info.start_time)
            end = datetime.fromisoformat(stage_info.end_time)
            stage_info.duration_seconds = (end - start).total_seconds()

        if artifacts:
            stage_info.artifacts = artifacts

        self._save_metadata(ctx)

    def fail_stage(
        self,
        ctx: RunContext,
        stage_name: str,
        error_message: str,
    ) -> None:
        """Mark a stage as failed.

        Parameters
        ----------
        ctx : RunContext
            The run context
        stage_name : str
            Name of the stage
        error_message : str
            Error message describing the failure
        """
        if stage_name not in ctx.stages:
            ctx.stages[stage_name] = StageInfo(name=stage_name)

        stage_info = ctx.stages[stage_name]
        stage_info.status = StageStatus.FAILED
        stage_info.end_time = utc_timestamp()
        stage_info.error_message = error_message

        if stage_info.start_time:
            start = datetime.fromisoformat(stage_info.start_time)
            end = datetime.fromisoformat(stage_info.end_time)
            stage_info.duration_seconds = (end - start).total_seconds()

        # Update run status
        ctx.status = RunStatus.FAILED

        self._save_metadata(ctx)

    def skip_stage(self, ctx: RunContext, stage_name: str, reason: str = "cached") -> None:
        """Mark a stage as skipped (e.g., due to caching).

        Parameters
        ----------
        ctx : RunContext
            The run context
        stage_name : str
            Name of the stage
        reason : str
            Reason for skipping (default: "cached")
        """
        stage_info = StageInfo(
            name=stage_name,
            status=StageStatus.SKIPPED,
            output_dir=ctx.stage_dir(stage_name),
            error_message=f"Skipped: {reason}",
        )
        ctx.stages[stage_name] = stage_info
        self._save_metadata(ctx)

    def finalize_run(self, ctx: RunContext, success: bool = True) -> None:
        """Finalize the run and save final metadata.

        Parameters
        ----------
        ctx : RunContext
            The run context
        success : bool
            Whether the run completed successfully (default: True)
        """
        ctx.end_time = utc_timestamp()

        if success:
            # Check if any stages failed
            failed_stages = [s for s in ctx.stages.values() if s.status == StageStatus.FAILED]
            if failed_stages:
                ctx.status = RunStatus.PARTIAL
            else:
                ctx.status = RunStatus.COMPLETED
        else:
            ctx.status = RunStatus.FAILED

        self._save_metadata(ctx)

    def _save_metadata(self, ctx: RunContext) -> None:
        """Save run metadata to YAML file."""
        # Build stages dict
        stages_dict = {}
        for stage_name, stage_info in ctx.stages.items():
            stages_dict[stage_name] = {
                "status": stage_info.status.value,
                "start_time": stage_info.start_time,
                "end_time": stage_info.end_time,
                "duration_seconds": stage_info.duration_seconds,
                "output_dir": str(stage_info.output_dir) if stage_info.output_dir else None,
                "log_file": str(stage_info.log_file) if stage_info.log_file else None,
                "error_message": stage_info.error_message,
                "artifacts": stage_info.artifacts,
            }

        metadata = {
            "run_id": ctx.run_id,
            "status": ctx.status.value,
            "current_stage": ctx.current_stage,
            "start_time": ctx.start_time,
            "end_time": ctx.end_time,
            "seed": ctx.seed,
            "device": ctx.device,
            "git_commit": ctx.git_commit,
            "git_dirty": ctx.git_dirty,
            "environment": ctx.environment,
            "resolved_config": str(ctx.config_dir / "resolved_config.yaml"),
            "artifact_manifest": str(ctx.master_manifest_path),
            "error_log": str(ctx.logs_dir / "error.log")
            if ctx.status == RunStatus.FAILED
            else None,
            "stages": stages_dict,
        }

        # Ensure config dir exists
        ctx.config_dir.mkdir(parents=True, exist_ok=True)

        with ctx.metadata_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(metadata, f, sort_keys=False)

    def get_run_dir(self, run_id: str) -> Path:
        """Get the directory for a run.

        Parameters
        ----------
        run_id : str
            The run identifier

        Returns
        -------
        Path
            The run directory path
        """
        return self.artifacts_root / run_id

    def list_runs(self) -> list[str]:
        """List all run IDs in the artifacts directory.

        Returns
        -------
        list of str
            List of run IDs
        """
        if not self.artifacts_root.exists():
            return []

        return [
            d.name
            for d in self.artifacts_root.iterdir()
            if d.is_dir() and (d / "config" / "run_metadata.yaml").exists()
        ]

    def load_run_context(self, run_id: str) -> RunContext | None:
        """Load an existing run context from disk.

        Parameters
        ----------
        run_id : str
            The run identifier

        Returns
        -------
        RunContext or None
            The loaded context, or None if not found
        """
        run_dir = self.artifacts_root / run_id
        metadata_path = run_dir / "config" / "run_metadata.yaml"

        if not metadata_path.exists():
            return None

        # Load config
        config_path = run_dir / "config" / "resolved_config.yaml"
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
        else:
            config = {}

        return self._resume_run(run_dir, config, force_rerun=False)
