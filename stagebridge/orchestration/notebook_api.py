"""High-level notebook-facing API for StageBridge pipeline orchestration.

This module provides clean, notebook-friendly functions for running
the StageBridge pipeline with progress tracking, validation, and
artifact management.
"""

from __future__ import annotations

import logging
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stagebridge.orchestration.artifact_registry import ArtifactRegistry
from stagebridge.orchestration.config_loader import (
    get_enabled_stages,
    load_config,
    load_smoke_test_config,
)
from stagebridge.orchestration.progress import (
    print_error_with_log,
    print_stage_header,
)
from stagebridge.orchestration.run_manager import (
    RunContext,
    RunManager,
    StageStatus,
)
from stagebridge.orchestration.validation import (
    ValidationResult,
    check_stage_can_resume,
    format_validation_errors,
    validate_stage_artifacts,
)


# Setup logging
_logger = logging.getLogger(__name__)


@dataclass
class StageResult:
    """Result from running a single pipeline stage."""

    stage_name: str
    success: bool
    skipped: bool = False
    skip_reason: str | None = None
    duration_seconds: float = 0.0
    output_dir: Path | None = None
    artifacts: list[str] = field(default_factory=list)
    error_message: str | None = None
    log_path: Path | None = None
    result_data: dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        """Return True if stage succeeded or was skipped."""
        return self.success or self.skipped


@dataclass
class RunSummary:
    """Summary of a complete pipeline run."""

    run_id: str
    status: str
    total_stages: int
    completed_stages: int
    failed_stages: int
    skipped_stages: int
    duration_seconds: float
    duration_formatted: str
    run_dir: Path
    stages: dict[str, dict[str, Any]] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _format_duration(seconds: float) -> str:
    """Format duration in human-readable form."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def _setup_stage_logging(ctx: RunContext, stage_name: str) -> logging.FileHandler | None:
    """Setup logging to stage log file."""
    log_path = ctx.stage_log(stage_name)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
        logging.getLogger().addHandler(handler)
        return handler
    except Exception:
        return None


def _teardown_stage_logging(handler: logging.FileHandler | None) -> None:
    """Remove stage log handler."""
    if handler:
        logging.getLogger().removeHandler(handler)
        handler.close()


# Global run manager instance
_run_manager: RunManager | None = None


def get_run_manager(artifacts_root: str | Path = "artifacts/runs") -> RunManager:
    """Get or create the global run manager.

    Parameters
    ----------
    artifacts_root : str or Path
        Root directory for artifacts

    Returns
    -------
    RunManager
        The run manager instance
    """
    global _run_manager
    if _run_manager is None:
        _run_manager = RunManager(artifacts_root=artifacts_root)
    return _run_manager


def initialize_run(
    config: dict[str, Any] | str | Path | None = None,
    *,
    resume_if_possible: bool = True,
    run_id: str | None = None,
    artifacts_root: str | Path = "artifacts/runs",
) -> RunContext:
    """Initialize a new pipeline run or resume an existing one.

    Parameters
    ----------
    config : dict, str, Path, or None
        Configuration source (dict, YAML path, or None for defaults)
    resume_if_possible : bool
        Whether to resume if run exists (default: True)
    run_id : str, optional
        Explicit run ID (auto-generated if not provided)
    artifacts_root : str or Path
        Root directory for artifacts (default: artifacts/runs)

    Returns
    -------
    RunContext
        The initialized run context
    """
    # Load and merge config
    resolved_config = load_config(config, validate=True)

    # Override resume setting
    resolved_config["resume_if_possible"] = resume_if_possible

    # Get or set run_id
    if run_id is None:
        run_id = resolved_config.get("run_id")

    # Get run manager
    manager = get_run_manager(artifacts_root)

    # Initialize run
    ctx = manager.initialize_run(
        resolved_config,
        run_id=run_id,
        resume_if_possible=resume_if_possible,
        force_rerun=resolved_config.get("force_rerun", False),
    )

    print(f"\nInitialized run: {ctx.run_id}")
    print(f"Run directory: {ctx.run_dir}")
    print(f"Resume mode: {'enabled' if resume_if_possible else 'disabled'}")
    print()

    return ctx


def _run_stage_impl(
    ctx: RunContext,
    stage_name: str,
    stage_func: Any,
    *,
    force_rerun: bool = False,
    stage_number: int = 0,
    total_stages: int = 0,
) -> StageResult:
    """Internal implementation for running a stage.

    Parameters
    ----------
    ctx : RunContext
        The run context
    stage_name : str
        Name of the stage
    stage_func : callable
        Function to execute the stage
    force_rerun : bool
        Force rerun even if cached
    stage_number : int
        Stage number for display
    total_stages : int
        Total stages for display

    Returns
    -------
    StageResult
        The stage result
    """
    manager = get_run_manager()
    registry = ArtifactRegistry(ctx.run_dir)

    # Check if we should run
    actual_force = force_rerun or ctx.force_rerun
    if not actual_force:
        can_resume, reason = check_stage_can_resume(ctx, stage_name)
        if can_resume:
            manager.skip_stage(ctx, stage_name, reason)
            return StageResult(
                stage_name=stage_name,
                success=True,
                skipped=True,
                skip_reason=reason,
                output_dir=ctx.stage_dir(stage_name),
            )

    # Print stage header
    if stage_number > 0:
        print_stage_header(stage_name, stage_number, total_stages)

    # Setup logging
    log_handler = _setup_stage_logging(ctx, stage_name)
    log_path = ctx.stage_log(stage_name)

    # Start stage tracking
    stage_info = manager.start_stage(ctx, stage_name)
    start_time = time.time()

    try:
        # Run the stage
        _logger.info(f"Starting stage: {stage_name}")
        result_data = stage_func(ctx)
        _logger.info(f"Stage completed: {stage_name}")

        # Record duration
        duration = time.time() - start_time

        # Register artifacts
        stage_dir = ctx.stage_dir(stage_name)
        artifacts = registry.register_artifacts_from_dir(stage_dir, stage_name)
        artifact_names = [a.name for a in artifacts]

        # Mark stage complete
        registry.mark_stage_complete(stage_name, stage_dir)
        registry.create_stage_manifest(
            stage_name,
            "completed",
            start_time=stage_info.start_time,
            end_time=datetime.now(timezone.utc).isoformat(),
            duration_seconds=duration,
        )

        manager.complete_stage(ctx, stage_name, artifact_names)

        return StageResult(
            stage_name=stage_name,
            success=True,
            duration_seconds=duration,
            output_dir=stage_dir,
            artifacts=artifact_names,
            log_path=log_path,
            result_data=result_data if isinstance(result_data, dict) else {},
        )

    except Exception as e:
        duration = time.time() - start_time
        error_msg = str(e)

        _logger.error(f"Stage failed: {stage_name} - {error_msg}")
        _logger.error(traceback.format_exc())

        # Mark stage failed
        manager.fail_stage(ctx, stage_name, error_msg)

        # Print error
        print_error_with_log(
            stage_name,
            error_msg,
            log_path=log_path,
        )

        return StageResult(
            stage_name=stage_name,
            success=False,
            duration_seconds=duration,
            output_dir=ctx.stage_dir(stage_name),
            error_message=error_msg,
            log_path=log_path,
        )

    finally:
        _teardown_stage_logging(log_handler)


# Stage implementations - these are placeholder implementations
# that should be replaced with actual pipeline logic


def _run_data_qc_impl(ctx: RunContext) -> dict[str, Any]:
    """Run data QC stage."""
    # This would call the actual data QC functions
    # For now, create placeholder outputs
    import json

    stage_dir = ctx.stage_dir("data_qc")
    stage_dir.mkdir(parents=True, exist_ok=True)

    qc_report = {
        "status": "completed",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": ctx.seed,
    }

    (stage_dir / "qc_report.json").write_text(json.dumps(qc_report, indent=2), encoding="utf-8")
    (stage_dir / "qc_summary.html").write_text(
        "<html><body><h1>QC Summary</h1></body></html>", encoding="utf-8"
    )

    return {"status": "completed", "qc_report": qc_report}


def _run_reference_impl(ctx: RunContext) -> dict[str, Any]:
    """Run reference preparation stage."""
    import json

    stage_dir = ctx.stage_dir("reference")
    stage_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "status": "completed",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Create placeholder outputs
    (stage_dir / "reference_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    # Note: reference_mapping.h5ad would be created by actual reference code
    # For smoke test, we create a placeholder marker
    (stage_dir / "reference_mapping.h5ad.placeholder").write_text("placeholder", encoding="utf-8")

    return {"status": "completed", "metrics": metrics}


def _run_spatial_backend_impl(ctx: RunContext) -> dict[str, Any]:
    """Run spatial backend benchmark stage."""
    import json

    stage_dir = ctx.stage_dir("spatial_backend")
    stage_dir.mkdir(parents=True, exist_ok=True)

    backends = ctx.config.get("spatial_backends", ["tangram"])
    benchmark = {
        "status": "completed",
        "backends_tested": backends,
        "selected": backends[0] if backends else "tangram",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    (stage_dir / "backend_benchmark.json").write_text(
        json.dumps(benchmark, indent=2), encoding="utf-8"
    )
    (stage_dir / "selected_backend.txt").write_text(benchmark["selected"], encoding="utf-8")

    return {"status": "completed", "benchmark": benchmark}


def _run_baselines_impl(ctx: RunContext) -> dict[str, Any]:
    """Run architecture baselines stage."""
    import json

    stage_dir = ctx.stage_dir("baselines")
    stage_dir.mkdir(parents=True, exist_ok=True)

    baselines = ctx.config.get("baselines", ["mlp", "gcn"])
    results = {
        "status": "completed",
        "baselines_tested": baselines,
        "results": {b: {"metric": 0.0} for b in baselines},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    (stage_dir / "baseline_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    return {"status": "completed", "results": results}


def _run_full_model_impl(ctx: RunContext) -> dict[str, Any]:
    """Run full model training stage."""
    import json

    stage_dir = ctx.stage_dir("full_model")
    stage_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "status": "completed",
        "final_loss": 0.0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    (stage_dir / "training_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    # Placeholder for model checkpoint
    (stage_dir / "model_checkpoint.pt.placeholder").write_text("placeholder", encoding="utf-8")

    return {"status": "completed", "metrics": metrics}


def _run_ablations_impl(ctx: RunContext) -> dict[str, Any]:
    """Run ablation studies stage."""
    import json

    stage_dir = ctx.stage_dir("ablations")
    stage_dir.mkdir(parents=True, exist_ok=True)

    ablations = ctx.config.get("ablations", ["no_spatial", "no_attention"])
    results = {
        "status": "completed",
        "ablations_tested": ablations,
        "results": {a: {"delta": 0.0} for a in ablations},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    (stage_dir / "ablation_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    return {"status": "completed", "results": results}


def _run_biology_impl(ctx: RunContext) -> dict[str, Any]:
    """Run biological validation stage."""
    import json

    stage_dir = ctx.stage_dir("biology")
    stage_dir.mkdir(parents=True, exist_ok=True)

    validation = {
        "status": "completed",
        "biological_metrics": {},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    (stage_dir / "biology_validation.json").write_text(
        json.dumps(validation, indent=2), encoding="utf-8"
    )

    return {"status": "completed", "validation": validation}


def _run_figures_impl(ctx: RunContext) -> dict[str, Any]:
    """Run publication figures stage."""
    import json

    stage_dir = ctx.stage_dir("figures")
    stage_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "status": "completed",
        "figures": [],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    (stage_dir / "figures_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    return {"status": "completed", "manifest": manifest}


# Public API functions


def run_data_qc(ctx: RunContext, *, force_rerun: bool = False) -> StageResult:
    """Run the data loading and QC stage.

    Parameters
    ----------
    ctx : RunContext
        The run context
    force_rerun : bool
        Force rerun even if cached (default: False)

    Returns
    -------
    StageResult
        The stage result
    """
    return _run_stage_impl(ctx, "data_qc", _run_data_qc_impl, force_rerun=force_rerun)


def run_reference_mapping(ctx: RunContext, *, force_rerun: bool = False) -> StageResult:
    """Run the reference preparation stage.

    Parameters
    ----------
    ctx : RunContext
        The run context
    force_rerun : bool
        Force rerun even if cached (default: False)

    Returns
    -------
    StageResult
        The stage result
    """
    return _run_stage_impl(ctx, "reference", _run_reference_impl, force_rerun=force_rerun)


def run_spatial_backend_benchmark(ctx: RunContext, *, force_rerun: bool = False) -> StageResult:
    """Run the spatial backend benchmark stage.

    Parameters
    ----------
    ctx : RunContext
        The run context
    force_rerun : bool
        Force rerun even if cached (default: False)

    Returns
    -------
    StageResult
        The stage result
    """
    return _run_stage_impl(
        ctx, "spatial_backend", _run_spatial_backend_impl, force_rerun=force_rerun
    )


def run_baselines(ctx: RunContext, *, force_rerun: bool = False) -> StageResult:
    """Run the architecture baselines stage.

    Parameters
    ----------
    ctx : RunContext
        The run context
    force_rerun : bool
        Force rerun even if cached (default: False)

    Returns
    -------
    StageResult
        The stage result
    """
    return _run_stage_impl(ctx, "baselines", _run_baselines_impl, force_rerun=force_rerun)


def run_full_model(ctx: RunContext, *, force_rerun: bool = False) -> StageResult:
    """Run the full model training stage.

    Parameters
    ----------
    ctx : RunContext
        The run context
    force_rerun : bool
        Force rerun even if cached (default: False)

    Returns
    -------
    StageResult
        The stage result
    """
    return _run_stage_impl(ctx, "full_model", _run_full_model_impl, force_rerun=force_rerun)


def run_ablations(ctx: RunContext, *, force_rerun: bool = False) -> StageResult:
    """Run the ablation studies stage.

    Parameters
    ----------
    ctx : RunContext
        The run context
    force_rerun : bool
        Force rerun even if cached (default: False)

    Returns
    -------
    StageResult
        The stage result
    """
    return _run_stage_impl(ctx, "ablations", _run_ablations_impl, force_rerun=force_rerun)


def run_biology(ctx: RunContext, *, force_rerun: bool = False) -> StageResult:
    """Run the biological validation stage.

    Parameters
    ----------
    ctx : RunContext
        The run context
    force_rerun : bool
        Force rerun even if cached (default: False)

    Returns
    -------
    StageResult
        The stage result
    """
    return _run_stage_impl(ctx, "biology", _run_biology_impl, force_rerun=force_rerun)


def run_publication_figures(ctx: RunContext, *, force_rerun: bool = False) -> StageResult:
    """Run the publication figures stage.

    Parameters
    ----------
    ctx : RunContext
        The run context
    force_rerun : bool
        Force rerun even if cached (default: False)

    Returns
    -------
    StageResult
        The stage result
    """
    return _run_stage_impl(ctx, "figures", _run_figures_impl, force_rerun=force_rerun)


def validate_stage(ctx: RunContext, stage_name: str) -> ValidationResult:
    """Validate artifacts for a stage.

    Parameters
    ----------
    ctx : RunContext
        The run context
    stage_name : str
        Name of the stage to validate

    Returns
    -------
    ValidationResult
        The validation result
    """
    result = validate_stage_artifacts(ctx, stage_name)

    if result.success:
        print(f"[OK] Stage '{stage_name}' validation passed")
    else:
        print(format_validation_errors(result, ctx.stage_log(stage_name)))

    return result


def summarize_run(ctx: RunContext) -> RunSummary:
    """Generate a human-readable summary of the run.

    Parameters
    ----------
    ctx : RunContext
        The run context

    Returns
    -------
    RunSummary
        The run summary
    """
    manager = get_run_manager()

    # Finalize run
    failed = any(s.status == StageStatus.FAILED for s in ctx.stages.values())
    manager.finalize_run(ctx, success=not failed)

    # Build summary
    completed = sum(1 for s in ctx.stages.values() if s.status == StageStatus.COMPLETED)
    failed_count = sum(1 for s in ctx.stages.values() if s.status == StageStatus.FAILED)
    skipped = sum(1 for s in ctx.stages.values() if s.status == StageStatus.SKIPPED)

    # Calculate duration
    if ctx.start_time and ctx.end_time:
        start = datetime.fromisoformat(ctx.start_time)
        end = datetime.fromisoformat(ctx.end_time)
        duration = (end - start).total_seconds()
    else:
        duration = 0.0

    # Collect errors
    errors = []
    for stage_name, stage_info in ctx.stages.items():
        if stage_info.status == StageStatus.FAILED and stage_info.error_message:
            errors.append(f"{stage_name}: {stage_info.error_message}")

    summary = RunSummary(
        run_id=ctx.run_id,
        status=ctx.status.value,
        total_stages=len(ctx.stages),
        completed_stages=completed,
        failed_stages=failed_count,
        skipped_stages=skipped,
        duration_seconds=duration,
        duration_formatted=_format_duration(duration),
        run_dir=ctx.run_dir,
        stages={
            name: {
                "status": info.status.value,
                "duration_seconds": info.duration_seconds,
                "output_dir": str(info.output_dir) if info.output_dir else None,
            }
            for name, info in ctx.stages.items()
        },
        errors=errors,
    )

    # Print summary
    _print_run_summary(summary)

    # Save master manifest
    registry = ArtifactRegistry(ctx.run_dir)
    registry.save_master_manifest(
        ctx.run_id,
        ctx.status.value,
        start_time=ctx.start_time,
        end_time=ctx.end_time,
    )

    return summary


def _print_run_summary(summary: RunSummary) -> None:
    """Print a formatted run summary."""
    print(f"\n{'=' * 60}")
    print("Run Summary")
    print(f"{'=' * 60}")
    print(f"Run ID: {summary.run_id}")
    print(f"Status: {summary.status.upper()}")
    print(f"Duration: {summary.duration_formatted}")
    print(f"Run directory: {summary.run_dir}")
    print()
    print(
        f"Stages: {summary.completed_stages} completed, {summary.skipped_stages} skipped, {summary.failed_stages} failed"
    )

    if summary.errors:
        print("\nErrors:")
        for error in summary.errors:
            print(f"  - {error}")

    print(f"{'=' * 60}\n")


def run_full_pipeline(
    config: dict[str, Any] | str | Path | None = None,
    *,
    resume_if_possible: bool = True,
    run_id: str | None = None,
) -> RunSummary:
    """Run the complete StageBridge pipeline.

    Parameters
    ----------
    config : dict, str, Path, or None
        Configuration source
    resume_if_possible : bool
        Whether to resume if run exists (default: True)
    run_id : str, optional
        Explicit run ID

    Returns
    -------
    RunSummary
        The run summary
    """
    # Initialize run
    ctx = initialize_run(
        config,
        resume_if_possible=resume_if_possible,
        run_id=run_id,
    )

    # Get enabled stages
    enabled = get_enabled_stages(ctx.config)

    # Stage functions
    stage_functions = {
        "data_qc": run_data_qc,
        "reference": run_reference_mapping,
        "spatial_backend": run_spatial_backend_benchmark,
        "baselines": run_baselines,
        "full_model": run_full_model,
        "ablations": run_ablations,
        "biology": run_biology,
        "figures": run_publication_figures,
    }

    # Run each enabled stage
    total = len(enabled)
    for i, stage_name in enumerate(enabled, 1):
        if stage_name in stage_functions:
            print_stage_header(stage_name, i, total)
            result = stage_functions[stage_name](ctx)

            if not result and not result.skipped:
                print(f"\nPipeline stopped due to stage failure: {stage_name}")
                break

    # Generate summary
    return summarize_run(ctx)


def run_smoke_pipeline() -> RunSummary:
    """Run a minimal smoke test pipeline.

    Returns
    -------
    RunSummary
        The run summary
    """
    config = load_smoke_test_config()
    return run_full_pipeline(
        config,
        resume_if_possible=False,
        run_id="smoke_test",
    )
