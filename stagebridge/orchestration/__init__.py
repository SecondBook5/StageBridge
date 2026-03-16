"""StageBridge orchestration infrastructure.

This package provides the notebook orchestration infrastructure for running
the StageBridge pipeline, including:

- Run management and lifecycle
- Configuration loading and validation
- Artifact tracking and manifests
- Progress reporting with tqdm
- Stage validation and resume logic

Usage
-----
>>> from stagebridge.orchestration import initialize_run, run_data_qc
>>> ctx = initialize_run("configs/default.yaml")
>>> result = run_data_qc(ctx)
"""

from stagebridge.orchestration.artifact_registry import (
    ArtifactInfo,
    ArtifactRegistry,
    StageManifest,
)
from stagebridge.orchestration.config_loader import (
    ConfigValidationError,
    get_enabled_stages,
    is_stage_enabled,
    load_config,
    load_default_config,
    load_smoke_test_config,
    save_config,
    validate_config,
)
from stagebridge.orchestration.notebook_api import (
    RunSummary,
    StageResult,
    initialize_run,
    run_ablations,
    run_baselines,
    run_biology,
    run_data_qc,
    run_full_model,
    run_full_pipeline,
    run_publication_figures,
    run_reference_mapping,
    run_smoke_pipeline,
    run_spatial_backend_benchmark,
    summarize_run,
    validate_stage,
)
from stagebridge.orchestration.progress import (
    PipelineProgress,
    StageProgress,
    get_progress_bar,
    print_error_with_log,
    print_stage_header,
    print_stage_result,
    stage_progress_context,
)
from stagebridge.orchestration.run_manager import (
    RunContext,
    RunManager,
    RunStatus,
    StageInfo,
    StageStatus,
)
from stagebridge.orchestration.validation import (
    ValidationResult,
    check_stage_can_resume,
    format_validation_errors,
    should_run_stage,
    validate_config_for_stage,
    validate_run_artifacts,
    validate_stage_artifacts,
)

__all__ = [
    # Run management
    "RunContext",
    "RunManager",
    "RunStatus",
    "StageInfo",
    "StageStatus",
    # Config
    "ConfigValidationError",
    "get_enabled_stages",
    "is_stage_enabled",
    "load_config",
    "load_default_config",
    "load_smoke_test_config",
    "save_config",
    "validate_config",
    # Artifacts
    "ArtifactInfo",
    "ArtifactRegistry",
    "StageManifest",
    # Progress
    "PipelineProgress",
    "StageProgress",
    "get_progress_bar",
    "print_error_with_log",
    "print_stage_header",
    "print_stage_result",
    "stage_progress_context",
    # Validation
    "ValidationResult",
    "check_stage_can_resume",
    "format_validation_errors",
    "should_run_stage",
    "validate_config_for_stage",
    "validate_run_artifacts",
    "validate_stage_artifacts",
    # Notebook API
    "RunSummary",
    "StageResult",
    "initialize_run",
    "run_ablations",
    "run_baselines",
    "run_biology",
    "run_data_qc",
    "run_full_model",
    "run_full_pipeline",
    "run_publication_figures",
    "run_reference_mapping",
    "run_smoke_pipeline",
    "run_spatial_backend_benchmark",
    "summarize_run",
    "validate_stage",
]
