"""Lightweight results-system API for Mission 2."""

from .manifest import RunMetadata, RunMetrics
from .milestone import (
    MilestoneArchiveResult,
    MilestonePromotionResult,
    archive_current_scratch_run,
    promote_current_scratch_run,
)
from .registry import (
    ensure_registry_files,
    find_results_registry_row,
    read_milestone_index,
    read_promoted_results,
    read_results_registry,
)
from .result_card import build_result_card
from .run_writer import (
    load_current_scratch_run,
    run_smoke_execution,
    write_pipeline_scratch_run,
    write_scratch_run,
)

__all__ = [
    "MilestoneArchiveResult",
    "MilestonePromotionResult",
    "RunMetadata",
    "archive_current_scratch_run",
    "RunMetrics",
    "build_result_card",
    "ensure_registry_files",
    "find_results_registry_row",
    "load_current_scratch_run",
    "promote_current_scratch_run",
    "read_milestone_index",
    "read_promoted_results",
    "read_results_registry",
    "run_smoke_execution",
    "write_pipeline_scratch_run",
    "write_scratch_run",
]
