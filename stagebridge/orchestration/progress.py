"""Progress tracking and reporting for StageBridge pipeline orchestration.

This module provides progress bars, status messages, and completion summaries
using tqdm for visual feedback during pipeline execution.
"""

from __future__ import annotations


import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator, Iterator

try:
    from tqdm import tqdm
    from tqdm.auto import tqdm as tqdm_auto

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


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


@dataclass
class StageProgress:
    """Progress tracker for a single pipeline stage."""

    stage_name: str
    total_steps: int = 0
    current_step: int = 0
    status: str = "pending"
    start_time: float | None = None
    end_time: float | None = None
    message: str = ""
    output_dir: Path | None = None
    _pbar: Any = None

    def start(self, total: int = 0, desc: str | None = None) -> None:
        """Start the stage progress.

        Parameters
        ----------
        total : int
            Total number of steps (0 for indeterminate)
        desc : str, optional
            Description override
        """
        self.start_time = time.time()
        self.total_steps = total
        self.current_step = 0
        self.status = "running"

        description = desc or self.stage_name

        if TQDM_AVAILABLE and total > 0:
            self._pbar = tqdm_auto(
                total=total,
                desc=description,
                unit="step",
                ncols=100,
                leave=True,
                file=sys.stdout,
            )
        elif TQDM_AVAILABLE:
            # Indeterminate progress
            self._pbar = tqdm_auto(
                desc=description,
                unit="step",
                ncols=100,
                leave=True,
                file=sys.stdout,
            )

    def update(self, n: int = 1, message: str | None = None) -> None:
        """Update progress.

        Parameters
        ----------
        n : int
            Number of steps to advance
        message : str, optional
            Status message
        """
        self.current_step += n
        if message:
            self.message = message

        if self._pbar is not None:
            self._pbar.update(n)
            if message:
                self._pbar.set_postfix_str(message)

    def set_description(self, desc: str) -> None:
        """Update the progress bar description.

        Parameters
        ----------
        desc : str
            New description
        """
        if self._pbar is not None:
            self._pbar.set_description(desc)

    def complete(self, message: str | None = None) -> float:
        """Mark stage as complete.

        Parameters
        ----------
        message : str, optional
            Completion message

        Returns
        -------
        float
            Duration in seconds
        """
        self.end_time = time.time()
        self.status = "completed"

        if message:
            self.message = message

        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None

        duration = self.duration
        self._print_completion(duration)

        return duration

    def fail(self, error_message: str) -> None:
        """Mark stage as failed.

        Parameters
        ----------
        error_message : str
            Error message
        """
        self.end_time = time.time()
        self.status = "failed"
        self.message = error_message

        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None

        self._print_failure(error_message)

    def skip(self, reason: str = "cached") -> None:
        """Mark stage as skipped.

        Parameters
        ----------
        reason : str
            Reason for skipping
        """
        self.status = "skipped"
        self.message = reason

        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None

        self._print_skip(reason)

    @property
    def duration(self) -> float:
        """Get duration in seconds."""
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.time()
        return end - self.start_time

    def _print_completion(self, duration: float) -> None:
        """Print completion message."""
        duration_str = _format_duration(duration)
        output_str = f" -> {self.output_dir}" if self.output_dir else ""
        print(f"[OK] {self.stage_name} completed in {duration_str}{output_str}")

    def _print_failure(self, error_message: str) -> None:
        """Print failure message."""
        print(f"[FAIL] {self.stage_name} failed: {error_message}")

    def _print_skip(self, reason: str) -> None:
        """Print skip message."""
        print(f"[SKIP] {self.stage_name}: {reason}")


@dataclass
class PipelineProgress:
    """Progress tracker for the entire pipeline."""

    total_stages: int = 0
    completed_stages: int = 0
    failed_stages: int = 0
    skipped_stages: int = 0
    current_stage: str | None = None
    start_time: float | None = None
    end_time: float | None = None
    stages: dict[str, StageProgress] = field(default_factory=dict)
    _pbar: Any = None

    def start(self, stage_names: list[str]) -> None:
        """Start pipeline progress tracking.

        Parameters
        ----------
        stage_names : list of str
            Names of stages to run
        """
        self.total_stages = len(stage_names)
        self.start_time = time.time()
        self.completed_stages = 0
        self.failed_stages = 0
        self.skipped_stages = 0

        print(f"\n{'=' * 60}")
        print(f"StageBridge Pipeline - {self.total_stages} stages")
        print(f"{'=' * 60}\n")

        if TQDM_AVAILABLE:
            self._pbar = tqdm_auto(
                total=self.total_stages,
                desc="Pipeline",
                unit="stage",
                ncols=100,
                leave=True,
                position=0,
                file=sys.stdout,
            )

    def start_stage(
        self,
        stage_name: str,
        total_steps: int = 0,
        output_dir: Path | None = None,
    ) -> StageProgress:
        """Start a new stage.

        Parameters
        ----------
        stage_name : str
            Name of the stage
        total_steps : int
            Number of steps in the stage
        output_dir : Path, optional
            Output directory for the stage

        Returns
        -------
        StageProgress
            The stage progress tracker
        """
        self.current_stage = stage_name

        stage_progress = StageProgress(
            stage_name=stage_name,
            output_dir=output_dir,
        )
        stage_progress.start(total=total_steps)
        self.stages[stage_name] = stage_progress

        return stage_progress

    def complete_stage(self, stage_name: str, message: str | None = None) -> None:
        """Mark a stage as complete.

        Parameters
        ----------
        stage_name : str
            Name of the stage
        message : str, optional
            Completion message
        """
        if stage_name in self.stages:
            self.stages[stage_name].complete(message)

        self.completed_stages += 1

        if self._pbar is not None:
            self._pbar.update(1)
            self._pbar.set_postfix_str(f"Completed: {stage_name}")

    def fail_stage(self, stage_name: str, error_message: str) -> None:
        """Mark a stage as failed.

        Parameters
        ----------
        stage_name : str
            Name of the stage
        error_message : str
            Error message
        """
        if stage_name in self.stages:
            self.stages[stage_name].fail(error_message)
        else:
            stage_progress = StageProgress(stage_name=stage_name)
            stage_progress.fail(error_message)
            self.stages[stage_name] = stage_progress

        self.failed_stages += 1

        if self._pbar is not None:
            self._pbar.update(1)
            self._pbar.set_postfix_str(f"Failed: {stage_name}")

    def skip_stage(self, stage_name: str, reason: str = "cached") -> None:
        """Mark a stage as skipped.

        Parameters
        ----------
        stage_name : str
            Name of the stage
        reason : str
            Reason for skipping
        """
        stage_progress = StageProgress(stage_name=stage_name)
        stage_progress.skip(reason)
        self.stages[stage_name] = stage_progress

        self.skipped_stages += 1

        if self._pbar is not None:
            self._pbar.update(1)
            self._pbar.set_postfix_str(f"Skipped: {stage_name}")

    def finish(self) -> dict[str, Any]:
        """Finish pipeline tracking and print summary.

        Returns
        -------
        dict
            Summary statistics
        """
        self.end_time = time.time()

        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None

        summary = self._build_summary()

        self._print_summary(summary)

        return summary

    @property
    def duration(self) -> float:
        """Get total duration in seconds."""
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.time()
        return end - self.start_time

    def _build_summary(self) -> dict[str, Any]:
        """Build summary statistics."""
        return {
            "total_stages": self.total_stages,
            "completed_stages": self.completed_stages,
            "failed_stages": self.failed_stages,
            "skipped_stages": self.skipped_stages,
            "duration_seconds": self.duration,
            "duration_formatted": _format_duration(self.duration),
            "status": "completed" if self.failed_stages == 0 else "partial",
            "stages": {
                name: {
                    "status": stage.status,
                    "duration_seconds": stage.duration,
                    "message": stage.message,
                }
                for name, stage in self.stages.items()
            },
        }

    def _print_summary(self, summary: dict[str, Any]) -> None:
        """Print the pipeline summary."""
        print(f"\n{'=' * 60}")
        print("Pipeline Summary")
        print(f"{'=' * 60}")
        print(f"Total time: {summary['duration_formatted']}")
        print(f"Stages completed: {summary['completed_stages']}/{summary['total_stages']}")
        if summary["skipped_stages"] > 0:
            print(f"Stages skipped: {summary['skipped_stages']}")
        if summary["failed_stages"] > 0:
            print(f"Stages failed: {summary['failed_stages']}")

        print(f"\nStatus: {summary['status'].upper()}")
        print(f"{'=' * 60}\n")


def get_progress_bar(
    iterable: Iterator | None = None,
    total: int | None = None,
    desc: str = "",
    unit: str = "it",
    leave: bool = True,
    position: int | None = None,
    disable: bool = False,
) -> Any:
    """Get a tqdm progress bar with consistent styling.

    Parameters
    ----------
    iterable : Iterator, optional
        Iterable to wrap
    total : int, optional
        Total count (if iterable doesn't have __len__)
    desc : str
        Description
    unit : str
        Unit name (default: "it")
    leave : bool
        Leave bar after completion (default: True)
    position : int, optional
        Bar position for nested bars
    disable : bool
        Disable the progress bar (default: False)

    Returns
    -------
    tqdm or iterable
        Progress bar wrapping the iterable
    """
    if not TQDM_AVAILABLE or disable:
        if iterable is not None:
            return iterable
        return range(total or 0)

    return tqdm_auto(
        iterable,
        total=total,
        desc=desc,
        unit=unit,
        ncols=100,
        leave=leave,
        position=position,
        file=sys.stdout,
    )


@contextmanager
def stage_progress_context(
    stage_name: str,
    total_steps: int = 0,
    output_dir: Path | None = None,
) -> Generator[StageProgress, None, None]:
    """Context manager for stage progress tracking.

    Parameters
    ----------
    stage_name : str
        Name of the stage
    total_steps : int
        Number of steps (0 for indeterminate)
    output_dir : Path, optional
        Output directory

    Yields
    ------
    StageProgress
        The stage progress tracker
    """
    progress = StageProgress(
        stage_name=stage_name,
        output_dir=output_dir,
    )
    progress.start(total=total_steps)

    try:
        yield progress
        progress.complete()
    except Exception as e:
        progress.fail(str(e))
        raise


def print_stage_header(stage_name: str, stage_number: int, total_stages: int) -> None:
    """Print a stage header.

    Parameters
    ----------
    stage_name : str
        Name of the stage
    stage_number : int
        Stage number (1-indexed)
    total_stages : int
        Total number of stages
    """
    print(f"\n{'─' * 60}")
    print(f"Stage {stage_number}/{total_stages}: {stage_name}")
    print(f"{'─' * 60}")


def print_stage_result(
    stage_name: str,
    success: bool,
    duration: float,
    output_dir: Path | None = None,
    message: str | None = None,
) -> None:
    """Print a stage result message.

    Parameters
    ----------
    stage_name : str
        Name of the stage
    success : bool
        Whether the stage succeeded
    duration : float
        Duration in seconds
    output_dir : Path, optional
        Output directory
    message : str, optional
        Additional message
    """
    duration_str = _format_duration(duration)

    if success:
        status = "[OK]"
        output_str = f" -> {output_dir}" if output_dir else ""
        print(f"{status} {stage_name} completed in {duration_str}{output_str}")
    else:
        status = "[FAIL]"
        msg_str = f": {message}" if message else ""
        print(f"{status} {stage_name} failed after {duration_str}{msg_str}")


def print_error_with_log(
    stage_name: str,
    error_message: str,
    log_path: Path | None = None,
    suggestion: str | None = None,
) -> None:
    """Print an error message with log file pointer.

    Parameters
    ----------
    stage_name : str
        Name of the stage that failed
    error_message : str
        Error message
    log_path : Path, optional
        Path to log file
    suggestion : str, optional
        Suggestion for fixing the error
    """
    print(f"\n[FAIL] Stage '{stage_name}' failed")
    print(f"Error: {error_message}")

    if suggestion:
        print(f"Suggestion: {suggestion}")

    if log_path:
        print(f"Log: {log_path}")

    print()
