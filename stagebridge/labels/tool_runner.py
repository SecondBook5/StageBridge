"""Common external-tool execution helpers for label-repair wrappers."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

from stagebridge.labels.common_schema import ToolCommand, ToolExecutionResult


def resolve_executable(executable: str) -> Path | None:
    """Resolve an executable from PATH or as an absolute filesystem path.

    Args:
        executable: Executable name or absolute path.
    """
    candidate = Path(str(executable))
    if candidate.is_absolute():
        return candidate if candidate.exists() else None
    resolved = shutil.which(str(executable))
    return None if resolved is None else Path(resolved)


def run_external_command(
    command: ToolCommand,
    *,
    dry_run: bool = False,
    resume: bool = True,
) -> ToolExecutionResult:
    """Execute one external-tool command with explicit diagnostics.

    Args:
        command: Structured command specification.
        dry_run: If true, do not execute the tool and return a dry-run record.
        resume: If true and the log already exists, append instead of overwriting.
    """
    resolved = resolve_executable(command.executable)
    argv = (str(command.executable), *[str(arg) for arg in command.args])
    if resolved is None:
        return ToolExecutionResult(
            command=argv,
            return_code=None,
            stdout_path=command.log_path,
            status="missing_executable",
            message=f"Executable not found: {command.executable}",
            backend_trace=f"{command.name}:missing_executable",
        )
    if dry_run:
        return ToolExecutionResult(
            command=(str(resolved), *[str(arg) for arg in command.args]),
            return_code=None,
            stdout_path=command.log_path,
            status="dry_run",
            message="Dry-run mode requested; command not executed.",
            backend_trace=f"{command.name}:dry_run",
        )

    command.workdir.mkdir(parents=True, exist_ok=True)
    log_path = command.log_path
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    if command.env:
        env.update({str(key): str(value) for key, value in command.env.items()})
    attempts = max(int(command.retries), 0) + 1
    mode = "a" if resume else "w"
    for attempt in range(attempts):
        try:
            if log_path is not None:
                with log_path.open(mode, encoding="utf-8") as handle:
                    handle.write(f"$ {' '.join((str(resolved), *map(str, command.args)))}\n")
                    completed = subprocess.run(
                        [str(resolved), *[str(arg) for arg in command.args]],
                        cwd=str(command.workdir),
                        env=env,
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        timeout=int(command.timeout_seconds),
                        check=False,
                        text=True,
                    )
            else:
                completed = subprocess.run(
                    [str(resolved), *[str(arg) for arg in command.args]],
                    cwd=str(command.workdir),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=int(command.timeout_seconds),
                    check=False,
                    text=True,
                )
            if completed.returncode == 0:
                return ToolExecutionResult(
                    command=(str(resolved), *[str(arg) for arg in command.args]),
                    return_code=int(completed.returncode),
                    stdout_path=log_path,
                    status="complete",
                    message="Command completed successfully.",
                    backend_trace=f"{command.name}:executed",
                )
        except subprocess.TimeoutExpired as exc:
            if attempt + 1 == attempts:
                return ToolExecutionResult(
                    command=(str(resolved), *[str(arg) for arg in command.args]),
                    return_code=None,
                    stdout_path=log_path,
                    status="failed",
                    message=f"Command timed out after {exc.timeout} seconds.",
                    backend_trace=f"{command.name}:timeout",
                )
    return ToolExecutionResult(
        command=(str(resolved), *[str(arg) for arg in command.args]),
        return_code=1,
        stdout_path=log_path,
        status="failed",
        message="Command returned a non-zero exit code.",
        backend_trace=f"{command.name}:failed",
    )
