"""Typed manifest helpers for the lightweight StageBridge results system."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

from omegaconf import DictConfig, OmegaConf


ALLOWED_STATUSES = frozenset({"failed", "partial", "complete", "promoted"})
SCRATCH_CURRENT_RELATIVE = Path("outputs") / "scratch" / "current"
REGISTRY_DIR_RELATIVE = Path("results") / "registry"
MILESTONES_DIR_RELATIVE = Path("results") / "milestones"
RESULTS_REGISTRY_COLUMNS = [
    "timestamp",
    "git_commit",
    "git_short_hash",
    "git_branch",
    "experiment_name",
    "mode",
    "stage_edges",
    "split_name",
    "wes_regularizer_enabled",
    "spatial_mapping_method",
    "context_model_mode",
    "status",
    "primary_metric",
    "promoted",
    "scratch_path",
    "milestone_id",
]
MILESTONE_INDEX_COLUMNS = [
    "milestone_id",
    "timestamp",
    "source_timestamp",
    "git_commit",
    "git_short_hash",
    "git_tag",
    "summary",
    "importance_level",
    "milestone_path",
]
PROMOTED_RESULT_KEYS = [
    "best_rna_only",
    "best_deep_sets",
    "best_deep_sets_transformer_hybrid",
    "best_set_only",
    "best_typed_hierarchical_transformer",
    "best_graph_of_sets",
    "best_graph_of_sets_wes",
    "best_aah_to_ais",
    "best_ais_to_mia",
    "best_full_v1_candidate",
]
DEFAULT_PROMOTED_RESULTS = {key: None for key in PROMOTED_RESULT_KEYS}
DEFAULT_PROMOTED_RESULTS["latest_promoted"] = None


def repo_root(base_dir: str | Path | None = None) -> Path:
    """Resolve the repository root."""
    if base_dir is not None:
        return Path(base_dir).resolve()
    return Path(__file__).resolve().parents[2]


def scratch_current_dir(base_dir: str | Path | None = None) -> Path:
    """Return the reusable scratch directory."""
    return repo_root(base_dir) / SCRATCH_CURRENT_RELATIVE


def registry_dir(base_dir: str | Path | None = None) -> Path:
    """Return the registry directory."""
    return repo_root(base_dir) / REGISTRY_DIR_RELATIVE


def milestones_dir(base_dir: str | Path | None = None) -> Path:
    """Return the milestone root directory."""
    return repo_root(base_dir) / MILESTONES_DIR_RELATIVE


def _cfg_to_dict(cfg: DictConfig | Mapping[str, Any] | None) -> dict[str, Any]:
    if cfg is None:
        return {}
    if isinstance(cfg, DictConfig):
        payload = OmegaConf.to_container(cfg, resolve=True)
        return dict(payload or {})
    return dict(cfg)


def nested_get(data: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    """Read a dotted key path from a nested mapping."""
    current: Any = data
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def normalize_stage_edges(stage_edges: Any) -> list[str]:
    """Normalize stage-edge values into ``['SRC->TGT', ...]`` form."""
    if stage_edges is None:
        return []
    if isinstance(stage_edges, str):
        return [stage_edges]
    if not isinstance(stage_edges, Sequence):
        return [str(stage_edges)]

    normalized: list[str] = []
    for item in stage_edges:
        if isinstance(item, str):
            normalized.append(item)
            continue
        if isinstance(item, Sequence) and len(item) == 2:
            normalized.append(f"{item[0]}->{item[1]}")
            continue
        normalized.append(str(item))
    return normalized


def stage_edges_label(stage_edges: Sequence[str]) -> str:
    """Convert stage edges into a CSV-friendly label."""
    return "|".join(str(edge) for edge in stage_edges)


def parse_stage_edges_label(label: str | None) -> list[str]:
    """Convert a CSV label back to a list of stage-edge strings."""
    if not label:
        return []
    return [part for part in str(label).split("|") if part]


def validate_status(status: str) -> str:
    """Validate a run status value."""
    if status not in ALLOWED_STATUSES:
        allowed = ", ".join(sorted(ALLOWED_STATUSES))
        raise ValueError(f"Unsupported run status '{status}'. Allowed: {allowed}")
    return status


def infer_mode_from_config(cfg: Mapping[str, Any]) -> str:
    """Infer a compact mode label from config content."""
    context_mode = str(nested_get(cfg, "context_model.mode", "rna_only"))
    wes_enabled = bool(nested_get(cfg, "transition_model.wes_regularizer.enabled", False))
    if context_mode == "graph_of_sets" and wes_enabled:
        return "graph_of_sets_wes"
    if context_mode == "graph_of_sets":
        return "graph_of_sets"
    if context_mode == "deep_sets":
        return "deep_sets"
    if context_mode == "set_only":
        return "set_only"
    if context_mode == "deep_sets_transformer_hybrid":
        return "deep_sets_transformer_hybrid"
    if context_mode == "typed_hierarchical_transformer":
        return "typed_hierarchical_transformer"
    return "rna_only"


@dataclass(slots=True, frozen=True)
class GitContext:
    """Lightweight git identity for one recorded run."""

    git_commit: str
    git_short_hash: str
    git_branch: str


@dataclass(slots=True)
class RunMetadata:
    """Required scratch-run metadata schema."""

    timestamp: str
    git_commit: str
    git_short_hash: str
    git_branch: str
    experiment_name: str
    mode: str
    stage_edges: list[str]
    seed: int
    split_name: str
    wes_regularizer_enabled: bool
    spatial_mapping_method: str
    context_model_mode: str
    notebook_source: str
    status: str

    def to_dict(self) -> dict[str, Any]:
        validate_status(self.status)
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RunMetadata":
        return cls(
            timestamp=str(payload["timestamp"]),
            git_commit=str(payload["git_commit"]),
            git_short_hash=str(payload["git_short_hash"]),
            git_branch=str(payload["git_branch"]),
            experiment_name=str(payload["experiment_name"]),
            mode=str(payload["mode"]),
            stage_edges=normalize_stage_edges(payload.get("stage_edges", [])),
            seed=int(payload["seed"]),
            split_name=str(payload["split_name"]),
            wes_regularizer_enabled=bool(payload["wes_regularizer_enabled"]),
            spatial_mapping_method=str(payload["spatial_mapping_method"]),
            context_model_mode=str(payload["context_model_mode"]),
            notebook_source=str(payload["notebook_source"]),
            status=validate_status(str(payload["status"])),
        )


@dataclass(slots=True)
class RunMetrics:
    """Structured scratch metrics payload."""

    primary_metric: float | int | str | None = None
    secondary_metrics: dict[str, Any] = field(default_factory=dict)
    calibration: dict[str, Any] = field(default_factory=dict)
    ablation_label: str | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RunMetrics":
        return cls(
            primary_metric=payload.get("primary_metric"),
            secondary_metrics=dict(payload.get("secondary_metrics", {})),
            calibration=dict(payload.get("calibration", {})),
            ablation_label=payload.get("ablation_label"),
            notes=payload.get("notes"),
        )


def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def git_context(base_dir: str | Path | None = None) -> GitContext:
    """Resolve git metadata for the repository."""
    root = repo_root(base_dir)

    def _run_git(args: list[str], fallback: str) -> str:
        try:
            completed = subprocess.run(
                ["git", *args],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )
        except Exception:
            return fallback
        value = completed.stdout.strip()
        return value or fallback

    commit = _run_git(["rev-parse", "HEAD"], "unknown")
    short_hash = _run_git(["rev-parse", "--short", "HEAD"], "unknown")
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], "unknown")
    return GitContext(git_commit=commit, git_short_hash=short_hash, git_branch=branch)


def build_run_metadata(
    cfg: DictConfig | Mapping[str, Any] | None,
    *,
    experiment_name: str | None = None,
    mode: str | None = None,
    stage_edges: Sequence[str] | None = None,
    seed: int | None = None,
    split_name: str | None = None,
    notebook_source: str = "StageBridge.ipynb",
    status: str = "complete",
    timestamp: str | None = None,
    base_dir: str | Path | None = None,
) -> RunMetadata:
    """Build run metadata from config and explicit overrides."""
    data = _cfg_to_dict(cfg)
    git = git_context(base_dir)
    resolved_stage_edges = normalize_stage_edges(
        stage_edges
        if stage_edges is not None
        else nested_get(data, "transition_model.disease_edges", [])
    )
    resolved_seed = int(
        seed if seed is not None else nested_get(data, "train.seed", nested_get(data, "seed", 42))
    )
    resolved_split = str(
        split_name
        if split_name is not None
        else nested_get(data, "splits.name", "unspecified_split")
    )
    resolved_experiment = str(
        experiment_name
        if experiment_name is not None
        else nested_get(data, "run_name", "stagebridge")
    )
    resolved_mode = str(mode if mode is not None else infer_mode_from_config(data))
    return RunMetadata(
        timestamp=str(timestamp or utc_timestamp()),
        git_commit=git.git_commit,
        git_short_hash=git.git_short_hash,
        git_branch=git.git_branch,
        experiment_name=resolved_experiment,
        mode=resolved_mode,
        stage_edges=resolved_stage_edges,
        seed=resolved_seed,
        split_name=resolved_split,
        wes_regularizer_enabled=bool(
            nested_get(data, "transition_model.wes_regularizer.enabled", False)
        ),
        spatial_mapping_method=str(
            nested_get(data, "spatial_mapping.method", "unspecified_mapping")
        ),
        context_model_mode=str(nested_get(data, "context_model.mode", "rna_only")),
        notebook_source=notebook_source,
        status=validate_status(status),
    )


def build_smoke_metrics(completed_steps: int, failed_steps: int) -> RunMetrics:
    """Return an honest infrastructure-smoke metrics payload."""
    return RunMetrics(
        primary_metric=None,
        secondary_metrics={
            "completed_steps": completed_steps,
            "failed_steps": failed_steps,
        },
        calibration={},
        ablation_label="smoke_infrastructure",
        notes="Infrastructure smoke only. No scientific metric was computed.",
    )
