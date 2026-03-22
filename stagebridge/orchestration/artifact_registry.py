"""Artifact tracking and manifest generation for StageBridge orchestration.

This module provides artifact registration, manifest generation,
and artifact validation for pipeline runs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from stagebridge.results.manifest import utc_timestamp


@dataclass
class ArtifactInfo:
    """Information about a single artifact."""

    name: str
    path: str
    stage: str
    artifact_type: str
    size_bytes: int | None = None
    checksum: str | None = None
    created_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "path": self.path,
            "stage": self.stage,
            "artifact_type": self.artifact_type,
            "size_bytes": self.size_bytes,
            "checksum": self.checksum,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArtifactInfo":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            path=data["path"],
            stage=data["stage"],
            artifact_type=data["artifact_type"],
            size_bytes=data.get("size_bytes"),
            checksum=data.get("checksum"),
            created_at=data.get("created_at"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class StageManifest:
    """Manifest for artifacts from a single stage."""

    stage_name: str
    status: str
    start_time: str | None = None
    end_time: str | None = None
    duration_seconds: float | None = None
    artifacts: list[ArtifactInfo] = field(default_factory=list)
    expected_artifacts: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "stage_name": self.stage_name,
            "status": self.status,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": self.duration_seconds,
            "artifacts": [a.to_dict() for a in self.artifacts],
            "expected_artifacts": self.expected_artifacts,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StageManifest":
        """Create from dictionary."""
        return cls(
            stage_name=data["stage_name"],
            status=data["status"],
            start_time=data.get("start_time"),
            end_time=data.get("end_time"),
            duration_seconds=data.get("duration_seconds"),
            artifacts=[ArtifactInfo.from_dict(a) for a in data.get("artifacts", [])],
            expected_artifacts=data.get("expected_artifacts", []),
            metadata=data.get("metadata", {}),
        )


# Expected artifacts by stage
EXPECTED_ARTIFACTS: dict[str, list[str]] = {
    "data_qc": [
        "qc_report.json",
        "qc_summary.html",
        "cell_counts.csv",
    ],
    "reference": [
        "reference_mapping.h5ad",
        "reference_metrics.json",
    ],
    "spatial_backend": [
        "backend_benchmark.json",
        "backend_comparison.csv",
        "selected_backend.txt",
    ],
    "baselines": [
        "baseline_results.json",
        "baseline_comparison.csv",
    ],
    "full_model": [
        "model_checkpoint.pt",
        "training_metrics.json",
        "training_curves.csv",
    ],
    "ablations": [
        "ablation_results.json",
        "ablation_comparison.csv",
    ],
    "biology": [
        "biology_validation.json",
        "biological_metrics.csv",
    ],
    "figures": [
        "figures_manifest.json",
    ],
}


def _compute_checksum(path: Path, algorithm: str = "sha256") -> str | None:
    """Compute file checksum."""
    if not path.exists() or not path.is_file():
        return None

    try:
        hasher = hashlib.new(algorithm)
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return f"{algorithm}:{hasher.hexdigest()}"
    except Exception:
        return None


def _get_file_size(path: Path) -> int | None:
    """Get file size in bytes."""
    try:
        if path.exists() and path.is_file():
            return path.stat().st_size
    except Exception:
        pass
    return None


class ArtifactRegistry:
    """Registry for tracking artifacts across pipeline stages.

    This class provides:
    - Artifact registration and tracking
    - Manifest generation
    - Validation of expected artifacts
    """

    def __init__(self, run_dir: Path | str) -> None:
        """Initialize the registry.

        Parameters
        ----------
        run_dir : Path or str
            The run directory
        """
        self.run_dir = Path(run_dir)
        self.manifests_dir = self.run_dir / "manifests"
        self.manifests_dir.mkdir(parents=True, exist_ok=True)

        self._artifacts: dict[str, list[ArtifactInfo]] = {}
        self._stage_manifests: dict[str, StageManifest] = {}

        # Load existing manifests
        self._load_manifests()

    def _load_manifests(self) -> None:
        """Load existing manifests from disk."""
        master_path = self.manifests_dir / "master_manifest.json"
        if master_path.exists():
            try:
                with master_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    for stage_data in data.get("stages", {}).values():
                        manifest = StageManifest.from_dict(stage_data)
                        self._stage_manifests[manifest.stage_name] = manifest
                        self._artifacts[manifest.stage_name] = manifest.artifacts
            except Exception:
                pass

    def register_artifact(
        self,
        name: str,
        path: str | Path,
        stage: str,
        artifact_type: str = "file",
        compute_checksum: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactInfo:
        """Register an artifact.

        Parameters
        ----------
        name : str
            Artifact name
        path : str or Path
            Path to the artifact
        stage : str
            Stage that produced the artifact
        artifact_type : str
            Type of artifact (file, directory, etc.)
        compute_checksum : bool
            Whether to compute checksum (default: True)
        metadata : dict, optional
            Additional metadata

        Returns
        -------
        ArtifactInfo
            The registered artifact info
        """
        path = Path(path)

        artifact = ArtifactInfo(
            name=name,
            path=str(path),
            stage=stage,
            artifact_type=artifact_type,
            size_bytes=_get_file_size(path),
            checksum=_compute_checksum(path) if compute_checksum else None,
            created_at=utc_timestamp(),
            metadata=metadata or {},
        )

        if stage not in self._artifacts:
            self._artifacts[stage] = []
        self._artifacts[stage].append(artifact)

        return artifact

    def register_artifacts_from_dir(
        self,
        directory: Path | str,
        stage: str,
        *,
        pattern: str = "*",
        compute_checksums: bool = True,
    ) -> list[ArtifactInfo]:
        """Register all files in a directory as artifacts.

        Parameters
        ----------
        directory : Path or str
            Directory to scan
        stage : str
            Stage that produced the artifacts
        pattern : str
            Glob pattern for files (default: "*")
        compute_checksums : bool
            Whether to compute checksums (default: True)

        Returns
        -------
        list of ArtifactInfo
            List of registered artifacts
        """
        directory = Path(directory)
        artifacts: list[ArtifactInfo] = []

        if not directory.exists():
            return artifacts

        for path in directory.glob(pattern):
            if path.is_file():
                artifact = self.register_artifact(
                    name=path.name,
                    path=path,
                    stage=stage,
                    artifact_type="file",
                    compute_checksum=compute_checksums,
                )
                artifacts.append(artifact)

        return artifacts

    def get_stage_artifacts(self, stage: str) -> list[ArtifactInfo]:
        """Get all artifacts for a stage.

        Parameters
        ----------
        stage : str
            Stage name

        Returns
        -------
        list of ArtifactInfo
            List of artifacts
        """
        return self._artifacts.get(stage, [])

    def create_stage_manifest(
        self,
        stage: str,
        status: str,
        *,
        start_time: str | None = None,
        end_time: str | None = None,
        duration_seconds: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> StageManifest:
        """Create a manifest for a stage.

        Parameters
        ----------
        stage : str
            Stage name
        status : str
            Stage status
        start_time : str, optional
            Start time
        end_time : str, optional
            End time
        duration_seconds : float, optional
            Duration in seconds
        metadata : dict, optional
            Additional metadata

        Returns
        -------
        StageManifest
            The stage manifest
        """
        manifest = StageManifest(
            stage_name=stage,
            status=status,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration_seconds,
            artifacts=self._artifacts.get(stage, []),
            expected_artifacts=EXPECTED_ARTIFACTS.get(stage, []),
            metadata=metadata or {},
        )

        self._stage_manifests[stage] = manifest

        # Save stage manifest
        stage_manifest_path = self.manifests_dir / f"{stage}_manifest.json"
        with stage_manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest.to_dict(), f, indent=2)

        return manifest

    def save_master_manifest(
        self,
        run_id: str,
        status: str,
        *,
        start_time: str | None = None,
        end_time: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Save the master manifest with all stage manifests.

        Parameters
        ----------
        run_id : str
            Run identifier
        status : str
            Overall run status
        start_time : str, optional
            Run start time
        end_time : str, optional
            Run end time
        metadata : dict, optional
            Additional metadata

        Returns
        -------
        Path
            Path to the master manifest
        """
        # Count artifacts
        total_artifacts = sum(len(arts) for arts in self._artifacts.values())
        total_size = sum(a.size_bytes or 0 for arts in self._artifacts.values() for a in arts)

        master = {
            "run_id": run_id,
            "status": status,
            "created_at": utc_timestamp(),
            "start_time": start_time,
            "end_time": end_time,
            "total_artifacts": total_artifacts,
            "total_size_bytes": total_size,
            "stages": {
                name: manifest.to_dict() for name, manifest in self._stage_manifests.items()
            },
            "metadata": metadata or {},
        }

        master_path = self.manifests_dir / "master_manifest.json"
        with master_path.open("w", encoding="utf-8") as f:
            json.dump(master, f, indent=2)

        return master_path

    def validate_stage_artifacts(
        self,
        stage: str,
        stage_dir: Path | str | None = None,
    ) -> tuple[bool, list[str]]:
        """Validate that expected artifacts exist for a stage.

        Parameters
        ----------
        stage : str
            Stage name
        stage_dir : Path or str, optional
            Stage output directory (default: run_dir/stage)

        Returns
        -------
        tuple of (bool, list of str)
            (success, list of missing/invalid artifacts)
        """
        if stage_dir is None:
            # Map stage name to subdirectory
            stage_to_subdir = {
                "data_qc": "qc",
                "reference": "references",
                "spatial_backend": "spatial_backends",
            }
            subdir = stage_to_subdir.get(stage, stage)
            stage_dir = self.run_dir / subdir

        stage_dir = Path(stage_dir)
        expected = EXPECTED_ARTIFACTS.get(stage, [])
        issues: list[str] = []

        # Check directory exists
        if not stage_dir.exists():
            issues.append(f"Stage directory does not exist: {stage_dir}")
            return False, issues

        # Check for completion marker
        completion_marker = stage_dir / ".completed"
        if not completion_marker.exists():
            issues.append(f"Completion marker missing: {completion_marker}")

        # Check expected artifacts
        for artifact_name in expected:
            artifact_path = stage_dir / artifact_name
            if not artifact_path.exists():
                issues.append(f"Missing artifact: {artifact_name}")
            elif artifact_path.stat().st_size == 0:
                issues.append(f"Empty artifact: {artifact_name}")

        # Check manifest
        stage_manifest_path = self.manifests_dir / f"{stage}_manifest.json"
        if stage_manifest_path.exists():
            try:
                with stage_manifest_path.open("r", encoding="utf-8") as f:
                    manifest_data = json.load(f)
                    if manifest_data.get("status") != "completed":
                        issues.append(
                            f"Stage manifest shows status: {manifest_data.get('status')}"
                        )
            except Exception as e:
                issues.append(f"Failed to read stage manifest: {e}")

        return len(issues) == 0, issues

    def mark_stage_complete(self, stage: str, stage_dir: Path | str | None = None) -> None:
        """Create a completion marker for a stage.

        Parameters
        ----------
        stage : str
            Stage name
        stage_dir : Path or str, optional
            Stage output directory
        """
        if stage_dir is None:
            stage_to_subdir = {
                "data_qc": "qc",
                "reference": "references",
                "spatial_backend": "spatial_backends",
            }
            subdir = stage_to_subdir.get(stage, stage)
            stage_dir = self.run_dir / subdir

        stage_dir = Path(stage_dir)
        stage_dir.mkdir(parents=True, exist_ok=True)

        completion_marker = stage_dir / ".completed"
        completion_marker.write_text(utc_timestamp(), encoding="utf-8")

    def is_stage_complete(self, stage: str, stage_dir: Path | str | None = None) -> bool:
        """Check if a stage has a completion marker.

        Parameters
        ----------
        stage : str
            Stage name
        stage_dir : Path or str, optional
            Stage output directory

        Returns
        -------
        bool
            True if stage is marked complete
        """
        if stage_dir is None:
            stage_to_subdir = {
                "data_qc": "qc",
                "reference": "references",
                "spatial_backend": "spatial_backends",
            }
            subdir = stage_to_subdir.get(stage, stage)
            stage_dir = self.run_dir / subdir

        stage_dir = Path(stage_dir)
        completion_marker = stage_dir / ".completed"
        return completion_marker.exists()

    def get_all_artifacts(self) -> dict[str, list[ArtifactInfo]]:
        """Get all registered artifacts by stage.

        Returns
        -------
        dict
            Dictionary mapping stage names to artifact lists
        """
        return dict(self._artifacts)

    def clear_stage(self, stage: str) -> None:
        """Clear artifacts for a stage (for re-running).

        Parameters
        ----------
        stage : str
            Stage name
        """
        self._artifacts[stage] = []
        if stage in self._stage_manifests:
            del self._stage_manifests[stage]

        # Remove stage manifest file
        stage_manifest_path = self.manifests_dir / f"{stage}_manifest.json"
        if stage_manifest_path.exists():
            stage_manifest_path.unlink()
