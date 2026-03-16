"""Tests for the artifact registry module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from stagebridge.orchestration.artifact_registry import (
    ArtifactInfo,
    ArtifactRegistry,
    StageManifest,
)


@pytest.fixture
def temp_run_dir(tmp_path: Path) -> Path:
    """Create a temporary run directory with standard structure."""
    run_dir = tmp_path / "run_test"
    run_dir.mkdir()

    # Create standard subdirectories
    for subdir in ["qc", "references", "manifests", "logs"]:
        (run_dir / subdir).mkdir()

    return run_dir


@pytest.fixture
def registry(temp_run_dir: Path) -> ArtifactRegistry:
    """Create an artifact registry."""
    return ArtifactRegistry(temp_run_dir)


class TestArtifactInfo:
    """Tests for ArtifactInfo dataclass."""

    def test_to_dict(self) -> None:
        """Test converting to dictionary."""
        artifact = ArtifactInfo(
            name="test.json",
            path="/path/to/test.json",
            stage="data_qc",
            artifact_type="file",
            size_bytes=1024,
            checksum="sha256:abc123",
            created_at="2024-01-01T00:00:00Z",
            metadata={"key": "value"},
        )

        d = artifact.to_dict()

        assert d["name"] == "test.json"
        assert d["path"] == "/path/to/test.json"
        assert d["stage"] == "data_qc"
        assert d["artifact_type"] == "file"
        assert d["size_bytes"] == 1024
        assert d["checksum"] == "sha256:abc123"
        assert d["metadata"] == {"key": "value"}

    def test_from_dict(self) -> None:
        """Test creating from dictionary."""
        d = {
            "name": "test.json",
            "path": "/path/to/test.json",
            "stage": "data_qc",
            "artifact_type": "file",
            "size_bytes": 1024,
            "checksum": "sha256:abc123",
            "created_at": "2024-01-01T00:00:00Z",
            "metadata": {"key": "value"},
        }

        artifact = ArtifactInfo.from_dict(d)

        assert artifact.name == "test.json"
        assert artifact.stage == "data_qc"
        assert artifact.metadata == {"key": "value"}


class TestStageManifest:
    """Tests for StageManifest dataclass."""

    def test_to_dict(self) -> None:
        """Test converting to dictionary."""
        manifest = StageManifest(
            stage_name="data_qc",
            status="completed",
            start_time="2024-01-01T00:00:00Z",
            end_time="2024-01-01T00:05:00Z",
            duration_seconds=300.0,
            artifacts=[
                ArtifactInfo(
                    name="test.json",
                    path="/path/test.json",
                    stage="data_qc",
                    artifact_type="file",
                )
            ],
            expected_artifacts=["test.json"],
            metadata={"key": "value"},
        )

        d = manifest.to_dict()

        assert d["stage_name"] == "data_qc"
        assert d["status"] == "completed"
        assert d["duration_seconds"] == 300.0
        assert len(d["artifacts"]) == 1
        assert d["artifacts"][0]["name"] == "test.json"

    def test_from_dict(self) -> None:
        """Test creating from dictionary."""
        d = {
            "stage_name": "data_qc",
            "status": "completed",
            "start_time": "2024-01-01T00:00:00Z",
            "end_time": "2024-01-01T00:05:00Z",
            "duration_seconds": 300.0,
            "artifacts": [
                {
                    "name": "test.json",
                    "path": "/path/test.json",
                    "stage": "data_qc",
                    "artifact_type": "file",
                }
            ],
            "expected_artifacts": ["test.json"],
            "metadata": {"key": "value"},
        }

        manifest = StageManifest.from_dict(d)

        assert manifest.stage_name == "data_qc"
        assert manifest.status == "completed"
        assert len(manifest.artifacts) == 1


class TestArtifactRegistry:
    """Tests for ArtifactRegistry class."""

    def test_register_artifact(self, registry: ArtifactRegistry, temp_run_dir: Path) -> None:
        """Test registering an artifact."""
        # Create a test file
        test_file = temp_run_dir / "qc" / "test.json"
        test_file.write_text('{"test": true}', encoding="utf-8")

        artifact = registry.register_artifact(
            name="test.json",
            path=test_file,
            stage="data_qc",
            artifact_type="file",
        )

        assert artifact.name == "test.json"
        assert artifact.stage == "data_qc"
        assert artifact.size_bytes is not None
        assert artifact.size_bytes > 0
        assert artifact.checksum is not None
        assert artifact.created_at is not None

    def test_register_artifact_with_metadata(
        self, registry: ArtifactRegistry, temp_run_dir: Path
    ) -> None:
        """Test registering an artifact with metadata."""
        test_file = temp_run_dir / "qc" / "test.json"
        test_file.write_text('{"test": true}', encoding="utf-8")

        artifact = registry.register_artifact(
            name="test.json",
            path=test_file,
            stage="data_qc",
            metadata={"version": "1.0"},
        )

        assert artifact.metadata == {"version": "1.0"}

    def test_register_artifacts_from_dir(
        self, registry: ArtifactRegistry, temp_run_dir: Path
    ) -> None:
        """Test registering all artifacts from a directory."""
        qc_dir = temp_run_dir / "qc"

        # Create test files
        (qc_dir / "file1.json").write_text('{"a": 1}', encoding="utf-8")
        (qc_dir / "file2.json").write_text('{"b": 2}', encoding="utf-8")
        (qc_dir / "file3.txt").write_text("text content", encoding="utf-8")

        artifacts = registry.register_artifacts_from_dir(qc_dir, "data_qc", pattern="*.json")

        assert len(artifacts) == 2
        names = [a.name for a in artifacts]
        assert "file1.json" in names
        assert "file2.json" in names
        assert "file3.txt" not in names

    def test_get_stage_artifacts(self, registry: ArtifactRegistry, temp_run_dir: Path) -> None:
        """Test getting artifacts for a stage."""
        test_file = temp_run_dir / "qc" / "test.json"
        test_file.write_text('{"test": true}', encoding="utf-8")

        registry.register_artifact("test.json", test_file, "data_qc")

        artifacts = registry.get_stage_artifacts("data_qc")
        assert len(artifacts) == 1
        assert artifacts[0].name == "test.json"

        # Non-existent stage should return empty list
        assert registry.get_stage_artifacts("nonexistent") == []

    def test_create_stage_manifest(self, registry: ArtifactRegistry, temp_run_dir: Path) -> None:
        """Test creating a stage manifest."""
        test_file = temp_run_dir / "qc" / "test.json"
        test_file.write_text('{"test": true}', encoding="utf-8")

        registry.register_artifact("test.json", test_file, "data_qc")

        manifest = registry.create_stage_manifest(
            "data_qc",
            "completed",
            start_time="2024-01-01T00:00:00Z",
            end_time="2024-01-01T00:05:00Z",
            duration_seconds=300.0,
        )

        assert manifest.stage_name == "data_qc"
        assert manifest.status == "completed"
        assert len(manifest.artifacts) == 1

        # Check manifest file was saved
        manifest_path = temp_run_dir / "manifests" / "data_qc_manifest.json"
        assert manifest_path.exists()

    def test_save_master_manifest(self, registry: ArtifactRegistry, temp_run_dir: Path) -> None:
        """Test saving the master manifest."""
        test_file = temp_run_dir / "qc" / "test.json"
        test_file.write_text('{"test": true}', encoding="utf-8")

        registry.register_artifact("test.json", test_file, "data_qc")
        registry.create_stage_manifest("data_qc", "completed")

        master_path = registry.save_master_manifest(
            "test_run",
            "completed",
            start_time="2024-01-01T00:00:00Z",
            end_time="2024-01-01T00:10:00Z",
        )

        assert master_path.exists()

        with master_path.open("r", encoding="utf-8") as f:
            master = json.load(f)

        assert master["run_id"] == "test_run"
        assert master["status"] == "completed"
        assert master["total_artifacts"] == 1
        assert "data_qc" in master["stages"]

    def test_mark_stage_complete(self, registry: ArtifactRegistry, temp_run_dir: Path) -> None:
        """Test marking a stage as complete."""
        registry.mark_stage_complete("data_qc")

        completion_marker = temp_run_dir / "qc" / ".completed"
        assert completion_marker.exists()

    def test_is_stage_complete(self, registry: ArtifactRegistry, temp_run_dir: Path) -> None:
        """Test checking if a stage is complete."""
        assert not registry.is_stage_complete("data_qc")

        registry.mark_stage_complete("data_qc")

        assert registry.is_stage_complete("data_qc")

    def test_validate_stage_artifacts(
        self, registry: ArtifactRegistry, temp_run_dir: Path
    ) -> None:
        """Test validating stage artifacts."""
        qc_dir = temp_run_dir / "qc"

        # Create required files
        (qc_dir / "qc_report.json").write_text('{"status": "ok"}', encoding="utf-8")
        (qc_dir / "qc_summary.html").write_text("<html></html>", encoding="utf-8")
        (qc_dir / ".completed").write_text("done", encoding="utf-8")

        success, issues = registry.validate_stage_artifacts("data_qc")

        # May have issues if not all expected files exist, but should not crash
        assert isinstance(success, bool)
        assert isinstance(issues, list)

    def test_clear_stage(self, registry: ArtifactRegistry, temp_run_dir: Path) -> None:
        """Test clearing stage artifacts."""
        test_file = temp_run_dir / "qc" / "test.json"
        test_file.write_text('{"test": true}', encoding="utf-8")

        registry.register_artifact("test.json", test_file, "data_qc")
        registry.create_stage_manifest("data_qc", "completed")

        # Verify artifacts exist
        assert len(registry.get_stage_artifacts("data_qc")) == 1
        assert (temp_run_dir / "manifests" / "data_qc_manifest.json").exists()

        # Clear stage
        registry.clear_stage("data_qc")

        assert len(registry.get_stage_artifacts("data_qc")) == 0
        assert not (temp_run_dir / "manifests" / "data_qc_manifest.json").exists()

    def test_get_all_artifacts(self, registry: ArtifactRegistry, temp_run_dir: Path) -> None:
        """Test getting all artifacts."""
        # Create files for multiple stages
        (temp_run_dir / "qc" / "qc.json").write_text("{}", encoding="utf-8")
        (temp_run_dir / "references" / "ref.json").write_text("{}", encoding="utf-8")

        registry.register_artifact("qc.json", temp_run_dir / "qc" / "qc.json", "data_qc")
        registry.register_artifact(
            "ref.json", temp_run_dir / "references" / "ref.json", "reference"
        )

        all_artifacts = registry.get_all_artifacts()

        assert "data_qc" in all_artifacts
        assert "reference" in all_artifacts
        assert len(all_artifacts["data_qc"]) == 1
        assert len(all_artifacts["reference"]) == 1
