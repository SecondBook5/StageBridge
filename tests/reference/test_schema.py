"""Tests for standardized output schema compliance."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stagebridge.reference.schema import (
    ReferenceEmbeddingSchema,
    ReferenceManifest,
    SCHEMA,
    export_reference_outputs,
    load_reference_outputs,
    validate_output_integrity,
    create_manifest,
)


def _create_mock_hlca_df(n_cells: int = 20, latent_dim: int = 8) -> pd.DataFrame:
    """Create mock HLCA embedding DataFrame."""
    df = pd.DataFrame(
        {
            "cell_id": [f"cell_{i}" for i in range(n_cells)],
            "donor_id": [f"D{i % 3}" for i in range(n_cells)],
            "sample_id": [f"S{i % 5}" for i in range(n_cells)],
            "stage_id": [["AAH", "AIS", "MIA", "LUAD"][i % 4] for i in range(n_cells)],
        }
    )
    for i in range(latent_dim):
        df[f"hlca_latent_{i}"] = np.random.randn(n_cells).astype(np.float32)
    return df


def _create_mock_luca_df(n_cells: int = 20, latent_dim: int = 8) -> pd.DataFrame:
    """Create mock LuCa embedding DataFrame."""
    df = pd.DataFrame(
        {
            "cell_id": [f"cell_{i}" for i in range(n_cells)],
            "donor_id": [f"D{i % 3}" for i in range(n_cells)],
            "sample_id": [f"S{i % 5}" for i in range(n_cells)],
            "stage_id": [["AAH", "AIS", "MIA", "LUAD"][i % 4] for i in range(n_cells)],
        }
    )
    for i in range(latent_dim):
        df[f"luca_latent_{i}"] = np.random.randn(n_cells).astype(np.float32)
    return df


def _create_mock_fused_df(n_cells: int = 20, fused_dim: int = 16) -> pd.DataFrame:
    """Create mock fused embedding DataFrame."""
    df = pd.DataFrame(
        {
            "cell_id": [f"cell_{i}" for i in range(n_cells)],
            "donor_id": [f"D{i % 3}" for i in range(n_cells)],
            "sample_id": [f"S{i % 5}" for i in range(n_cells)],
            "stage_id": [["AAH", "AIS", "MIA", "LUAD"][i % 4] for i in range(n_cells)],
            "reference_mode_used": ["both"] * n_cells,
        }
    )
    for i in range(fused_dim):
        df[f"fused_latent_{i}"] = np.random.randn(n_cells).astype(np.float32)
    return df


def _create_mock_confidence_df(n_cells: int = 20) -> pd.DataFrame:
    """Create mock confidence DataFrame."""
    return pd.DataFrame(
        {
            "cell_id": [f"cell_{i}" for i in range(n_cells)],
            "hlca_confidence": np.random.uniform(0.5, 1.0, n_cells).astype(np.float32),
            "luca_confidence": np.random.uniform(0.5, 1.0, n_cells).astype(np.float32),
        }
    )


class TestReferenceEmbeddingSchema:
    """Tests for ReferenceEmbeddingSchema."""

    def test_schema_constants(self) -> None:
        """Schema has expected constants."""
        assert SCHEMA.HLCA_LATENT_PREFIX == "hlca_latent_"
        assert SCHEMA.LUCA_LATENT_PREFIX == "luca_latent_"
        assert SCHEMA.FUSED_LATENT_PREFIX == "fused_latent_"
        assert "cell_id" in SCHEMA.METADATA_COLS
        assert "donor_id" in SCHEMA.METADATA_COLS
        assert "hlca_confidence" in SCHEMA.CONFIDENCE_COLS


class TestReferenceManifest:
    """Tests for ReferenceManifest."""

    def test_to_dict(self) -> None:
        """Manifest converts to serializable dict."""
        manifest = create_manifest(
            run_id="test_run_001",
            hlca_dim=16,
            luca_dim=12,
            fused_dim=28,
            n_cells=1000,
            fusion_method="concat",
            mapping_method="knn_projection",
            hlca_path="/path/to/hlca.h5ad",
            luca_path="/path/to/luca.h5ad",
            query_path="/path/to/query.h5ad",
        )

        d = manifest.to_dict()

        assert d["run_id"] == "test_run_001"
        assert d["hlca_latent_dim"] == 16
        assert d["fusion_method"] == "concat"

        # Should be JSON serializable
        json.dumps(d)

    def test_from_dict(self) -> None:
        """Manifest can be recreated from dict."""
        original = create_manifest(
            run_id="test",
            hlca_dim=8,
            luca_dim=8,
            fused_dim=16,
            n_cells=100,
            fusion_method="average",
            mapping_method="pca_projection",
            hlca_path="/path/hlca",
            luca_path=None,
            query_path="/path/query",
        )

        d = original.to_dict()
        restored = ReferenceManifest.from_dict(d)

        assert restored.run_id == original.run_id
        assert restored.hlca_latent_dim == original.hlca_latent_dim
        assert restored.luca_reference_path == original.luca_reference_path


class TestExportAndLoadOutputs:
    """Tests for export_reference_outputs and load_reference_outputs."""

    def test_export_creates_files(self, tmp_path: Path) -> None:
        """Export creates all expected files."""
        hlca_df = _create_mock_hlca_df(n_cells=20, latent_dim=8)
        luca_df = _create_mock_luca_df(n_cells=20, latent_dim=8)
        fused_df = _create_mock_fused_df(n_cells=20, fused_dim=16)
        conf_df = _create_mock_confidence_df(n_cells=20)

        manifest = create_manifest(
            run_id="export_test",
            hlca_dim=8,
            luca_dim=8,
            fused_dim=16,
            n_cells=20,
            fusion_method="concat",
            mapping_method="knn_projection",
            hlca_path="/path/hlca",
            luca_path="/path/luca",
            query_path="/path/query",
        )

        paths = export_reference_outputs(
            hlca_df=hlca_df,
            luca_df=luca_df,
            fused_df=fused_df,
            confidence_df=conf_df,
            manifest=manifest,
            feature_overlap={"hlca": {"overlap_fraction": 0.8}},
            output_dir=tmp_path,
        )

        assert (tmp_path / "hlca_embedding.parquet").exists()
        assert (tmp_path / "luca_embedding.parquet").exists()
        assert (tmp_path / "fused_embedding.parquet").exists()
        assert (tmp_path / "reference_confidence.parquet").exists()
        assert (tmp_path / "reference_manifest.json").exists()
        assert (tmp_path / "feature_overlap_report.json").exists()
        assert (tmp_path / "plots").is_dir()

    def test_round_trip(self, tmp_path: Path) -> None:
        """Data survives export/load round trip."""
        hlca_df = _create_mock_hlca_df(n_cells=15, latent_dim=4)
        luca_df = _create_mock_luca_df(n_cells=15, latent_dim=4)
        fused_df = _create_mock_fused_df(n_cells=15, fused_dim=8)
        conf_df = _create_mock_confidence_df(n_cells=15)

        manifest = create_manifest(
            run_id="roundtrip_test",
            hlca_dim=4,
            luca_dim=4,
            fused_dim=8,
            n_cells=15,
            fusion_method="concat",
            mapping_method="knn_projection",
            hlca_path="/path/hlca",
            luca_path="/path/luca",
            query_path="/path/query",
        )

        export_reference_outputs(
            hlca_df=hlca_df,
            luca_df=luca_df,
            fused_df=fused_df,
            confidence_df=conf_df,
            manifest=manifest,
            feature_overlap={},
            output_dir=tmp_path,
        )

        loaded = load_reference_outputs(tmp_path)

        # Check DataFrames
        assert loaded["hlca_df"].shape == hlca_df.shape
        assert loaded["luca_df"].shape == luca_df.shape
        assert loaded["fused_df"].shape == fused_df.shape
        assert loaded["confidence_df"].shape == conf_df.shape

        # Check manifest
        assert loaded["manifest"].run_id == "roundtrip_test"
        assert loaded["manifest"].hlca_latent_dim == 4

    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        """Loading from directory with missing files raises."""
        # Create empty directory
        (tmp_path / "partial").mkdir()

        with pytest.raises(FileNotFoundError):
            load_reference_outputs(tmp_path / "partial")


class TestValidateOutputIntegrity:
    """Tests for validate_output_integrity function."""

    def test_valid_outputs_pass(self, tmp_path: Path) -> None:
        """Valid outputs pass integrity check."""
        hlca_df = _create_mock_hlca_df(n_cells=10, latent_dim=4)
        luca_df = _create_mock_luca_df(n_cells=10, latent_dim=4)
        fused_df = _create_mock_fused_df(n_cells=10, fused_dim=8)
        conf_df = _create_mock_confidence_df(n_cells=10)

        manifest = create_manifest(
            run_id="valid_test",
            hlca_dim=4,
            luca_dim=4,
            fused_dim=8,
            n_cells=10,
            fusion_method="concat",
            mapping_method="knn_projection",
            hlca_path="/path/hlca",
            luca_path="/path/luca",
            query_path="/path/query",
        )

        export_reference_outputs(
            hlca_df=hlca_df,
            luca_df=luca_df,
            fused_df=fused_df,
            confidence_df=conf_df,
            manifest=manifest,
            feature_overlap={},
            output_dir=tmp_path,
        )

        report = validate_output_integrity(tmp_path)

        assert report["valid"]
        assert len(report["errors"]) == 0

    def test_missing_file_fails(self, tmp_path: Path) -> None:
        """Missing file causes validation failure."""
        # Create partial outputs
        hlca_df = _create_mock_hlca_df(n_cells=10, latent_dim=4)
        hlca_df.to_parquet(tmp_path / "hlca_embedding.parquet")

        report = validate_output_integrity(tmp_path)

        assert not report["valid"]
        assert len(report["errors"]) > 0

    def test_cell_id_mismatch_fails(self, tmp_path: Path) -> None:
        """Cell ID mismatch causes validation failure."""
        # Create outputs with mismatched cell IDs
        hlca_df = _create_mock_hlca_df(n_cells=10, latent_dim=4)
        luca_df = _create_mock_luca_df(n_cells=10, latent_dim=4)
        luca_df["cell_id"] = [f"different_{i}" for i in range(10)]  # Different IDs!

        fused_df = _create_mock_fused_df(n_cells=10, fused_dim=8)
        conf_df = _create_mock_confidence_df(n_cells=10)

        manifest = create_manifest(
            run_id="mismatch_test",
            hlca_dim=4,
            luca_dim=4,
            fused_dim=8,
            n_cells=10,
            fusion_method="concat",
            mapping_method="knn_projection",
            hlca_path="/path/hlca",
            luca_path="/path/luca",
            query_path="/path/query",
        )

        export_reference_outputs(
            hlca_df=hlca_df,
            luca_df=luca_df,
            fused_df=fused_df,
            confidence_df=conf_df,
            manifest=manifest,
            feature_overlap={},
            output_dir=tmp_path,
        )

        report = validate_output_integrity(tmp_path)

        assert not report["valid"]
        assert any("mismatch" in e.lower() for e in report["errors"])
