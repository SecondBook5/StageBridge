"""Tests for reference geometry validation.

These tests verify that the validation module correctly identifies:
1. Missing files
2. Schema violations
3. NaN in embeddings
4. Confidence calibration issues
5. Cell count mismatches
6. Latent dimension mismatches
7. Missing donors
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stagebridge.validation.reference_geometry import (
    ReferenceGeometryValidator,
    ReferenceGeometryValidationReport,
    validate_reference_geometry,
    validate_reference_geometry_quick,
    EXPECTED_FILES,
    EXPECTED_HLCA_DIM,
    EXPECTED_LUCA_DIM,
    EXPECTED_FUSED_DIM,
    METADATA_COLS,
    CONFIDENCE_REQUIRED_COLS,
)


# --- Fixtures for creating mock data ---


def _create_hlca_df(
    n_cells: int = 100,
    hlca_dim: int = EXPECTED_HLCA_DIM,
    add_nan: bool = False,
    nan_fraction: float = 0.01,
) -> pd.DataFrame:
    """Create mock HLCA embedding DataFrame."""
    np.random.seed(42)
    df = pd.DataFrame({
        "cell_id": [f"cell_{i}" for i in range(n_cells)],
        "donor_id": [f"D{i % 5}" for i in range(n_cells)],
        "sample_id": [f"S{i % 10}" for i in range(n_cells)],
        "stage_id": [["AAH", "AIS", "MIA", "LUAD"][i % 4] for i in range(n_cells)],
    })
    for i in range(hlca_dim):
        df[f"hlca_latent_{i}"] = np.random.randn(n_cells).astype(np.float32)

    if add_nan:
        nan_count = max(1, int(n_cells * nan_fraction))
        nan_idx = np.random.choice(n_cells, nan_count, replace=False)
        df.loc[nan_idx, "hlca_latent_0"] = np.nan

    return df


def _create_luca_df(
    n_cells: int = 100,
    luca_dim: int = EXPECTED_LUCA_DIM,
    cell_id_base: str = "cell",
    add_nan: bool = False,
) -> pd.DataFrame:
    """Create mock LuCA embedding DataFrame."""
    np.random.seed(43)
    df = pd.DataFrame({
        "cell_id": [f"{cell_id_base}_{i}" for i in range(n_cells)],
        "donor_id": [f"D{i % 5}" for i in range(n_cells)],
        "sample_id": [f"S{i % 10}" for i in range(n_cells)],
        "stage_id": [["AAH", "AIS", "MIA", "LUAD"][i % 4] for i in range(n_cells)],
    })
    for i in range(luca_dim):
        df[f"luca_latent_{i}"] = np.random.randn(n_cells).astype(np.float32)

    if add_nan:
        df.loc[0, "luca_latent_0"] = np.nan

    return df


def _create_fused_df(
    n_cells: int = 100,
    fused_dim: int = EXPECTED_FUSED_DIM,
    cell_id_base: str = "cell",
) -> pd.DataFrame:
    """Create mock fused embedding DataFrame."""
    np.random.seed(44)
    df = pd.DataFrame({
        "cell_id": [f"{cell_id_base}_{i}" for i in range(n_cells)],
        "donor_id": [f"D{i % 5}" for i in range(n_cells)],
        "sample_id": [f"S{i % 10}" for i in range(n_cells)],
        "stage_id": [["AAH", "AIS", "MIA", "LUAD"][i % 4] for i in range(n_cells)],
        "reference_mode_used": ["both"] * n_cells,
    })
    for i in range(fused_dim):
        df[f"fused_latent_{i}"] = np.random.randn(n_cells).astype(np.float32)

    return df


def _create_confidence_df(
    n_cells: int = 100,
    cell_id_base: str = "cell",
    hlca_mean: float = 0.5,
    luca_mean: float = 0.5,
    add_nan: bool = False,
) -> pd.DataFrame:
    """Create mock confidence DataFrame with calibrated values.

    For percentile rank calibration, mean should be ~0.5.
    """
    np.random.seed(45)

    # Generate values centered around the specified mean
    # Using beta distribution for bounded [0,1] values
    hlca_conf = np.clip(np.random.normal(hlca_mean, 0.15, n_cells), 0, 1).astype(np.float32)
    luca_conf = np.clip(np.random.normal(luca_mean, 0.15, n_cells), 0, 1).astype(np.float32)

    df = pd.DataFrame({
        "cell_id": [f"{cell_id_base}_{i}" for i in range(n_cells)],
        "hlca_confidence": hlca_conf,
        "luca_confidence": luca_conf,
        "hlca_raw_distance": np.random.uniform(0.1, 2.0, n_cells).astype(np.float32),
        "luca_raw_distance": np.random.uniform(0.1, 2.0, n_cells).astype(np.float32),
        "hlca_confidence_method": ["percentile_rank"] * n_cells,
        "luca_confidence_method": ["percentile_rank"] * n_cells,
        "reference_mode_used": ["both"] * n_cells,
    })

    if add_nan:
        df.loc[0, "hlca_confidence"] = np.nan

    return df


def _create_manifest(
    n_cells: int = 100,
    hlca_dim: int = EXPECTED_HLCA_DIM,
    luca_dim: int = EXPECTED_LUCA_DIM,
    fused_dim: int = EXPECTED_FUSED_DIM,
) -> dict:
    """Create mock manifest."""
    return {
        "run_id": "test_run_001",
        "mapping_method": "knn_projection",
        "fusion_method": "concat",
        "n_cells": n_cells,
        "hlca_latent_dim": hlca_dim,
        "luca_latent_dim": luca_dim,
        "fused_dim": fused_dim,
        "hlca_reference_path": "/path/to/hlca.h5ad",
        "luca_reference_path": "/path/to/luca.h5ad",
        "query_data_path": "/path/to/query.h5ad",
    }


def _create_feature_overlap() -> dict:
    """Create mock feature overlap report."""
    return {
        "hlca": {
            "overlap_genes": 18000,
            "query_genes": 20000,
            "reference_genes": 22000,
            "overlap_fraction": 0.82,
        },
        "luca": {
            "overlap_genes": 17500,
            "query_genes": 20000,
            "reference_genes": 19000,
            "overlap_fraction": 0.87,
        },
    }


@pytest.fixture
def valid_output_dir(tmp_path: Path) -> Path:
    """Create a valid reference_geometry output directory."""
    output_dir = tmp_path / "reference_geometry"
    output_dir.mkdir()

    n_cells = 100

    # Create all required files
    _create_hlca_df(n_cells).to_parquet(output_dir / "hlca_embedding.parquet")
    _create_luca_df(n_cells).to_parquet(output_dir / "luca_embedding.parquet")
    _create_fused_df(n_cells).to_parquet(output_dir / "fused_embedding.parquet")
    _create_confidence_df(n_cells).to_parquet(output_dir / "reference_confidence.parquet")

    with open(output_dir / "reference_manifest.json", "w") as f:
        json.dump(_create_manifest(n_cells), f)

    with open(output_dir / "feature_overlap_report.json", "w") as f:
        json.dump(_create_feature_overlap(), f)

    return output_dir


@pytest.fixture
def output_dir_missing_files(tmp_path: Path) -> Path:
    """Create output directory with missing files."""
    output_dir = tmp_path / "reference_geometry"
    output_dir.mkdir()

    # Only create some files
    _create_hlca_df(50).to_parquet(output_dir / "hlca_embedding.parquet")
    # Missing: luca, fused, confidence, manifest, feature_overlap

    return output_dir


@pytest.fixture
def output_dir_with_nan(tmp_path: Path) -> Path:
    """Create output directory with NaN in embeddings."""
    output_dir = tmp_path / "reference_geometry"
    output_dir.mkdir()

    n_cells = 100

    # Create files with NaN
    _create_hlca_df(n_cells, add_nan=True).to_parquet(output_dir / "hlca_embedding.parquet")
    _create_luca_df(n_cells, add_nan=True).to_parquet(output_dir / "luca_embedding.parquet")
    _create_fused_df(n_cells).to_parquet(output_dir / "fused_embedding.parquet")
    _create_confidence_df(n_cells, add_nan=True).to_parquet(output_dir / "reference_confidence.parquet")

    with open(output_dir / "reference_manifest.json", "w") as f:
        json.dump(_create_manifest(n_cells), f)

    with open(output_dir / "feature_overlap_report.json", "w") as f:
        json.dump(_create_feature_overlap(), f)

    return output_dir


# --- Test Classes ---


class TestValidationReportBasics:
    """Tests for ReferenceGeometryValidationReport dataclass."""

    def test_to_dict(self) -> None:
        """Report converts to serializable dict."""
        report = ReferenceGeometryValidationReport(
            valid=True,
            validation_status="VALID",
            cell_count=1000,
            donor_count=5,
        )

        d = report.to_dict()

        assert d["valid"] is True
        assert d["cell_count"] == 1000
        assert d["donor_count"] == 5

        # Should be JSON serializable
        json.dumps(d)

    def test_to_markdown(self) -> None:
        """Report generates markdown summary."""
        report = ReferenceGeometryValidationReport(
            valid=True,
            validation_status="VALID",
            cell_count=1000,
            hlca_dim=30,
            luca_dim=10,
            fused_dim=40,
            files_found=["hlca_embedding.parquet", "luca_embedding.parquet"],
        )

        md = report.to_markdown()

        assert "# Reference Geometry Validation Report" in md
        assert "VALID" in md
        assert "1,000" in md
        assert "hlca_embedding.parquet" in md


class TestFileExistence:
    """Tests for file existence checks."""

    def test_all_files_present(self, valid_output_dir: Path) -> None:
        """Validator finds all expected files."""
        validator = ReferenceGeometryValidator(valid_output_dir)
        report = validator.validate()

        assert len(report.files_found) >= 6
        assert len(report.files_missing) == 0

    def test_missing_files_detected(self, output_dir_missing_files: Path) -> None:
        """Validator detects missing files."""
        validator = ReferenceGeometryValidator(output_dir_missing_files)
        report = validator.validate()

        assert not report.valid
        assert "INVALID" in report.validation_status
        assert len(report.files_missing) > 0
        assert "luca_embedding.parquet" in report.files_missing

    def test_missing_critical_file_blocks(self, tmp_path: Path) -> None:
        """Missing critical files block validation."""
        output_dir = tmp_path / "reference_geometry"
        output_dir.mkdir()

        # Only create non-critical files
        with open(output_dir / "reference_manifest.json", "w") as f:
            json.dump(_create_manifest(100), f)

        validator = ReferenceGeometryValidator(output_dir)
        report = validator.validate()

        assert not report.valid
        assert "Critical files missing" in report.errors[0]


class TestSchemaValidation:
    """Tests for schema validation."""

    def test_valid_schema(self, valid_output_dir: Path) -> None:
        """Validator accepts correct schemas."""
        validator = ReferenceGeometryValidator(valid_output_dir)
        report = validator.validate()

        assert len(report.schema_errors) == 0

    def test_missing_metadata_column(self, tmp_path: Path) -> None:
        """Validator detects missing metadata columns."""
        output_dir = tmp_path / "reference_geometry"
        output_dir.mkdir()

        # Create HLCA with missing donor_id
        df = _create_hlca_df(50)
        del df["donor_id"]
        df.to_parquet(output_dir / "hlca_embedding.parquet")

        # Create other required files
        _create_luca_df(50).to_parquet(output_dir / "luca_embedding.parquet")
        _create_fused_df(50).to_parquet(output_dir / "fused_embedding.parquet")
        _create_confidence_df(50).to_parquet(output_dir / "reference_confidence.parquet")
        with open(output_dir / "reference_manifest.json", "w") as f:
            json.dump(_create_manifest(50), f)
        with open(output_dir / "feature_overlap_report.json", "w") as f:
            json.dump(_create_feature_overlap(), f)

        validator = ReferenceGeometryValidator(output_dir)
        report = validator.validate()

        assert any("donor_id" in e for e in report.schema_errors)

    def test_missing_latent_columns(self, tmp_path: Path) -> None:
        """Validator detects missing latent columns."""
        output_dir = tmp_path / "reference_geometry"
        output_dir.mkdir()

        # Create HLCA with no latent columns
        df = pd.DataFrame({
            "cell_id": [f"cell_{i}" for i in range(50)],
            "donor_id": [f"D{i % 5}" for i in range(50)],
            "sample_id": [f"S{i}" for i in range(50)],
            "stage_id": ["AAH"] * 50,
        })
        df.to_parquet(output_dir / "hlca_embedding.parquet")

        _create_luca_df(50).to_parquet(output_dir / "luca_embedding.parquet")
        _create_fused_df(50).to_parquet(output_dir / "fused_embedding.parquet")
        _create_confidence_df(50).to_parquet(output_dir / "reference_confidence.parquet")
        with open(output_dir / "reference_manifest.json", "w") as f:
            json.dump(_create_manifest(50), f)
        with open(output_dir / "feature_overlap_report.json", "w") as f:
            json.dump(_create_feature_overlap(), f)

        validator = ReferenceGeometryValidator(output_dir)
        report = validator.validate()

        assert any("hlca_latent" in e for e in report.schema_errors)
        assert report.hlca_dim == 0


class TestNaNDetection:
    """Tests for NaN detection in embeddings."""

    def test_no_nan_in_valid_outputs(self, valid_output_dir: Path) -> None:
        """Validator passes when no NaN present."""
        validator = ReferenceGeometryValidator(valid_output_dir)
        report = validator.validate()

        assert not report.has_nan
        assert len(report.nan_report) == 0

    def test_nan_detected_in_embeddings(self, output_dir_with_nan: Path) -> None:
        """Validator detects NaN in embeddings."""
        validator = ReferenceGeometryValidator(output_dir_with_nan)
        report = validator.validate()

        assert report.has_nan
        assert "hlca_embedding" in report.nan_report
        assert any("NaN" in e for e in report.errors)

    def test_nan_detected_in_confidence(self, output_dir_with_nan: Path) -> None:
        """Validator detects NaN in confidence scores."""
        validator = ReferenceGeometryValidator(output_dir_with_nan)
        report = validator.validate()

        # NaN was added to confidence in fixture
        assert report.has_nan


class TestConfidenceCalibration:
    """Tests for confidence calibration checks."""

    def test_calibrated_confidence_passes(self, valid_output_dir: Path) -> None:
        """Validator passes for properly calibrated confidence (mean ~0.5)."""
        validator = ReferenceGeometryValidator(valid_output_dir)
        report = validator.validate()

        assert report.confidence_calibration_ok
        assert "hlca" in report.confidence_report
        assert "luca" in report.confidence_report

    def test_miscalibrated_confidence_warns(self, tmp_path: Path) -> None:
        """Validator warns when confidence is not centered at 0.5."""
        output_dir = tmp_path / "reference_geometry"
        output_dir.mkdir()

        n_cells = 100
        _create_hlca_df(n_cells).to_parquet(output_dir / "hlca_embedding.parquet")
        _create_luca_df(n_cells).to_parquet(output_dir / "luca_embedding.parquet")
        _create_fused_df(n_cells).to_parquet(output_dir / "fused_embedding.parquet")

        # Create confidence with biased mean (not ~0.5)
        _create_confidence_df(n_cells, hlca_mean=0.9, luca_mean=0.85).to_parquet(
            output_dir / "reference_confidence.parquet"
        )

        with open(output_dir / "reference_manifest.json", "w") as f:
            json.dump(_create_manifest(n_cells), f)
        with open(output_dir / "feature_overlap_report.json", "w") as f:
            json.dump(_create_feature_overlap(), f)

        validator = ReferenceGeometryValidator(output_dir, confidence_mean_tolerance=0.15)
        report = validator.validate()

        assert not report.confidence_calibration_ok
        assert any("confidence mean" in w for w in report.warnings)

    def test_confidence_out_of_range_errors(self, tmp_path: Path) -> None:
        """Validator errors when confidence is outside [0,1]."""
        output_dir = tmp_path / "reference_geometry"
        output_dir.mkdir()

        n_cells = 50
        _create_hlca_df(n_cells).to_parquet(output_dir / "hlca_embedding.parquet")
        _create_luca_df(n_cells).to_parquet(output_dir / "luca_embedding.parquet")
        _create_fused_df(n_cells).to_parquet(output_dir / "fused_embedding.parquet")

        # Create confidence with invalid values
        conf_df = _create_confidence_df(n_cells)
        conf_df.loc[0, "hlca_confidence"] = 1.5  # Invalid!
        conf_df.to_parquet(output_dir / "reference_confidence.parquet")

        with open(output_dir / "reference_manifest.json", "w") as f:
            json.dump(_create_manifest(n_cells), f)
        with open(output_dir / "feature_overlap_report.json", "w") as f:
            json.dump(_create_feature_overlap(), f)

        validator = ReferenceGeometryValidator(output_dir)
        report = validator.validate()

        assert any("out of [0,1]" in e for e in report.errors)


class TestCellCountValidation:
    """Tests for cell count validation."""

    def test_cell_count_consistency(self, valid_output_dir: Path) -> None:
        """Validator checks cell counts are consistent across files."""
        validator = ReferenceGeometryValidator(valid_output_dir)
        report = validator.validate()

        assert report.cell_count_match
        assert report.cell_count == 100

    def test_cell_count_mismatch_detected(self, tmp_path: Path) -> None:
        """Validator detects cell count mismatches between files."""
        output_dir = tmp_path / "reference_geometry"
        output_dir.mkdir()

        # Create files with different cell counts
        _create_hlca_df(100).to_parquet(output_dir / "hlca_embedding.parquet")
        _create_luca_df(80).to_parquet(output_dir / "luca_embedding.parquet")  # Different!
        _create_fused_df(100).to_parquet(output_dir / "fused_embedding.parquet")
        _create_confidence_df(100).to_parquet(output_dir / "reference_confidence.parquet")

        with open(output_dir / "reference_manifest.json", "w") as f:
            json.dump(_create_manifest(100), f)
        with open(output_dir / "feature_overlap_report.json", "w") as f:
            json.dump(_create_feature_overlap(), f)

        validator = ReferenceGeometryValidator(output_dir)
        report = validator.validate()

        assert not report.cell_count_match
        assert any("Cell count mismatch" in e for e in report.errors)

    def test_expected_cell_count_validation(self, valid_output_dir: Path) -> None:
        """Validator checks against expected cell count."""
        validator = ReferenceGeometryValidator(
            valid_output_dir,
            expected_cell_count=200,  # Different from actual 100
        )
        report = validator.validate()

        assert not report.cell_count_match
        assert report.expected_cell_count == 200


class TestLatentDimensionValidation:
    """Tests for latent dimension validation."""

    def test_correct_dimensions_pass(self, valid_output_dir: Path) -> None:
        """Validator passes for correct latent dimensions."""
        validator = ReferenceGeometryValidator(valid_output_dir)
        report = validator.validate()

        assert report.latent_dims_ok
        assert report.hlca_dim == EXPECTED_HLCA_DIM
        assert report.luca_dim == EXPECTED_LUCA_DIM
        assert report.fused_dim == EXPECTED_FUSED_DIM

    def test_wrong_dimensions_warn(self, tmp_path: Path) -> None:
        """Validator warns for incorrect latent dimensions."""
        output_dir = tmp_path / "reference_geometry"
        output_dir.mkdir()

        n_cells = 50
        # Create with wrong dimensions
        _create_hlca_df(n_cells, hlca_dim=16).to_parquet(output_dir / "hlca_embedding.parquet")
        _create_luca_df(n_cells, luca_dim=8).to_parquet(output_dir / "luca_embedding.parquet")
        _create_fused_df(n_cells, fused_dim=24).to_parquet(output_dir / "fused_embedding.parquet")
        _create_confidence_df(n_cells).to_parquet(output_dir / "reference_confidence.parquet")

        with open(output_dir / "reference_manifest.json", "w") as f:
            json.dump(_create_manifest(n_cells, hlca_dim=16, luca_dim=8, fused_dim=24), f)
        with open(output_dir / "feature_overlap_report.json", "w") as f:
            json.dump(_create_feature_overlap(), f)

        validator = ReferenceGeometryValidator(output_dir)
        report = validator.validate()

        assert not report.latent_dims_ok
        assert report.hlca_dim == 16
        assert any("HLCA dim" in w for w in report.warnings)

    def test_custom_expected_dimensions(self, tmp_path: Path) -> None:
        """Validator accepts custom expected dimensions."""
        output_dir = tmp_path / "reference_geometry"
        output_dir.mkdir()

        n_cells = 50
        _create_hlca_df(n_cells, hlca_dim=16).to_parquet(output_dir / "hlca_embedding.parquet")
        _create_luca_df(n_cells, luca_dim=8).to_parquet(output_dir / "luca_embedding.parquet")
        _create_fused_df(n_cells, fused_dim=24).to_parquet(output_dir / "fused_embedding.parquet")
        _create_confidence_df(n_cells).to_parquet(output_dir / "reference_confidence.parquet")

        with open(output_dir / "reference_manifest.json", "w") as f:
            json.dump(_create_manifest(n_cells, hlca_dim=16, luca_dim=8, fused_dim=24), f)
        with open(output_dir / "feature_overlap_report.json", "w") as f:
            json.dump(_create_feature_overlap(), f)

        validator = ReferenceGeometryValidator(
            output_dir,
            expected_hlca_dim=16,
            expected_luca_dim=8,
            expected_fused_dim=24,
        )
        report = validator.validate()

        assert report.latent_dims_ok


class TestDonorPreservation:
    """Tests for donor preservation validation."""

    def test_all_donors_present(self, valid_output_dir: Path) -> None:
        """Validator passes when all expected donors are present."""
        expected_donors = [f"D{i}" for i in range(5)]

        validator = ReferenceGeometryValidator(
            valid_output_dir,
            expected_donors=expected_donors,
        )
        report = validator.validate()

        assert report.donors_preserved
        assert len(report.missing_donors) == 0

    def test_missing_donors_detected(self, valid_output_dir: Path) -> None:
        """Validator detects missing donors."""
        expected_donors = [f"D{i}" for i in range(5)] + ["D_MISSING"]

        validator = ReferenceGeometryValidator(
            valid_output_dir,
            expected_donors=expected_donors,
        )
        report = validator.validate()

        assert not report.donors_preserved
        assert "D_MISSING" in report.missing_donors


class TestValidationStatus:
    """Tests for final validation status determination."""

    def test_valid_status(self, valid_output_dir: Path) -> None:
        """Validator returns VALID for clean outputs."""
        validator = ReferenceGeometryValidator(valid_output_dir)
        report = validator.validate()

        assert report.valid
        assert report.validation_status == "VALID"

    def test_invalid_status_on_error(self, output_dir_missing_files: Path) -> None:
        """Validator returns INVALID when errors exist."""
        validator = ReferenceGeometryValidator(output_dir_missing_files)
        report = validator.validate()

        assert not report.valid
        assert report.validation_status == "INVALID"

    def test_valid_with_warnings_status(self, tmp_path: Path) -> None:
        """Validator returns VALID_WITH_WARNINGS when only warnings exist."""
        output_dir = tmp_path / "reference_geometry"
        output_dir.mkdir()

        n_cells = 50
        _create_hlca_df(n_cells, hlca_dim=16).to_parquet(output_dir / "hlca_embedding.parquet")
        _create_luca_df(n_cells, luca_dim=8).to_parquet(output_dir / "luca_embedding.parquet")
        _create_fused_df(n_cells, fused_dim=24).to_parquet(output_dir / "fused_embedding.parquet")
        _create_confidence_df(n_cells).to_parquet(output_dir / "reference_confidence.parquet")

        with open(output_dir / "reference_manifest.json", "w") as f:
            json.dump(_create_manifest(n_cells), f)
        with open(output_dir / "feature_overlap_report.json", "w") as f:
            json.dump(_create_feature_overlap(), f)

        # This will have warnings about dimension mismatch but no errors
        validator = ReferenceGeometryValidator(output_dir)
        report = validator.validate()

        assert report.valid
        assert report.validation_status == "VALID_WITH_WARNINGS"
        assert len(report.warnings) > 0

    def test_strict_mode_converts_warnings_to_errors(self, tmp_path: Path) -> None:
        """Strict mode treats warnings as errors."""
        output_dir = tmp_path / "reference_geometry"
        output_dir.mkdir()

        n_cells = 50
        _create_hlca_df(n_cells, hlca_dim=16).to_parquet(output_dir / "hlca_embedding.parquet")
        _create_luca_df(n_cells, luca_dim=8).to_parquet(output_dir / "luca_embedding.parquet")
        _create_fused_df(n_cells, fused_dim=24).to_parquet(output_dir / "fused_embedding.parquet")
        _create_confidence_df(n_cells).to_parquet(output_dir / "reference_confidence.parquet")

        with open(output_dir / "reference_manifest.json", "w") as f:
            json.dump(_create_manifest(n_cells), f)
        with open(output_dir / "feature_overlap_report.json", "w") as f:
            json.dump(_create_feature_overlap(), f)

        validator = ReferenceGeometryValidator(output_dir, strict=True)
        report = validator.validate()

        assert not report.valid
        assert report.validation_status == "INVALID"


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_validate_reference_geometry_saves_reports(self, valid_output_dir: Path) -> None:
        """validate_reference_geometry saves JSON and markdown reports."""
        report = validate_reference_geometry(
            valid_output_dir,
            save_report=True,
        )

        assert report.valid
        assert (valid_output_dir / "validation_report.json").exists()
        assert (valid_output_dir / "validation_report.md").exists()

    def test_validate_reference_geometry_no_save(self, valid_output_dir: Path) -> None:
        """validate_reference_geometry can skip saving reports."""
        report = validate_reference_geometry(
            valid_output_dir,
            save_report=False,
        )

        assert report.valid
        # Should not create report files
        assert not (valid_output_dir / "validation_report.json").exists()

    def test_quick_validation_returns_bool(self, valid_output_dir: Path) -> None:
        """validate_reference_geometry_quick returns boolean."""
        result = validate_reference_geometry_quick(valid_output_dir)
        assert result is True

    def test_quick_validation_false_on_error(self, output_dir_missing_files: Path) -> None:
        """validate_reference_geometry_quick returns False on error."""
        result = validate_reference_geometry_quick(output_dir_missing_files)
        assert result is False


class TestReportSerialization:
    """Tests for report serialization."""

    def test_to_json_creates_file(self, valid_output_dir: Path, tmp_path: Path) -> None:
        """Report can be saved to JSON file."""
        validator = ReferenceGeometryValidator(valid_output_dir)
        report = validator.validate()

        output_path = tmp_path / "test_report.json"
        report.to_json(output_path)

        assert output_path.exists()

        with open(output_path) as f:
            loaded = json.load(f)

        assert loaded["valid"] == report.valid
        assert loaded["cell_count"] == report.cell_count

    def test_report_is_json_serializable(self, valid_output_dir: Path) -> None:
        """Report dict is fully JSON serializable."""
        validator = ReferenceGeometryValidator(valid_output_dir)
        report = validator.validate()

        # This should not raise
        json_str = json.dumps(report.to_dict())
        assert len(json_str) > 0
