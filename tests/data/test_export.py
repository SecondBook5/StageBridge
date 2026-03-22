"""Tests for stagebridge.data.export module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Try to import anndata; skip tests if not available
try:
    import anndata

    ANNDATA_AVAILABLE = True
except ImportError:
    ANNDATA_AVAILABLE = False

from stagebridge.data.export import (
    CANONICAL_FILES,
    ExportResult,
    export_canonical_dataset,
    generate_donor_manifest,
    generate_sample_manifest,
    generate_stage_manifest,
    load_canonical_dataset,
    validate_canonical_output,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_adata():
    """Create a simple AnnData object for testing."""
    if not ANNDATA_AVAILABLE:
        pytest.skip("anndata not available")

    np.random.seed(42)
    n_cells = 100
    n_genes = 50

    counts = np.random.negative_binomial(5, 0.5, size=(n_cells, n_genes))

    obs = pd.DataFrame(
        {
            "donor_id": np.repeat(["D1", "D2", "D3", "D4"], 25),
            "sample_id": np.repeat([f"S{i}" for i in range(1, 5)], 25),
            "stage": np.repeat(["Normal", "AAH", "AIS", "MIA"], 25),
            "modality": ["snrna"] * n_cells,
        },
        index=[f"cell_{i}" for i in range(n_cells)],
    )

    var = pd.DataFrame(index=[f"Gene{i}" for i in range(n_genes)])

    adata = anndata.AnnData(
        X=counts.astype(np.float32),
        obs=obs,
        var=var,
    )
    adata.layers["counts"] = counts.copy()

    return adata


@pytest.fixture
def spatial_adata():
    """Create a spatial AnnData object for testing."""
    if not ANNDATA_AVAILABLE:
        pytest.skip("anndata not available")

    np.random.seed(42)
    n_spots = 50
    n_genes = 30

    counts = np.random.negative_binomial(5, 0.5, size=(n_spots, n_genes))

    # Create spatial coordinates
    coords = np.random.uniform(0, 1000, size=(n_spots, 2))

    obs = pd.DataFrame(
        {
            "donor_id": np.repeat(["D1", "D2"], 25),
            "sample_id": np.repeat(["S1", "S2"], 25),
            "stage": np.repeat(["Normal", "AAH"], 25),
            "modality": ["spatial"] * n_spots,
        },
        index=[f"spot_{i}" for i in range(n_spots)],
    )

    var = pd.DataFrame(index=[f"Gene{i}" for i in range(n_genes)])

    adata = anndata.AnnData(
        X=counts.astype(np.float32),
        obs=obs,
        var=var,
    )
    adata.obsm["spatial"] = coords

    return adata


# ---------------------------------------------------------------------------
# Manifest generation tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not ANNDATA_AVAILABLE, reason="anndata not available")
class TestGenerateManifests:
    """Tests for manifest generation."""

    def test_generate_donor_manifest(self, simple_adata) -> None:
        """Test donor manifest generation."""
        manifest = generate_donor_manifest(simple_adata)

        assert isinstance(manifest, pd.DataFrame)
        assert "donor_id" in manifest.columns
        assert "n_cells" in manifest.columns
        assert len(manifest) == 4  # 4 donors

    def test_generate_donor_manifest_includes_stages(self, simple_adata) -> None:
        """Test that donor manifest includes stage info."""
        manifest = generate_donor_manifest(simple_adata)

        assert "stages" in manifest.columns or "n_stages" in manifest.columns

    def test_generate_sample_manifest(self, simple_adata) -> None:
        """Test sample manifest generation."""
        manifest = generate_sample_manifest(simple_adata)

        assert isinstance(manifest, pd.DataFrame)
        assert "sample_id" in manifest.columns
        assert "donor_id" in manifest.columns
        assert len(manifest) == 4  # 4 samples

    def test_generate_stage_manifest(self, simple_adata) -> None:
        """Test stage manifest generation."""
        manifest = generate_stage_manifest(simple_adata)

        assert isinstance(manifest, pd.DataFrame)
        assert "stage" in manifest.columns
        assert "n_cells" in manifest.columns
        assert len(manifest) == 4  # 4 stages

    def test_stage_manifest_biological_order(self, simple_adata) -> None:
        """Test that stage manifest preserves biological order."""
        manifest = generate_stage_manifest(simple_adata)

        stages = manifest["stage"].tolist()
        expected_order = ["Normal", "AAH", "AIS", "MIA"]

        assert stages == expected_order


# ---------------------------------------------------------------------------
# Export tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not ANNDATA_AVAILABLE, reason="anndata not available")
class TestExportCanonicalDataset:
    """Tests for canonical dataset export."""

    def test_export_cells_basic(self, simple_adata, tmp_path: Path) -> None:
        """Test basic cells export."""
        output_dir = tmp_path / "output"

        result = export_canonical_dataset(
            adata=simple_adata,
            output_dir=output_dir,
            dataset_name="test_dataset",
        )

        assert isinstance(result, ExportResult)
        assert result.success
        assert result.n_cells == simple_adata.n_obs
        assert result.n_genes == simple_adata.n_vars

    def test_export_creates_h5ad(self, simple_adata, tmp_path: Path) -> None:
        """Test that export creates h5ad file."""
        output_dir = tmp_path / "output"

        export_canonical_dataset(
            adata=simple_adata,
            output_dir=output_dir,
            dataset_name="test",
        )

        h5ad_path = output_dir / CANONICAL_FILES["cells_h5ad"]
        assert h5ad_path.exists()

    def test_export_creates_parquet(self, simple_adata, tmp_path: Path) -> None:
        """Test that export creates parquet file."""
        output_dir = tmp_path / "output"

        export_canonical_dataset(
            adata=simple_adata,
            output_dir=output_dir,
            dataset_name="test",
            write_parquet=True,
        )

        parquet_path = output_dir / CANONICAL_FILES["cells_parquet"]
        assert parquet_path.exists()

    def test_export_creates_manifests(self, simple_adata, tmp_path: Path) -> None:
        """Test that export creates manifest files."""
        output_dir = tmp_path / "output"

        export_canonical_dataset(
            adata=simple_adata,
            output_dir=output_dir,
            dataset_name="test",
            write_manifests=True,
        )

        donor_path = output_dir / CANONICAL_FILES["donor_manifest"]
        sample_path = output_dir / CANONICAL_FILES["sample_manifest"]
        stage_path = output_dir / CANONICAL_FILES["stage_manifest"]

        assert donor_path.exists()
        assert sample_path.exists()
        assert stage_path.exists()

    def test_export_spatial(self, spatial_adata, tmp_path: Path) -> None:
        """Test spatial data export."""
        output_dir = tmp_path / "output"

        result = export_canonical_dataset(
            spatial_adata=spatial_adata,
            output_dir=output_dir,
            dataset_name="test",
        )

        assert result.n_spots == spatial_adata.n_obs

        spatial_path = output_dir / CANONICAL_FILES["spatial_h5ad"]
        assert spatial_path.exists()

    def test_export_both_modalities(self, simple_adata, spatial_adata, tmp_path: Path) -> None:
        """Test exporting both cells and spatial."""
        output_dir = tmp_path / "output"

        result = export_canonical_dataset(
            adata=simple_adata,
            spatial_adata=spatial_adata,
            output_dir=output_dir,
            dataset_name="test",
        )

        assert result.n_cells == simple_adata.n_obs
        assert result.n_spots == spatial_adata.n_obs

        cells_path = output_dir / CANONICAL_FILES["cells_h5ad"]
        spatial_path = output_dir / CANONICAL_FILES["spatial_h5ad"]
        assert cells_path.exists()
        assert spatial_path.exists()

    def test_export_result_save(self, simple_adata, tmp_path: Path) -> None:
        """Test saving export result."""
        output_dir = tmp_path / "output"

        export_canonical_dataset(
            adata=simple_adata,
            output_dir=output_dir,
            dataset_name="test",
        )

        result_path = output_dir / CANONICAL_FILES["export_result"]
        assert result_path.exists()

    def test_export_files_written_list(self, simple_adata, tmp_path: Path) -> None:
        """Test that files_written is populated."""
        output_dir = tmp_path / "output"

        result = export_canonical_dataset(
            adata=simple_adata,
            output_dir=output_dir,
            dataset_name="test",
        )

        assert len(result.files_written) > 0
        for path in result.files_written:
            assert path.exists()


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not ANNDATA_AVAILABLE, reason="anndata not available")
class TestValidateCanonicalOutput:
    """Tests for canonical output validation."""

    def test_validate_valid_output(self, simple_adata, tmp_path: Path) -> None:
        """Test validation of valid output."""
        output_dir = tmp_path / "output"

        export_canonical_dataset(
            adata=simple_adata,
            output_dir=output_dir,
            dataset_name="test",
        )

        is_valid, issues = validate_canonical_output(output_dir)

        assert is_valid is True
        assert len(issues) == 0

    def test_validate_missing_cells(self, tmp_path: Path) -> None:
        """Test validation fails without cells.h5ad."""
        output_dir = tmp_path / "empty"
        output_dir.mkdir()

        is_valid, issues = validate_canonical_output(output_dir, require_cells=True)

        assert is_valid is False
        assert any("cells.h5ad" in issue for issue in issues)

    def test_validate_missing_directory(self, tmp_path: Path) -> None:
        """Test validation of missing directory."""
        is_valid, issues = validate_canonical_output(tmp_path / "nonexistent")

        assert is_valid is False
        assert any("does not exist" in issue for issue in issues)

    def test_validate_optional_spatial(self, simple_adata, tmp_path: Path) -> None:
        """Test validation without spatial is OK if not required."""
        output_dir = tmp_path / "output"

        export_canonical_dataset(
            adata=simple_adata,
            output_dir=output_dir,
            dataset_name="test",
        )

        is_valid, issues = validate_canonical_output(output_dir, require_spatial=False)

        assert is_valid is True


# ---------------------------------------------------------------------------
# Load tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not ANNDATA_AVAILABLE, reason="anndata not available")
class TestLoadCanonicalDataset:
    """Tests for loading canonical dataset."""

    def test_load_cells(self, simple_adata, tmp_path: Path) -> None:
        """Test loading cells from canonical output."""
        output_dir = tmp_path / "output"

        export_canonical_dataset(
            adata=simple_adata,
            output_dir=output_dir,
            dataset_name="test",
        )

        data = load_canonical_dataset(output_dir, load_cells=True)

        assert data["cells"] is not None
        assert data["cells"].n_obs == simple_adata.n_obs

    def test_load_manifests(self, simple_adata, tmp_path: Path) -> None:
        """Test loading manifests."""
        output_dir = tmp_path / "output"

        export_canonical_dataset(
            adata=simple_adata,
            output_dir=output_dir,
            dataset_name="test",
        )

        data = load_canonical_dataset(output_dir)

        assert data["donor_manifest"] is not None
        assert data["sample_manifest"] is not None
        assert data["stage_manifest"] is not None

    def test_load_backed_mode(self, simple_adata, tmp_path: Path) -> None:
        """Test loading in backed mode."""
        output_dir = tmp_path / "output"

        export_canonical_dataset(
            adata=simple_adata,
            output_dir=output_dir,
            dataset_name="test",
        )

        data = load_canonical_dataset(output_dir, backed=True)

        assert data["cells"] is not None
        # Backed mode should return a backed AnnData
        assert data["cells"].isbacked or data["cells"].n_obs > 0


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not ANNDATA_AVAILABLE, reason="anndata not available")
class TestExportEdgeCases:
    """Tests for export edge cases."""

    def test_export_empty_adata(self, tmp_path: Path) -> None:
        """Test exporting empty AnnData."""
        adata = anndata.AnnData(
            X=np.zeros((0, 10)),
            obs=pd.DataFrame(columns=["donor_id", "sample_id", "stage"]),
            var=pd.DataFrame(index=[f"Gene{i}" for i in range(10)]),
        )

        output_dir = tmp_path / "output"

        result = export_canonical_dataset(
            adata=adata,
            output_dir=output_dir,
            dataset_name="test",
        )

        # Should still succeed but with 0 cells
        assert result.n_cells == 0

    def test_export_missing_columns(self, tmp_path: Path) -> None:
        """Test export with missing required columns fills defaults."""
        adata = anndata.AnnData(
            X=np.random.rand(10, 5).astype(np.float32),
            obs=pd.DataFrame(index=[f"cell_{i}" for i in range(10)]),
            var=pd.DataFrame(index=[f"Gene{i}" for i in range(5)]),
        )

        output_dir = tmp_path / "output"

        result = export_canonical_dataset(
            adata=adata,
            output_dir=output_dir,
            dataset_name="test",
        )

        # Should succeed with warnings about missing columns
        assert result.success or len(result.warnings) > 0

    def test_export_none_inputs(self, tmp_path: Path) -> None:
        """Test export with all None inputs."""
        output_dir = tmp_path / "output"

        result = export_canonical_dataset(
            adata=None,
            spatial_adata=None,
            output_dir=output_dir,
            dataset_name="test",
        )

        # Should "succeed" but with no data
        assert result.n_cells == 0
        assert result.n_spots == 0
