"""Tests for stagebridge.data.ingest module."""

from __future__ import annotations

import gzip
import tarfile
import zipfile
from pathlib import Path

import pytest

from stagebridge.data.ingest import (
    IngestResult,
    ProvenanceRecord,
    compute_checksum,
    discover_raw_files,
    record_provenance,
    unpack_archive,
    validate_ingest_result,
    verify_checksum,
    _get_format,
    _infer_file_type,
    _infer_modality,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_data_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with sample data files."""
    data_dir = tmp_path / "raw_data"
    data_dir.mkdir()

    # Create sample files
    (data_dir / "matrix.mtx").write_text("%%MatrixMarket matrix\n1 1 1\n1 1 100")
    (data_dir / "metadata.csv").write_text("cell_id,donor_id,stage\ncell1,D1,Normal")
    (data_dir / "barcodes.tsv").write_text("ACGT\nTGCA")
    (data_dir / "tissue_positions.csv").write_text("barcode,x,y\nACGT,100,200")
    (data_dir / "tissue_image.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

    # Create subdirectory
    sub_dir = data_dir / "sample1"
    sub_dir.mkdir()
    (sub_dir / "counts.h5ad").write_bytes(b"HDF5" + b"\x00" * 100)

    return data_dir


@pytest.fixture
def temp_archive_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with archive files."""
    archive_dir = tmp_path / "archives"
    archive_dir.mkdir()

    # Create test content
    content_dir = tmp_path / "content"
    content_dir.mkdir()
    (content_dir / "file1.txt").write_text("content1")
    (content_dir / "file2.txt").write_text("content2")

    # Create tar.gz
    tar_path = archive_dir / "test.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(content_dir / "file1.txt", arcname="file1.txt")
        tar.add(content_dir / "file2.txt", arcname="file2.txt")

    # Create zip
    zip_path = archive_dir / "test.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(content_dir / "file1.txt", "file1.txt")
        zf.write(content_dir / "file2.txt", "file2.txt")

    # Create gzip
    gz_path = archive_dir / "file.txt.gz"
    with gzip.open(gz_path, "wt") as f:
        f.write("gzip content")

    return archive_dir


# ---------------------------------------------------------------------------
# Format detection tests
# ---------------------------------------------------------------------------


class TestFormatDetection:
    """Tests for file format detection."""

    def test_get_format_h5ad(self) -> None:
        """Test h5ad format detection."""
        assert _get_format(Path("data.h5ad")) == "h5ad"

    def test_get_format_mtx(self) -> None:
        """Test mtx format detection."""
        assert _get_format(Path("matrix.mtx")) == "mtx"

    def test_get_format_tar_gz(self) -> None:
        """Test tar.gz format detection."""
        assert _get_format(Path("archive.tar.gz")) == "tar.gz"

    def test_get_format_csv(self) -> None:
        """Test csv format detection."""
        assert _get_format(Path("metadata.csv")) == "csv"

    def test_get_format_unknown(self) -> None:
        """Test unknown format."""
        assert _get_format(Path("file.xyz")) == "xyz"

    def test_infer_file_type_matrix(self) -> None:
        """Test matrix file type inference."""
        assert _infer_file_type(Path("matrix.mtx")) == "matrix"
        assert _infer_file_type(Path("counts.h5ad")) == "matrix"

    def test_infer_file_type_metadata(self) -> None:
        """Test metadata file type inference."""
        assert _infer_file_type(Path("metadata.csv")) == "metadata"
        assert _infer_file_type(Path("cell_info.tsv")) == "metadata"

    def test_infer_file_type_coordinates(self) -> None:
        """Test coordinate file type inference."""
        assert _infer_file_type(Path("tissue_positions.csv")) == "coordinates"

    def test_infer_file_type_image(self) -> None:
        """Test image file type inference."""
        assert _infer_file_type(Path("tissue_image.png")) == "image"
        assert _infer_file_type(Path("hires_image.tif")) == "image"

    def test_infer_file_type_archive(self) -> None:
        """Test archive file type inference."""
        assert _infer_file_type(Path("data.tar.gz")) == "archive"
        assert _infer_file_type(Path("data.zip")) == "archive"

    def test_infer_modality_snrna(self) -> None:
        """Test snRNA modality inference."""
        assert _infer_modality(Path("/data/snrna/sample.h5ad")) == "snRNA"
        assert _infer_modality(Path("/scrna_data/counts.mtx")) == "snRNA"

    def test_infer_modality_spatial(self) -> None:
        """Test spatial modality inference."""
        assert _infer_modality(Path("/spatial/visium_sample/")) == "spatial"

    def test_infer_modality_none(self) -> None:
        """Test unknown modality."""
        assert _infer_modality(Path("/generic/data.h5ad")) is None


# ---------------------------------------------------------------------------
# Checksum tests
# ---------------------------------------------------------------------------


class TestChecksum:
    """Tests for checksum computation."""

    def test_compute_checksum_sha256(self, tmp_path: Path) -> None:
        """Test SHA256 checksum computation."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")

        checksum = compute_checksum(test_file, algorithm="sha256")
        assert checksum.startswith("sha256:")
        assert len(checksum.split(":")[1]) == 64  # SHA256 hex length

    def test_compute_checksum_md5(self, tmp_path: Path) -> None:
        """Test MD5 checksum computation."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")

        checksum = compute_checksum(test_file, algorithm="md5")
        assert checksum.startswith("md5:")
        assert len(checksum.split(":")[1]) == 32  # MD5 hex length

    def test_compute_checksum_missing_file(self, tmp_path: Path) -> None:
        """Test checksum for missing file raises error."""
        with pytest.raises(FileNotFoundError):
            compute_checksum(tmp_path / "nonexistent.txt")

    def test_verify_checksum_valid(self, tmp_path: Path) -> None:
        """Test checksum verification with valid checksum."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")

        checksum = compute_checksum(test_file)
        assert verify_checksum(test_file, checksum) is True

    def test_verify_checksum_invalid(self, tmp_path: Path) -> None:
        """Test checksum verification with invalid checksum."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")

        assert verify_checksum(test_file, "sha256:0" * 64) is False


# ---------------------------------------------------------------------------
# File discovery tests
# ---------------------------------------------------------------------------


class TestDiscoverRawFiles:
    """Tests for raw file discovery."""

    def test_discover_files_basic(self, temp_data_dir: Path) -> None:
        """Test basic file discovery."""
        result = discover_raw_files(temp_data_dir)

        assert isinstance(result, IngestResult)
        assert result.n_files > 0
        assert result.source_dir == temp_data_dir
        assert result.discovered_at is not None

    def test_discover_files_categorizes_matrix(self, temp_data_dir: Path) -> None:
        """Test that matrix files are categorized correctly."""
        result = discover_raw_files(temp_data_dir)

        matrix_names = [f.path.name for f in result.matrix_files]
        assert "matrix.mtx" in matrix_names or "counts.h5ad" in matrix_names

    def test_discover_files_categorizes_metadata(self, temp_data_dir: Path) -> None:
        """Test that metadata files are categorized correctly."""
        result = discover_raw_files(temp_data_dir)

        metadata_names = [f.path.name for f in result.metadata_files]
        assert "metadata.csv" in metadata_names

    def test_discover_files_categorizes_coordinates(self, temp_data_dir: Path) -> None:
        """Test that coordinate files are categorized correctly."""
        result = discover_raw_files(temp_data_dir)

        coord_names = [f.path.name for f in result.coordinate_files]
        assert "tissue_positions.csv" in coord_names

    def test_discover_files_categorizes_images(self, temp_data_dir: Path) -> None:
        """Test that image files are categorized correctly."""
        result = discover_raw_files(temp_data_dir)

        image_names = [f.path.name for f in result.image_files]
        assert "tissue_image.png" in image_names

    def test_discover_files_with_checksums(self, temp_data_dir: Path) -> None:
        """Test file discovery with checksum computation."""
        result = discover_raw_files(temp_data_dir, compute_checksums=True)

        # At least one file should have a checksum
        files_with_checksums = [f for f in result.files if f.checksum is not None]
        assert len(files_with_checksums) > 0

    def test_discover_files_excludes_patterns(self, temp_data_dir: Path) -> None:
        """Test file discovery with exclusion patterns."""
        # Create a file that should be excluded
        (temp_data_dir / "test.pyc").write_bytes(b"bytecode")

        result = discover_raw_files(temp_data_dir, exclude_patterns=["*.pyc"])

        file_names = [f.path.name for f in result.files]
        assert "test.pyc" not in file_names

    def test_discover_files_missing_dir(self, tmp_path: Path) -> None:
        """Test discovery with missing directory raises error."""
        with pytest.raises(FileNotFoundError):
            discover_raw_files(tmp_path / "nonexistent")

    def test_discover_files_file_instead_of_dir(self, tmp_path: Path) -> None:
        """Test discovery with file instead of directory raises error."""
        test_file = tmp_path / "file.txt"
        test_file.write_text("content")

        with pytest.raises(NotADirectoryError):
            discover_raw_files(test_file)


# ---------------------------------------------------------------------------
# Archive unpacking tests
# ---------------------------------------------------------------------------


class TestUnpackArchive:
    """Tests for archive unpacking."""

    def test_unpack_tar_gz(self, temp_archive_dir: Path, tmp_path: Path) -> None:
        """Test unpacking tar.gz archive."""
        output_dir = tmp_path / "output"
        tar_path = temp_archive_dir / "test.tar.gz"

        result = unpack_archive(tar_path, output_dir)

        assert result == output_dir
        assert (output_dir / "file1.txt").exists()
        assert (output_dir / "file2.txt").exists()
        assert (output_dir / "file1.txt").read_text() == "content1"

    def test_unpack_zip(self, temp_archive_dir: Path, tmp_path: Path) -> None:
        """Test unpacking zip archive."""
        output_dir = tmp_path / "output"
        zip_path = temp_archive_dir / "test.zip"

        result = unpack_archive(zip_path, output_dir)

        assert result == output_dir
        assert (output_dir / "file1.txt").exists()
        assert (output_dir / "file2.txt").exists()

    def test_unpack_gzip(self, temp_archive_dir: Path, tmp_path: Path) -> None:
        """Test unpacking gzip file."""
        output_dir = tmp_path / "output"
        gz_path = temp_archive_dir / "file.txt.gz"

        unpack_archive(gz_path, output_dir)

        assert (output_dir / "file.txt").exists()
        assert (output_dir / "file.txt").read_text() == "gzip content"

    def test_unpack_default_output_dir(self, temp_archive_dir: Path) -> None:
        """Test unpacking to default output directory."""
        tar_path = temp_archive_dir / "test.tar.gz"

        result = unpack_archive(tar_path)

        assert result == temp_archive_dir
        assert (temp_archive_dir / "file1.txt").exists()

    def test_unpack_missing_archive(self, tmp_path: Path) -> None:
        """Test unpacking missing archive raises error."""
        with pytest.raises(FileNotFoundError):
            unpack_archive(tmp_path / "nonexistent.tar.gz")


# ---------------------------------------------------------------------------
# Provenance tests
# ---------------------------------------------------------------------------


class TestProvenance:
    """Tests for provenance recording."""

    def test_record_provenance_basic(self, temp_data_dir: Path, tmp_path: Path) -> None:
        """Test basic provenance recording."""
        result = discover_raw_files(temp_data_dir, compute_checksums=True)
        output_path = tmp_path / "provenance.json"

        record = record_provenance(
            result.files,
            source_url="https://example.com/data",
            output_path=output_path,
            notes="Test provenance",
        )

        assert isinstance(record, ProvenanceRecord)
        assert record.source_url == "https://example.com/data"
        assert record.notes == "Test provenance"
        assert len(record.files) == len(result.files)
        assert output_path.exists()

    def test_record_provenance_with_git_commit(self, temp_data_dir: Path) -> None:
        """Test provenance with git commit."""
        result = discover_raw_files(temp_data_dir)

        record = record_provenance(
            result.files,
            git_commit="abc123",
        )

        assert record.git_commit == "abc123"

    def test_record_provenance_checksums(self, temp_data_dir: Path) -> None:
        """Test provenance includes checksums."""
        result = discover_raw_files(temp_data_dir, compute_checksums=True)

        record = record_provenance(result.files)

        # Should have checksums for files that were computed
        files_with_checksums = [f for f in result.files if f.checksum is not None]
        assert len(record.checksums) == len(files_with_checksums)


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestValidation:
    """Tests for ingest result validation."""

    def test_validate_ingest_result_valid(self, temp_data_dir: Path) -> None:
        """Test validation of valid ingest result."""
        result = discover_raw_files(temp_data_dir)

        is_valid, issues = validate_ingest_result(result)

        assert is_valid is True
        assert len(issues) == 0

    def test_validate_ingest_result_no_matrix(self, tmp_path: Path) -> None:
        """Test validation fails without matrix files."""
        # Create directory with only metadata
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "metadata.csv").write_text("col1,col2\n1,2")

        result = discover_raw_files(data_dir)

        is_valid, issues = validate_ingest_result(result)

        assert is_valid is False
        assert any("matrix" in issue.lower() for issue in issues)
