"""
Raw data ingestion, unpacking, and provenance tracking for StageBridge.

This module handles:
- Raw file discovery (matrix files, metadata, coordinates, imaging)
- Archive unpacking (tar, gz, zip)
- Provenance recording (source URLs, checksums, timestamps)
- File validation and integrity checks

Usage:
    from stagebridge.data.ingest import discover_raw_files, unpack_archive, record_provenance

    result = discover_raw_files("/path/to/raw/data")
    for archive in result.archives:
        unpack_archive(archive, output_dir)
    record_provenance(result.files, source_url="https://...", output_path=manifest)
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import tarfile
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from stagebridge.logging_utils import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class DiscoveredFile:
    """Information about a discovered raw file."""

    path: Path
    file_type: str  # matrix, metadata, coordinates, image, archive, other
    format: str  # h5ad, mtx, csv, tsv, parquet, json, tif, png, tar, gz, zip, etc.
    size_bytes: int
    checksum: str | None = None
    modality: str | None = None  # snRNA, snATAC, spatial, wes, etc.
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "path": str(self.path),
            "file_type": self.file_type,
            "format": self.format,
            "size_bytes": self.size_bytes,
            "checksum": self.checksum,
            "modality": self.modality,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DiscoveredFile":
        """Create from dictionary."""
        return cls(
            path=Path(data["path"]),
            file_type=data["file_type"],
            format=data["format"],
            size_bytes=data["size_bytes"],
            checksum=data.get("checksum"),
            modality=data.get("modality"),
            notes=data.get("notes", ""),
        )


@dataclass
class IngestResult:
    """Result of raw data ingestion and discovery."""

    source_dir: Path
    discovered_at: str
    files: list[DiscoveredFile] = field(default_factory=list)
    archives: list[DiscoveredFile] = field(default_factory=list)
    matrix_files: list[DiscoveredFile] = field(default_factory=list)
    metadata_files: list[DiscoveredFile] = field(default_factory=list)
    coordinate_files: list[DiscoveredFile] = field(default_factory=list)
    image_files: list[DiscoveredFile] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def total_size_bytes(self) -> int:
        """Total size of all discovered files."""
        return sum(f.size_bytes for f in self.files)

    @property
    def n_files(self) -> int:
        """Total number of discovered files."""
        return len(self.files)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "source_dir": str(self.source_dir),
            "discovered_at": self.discovered_at,
            "total_files": self.n_files,
            "total_size_bytes": self.total_size_bytes,
            "files": [f.to_dict() for f in self.files],
            "archives": [f.to_dict() for f in self.archives],
            "matrix_files": [f.to_dict() for f in self.matrix_files],
            "metadata_files": [f.to_dict() for f in self.metadata_files],
            "coordinate_files": [f.to_dict() for f in self.coordinate_files],
            "image_files": [f.to_dict() for f in self.image_files],
            "errors": self.errors,
            "warnings": self.warnings,
        }


@dataclass
class ProvenanceRecord:
    """Provenance tracking for data files."""

    source_url: str | None
    download_date: str | None
    files: list[dict[str, Any]]
    checksums: dict[str, str]  # path -> checksum
    total_size_bytes: int
    notes: str = ""
    git_commit: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "source_url": self.source_url,
            "download_date": self.download_date,
            "files": self.files,
            "checksums": self.checksums,
            "total_size_bytes": self.total_size_bytes,
            "notes": self.notes,
            "git_commit": self.git_commit,
        }


# ---------------------------------------------------------------------------
# File type detection
# ---------------------------------------------------------------------------

# File extension to format mapping
FORMAT_MAP: dict[str, str] = {
    ".h5ad": "h5ad",
    ".h5": "h5",
    ".hdf5": "hdf5",
    ".mtx": "mtx",
    ".mtx.gz": "mtx.gz",
    ".csv": "csv",
    ".csv.gz": "csv.gz",
    ".tsv": "tsv",
    ".tsv.gz": "tsv.gz",
    ".parquet": "parquet",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".tar": "tar",
    ".tar.gz": "tar.gz",
    ".tgz": "tgz",
    ".gz": "gz",
    ".zip": "zip",
    ".tif": "tif",
    ".tiff": "tiff",
    ".png": "png",
    ".jpg": "jpg",
    ".jpeg": "jpeg",
    ".bam": "bam",
    ".bed": "bed",
    ".vcf": "vcf",
    ".vcf.gz": "vcf.gz",
}

# Patterns for identifying file types
MATRIX_PATTERNS = [
    "matrix.mtx",
    "counts.mtx",
    "expression",
    "raw_counts",
    "filtered_feature_bc_matrix",
    "raw_feature_bc_matrix",
]
METADATA_PATTERNS = [
    "metadata",
    "obs",
    "cell_info",
    "sample_info",
    "donor",
    "clinical",
    "manifest",
    "annotations",
    "barcodes",
]
COORDINATE_PATTERNS = [
    "coordinates",
    "spatial",
    "positions",
    "tissue_positions",
    "scalefactors",
    "tissue_hires",
    "tissue_lowres",
]
IMAGE_PATTERNS = [
    "image",
    "tissue",
    "hires",
    "lowres",
    "fullres",
]
ARCHIVE_EXTENSIONS = {".tar", ".tar.gz", ".tgz", ".gz", ".zip"}


def _get_format(path: Path) -> str:
    """Determine file format from path."""
    name = path.name.lower()

    # Check for compound extensions first
    for ext in [".tar.gz", ".mtx.gz", ".csv.gz", ".tsv.gz", ".vcf.gz"]:
        if name.endswith(ext):
            return FORMAT_MAP.get(ext, ext.lstrip("."))

    # Then check single extension
    suffix = path.suffix.lower()
    return FORMAT_MAP.get(suffix, suffix.lstrip(".") or "unknown")


def _infer_file_type(path: Path) -> str:
    """Infer file type from path name and extension."""
    name = path.name.lower()
    fmt = _get_format(path)

    # Check for archives
    if fmt in {"tar", "tar.gz", "tgz", "gz", "zip"}:
        return "archive"

    # Check for images
    if fmt in {"tif", "tiff", "png", "jpg", "jpeg"}:
        return "image"

    # Check patterns
    for pattern in MATRIX_PATTERNS:
        if pattern in name:
            return "matrix"

    for pattern in METADATA_PATTERNS:
        if pattern in name:
            return "metadata"

    for pattern in COORDINATE_PATTERNS:
        if pattern in name:
            return "coordinates"

    for pattern in IMAGE_PATTERNS:
        if pattern in name:
            return "image"

    # Infer from format
    if fmt in {"h5ad", "h5", "hdf5", "mtx", "mtx.gz"}:
        return "matrix"

    if fmt in {"csv", "csv.gz", "tsv", "tsv.gz", "parquet", "json"}:
        # Could be metadata or other
        return "metadata"

    return "other"


def _infer_modality(path: Path) -> str | None:
    """Infer data modality from path."""
    path_str = str(path).lower()

    if "snrna" in path_str or "scrna" in path_str or "rna" in path_str:
        return "snRNA"
    if "snatac" in path_str or "scatac" in path_str or "atac" in path_str:
        return "snATAC"
    if "spatial" in path_str or "visium" in path_str or "10x_spatial" in path_str:
        return "spatial"
    if "wes" in path_str or "exome" in path_str:
        return "wes"
    if "wgs" in path_str or "genome" in path_str:
        return "wgs"

    return None


# ---------------------------------------------------------------------------
# Checksum computation
# ---------------------------------------------------------------------------


def compute_checksum(
    path: Path,
    algorithm: Literal["md5", "sha256", "sha1"] = "sha256",
    chunk_size: int = 8192,
) -> str:
    """Compute file checksum.

    Parameters
    ----------
    path : Path
        Path to the file.
    algorithm : str
        Hash algorithm to use (default: sha256).
    chunk_size : int
        Read chunk size in bytes.

    Returns
    -------
    str
        Checksum in format "algorithm:hexdigest".
    """
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Cannot compute checksum: {path} does not exist or is not a file")

    hasher = hashlib.new(algorithm)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hasher.update(chunk)

    return f"{algorithm}:{hasher.hexdigest()}"


def verify_checksum(path: Path, expected: str) -> bool:
    """Verify file checksum matches expected value.

    Parameters
    ----------
    path : Path
        Path to the file.
    expected : str
        Expected checksum in format "algorithm:hexdigest".

    Returns
    -------
    bool
        True if checksum matches.
    """
    if ":" not in expected:
        raise ValueError(f"Invalid checksum format: {expected}. Expected 'algorithm:hexdigest'")

    algorithm, _ = expected.split(":", 1)
    actual = compute_checksum(path, algorithm=algorithm)
    return actual == expected


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


def discover_raw_files(
    source_dir: str | Path,
    *,
    compute_checksums: bool = False,
    follow_symlinks: bool = True,
    max_depth: int | None = None,
    exclude_patterns: list[str] | None = None,
) -> IngestResult:
    """Discover raw data files in a directory.

    Scans the directory tree and categorizes files by type:
    - Matrix files (h5ad, mtx, etc.)
    - Metadata files (csv, json, etc.)
    - Coordinate files (spatial positions)
    - Image files (tissue images)
    - Archives (tar, gz, zip)

    Parameters
    ----------
    source_dir : Path
        Directory to scan.
    compute_checksums : bool
        Whether to compute file checksums (slower but useful for provenance).
    follow_symlinks : bool
        Whether to follow symbolic links.
    max_depth : int, optional
        Maximum directory depth to scan.
    exclude_patterns : list[str], optional
        Patterns to exclude (e.g., ["__pycache__", ".git"]).

    Returns
    -------
    IngestResult
        Discovery result with categorized files.
    """
    source_dir = Path(source_dir).resolve()

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

    if not source_dir.is_dir():
        raise NotADirectoryError(f"Source path is not a directory: {source_dir}")

    exclude_patterns = exclude_patterns or ["__pycache__", ".git", ".svn", ".hg", "*.pyc"]

    result = IngestResult(
        source_dir=source_dir,
        discovered_at=datetime.now(timezone.utc).isoformat(),
    )

    log.info("Discovering raw files in %s ...", source_dir)

    def _should_exclude(path: Path) -> bool:
        name = path.name
        for pattern in exclude_patterns:
            if pattern.startswith("*"):
                if name.endswith(pattern[1:]):
                    return True
            elif pattern in name:
                return True
        return False

    def _walk_dir(dir_path: Path, current_depth: int = 0) -> None:
        if max_depth is not None and current_depth > max_depth:
            return

        try:
            entries = list(dir_path.iterdir())
        except PermissionError as e:
            result.warnings.append(f"Permission denied: {dir_path}")
            return

        for entry in entries:
            if _should_exclude(entry):
                continue

            if entry.is_symlink() and not follow_symlinks:
                continue

            if entry.is_dir():
                _walk_dir(entry, current_depth + 1)
            elif entry.is_file():
                try:
                    _process_file(entry)
                except Exception as e:
                    result.errors.append(f"Error processing {entry}: {e}")

    def _process_file(path: Path) -> None:
        try:
            size = path.stat().st_size
        except OSError:
            size = 0

        fmt = _get_format(path)
        file_type = _infer_file_type(path)
        modality = _infer_modality(path)

        checksum = None
        if compute_checksums:
            try:
                checksum = compute_checksum(path)
            except Exception as e:
                result.warnings.append(f"Could not compute checksum for {path}: {e}")

        discovered = DiscoveredFile(
            path=path,
            file_type=file_type,
            format=fmt,
            size_bytes=size,
            checksum=checksum,
            modality=modality,
        )

        result.files.append(discovered)

        # Categorize
        if file_type == "archive":
            result.archives.append(discovered)
        elif file_type == "matrix":
            result.matrix_files.append(discovered)
        elif file_type == "metadata":
            result.metadata_files.append(discovered)
        elif file_type == "coordinates":
            result.coordinate_files.append(discovered)
        elif file_type == "image":
            result.image_files.append(discovered)

    _walk_dir(source_dir)

    log.info(
        "Discovered %d files: %d matrices, %d metadata, %d coordinates, %d images, %d archives",
        result.n_files,
        len(result.matrix_files),
        len(result.metadata_files),
        len(result.coordinate_files),
        len(result.image_files),
        len(result.archives),
    )

    if result.errors:
        log.warning("Encountered %d errors during discovery", len(result.errors))

    return result


# ---------------------------------------------------------------------------
# Archive unpacking
# ---------------------------------------------------------------------------


def unpack_archive(
    archive_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    overwrite: bool = False,
    remove_archive: bool = False,
) -> Path:
    """Unpack an archive file.

    Supports tar, tar.gz, tgz, gz, and zip formats.

    Parameters
    ----------
    archive_path : Path
        Path to the archive file.
    output_dir : Path, optional
        Output directory (default: same directory as archive).
    overwrite : bool
        Whether to overwrite existing files.
    remove_archive : bool
        Whether to remove the archive after unpacking.

    Returns
    -------
    Path
        Path to the output directory containing unpacked files.
    """
    archive_path = Path(archive_path).resolve()

    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    if output_dir is None:
        output_dir = archive_path.parent
    else:
        output_dir = Path(output_dir).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    fmt = _get_format(archive_path)
    log.info("Unpacking %s (%s) to %s ...", archive_path.name, fmt, output_dir)

    if fmt in {"tar", "tar.gz", "tgz"}:
        _unpack_tar(archive_path, output_dir, overwrite)
    elif fmt == "gz":
        _unpack_gzip(archive_path, output_dir, overwrite)
    elif fmt == "zip":
        _unpack_zip(archive_path, output_dir, overwrite)
    else:
        raise ValueError(f"Unsupported archive format: {fmt}")

    if remove_archive:
        log.info("Removing archive: %s", archive_path)
        archive_path.unlink()

    return output_dir


def _unpack_tar(archive_path: Path, output_dir: Path, overwrite: bool) -> None:
    """Unpack tar or tar.gz archive."""
    mode = "r:gz" if archive_path.name.endswith((".tar.gz", ".tgz")) else "r"

    with tarfile.open(archive_path, mode) as tar:
        members = tar.getmembers()

        for member in members:
            dest = output_dir / member.name

            if dest.exists() and not overwrite:
                log.debug("Skipping existing file: %s", dest)
                continue

            # Security check: prevent path traversal
            if ".." in member.name or member.name.startswith("/"):
                log.warning("Skipping potentially unsafe path: %s", member.name)
                continue

            tar.extract(member, output_dir)

        log.info("Extracted %d files from tar archive", len(members))


def _unpack_gzip(archive_path: Path, output_dir: Path, overwrite: bool) -> None:
    """Unpack gzip file."""
    # Determine output filename
    if archive_path.name.endswith(".gz"):
        output_name = archive_path.stem
    else:
        output_name = archive_path.name + ".unpacked"

    output_path = output_dir / output_name

    if output_path.exists() and not overwrite:
        log.info("Skipping existing file: %s", output_path)
        return

    with gzip.open(archive_path, "rb") as f_in:
        with output_path.open("wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    log.info("Extracted gzip to: %s", output_path)


def _unpack_zip(archive_path: Path, output_dir: Path, overwrite: bool) -> None:
    """Unpack zip archive."""
    with zipfile.ZipFile(archive_path, "r") as zf:
        members = zf.namelist()

        for member in members:
            dest = output_dir / member

            if dest.exists() and not overwrite:
                log.debug("Skipping existing file: %s", dest)
                continue

            # Security check
            if ".." in member or member.startswith("/"):
                log.warning("Skipping potentially unsafe path: %s", member)
                continue

            zf.extract(member, output_dir)

        log.info("Extracted %d files from zip archive", len(members))


# ---------------------------------------------------------------------------
# Provenance tracking
# ---------------------------------------------------------------------------


def record_provenance(
    files: list[DiscoveredFile],
    source_url: str | None = None,
    download_date: str | None = None,
    output_path: str | Path | None = None,
    notes: str = "",
    git_commit: str | None = None,
) -> ProvenanceRecord:
    """Record provenance information for data files.

    Parameters
    ----------
    files : list[DiscoveredFile]
        List of discovered files to record.
    source_url : str, optional
        Source URL (e.g., GEO accession).
    download_date : str, optional
        Download date (ISO format). If not provided, uses current time.
    output_path : Path, optional
        Path to write provenance JSON file.
    notes : str
        Additional notes.
    git_commit : str, optional
        Git commit hash for reproducibility.

    Returns
    -------
    ProvenanceRecord
        The provenance record.
    """
    if download_date is None:
        download_date = datetime.now(timezone.utc).isoformat()

    # Collect checksums
    checksums = {}
    for f in files:
        if f.checksum:
            checksums[str(f.path)] = f.checksum

    record = ProvenanceRecord(
        source_url=source_url,
        download_date=download_date,
        files=[f.to_dict() for f in files],
        checksums=checksums,
        total_size_bytes=sum(f.size_bytes for f in files),
        notes=notes,
        git_commit=git_commit,
    )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(record.to_dict(), f, indent=2)

        log.info("Wrote provenance record to %s", output_path)

    return record


def load_provenance(path: str | Path) -> ProvenanceRecord:
    """Load provenance record from JSON file.

    Parameters
    ----------
    path : Path
        Path to provenance JSON file.

    Returns
    -------
    ProvenanceRecord
        The loaded provenance record.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Provenance file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return ProvenanceRecord(
        source_url=data.get("source_url"),
        download_date=data.get("download_date"),
        files=data.get("files", []),
        checksums=data.get("checksums", {}),
        total_size_bytes=data.get("total_size_bytes", 0),
        notes=data.get("notes", ""),
        git_commit=data.get("git_commit"),
    )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def validate_ingest_result(result: IngestResult) -> tuple[bool, list[str]]:
    """Validate an ingest result.

    Checks:
    - At least one matrix file found
    - No critical errors
    - Files are accessible

    Parameters
    ----------
    result : IngestResult
        The ingest result to validate.

    Returns
    -------
    tuple[bool, list[str]]
        (is_valid, list of issues)
    """
    issues = []

    if not result.matrix_files:
        issues.append("No matrix files found")

    if result.errors:
        issues.extend(f"Error: {e}" for e in result.errors)

    # Check file accessibility
    for f in result.files[:10]:  # Check first 10 files
        if not f.path.exists():
            issues.append(f"File no longer exists: {f.path}")

    return len(issues) == 0, issues
