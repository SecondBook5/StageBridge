"""Validation checks for dual-reference mapping pipeline outputs.

Expected outputs in reference_geometry/:
- hlca_embedding.parquet (cell_id, donor_id, sample_id, stage_id, hlca_latent_0..29)
- luca_embedding.parquet (cell_id, donor_id, sample_id, stage_id, luca_latent_0..9)
- fused_embedding.parquet (cell_id, donor_id, sample_id, stage_id, reference_mode_used, fused_latent_0..39)
- reference_confidence.parquet (cell_id, hlca_confidence, luca_confidence, hlca_raw_distance, luca_raw_distance, ...)
- reference_manifest.json
- feature_overlap_report.json
- diagnostics_report.json

Validation checks:
1. Artifact contract: All expected files exist with correct columns
2. No NaN in embeddings: Final outputs should have zero NaN
3. Confidence calibration: Mean confidence should be ~0.5 (percentile rank calibration)
4. Donor preservation: All donors in query should appear in outputs
5. Cell count match: Output cell count should match expected
6. Latent dimensions: HLCA=30, LuCA=10, Fused=40
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from stagebridge.logging_utils import get_logger

log = get_logger(__name__)


# Expected output file names
EXPECTED_FILES = {
    "hlca_embedding": "hlca_embedding.parquet",
    "luca_embedding": "luca_embedding.parquet",
    "fused_embedding": "fused_embedding.parquet",
    "confidence": "reference_confidence.parquet",
    "manifest": "reference_manifest.json",
    "feature_overlap": "feature_overlap_report.json",
    "diagnostics": "diagnostics_report.json",  # Optional
}

# Expected latent dimensions
EXPECTED_HLCA_DIM = 30
EXPECTED_LUCA_DIM = 10
EXPECTED_FUSED_DIM = 40

# Expected metadata columns
METADATA_COLS = ["cell_id", "donor_id", "sample_id", "stage_id"]

# Confidence columns (required and optional)
CONFIDENCE_REQUIRED_COLS = ["cell_id", "hlca_confidence", "luca_confidence"]
CONFIDENCE_OPTIONAL_COLS = [
    "hlca_raw_distance",
    "luca_raw_distance",
    "hlca_confidence_method",
    "luca_confidence_method",
    "reference_mode_used",
]


@dataclass
class ReferenceGeometryValidationReport:
    """Comprehensive validation report for reference geometry outputs."""

    # Overall status
    valid: bool = True
    validation_status: Literal["VALID", "INVALID", "VALID_WITH_WARNINGS"] = "VALID"

    # File existence
    files_found: list[str] = field(default_factory=list)
    files_missing: list[str] = field(default_factory=list)

    # Schema validation
    schema_errors: list[str] = field(default_factory=list)
    schema_warnings: list[str] = field(default_factory=list)

    # NaN checks
    nan_report: dict[str, Any] = field(default_factory=dict)
    has_nan: bool = False

    # Confidence calibration
    confidence_report: dict[str, Any] = field(default_factory=dict)
    confidence_calibration_ok: bool = True

    # Cell/donor counts
    cell_count: int = 0
    donor_count: int = 0
    expected_cell_count: int | None = None
    cell_count_match: bool = True

    # Latent dimensions
    hlca_dim: int = 0
    luca_dim: int = 0
    fused_dim: int = 0
    latent_dims_ok: bool = True

    # Donor preservation
    donors_preserved: bool = True
    missing_donors: list[str] = field(default_factory=list)

    # Manifest info
    manifest_info: dict[str, Any] = field(default_factory=dict)

    # Errors and warnings
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert report to JSON-serializable dictionary."""
        return {
            "valid": self.valid,
            "validation_status": self.validation_status,
            "files_found": self.files_found,
            "files_missing": self.files_missing,
            "schema_errors": self.schema_errors,
            "schema_warnings": self.schema_warnings,
            "nan_report": self.nan_report,
            "has_nan": self.has_nan,
            "confidence_report": self.confidence_report,
            "confidence_calibration_ok": self.confidence_calibration_ok,
            "cell_count": self.cell_count,
            "donor_count": self.donor_count,
            "expected_cell_count": self.expected_cell_count,
            "cell_count_match": self.cell_count_match,
            "hlca_dim": self.hlca_dim,
            "luca_dim": self.luca_dim,
            "fused_dim": self.fused_dim,
            "latent_dims_ok": self.latent_dims_ok,
            "donors_preserved": self.donors_preserved,
            "missing_donors": self.missing_donors,
            "manifest_info": self.manifest_info,
            "errors": self.errors,
            "warnings": self.warnings,
        }

    def to_json(self, output_path: Path) -> None:
        """Write report to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def to_markdown(self) -> str:
        """Generate markdown summary of the report."""
        lines = [
            "# Reference Geometry Validation Report",
            "",
            f"**Status**: {self.validation_status}",
            "",
            "## Summary",
            "",
            f"- Cell count: {self.cell_count:,}",
            f"- Donor count: {self.donor_count}",
            f"- HLCA latent dim: {self.hlca_dim}",
            f"- LuCA latent dim: {self.luca_dim}",
            f"- Fused latent dim: {self.fused_dim}",
            "",
            "## File Checks",
            "",
        ]

        if self.files_found:
            lines.append("**Found:**")
            for f in self.files_found:
                lines.append(f"- [x] {f}")
            lines.append("")

        if self.files_missing:
            lines.append("**Missing:**")
            for f in self.files_missing:
                lines.append(f"- [ ] {f} (REQUIRED)")
            lines.append("")

        lines.extend(
            [
                "## Quality Checks",
                "",
                f"- NaN in embeddings: {'FAIL' if self.has_nan else 'PASS'}",
                f"- Confidence calibration: {'PASS' if self.confidence_calibration_ok else 'WARNING'}",
                f"- Latent dimensions: {'PASS' if self.latent_dims_ok else 'FAIL'}",
                f"- Donors preserved: {'PASS' if self.donors_preserved else 'FAIL'}",
                "",
            ]
        )

        if self.errors:
            lines.extend(
                [
                    "## Errors (BLOCKING)",
                    "",
                ]
            )
            for err in self.errors:
                lines.append(f"- {err}")
            lines.append("")

        if self.warnings:
            lines.extend(
                [
                    "## Warnings",
                    "",
                ]
            )
            for warn in self.warnings:
                lines.append(f"- {warn}")
            lines.append("")

        if self.confidence_report:
            lines.extend(
                [
                    "## Confidence Calibration Details",
                    "",
                ]
            )
            if "hlca" in self.confidence_report:
                h = self.confidence_report["hlca"]
                lines.append(
                    f"- HLCA: mean={h.get('mean', 'N/A'):.3f}, std={h.get('std', 'N/A'):.3f}"
                )
            if "luca" in self.confidence_report:
                luca_stats = self.confidence_report["luca"]
                lines.append(
                    f"- LuCA: mean={luca_stats.get('mean', 'N/A'):.3f}, std={luca_stats.get('std', 'N/A'):.3f}"
                )
            lines.append("")

        return "\n".join(lines)


class ReferenceGeometryValidator:
    """Validator for dual-reference mapping pipeline outputs."""

    def __init__(
        self,
        output_dir: Path,
        *,
        expected_cell_count: int | None = None,
        expected_donors: list[str] | None = None,
        expected_hlca_dim: int = EXPECTED_HLCA_DIM,
        expected_luca_dim: int = EXPECTED_LUCA_DIM,
        expected_fused_dim: int = EXPECTED_FUSED_DIM,
        confidence_mean_tolerance: float = 0.15,  # Allow mean in [0.35, 0.65]
        strict: bool = False,
    ):
        """Initialize validator.

        Parameters
        ----------
        output_dir : Path
            Path to reference_geometry/ directory
        expected_cell_count : int, optional
            Expected number of cells (if known from query data)
        expected_donors : list[str], optional
            Expected donor IDs (if known from query data)
        expected_hlca_dim : int
            Expected HLCA latent dimension (default: 30)
        expected_luca_dim : int
            Expected LuCA latent dimension (default: 10)
        expected_fused_dim : int
            Expected fused latent dimension (default: 40)
        confidence_mean_tolerance : float
            How far from 0.5 the mean confidence can be (percentile rank should be ~0.5)
        strict : bool
            If True, warnings are treated as errors
        """
        self.output_dir = Path(output_dir)
        self.expected_cell_count = expected_cell_count
        self.expected_donors = expected_donors
        self.expected_hlca_dim = expected_hlca_dim
        self.expected_luca_dim = expected_luca_dim
        self.expected_fused_dim = expected_fused_dim
        self.confidence_mean_tolerance = confidence_mean_tolerance
        self.strict = strict

        self._report = ReferenceGeometryValidationReport()
        self._dataframes: dict[str, pd.DataFrame] = {}
        self._manifest: dict[str, Any] = {}

    def validate(self) -> ReferenceGeometryValidationReport:
        """Run all validation checks and return report."""
        self._report = ReferenceGeometryValidationReport()

        # 1. Check file existence
        self._check_files_exist()

        # If critical files are missing, abort early
        if (
            "hlca_embedding" in self._report.files_missing
            or "fused_embedding" in self._report.files_missing
            or "confidence" in self._report.files_missing
        ):
            self._report.valid = False
            self._report.validation_status = "INVALID"
            self._report.errors.append("Critical files missing - cannot proceed with validation")
            return self._report

        # 2. Load dataframes
        self._load_dataframes()

        # 3. Check schemas
        self._check_schemas()

        # 4. Check for NaN
        self._check_nan()

        # 5. Check confidence calibration
        self._check_confidence_calibration()

        # 6. Check cell count
        self._check_cell_count()

        # 7. Check latent dimensions
        self._check_latent_dimensions()

        # 8. Check donor preservation
        self._check_donor_preservation()

        # 9. Load and validate manifest
        self._check_manifest()

        # Determine final status
        self._determine_final_status()

        return self._report

    def _check_files_exist(self) -> None:
        """Check that all expected files exist."""
        for name, filename in EXPECTED_FILES.items():
            filepath = self.output_dir / filename
            if filepath.exists():
                self._report.files_found.append(filename)
            else:
                if name == "diagnostics":
                    # diagnostics_report.json is optional
                    self._report.warnings.append(f"Optional file missing: {filename}")
                else:
                    self._report.files_missing.append(filename)
                    self._report.errors.append(f"Required file missing: {filename}")

    def _load_dataframes(self) -> None:
        """Load parquet files into dataframes."""
        for name in ["hlca_embedding", "luca_embedding", "fused_embedding", "confidence"]:
            filepath = self.output_dir / EXPECTED_FILES[name]
            if filepath.exists():
                try:
                    self._dataframes[name] = pd.read_parquet(filepath)
                except Exception as e:
                    self._report.errors.append(f"Failed to load {filepath.name}: {e}")

    def _check_schemas(self) -> None:
        """Check that dataframes have expected columns."""
        # Check HLCA embedding
        if "hlca_embedding" in self._dataframes:
            df = self._dataframes["hlca_embedding"]
            self._check_metadata_cols(df, "hlca_embedding")
            latent_cols = [c for c in df.columns if c.startswith("hlca_latent_")]
            self._report.hlca_dim = len(latent_cols)
            if len(latent_cols) == 0:
                self._report.schema_errors.append("hlca_embedding: No hlca_latent_* columns found")

        # Check LuCA embedding
        if "luca_embedding" in self._dataframes:
            df = self._dataframes["luca_embedding"]
            self._check_metadata_cols(df, "luca_embedding")
            latent_cols = [c for c in df.columns if c.startswith("luca_latent_")]
            self._report.luca_dim = len(latent_cols)
            if len(latent_cols) == 0:
                self._report.schema_errors.append("luca_embedding: No luca_latent_* columns found")

        # Check fused embedding
        if "fused_embedding" in self._dataframes:
            df = self._dataframes["fused_embedding"]
            self._check_metadata_cols(df, "fused_embedding")
            latent_cols = [c for c in df.columns if c.startswith("fused_latent_")]
            self._report.fused_dim = len(latent_cols)
            if len(latent_cols) == 0:
                self._report.schema_errors.append(
                    "fused_embedding: No fused_latent_* columns found"
                )
            if "reference_mode_used" not in df.columns:
                self._report.schema_warnings.append(
                    "fused_embedding: Missing reference_mode_used column"
                )

        # Check confidence
        if "confidence" in self._dataframes:
            df = self._dataframes["confidence"]
            for col in CONFIDENCE_REQUIRED_COLS:
                if col not in df.columns:
                    self._report.schema_errors.append(
                        f"reference_confidence: Missing required column '{col}'"
                    )

        if self._report.schema_errors:
            self._report.errors.extend(self._report.schema_errors)

    def _check_metadata_cols(self, df: pd.DataFrame, name: str) -> None:
        """Check that metadata columns exist."""
        for col in METADATA_COLS:
            if col not in df.columns:
                self._report.schema_errors.append(f"{name}: Missing metadata column '{col}'")

    def _check_nan(self) -> None:
        """Check for NaN values in embeddings."""
        nan_report = {}

        for name, df in self._dataframes.items():
            if name == "confidence":
                # Check confidence columns
                nan_counts = {}
                for col in ["hlca_confidence", "luca_confidence"]:
                    if col in df.columns:
                        nan_count = int(df[col].isna().sum())
                        if nan_count > 0:
                            nan_counts[col] = nan_count
                            self._report.has_nan = True
                if nan_counts:
                    nan_report["confidence"] = nan_counts
            else:
                # Check latent columns
                prefix = name.replace("_embedding", "_latent_")
                if name == "fused_embedding":
                    prefix = "fused_latent_"
                latent_cols = [c for c in df.columns if c.startswith(prefix)]
                if latent_cols:
                    nan_mask = df[latent_cols].isna()
                    total_nan = int(nan_mask.sum().sum())
                    cells_with_nan = int(nan_mask.any(axis=1).sum())
                    if total_nan > 0:
                        nan_report[name] = {
                            "total_nan": total_nan,
                            "cells_with_nan": cells_with_nan,
                            "fraction": cells_with_nan / len(df) if len(df) > 0 else 0,
                        }
                        self._report.has_nan = True

        self._report.nan_report = nan_report
        if self._report.has_nan:
            self._report.errors.append(f"NaN values detected in embeddings: {nan_report}")

    def _check_confidence_calibration(self) -> None:
        """Check that confidence scores are properly calibrated (mean ~0.5 for percentile rank)."""
        if "confidence" not in self._dataframes:
            self._report.confidence_calibration_ok = False
            return

        df = self._dataframes["confidence"]
        conf_report = {}

        for ref, col in [("hlca", "hlca_confidence"), ("luca", "luca_confidence")]:
            if col in df.columns:
                conf_values = df[col].dropna()
                if len(conf_values) > 0:
                    mean_conf = float(conf_values.mean())
                    std_conf = float(conf_values.std())
                    min_conf = float(conf_values.min())
                    max_conf = float(conf_values.max())

                    conf_report[ref] = {
                        "mean": mean_conf,
                        "std": std_conf,
                        "min": min_conf,
                        "max": max_conf,
                        "n_cells": len(conf_values),
                    }

                    # Check calibration: for percentile rank, mean should be ~0.5
                    if abs(mean_conf - 0.5) > self.confidence_mean_tolerance:
                        self._report.confidence_calibration_ok = False
                        self._report.warnings.append(
                            f"{ref.upper()} confidence mean={mean_conf:.3f}, expected ~0.5 for percentile rank calibration"
                        )

                    # Check valid range [0, 1]
                    if min_conf < 0 or max_conf > 1:
                        self._report.errors.append(
                            f"{ref.upper()} confidence out of [0,1] range: min={min_conf:.3f}, max={max_conf:.3f}"
                        )

        self._report.confidence_report = conf_report

    def _check_cell_count(self) -> None:
        """Check cell counts across files."""
        cell_counts = {}
        for name, df in self._dataframes.items():
            cell_counts[name] = len(df)

        # All files should have same cell count
        unique_counts = set(cell_counts.values())
        if len(unique_counts) > 1:
            self._report.errors.append(f"Cell count mismatch across files: {cell_counts}")
            self._report.cell_count_match = False

        if cell_counts:
            self._report.cell_count = list(cell_counts.values())[0]

        # Check against expected
        if self.expected_cell_count is not None:
            if self._report.cell_count != self.expected_cell_count:
                self._report.cell_count_match = False
                self._report.errors.append(
                    f"Cell count mismatch: got {self._report.cell_count}, expected {self.expected_cell_count}"
                )
            self._report.expected_cell_count = self.expected_cell_count

    def _check_latent_dimensions(self) -> None:
        """Check latent dimensions match expected values."""
        dim_errors = []

        if self._report.hlca_dim != self.expected_hlca_dim:
            dim_errors.append(
                f"HLCA dim={self._report.hlca_dim}, expected {self.expected_hlca_dim}"
            )

        if self._report.luca_dim != self.expected_luca_dim:
            dim_errors.append(
                f"LuCA dim={self._report.luca_dim}, expected {self.expected_luca_dim}"
            )

        if self._report.fused_dim != self.expected_fused_dim:
            dim_errors.append(
                f"Fused dim={self._report.fused_dim}, expected {self.expected_fused_dim}"
            )

        if dim_errors:
            self._report.latent_dims_ok = False
            for err in dim_errors:
                self._report.warnings.append(f"Latent dimension mismatch: {err}")

    def _check_donor_preservation(self) -> None:
        """Check that all expected donors appear in outputs."""
        if "hlca_embedding" in self._dataframes:
            df = self._dataframes["hlca_embedding"]
            if "donor_id" in df.columns:
                output_donors = set(df["donor_id"].unique())
                self._report.donor_count = len(output_donors)

                if self.expected_donors is not None:
                    expected_set = set(self.expected_donors)
                    missing = expected_set - output_donors
                    if missing:
                        self._report.donors_preserved = False
                        self._report.missing_donors = list(missing)
                        self._report.errors.append(f"Missing donors in output: {missing}")

    def _check_manifest(self) -> None:
        """Load and validate manifest."""
        manifest_path = self.output_dir / EXPECTED_FILES["manifest"]
        if manifest_path.exists():
            try:
                with open(manifest_path) as f:
                    self._manifest = json.load(f)
                self._report.manifest_info = {
                    "run_id": self._manifest.get("run_id"),
                    "mapping_method": self._manifest.get("mapping_method"),
                    "fusion_method": self._manifest.get("fusion_method"),
                    "n_cells": self._manifest.get("n_cells"),
                    "hlca_dim": self._manifest.get("hlca_latent_dim")
                    or self._manifest.get("hlca_dim"),
                    "luca_dim": self._manifest.get("luca_latent_dim")
                    or self._manifest.get("luca_dim"),
                }
            except json.JSONDecodeError as e:
                self._report.errors.append(f"Invalid JSON in manifest: {e}")

    def _determine_final_status(self) -> None:
        """Determine final validation status based on errors and warnings."""
        if self._report.errors:
            self._report.valid = False
            self._report.validation_status = "INVALID"
        elif self._report.warnings:
            if self.strict:
                self._report.valid = False
                self._report.validation_status = "INVALID"
            else:
                self._report.valid = True
                self._report.validation_status = "VALID_WITH_WARNINGS"
        else:
            self._report.valid = True
            self._report.validation_status = "VALID"


def validate_reference_geometry(
    output_dir: str | Path,
    *,
    expected_cell_count: int | None = None,
    expected_donors: list[str] | None = None,
    expected_hlca_dim: int = EXPECTED_HLCA_DIM,
    expected_luca_dim: int = EXPECTED_LUCA_DIM,
    expected_fused_dim: int = EXPECTED_FUSED_DIM,
    strict: bool = False,
    save_report: bool = True,
) -> ReferenceGeometryValidationReport:
    """Validate reference geometry outputs and optionally save report.

    Parameters
    ----------
    output_dir : str | Path
        Path to reference_geometry/ directory
    expected_cell_count : int, optional
        Expected number of cells
    expected_donors : list[str], optional
        Expected donor IDs
    expected_hlca_dim : int
        Expected HLCA latent dimension
    expected_luca_dim : int
        Expected LuCA latent dimension
    expected_fused_dim : int
        Expected fused latent dimension
    strict : bool
        If True, warnings are treated as errors
    save_report : bool
        If True, save JSON and markdown reports to output_dir

    Returns
    -------
    ReferenceGeometryValidationReport
        Validation report
    """
    output_dir = Path(output_dir)

    validator = ReferenceGeometryValidator(
        output_dir=output_dir,
        expected_cell_count=expected_cell_count,
        expected_donors=expected_donors,
        expected_hlca_dim=expected_hlca_dim,
        expected_luca_dim=expected_luca_dim,
        expected_fused_dim=expected_fused_dim,
        strict=strict,
    )

    report = validator.validate()

    if save_report:
        # Save JSON report
        report.to_json(output_dir / "validation_report.json")

        # Save markdown report
        md_content = report.to_markdown()
        with open(output_dir / "validation_report.md", "w") as f:
            f.write(md_content)

        log.info("Validation reports saved to %s", output_dir)

    return report


def validate_reference_geometry_quick(
    output_dir: str | Path,
) -> bool:
    """Quick validation check - returns True if outputs are valid.

    Parameters
    ----------
    output_dir : str | Path
        Path to reference_geometry/ directory

    Returns
    -------
    bool
        True if validation passes, False otherwise
    """
    report = validate_reference_geometry(
        output_dir,
        save_report=False,
    )
    return report.valid
