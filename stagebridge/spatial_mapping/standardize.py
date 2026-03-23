"""
Output standardization for spatial backends.

Ensures all backend outputs conform to a common schema for downstream use.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import numpy as np
import pandas as pd

from .backend_base import BackendMappingResult


@dataclass
class StandardizedOutput:
    """
    Standardized output schema for spatial backend results.

    This schema ensures consistent format across all backends for downstream
    StageBridge modules.
    """

    # Required: Cell type proportions (n_spots x n_celltypes)
    cell_type_proportions: pd.DataFrame

    # Required: Per-spot confidence scores (n_spots,)
    confidence: pd.Series

    # Required: Backend metadata
    backend_name: str
    backend_version: str | None = None
    backend_config: dict[str, Any] | None = None

    # Optional: Reconstructed/imputed expression
    reconstructed_expression: pd.DataFrame | None = None

    # Optional: Cell-level assignments (for cell-resolution backends)
    cell_assignments: pd.DataFrame | None = None

    # Optional: State-aware outputs
    state_proportions: pd.DataFrame | None = None

    # Metrics
    upstream_metrics: dict[str, float] | None = None

    def validate(self) -> list[str]:
        """
        Validate that output conforms to required schema.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Check required fields
        if self.cell_type_proportions is None:
            errors.append("cell_type_proportions is required")
        else:
            # Validate proportions DataFrame
            if not isinstance(self.cell_type_proportions, pd.DataFrame):
                errors.append("cell_type_proportions must be a DataFrame")
            else:
                # Check values are in [0, 1]
                values = self.cell_type_proportions.values
                if values.min() < -1e-6:
                    errors.append(
                        f"cell_type_proportions has negative values: min={values.min():.6f}"
                    )
                if values.max() > 1 + 1e-6:
                    errors.append(f"cell_type_proportions has values > 1: max={values.max():.6f}")

                # Check rows sum to ~1
                row_sums = values.sum(axis=1)
                if not np.allclose(row_sums, 1.0, atol=1e-4):
                    errors.append(
                        f"cell_type_proportions rows don't sum to 1: "
                        f"range [{row_sums.min():.4f}, {row_sums.max():.4f}]"
                    )

        if self.confidence is None:
            errors.append("confidence is required")
        else:
            if not isinstance(self.confidence, pd.Series):
                errors.append("confidence must be a Series")
            else:
                # Check values are in [0, 1]
                if self.confidence.min() < -1e-6:
                    errors.append(
                        f"confidence has negative values: min={self.confidence.min():.6f}"
                    )
                if self.confidence.max() > 1 + 1e-6:
                    errors.append(f"confidence has values > 1: max={self.confidence.max():.6f}")

        # Check index alignment
        if self.cell_type_proportions is not None and self.confidence is not None:
            if not self.cell_type_proportions.index.equals(self.confidence.index):
                errors.append("cell_type_proportions and confidence have mismatched indices")

        # Check backend name
        if not self.backend_name:
            errors.append("backend_name is required")

        return errors

    def save(self, output_dir: Path) -> None:
        """
        Save standardized output to directory.

        Creates:
        - cell_type_proportions.parquet
        - mapping_confidence.parquet
        - backend_metadata.json
        - upstream_metrics.json (if available)
        - reconstructed_expression.parquet (if available)
        - cell_assignments.parquet (if available)
        - state_proportions.parquet (if available)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save required outputs
        self.cell_type_proportions.to_parquet(output_dir / "cell_type_proportions.parquet")
        self.confidence.to_frame("confidence").to_parquet(
            output_dir / "mapping_confidence.parquet"
        )

        # Save metadata
        metadata = {
            "backend_name": self.backend_name,
            "backend_version": self.backend_version,
            "backend_config": self.backend_config,
            "n_spots": len(self.cell_type_proportions),
            "n_celltypes": self.cell_type_proportions.shape[1],
            "cell_types": self.cell_type_proportions.columns.tolist(),
        }

        with open(output_dir / "backend_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Save metrics
        if self.upstream_metrics:
            with open(output_dir / "upstream_metrics.json", "w") as f:
                json.dump(self.upstream_metrics, f, indent=2)

        # Save optional outputs
        if self.reconstructed_expression is not None:
            self.reconstructed_expression.to_parquet(
                output_dir / "reconstructed_expression.parquet"
            )

        if self.cell_assignments is not None:
            self.cell_assignments.to_parquet(output_dir / "cell_assignments.parquet")

        if self.state_proportions is not None:
            self.state_proportions.to_parquet(output_dir / "state_proportions.parquet")

    @classmethod
    def load(cls, output_dir: Path) -> "StandardizedOutput":
        """Load standardized output from directory."""
        output_dir = Path(output_dir)

        # Load required outputs
        cell_type_proportions = pd.read_parquet(output_dir / "cell_type_proportions.parquet")
        confidence = pd.read_parquet(output_dir / "mapping_confidence.parquet")["confidence"]

        # Load metadata
        with open(output_dir / "backend_metadata.json") as f:
            metadata = json.load(f)

        # Load optional metrics
        upstream_metrics = None
        if (output_dir / "upstream_metrics.json").exists():
            with open(output_dir / "upstream_metrics.json") as f:
                upstream_metrics = json.load(f)

        # Load optional outputs
        reconstructed_expression = None
        if (output_dir / "reconstructed_expression.parquet").exists():
            reconstructed_expression = pd.read_parquet(
                output_dir / "reconstructed_expression.parquet"
            )

        cell_assignments = None
        if (output_dir / "cell_assignments.parquet").exists():
            cell_assignments = pd.read_parquet(output_dir / "cell_assignments.parquet")

        state_proportions = None
        if (output_dir / "state_proportions.parquet").exists():
            state_proportions = pd.read_parquet(output_dir / "state_proportions.parquet")

        return cls(
            cell_type_proportions=cell_type_proportions,
            confidence=confidence,
            backend_name=metadata["backend_name"],
            backend_version=metadata.get("backend_version"),
            backend_config=metadata.get("backend_config"),
            reconstructed_expression=reconstructed_expression,
            cell_assignments=cell_assignments,
            state_proportions=state_proportions,
            upstream_metrics=upstream_metrics,
        )


def standardize_backend_output(
    result: BackendMappingResult,
    backend_name: str,
    backend_version: str | None = None,
) -> StandardizedOutput:
    """
    Convert a BackendMappingResult to StandardizedOutput.

    This ensures the result conforms to the common schema.

    Args:
        result: Raw BackendMappingResult from a backend
        backend_name: Name of the backend
        backend_version: Optional version string

    Returns:
        StandardizedOutput conforming to schema
    """
    # Ensure proportions are normalized
    proportions = result.cell_type_proportions.copy()

    # Clip to valid range
    proportions = proportions.clip(lower=0)

    # Renormalize rows to sum to 1
    row_sums = proportions.sum(axis=1)
    proportions = proportions.div(row_sums, axis=0).fillna(0)

    # Handle any remaining edge cases
    zero_rows = proportions.sum(axis=1) == 0
    if zero_rows.any():
        # Assign uniform distribution to zero rows
        n_types = proportions.shape[1]
        proportions.loc[zero_rows] = 1.0 / n_types

    # Ensure confidence is in [0, 1]
    confidence = result.confidence.clip(lower=0, upper=1)

    return StandardizedOutput(
        cell_type_proportions=proportions,
        confidence=confidence,
        backend_name=backend_name,
        backend_version=backend_version,
        backend_config=result.metadata,
        reconstructed_expression=result.reconstructed_expression,
        cell_assignments=result.cell_assignments,
        upstream_metrics=result.upstream_metrics,
    )


def validate_standardized_output(result: StandardizedOutput) -> tuple[bool, list[str]]:
    """
    Validate a standardized output conforms to schema.

    Args:
        result: StandardizedOutput to validate

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = result.validate()
    return len(errors) == 0, errors


def merge_standardized_outputs(
    outputs: dict[str, StandardizedOutput],
    output_dir: Path,
) -> None:
    """
    Merge multiple standardized outputs into a single directory.

    Creates a comparison-ready structure:
    output_dir/
        tangram/
        destvi/
        tacco/
        comparison_index.json

    Args:
        outputs: Dictionary mapping backend name to StandardizedOutput
        output_dir: Directory to save merged outputs
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save each backend
    for backend_name, output in outputs.items():
        backend_dir = output_dir / backend_name.lower()
        output.save(backend_dir)

    # Create comparison index
    index = {
        "backends": list(outputs.keys()),
        "n_spots": {name: len(out.cell_type_proportions) for name, out in outputs.items()},
        "n_celltypes": {name: out.cell_type_proportions.shape[1] for name, out in outputs.items()},
    }

    with open(output_dir / "comparison_index.json", "w") as f:
        json.dump(index, f, indent=2)


def load_all_standardized_outputs(
    output_dir: Path,
) -> dict[str, StandardizedOutput]:
    """
    Load all standardized outputs from a comparison directory.

    Args:
        output_dir: Directory containing backend subdirectories

    Returns:
        Dictionary mapping backend name to StandardizedOutput
    """
    output_dir = Path(output_dir)

    # Load comparison index if exists
    index_path = output_dir / "comparison_index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        backend_names = index["backends"]
    else:
        # Discover backends from subdirectories
        backend_names = [
            d.name
            for d in output_dir.iterdir()
            if d.is_dir() and (d / "cell_type_proportions.parquet").exists()
        ]

    outputs = {}
    for name in backend_names:
        backend_dir = output_dir / name.lower()
        if backend_dir.exists():
            outputs[name] = StandardizedOutput.load(backend_dir)

    return outputs
