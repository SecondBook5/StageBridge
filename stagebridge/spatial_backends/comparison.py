"""
Backend comparison logic for spatial backend benchmark.

Provides infrastructure to run multiple backends and compare their outputs.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json
import time
import traceback
import numpy as np
import pandas as pd
import anndata as ad

from .base import SpatialBackend, BackendMappingResult
from .metrics import (
    MetricsReport,
    compute_comprehensive_metrics,
    compute_donor_robustness,
)
from .standardize import (
    StandardizedOutput,
    standardize_backend_output,
    validate_standardized_output,
)


@dataclass
class BackendRunResult:
    """Result of running a single backend."""

    backend_name: str
    success: bool
    result: BackendMappingResult | None = None
    standardized: StandardizedOutput | None = None
    metrics: MetricsReport | None = None
    error: str | None = None
    traceback: str | None = None
    runtime_seconds: float = 0.0
    memory_mb: float | None = None


@dataclass
class ComparisonResult:
    """
    Complete comparison result across all backends.

    Contains individual results, comparison table, and rankings.
    """

    # Individual results per backend
    results: dict[str, BackendRunResult] = field(default_factory=dict)

    # Comparison DataFrame
    comparison_table: pd.DataFrame | None = None

    # Rankings by different criteria
    rankings: dict[str, list[str]] = field(default_factory=dict)

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_successful_backends(self) -> list[str]:
        """Get list of backends that ran successfully."""
        return [name for name, result in self.results.items() if result.success]

    def get_failed_backends(self) -> list[str]:
        """Get list of backends that failed."""
        return [name for name, result in self.results.items() if not result.success]

    def save(self, output_dir: Path) -> None:
        """Save comparison result to directory."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save comparison table
        if self.comparison_table is not None:
            self.comparison_table.to_parquet(output_dir / "comparison_table.parquet")
            self.comparison_table.to_csv(output_dir / "comparison_table.csv")

        # Save rankings
        with open(output_dir / "rankings.json", "w") as f:
            json.dump(self.rankings, f, indent=2)

        # Save metadata
        meta = {
            "successful_backends": self.get_successful_backends(),
            "failed_backends": self.get_failed_backends(),
            **self.metadata,
        }
        with open(output_dir / "comparison_metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        # Save individual backend results
        for name, result in self.results.items():
            backend_dir = output_dir / name.lower()
            backend_dir.mkdir(parents=True, exist_ok=True)

            # Save standardized output
            if result.standardized:
                result.standardized.save(backend_dir)

            # Save metrics
            if result.metrics:
                with open(backend_dir / "backend_metrics.json", "w") as f:
                    json.dump(result.metrics.to_dict(), f, indent=2)

            # Save error info if failed
            if not result.success:
                with open(backend_dir / "error.txt", "w") as f:
                    f.write(f"Error: {result.error}\n\n")
                    if result.traceback:
                        f.write(f"Traceback:\n{result.traceback}\n")

    @classmethod
    def load(cls, output_dir: Path) -> "ComparisonResult":
        """Load comparison result from directory."""
        output_dir = Path(output_dir)

        # Load comparison table
        comparison_table = None
        if (output_dir / "comparison_table.parquet").exists():
            comparison_table = pd.read_parquet(output_dir / "comparison_table.parquet")

        # Load rankings
        rankings = {}
        if (output_dir / "rankings.json").exists():
            with open(output_dir / "rankings.json") as f:
                rankings = json.load(f)

        # Load metadata
        metadata = {}
        if (output_dir / "comparison_metadata.json").exists():
            with open(output_dir / "comparison_metadata.json") as f:
                metadata = json.load(f)

        return cls(
            comparison_table=comparison_table,
            rankings=rankings,
            metadata=metadata,
        )


def run_single_backend(
    backend: SpatialBackend,
    backend_name: str,
    snrna: ad.AnnData,
    spatial: ad.AnnData,
    output_dir: Path | None = None,
    spatial_coords: np.ndarray | None = None,
    transition_data: dict[str, Any] | None = None,
) -> BackendRunResult:
    """
    Run a single backend and collect metrics.

    Args:
        backend: Initialized backend instance
        backend_name: Name of the backend
        snrna: Single-cell reference data
        spatial: Spatial data
        output_dir: Optional output directory
        spatial_coords: Spatial coordinates for coherence metrics
        transition_data: Optional transition data for downstream metrics

    Returns:
        BackendRunResult with success status and results/error
    """
    print(f"\n{'=' * 60}")
    print(f"Running backend: {backend_name}")
    print(f"{'=' * 60}")

    start_time = time.time()

    try:
        # Run mapping
        result = backend.map(snrna, spatial, output_dir=output_dir)

        runtime = time.time() - start_time
        print(f"Backend {backend_name} completed in {runtime:.2f}s")

        # Standardize output
        standardized = standardize_backend_output(
            result,
            backend_name=backend_name,
        )

        # Validate
        is_valid, errors = validate_standardized_output(standardized)
        if not is_valid:
            print(f"Warning: Validation errors for {backend_name}: {errors}")

        # Compute metrics
        if spatial_coords is None and "spatial" in spatial.obsm:
            spatial_coords = spatial.obsm["spatial"]

        spatial_expression = pd.DataFrame(
            spatial.X if not hasattr(spatial.X, "toarray") else spatial.X.toarray(),
            index=spatial.obs_names,
            columns=spatial.var_names,
        )

        metrics = compute_comprehensive_metrics(
            result,
            spatial_coords=spatial_coords,
            spatial_expression=spatial_expression,
            transition_data=transition_data,
            runtime_seconds=runtime,
        )

        return BackendRunResult(
            backend_name=backend_name,
            success=True,
            result=result,
            standardized=standardized,
            metrics=metrics,
            runtime_seconds=runtime,
        )

    except Exception as e:
        runtime = time.time() - start_time
        error_msg = str(e)
        tb = traceback.format_exc()

        print(f"Backend {backend_name} FAILED after {runtime:.2f}s")
        print(f"Error: {error_msg}")

        return BackendRunResult(
            backend_name=backend_name,
            success=False,
            error=error_msg,
            traceback=tb,
            runtime_seconds=runtime,
        )


def run_backend_comparison(
    backends: dict[str, SpatialBackend],
    snrna: ad.AnnData,
    spatial: ad.AnnData,
    output_dir: Path,
    spatial_coords: np.ndarray | None = None,
    transition_data: dict[str, Any] | None = None,
    required_backends: list[str] | None = None,
) -> ComparisonResult:
    """
    Run all backends and produce comparison.

    Args:
        backends: Dictionary mapping backend name to initialized backend
        snrna: Single-cell reference data
        spatial: Spatial data
        output_dir: Output directory for results
        spatial_coords: Spatial coordinates for coherence metrics
        transition_data: Optional transition data for downstream metrics
        required_backends: List of required backends (fail if any fail)

    Returns:
        ComparisonResult with all results and comparison table
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if required_backends is None:
        required_backends = ["tangram", "destvi", "tacco"]

    # Run each backend
    results = {}
    for name, backend in backends.items():
        backend_output_dir = output_dir / name.lower()

        result = run_single_backend(
            backend=backend,
            backend_name=name,
            snrna=snrna,
            spatial=spatial,
            output_dir=backend_output_dir,
            spatial_coords=spatial_coords,
            transition_data=transition_data,
        )
        results[name] = result

    # Check required backends
    failed_required = [
        name
        for name in required_backends
        if name.lower() in [n.lower() for n in results.keys()]
        and not results.get(name, results.get(name.lower(), BackendRunResult(name, False))).success
    ]

    if failed_required:
        print(f"\nWARNING: Required backends failed: {failed_required}")

    # Build comparison table
    comparison_table = build_comparison_table(results)

    # Rank backends
    rankings = rank_backends(comparison_table)

    # Compile result
    comparison = ComparisonResult(
        results=results,
        comparison_table=comparison_table,
        rankings=rankings,
        metadata={
            "n_spots": len(spatial),
            "n_cells": len(snrna),
            "n_genes": len(snrna.var_names),
            "required_backends": required_backends,
        },
    )

    # Save results
    comparison.save(output_dir)

    return comparison


def build_comparison_table(
    results: dict[str, BackendRunResult],
) -> pd.DataFrame:
    """
    Build comparison DataFrame from backend results.

    Args:
        results: Dictionary of backend results

    Returns:
        DataFrame with one row per backend and metrics as columns
    """
    rows = []

    for name, result in results.items():
        row = {
            "backend": name,
            "success": result.success,
            "runtime_seconds": result.runtime_seconds,
        }

        if result.metrics:
            row.update(result.metrics.to_dict())

        if result.error:
            row["error"] = result.error[:100]  # Truncate

        rows.append(row)

    df = pd.DataFrame(rows)

    # Reorder columns
    priority_cols = ["backend", "success", "runtime_seconds"]
    other_cols = [c for c in df.columns if c not in priority_cols]
    df = df[priority_cols + sorted(other_cols)]

    return df


def rank_backends(
    comparison_table: pd.DataFrame,
    weights: dict[str, float] | None = None,
) -> dict[str, list[str]]:
    """
    Rank backends by different criteria.

    Args:
        comparison_table: Comparison DataFrame
        weights: Optional weights for overall ranking

    Returns:
        Dictionary mapping criterion to ranked list of backends
    """
    if weights is None:
        weights = {
            "upstream": 0.3,
            "downstream": 0.4,
            "spatial": 0.2,
            "runtime": 0.1,
        }

    # Only rank successful backends
    df = comparison_table[comparison_table["success"]].copy()

    if len(df) == 0:
        return {"overall": [], "upstream": [], "downstream": [], "spatial": []}

    rankings = {}

    # Upstream quality ranking
    upstream_cols = [c for c in df.columns if c.startswith("upstream_")]
    if upstream_cols:
        # Higher is better for most upstream metrics
        df["upstream_score"] = df[upstream_cols].mean(axis=1)
        rankings["upstream"] = df.sort_values("upstream_score", ascending=False)[
            "backend"
        ].tolist()
    else:
        rankings["upstream"] = df["backend"].tolist()

    # Downstream utility ranking
    downstream_cols = [c for c in df.columns if c.startswith("downstream_")]
    if downstream_cols:
        df["downstream_score"] = df[downstream_cols].mean(axis=1)
        rankings["downstream"] = df.sort_values("downstream_score", ascending=False)[
            "backend"
        ].tolist()
    else:
        rankings["downstream"] = df["backend"].tolist()

    # Spatial coherence ranking
    spatial_cols = [c for c in df.columns if c.startswith("spatial_")]
    if spatial_cols:
        df["spatial_score"] = df[spatial_cols].mean(axis=1)
        rankings["spatial"] = df.sort_values("spatial_score", ascending=False)["backend"].tolist()
    else:
        rankings["spatial"] = df["backend"].tolist()

    # Runtime ranking (lower is better)
    rankings["runtime"] = df.sort_values("runtime_seconds", ascending=True)["backend"].tolist()

    # Overall weighted ranking
    score_cols = ["upstream_score", "downstream_score", "spatial_score"]
    available_scores = [c for c in score_cols if c in df.columns]

    if available_scores:
        # Normalize runtime to [0, 1] (inverted: faster = higher)
        max_runtime = df["runtime_seconds"].max()
        if max_runtime > 0:
            df["runtime_score"] = 1 - (df["runtime_seconds"] / max_runtime)
        else:
            df["runtime_score"] = 1.0

        # Compute weighted overall score
        df["overall_score"] = 0.0
        for score_name, weight in weights.items():
            col = f"{score_name}_score"
            if col in df.columns:
                df["overall_score"] += weight * df[col].fillna(0.5)

        rankings["overall"] = df.sort_values("overall_score", ascending=False)["backend"].tolist()
    else:
        rankings["overall"] = df["backend"].tolist()

    return rankings


def run_donor_comparison(
    backends: dict[str, SpatialBackend],
    snrna_by_donor: dict[str, ad.AnnData],
    spatial_by_donor: dict[str, ad.AnnData],
    output_dir: Path,
) -> dict[str, dict[str, float]]:
    """
    Run backends on multiple donors and compute robustness metrics.

    Args:
        backends: Dictionary of backend instances
        snrna_by_donor: Dictionary mapping donor ID to snRNA data
        spatial_by_donor: Dictionary mapping donor ID to spatial data
        output_dir: Output directory

    Returns:
        Dictionary mapping backend name to robustness metrics
    """
    output_dir = Path(output_dir)

    robustness_by_backend = {}

    for backend_name, backend in backends.items():
        print(f"\nRunning {backend_name} across donors...")

        results_by_donor = {}

        for donor_id in snrna_by_donor.keys():
            if donor_id not in spatial_by_donor:
                continue

            donor_dir = output_dir / backend_name.lower() / f"donor_{donor_id}"

            try:
                result = backend.map(
                    snrna_by_donor[donor_id],
                    spatial_by_donor[donor_id],
                    output_dir=donor_dir,
                )
                results_by_donor[donor_id] = result
            except Exception as e:
                print(f"  Donor {donor_id} failed: {e}")

        # Compute robustness
        if len(results_by_donor) >= 2:
            robustness = compute_donor_robustness(results_by_donor)
            robustness_by_backend[backend_name] = robustness
        else:
            robustness_by_backend[backend_name] = {
                "donor_consistency": np.nan,
                "celltype_stability": np.nan,
                "n_donors": len(results_by_donor),
            }

    return robustness_by_backend
