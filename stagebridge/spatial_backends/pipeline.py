"""
Main benchmark pipeline for spatial backend comparison.

Provides end-to-end pipeline for running, comparing, and selecting spatial backends.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
import json
import time
import numpy as np
import pandas as pd
import anndata as ad

from .base import SpatialBackend, BackendMappingResult
from .tangram_wrapper import TangramBackend
from .destvi_wrapper import DestVIBackend
from .tacco_wrapper import TACCOBackend
from .comparison import (
    ComparisonResult,
    run_backend_comparison,
    run_donor_comparison,
)
from .selection import (
    BackendSelection,
    select_canonical_backend,
    generate_selection_report,
    save_canonical_decision,
)
from .visualize import (
    plot_spatial_maps_comparison,
    plot_metrics_comparison,
    plot_confidence_distributions,
    plot_donor_robustness,
    create_comparison_summary_figure,
)
from .standardize import StandardizedOutput


@dataclass
class SpatialBenchmarkConfig:
    """
    Configuration for spatial backend benchmark.

    Controls which backends to run, parameters, and output settings.
    """

    # Backends to run
    backends_to_run: list[str] = field(default_factory=lambda: ["tangram", "destvi", "tacco"])

    # Required backends (fail if any fail)
    required_backends: list[str] = field(default_factory=lambda: ["tangram", "destvi", "tacco"])

    # Backend-specific configurations
    tangram_config: dict[str, Any] = field(default_factory=dict)
    destvi_config: dict[str, Any] = field(default_factory=dict)
    tacco_config: dict[str, Any] = field(default_factory=dict)

    # Selection weights
    selection_weights: dict[str, float] = field(
        default_factory=lambda: {
            "upstream": 0.25,
            "downstream": 0.40,
            "spatial": 0.20,
            "robustness": 0.10,
            "runtime": 0.05,
        }
    )

    # Smoke mode (reduced computation)
    smoke_mode: bool = False
    smoke_n_spots: int = 500
    smoke_n_cells: int = 2000
    smoke_n_epochs: int = 50

    # Output settings
    save_plots: bool = True
    save_intermediate: bool = True

    # Random seed
    random_seed: int = 42

    def get_backend_config(self, backend_name: str) -> dict[str, Any]:
        """Get configuration for a specific backend."""
        configs = {
            "tangram": self.tangram_config,
            "destvi": self.destvi_config,
            "tacco": self.tacco_config,
        }

        config = configs.get(backend_name.lower(), {}).copy()

        # Apply smoke mode modifications
        if self.smoke_mode:
            if backend_name.lower() == "tangram":
                config.setdefault("n_epochs", self.smoke_n_epochs)
            elif backend_name.lower() == "destvi":
                config.setdefault("n_epochs_condsc", self.smoke_n_epochs)
                config.setdefault("n_epochs_destvi", self.smoke_n_epochs * 5)

        return config


@dataclass
class BenchmarkProgress:
    """Tracks progress of benchmark execution."""

    total_backends: int = 0
    completed_backends: int = 0
    current_backend: str | None = None
    status: str = "not_started"
    errors: list[str] = field(default_factory=list)

    def update(
        self,
        backend: str | None = None,
        status: str | None = None,
        error: str | None = None,
    ):
        """Update progress state."""
        if backend:
            self.current_backend = backend
        if status:
            self.status = status
        if error:
            self.errors.append(error)

    def backend_complete(self, backend: str, success: bool):
        """Mark a backend as complete."""
        self.completed_backends += 1
        if not success:
            self.errors.append(f"{backend} failed")


def run_spatial_benchmark(
    config: SpatialBenchmarkConfig,
    snrna: ad.AnnData,
    spatial: ad.AnnData,
    output_dir: Path,
    transition_data: dict[str, Any] | None = None,
    progress_callback: Callable[[BenchmarkProgress], None] | None = None,
) -> tuple[ComparisonResult, BackendSelection]:
    """
    Run complete spatial backend benchmark pipeline.

    This is the main entry point for the benchmark. It:
    1. Initializes all backends
    2. Runs each backend on the data
    3. Computes metrics and comparisons
    4. Selects canonical backend
    5. Generates reports and visualizations

    Args:
        config: Benchmark configuration
        snrna: Single-cell reference data
        spatial: Spatial transcriptomics data
        output_dir: Output directory for results
        transition_data: Optional transition data for downstream metrics
        progress_callback: Optional callback for progress updates

    Returns:
        Tuple of (ComparisonResult, BackendSelection)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize progress tracking
    progress = BenchmarkProgress(
        total_backends=len(config.backends_to_run),
        status="initializing",
    )

    if progress_callback:
        progress_callback(progress)

    # Apply smoke mode if needed
    if config.smoke_mode:
        snrna, spatial = _apply_smoke_mode(
            snrna,
            spatial,
            n_cells=config.smoke_n_cells,
            n_spots=config.smoke_n_spots,
            seed=config.random_seed,
        )
        print(f"Smoke mode: {len(snrna)} cells, {len(spatial)} spots")

    # Save benchmark config
    _save_config(config, output_dir)

    # Initialize backends
    progress.update(status="initializing_backends")
    if progress_callback:
        progress_callback(progress)

    backends = _initialize_backends(config)

    # Get spatial coordinates
    spatial_coords = None
    if "spatial" in spatial.obsm:
        spatial_coords = spatial.obsm["spatial"]

    # Run comparison
    progress.update(status="running_backends")
    if progress_callback:
        progress_callback(progress)

    comparison = run_backend_comparison(
        backends=backends,
        snrna=snrna,
        spatial=spatial,
        output_dir=output_dir,
        spatial_coords=spatial_coords,
        transition_data=transition_data,
        required_backends=config.required_backends,
    )

    # Update progress for each backend
    for name, result in comparison.results.items():
        progress.backend_complete(name, result.success)
        if progress_callback:
            progress_callback(progress)

    # Select canonical backend
    progress.update(status="selecting_canonical")
    if progress_callback:
        progress_callback(progress)

    try:
        selection = select_canonical_backend(
            comparison,
            weights=config.selection_weights,
        )
    except ValueError as e:
        # No successful backends
        progress.update(status="failed", error=str(e))
        if progress_callback:
            progress_callback(progress)
        raise

    # Generate report
    generate_selection_report(
        comparison,
        selection,
        output_path=output_dir / "backend_selection_report.md",
    )

    # Save canonical decision
    save_canonical_decision(selection, output_dir)

    # Generate visualizations
    if config.save_plots:
        progress.update(status="generating_plots")
        if progress_callback:
            progress_callback(progress)

        _generate_benchmark_plots(
            comparison=comparison,
            selection=selection,
            spatial_coords=spatial_coords,
            output_dir=output_dir,
        )

    progress.update(status="completed")
    if progress_callback:
        progress_callback(progress)

    return comparison, selection


def _apply_smoke_mode(
    snrna: ad.AnnData,
    spatial: ad.AnnData,
    n_cells: int,
    n_spots: int,
    seed: int,
) -> tuple[ad.AnnData, ad.AnnData]:
    """Subsample data for smoke mode."""
    np.random.seed(seed)

    # Subsample cells
    if len(snrna) > n_cells:
        cell_idx = np.random.choice(len(snrna), n_cells, replace=False)
        snrna = snrna[cell_idx].copy()

    # Subsample spots
    if len(spatial) > n_spots:
        spot_idx = np.random.choice(len(spatial), n_spots, replace=False)
        spatial = spatial[spot_idx].copy()

    return snrna, spatial


def _save_config(config: SpatialBenchmarkConfig, output_dir: Path) -> None:
    """Save benchmark configuration to JSON."""
    config_dict = {
        "backends_to_run": config.backends_to_run,
        "required_backends": config.required_backends,
        "tangram_config": config.tangram_config,
        "destvi_config": config.destvi_config,
        "tacco_config": config.tacco_config,
        "selection_weights": config.selection_weights,
        "smoke_mode": config.smoke_mode,
        "random_seed": config.random_seed,
    }

    with open(output_dir / "benchmark_config.json", "w") as f:
        json.dump(config_dict, f, indent=2)


def _initialize_backends(
    config: SpatialBenchmarkConfig,
) -> dict[str, SpatialBackend]:
    """Initialize all requested backends."""
    backends = {}

    backend_classes = {
        "tangram": TangramBackend,
        "destvi": DestVIBackend,
        "tacco": TACCOBackend,
    }

    for name in config.backends_to_run:
        name_lower = name.lower()
        if name_lower in backend_classes:
            backend_config = config.get_backend_config(name_lower)
            backends[name] = backend_classes[name_lower](**backend_config)
        else:
            print(f"Warning: Unknown backend '{name}', skipping")

    return backends


def _generate_benchmark_plots(
    comparison: ComparisonResult,
    selection: BackendSelection,
    spatial_coords: np.ndarray | None,
    output_dir: Path,
) -> None:
    """Generate all benchmark visualization plots."""
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Collect successful results
    results = {}
    for name, result in comparison.results.items():
        if result.success and result.standardized:
            results[name] = result.standardized

    if not results:
        print("No successful results to plot")
        return

    # Plot 1: Spatial maps comparison
    if spatial_coords is not None:
        try:
            plot_spatial_maps_comparison(
                results=results,
                spatial_coords=spatial_coords,
                output_path=plots_dir / "spatial_maps_comparison.png",
            )
        except Exception as e:
            print(f"Warning: Failed to generate spatial maps: {e}")

    # Plot 2: Metrics comparison
    if comparison.comparison_table is not None:
        try:
            plot_metrics_comparison(
                comparison_table=comparison.comparison_table,
                output_path=plots_dir / "metrics_comparison.png",
            )
        except Exception as e:
            print(f"Warning: Failed to generate metrics comparison: {e}")

    # Plot 3: Confidence distributions
    try:
        plot_confidence_distributions(
            results=results,
            output_path=plots_dir / "confidence_distributions.png",
        )
    except Exception as e:
        print(f"Warning: Failed to generate confidence distributions: {e}")

    # Plot 4: Summary figure
    if spatial_coords is not None:
        try:
            create_comparison_summary_figure(
                comparison_result=comparison,
                results=results,
                spatial_coords=spatial_coords,
                output_path=plots_dir / "comparison_summary.png",
            )
        except Exception as e:
            print(f"Warning: Failed to generate summary figure: {e}")


def run_smoke_benchmark(
    snrna: ad.AnnData,
    spatial: ad.AnnData,
    output_dir: Path,
) -> tuple[ComparisonResult, BackendSelection]:
    """
    Run a quick smoke test benchmark with reduced parameters.

    Useful for testing the pipeline and validating schema.

    Args:
        snrna: Single-cell reference data
        spatial: Spatial transcriptomics data
        output_dir: Output directory

    Returns:
        Tuple of (ComparisonResult, BackendSelection)
    """
    config = SpatialBenchmarkConfig(
        smoke_mode=True,
        smoke_n_spots=200,
        smoke_n_cells=500,
        smoke_n_epochs=10,
        save_plots=True,
    )

    return run_spatial_benchmark(
        config=config,
        snrna=snrna,
        spatial=spatial,
        output_dir=output_dir,
    )


def load_benchmark_results(
    output_dir: Path,
) -> tuple[ComparisonResult, BackendSelection]:
    """
    Load previously saved benchmark results.

    Args:
        output_dir: Directory containing benchmark outputs

    Returns:
        Tuple of (ComparisonResult, BackendSelection)
    """
    from .selection import load_canonical_decision

    comparison = ComparisonResult.load(output_dir)
    selection = load_canonical_decision(output_dir)

    return comparison, selection


def get_canonical_backend_result(
    output_dir: Path,
) -> StandardizedOutput:
    """
    Load the canonical backend's standardized output.

    Args:
        output_dir: Directory containing benchmark outputs

    Returns:
        StandardizedOutput for canonical backend
    """
    from .selection import load_canonical_decision
    from .standardize import StandardizedOutput

    selection = load_canonical_decision(output_dir)
    canonical_dir = output_dir / selection.canonical_backend.lower()

    return StandardizedOutput.load(canonical_dir)
