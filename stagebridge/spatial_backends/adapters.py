"""Adapters that bridge spatial_mapping implementations to the benchmark interface.

This module provides adapter classes that wrap the existing spatial_mapping
implementations (tangram_mapper, destvi_mapper, tacco_mapper) to conform to
the SpatialBackend interface used by the benchmark infrastructure.

The adapters:
1. Call the existing production implementations
2. Convert SpatialMappingResult to BackendMappingResult
3. Provide consistent interface for benchmarking
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from stagebridge.spatial_mapping.base import SpatialMappingResult
from .base import SpatialBackend, BackendMappingResult, compute_cell_type_entropy


@dataclass
class AdapterConfig:
    """Configuration for spatial backend adapters."""

    # Execution mode: 'load_precomputed', 'rebuild_cached', 'force_rebuild'
    execution_mode: str = "force_rebuild"

    # Stage filtering
    stages: list[str] | None = None
    donors: list[str] | None = None
    max_spots_per_stage: int | None = None

    # Random seed
    seed: int = 42

    # Additional backend-specific config
    extra: dict[str, Any] | None = None


def _convert_to_backend_result(
    mapping_result: SpatialMappingResult,
    runtime_seconds: float = 0.0,
) -> BackendMappingResult:
    """Convert SpatialMappingResult to BackendMappingResult.

    Parameters
    ----------
    mapping_result : SpatialMappingResult
        Result from spatial_mapping implementation
    runtime_seconds : float
        Execution time

    Returns
    -------
    BackendMappingResult
        Standardized result for benchmarking
    """
    # Extract cell type proportions as DataFrame
    if mapping_result.compositions is not None and mapping_result.obs is not None:
        cell_type_proportions = pd.DataFrame(
            mapping_result.compositions,
            index=mapping_result.obs.index,
            columns=list(mapping_result.feature_names),
        )
    else:
        # Empty result
        cell_type_proportions = pd.DataFrame()

    # Compute confidence from entropy (low entropy = high confidence)
    if not cell_type_proportions.empty:
        entropy = compute_cell_type_entropy(cell_type_proportions)
        confidence = 1.0 - entropy
    else:
        confidence = pd.Series(dtype=float)

    # Extract upstream metrics from QC
    upstream_metrics: dict[str, float] = {}
    if mapping_result.qc:
        for key, value in mapping_result.qc.items():
            if isinstance(value, (int, float)):
                upstream_metrics[key] = float(value)

    # Add standard metrics
    if not cell_type_proportions.empty:
        upstream_metrics["n_spots"] = len(cell_type_proportions)
        upstream_metrics["n_celltypes"] = cell_type_proportions.shape[1]
        upstream_metrics["mean_entropy"] = float(
            compute_cell_type_entropy(cell_type_proportions).mean()
        )
        upstream_metrics["coverage"] = (
            float((confidence > 0.5).mean()) if len(confidence) > 0 else 0.0
        )

    # Build metadata
    metadata: dict[str, Any] = {
        "backend": mapping_result.method,
        "status": mapping_result.status,
        "provider_version": mapping_result.provider_version,
        "execution_mode": mapping_result.execution_mode,
        "runtime_seconds": runtime_seconds,
    }
    if mapping_result.provenance:
        metadata["provenance"] = mapping_result.provenance
    if mapping_result.notes:
        metadata["notes"] = mapping_result.notes

    return BackendMappingResult(
        cell_type_proportions=cell_type_proportions,
        confidence=confidence,
        upstream_metrics=upstream_metrics,
        metadata=metadata,
    )


class TangramAdapter(SpatialBackend):
    """Adapter wrapping the existing Tangram implementation.

    This adapter calls stagebridge.spatial_mapping.tangram_mapper.run_tangram()
    and converts the result to BackendMappingResult for benchmarking.
    """

    def __init__(self, config: AdapterConfig | None = None, **kwargs):
        """Initialize Tangram adapter.

        Parameters
        ----------
        config : AdapterConfig, optional
            Adapter configuration
        **kwargs
            Additional config passed to parent
        """
        super().__init__(**kwargs)
        self.adapter_config = config or AdapterConfig()

    def map(
        self,
        snrna: Any,
        spatial: Any,
        output_dir: Path | None = None,
    ) -> BackendMappingResult:
        """Run Tangram mapping using the existing implementation.

        Note: This adapter expects a config dict in self.config that can be
        passed to run_tangram(). For direct AnnData inputs, use the
        TangramBackend class instead.
        """
        from stagebridge.spatial_mapping.tangram_mapper import run_tangram

        # Build config for run_tangram
        cfg = self._build_cfg()

        start_time = time.time()
        result = run_tangram(
            cfg,
            stages=self.adapter_config.stages,
            donors=self.adapter_config.donors,
            max_spots_per_stage=self.adapter_config.max_spots_per_stage,
            seed=self.adapter_config.seed,
        )
        runtime = time.time() - start_time

        backend_result = _convert_to_backend_result(result, runtime)

        if output_dir:
            backend_result.save(output_dir)

        return backend_result

    def _build_cfg(self) -> dict[str, Any]:
        """Build configuration dict for run_tangram."""
        cfg: dict[str, Any] = dict(self.config)
        cfg.setdefault("spatial_mapping", {})
        cfg["spatial_mapping"]["method"] = "tangram"
        cfg["spatial_mapping"]["execution_mode"] = self.adapter_config.execution_mode
        if self.adapter_config.extra:
            cfg["spatial_mapping"].update(self.adapter_config.extra)
        return cfg

    def compute_upstream_metrics(self, snrna, spatial, result) -> dict[str, float]:
        """Return metrics from result (already computed during mapping)."""
        return result.upstream_metrics if result else {}

    def estimate_confidence(self, snrna, spatial, result) -> pd.Series:
        """Return confidence from result (already computed during mapping)."""
        return result.confidence if result else pd.Series(dtype=float)


class DestVIAdapter(SpatialBackend):
    """Adapter wrapping the existing DestVI implementation."""

    def __init__(self, config: AdapterConfig | None = None, **kwargs):
        super().__init__(**kwargs)
        self.adapter_config = config or AdapterConfig()

    def map(
        self,
        snrna: Any,
        spatial: Any,
        output_dir: Path | None = None,
    ) -> BackendMappingResult:
        """Run DestVI mapping using the existing implementation."""
        from stagebridge.spatial_mapping.destvi_mapper import run_destvi

        cfg = self._build_cfg()

        start_time = time.time()
        result = run_destvi(
            cfg,
            stages=self.adapter_config.stages,
            donors=self.adapter_config.donors,
            max_spots_per_stage=self.adapter_config.max_spots_per_stage,
            seed=self.adapter_config.seed,
        )
        runtime = time.time() - start_time

        backend_result = _convert_to_backend_result(result, runtime)

        if output_dir:
            backend_result.save(output_dir)

        return backend_result

    def _build_cfg(self) -> dict[str, Any]:
        cfg: dict[str, Any] = dict(self.config)
        cfg.setdefault("spatial_mapping", {})
        cfg["spatial_mapping"]["method"] = "destvi"
        cfg["spatial_mapping"]["execution_mode"] = self.adapter_config.execution_mode
        if self.adapter_config.extra:
            cfg["spatial_mapping"].update(self.adapter_config.extra)
        return cfg

    def compute_upstream_metrics(self, snrna, spatial, result) -> dict[str, float]:
        return result.upstream_metrics if result else {}

    def estimate_confidence(self, snrna, spatial, result) -> pd.Series:
        return result.confidence if result else pd.Series(dtype=float)


class TACCOAdapter(SpatialBackend):
    """Adapter wrapping the existing TACCO implementation."""

    def __init__(self, config: AdapterConfig | None = None, **kwargs):
        super().__init__(**kwargs)
        self.adapter_config = config or AdapterConfig()

    def map(
        self,
        snrna: Any,
        spatial: Any,
        output_dir: Path | None = None,
    ) -> BackendMappingResult:
        """Run TACCO mapping using the existing implementation."""
        from stagebridge.spatial_mapping.tacco_mapper import run_tacco

        cfg = self._build_cfg()

        start_time = time.time()
        result = run_tacco(
            cfg,
            stages=self.adapter_config.stages,
            donors=self.adapter_config.donors,
            max_spots_per_stage=self.adapter_config.max_spots_per_stage,
            seed=self.adapter_config.seed,
        )
        runtime = time.time() - start_time

        backend_result = _convert_to_backend_result(result, runtime)

        if output_dir:
            backend_result.save(output_dir)

        return backend_result

    def _build_cfg(self) -> dict[str, Any]:
        cfg: dict[str, Any] = dict(self.config)
        cfg.setdefault("spatial_mapping", {})
        cfg["spatial_mapping"]["method"] = "tacco"
        cfg["spatial_mapping"]["execution_mode"] = self.adapter_config.execution_mode
        if self.adapter_config.extra:
            cfg["spatial_mapping"].update(self.adapter_config.extra)
        return cfg

    def compute_upstream_metrics(self, snrna, spatial, result) -> dict[str, float]:
        return result.upstream_metrics if result else {}

    def estimate_confidence(self, snrna, spatial, result) -> pd.Series:
        return result.confidence if result else pd.Series(dtype=float)


# Registry of adapters
ADAPTERS: dict[str, type[SpatialBackend]] = {
    "tangram": TangramAdapter,
    "destvi": DestVIAdapter,
    "tacco": TACCOAdapter,
}


def get_adapter(
    method: str,
    config: AdapterConfig | None = None,
    **kwargs,
) -> SpatialBackend:
    """Get a spatial backend adapter by method name.

    Parameters
    ----------
    method : str
        Backend method name: 'tangram', 'destvi', 'tacco'
    config : AdapterConfig, optional
        Adapter configuration
    **kwargs
        Passed to adapter constructor

    Returns
    -------
    SpatialBackend
        Configured adapter instance
    """
    method_lower = method.lower()
    if method_lower not in ADAPTERS:
        available = ", ".join(sorted(ADAPTERS.keys()))
        raise ValueError(f"Unknown backend '{method}'. Available: {available}")

    return ADAPTERS[method_lower](config=config, **kwargs)
