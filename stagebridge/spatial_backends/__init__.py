"""
Spatial transcriptomics mapping backend wrappers.

Provides unified interface for multiple spatial mapping methods:
- Tangram: Marker-based mapping with gradient-based optimization
- DestVI: VAE-based probabilistic mapping with amortized inference
- TACCO: Compositional transfer with optimal transport

Two interface modes are available:

**Direct backends** (TangramBackend, DestVIBackend, TACCOBackend):
    Take AnnData objects directly. Suitable for benchmarking and testing.

**Adapters** (TangramAdapter, DestVIAdapter, TACCOAdapter):
    Wrap the production implementations in stagebridge.spatial_mapping.
    Use config-driven execution with caching and execution modes.

Benchmark infrastructure:
- metrics: Upstream and downstream evaluation metrics
- comparison: Backend comparison logic
- selection: Canonical backend selection with justification
- visualize: Comparison visualizations
- pipeline: End-to-end benchmark pipeline
- standardize: Output standardization
"""

from .base import SpatialBackend, BackendMappingResult

# Direct AnnData backends (for benchmarking)
from .tangram_wrapper import TangramBackend
from .destvi_wrapper import DestVIBackend
from .tacco_wrapper import TACCOBackend

# Adapters wrapping spatial_mapping implementations (for production pipelines)
from .adapters import (
    AdapterConfig,
    TangramAdapter,
    DestVIAdapter,
    TACCOAdapter,
    get_adapter,
)

# Benchmark infrastructure
from .metrics import (
    MetricsReport,
    compute_upstream_metrics,
    compute_downstream_utility,
    compute_spatial_coherence,
    compute_donor_robustness,
    compute_comprehensive_metrics,
)
from .comparison import (
    BackendRunResult,
    ComparisonResult,
    run_backend_comparison,
    run_single_backend,
    build_comparison_table,
    rank_backends,
)
from .selection import (
    BackendSelection,
    select_canonical_backend,
    generate_selection_report,
    save_canonical_decision,
    load_canonical_decision,
)
from .standardize import (
    StandardizedOutput,
    standardize_backend_output,
    validate_standardized_output,
)
from .pipeline import (
    SpatialBenchmarkConfig,
    BenchmarkProgress,
    run_spatial_benchmark,
    run_smoke_benchmark,
    load_benchmark_results,
    get_canonical_backend_result,
)

__all__ = [
    # Base classes
    "SpatialBackend",
    "BackendMappingResult",
    # Direct backends (AnnData interface)
    "TangramBackend",
    "DestVIBackend",
    "TACCOBackend",
    # Adapters (wrap spatial_mapping implementations)
    "AdapterConfig",
    "TangramAdapter",
    "DestVIAdapter",
    "TACCOAdapter",
    "get_adapter",
    # Metrics
    "MetricsReport",
    "compute_upstream_metrics",
    "compute_downstream_utility",
    "compute_spatial_coherence",
    "compute_donor_robustness",
    "compute_comprehensive_metrics",
    # Comparison
    "BackendRunResult",
    "ComparisonResult",
    "run_backend_comparison",
    "run_single_backend",
    "build_comparison_table",
    "rank_backends",
    # Selection
    "BackendSelection",
    "select_canonical_backend",
    "generate_selection_report",
    "save_canonical_decision",
    "load_canonical_decision",
    # Standardization
    "StandardizedOutput",
    "standardize_backend_output",
    "validate_standardized_output",
    # Pipeline
    "SpatialBenchmarkConfig",
    "BenchmarkProgress",
    "run_spatial_benchmark",
    "run_smoke_benchmark",
    "load_benchmark_results",
    "get_canonical_backend_result",
    # Factory functions
    "get_backend",
]


def get_backend(name: str, use_adapter: bool = False) -> type[SpatialBackend]:
    """
    Get spatial mapping backend by name.

    Args:
        name: Backend name ('tangram', 'destvi', or 'tacco')
        use_adapter: If True, return adapter wrapping spatial_mapping implementation.
                     If False (default), return direct AnnData backend.

    Returns:
        Backend class
    """
    direct_backends = {
        "tangram": TangramBackend,
        "destvi": DestVIBackend,
        "tacco": TACCOBackend,
    }

    adapter_backends = {
        "tangram": TangramAdapter,
        "destvi": DestVIAdapter,
        "tacco": TACCOAdapter,
    }

    backends = adapter_backends if use_adapter else direct_backends

    if name.lower() not in backends:
        raise ValueError(f"Unknown backend: {name}. Available: {list(backends.keys())}")

    return backends[name.lower()]
