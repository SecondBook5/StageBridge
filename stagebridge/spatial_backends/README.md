# Spatial Deconvolution Backends

## spatial_backends/ vs spatial_mapping/

This project has **two** spatial directories with different purposes:

### `stagebridge/spatial_backends/` (this directory)
**Purpose:** Unified wrapper system for benchmarking multiple deconvolution methods

- `base.py` - Abstract base class for all backends
- `tangram_wrapper.py` - Tangram integration
- `destvi_wrapper.py` - DestVI integration
- `tacco_wrapper.py` - TACCO integration
- `cell2location_wrapper.py` - Cell2Location integration
- `pipeline.py` - Orchestration for running all backends
- `metrics.py` - Comparison metrics (cosine, correlation, etc.)
- `visualize.py` - Side-by-side comparison plots

**Usage:**
```python
from stagebridge.spatial_backends import TangramBackend, run_backend_comparison
```

### `stagebridge/spatial_mapping/`
**Purpose:** Production spatial mapping implementations (used after backend selection)

- Core implementations of spatial mapping algorithms
- Used by the main training pipeline
- Optimized for production use

## Summary

| Module | Purpose | When to use |
|--------|---------|-------------|
| `spatial_backends/` | Benchmarking & comparison | Evaluating which method to use |
| `spatial_mapping/` | Production implementation | After method is selected |

## Canonical Backend

After benchmarking, **Tangram** was selected as the canonical backend for V1.
