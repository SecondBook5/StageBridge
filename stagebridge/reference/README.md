# StageBridge Reference Module

Dual-reference geometry layer for mapping query cells to HLCA (healthy) and LuCA (disease-aware) reference spaces.

## Reference Sources

### HLCA (Human Lung Cell Atlas)

- **Source**: CZI cellxgene via scvi-tools Hugging Face Hub
- **Repository**: `scvi-tools/human-lung-cell-atlas-scanvi`
- **Download**: Via `scvi.hub.HubModel.pull_from_huggingface_hub()`
- **Cells**: ~584K healthy lung cells
- **Latent key**: `X_scanvi_emb` (30 dimensions)
- **Required columns**: `ann_level_1`, `ann_level_2`, `ann_level_3`

### LuCA (Lung Cancer Atlas)

- **Source**: Zenodo / LungCancerAtlas GitHub
- **Documentation**: https://github.com/LungCancerAtlas/
- **Latent key**: `X_scVI` (10 dimensions)
- **Required columns**: `cell_type`

**CRITICAL: LuCA Core vs Extended**

LuCA has two versions:
- **LuCA Core** (~790K cells): Complete latents, 100% valid - USE THIS
- **LuCA Extended** (~1.3M cells): 31% NaN in latents - DO NOT USE

Always verify latent integrity before mapping:

```bash
python -m stagebridge.reference.diagnose_reference /path/to/luca.h5ad --latent-key X_scVI --diagnose-only
```

Expected output for usable reference:
```
=== Latent Integrity Report ===
  Total cells: 790,000
  Latent dim: 10
  Valid cells: 790,000 (100.0%)
  Cells with any NaN: 0
  Recommendation: usable
```

## Module Structure

```
stagebridge/reference/
├── __init__.py          # Public API exports
├── README.md            # This file
├── loaders.py           # Reference loading with validation
├── map_query.py         # Query-to-reference mapping (in-memory)
├── map_query_chunked.py # Memory-efficient chunked mapping (HPC/large refs)
├── fuse.py              # Dual-reference fusion strategies
├── confidence.py        # Confidence scoring (percentile rank calibration)
├── schema.py            # Output schemas and export
├── pipeline.py          # High-level pipeline interface
├── diagnose_reference.py # Latent integrity checking tool
├── diagnostics.py       # Latent space diagnostics (stage preservation, etc.)
├── hlca_mapper.py       # Legacy HLCA-specific mapper (scvi/scArches)
├── label_transfer.py    # Reference label transfer
├── latent_store.py      # Latent storage utilities
├── prepare.py           # Reference preparation helpers
└── visualize.py         # Reference visualization
```

### Core Modules (Active in V1)

| Module | Purpose | Used By |
|--------|---------|---------|
| `loaders.py` | Load/validate HLCA and LuCA references | `pipeline.py`, `run_reference.py` |
| `map_query.py` | k-NN projection in gene space | `pipeline.py` |
| `map_query_chunked.py` | Memory-efficient mapping for HPC | `run_reference.py --hpc` |
| `fuse.py` | Concatenate/fuse dual embeddings | `pipeline.py` |
| `confidence.py` | Percentile-rank confidence calibration | `pipeline.py`, `run_reference.py` |
| `schema.py` | Output schema and export | `pipeline.py` |
| `diagnose_reference.py` | Check for NaN in reference latents | CLI tool |

### Support Modules

| Module | Purpose | Notes |
|--------|---------|-------|
| `hlca_mapper.py` | Legacy HLCA mapper with scvi/scArches | Used by `test_reference_branch.py` |
| `diagnostics.py` | Stage preservation, donor leakage checks | Used by `hlca_mapper.py` |
| `latent_store.py` | Latent storage utilities | Used by `hlca_mapper.py` |
| `label_transfer.py` | Transfer reference labels to query | Used by `hlca_mapper.py` |
| `prepare.py` | Reference preparation | Not currently imported |
| `visualize.py` | Visualization helpers | Not currently imported |

## Key Design Decisions

### 1. Percentile-Rank Confidence Calibration

**Problem**: LuCA (~1.3M cells) is denser than HLCA (~584K cells). Raw k-NN distances are systematically smaller for LuCA, biasing confidence scores.

**Solution**: Convert distances to percentile ranks before computing confidence:

```python
from scipy.stats import rankdata

ranks = rankdata(distances)
confidence = 1.0 - (ranks - 1) / (len(ranks) - 1)  # Lower distance = higher rank = higher confidence
```

This ensures confidence is comparable across references regardless of density.

### 2. PCA Reduction for High-Dimensional Gene Space

When common genes exceed 2000, the mapping uses PCA reduction:
1. Fit IncrementalPCA on sampled reference cells
2. Transform both reference and query to PCA space
3. Build FAISS index in reduced space

This prevents memory issues and improves k-NN quality.

### 3. NaN Handling

**NEVER zero-fill NaN embeddings.** Instead:
1. Diagnose reference integrity before mapping
2. Filter out cells with NaN latents
3. Report but do not silently fix

Use `diagnose_reference.py` to check integrity:
```bash
python -m stagebridge.reference.diagnose_reference /path/to/ref.h5ad --diagnose-only
```

### 4. Gene Name Handling

References may use ENSG IDs while query uses gene symbols. The mappers automatically use `feature_name` column when available:

```python
# In map_query.py and map_query_chunked.py:
if first_ref_gene.startswith("ENSG") and "feature_name" in ref_adata.var.columns:
    ref_symbols = ref_adata.var["feature_name"].astype(str).tolist()
```

## Output Schema

```
reference_geometry/
├── hlca_embedding.parquet      # cell_id, donor_id, sample_id, stage_id, hlca_latent_0..29
├── luca_embedding.parquet      # cell_id, donor_id, sample_id, stage_id, luca_latent_0..9
├── fused_embedding.parquet     # cell_id + hlca_latent + luca_latent + fused_latent
├── reference_confidence.parquet # cell_id, hlca_confidence, luca_confidence, ..._method, reference_mode_used
├── reference_manifest.json     # Run metadata and parameters
└── feature_overlap_report.json # Gene overlap statistics
```

## Usage

### Pipeline Mode (Recommended)

```bash
python -m stagebridge.pipelines.run_reference \
    --data-root $DATA \
    --hlca $DATA/references/hlca/hlca_core.h5ad \
    --luca $DATA/references/luca/luca_core.h5ad
```

### HPC Mode (Large References)

```bash
python -m stagebridge.pipelines.run_reference \
    --data-root $DATA \
    --hpc \
    --chunk-size 50000
```

### Python API

```python
from stagebridge.reference import (
    load_hlca_reference,
    load_luca_reference,
    run_reference_pipeline,
    ReferenceGeometryConfig,
)

config = ReferenceGeometryConfig(
    hlca_reference_path="/path/to/hlca.h5ad",
    luca_reference_path="/path/to/luca_core.h5ad",
    query_data_path="/path/to/query.h5ad",
    mapping_method="knn_projection",
    k_neighbors=50,
)

result = run_reference_pipeline(config)
print(f"Mapped {result.n_cells} cells to fused {result.fused_dim}-dim space")
```

### Diagnose Reference Tool

```bash
# Check for NaN in latents
python -m stagebridge.reference.diagnose_reference /path/to/ref.h5ad --diagnose-only

# Clean reference (remove cells with NaN latents)
python -m stagebridge.reference.diagnose_reference /path/to/ref.h5ad \
    --output /path/to/ref_cleaned.h5ad
```

## Validation Checklist

Before running reference mapping:

1. [ ] HLCA downloaded via scvi-tools hub
2. [ ] LuCA is CORE version (not extended)
3. [ ] `diagnose_reference.py` shows 100% valid cells
4. [ ] Gene overlap > 30% (check `feature_overlap_report.json`)

## Troubleshooting

### "Reference latent has X% NaN cells"

Use LuCA Core instead of Extended:
```bash
python -m stagebridge.reference.diagnose_reference /path/to/luca.h5ad --diagnose-only
# If NaN > 0%, get the core version
```

### Low gene overlap warning

Check if reference uses ENSG IDs:
```bash
python -c "import anndata; a = anndata.read_h5ad('ref.h5ad', backed='r'); print(a.var_names[:5])"
```

If ENSG IDs, ensure `feature_name` column exists in `var`.

### Memory errors on large references

Use HPC mode with chunking:
```bash
python -m stagebridge.pipelines.run_reference --hpc --chunk-size 30000
```
