# StageBridge Data Model Specification (V1)

**Last Updated:** 2026-03-15
**Status:** V1-Minimal Canonical Schema

---

## 1. Overview

This document defines the canonical data model for StageBridge V1. All dataset-specific preprocessing must map into this generic schema. The data model is designed to be:
- **Cell-centric:** Primary learning unit is the cell
- **Modality-agnostic:** Supports snRNA, spatial, optional genomics
- **Stage-flexible:** Configurable progression graphs
- **Spatially-aware:** First-class support for neighborhood structure
- **Reproducible:** Complete provenance tracking

---

## 2. Core Entities

### 2.1 Cell Token

The fundamental unit of the model.

**Schema: `cells.parquet`**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `cell_id` | string |  | Unique cell identifier |
| `donor_id` | string |  | Donor/patient identifier |
| `lesion_id` | string |  | Lesion/sample identifier |
| `stage` | string |  | Disease stage (e.g., "AIS", "MIA", "invasive") |
| `modality` | string |  | "snrna" or "spatial" |
| `cell_type` | string |  | Annotated cell type |
| `x_coord` | float |  | Spatial X (for spatial modality) |
| `y_coord` | float |  | Spatial Y (for spatial modality) |
| `z_healthy` | array[float] |  | HLCA latent coordinates (dim: 64-128) |
| `z_disease` | array[float] |  | LuCA latent coordinates (dim: 64-128) |
| `z_fused` | array[float] |  | Fused latent coordinates (dim: 128-256) |
| `expr_raw` | array[float] |  | Raw expression (HVGs only, ~2000 genes) |
| `expr_normalized` | array[float] |  | log1p normalized expression |
| `n_counts` | int |  | Total UMI counts |
| `n_genes` | int |  | Number of detected genes |
| `pct_mito` | float |  | Percent mitochondrial |
| `clone_id` | string |  | Clone/lineage identifier (if WES available) |
| `spatial_backend` | string |  | Spatial mapping method ("tangram", "destvi", "tacco") |
| `mapping_confidence` | float |  | Confidence score from spatial mapping |
| `split` | string |  | "train", "val", or "test" |

**Size Estimate:**
- LUAD dataset: ~500K cells × 2KB/cell ≈ 1GB

**Notes:**
- `z_healthy`, `z_disease`, `z_fused` computed by Layer A
- For snRNA cells, spatial coords may be NaN
- For spatial spots, expression is deconvolved/mapped
- `spatial_backend` tracks which mapping method was used

### 2.2 Neighborhood / Niche Object

Spatial context around each receiver cell.

**Schema: `neighborhoods.parquet`**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `niche_id` | string |  | Unique niche identifier (typically = cell_id) |
| `receiver_cell_id` | string |  | Center/receiver cell |
| `neighbor_cell_ids` | list[string] |  | Ordered list of neighbor IDs |
| `neighbor_distances` | list[float] |  | Euclidean distances (μm) |
| `ring_assignments` | list[int] |  | Ring index for each neighbor (0-3) |
| `niche_composition` | dict |  | Cell type counts in neighborhood |
| `niche_diversity` | float |  | Shannon entropy of composition |
| `niche_density` | float |  | Cells per unit area |
| `hlca_similarity_mean` | float |  | Mean HLCA similarity in neighborhood |
| `luca_similarity_mean` | float |  | Mean LuCA similarity in neighborhood |
| `pathway_scores` | dict |  | Ligand-receptor or pathway activities |
| `graph_method` | string |  | "knn" or "radius" |
| `k_neighbors` | int |  | K value (if KNN) |
| `radius_um` | float |  | Radius value (if radius-based) |

**Distance Bins (Default):**
- Ring 0: 0-50 μm
- Ring 1: 50-100 μm
- Ring 2: 100-200 μm
- Ring 3: 200+ μm

**Size Estimate:**
- 500K cells × 200 neighbors/cell × 20 bytes ≈ 2GB

**Notes:**
- Only computed for cells with spatial coordinates
- snRNA cells have no neighborhoods (NaN)
- Neighborhood graphs can be precomputed or built on-the-fly
- Multiple graph construction methods can coexist

### 2.3 Stage-Edge Batch

Training batches for transition learning.

**Schema: `stage_edges.parquet`**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `edge_id` | string |  | "stage_src_to_stage_tgt" |
| `source_stage` | string |  | Source stage name |
| `target_stage` | string |  | Target stage name |
| `source_cell_ids` | list[string] |  | Source cell IDs |
| `target_cell_ids` | list[string] |  | Target cell IDs |
| `n_source_cells` | int |  | Number of source cells |
| `n_target_cells` | int |  | Number of target cells |
| `donor_ids` | list[string] |  | Donors contributing to this edge |
| `lesion_ids` | list[string] |  | Lesions contributing to this edge |
| `edge_weight` | float |  | Edge weight for sampling (e.g., by prevalence) |
| `has_genomics` | bool |  | Whether WES data available |

**LUAD Example Edges:**
- `normal_to_ais`: Normal alveolar → Adenocarcinoma in situ
- `ais_to_mia`: AIS → Minimally invasive adenocarcinoma
- `mia_to_invasive`: MIA → Invasive adenocarcinoma
- `normal_to_invasive`: Normal → Invasive (skip connection)

**Size Estimate:**
- ~10 edges × 1MB/edge ≈ 10MB

**Notes:**
- Edges define the transition graph structure
- Edges can be bidirectional or unidirectional
- Edge weights can balance rare transitions
- Multiple edges can connect the same stage pair (e.g., different cell type transitions)

### 2.4 Split Manifest

Train/validation/test donor assignments.

**Schema: `split_manifest.json`**

```json
{
  "split_strategy": "donor_held_out",
  "n_folds": 5,
  "random_seed": 42,
  "splits": {
    "fold_0": {
      "train_donors": ["D001", "D002", ..., "D012"],
      "val_donors": ["D013", "D014", "D015"],
      "test_donors": ["D016", "D017", "D018"]
    },
    "fold_1": {
      ...
    }
  },
  "donor_metadata": {
    "D001": {
      "age": 65,
      "sex": "M",
      "smoking_status": "former",
      "stage_distribution": {"normal": 1000, "ais": 500, "mia": 200},
      "has_wes": true
    },
    ...
  },
  "stratification_vars": ["stage", "smoking_status"],
  "creation_date": "2026-03-15T10:30:00Z",
  "git_commit": "abc123def"
}
```

**Requirements:**
- Donor-level splits (cells are nested within donors)
- All stages represented in each split
- Balanced stage distribution where possible
- Stratification by key covariates
- Complete provenance tracking

### 2.5 Feature Specification

Standardized feature definitions.

**Schema: `feature_spec.yaml`**

```yaml
version: "1.0"
dataset: "luad_evo"
creation_date: "2026-03-15"

expression:
  modality: "gene_expression"
  normalization: "log1p"
  scaling: "total_1e4"
  n_genes: 2000
  gene_list_path: "hvgs_2000.txt"

latent_space:
  hlca:
    dim: 128
    reference_atlas: "HLCA_v2"
    alignment_method: "scvi"
  luca:
    dim: 128
    reference_atlas: "LuCA_v1"
    alignment_method: "scvi"
  fused:
    dim: 256
    fusion_method: "concat"  # or "learned"

spatial:
  coordinate_units: "micrometers"
  origin: "top_left"
  neighborhood_method: "knn"
  k_neighbors: 100
  distance_bins: [0, 50, 100, 200, 1000]

genomics:
  available: true
  features:
    - tmb: "Tumor mutation burden"
    - signature_sbs1: "Clock-like signature"
    - signature_sbs4: "Smoking signature"
    - clone_id: "Phylogenetic clone assignment"
  source: "wes_features.parquet"

cell_types:
  ontology: "cell_ontology_v2023"
  categories:
    - "AT1"
    - "AT2"
    - "Basal"
    - "Club"
    - "Ciliated"
    - "Neuroendocrine"
    - "Macrophage"
    - "T cell"
    - "B cell"
    - "Endothelial"
    - "Fibroblast"

stages:
  progression_graph:
    nodes:
      - "normal"
      - "ais"
      - "mia"
      - "invasive"
    edges:
      - {source: "normal", target: "ais"}
      - {source: "ais", target: "mia"}
      - {source: "mia", target: "invasive"}
      - {source: "normal", target: "invasive"}  # skip connection
  stage_order: ["normal", "ais", "mia", "invasive"]
```

### 2.6 WES Features

Genomic features per donor/lesion.

**Schema: `wes_features.parquet`**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `sample_id` | string |  | Donor or lesion ID |
| `tmb` | float |  | Tumor mutation burden (mutations/Mb) |
| `signature_sbs1` | float |  | Clock-like signature weight |
| `signature_sbs4` | float |  | Smoking signature weight |
| `signature_sbs13` | float |  | APOBEC signature weight |
| `clone_id` | string |  | Major clone identifier |
| `purity` | float |  | Tumor purity estimate |
| `ploidy` | float |  | Average ploidy |
| `driver_mutations` | list[string] |  | Known driver mutations (e.g., "KRAS_G12C") |
| `cnv_burden` | float |  | Copy number variation burden |

**Size Estimate:**
- ~20 donors × 5 lesions/donor × 500 bytes ≈ 50KB

**Notes:**
- One row per sequenced sample
- Links to cells via `donor_id` or `lesion_id`
- Can be aggregated to donor-level or lesion-level

---

## 3. Spatial Backend Outputs

Each spatial backend produces standardized outputs.

### 3.1 Directory Structure

```
data/processed/<dataset>/spatial_backend/
 tangram/
    cell_type_proportions.parquet
    mapping_confidence.parquet
    gene_imputation.h5ad  # optional
    upstream_metrics.json
    backend_metadata.json
 destvi/
    cell_type_proportions.parquet
    mapping_confidence.parquet
    gene_imputation.h5ad
    upstream_metrics.json
    backend_metadata.json
 tacco/
     cell_type_proportions.parquet
     mapping_confidence.parquet
     upstream_metrics.json
     backend_metadata.json
```

### 3.2 Cell Type Proportions

**Schema: `cell_type_proportions.parquet`**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `spot_id` | string |  | Spatial spot identifier |
| `cell_type` | string |  | Cell type label |
| `proportion` | float |  | Estimated proportion (0-1) |
| `n_cells_est` | float |  | Estimated number of cells |

**Notes:**
- One row per (spot, cell_type) pair
- Proportions sum to 1.0 per spot
- Cell types match `feature_spec.yaml` ontology

### 3.3 Mapping Confidence

**Schema: `mapping_confidence.parquet`**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `spot_id` | string |  | Spatial spot identifier |
| `confidence_score` | float |  | Overall mapping confidence (0-1) |
| `entropy` | float |  | Entropy of proportion distribution |
| `n_cells` | int |  | Number of cells detected |

### 3.4 Backend Metadata

**Schema: `backend_metadata.json`**

```json
{
  "backend_name": "tangram",
  "backend_version": "1.2.0",
  "run_date": "2026-03-15T12:00:00Z",
  "reference_dataset": "snrna_merged.h5ad",
  "spatial_dataset": "spatial_merged.h5ad",
  "hyperparameters": {
    "mode": "cells",
    "density_prior": "rna_count_based",
    "lambda_g1": 1.0,
    "lambda_d": 0.5
  },
  "runtime_seconds": 3600,
  "git_commit": "abc123def"
}
```

### 3.5 Upstream Metrics

**Schema: `upstream_metrics.json`**

```json
{
  "spatial_coherence": {
    "moran_i_mean": 0.45,
    "moran_i_std": 0.12,
    "geary_c_mean": 0.65
  },
  "proportion_quality": {
    "entropy_mean": 1.8,
    "entropy_std": 0.4,
    "sparsity": 0.3
  },
  "confidence_stats": {
    "mean": 0.75,
    "median": 0.80,
    "q25": 0.65,
    "q75": 0.88
  },
  "computational": {
    "runtime_seconds": 3600,
    "peak_memory_gb": 48
  }
}
```

---

## 4. Canonical File Outputs (Step 0)

After running `run_data_prep.py`, the following files must exist:

```
data/processed/<dataset>/
 snrna_merged.h5ad                    # 19GB (LUAD)
 snrna_qc_normalized.h5ad             # 15GB (post-QC)
 snrna_manifest.csv                   # Sample metadata
 spatial_merged.h5ad                  # 35GB (LUAD)
 spatial_qc_normalized.h5ad           # 28GB (post-QC)
 spatial_manifest.csv                 # Sample metadata
 wes_features.parquet                 # 50KB
 cells.parquet                        # 1GB
 neighborhoods.parquet                # 2GB
 stage_edges.parquet                  # 10MB
 split_manifest.json                  # 10KB
 feature_spec.yaml                    # 5KB
 spatial_backend/
    tangram/...
    destvi/...
    tacco/...
 audit_report.json                    # QC summary
```

**Total Size Estimate:** ~100GB for LUAD dataset

---

## 5. Data Loading API

### 5.1 Cell Loader

```python
from stagebridge.data import CellDataset

dataset = CellDataset(
    cells_path="data/processed/luad_evo/cells.parquet",
    neighborhoods_path="data/processed/luad_evo/neighborhoods.parquet",
    split="train",
    spatial_backend="tangram",
    load_neighborhoods=True,
    load_expression=True,
    load_latents=True
)

# Access single cell
cell = dataset[0]
assert "cell_id" in cell
assert "z_fused" in cell
assert "niche_embedding" in cell  # if neighborhoods loaded
```

### 5.2 Stage-Edge Loader

```python
from stagebridge.data import StageEdgeBatchLoader

loader = StageEdgeBatchLoader(
    cells_path="data/processed/luad_evo/cells.parquet",
    edges_path="data/processed/luad_evo/stage_edges.parquet",
    split="train",
    batch_size=64,
    edge_sampling="uniform"  # or "weighted"
)

for batch in loader:
    src_cells = batch["source_cells"]  # (B, D)
    tgt_cells = batch["target_cells"]  # (B, D)
    src_niches = batch["source_niches"]  # (B, N, D)
    edge_ids = batch["edge_ids"]  # (B,)
```

### 5.3 Spatial Backend Loader

```python
from stagebridge.data import SpatialBackendLoader

backend = SpatialBackendLoader(
    backend_name="tangram",
    backend_dir="data/processed/luad_evo/spatial_backend/tangram"
)

proportions = backend.load_proportions()  # DataFrame
confidence = backend.load_confidence()  # DataFrame
metadata = backend.load_metadata()  # dict
metrics = backend.load_upstream_metrics()  # dict
```

---

## 6. Validation and Integrity Checks

### 6.1 Required Checks (Run After Step 0)

```python
from stagebridge.data import validate_data_model

report = validate_data_model("data/processed/luad_evo")

# Required checks:
assert report["cells_exist"], "cells.parquet missing"
assert report["neighborhoods_exist"], "neighborhoods.parquet missing"
assert report["edges_exist"], "stage_edges.parquet missing"
assert report["splits_exist"], "split_manifest.json missing"
assert report["feature_spec_exist"], "feature_spec.yaml missing"

# Integrity checks:
assert report["all_cell_ids_unique"], "Duplicate cell IDs found"
assert report["all_donors_in_splits"], "Orphan donors found"
assert report["all_stages_in_edges"], "Missing stage edges"
assert report["neighborhoods_match_cells"], "Neighborhood cell IDs don't match"

# Spatial backend checks:
assert len(report["spatial_backends"]) >= 3, "Need 3+ spatial backends"
assert "tangram" in report["spatial_backends"], "Tangram required"
assert "destvi" in report["spatial_backends"], "DestVI required"
assert "tacco" in report["spatial_backends"], "TACCO required"

# Completeness checks:
assert report["pct_cells_with_latents"] > 0.95, "Missing latents"
assert report["pct_spatial_cells_with_neighborhoods"] > 0.95, "Missing neighborhoods"
```

### 6.2 Automated Validation Script

```bash
python -m stagebridge.data.validate \
    --data-dir data/processed/luad_evo \
    --output validation_report.json
```

---

## 7. Data Versioning and Provenance

### 7.1 Dataset Versioning

Each processed dataset should have a version file:

**`data/processed/<dataset>/VERSION`**
```
dataset: luad_evo
version: 1.0.0
creation_date: 2026-03-15T10:00:00Z
git_commit: abc123def456
stagebridge_version: 0.1.0
raw_data_sources:
  - GSE308103 (snRNA)
  - GSE307534 (Visium)
  - GSE307529 (WES)
qc_params:
  min_genes: 200
  min_cells: 3
  max_pct_mito: 20
  min_counts: 500
spatial_backends:
  - tangram==1.2.0
  - destvi==0.9.1
  - tacco==0.3.0
```

### 7.2 Audit Trail

**`audit_report.json`** generated by Step 0:

```json
{
  "pipeline": "data_prep",
  "version": "1.0",
  "start_time": "2026-03-15T08:00:00Z",
  "end_time": "2026-03-15T18:00:00Z",
  "duration_hours": 10,

  "snrna": {
    "n_samples": 18,
    "cells_before_qc": 520000,
    "cells_after_qc": 485000,
    "genes_before_qc": 32000,
    "genes_after_qc": 2000,
    "qc_filters_applied": true
  },

  "spatial": {
    "n_samples": 56,
    "spots_before_qc": 340000,
    "spots_after_qc": 325000,
    "genes_before_qc": 32000,
    "genes_after_qc": 2000,
    "qc_filters_applied": true
  },

  "wes": {
    "n_samples": 18,
    "features_extracted": 9,
    "samples_with_wes": 18
  },

  "spatial_backends": {
    "tangram": {"status": "success", "runtime_seconds": 3600},
    "destvi": {"status": "success", "runtime_seconds": 7200},
    "tacco": {"status": "success", "runtime_seconds": 1800}
  },

  "artifacts_generated": [
    "cells.parquet",
    "neighborhoods.parquet",
    "stage_edges.parquet",
    "split_manifest.json",
    "feature_spec.yaml"
  ],

  "warnings": [],
  "errors": []
}
```

---

## 8. Extension Points (V2+)

### 8.1 Additional Modalities (V2)

Future versions may add:
- **Imaging features:** H&E, IF, IHC quantifications
- **Proteomics:** CODEX, CyCIF multiplexed imaging
- **Metabolomics:** Spatial metabolomics
- **Epigenomics:** scATAC-seq, scCUT&Tag

Schema extensions:
- `cells.parquet` adds columns: `imaging_features`, `protein_abundances`, etc.
- New files: `imaging_features.parquet`, `protein_features.parquet`

### 8.2 Cross-Organ Edges (V3)

For metastasis modeling:
- **Cross-organ edges:** Lung → Brain, Lung → Bone, etc.
- Schema extension: `stage_edges.parquet` adds `source_organ`, `target_organ`

### 8.3 Temporal Data (V3)

For longitudinal studies:
- **Timepoint field:** Add `timepoint` to `cells.parquet`
- **Temporal edges:** Edges between same donor at different times

---

## 9. Data Model Compliance Checklist

A dataset is V1-compliant if:

-  `cells.parquet` exists with all required fields
-  `neighborhoods.parquet` exists for spatial cells
-  `stage_edges.parquet` defines transition graph
-  `split_manifest.json` has donor-held-out splits
-  `feature_spec.yaml` documents all features
-  At least 3 spatial backends run and standardized
-  WES features available (even if optional)
-  All cell IDs are unique
-  All referenced IDs exist (no orphans)
-  Validation script passes all checks
-  Audit report generated
-  Version file exists with provenance

---

**End of Data Model Specification**
