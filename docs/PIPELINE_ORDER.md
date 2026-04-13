# StageBridge V1 Pipeline Order

Complete execution order for the StageBridge publication pipeline.

## Prerequisites (Manual, Before Snakemake)

```bash
# Step 0: Data preparation (QC, merge datasets)
python -m stagebridge.pipelines.run_data_prep --data-root $DATA

# Creates:
#   $DATA/processed/luad_evo/snrna_qc_normalized.h5ad
#   $DATA/processed/luad_evo/spatial_merged.h5ad
#   $DATA/processed/luad_evo/wes_features.parquet

# Step 0b: Add Ensembl IDs (required for model-based mapping)
python -m stagebridge.pipelines.add_ensembl_ids \
    --input $DATA/processed/luad_evo/snrna_qc_normalized.h5ad \
    --output $DATA/processed/luad_evo/snrna_qc_normalized_with_ensg.h5ad

# Step 0c: Download references
python -m stagebridge.pipelines.download_references --data-root $DATA
```

## Snakemake Pipeline (Automated)

Run with: `snakemake --profile workflow/slurm --jobs 20`

### Stage 1-2: Reference Mapping

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  INPUT: snrna_qc_normalized_with_ensg.h5ad (~800k cells)                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
           ┌────────────────┐              ┌────────────────┐
           │  hlca_mapping  │              │  luca_mapping  │
           │  (scArches)    │              │  (scArches)    │
           │  30-dim latent │              │  10-dim latent │
           └────────────────┘              └────────────────┘
                    │                               │
                    │    ┌──────────────────────────┤
                    │    │                          │
                    ▼    ▼                          │
           ┌────────────────┐                       │
           │fuse_embeddings │                       │
           │ 40-dim fused   │                       │
           └────────────────┘                       │
                    │                               │
                    │                               │
                    ▼                               ▼
           ┌────────────────────────────────────────────────┐
           │           add_cell_type_labels                 │
           │  Adds cell_type (HLCA) + luca_cell_type (LuCA) │
           └────────────────────────────────────────────────┘
                                    │
                                    ▼
           ┌────────────────────────────────────────────────┐
           │             validate_markers                    │
           │  QC check: marker enrichment by cell type       │
           └────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
```

### Stage 3: Spatial Benchmark + Clonal Extraction (Parallel)

```
           ┌────────────────┐              ┌────────────────────────┐
           │clonal_extraction│             │ spatial_backend_sample │
           │ (inferCNVpy)   │              │ (8 backends × 2 labels │
           │                │              │  × 56 samples = 896)   │
           └────────────────┘              └────────────────────────┘
                    │                               │
                    │                      ┌────────┴────────┐
                    │                      ▼                 ▼
                    │              aggregate_samples   unified_figures
                    │                      │                 │
                    │                      └────────┬────────┘
                    │                               ▼
                    │                      spatial_comparison
                    │                      (selects best backend)
                    │                               │
                    │                               ▼
                    │                      canonical_backend_aggregate
                    │                               │
                    └───────────────┬───────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  OUTPUTS:                                                                    │
│    - clonal_patterns.json (donor → 1a/1b/2 pattern)                         │
│    - backend_comparison.json (8 backends ranked)                             │
│    - canonical/cell_type_proportions.parquet (best backend results)          │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Stage 4: Data Preparation

```
           ┌────────────────────────────────────────────────┐
           │              data_preparation                   │
           │                                                 │
           │  INPUTS:                                        │
           │    - fused_embedding.parquet (40-dim)          │
           │    - canonical/cell_type_proportions.parquet   │
           │    - snrna_with_celltypes.h5ad                 │
           │    - spatial_merged.h5ad                       │
           │    - wes_features.parquet                      │
           │    - clonal_patterns.json  ← NEW               │
           │                                                 │
           │  OUTPUTS:                                       │
           │    - cells.parquet (with clonal_pattern col)   │
           │    - neighborhoods.parquet                      │
           │    - split_manifest.json (donor-held-out)      │
           └────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
           ┌────────────────┐              ┌────────────────┐
           │validate_splits │              │semi_synthetic  │
           │(leakage check) │              │  benchmark     │
           └────────────────┘              └────────────────┘
                    │                               │
                    └───────────────┬───────────────┘
                                    ▼
```

### Stage 5: Training + Validation

```
           ┌────────────────────────────────────────────────┐
           │                    hpo                          │
           │           (30 Optuna trials)                    │
           └────────────────────────────────────────────────┘
                                    │
           ┌────────────────────────┼────────────────────────┐
           ▼                        ▼                        ▼
   ┌────────────────┐      ┌────────────────┐      ┌────────────────┐
   │   training     │      │training_frozen │      │   baseline     │
   │  (5×3 = 15)    │      │  (5×3 = 15)    │      │ (4×5×3 = 60)   │
   │                │      │                │      │                │
   │ SSL: 100 epochs│      │ SSL: 100 epochs│      │ pooling_mlp    │
   │ Trans: 50 eps  │      │ Trans: FROZEN  │      │ deep_sets      │
   └────────────────┘      └────────────────┘      │ set_transformer│
           │                        │              │ graph_sage     │
           ▼                        │              └────────────────┘
   ┌────────────────┐               │                      │
   │aggregate_cv    │               │                      ▼
   │   results      │               │              ┌────────────────┐
   └────────────────┘               │              │aggregate_      │
           │                        │              │  baselines     │
           ▼                        │              └────────────────┘
   ┌────────────────┐               │                      │
   │ h3_validation  │◄──────────────┘                      │
   │                │                                      │
   │ H3.1: trans~   │                                      │
   │   shared clone │                                      │
   │ H3.2: niche by │                                      │
   │   pattern      │                                      │
   └────────────────┘                                      │
           │                                               │
           └───────────────────────┬───────────────────────┘
                                   ▼
```

### Stage 6-7: Ablation + Figures

```
           ┌────────────────────────────────────────────────┐
           │                 ablation                        │
           │            (14 configurations)                  │
           │                                                 │
           │  no_niche, pooled_niche, no_hierarchy,         │
           │  no_hlca, no_luca, single_reference,           │
           │  no_attention, no_flow, deterministic,         │
           │  no_genomics, no_uncertainty, reduced_latent,  │
           │  no_ssl, baseline_mlp                          │
           └────────────────────────────────────────────────┘
                                    │
                                    ▼
           ┌────────────────────────────────────────────────┐
           │              ablation_summary                   │
           └────────────────────────────────────────────────┘
                                    │
                                    ▼
           ┌────────────────────────────────────────────────┐
           │            publication_figures                  │
           │                                                 │
           │  INPUTS: all results from above                 │
           │  OUTPUTS: main/ and supplementary/ figures      │
           └────────────────────────────────────────────────┘
```

## Key Data Flow: Clonal Patterns

```
1. snrna_with_celltypes.h5ad
   └── clonal_extraction (inferCNVpy)
       └── clonal_patterns.json {donor_id: "1a"|"1b"|"2"|"stable"}

2. clonal_patterns.json
   └── data_preparation
       └── cells.parquet
           ├── clonal_pattern: str ("1a", "1b", "2", "stable", "unknown")
           └── clonal_pattern_idx: int (0, 1, 2, 3, -1)

3. cells.parquet + trained model
   └── h3_validation
       └── h3_validation.json
           ├── h3_1: {auc, odds_ratio, pvalue, h3_1_supported}
           └── h3_2: {mean_influence_1a/1b/2, pvalue, effect_size, h3_2_supported}
```

## Hypothesis Validation

| Hypothesis | Test | Metric | Threshold |
|------------|------|--------|-----------|
| H3.1 | Transition prob ~ shared clones | AUC | > 0.6 |
| H3.1 | Association strength | Odds Ratio | > 1.5 |
| H3.2 | Niche influence: 1a > 2 | Mann-Whitney p | < 0.05 |
| H3.2 | Effect magnitude | Rank-biserial r | > 0.2 |

## Running Individual Stages

```bash
# Just clonal extraction
snakemake --profile workflow/slurm $DATA/runs/clonal/clonal_patterns.json

# Just spatial backends (all 896 jobs)
snakemake --profile workflow/slurm $DATA/runs/spatial_benchmark/backend_comparison.json

# Just training (all 15 runs)
snakemake --profile workflow/slurm $DATA/runs/v1_complete/aggregated/cv_results.json

# Just H3 validation
snakemake --profile workflow/slurm $DATA/runs/v1_complete/h3_validation/h3_validation.json

# Everything
snakemake --profile workflow/slurm --jobs 20
```
