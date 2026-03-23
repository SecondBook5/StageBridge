# StageBridge Development Rules

## Before Modifying ANY Code

1. **Ask the research-director agent first** - It knows the doctrine and canonical flow
2. **Read existing code before writing new code** - Check if it already exists
3. **Understand the data sizes** - snRNA: ~800k cells, Spatial: ~640k spots
4. **Check the pipelines README** - `stagebridge/pipelines/README.md` has the execution order

## Pre-Flight Checklist

Before implementing anything, answer these questions:

- [ ] Does this already exist in the codebase? (Search first)
- [ ] What is the minimal change needed?
- [ ] What are the memory implications for real data sizes?
- [ ] Have I consulted the research-director agent?
- [ ] Am I following the established pipeline order?

## Pipeline Order (Do Not Deviate)

```
1. run_data_prep.py              (QC/merge)
2. download_references.py        (HLCA/LuCA)
3. add_ensembl_ids.py            (gene ID prep for model-based mapping)
4. run_reference.py --hlca-only  (HLCA mapping first)
5. run_reference.py --luca-only  (LuCA mapping, may need pandas 1.5.x env)
6. run_spatial_benchmark.py      (Tangram/DestVI/TACCO/Cell2Location)
7. complete_data_prep.py         (canonical format)
8. run_v1_complete.py            (training)
```

Note: Steps 4-5 use model-based scArches surgery. LuCA may require separate
environment with pandas 1.5.x due to model compatibility issues.

## Key Principles

1. **Use existing code** - Don't reinvent. Search `stagebridge/` first.
2. **Step-by-step verification** - Run each pipeline step individually, verify output, then proceed.
3. **Memory awareness** - Any operation on 600k+ cells needs chunking or streaming.
4. **No shortcuts** - Don't skip reference mapping or spatial benchmarking.
5. **Spatial backends handle spatial processing** - Use `--spatial-merge-only` flag.

## When Adding New Features

1. Consult research-director agent for doctrine compliance
2. Check if existing modules can be extended
3. Test with small synthetic data first
4. Document in the appropriate README

## Data Locations

```
$DATA/
├── raw/geo/                                    # Raw downloads
├── references/                                 # HLCA + LuCA atlases
│   ├── hlca/hlca_reference.h5ad
│   ├── hlca/hub_cache/                         # scANVI model from HuggingFace
│   └── luca/luca_core_atlas.h5ad               # Use CORE, not Extended
├── processed/luad_evo/
│   ├── snrna_qc_normalized.h5ad                # After QC
│   ├── snrna_qc_normalized_with_ensg.h5ad      # With ENSG IDs (for model mapping)
│   ├── spatial_merged.h5ad                     # Ready for spatial backends
│   ├── reference_geometry/                     # Dual-reference embeddings
│   │   ├── hlca_embedding.parquet
│   │   ├── luca_embedding.parquet
│   │   └── fused_embedding.parquet
│   └── canonical/                              # Ready for training
└── runs/                                       # Training outputs
```

## Common Mistakes to Avoid

- Adding placeholder/hacky implementations instead of using existing code
- Running full QC on spatial data (backends handle it)
- Making matrix copies of 600k+ cell data
- Skipping the research-director agent consultation
- Implementing without reading existing pipelines first

## HPC Environment Notes

- **PyTorch CUDA**: Use cu124 (CUDA 12.4), not cu130. Even if nvidia-smi shows 13.x
- **GPU detection**: Always `export CUDA_VISIBLE_DEVICES=0,1,2,3` before running
- **Verify GPU**: `python -c "import torch; print(torch.cuda.is_available())"`
- **LuCA pandas**: Model requires pandas 1.5.x (create separate env if needed)
- **Gene IDs**: HLCA model expects ENSG IDs - run add_ensembl_ids.py first
