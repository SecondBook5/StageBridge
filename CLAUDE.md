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
1. run_data_prep.py          (QC/merge)
2. download_references.py    (HLCA/LuCA)
3. run_reference.py          (mapping)
4. run_spatial_benchmark.py  (Tangram/DestVI/TACCO)
5. complete_data_prep.py     (canonical format)
6. run_v1_complete.py        (training)
```

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
├── raw/geo/                          # Raw downloads
├── references/                       # HLCA + LuCA atlases
├── processed/luad_evo/
│   ├── snrna_qc_normalized.h5ad     # Ready for reference mapping
│   ├── spatial_merged.h5ad          # Ready for spatial backends
│   └── canonical/                    # Ready for training
└── runs/                             # Training outputs
```

## Common Mistakes to Avoid

- Adding placeholder/hacky implementations instead of using existing code
- Running full QC on spatial data (backends handle it)
- Making matrix copies of 600k+ cell data
- Skipping the research-director agent consultation
- Implementing without reading existing pipelines first
