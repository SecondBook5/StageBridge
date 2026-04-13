# Clonal Pattern Extraction Pipeline

## Overview

This module extracts clonal evolution patterns (1a, 1b, 2) from spatial transcriptomics data
using CNV inference, following the methodology of Peng et al. 2025.

## Pattern Definitions

| Pattern | Description | Clonal Relationship |
|---------|-------------|---------------------|
| **1a** | Direct lineage | Precursor clones present in LUAD + LUAD has additional subclones |
| **1b** | Branched evolution | Shared clones + stage-specific clones in both |
| **2** | Independent origins | No shared clones between precursor and LUAD |

## Workflow

```
1. Load spatial data per patient
   |
2. Filter to epithelial spots only
   |
3. Run CNV inference (infercnvpy or copyKAT)
   |
4. Cluster spots by CNV profile -> identify clones
   |
5. Map clones to tissue type (precursor vs LUAD)
   |
6. Build phylogenetic tree
   |
7. Classify pattern based on clone sharing
   |
8. Output: patient -> pattern mapping
```

## Dependencies

```bash
pip install infercnvpy scanpy
```

## Usage

```python
from stagebridge.clonal import extract_clonal_patterns

# Run on all patients
patterns = extract_clonal_patterns(
    spatial_adata=spatial_adata,
    reference_cells="Normal",  # Use normal epithelial as reference
    output_dir="results/clonal/"
)

# Result: {'P1': '1b', 'P2': '1a', ...}
```

## References

- Peng et al. 2025 - Original methodology using SpatialInferCNV
- infercnvpy - Python implementation of inferCNV
- copyKAT - Alternative CNV inference method
