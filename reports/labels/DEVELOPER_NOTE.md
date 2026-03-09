# Developer Note

## Current assumptions
- `sample_id` is used as the lesion identifier.
- Existing `wes_features.parquet` is treated as a lesion-level WES proxy layer.
- CNA, clonal, phylogeny, and pathology backends default to parse-only and may produce empty normalized summaries when no external results are configured.

## External tools expected but not bundled
- FACETS
- CNVkit
- Sequenza
- PyClone-VI
- PhylogicNDT
- Pairtree
- Treeomics
- QuPath
- QuST

## Current data limitations
- `AAH->AIS` still lacks enough negative donor support for donor-held-out binary benchmarking.
- The current run used no parsed external CNA/clonal/phylogeny/pathology outputs, so refinement relied on curated labels, heuristic provenance, later-stage support, and existing WES proxy features.
- AAH label repair currently supports a conservative continuous-risk recommendation rather than a repaired binary benchmark.
