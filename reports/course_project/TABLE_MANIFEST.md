# Table Manifest

## Table 1: Core Mode Comparison
- **Files**: `tables/core_mode_comparison.csv`, `tables/core_mode_comparison.md`
- **Description**: Primary comparison of four context-encoding modes (rna_only, pooled, set_only, graph_of_sets) across two transition edges (AAH->AIS, AIS->MIA). Includes transformer usage flags, context type, and course interpretation.
- **Source**: Registry rows 18-21 (AAH->AIS) and 23-27 (AIS->MIA)
- **Paper section**: Section 6 (Preliminary Results), Table 1

## Table 2: Context Ablation Summary
- **Files**: `tables/context_ablation_summary.csv`, `tables/context_ablation_summary.md`
- **Description**: Per-edge ablation results: whether Set Transformer beats pooled, RNA-only, and whether Graph-of-Sets earned its place. Evidence graded as pass/weak_pass/inconclusive/fail.
- **Source**: Derived from core mode comparison
- **Paper section**: Section 6.2 (Key Findings)

## Table 3: Extension Status Summary
- **Files**: `tables/extension_status_summary.csv`, `tables/extension_status_summary.md`
- **Description**: Status of optional extension components (Graph-of-Sets, WES, diffusion). Includes tested flag, evidence summary, and role in course story.
- **Source**: Gate evaluation results (registry rows 31-34)
- **Paper section**: Section 6.3 (Extension Status)
