# Course Package Status

## What is Complete

### Paper Draft
- Abstract, Introduction, Related Work, Research Problem, and Method sections are near-final quality
- Experimental Plan, Preliminary Results, Statistical Evaluation, and Conclusion are drafted
- 20 references cited from reputable venues (ICML, NeurIPS, Nature Methods, etc.)
- Mathematical formulation includes Set Transformer equations, Schrodinger bridge interpolant, drift loss

### Figures (9 total)
- Figure 1: Architecture block diagram (conceptual)
- Figure 2: Core mode comparison by edge (from registry data)
- Figure 3: Context ablation decision matrix (from registry data)
- Figure 4: Extension component status (from gate results)
- Figure 5: UMAP latent space by disease stage (from 798K cells, subsampled to 20K)
- Figure 6: Stage centroid separation heatmap (from full 798K cells)
- Figure 7: Typed niche token composition heatmap + stacked bar (from 640K spots)
- Figure 8: Latent dimension violin plots by stage (from 20K cells)
- Figure 9: Cell-type abundances by stage heatmap (from 640K spots)

### Tables (3 sets, CSV + MD)
- Core mode comparison (4 modes x 2 edges)
- Context ablation summary (per-edge pass/fail)
- Extension status summary (3 components)

### Human-Readable Summaries
- Executive results summary
- Run provenance (which registry rows feed each figure)
- Model interpretation guide

### Manifests and Checklists
- Figure manifest with captions and source data
- Table manifest with descriptions
- Submission checklist mapped to rubric

## What is Preliminary

- Experimental plan section (draft, needs final experiment list)
- Conclusion section (draft, needs final language)
- Statistical evaluation plan (draft, bootstrap/Wilcoxon planned but not executed)
- Notebook exports (to be finalized)

## Which Runs Were Used

All results come from the results registry (`results/registry/results_registry.csv`):
- Registry rows 18-21: AAH->AIS mode comparison (rna_only, set_only, pooled, graph_of_sets)
- Registry rows 23-27: AIS->MIA mode comparison
- Registry rows 31-34: WES and diffusion gate evaluations
- All runs on commit `20e0d11`, main branch

## What is Still Placeholder

- Notebook demo exports (notebook_demo_results.csv, notebook_demo_comparison.png)
- Notebook demo summary (NOTEBOOK_DEMO_SUMMARY.md)
- Bootstrap confidence intervals and p-values
- Full 5-stage chain evaluation
- Brain metastasis extension results
