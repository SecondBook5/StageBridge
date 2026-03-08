# Figure Manifest

## Figure 1: StageBridge Architecture Block Diagram
- **File**: `figures/figure_1_stagebridge_block_diagram.png` (+ `.pdf`, `.mmd`)
- **Caption**: StageBridge architecture overview. Input cells are embedded in HLCA latent space, spatial mapping produces typed niche tokens (epithelial, stromal, immune, vascular), a Set Transformer encodes the token set into a context vector c_s via ISAB, SAB, and PMA, and the context conditions an OT-based transition model that predicts target-stage cell distributions. An optional Graph Transformer provides tissue-level context.
- **Source**: Conceptual diagram, generated programmatically
- **Paper section**: Section 4 (Method)

## Figure 2: Context Mode Comparison by Edge
- **File**: `figures/figure_2_mode_comparison_by_edge.png` (+ `.pdf`)
- **Caption**: Donor-held-out Sinkhorn divergence (lower = better) across four context-encoding modes for two transition edges. Set Transformer achieves the best score on AIS->MIA (15.76) and outperforms pooled context on both edges.
- **Source**: Registry rows 18-21 (AAH->AIS) and 23-27 (AIS->MIA)
- **Paper section**: Section 6 (Preliminary Results)

## Figure 3: Context Ablation Decision Matrix
- **File**: `figures/figure_3_context_ablation_summary.png` (+ `.pdf`)
- **Caption**: Ablation decision matrix showing pairwise mode comparisons on each edge. Green = pass, red = fail. Set Transformer consistently outperforms pooled context; Graph-of-Sets does not earn flagship status.
- **Source**: Derived from registry comparison data
- **Paper section**: Section 6 (Preliminary Results)

## Figure 4: Extension Component Status
- **File**: `figures/figure_4_extension_status.png` (+ `.pdf`)
- **Caption**: Relative evidence strength for optional extension components. Only the Set Transformer (main story) exceeds the inclusion threshold. Graph-of-Sets, WES regularization, and state-dependent diffusion remain extensions with mixed evidence.
- **Source**: Derived from gate evaluation results (registry rows 31-34)
- **Paper section**: Section 6.3 (Extension Status)

## Figure 5: HLCA Latent Space by Disease Stage
- **File**: `figures/figure_5_latent_space_by_stage.png` (+ `.pdf`)
- **Caption**: Left: UMAP embedding of 20,000 cells from the HLCA-derived 32-dimensional latent space, colored by disease stage. Right: Stage density contours showing overlapping but distinguishable stage distributions. Normal and LUAD occupy distinct regions; intermediate stages (AAH, AIS, MIA) show progressive shift.
- **Source**: snrna_hlca_latent_full.h5ad (GSE308103, 798,100 cells, subsampled to 20K)
- **Paper section**: Section 4.1 (Data Representation)

## Figure 6: Stage Centroid Separation Heatmap
- **File**: `figures/figure_6_stage_centroid_heatmap.png` (+ `.pdf`)
- **Caption**: Pairwise Euclidean distances between stage centroids in the HLCA latent space. LUAD is most distant from Normal (1.05), confirming the latent space captures disease progression. Adjacent stages have smaller distances (Normal-AAH: 0.40, AAH-AIS: 0.41, AIS-MIA: 0.46).
- **Source**: Full snrna_hlca_latent_full.h5ad (798,100 cells)
- **Paper section**: Section 4.1 (Data Representation)

## Figure 7: Typed Niche Token Composition
- **File**: `figures/figure_7_niche_token_composition.png` (+ `.pdf`)
- **Caption**: Left: Heatmap of mean typed token values (epithelial, stromal, immune, vascular) across disease stages. Right: Stacked bar showing niche composition balance across the disease ladder. Immune token proportion decreases from Normal to MIA, while vascular increases.
- **Source**: niche_tokens_full.parquet (639,816 spatial spots)
- **Paper section**: Section 4.2 (Typed Niche Token Construction)

## Figure 8: Latent Dimension Violin Plots
- **File**: `figures/figure_8_latent_dimension_violins.png` (+ `.pdf`)
- **Caption**: Per-stage distributions of the first six HLCA latent dimensions, shown as violin plots. Several dimensions show stage-dependent shifts (e.g., HLCA dim 1 and 2), confirming the latent space encodes biologically meaningful variation across the progression trajectory.
- **Source**: snrna_hlca_latent_full.h5ad (subsampled to 20K cells)
- **Paper section**: Section 4.1 (Data Representation)

## Figure 9: Cell-Type Abundances by Stage
- **File**: `figures/figure_9_celltype_by_stage_heatmap.png` (+ `.pdf`)
- **Caption**: Mean Tangram-estimated abundances of key cell types (AT2, Macrophages, T cells, Fibroblasts) across disease stages. Macrophage and T cell abundances decrease from Normal to LUAD, while fibroblast abundance increases, reflecting known microenvironmental remodeling during progression.
- **Source**: niche_tokens_full.parquet (639,816 spots, Tangram deconvolution)
- **Paper section**: Section 4.2 (Typed Niche Token Construction)

## Figure 10: Schrödinger Bridge Interpolation
- **File**: `figures/figure_10_schrodinger_bridge.png` (+ `.pdf`)
- **Caption**: Schrödinger bridge interpolation between AIS and MIA stages. Left: source (AIS) and target (MIA) distributions in UMAP space. Center: interpolated distributions at t ∈ {0.25, 0.5, 0.75} showing smooth transport paths. Right: noise schedule σ√(t(1−t)) governing stochastic perturbation magnitude.
- **Source**: snrna_hlca_latent_full.h5ad (subsampled, UMAP-projected) with simulated interpolation
- **Paper section**: Section 4.4 (Transition Model)

## Figure 11: Transition Flow UMAP
- **File**: `figures/figure_11_transition_flow_umap.png` (+ `.pdf`)
- **Caption**: UMAP embedding with stage transition arrows illustrating learned flow directions between adjacent disease stages. Arrow directions computed from stage centroid differences in latent space, showing the progressive Normal → AAH → AIS → MIA → LUAD trajectory.
- **Source**: snrna_hlca_latent_full.h5ad (subsampled to 20K cells)
- **Paper section**: Section 6 (Preliminary Results)

## Figure 12: Architecture Comparison
- **File**: `figures/figure_12_architecture_comparison.png` (+ `.pdf`)
- **Caption**: Left: Architectural summary table comparing four context-encoding modes (RNA-only, pooled, Set Transformer, Graph-of-Sets) on key properties (attention, spatial awareness, scalability). Right: Sinkhorn divergence comparison across edges, highlighting Set Transformer as the best-performing architecture.
- **Source**: Registry rows 18-27
- **Paper section**: Section 4.3 (Context Encoding)

## Figure 13: WES Mutation Landscape
- **File**: `figures/figure_13_wes_landscape.png` (+ `.pdf`)
- **Caption**: Left: Heatmap of WES-derived mutation features across patients and disease stages. Right: Tumor mutational burden (TMB) by stage, showing progressive accumulation of somatic mutations from Normal through LUAD.
- **Source**: wes_features.parquet (GSE307529)
- **Paper section**: Section 4.5 (WES Regularization)

## Figure 14: Spatial Niche Maps
- **File**: `figures/figure_14_spatial_niche_map.png` (+ `.pdf`)
- **Caption**: Spatial niche maps for representative AIS and MIA tissue sections. Left panels: dominant cell-type maps showing spatial organization. Right panel: niche entropy quantifying microenvironment heterogeneity. MIA sections show higher entropy, consistent with increased stromal and immune infiltration during invasion.
- **Source**: niche_tokens_full.parquet (spatial coordinates + deconvolution)
- **Paper section**: Section 4.2 (Typed Niche Token Construction)
