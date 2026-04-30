# StageBridge Figure Inventory

## Status Summary

**WORKING** (from V1, based on real data):
- Embeddings (UMAP, PHATE, tSNE)
- Cell type annotations
- IL1B expression analysis
- Spatial deconvolution
- Cell composition
- Some biological validation

**BROKEN/MISSING**:
- Model training checkpoints (empty!)
- Evaluation metrics (all NaN)
- Baseline comparisons (no results)
- Ablation studies (no results)
- LIANA L-R analysis (directory empty)

---

## Figures from V1 (in figures/from_v1/)

### Embeddings & Annotations
| Figure | Description | Status |
|--------|-------------|--------|
| `panel_umap_stage.png` | UMAP colored by disease stage | WORKING |
| `panel_umap_celltype.png` | UMAP colored by cell type | WORKING |
| `panel_umap_il1b.png` | UMAP colored by IL1B expression | WORKING |
| `panel_umap_gamma.png` | UMAP colored by gamma (proliferation?) | WORKING |
| `panel_umap_flow.png` | UMAP with velocity/flow arrows | WORKING |
| `panel_phate_stage.png` | PHATE colored by stage | WORKING |
| `panel_phate_pseudotime.png` | PHATE colored by pseudotime | WORKING |
| `fused_umap.png` | Fused embedding UMAP | WORKING |
| `fused_umap_by_donor.png` | UMAP by donor (batch check) | WORKING |
| `fused_phate.png` | Fused embedding PHATE | WORKING |
| `fused_tsne.png` | Fused embedding tSNE | WORKING |
| `fig1_embedding_overview.png` | Publication-style overview | WORKING |

### Spatial Deconvolution
| Figure | Description | Status |
|--------|-------------|--------|
| `destvi_celltype_heatmap.png` | Cell type proportions heatmap | WORKING |
| `destvi_celltype_by_stage.png` | Cell types by disease stage | WORKING |
| `panel_spatial_P4_stage.png` | Spatial map P4 by stage | WORKING |
| `panel_spatial_P4_il1b.png` | Spatial map P4 IL1B | WORKING |
| `panel_spatial_P4_gamma.png` | Spatial map P4 gamma | WORKING |
| `panel_spatial_donor_P1.png` | Spatial map donor P1 | WORKING |
| `panel_spatial_flow.png` | Spatial flow visualization | WORKING |
| `panel_spatial_transition_zones.png` | Transition zones | WORKING |
| `advanced_spatial_vs_snrna.png` | Spatial vs snRNA comparison | WORKING |

### Biology / IL1B
| Figure | Description | Status |
|--------|-------------|--------|
| `panel_il1b_expression.png` | IL1B expression overview | WORKING |
| `panel_il1b_violin.png` | IL1B violin by stage | WORKING |
| `advanced_il1b_analysis.png` | Advanced IL1B analysis | WORKING |
| `fig_il1b_association.png` | IL1B associations | WORKING |
| `panel_cell_composition.png` | Cell type composition | WORKING |

### Phase Portraits / Dynamics
| Figure | Description | Status |
|--------|-------------|--------|
| `panel_phase_portrait.png` | Phase portrait | WORKING |
| `panel_phase_portrait_flow.png` | Phase portrait with flow | WORKING |
| `fig_ot_dynamics.png` | OT dynamics | WORKING |
| `panel_phase_prolif.png` | Phase + proliferation | WORKING |

### Clonal / Evolution
| Figure | Description | Status |
|--------|-------------|--------|
| `fig2b_clonal_evolution.png` | Clonal evolution | WORKING |
| `panel_mutation_landscape.png` | Mutation landscape | WORKING |
| `panel_mutation_violin.png` | Mutations by stage | WORKING |

### Reference Mapping
| Figure | Description | Status |
|--------|-------------|--------|
| `fig4_reference_comparison.png` | HLCA vs LuCA comparison | WORKING |
| `embedding_overview.png` | Embedding overview | WORKING |

### Publication Composites
| Figure | Description | Status |
|--------|-------------|--------|
| `fig_publication_main.png` | Main publication figure | WORKING |
| `figure_architecture.png` | Architecture diagram | WORKING |
| `figure_biology_validation.png` | Biology validation | WORKING |
| `figure_transition_stages.png` | Stage transitions | WORKING |

---

## CRITICAL GAPS

### 1. Model Training Failed
- All checkpoints are EMPTY
- Evaluation metrics are NaN
- Need to re-run training

### 2. LIANA L-R Analysis Missing
- Directory exists but empty
- Critical for IL1B-IL1R1 hypothesis
- Need to run LIANA

### 3. Baselines/Ablations Empty
- Folders exist but no results
- Can't show model comparison
- Need to re-run

### 4. Missing Standard QC
- No nGenes/nUMI/mito violin plots
- No marker gene dotplot
- Reviewers will ask

---

## For Poster: What's Usable Now

1. **Data Story** (WORKING):
   - UMAP/PHATE embeddings by stage
   - Cell type composition
   - Spatial deconvolution maps
   - IL1B expression patterns

2. **Biological Insight** (WORKING):
   - IL1B association with stage
   - Clonal evolution patterns
   - Spatial transition zones
   - Phase portraits (but may be synthetic?)

3. **Model Story** (BROKEN):
   - No training curves
   - No baseline comparison
   - No ablation results
   - No held-out evaluation

---

## Action Items

1. [ ] Verify which figures use real vs synthetic data
2. [ ] Re-run model training (or find checkpoints)
3. [ ] Run LIANA analysis
4. [ ] Generate QC violin plots
5. [ ] Create marker gene dotplot
