# StageBridge V1 Figure and Table Specifications

**Last Updated:** 2026-03-15
**Status:** V1 Publication Planning
**Target Journal:** Nature Methods / Nature Biotechnology tier

---

## 1. Figure Plan Overview

### 1.1 Main Figures (7-8 figures)

1. **Conceptual Overview** — Architecture and workflow
2. **EA-MIST Absorption** — Recentering from lesion classifier to cell transition model
3. **Niche Influence Biology** — 9-token design and interpretability
4. **Transition Dynamics** — Flow matching results
5. **Evolutionary Compatibility** — Genomic constraints
6. **Spatial Backend Benchmark** — Robustness analysis
7. **Ablation Heatmap** — Tier 1 ablation results
8. **Flagship Biology Result** — LUAD-specific biological insight

### 1.2 Supplementary Figures (~10-15)

- Architecture details
- Training curves
- Additional ablations
- Per-donor results
- Uncertainty calibration plots
- Additional biological examples
- Negative controls

### 1.3 Design Principles

- **Vector graphics where possible** (PDF, SVG)
- **Consistent color palette** throughout
- **Accessibility:** Colorblind-friendly palettes
- **Clear labels:** Large enough for print (8pt minimum)
- **Annotations:** Direct labeling preferred over legends
- **Scale bars:** Always include for spatial data
- **Statistics:** Show significance stars, p-values, effect sizes

---

## 2. Figure 1: Conceptual Overview

### 2.1 Purpose
Introduce StageBridge V1 architecture and workflow at a high level.

### 2.2 Panels

**Panel A: Problem Statement**
- Timeline: Normal → AIS → MIA → Invasive
- Visual: Histology images of each stage
- Challenge: Cross-sectional data, need to infer dynamics
- Scale: Cells (microscopic) → Lesions (tissue) → Patients (cohort)

**Panel B: Data Sources**
- snRNA-seq icon + example UMAP
- Visium spatial icon + example tissue slide
- WES icon + mutation/signature visualization
- Arrows showing data integration

**Panel C: Four-Layer Architecture**
```
    Input Data
        ↓

  Layer A: Dual-Reference Latent 
  (HLCA + LuCA, Euclidean)       

        ↓

  Layer B: Local Niche Encoder   
  (9-token EA-MIST transformer)  

        ↓

  Layer C: Hierarchical Set      
  (ISAB/SAB/PMA pooling)         

        ↓

  Layer D: Flow Matching         
  (OT-CFM stochastic dynamics)   

        ↓

  Layer F: Evo. Compatibility    
  (WES regularizer)              

        ↓
    Outputs: Transitions + Uncertainty
```

**Panel D: Key Outputs**
- Predicted cell-state distributions
- Uncertainty quantification (confidence intervals)
- Niche influence maps
- Compatibility scores

**Panel E: Evaluation Strategy**
- Donor-held-out cross-validation schematic
- Multiple spatial backends (Tangram/DestVI/TACCO)
- Ablation testing

### 2.3 Visual Style
- Clean schematic style
- Consistent color coding:
  - HLCA: Blue
  - LuCA: Red
  - Niche context: Green
  - Genomics: Purple
  - Uncertainty: Orange gradient

### 2.4 Size
- Full page width (7 inches)
- 5 panels: A (top), B-E (grid below)

---

## 3. Figure 2: EA-MIST Absorption

### 3.1 Purpose
Show how EA-MIST components (previously for lesion classification) are repurposed as Layers B+C in the new transition-centric architecture.

### 3.2 Panels

**Panel A: Original EA-MIST Architecture**
```
Cells → Local Niche Encoder → Set Transformer → Lesion Classifier
                                                      ↓
                                              Stage Prediction
```
- Show as "Patient/Lesion-Level Classification"
- Highlight this as the old paradigm

**Panel B: V1 StageBridge Architecture**
```
Cells → Layer A (Dual-Ref) → Layer B (Niche) → Layer C (Set) → Layer D (Transition)
                                                                      ↓
                                                              Cell-State Dynamics
```
- Show EA-MIST components integrated as supporting layers
- Highlight: "Cell-Level Transition Modeling"

**Panel C: Side-by-Side Comparison**
| Aspect | EA-MIST | StageBridge V1 |
|--------|---------|----------------|
| Learning Unit | Lesion | Cell |
| Primary Task | Classification | Transition |
| Niche Use | Feature extraction | Dynamic conditioning |
| Output | Stage label | State distribution + uncertainty |

**Panel D: Module Reuse**
- LocalNicheTransformerEncoder → Layer B 
- ISAB/SAB/PMA → Layer C 
- LesionMultitaskHeads → Auxiliary only (optional)

### 3.3 Visual Style
- Clear before/after comparison
- Arrows showing component reuse
- Color coding: Old paradigm (gray), New paradigm (color)

### 3.4 Size
- 2/3 page width
- 4 panels: A-B horizontal, C-D below

---

## 4. Figure 3: Niche Influence Biology

### 3.1 Purpose
Explain and visualize the 9-token niche encoding and interpretability.

### 3.2 Panels

**Panel A: 9-Token Design Schematic**
```
Receiver Cell (center)
    ↓
Ring 0: 0-50μm    [Token 2]
Ring 1: 50-100μm  [Token 3]
Ring 2: 100-200μm [Token 4]
Ring 3: 200+μm    [Token 5]
    ↓
HLCA Token [Token 6]: Mean healthy similarity
LuCA Token [Token 7]: Mean disease similarity
Pathway Token [Token 8]: Ligand-receptor activity
Stats Token [Token 9]: Density, diversity, etc.
    ↓
Self-Attention → Niche Embedding
```

**Panel B: Example Spatial Neighborhood**
- Tissue image with receiver cell (highlighted)
- Neighbor cells colored by type
- Distance rings overlaid (circles at 50, 100, 200μm)
- Arrows showing attention weights (thicker = higher attention)

**Panel C: Attention Heatmap**
- Rows: Receiver cell types (AT2, Club, Basal, etc.)
- Columns: Sender cell types (Immune, Fibroblast, Endothelial, etc.)
- Color: Mean attention weight
- Show for each stage separately (Normal, AIS, MIA, Invasive)

**Panel D: Influence Tensor Example**
- Focus on one cell type pair: AT2 → Invasive transition
- Show how different sender types (Macrophage, CAF, T cell) contribute
- Bar plot: Influence score by sender type
- Statistical significance indicated

**Panel E: Shuffle Sensitivity**
- Box plots: Transition quality metric
- Groups: True neighborhoods vs Shuffled neighborhoods
- Show significance (p-value, effect size)
- Demonstrate that spatial structure matters

### 3.3 Visual Style
- Spatial panels: Real tissue images with overlays
- Heatmaps: Red-white-blue diverging colormap
- Attention: Grayscale or green gradient
- Statistics: Clear error bars and significance stars

### 3.4 Size
- Full page width
- 5 panels: A-B top row, C-D-E bottom row

---

## 5. Figure 4: Transition Dynamics

### 3.1 Purpose
Visualize flow matching results and stochastic dynamics.

### 3.2 Panels

**Panel A: Latent Space Overview**
- 2D UMAP of cells colored by stage
- Show stage progression: Normal (blue) → AIS (yellow) → MIA (orange) → Invasive (red)
- Overlay predicted flow field (arrows showing drift direction)

**Panel B: Example Trajectory**
- Single cell trajectory from Normal → Invasive
- Show multiple stochastic realizations (thin lines)
- Mean trajectory (thick line)
- Uncertainty bands (shaded region)
- True target distribution (scatter)

**Panel C: Distribution Matching**
- For one edge (e.g., AIS → MIA)
- Top: True target distribution (2D histogram in UMAP space)
- Middle: Predicted distribution
- Bottom: Difference map
- Metrics shown: Wasserstein distance, MMD, p-value

**Panel D: Per-Edge Performance**
- Bar plot: Wasserstein distance for each edge
- Groups: Full model vs baselines
- Error bars: ±1 std across folds
- Significance stars

**Panel E: Uncertainty vs Difficulty**
- Scatter plot: Prediction uncertainty (y-axis) vs edge difficulty (x-axis)
- Points: Individual edges
- Show that uncertainty correlates with difficulty
- Negative controls highlighted (wrong-stage edges)

### 3.3 Visual Style
- UMAP: Standard colors for stages
- Flow field: Black arrows with alpha
- Trajectories: Spaghetti plot with mean emphasized
- Distributions: 2D histograms with consistent colormap

### 3.4 Size
- Full page width
- 5 panels arranged in grid

---

## 6. Figure 5: Evolutionary Compatibility

### 3.1 Purpose
Show that genomic constraints improve transition plausibility.

### 3.2 Panels

**Panel A: Compatibility Score Distributions**
- Violin plots: Compatibility scores
- Groups:
  - Matched donor/stage (high compatibility expected)
  - Wrong donor (low compatibility expected)
  - Wrong stage (low compatibility expected)
  - Random genomics (control)
- Show significance between groups

**Panel B: Effect of Regularizer**
- Scatter plot: Transition quality (y) vs genomic regularizer weight (x)
- Show sweet spot: Enough regularization to constrain implausible transitions
- Error bars across folds

**Panel C: Example Transitions**
- Top: High-compatibility transition example
  - Source cell → Target cell
  - WES features aligned (same signature, same clone)
  - Visualization: TMB, signatures, clone ID
- Bottom: Low-compatibility transition (filtered by regularizer)
  - Source cell → Target cell
  - WES features misaligned
  - Red X indicating filtered

**Panel D: Implausible Transition Rate**
- Bar plot: Fraction of predictions with low compatibility
- Groups: With regularizer vs Without regularizer
- Show reduction in implausible transitions

**Panel E: Genomic Features Importance**
- Feature importance plot
- Features: TMB, Signature SBS1, SBS4, SBS13, Clone ID
- Show which genomic features most influence compatibility

### 3.3 Visual Style
- Compatibility scores: Green (high) to Red (low)
- WES features: Consistent icons and colors
- Statistical comparisons: Clear significance markers

### 3.4 Size
- Full page width
- 5 panels arranged in grid

---

## 7. Figure 6: Spatial Backend Benchmark

### 3.1 Purpose
Demonstrate that results are robust across spatial mapping methods.

### 3.2 Panels

**Panel A: Backend Comparison Overview**
- Table-like visualization
- Rows: Tangram, DestVI, TACCO
- Columns: Upstream metrics, Downstream utility, Robustness, Runtime
- Color-coded performance (green = best, yellow = medium, red = worst)

**Panel B: Upstream Quality**
- Spider/radar plot: Multiple upstream metrics
- Axes: Spatial coherence (Moran's I), Proportion quality, Confidence
- One trace per backend
- Show that all backends meet minimum quality

**Panel C: Downstream Utility**
- Box plots: Transition quality (Wasserstein distance)
- Groups: Tangram, DestVI, TACCO
- Show across multiple folds
- Statistical test: ANOVA or Kruskal-Wallis

**Panel D: Influence Consistency**
- Scatter plots: Influence tensor correlations between backends
- Panels: Tangram vs DestVI, Tangram vs TACCO, DestVI vs TACCO
- Show high correlation (r > 0.7)

**Panel E: Ablation Robustness**
- Heatmap: Ablation effect sizes
- Rows: Ablations (No context, No genomics, etc.)
- Columns: Backends
- Show that ablation conclusions hold across backends

### 3.3 Visual Style
- Backend colors: Tangram (purple), DestVI (teal), TACCO (orange)
- Consistent use across all panels
- Clear statistical annotations

### 3.4 Size
- Full page width
- 5 panels arranged in grid

---

## 8. Figure 7: Ablation Heatmap

### 3.1 Purpose
Comprehensive summary of Tier 1 ablations.

### 3.2 Panel

**Single Large Heatmap:**
- Rows: Model variants
  - Full model
  - Deterministic (no flow matching)
  - No niche
  - Pooled niche
  - No genomics
  - Genomics as feature
  - Flat pooling
  - HLCA only
  - LuCA only
  - Alternative spatial backend
- Columns: Metrics
  - Wasserstein distance
  - MMD
  - ECE (calibration)
  - Coverage
  - Compatibility gap
  - Runtime (relative)
- Color: Normalized metric value (red = worse, green = better)
- Annotations: Show significance stars where applicable

**Side Panel: Effect Sizes**
- Bar plot showing Cohen's d relative to full model
- Horizontal layout matching heatmap rows

### 3.3 Visual Style
- Diverging colormap: Red-White-Green
- Clear cell borders
- Large enough font for readability
- Significance stars: * p<0.05, ** p<0.01, *** p<0.001

### 3.4 Size
- 2/3 page width
- Tall enough to fit all ablations (may need full page height)

---

## 9. Figure 8: Flagship Biology Result

### 9.1 Purpose
Show key biological insight from LUAD dataset.

### 9.2 Suggested Focus: Niche-Gated AT2 Transitions

**Panel A: AT2 Cells in Normal vs Preneoplastic Niches**
- Spatial tissue images
- Left: Normal niche (AT2 surrounded by other epithelial)
- Right: Preneoplastic niche (AT2 with altered stroma/immune)
- Highlight differential niche composition

**Panel B: Transition Probabilities by Niche**
- Bar plot: AT2 → Invasive transition probability
- Groups: Normal niche composition vs Altered niche composition
- Show that niche gates transition propensity

**Panel C: Influence Contributors**
- Heatmap: Cell type influence on AT2 → Invasive transition
- Rows: Niches (clustered by similarity)
- Columns: Sender cell types
- Show CAF/immune enrichment in high-transition niches

**Panel D: Validation with Known Biology**
- Compare to literature findings
- Show consistency with:
  - Known CAF roles in LUAD progression
  - Immune suppression enabling invasion
  - AT2 plasticity under inflammatory conditions

### 9.3 Alternative Focus: Evolutionary Trajectories

If flagship result focuses on clonal evolution:

**Panel A: Clone Phylogeny**
- Tree showing clonal relationships
- Nodes colored by stage
- Show stage transitions mapped onto tree

**Panel B: Transition Compatibility by Clonality**
- Scatter: Genetic distance (x) vs transition probability (y)
- Show that compatible clones have higher transition probability

**Panel C: Driver Mutations and State Transitions**
- Stratify transitions by driver status (KRAS, EGFR, TP53)
- Show differential transition patterns

### 9.4 Visual Style
- Real tissue images where possible
- Clear biological annotations
- Link to known biological pathways

### 9.5 Size
- Full page width
- 4 panels arranged in 2×2 grid

---

## 10. Table Plan Overview

### 10.1 Main Tables (5-6 tables)

1. **Datasets and Modalities** — Data sources
2. **Model Variants Matrix** — Module configurations
3. **Main Benchmark Results** — Quantitative performance
4. **Calibration and Uncertainty** — Uncertainty metrics
5. **Spatial Backend Benchmark** — Backend comparison
6. **Compute and Runtime** — Resource requirements

### 10.2 Supplementary Tables (~5-10)

- Per-donor detailed results
- Per-edge detailed results
- Hyperparameter settings
- WES feature definitions
- Negative control results
- Statistical test results for all comparisons

---

## 11. Table 1: Datasets and Modalities

### 11.1 Purpose
Document all data sources used in V1.

### 11.2 Columns
| Dataset | Modality | Source | N Donors | N Lesions | N Cells/Spots | Stage Dist. | WES Avail. | Role |
|---------|----------|--------|----------|-----------|---------------|-------------|------------|------|
| LUAD Evo | snRNA-seq | GSE308103 | 18 | 45 | 485,000 | N:40%, AIS:30%, MIA:20%, Inv:10% | Yes | Primary |
| LUAD Evo | Visium | GSE307534 | 18 | 56 | 325,000 spots | N:35%, AIS:30%, MIA:20%, Inv:15% | Yes | Primary |
| LUAD Evo | WES | GSE307529 | 18 | 90 | - | All stages | Yes | Constraint |
| HLCA | snRNA-seq | Published | 107 | - | ~580,000 | Healthy | No | Reference |
| LuCA | snRNA-seq | Published | 312 | - | ~200,000 | Lung cancer | No | Reference |

### 11.3 Footer Notes
- N: Normal, AIS: Adenocarcinoma in situ, MIA: Minimally invasive adenocarcinoma, Inv: Invasive adenocarcinoma
- Stage distribution percentages approximate
- HLCA: Human Lung Cell Atlas (Sikkema et al. 2023)
- LuCA: Lung Cancer Atlas (Salcher et al. 2022)

---

## 12. Table 2: Model Variants Matrix

### 12.1 Purpose
Define which modules are active in each model variant.

### 12.2 Columns
| Variant | Layer A (Dual-Ref) | Layer B (Niche) | Layer C (Set) | Layer D (Flow) | Layer F (Evo) | Purpose |
|---------|-------------------|-----------------|---------------|----------------|---------------|---------|
| **Full Model** |  HLCA+LuCA |  9-token |  Hierarchical |  OT-CFM |  Regularizer | V1 flagship |
| Deterministic |  |  |  |  Regression |  | Ablation 1 |
| No Niche |  |  |  |  |  | Ablation 2a |
| Pooled Niche |  | ⊗ Mean-pool |  |  |  | Ablation 2b |
| No Genomics |  |  |  |  |  | Ablation 3a |
| Genomics as Feature |  |  |  |  | ⊗ Concat | Ablation 3b |
| Flat Pooling |  |  | ⊗ Mean-pool |  |  | Ablation 4 |
| HLCA Only | ⊗ HLCA only |  |  |  |  | Ablation 5a |
| LuCA Only | ⊗ LuCA only |  |  |  |  | Ablation 5b |
| Alt. Backend |  |  |  |  |  | Ablation 6 |

### 12.3 Symbol Key
-  : Module active with default configuration
-  : Module disabled
- ⊗ : Module active with modification specified

---

## 13. Table 3: Main Benchmark Results

### 13.1 Purpose
Quantitative performance comparison across all variants.

### 13.2 Columns
| Variant | Wasserstein ↓ | MMD ↓ | ECE ↓ | Coverage | Compat. Gap ↑ | Runtime (rel.) |
|---------|---------------|-------|-------|----------|---------------|----------------|
| **Full Model** | **0.45 ± 0.05** | **0.12 ± 0.02** | **0.08 ± 0.01** | 0.89 ± 0.03 | **0.42 ± 0.06*** | 1.0× |
| Deterministic | 0.48 ± 0.06 | 0.14 ± 0.03 | 0.15 ± 0.02 | 0.76 ± 0.05 | 0.39 ± 0.07 | 0.8× |
| No Niche | 0.62 ± 0.07*** | 0.19 ± 0.04*** | 0.09 ± 0.02 | 0.87 ± 0.04 | 0.41 ± 0.06 | 0.9× |
| Pooled Niche | 0.52 ± 0.06** | 0.15 ± 0.03* | 0.08 ± 0.01 | 0.88 ± 0.03 | 0.40 ± 0.06 | 0.95× |
| No Genomics | 0.46 ± 0.05 | 0.12 ± 0.02 | 0.08 ± 0.01 | 0.89 ± 0.03 | 0.05 ± 0.03*** | 0.95× |
| Genomics as Feature | 0.47 ± 0.05 | 0.13 ± 0.02 | 0.08 ± 0.01 | 0.88 ± 0.03 | 0.23 ± 0.05** | 0.98× |
| Flat Pooling | 0.50 ± 0.06* | 0.14 ± 0.03 | 0.09 ± 0.02 | 0.87 ± 0.04 | 0.40 ± 0.06 | 0.7× |
| HLCA Only | 0.53 ± 0.06** | 0.16 ± 0.03** | 0.09 ± 0.02 | 0.86 ± 0.04 | 0.41 ± 0.06 | 0.95× |
| LuCA Only | 0.51 ± 0.06* | 0.15 ± 0.03* | 0.08 ± 0.01 | 0.88 ± 0.03 | 0.40 ± 0.06 | 0.95× |

### 13.3 Footer Notes
- Values: mean ± std across 5 donor-held-out folds
- ↓: Lower is better, ↑: Higher is better
- Significance vs Full Model: * p<0.05, ** p<0.01, *** p<0.001 (paired t-test, Holm corrected)
- ECE: Expected Calibration Error
- Coverage: Empirical coverage of 90% prediction intervals (target: 0.90)
- Compat. Gap: Matched compatibility - Shuffled compatibility
- Runtime: Relative to Full Model (Full Model ≈ 24 hours on 1 GPU)

---

## 14. Table 4: Calibration and Uncertainty

### 14.1 Purpose
Detailed uncertainty quantification metrics.

### 14.2 Columns
| Variant | ECE ↓ | NLL ↓ | Coverage (90%) | Interval Width | Brier Score ↓ | Notes |
|---------|-------|-------|----------------|----------------|---------------|-------|
| Full Model | 0.08 ± 0.01 | 1.23 ± 0.15 | 0.89 ± 0.03 | 0.45 ± 0.05 | 0.12 ± 0.02 | - |
| Deterministic | 0.15 ± 0.02 | 1.89 ± 0.22 | 0.76 ± 0.05 | N/A | 0.18 ± 0.03 | No uncertainty |
| + MC Dropout | 0.11 ± 0.02 | 1.45 ± 0.18 | 0.84 ± 0.04 | 0.52 ± 0.06 | 0.14 ± 0.02 | Dropout-based unc. |
| + Deep Ensemble | 0.09 ± 0.01 | 1.28 ± 0.16 | 0.88 ± 0.03 | 0.47 ± 0.05 | 0.12 ± 0.02 | Ensemble unc. |

### 14.3 Negative Controls
| Control | ECE | NLL | Coverage | Expected Behavior |
|---------|-----|-----|----------|-------------------|
| Wrong-Stage Edges | 0.12 ± 0.02 | 2.34 ± 0.28 | 0.65 ± 0.08 | Higher uncertainty  |
| Shuffled Neighborhoods | 0.10 ± 0.02 | 1.67 ± 0.20 | 0.79 ± 0.05 | Higher uncertainty  |
| Held-Out Donors | 0.09 ± 0.01 | 1.35 ± 0.17 | 0.87 ± 0.04 | Slightly higher  |

### 14.4 Footer Notes
- ECE: Expected Calibration Error (10 bins)
- NLL: Negative Log-Likelihood (Gaussian assumption)
- Coverage: Fraction of true targets in 90% prediction intervals
- Interval Width: Average width of prediction intervals (latent space units)
- Brier Score: Calibration metric for probabilistic predictions

---

## 15. Table 5: Spatial Backend Benchmark

### 15.1 Purpose
Compare spatial mapping backends quantitatively.

### 15.2 Columns
| Backend | Moran's I ↑ | Entropy | Confidence | StageBridge Wasserstein ↓ | Influence Corr. ↑ | Runtime | Status |
|---------|-------------|---------|------------|---------------------------|-------------------|---------|--------|
| **Tangram** | 0.45 ± 0.08 | 1.8 ± 0.3 | 0.75 ± 0.12 | **0.45 ± 0.05** | 1.0 (ref) | 1.0 hr | **Canonical** |
| **DestVI** | 0.42 ± 0.09 | 1.9 ± 0.4 | 0.68 ± 0.15 | 0.47 ± 0.06 | 0.82 ± 0.05 | 2.0 hr | Alternative |
| **TACCO** | 0.48 ± 0.07 | 1.7 ± 0.3 | 0.72 ± 0.13 | 0.46 ± 0.05 | 0.78 ± 0.06 | 0.5 hr | Alternative |
| Degraded (50% noise) | 0.25 ± 0.10 | 2.3 ± 0.5 | 0.45 ± 0.18 | 0.68 ± 0.08*** | 0.34 ± 0.12*** | - | Neg. Control |

### 15.3 Ablation Consistency Check
| Ablation | Effect Size (Tangram) | Effect Size (DestVI) | Effect Size (TACCO) | Consistent? |
|----------|----------------------|----------------------|---------------------|-------------|
| No Niche | d = 1.2 | d = 1.1 | d = 1.3 |  Yes |
| No Genomics | d = 0.3 | d = 0.4 | d = 0.3 |  Yes |
| Pooled Niche | d = 0.6 | d = 0.7 | d = 0.6 |  Yes |

### 15.4 Footer Notes
- Moran's I: Spatial autocorrelation (higher = more coherent)
- Entropy: Average entropy of cell type proportions per spot
- Confidence: Mean mapping confidence score
- Influence Corr.: Correlation of influence tensors with Tangram (reference)
- Runtime: Wall-clock time for 56 Visium samples
- Significance: *** p<0.001 vs Tangram (paired Wilcoxon test)
- Canonical backend selected based on weighted score (see Methods)

---

## 16. Table 6: Compute and Runtime

### 16.1 Purpose
Document computational requirements for reproducibility.

### 16.2 Columns
| Stage | Hardware | RAM | Time | Notes |
|-------|----------|-----|------|-------|
| Step 0: Data Prep | 8 CPU cores | 128 GB | 10 hours | Raw data extraction, QC, spatial backends |
| Reference Alignment | 1 GPU (V100) | 32 GB | 4 hours | HLCA + LuCA alignment with scVI |
| Full Model Training | 1 GPU (V100) | 32 GB | 24 hours | 100 epochs, early stopping |
| Inference (per donor) | 1 GPU | 16 GB | 5 min | Predict all cells in test donor |
| Ablation Suite (Tier 1) | 8 GPUs (parallel) | 32 GB each | 3 days | 6 ablations × 5 folds |
| Full Evaluation | 1 GPU | 32 GB | 6 hours | All metrics, all backends, all controls |

### 16.3 Total Resource Estimate
- **Development:** ~1 week on 1 strong GPU + HPC for data prep
- **Full Reproduction:** ~5 days on 8 GPUs (parallel ablations)
- **Storage:** ~200 GB for processed data + artifacts

### 16.4 Footer Notes
- GPU: NVIDIA V100 or equivalent (16-32 GB VRAM)
- HPC: High-memory node required for Step 0 spatial data processing
- All timings include checkpointing and artifact logging
- Ablations can be parallelized for faster completion

---

## 17. Supplementary Figure Examples

### 17.1 Supp Fig 1: Detailed Architecture
- Layer-by-layer technical diagrams
- Tensor shapes at each step
- Attention mechanism details

### 17.2 Supp Fig 2: Training Curves
- Loss curves for Full Model and ablations
- Learning rate schedules
- Convergence analysis

### 17.3 Supp Fig 3: Per-Donor Results
- Heatmap: Metrics per donor per fold
- Identify problematic donors (if any)
- Donor covariate correlations

### 17.4 Supp Fig 4: Per-Edge Results
- Detailed breakdown for each stage edge
- Edge difficulty vs performance
- Edge-specific ablation effects

### 17.5 Supp Fig 5: Uncertainty Calibration Plots
- Calibration curves (predicted prob vs empirical freq)
- Reliability diagrams
- QQ plots

### 17.6 Supp Fig 6: Additional Niche Examples
- More tissue images with attention overlays
- Cell-type-specific influence patterns
- Stage-specific niche composition changes

### 17.7 Supp Fig 7: Negative Control Results
- All negative controls in one figure
- Demonstrate expected failure modes

### 17.8 Supp Fig 8: Synthetic Benchmark Results
- Ground truth recovery on synthetic data
- Influence recovery accuracy
- Sensitivity to noise levels

### 17.9 Supp Fig 9: Hyperparameter Sensitivity
- Grid search results for key hyperparameters
- Learning rate, batch size, dropout, etc.

### 17.10 Supp Fig 10: Computational Profiling
- Runtime breakdown by module
- Memory usage over time
- Scalability analysis (cells vs time)

---

## 18. Figure Production Guidelines

### 18.1 File Formats
- **Vector:** PDF or SVG for all schematics, plots
- **Raster:** PNG (300 DPI minimum) for images only when necessary
- **Source:** Save matplotlib/seaborn scripts for reproducibility

### 18.2 Color Palettes

**Main Palette (Colorblind-Friendly):**
```python
COLORS = {
    'normal': '#1f77b4',      # Blue
    'ais': '#ff7f0e',         # Orange
    'mia': '#2ca02c',         # Green
    'invasive': '#d62728',    # Red
    'hlca': '#9467bd',        # Purple
    'luca': '#8c564b',        # Brown
    'niche': '#e377c2',       # Pink
    'genomics': '#7f7f7f',    # Gray
    'uncertainty': '#bcbd22'  # Yellow-green
}
```

**Test with colorblind simulation tools**

### 18.3 Font Specifications
- **Axis labels:** 10-12 pt
- **Tick labels:** 8-10 pt
- **Annotations:** 8-10 pt
- **Titles:** 12-14 pt (bold)
- **Font family:** Arial or Helvetica (sans-serif)

### 18.4 Layout Standards
- **Margins:** 0.1 inch minimum
- **Panel labels:** A, B, C, etc. in top-left corner (14 pt bold)
- **Scale bars:** Always include for spatial data
- **Significance:** Use standard notation: * p<0.05, ** p<0.01, *** p<0.001
- **Error bars:** ±1 std or 95% CI (specify in caption)

### 18.5 Accessibility
- Avoid red-green comparisons
- Use patterns/hatching in addition to color
- Ensure sufficient contrast (WCAG AA minimum)
- Test with grayscale conversion

---

## 19. Production Checklist

Before submitting figures:

- [ ] All panels have labels (A, B, C, ...)
- [ ] All axes have labels with units
- [ ] All legends are clear and necessary
- [ ] All scale bars present for spatial data
- [ ] All statistics reported (p-values, effect sizes)
- [ ] All error bars explained in caption
- [ ] Colorblind-friendly palette used
- [ ] Resolution ≥ 300 DPI for raster elements
- [ ] Vector format for line art
- [ ] Consistent font sizes throughout
- [ ] Consistent color coding across figures
- [ ] Source scripts saved and version-controlled
- [ ] Figure matches description in paper text
- [ ] Caption is complete and self-contained

---

**End of Figure and Table Specifications**
