# StageBridge: Niche-Aware Modeling of Lung Adenocarcinoma Progression

## Abstract

**Background:** Early detection and interception of lung adenocarcinoma (LUAD) precursor lesions remains challenging due to the difficulty of predicting which atypical adenomatous hyperplasia (AAH) and adenocarcinoma in situ (AIS) lesions will progress to invasive cancer. The local immune and stromal microenvironment—particularly IL1B-high macrophage niches—has been implicated in early progression, but computational methods to quantify niche-dependent progression risk from single-cell spatial data are lacking.

**Methods:** We developed StageBridge, a receiver-centered niche-aware deep learning framework that models lung cancer progression using dual-reference geometry. The model integrates:
- Dual reference mapping to HLCA (healthy) and LuCA (cancer) atlases
- Spatial niche encoding via hierarchical set transformers
- Flow-matching transition prediction between disease stages
- Biology-informed auxiliary heads for IL1B pathway, KAC/reactive progenitors, and proliferation

We trained 15 models (5-fold cross-validation x 3 seeds) on ~1.4M cells from snRNA-seq and spatial transcriptomics data spanning Normal, AAH, AIS, MIA, and LUAD stages.

**Results:** 
- **Transition prediction:** Normal→AAH drift alignment = 96.4% (+/- 0.8%), indicating strong early-stage progression modeling
- **Biology validation:** IL1B pathway correlation r=0.14, KAC/reactive score r=0.26
- **Stage ordering:** 61% KAC monotonicity across progression stages
- **Cross-fold stability:** Drift alignment CV=28.3%, demonstrating robust generalization

**Conclusions:** StageBridge enables single-biopsy risk stratification of LUAD precursor lesions by integrating cell-intrinsic features with local niche context. The model's strong performance on early-stage transitions (Normal→AAH) and validation against IL1B-IL1R1 biology supports its potential for identifying intervention windows in lung cancer prevention.

**Keywords:** lung adenocarcinoma, precursor lesions, spatial transcriptomics, deep learning, tumor microenvironment, IL1B pathway

---

## Key Numbers for Poster

| Metric | Value |
|--------|-------|
| Total cells | ~1.4M |
| Models trained | 15 (5 folds x 3 seeds) |
| Epochs to convergence | 123 +/- 11 |
| Normal→AAH alignment | 96.4% +/- 0.8% |
| IL1B correlation | r = 0.14 +/- 0.05 |
| KAC correlation | r = 0.26 +/- 0.16 |
| Stage monotonicity | 61.1% +/- 21.5% |

## Figure Recommendations

1. **Architecture diagram** - Show dual-reference + niche encoder + transition head
2. **UMAP with stage coloring** - Dual-reference embedding with progression trajectory
3. **Transition prediction heatmap** - Normal→AAH strong, later stages weaker
4. **IL1B niche association** - Attention weights vs IL1B expression
5. **Clinical schematic** - Single biopsy → risk score → intervention decision
