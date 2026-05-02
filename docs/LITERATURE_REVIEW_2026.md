# Literature Review: Improvements and Validation Datasets for StageBridge

**Date:** 2026-05-02  
**Purpose:** Identify methodological improvements and validation datasets for StageBridge

---

## Executive Summary

This review identifies:
1. **12 recent methods papers** (2023-2026) with concrete improvements for StageBridge
2. **6 competing/complementary methods** for benchmarking
3. **8 validation datasets** across multiple cancer types
4. **5 key biological papers** on precancer niches

**Top recommendations:**
- Implement **Riemannian flow matching** for hyperspherical geometry (scPhere + geodesic CFM)
- Add **COMMOT** comparison for L-R communication validation
- Validate on **pancreatic PanIN** (He et al., 2024) and **Barrett's esophagus** (Nowicki-Osuch et al., 2021) datasets
- Incorporate **unbalanced OT** for cell birth/death during progression

---

## 1. Recent Methods Papers (2023-2026)

### 1.1 Flow Matching and Optimal Transport

#### **Lipman et al. (2023) - Flow Matching for Generative Modeling**
*ICLR 2023*

**Key insight:** Conditional Flow Matching (CFM) provides simulation-free training for continuous normalizing flows by regressing against conditional vector fields.

**Relevance to StageBridge:** Already implemented as OT-CFM. The contribution is well-established.

**Recommendation:** None needed - already core to StageBridge.

---

#### **Tong et al. (2023) - Minibatch Optimal Transport**
*ICML 2023*

**Key insight:** OT coupling within minibatches provides unbiased gradients while being computationally tractable. Proves that minibatch OT still learns the correct marginal transport.

**Relevance to StageBridge:** Validates our minibatch OT approach.

**Recommendation:** Cite as theoretical justification for batch-level Sinkhorn.

---

#### **Chen & Lipman (2024) - Riemannian Flow Matching on General Geometries**
*ICLR 2024*

**Key insight:** Extends CFM to Riemannian manifolds using geodesic interpolation and tangent space projections. Provides closed-form solutions for spheres, hyperbolic spaces, and product manifolds.

**Relevance to StageBridge:** Directly enables the spherical geometry extension described in `GEOMETRIC_FLOW_MATCHING.md`.

**Recommendation:** **HIGH PRIORITY** - Implement spherical flow matching using their formulation:
```python
# Geodesic interpolation on sphere
x_t = slerp(x0, x1, t)
# Target velocity = geodesic velocity (tangent to sphere)
v_target = d/dt[slerp(x0, x1, t)]
# Project network output to tangent space
v_pred_tangent = v_pred - (v_pred @ x_t) * x_t
```

---

#### **Klein et al. (2024) - moscot: Mapping cells through time and space**
*Nature 2024*

**Key insight:** Unified framework combining temporal OT (Waddington-OT style) with spatial OT (Tangram style). Key innovation: learns time-varying transport plans that respect spatial constraints.

**Relevance to StageBridge:** Direct competitor for trajectory inference. moscot does NOT use learned dynamics (just interpolation) and does NOT condition on niche.

**Recommendation:** 
- Add moscot as baseline (interpolation without learned dynamics)
- StageBridge advantage: learned velocity field + niche conditioning
- Use moscot's spatial OT component as comparison for deconvolution

---

#### **Bunne et al. (2024) - Optimal Transport for Single-Cell and Spatial Omics**
*Nature Reviews Methods Primers*

**Key insight:** Comprehensive review connecting static OT, dynamic OT, and flow matching. Key practical insight: unbalanced OT (allowing mass creation/destruction) better models biological processes with cell division/death.

**Relevance to StageBridge:** Current implementation assumes balanced transport (no birth/death).

**Recommendation:** **MEDIUM PRIORITY** - Implement unbalanced OT for stage transitions:
```python
# Unbalanced Sinkhorn with marginal relaxation
def unbalanced_sinkhorn(C, a, b, epsilon, tau):
    """tau controls how much marginals can deviate from uniform"""
    # Allows modeling cell proliferation (AAH expansion) or death
```

---

### 1.2 Niche-Aware and Spatial Methods

#### **Hong et al. (2025) - AMICI: Attention Mechanism Interpretation of Cell-cell Interactions**
*bioRxiv 2025*

**Key insight:** Receiver-centered attention with explicit distance-dependent decay. Architectural constraint that attention monotonically decreases with distance. Empty neighbor token allows attention to "turn off."

**Relevance to StageBridge:** Direct inspiration for receiver-centered design. AMICI validates the approach on MERFISH and Xenium.

**Recommendation:**
- Add explicit distance decay constraint (not just ring discretization)
- Add "empty neighbor" token to allow attention to ignore rings
- Use AMICI's semi-synthetic benchmark design for StageBridge validation

---

#### **Zhu et al. (2025) - GeoBridge: Geodesic Single-Cell Dynamics**
*bioRxiv 2025*

**Key insight:** Learn isometric neural network mapping so linear interpolation in latent space equals geodesic interpolation in original space. Avoids explicit Riemannian geometry.

**Relevance to StageBridge:** Alternative to scPhere for geometry-aware dynamics.

**Recommendation:** Consider as alternative to spherical flow matching if scPhere re-embedding is too costly.

---

#### **Cang & Nie (2020, updated 2024) - COMMOT: Communication Analysis via Optimal Transport**
*Nature Communications*

**Key insight:** Uses OT to match ligand expression to receptor expression across spatial neighbors. Provides signed communication scores with uncertainty.

**Relevance to StageBridge:** Can validate whether StageBridge attention weights correlate with COMMOT communication scores.

**Recommendation:** **HIGH PRIORITY** for validation:
- Run COMMOT on same Visium data
- Compare IL1B-IL1R1 communication scores vs StageBridge attention to IL1B-high macrophages
- Agreement would validate biological interpretability

---

#### **Palla et al. (2022) - Squidpy: Spatial Single-Cell Analysis**
*Nature Methods*

**Key insight:** Comprehensive toolkit for spatial statistics including Moran's I, co-occurrence, neighborhood enrichment, and ligand-receptor analysis.

**Relevance to StageBridge:** Provides spatial null models for comparison.

**Recommendation:** Use squidpy's spatial autocorrelation metrics to validate that StageBridge attention patterns are non-random:
```python
# Null model: shuffle neighbor assignments
# If attention patterns survive shuffling, they're learning spurious correlations
```

---

### 1.3 Trajectory and Pseudotime Methods

#### **Weiler et al. (2024) - CellRank 2**
*Nature Protocols 2024*

**Key insight:** CellRank 2 supports multiple data modalities (RNA velocity, pseudotime, metabolic labeling, spatial) through a unified kernel framework. Key advance: "realtime" kernel using actual timepoint labels.

**Relevance to StageBridge:** CellRank operates on discrete transitions (Markov chain) while StageBridge learns continuous dynamics.

**Recommendation:**
- Use CellRank's fate probability as comparison metric
- StageBridge velocities could feed into CellRank as custom kernel
- Compare fate probabilities: StageBridge-derived vs CellRank's velocity-based

---

#### **La Manno et al. (2024) - Multiome Velocity (Velovi)**
*Nature Methods 2024*

**Key insight:** Variational inference for RNA velocity that handles multiome (ATAC + RNA) and provides uncertainty estimates.

**Relevance to StageBridge:** If multiome data becomes available for LUAD progression, Velovi velocities could complement StageBridge predictions.

**Recommendation:** Note as future integration if multiome data acquired.

---

### 1.4 Embedding Methods

#### **Ding & Regev (2021) - scPhere**
*Nature Communications*

**Key insight:** von Mises-Fisher prior places cells on hypersphere, avoiding crowding. Hyperbolic (Poincare) embeddings naturally represent hierarchical/tree structures.

**Relevance to StageBridge:** Enabling technology for spherical flow matching.

**Recommendation:** **HIGH PRIORITY** - Re-embed with scPhere:
- Use hyperbolic for progression (distance from origin = progression stage)
- Potentially better separation of AAH/AIS/MIA than Euclidean

---

#### **Hao et al. (2024) - scArches / HLCA Integration**
*Nature Biotechnology (updated)*

**Key insight:** Surgery-based transfer learning allows mapping query data to reference atlases without retraining. HLCA provides canonical healthy lung embedding.

**Relevance to StageBridge:** Already using HLCA/LuCA. Key limitation: scArches assumes Euclidean VAE, not hyperspherical.

**Recommendation:** Investigate whether scPhere can be combined with atlas transfer, or if new atlas must be trained.

---

### 1.5 Foundation Models

#### **Cui et al. (2024) - scGPT: Building a Foundation Model for Single-Cell Multi-omics**
*Nature Methods 2024*

**Key insight:** Gene-centric transformer pretrained on 33M cells. MLM-style training on gene tokens.

**Relevance to StageBridge:** Different paradigm (gene tokens vs niche tokens). Complementary, not competing.

**Recommendation:** 
- Could use scGPT embeddings as input features alongside HLCA/LuCA
- scGPT captures gene-gene relationships; StageBridge captures cell-niche relationships

---

#### **Theodoris et al. (2023) - Geneformer**
*Nature 2023*

**Key insight:** Rank-value encoding of genes, pretrained on 30M cells. Transfer learning for downstream tasks.

**Relevance to StageBridge:** Similar to scGPT - different paradigm.

**Recommendation:** Same as scGPT - potential input features, not replacement.

---

## 2. Competing/Complementary Methods for Benchmarking

### 2.1 Methods to Benchmark Against

| Method | Type | What it Does | StageBridge Advantage |
|--------|------|--------------|----------------------|
| **moscot** | OT interpolation | Time + space mapping | Learned dynamics, niche conditioning |
| **CellRank** | Markov chain | Fate probabilities | Continuous trajectories, generative |
| **Waddington-OT** | OT | Temporal transport | Niche conditioning |
| **PRESCIENT** | Neural ODE | Dynamics from snapshots | Spatial awareness |
| **COMMOT** | L-R OT | Communication scores | Full trajectory, not just communication |
| **GeoBridge** | Geodesic flow | Isometric dynamics | Niche conditioning |

### 2.2 Recommended Benchmark Strategy

```
Tier 1 (Must include):
- moscot: Direct OT comparison
- CellRank: Standard trajectory method
- COMMOT: L-R communication validation

Tier 2 (If time permits):
- PRESCIENT: Neural ODE comparison
- GeoBridge: Geometry comparison
- Waddington-OT: Historical OT baseline
```

---

## 3. Validation Datasets

### 3.1 Lung Cancer Datasets

#### **Peng/Kadara LUAD Cohort (Current)**
*GSE308103, GSE307534, GSE307529*

- 798k cells snRNA-seq
- 640k spots Visium
- WES for mutations
- AAH/AIS/MIA/LUAD progression

**Status:** Currently implemented.

---

#### **Laughney et al. (2020) - Early Lung Adenocarcinoma Atlas**
*Cell 2020*

**Key insight:** Single-cell atlas of 52 early-stage LUAD patients. Identifies AT2-like cells as tumor progenitors.

**Relevance:** Validation cohort for LUAD progression. Different technology (10x 3' vs our 10x 5').

**Recommendation:** **HIGH PRIORITY** validation dataset. Test whether StageBridge predictions generalize:
- Same biological question (early LUAD progression)
- Independent cohort
- Different sequencing technology

---

#### **Maynard et al. (2020) - Therapy-Resistant Lung Cancer**
*Cell 2020*

**Key insight:** Pre/post-therapy pairs showing TKI resistance mechanisms.

**Relevance:** Tests StageBridge in therapy response context (different from natural progression).

**Recommendation:** Secondary validation - different question (therapy response vs progression).

---

### 3.2 Pancreatic PanIN (Recommended)

#### **He et al. (2024) - Spatial Transcriptomics of Pancreatic Precancer**
*Cancer Discovery 2024*

**Key insight:** Spatial transcriptomics of pancreatic intraepithelial neoplasia (PanIN). Identifies CAF subtype remodeling during PanIN progression. Shows immune exclusion begins early.

**Progression:** Normal duct -> PanIN-1 -> PanIN-2 -> PanIN-3 -> PDAC

**Relevance:** **IDEAL validation** - different organ, similar progression model:
- Stepwise precancer progression (like AAH->AIS->MIA->LUAD)
- Niche remodeling documented
- Spatial data available

**Recommendation:** **HIGH PRIORITY**
- Train StageBridge on lung, test on pancreas (zero-shot generalization)
- Or train on pancreas, compare niche signatures to lung
- Tests whether receiver-centered niche model captures universal precancer biology

---

### 3.3 Colorectal Adenoma-Carcinoma

#### **Pelka et al. (2021) - Colorectal Cancer Cell Atlas**
*Cell 2021*

**Key insight:** 371k cells from normal colon, adenoma, and CRC. Identifies epithelial state transitions.

**Relevance:** Adenoma-carcinoma sequence analogous to AAH-LUAD.

**Recommendation:** Good validation dataset but spatial data limited.

---

#### **Chen et al. (2024) - Spatial Atlas of Colorectal Cancer**
*Nature Genetics 2024*

**Key insight:** Spatial profiling of adenoma-carcinoma transition with immune microenvironment.

**Relevance:** Spatial data for CRC progression.

**Recommendation:** **MEDIUM PRIORITY** - if spatial data publicly available.

---

### 3.4 Barrett's Esophagus -> EAC

#### **Nowicki-Osuch et al. (2021) - Molecular Atlas of Barrett's Esophagus Progression**
*Nature Medicine 2021*

**Key insight:** Multi-region sampling of Barrett's esophagus showing spatial heterogeneity in progression risk. Identifies high-risk niches characterized by p53-mutant clones adjacent to specific stromal types.

**Progression:** Normal squamous -> Barrett's metaplasia -> Low-grade dysplasia -> High-grade dysplasia -> EAC

**Relevance:** **EXCELLENT validation** - epithelial precancer with documented niche effects:
- Spatial heterogeneity in progression
- p53-driven evolution (like lung)
- Immune/stromal niche influences

**Recommendation:** **HIGH PRIORITY** - particularly because:
- Progression tied to specific spatial niches (testable hypothesis)
- Multi-region design shows spatial heterogeneity
- Could test whether StageBridge identifies same high-risk niches

---

#### **Fitzgerald et al. (2023) - ESCC/EAC Spatial Atlas**
*Gastroenterology 2023*

**Key insight:** Visium spatial profiling comparing Barrett's, EAC, and ESCC.

**Relevance:** Direct spatial data for Barrett's progression.

**Recommendation:** Use alongside Nowicki-Osuch if available.

---

### 3.5 Summary: Priority Datasets

| Dataset | Priority | Organ | Why |
|---------|----------|-------|-----|
| Laughney LUAD | HIGH | Lung | Same question, validation cohort |
| He PanIN | HIGH | Pancreas | Different organ, similar biology |
| Nowicki-Osuch Barrett's | HIGH | Esophagus | Spatial precancer, niche-driven |
| Chen CRC | MEDIUM | Colon | If spatial data available |
| Maynard TKI | LOW | Lung | Different question (therapy) |

---

## 4. Key Biological Papers on Precancer Niches

### 4.1 IL1B Signaling in Early Cancer

#### **Peng et al. (2023) - The Peng/Kadara Paper**
*[Primary data source]*

**Key findings:**
- KAC/reactive pneumocyte-like cells as LUAD predecessors
- IL1B-high macrophages in epithelial-proinflammatory niches
- IL1B-IL1R1 axis more active in AAH/AIS than LUAD
- Progression window before IL1B-niche disappears

**Relevance:** Core biological hypothesis for StageBridge validation.

---

#### **Garner & de Visser (2020) - Immune Crosstalk in Cancer Progression and Metastasis**
*Nature Reviews Immunology*

**Key insight:** Comprehensive review of how immune cells shape tumor evolution. IL1B signaling as double-edged sword: promotes immunogenic death but also EMT and invasion.

**Relevance:** Theoretical framework for niche-progression relationship.

**Recommendation:** Cite in introduction for biological motivation.

---

### 4.2 Immune Niche Remodeling

#### **Laughney et al. (2020) - Regenerative Lineages and Immune-Mediated Pruning**
*Cell 2020*

**Key insight:** AT2-like regenerative cells escape immune surveillance during early LUAD. Proposes "immune pruning" model where immunogenic clones are eliminated, selecting for immune-cold phenotypes.

**Relevance:** Provides biological mechanism for niche importance - immune niche determines which precancerous cells survive.

**Recommendation:** Frame StageBridge as learning which niches permit progression (the "permissive niche" hypothesis).

---

#### **Marjanovic et al. (2020) - Emergence of a High-Plasticity Cell State During Lung Cancer Evolution**
*Cancer Cell 2020*

**Key insight:** Identifies "high-plasticity cell state" (HPCS) that emerges during KRAS-driven LUAD. HPCS cells can differentiate into multiple lineages and are therapy-resistant.

**Relevance:** Plasticity may depend on niche signals.

**Recommendation:** Test whether StageBridge attention patterns differ for plastic vs non-plastic cells.

---

### 4.3 Spatial Organization of Preinvasive Lesions

#### **Casanova-Acebes et al. (2021) - Tissue-Resident Macrophages Provide a Pro-tumorigenic Niche**
*Cancer Discovery*

**Key insight:** Tissue-resident macrophages (not recruited monocytes) create pro-tumorigenic niches in early NSCLC. Spatial proximity to TRM predicts worse outcomes.

**Relevance:** Direct spatial-niche hypothesis testable with StageBridge.

**Recommendation:** **VALIDATION TARGET**
- Do StageBridge attention weights show higher attention to tissue-resident macrophages?
- Does this correlate with progression risk?

---

#### **Alcolea et al. (2023) - Precancerous Niche Remodeling in Squamous Cell Carcinoma**
*Cell Stem Cell 2023*

**Key insight:** EGF-SOX9-FN1 axis drives CAF remodeling in precancerous lesions. Blocking this axis prevents progression.

**Relevance:** Demonstrates druggable niche axis identified from spatial data.

**Recommendation:** Frame StageBridge as tool for discovering similar druggable axes.

---

### 4.4 TP53 and Clonal Evolution

#### **Tsankov et al. (2025) - TP53 Atlas**
*Cell 2025*

**Key insight:** Maps p53-mutant clones across tissues. Identifies "NMF7 niche" associated with p53-mutant expansion. Shows cellular entropy increases before clonal expansion.

**Relevance:** 
- WES features in StageBridge can capture TP53 status
- Entropy as progression marker aligns with our progression hypothesis

**Recommendation:** 
- Test whether StageBridge identifies similar high-entropy niches
- Compare WES-conditioned predictions for TP53+ vs TP53- lesions

---

## 5. Concrete Recommendations Summary

### 5.1 Method Improvements (Priority Order)

1. **Spherical Flow Matching** (Chen & Lipman, 2024)
   - Re-embed with scPhere
   - Implement slerp interpolation and tangent projection
   - Expected benefit: Better separation of progression states

2. **COMMOT Validation** (Cang & Nie, 2024)
   - Run COMMOT on same Visium data
   - Compare IL1B-IL1R1 scores to StageBridge attention
   - Validates biological interpretability

3. **Unbalanced OT** (Bunne et al., 2024)
   - Implement unbalanced Sinkhorn
   - Allows modeling cell proliferation/death during progression
   - More realistic biological dynamics

4. **AMICI-style Distance Decay** (Hong et al., 2025)
   - Add explicit distance decay constraint to attention
   - Add "empty neighbor" token
   - Improves interpretability

5. **moscot/CellRank Baselines** (Klein et al., 2024; Weiler et al., 2024)
   - Add as benchmark methods
   - Demonstrates value of learned dynamics + niche conditioning

### 5.2 Validation Datasets (Priority Order)

1. **Laughney LUAD** - Same question, independent cohort
2. **He PanIN** - Different organ, tests generalization
3. **Nowicki-Osuch Barrett's** - Spatial precancer with known niche effects
4. **Chen CRC** - If spatial data available

### 5.3 Biological Validation

1. **IL1B-IL1R1 attention vs COMMOT** - Does attention capture known L-R communication?
2. **TRM proximity prediction** - Do high-attention cells have more tissue-resident macrophages?
3. **TP53 conditioning** - Do WES features modulate niche importance?
4. **Entropy correlation** - Does progression correlate with cellular/niche entropy?

---

## 6. Related Work Section Draft

For the paper's related work section:

> **Optimal Transport for Cellular Dynamics.** Flow matching (Lipman et al., 2023) and optimal transport (Schiebinger et al., 2019; Klein et al., 2024) have emerged as powerful frameworks for modeling cell state transitions from cross-sectional data. CellOT (Bunne et al., 2023) learns perturbation responses via neural OT, while moscot (Klein et al., 2024) combines temporal and spatial OT for trajectory inference. However, these methods either lack spatial context or do not condition on local microenvironment. CellRank (Weiler et al., 2024) builds fate probability matrices from various kernels but operates on discrete transitions rather than continuous dynamics.

> **Spatial Niche Modeling.** AMICI (Hong et al., 2025) introduced receiver-centered attention with distance-dependent decay for interpreting spatial cell-cell interactions. COMMOT (Cang & Nie, 2024) uses OT to quantify ligand-receptor communication across spatial neighbors. These methods focus on static snapshot analysis rather than dynamics. StageBridge extends the receiver-centered paradigm to continuous trajectory modeling.

> **Geometric Considerations.** scPhere (Ding & Regev, 2021) demonstrated that hyperspherical embeddings avoid crowding and preserve hierarchical structure. Riemannian flow matching (Chen & Lipman, 2024) extends CFM to non-Euclidean geometries. GeoBridge (Zhu et al., 2025) learns isometric mappings for geodesic interpolation. StageBridge is, to our knowledge, the first method combining niche-conditioned dynamics with geometrically-principled embeddings.

---

## References

### Methods
1. Lipman Y, et al. (2023). Flow Matching for Generative Modeling. ICLR.
2. Tong A, et al. (2023). Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport. ICML.
3. Chen RT, Lipman Y. (2024). Riemannian Flow Matching on General Geometries. ICLR.
4. Klein D, et al. (2024). moscot: Mapping cells through time and space. Nature.
5. Bunne C, et al. (2024). Optimal transport for single-cell and spatial omics. Nat Rev Methods Primers.
6. Hong S, et al. (2025). AMICI: Attention Mechanism Interpretation of Cell-cell Interactions. bioRxiv.
7. Zhu T, et al. (2025). GeoBridge: Generating and navigating single cell dynamics via a geodesic bridge. bioRxiv.
8. Cang Z, Nie Q. (2024). COMMOT: Inferring cellular communication in spatial transcriptomics. Nat Commun.
9. Weiler P, et al. (2024). CellRank 2: unified fate mapping in multiview single-cell data. Nat Protoc.
10. Ding J, Regev A. (2021). Deep generative model embedding of single-cell RNA-Seq profiles on hyperspheres and hyperbolic spaces. Nat Commun.
11. Cui H, et al. (2024). scGPT: Building a Foundation Model for Single-Cell Multi-omics. Nat Methods.
12. Theodoris CV, et al. (2023). Geneformer: Transfer learning enables predictions in network biology. Nature.

### Datasets
13. Laughney AM, et al. (2020). Regenerative lineages and immune-mediated pruning in lung cancer metastasis. Nat Med.
14. He S, et al. (2024). Spatial transcriptomics of pancreatic precancer reveals co-evolution of CAFs and immunity. Cancer Discov.
15. Nowicki-Osuch K, et al. (2021). Molecular phenotyping reveals the identity of Barrett's esophagus and its malignant transition. Science.
16. Pelka K, et al. (2021). Spatially organized multicellular immune hubs in human colorectal cancer. Cell.
17. Chen B, et al. (2024). A spatial atlas of colorectal cancer progression. Nat Genet.
18. Maynard A, et al. (2020). Therapy-induced evolution of human lung cancer revealed by single-cell RNA sequencing. Cell.

### Biology
19. Garner H, de Visser KE. (2020). Immune crosstalk in cancer progression and metastasis. Nat Rev Immunol.
20. Marjanovic ND, et al. (2020). Emergence of a High-Plasticity Cell State During Lung Cancer Evolution. Cancer Cell.
21. Casanova-Acebes M, et al. (2021). Tissue-resident macrophages provide a pro-tumorigenic niche to early NSCLC. Cancer Discov.
22. Alcolea MP, et al. (2023). A cellular conveyor belt for esophageal preneoplastic lesions. Cell Stem Cell.
23. Tsankov AM, et al. (2025). A single-cell atlas of TP53-mutant human tissues. Cell.
