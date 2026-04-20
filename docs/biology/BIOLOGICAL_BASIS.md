# Biological Basis for StageBridge Validation

This document tracks the key biological findings from the literature that inform
StageBridge's hypothesis testing and validation modules.

---

## Core StageBridge Hypothesis

**"Cross-sectional progression becomes more identifiable when conditioned on
receiver-centered local niche context."**

The niche (local microenvironment) determines which cells progress and which
are eliminated or remain quiescent.

---

## Paper 1: Precancerous Niche Remodelling (Alcolea et al., Nature 2026)

**DOI:** 10.1038/s41586-026-10157-8

### Key Findings

1. **Niche+ vs Niche- tumors**: Only ~30% of nascent tumors have stromal
   remodeling (Niche+). These persist; Niche- tumors are eliminated.

2. **EGF-SOX9-FN1 Axis**:
   - Stressed epithelial cells ("Tumour 12" state) express:
     - AP-1/stress TFs: JUN, FOS, FOSB, ATF3, EGR1, RUNX1
     - EGF ligands: AREG, HBEGF
     - SOX9 (marks cells in niche contact)
   - EGF ligands recruit PDGFRa-low fibroblasts from lamina propria
   - Fibroblasts deposit FN1 (fibronectin) creating supportive scaffold
   - FN1+ scaffold feeds back to sustain tumor survival

3. **Pre-CAF state**: Niche fibroblasts are FN1+/aYAP+ but lack full CAF
   markers (FAP-, ACTA2-). This is a transitional "pre-CAF" state.

4. **Therapeutic targets**: Blocking EGF (gefitinib) or FN1 assembly (FUD
   peptide) prevents tumor persistence.

### Implications for StageBridge

- **Receiver-centered modeling is correct**: Cells receiving niche signals
  (SOX9+ cells in contact with fibroblasts) have different fates
- **AP-1 validation**: AP-1 marks the stress state that initiates niche formation
- **Testable prediction**: High AP-1 + high EGF ligands + SOX9 should predict
  higher model-assigned progression risk

### Gene Signatures

```python
# Tumour 12 stress state (epithelial)
TUMOUR12_STRESS_TFS = ["JUN", "FOS", "FOSB", "ATF3", "EGR1", "EGR3", "RUNX1", "MYC"]
TUMOUR12_EGF_LIGANDS = ["AREG", "HBEGF", "NRG1", "NRG2"]
SOX9_NICHE_MARKERS = ["SOX9", "KRT6A", "KRT17"]

# Precancer niche fibroblasts
PRECANCER_NICHE_MATRIX = ["FN1", "COL1A1", "COL1A2", "COL3A1"]
PRECANCER_NICHE_ACTIVATION = ["YAP1", "WWTR1", "CTGF", "CYR61"]  # Hippo pathway
# Note: Full CAF markers (FAP, ACTA2) should be LOW in pre-CAF state
```

---

## Paper 2: TP53 Cellular and Spatial Atlas (Tsankov et al., Nature Cancer 2025)

**DOI:** 10.1038/s43018-025-01053-7

### Key Findings

1. **TP53-mutant LUAD phenotype**:
   - Loss of alveolar (AT2) identity
   - Increased cellular entropy (plasticity)
   - Upregulation of cell cycle, hypoxia, glycolysis, pEMT programs
   - Decreased AT2-like signature

2. **18 malignant programs identified**, including:
   - Senescence signature
   - pEMT (partial EMT)
   - Hypoxia
   - Cell cycle (CC.G2/M)

3. **NMF7 Multicellular Niche** (hypoxic, EMT-promoting):
   - TAM.SPP1 (SPP1+ tumor-associated macrophages)
   - CAF.COLs (collagen-producing CAFs)
   - Myofibroblasts
   - Spatially enriched in tumor periphery
   - Correlated with hypoxia and EMT programs

4. **Stromal remodeling in TP53mut**:
   - Decreased pericytes and endothelial cells
   - Increased CAF.ADH1B (cytokine/immune modulation)
   - TGFB2-TGFBR2 spatial correlation increased

5. **Immune exhaustion**: T cells in TP53mut show exhausted phenotype
   (TIGIT+, CTLA4+, PD1+)

### Implications for StageBridge

- **Entropy as plasticity measure**: High entropy cells = high plasticity =
  higher progression potential
- **Niche composition matters**: NMF7 niche promotes EMT; model should learn
  that cells in NMF7-like niches have different trajectories
- **IL1B connection**: SPP1+ TAMs are IL1B producers (connects to H1.2 hypothesis)

### Gene Signatures

```python
# NMF7 niche components
NMF7_NICHE = ["SPP1", "COL1A1", "VIM", "HIF1A"]
TAM_SPP1_MARKERS = ["SPP1", "APOE", "TREM2", "CD163"]
CAF_COLS_MARKERS = ["COL1A2", "COL3A1", "TWIST1"]

# TP53mut-enriched programs
TP53MUT_PROGRAMS = ["CDKN1A", "MDM2", "BAX"]  # p53 targets (paradoxically low in mut)
PEMT_MARKERS = ["VIM", "CDH2", "SNAI1", "ZEB1"]
HYPOXIA_MARKERS = ["HIF1A", "VEGFA", "SLC2A1", "LDHA"]
```

---

## Paper 3: Progressive Plasticity in CRC Metastasis (Ganesh et al., Nature 2025)

**DOI:** 10.1038/s41586-024-08150-0

### Key Findings

1. **Metastatic progression involves dedifferentiation**:
   - Primary tumors: ISC-like (LGR5+) canonical intestinal states
   - Metastases: Non-canonical states (squamous, neuroendocrine)
   - Pathway: ISC-like → Fetal progenitor → Non-canonical

2. **Fetal progenitor intermediate**:
   - Highly plastic cell state
   - Bridges canonical and non-canonical fates
   - 14 core genes define fetal signature (including WNT-associated: TCF7, PTK7)

3. **PROX1 as lineage restriction factor**:
   - High PROX1 = maintains intestinal lineage identity
   - Loss of PROX1 = licenses non-canonical differentiation
   - Context-dependent: effect differs in primary vs metastatic cells

4. **Microenvironment determines fate**:
   - Same cells in HISC medium → canonical differentiation
   - Same cells in IGFF medium → non-canonical differentiation
   - **This is the core StageBridge hypothesis!**

### Implications for StageBridge

- **Fetal/progenitor state is key**: Cells must pass through a plastic
  intermediate to change lineage
- **Lineage restriction factors**: In lung, HOPX/NKX2-1 may play similar role
  to PROX1 (but see Paper 5 for nuance)
- **Niche determines trajectory**: The model should learn that same cell state
  in different niches leads to different outcomes

---

## Paper 4: Adaptive Genome Regulation in Cancer (Cell 2024)

### Key Findings

1. **AP-1 as master regulator of stress-adaptive chromatin**:
   - JUN, FOS, ATF family TFs
   - Remodel chromatin in response to stress
   - Enable phenotypic plasticity

2. **AP-1 targets include**:
   - Senescence regulators
   - EMT genes
   - Metabolic reprogramming genes

3. **Mechanistic link**: AP-1 is the hub where multiple stress pathways
   (JNK, ERK, p38) converge to enable cell state flexibility

### Implications for StageBridge

- **AP-1 validation is well-founded**: AP-1 activity should mark cells with
  enhanced plasticity and progression potential
- **Connects stress response to plasticity**: Cells under stress activate AP-1,
  which enables state changes

---

## Paper 5: Lineage-Specific Oncogenic Driver Intolerance (Gardner et al., Science 2024)

**DOI:** 10.1126/science.adj1415

### Key Findings (CRITICAL for understanding lineage/plasticity)

1. **Lineage-specific driver tolerance**:
   - AT2 cells: Tolerate EGFR activation → LUAD
   - AT2 cells: INTOLERANT to MYC alone (cells die or don't expand)
   - PNECs: Tolerate MYC → SCLC
   - PNECs: INTOLERANT to EGFR (impairs fitness)

2. **Histological transformation (HT) requires**:
   - Loss of original driver dependency (e.g., EGFR withdrawal)
   - Acquisition of new driver program (e.g., MYC)
   - Passage through undifferentiated "bottleneck" state

3. **The Bottleneck State** (key insight!):
   - Highly undifferentiated, stem-like
   - LOW expression of all lineage markers (AT1, AT2, PNEC, basal, secretory)
   - HIGH expression of:
     - **Tm4sf1** - stem-like marker
     - **Sox9** - airway stemness (same as precancerous niche paper!)
     - **Creb5** - neuronal plasticity
     - **Myc** and **Sox2** target genes
   - This is the "permissive state" that enables lineage conversion

4. **Genetic requirements for AT2 → SCLC transformation**:
   - Loss of EGFR dependence (necessary)
   - MYC expression (necessary but not sufficient in AT2)
   - Loss of PTEN (enables MYC tolerance via PI3K/AKT)
   - Loss of RB1 (required for full neuroendocrine phenotype)
   - Loss of TP53 (contributes but not essential for transformation)

5. **Basal cells as transformation-competent**:
   - Basal stem cells can serve as cell of origin for SCLC
   - More plastic than AT2 cells
   - May be the "bottleneck" cell type

### Critical Implications for StageBridge

**This changes how we think about lineage factors:**

- It's NOT just "loss of HOPX = plasticity"
- It's "cells must enter an undifferentiated bottleneck state to change fate"
- The bottleneck is characterized by:
  - Loss of differentiated lineage markers
  - Gain of stem-like/progenitor markers (Tm4sf1, Sox9, Creb5)
  - Tolerance to new oncogenic programs

**Testable predictions:**

1. Cells in bottleneck state (high Tm4sf1/Sox9, low lineage markers) should
   have higher model-predicted plasticity/entropy

2. Niche context should determine whether bottleneck cells progress or
   redifferentiate

3. The model should identify cells transitioning INTO the bottleneck
   (losing lineage identity) as high-risk

### Gene Signatures

```python
# Bottleneck/stem-like state (the KEY plastic intermediate)
BOTTLENECK_MARKERS = ["TM4SF1", "SOX9", "CREB5"]
STEM_LIKE_PROGRAM = ["SOX2", "MYC", "KLF4"]

# Lineage markers (should be LOW in bottleneck)
AT2_LINEAGE = ["SFTPC", "SFTPB", "SFTPA1", "NKX2-1", "HOPX"]
AT1_LINEAGE = ["AGER", "PDPN", "AQP5"]
PNEC_LINEAGE = ["ASCL1", "CHGA", "SYP", "INSM1"]
BASAL_LINEAGE = ["KRT5", "KRT17", "TP63"]
SECRETORY_LINEAGE = ["SCGB1A1", "SCGB3A2"]

# Driver programs
EGFR_PROGRAM = ["EGFR", "ERBB2", "AREG", "HBEGF"]  # MAPK downstream
MYC_PROGRAM = ["MYC", "MYCN", "MAX"]

# Transformation-enabling genetic events
TRANSFORMATION_ENABLERS = {
    "PTEN_loss": "Enables MYC tolerance in AT2 via PI3K/AKT",
    "RB1_loss": "Required for neuroendocrine phenotype",
    "TP53_loss": "Contributes to transformation, not essential",
}
```

---

## Synthesis: How These Papers Inform StageBridge

### The Unified Model

1. **Precancerous progression requires niche cooperation** (Paper 1)
   - Stressed epithelial cells (AP-1+) signal to fibroblasts
   - Fibroblasts create supportive FN1+ scaffold
   - Only niche-supported tumors persist

2. **Niche composition determines EMT/plasticity** (Paper 2)
   - NMF7-like niches (SPP1+ TAMs, CAFs) promote hypoxia and EMT
   - Cells in these niches have higher entropy (plasticity)

3. **Transformation requires plastic intermediate** (Papers 3, 5)
   - Cells must enter undifferentiated "bottleneck" state
   - Characterized by loss of lineage markers, gain of stem markers
   - Microenvironment then determines which new fate is adopted

4. **AP-1 is the stress hub** (Paper 4)
   - AP-1 activation enables chromatin remodeling and plasticity
   - Marks cells capable of state transitions

### What StageBridge Should Learn

1. **Receiver cells in supportive niches** (FN1+, near activated fibroblasts)
   should have higher predicted progression

2. **Cells with bottleneck signatures** (high TM4SF1/SOX9, low lineage markers)
   should have high plasticity scores

3. **Niche context should modify fate**: Same cell state in different niches
   should have different predicted trajectories

4. **AP-1 + niche signals = progression risk**: The combination of intrinsic
   stress response AND extrinsic niche support predicts outcome

---

## Validation Gene Sets Summary

```python
# For run_ap1_validation.py
AP1_CORE = ["JUN", "JUNB", "JUND", "FOS", "FOSB", "ATF3", "ATF4"]
TUMOUR12_STATE = AP1_CORE + ["EGR1", "RUNX1", "SOX9", "AREG", "HBEGF"]

# For run_senescence_validation.py  
SASP_CORE = ["IL6", "IL1B", "IL1A", "CXCL8", "CCL2", "GDF15"]
NICHE_ECM = ["FN1", "COL1A1", "COL1A2", "COL3A1"]

# For plasticity/bottleneck validation (NEW)
BOTTLENECK_HIGH = ["TM4SF1", "SOX9", "CREB5", "SOX2"]
BOTTLENECK_LOW = ["SFTPC", "HOPX", "NKX2-1", "ASCL1", "KRT5"]  # All lineage markers

# For NMF7/hypoxic niche
NMF7_NICHE = ["SPP1", "VIM", "HIF1A", "COL1A1"]
```

---

## References

1. Alcolea et al. (2026). Precancerous niche remodelling dictates nascent
   tumour persistence. Nature.

2. Tsankov et al. (2025). A cellular and spatial atlas of TP53-associated
   tissue remodeling defines a multicellular tumor ecosystem in lung
   adenocarcinoma. Nature Cancer.

3. Ganesh et al. (2025). Progressive plasticity during colorectal cancer
   metastasis. Nature.

4. Marjanovic et al. (2024). A mechanism for adaptive genome regulation
   in cancer. Cell.

5. Gardner et al. (2024). Lineage-specific intolerance to oncogenic drivers
   restricts histological transformation. Science.
