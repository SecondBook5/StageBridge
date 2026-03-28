# StageBridge Biological Context

This document consolidates the biological motivation, hypothesis, and expected outputs for StageBridge.

---

## 1. The LUAD Initiation Problem

### The Stage Ladder

Lung adenocarcinoma (LUAD) develops through a stereotyped morphological progression:

1. **Normal** - Normal alveolar epithelium. Type II pneumocytes maintain the alveolar surface.
2. **AAH** (Atypical Adenomatous Hyperplasia) - Focal proliferation of mildly atypical pneumocytes along alveolar walls. The earliest preneoplastic lesion.
3. **AIS** (Adenocarcinoma In Situ) - Lepidic growth of neoplastic cells without stromal invasion. Complete resection is curative.
4. **MIA** (Minimally Invasive Adenocarcinoma) - Predominantly lepidic pattern with <=5mm invasion. Near-100% disease-free survival after resection.
5. **LUAD** (Invasive Lung Adenocarcinoma) - Tumor with invasion exceeding 5mm. Varied histological subtypes.

### Transition Biology

- **Normal to AAH** - Initiating mutations (often KRAS) drive focal hyperplasia. Microenvironment largely intact.
- **AAH to AIS** - Transition from hyperplasia to in-situ carcinoma. Spatial tissue reorganization.
- **AIS to MIA** - Onset of invasion. Local microenvironment (fibroblast activation, immune evasion) may gate invasion.
- **MIA to LUAD** - Established invasion. Tumor heterogeneity increases.

### The Peng/Kadara Cohort

GSE308103 (snRNA-seq), GSE307534 (Visium), GSE307529 (WES) - matched data across all five stages enabling:
- Cell-level transcriptomes in spatial context
- Evolutionary state linked to specific transitions
- Cross-sectional inference of transition dynamics

---

## 2. Target Mechanism (from Peng/Kadara)

### Key Findings

- **KAC / reactive pneumocyte-like alveolar progenitors** are early predecessors of LUAD
- They reside in **epithelial-proinflammatory niches** enriched for:
  - IL1B-high macrophages
  - IL1B-IL1R1 signaling axis
- These niches are **more common in precursor lesions (AAH, AIS) than in LUAD**
- Targeting this inflammatory niche **reduces alveolar progenitors and LUAD pathogenesis**

### Scientific Question

> Which epithelial cell states and local immune/stromal niches characterize early, progression-prone LUAD precursor lesions?

### Clinical Question

> Can a single cross-sectional biopsy identify precursor lesions or regions most likely to progress and best suited for interception?

---

## 3. The Niche Gating Hypothesis

### Statement

Local epithelial-stromal-immune neighborhood structure modulates cell-state transitions between LUAD initiation stages.

### What "Niche-Gated" Means

A transition is niche-gated if dynamics depend on local tissue context:
- Two cells with similar transcriptional profiles but different niches have different trajectories
- Removing niche conditioning degrades transition model performance
- The model learns niche-type-specific contributions to dynamics

### How StageBridge Tests This

**Primary Test: Context Ablation**
- Conditioned: Velocity field receives context vector from Layer C
- Unconditioned: No niche information

**Secondary Tests:**
1. Niche perturbation - Shuffle niche contexts
2. Niche regime analysis - Cluster niches by composition
3. Context sensitivity - Gradient of velocity field w.r.t. context

---

## 4. Required Model Outputs

StageBridge must produce not just latents and attention maps, but:

| Output | Description |
|--------|-------------|
| **Progression-risk score** | Per epithelial cell or per lesion region |
| **Niche-risk score** | Tied to concrete neighborhoods |
| **Stage-specific ecosystem summary** | Which niches are enriched at each stage |
| **Candidate intervention axis** | e.g., IL1B-high macrophage / inflammatory niche |

### Transition Quality Metrics
- Sinkhorn distance, MMD, trajectory smoothness
- Niche conditioning effect (conditioned vs unconditioned)
- Gene programs at key transitions

---

## 5. Biological Validation

### Predefined Axes (from Peng/Kadara)

1. Alveolar progenitor / KAC-like state axis
2. Proinflammatory macrophage niche axis
3. Epithelial dedifferentiation / inflammatory coupling

### Validation Questions

- Do epithelial cells in inflammatory niches map further toward LuCA reference?
- Do they show stronger progression scores?
- Do they occupy distinct receiver-centered niche representations?

### Perturbation-Style Interpretation

Remove IL1B-high macrophage-like neighbors from niche context. How much does the receiver's progression score change?

This is stronger than raw attention weights.

---

## References

- **Peng/Kadara**: LUAD precursor paper - KAC progenitors, IL1B-high macrophages
- **OSDR**: Tissue dynamics from snapshot, clinical prediction from early biopsies
- **HLCA**: Human Lung Cell Atlas (healthy reference)
- **LuCA**: Lung Cancer Atlas (disease reference)
