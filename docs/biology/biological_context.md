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

### Why This Ladder Is Biologically Interesting

The Normal-to-LUAD progression is one of the best-characterized solid tumor initiation sequences. Each transition has distinct molecular and microenvironmental correlates:

- **Normal to AAH** - Initiating mutations (often KRAS) drive focal hyperplasia. The tissue microenvironment is largely intact.
- **AAH to AIS** - Transition from hyperplasia to in-situ carcinoma. Spatial tissue reorganization is expected.
- **AIS to MIA** - Onset of invasion. Local microenvironment (fibroblast activation, immune evasion) may gate whether invasion begins.
- **MIA to LUAD** - Established invasion. Tumor heterogeneity increases. The niche becomes tumor-shaped rather than tissue-shaped.

### What Makes This Tractable

The Peng et al. cohort (GSE308103, GSE307534, GSE307529) provides matched snRNA-seq, Visium spatial, and WES data across all five stages from the same patients. Having matched modalities across the full ladder means:
- Cell-level transcriptomes can be placed in spatial context
- Evolutionary state (mutations, CNVs) can be linked to specific transitions
- Cross-sectional snapshots can be used to infer transition dynamics

---

## 2. The Niche Gating Hypothesis

### Statement

Local epithelial-stromal-immune neighborhood structure modulates cell-state transitions between LUAD initiation stages. The composition and spatial arrangement of the tissue microenvironment around cells influences the probability, direction, and dynamics of progression to subsequent disease stages.

### What "Niche-Gated" Means

A transition is niche-gated if the learned dynamics depend on local tissue context - not just the intrinsic state of the transitioning cell:

- Two cells at the AAH stage with similar transcriptional profiles but different surrounding niches should have different predicted trajectories.
- Removing niche conditioning should measurably degrade transition model performance.
- The model should learn niche-type-specific contributions to transition dynamics.

### Why This Hypothesis Is Plausible

1. **Stromal remodeling** - Cancer-associated fibroblasts create permissive environments for invasion.
2. **Immune surveillance** - Immune cell composition changes across the initiation ladder.
3. **Vascular remodeling** - Angiogenesis and vascular patterning change as tumors progress.
4. **Spatial evidence** - The Peng cohort includes matched Visium spatial data.

### How StageBridge Tests This

**Primary Test: Context Ablation**
- **Conditioned:** Velocity field receives context vector from Layer C
- **Unconditioned:** Velocity field receives no niche information

If niche-gated hypothesis is correct: conditioned model should produce better transitions.

**Secondary Tests:**
1. Niche perturbation - Shuffle niche contexts; observe change in predicted trajectories
2. Niche regime analysis - Cluster niches by composition; compare transition dynamics
3. Context sensitivity - Measure gradient of velocity field with respect to context vector

---

## 3. Expected Dynamical Outputs

### Trajectory Structure

| Property | Description | Biological Meaning |
|----------|-------------|-------------------|
| **Convergence** | Do trajectories from different sources converge? | Common attractor states |
| **Divergence** | Do similar sources diverge based on context? | Niche-dependent fate decisions |
| **Smoothness** | How smooth are the velocity fields? | Continuous vs discontinuous transitions |

### Niche Regimes

| Regime Type | Description | Example |
|-------------|-------------|---------|
| **Permissive** | Transitions proceed readily | High proliferation signal |
| **Restrictive** | Transitions slowed or blocked | Immune surveillance |
| **Divergent** | Trajectories bifurcate | Stromal vs epithelial fate |

### Gene/Program Attribution

| Transition | Expected Programs |
|------------|-------------------|
| Normal->AAH | Surfactant, early proliferation |
| AAH->AIS | Cell cycle, metabolic shift |
| AIS->MIA | EMT-related, invasion programs |
| MIA->LUAD | Immune evasion, angiogenesis |

### V1 Required Outputs

1. **Transition quality metrics** - Sinkhorn distance, MMD, trajectory smoothness
2. **Niche conditioning effect** - Comparison of conditioned vs unconditioned
3. **Niche regime identification** - At least preliminary clustering
4. **Context sensitivity analysis** - Quantify niche contribution to dynamics
5. **Biological validation** - Gene programs at key transitions

---

## 4. StageBridge Approach

StageBridge models this as a **cell-state transition problem**:

1. Cells are embedded in dual-reference latent space (HLCA + LuCA)
2. Local niches are encoded as context vectors
3. Flow matching learns niche-conditioned trajectories between stages
4. Evolutionary constraints from WES regularize biologically plausible paths

The question becomes: do niche-conditioned transitions differ from unconditioned transitions? If yes, the niche gates progression.

This is the core scientific question StageBridge V1 is designed to test.
