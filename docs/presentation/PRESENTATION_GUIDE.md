# StageBridge Presentation Guide

**Created:** 2026-04-29
**Purpose:** Thesis defense, poster, lab meeting (modular - expand/contract per venue)
**Total slides:** ~37

---

## Figure Sources Legend

- **[YOUR FIG]** = Your preliminary figures (Downloads folder or internal review PDF)
- **[CITE]** = Published figure, cite and adapt
- **[CREATE]** = Need to make new diagram
- **[TEXT]** = Text-only slide

---

## Section 1: The Clinical Problem (3 slides)

### Slide 1: "LUAD Develops Through Stereotyped Stages"
**Content:**
- Normal → AAH → AIS → MIA → LUAD (IAC)
- Histological progression with accumulating molecular abnormalities
- SCNA, allelic imbalance, TMB/APOBEC, methylation changes

**Figure:** [CITE] Mascaux et al. 2025, J Thorac Oncol, Figure 1 (adenomatous carcinogenesis panel)

**Talking points:**
- "Lung adenocarcinoma doesn't appear de novo - it progresses through these defined histological stages"
- "Pathologists can identify these stages in tissue sections"
- "The question is: which of these precursor lesions will actually progress?"

---

### Slide 2: "Immune Evasion Occurs Early"
**Content:**
- T-cell suppression begins in early tumorigenesis
- Immune checkpoint engagement, loss of immunogenicity, suppressive cytokines
- This is the window where niche context matters most

**Figure:** [CITE] Mascaux et al. 2025, J Thorac Oncol, Figure 4 (immune evasion schematic)

**Talking points:**
- "Immune evasion isn't just a late-stage phenomenon"
- "It begins early, in the preinvasive lesions"
- "This creates a window where the microenvironment may determine fate"

---

### Slide 3: "The Data: Peng/Kadara Cohort"
**Content:**
- 18 donors, matched snRNA-seq + Visium + WES
- Stages coexist spatially in same tissue section
- Key finding: IL1B-high inflammatory niches more common in precursors

**Figure:** [YOUR FIG] Panel B - Spatial tissue example (P1: stage + IL1B + context)
- File: `panel_spatial_donor_P1.png` or internal review PDF page 3

**Talking points:**
- "Our data comes from the Peng cohort - 18 patients with matched multi-modal data"
- "Critically, multiple stages coexist in the same tissue section"
- "This is what lets us learn dynamics from a snapshot"

---

## Section 2: Why Snapshots Can Reveal Dynamics (2 slides)

### Slide 4: "Learning Dynamics from Snapshots"
**Content:**
- No time-series available - single snapshot per patient
- But coexisting stages = "frozen progression" (field cancerization)
- Prior work showed neighborhood → dynamics is possible

**Figure:** [CREATE] Simple diagram showing tissue section with coexisting stages annotated

**Talking points:**
- "We don't have time-series data - that would require repeated biopsies"
- "But these stages coexist spatially, like frames of a movie laid side by side"
- "The assumption is that they represent different points along the same trajectory"

---

### Slide 5: "Our Hypothesis"
**Content:**
> "Cross-sectional cell state transitions become more identifiable when conditioned on receiver-centered local niche context"

**Figure:** [TEXT] - Hypothesis statement prominently displayed

**Talking points:**
- "This is the central claim of StageBridge"
- "Not just that cells transition, but that the LOCAL MICROENVIRONMENT predicts which cells transition and where they go"
- "Receiver-centered means we model what the focal cell receives from its neighbors"

---

## Section 3: Prior Art & Gap (3 slides)

### Slide 6: "OSDR: Neighborhood → Population Dynamics"
**Content:**
- Neighborhood composition predicts Ki67 (division rate)
- Phase portraits with stable fixed points (hot/cold fibrosis)
- Validated on breast cancer: predicted treatment response
- Limitation: population-level (cell counts), not cell-state trajectories

**Figure:** [CITE] Somer et al. 2026, Nature, Figure 1

**Talking points:**
- "OSDR showed that you CAN learn dynamics from a snapshot"
- "Neighborhood composition predicts division rate"
- "But it models population sizes - how many fibroblasts - not individual cell fates"

---

### Slide 7: "AMICI: Receiver-Centered Attention"
**Content:**
- Receiver = query, neighbors = keys/values
- Distance-dependent decay (closer = stronger)
- Predicts receiver gene expression from neighbor context
- Limitation: static expression, not transitions

**Figure:** [CITE] Hong et al. 2025, bioRxiv, Figure 1a-c

**Talking points:**
- "AMICI introduced receiver-centered attention for spatial transcriptomics"
- "The focal cell queries its neighbors - attention weighted by distance"
- "But it predicts expression at one timepoint, not transitions between states"

---

### Slide 8: "The Gap StageBridge Fills"
**Content:**

| Method | Models | Level | Dynamics |
|--------|--------|-------|----------|
| OSDR | Division rate | Population | Yes |
| AMICI | Gene expression | Cell | No |
| **StageBridge** | **State transitions** | **Cell** | **Yes** |

**Figure:** [CREATE] Table

**Talking points:**
- "StageBridge combines insights from both"
- "Cell-level like AMICI, dynamics like OSDR"
- "Niche-conditioned state transitions - that's the novel contribution"

---

## Section 4: Method (10-12 slides)

### Slide 9: "StageBridge Architecture Overview"
**Content:**
- Four components: Dual-Reference → 9-Token Niche → Set Transformer → OT-CFM

**Figure:** [YOUR FIG] Figure 1 from your paper (full architecture)

**Talking points:**
- "Four layers, each with a specific purpose"
- "I'll walk through each one"

---

### Slide 10: "Dual-Reference Geometry"
**Content:**
- HLCA (30d): healthy lung - captures normal cell type variation
- LuCA (10d): cancer - captures disease-specific states
- Variance contribution: HLCA 40.8%, LuCA 59.2%

**Figure:** [YOUR FIG] Panel H (dual-reference embedding analysis)
- File: `fig_reference_contribution.png` or internal review PDF page 9

**Talking points:**
- "We map cells to TWO reference atlases"
- "HLCA tells us where the cell sits relative to healthy lung"
- "LuCA tells us where it sits relative to disease states"
- "Both contribute - about 40/60 split in variance explained"

---

### Slide 11: "Dual-Reference: Current vs Future"
**Content:**
- Current: Concatenation [z_HLCA || z_LuCA] - pragmatic
- Problem: Different latent geometries, no alignment
- Future: Gromov-Wasserstein fusion - preserves relational structure

**Figure:** [CREATE] GW concept diagram

**Talking points:**
- "Currently we just concatenate - simple but not principled"
- "The two spaces have different geometries"
- "Future work: Gromov-Wasserstein alignment respects who-is-close-to-whom in each space"

---

### Slide 12: "9-Token Niche Representation"
**Content:**
- Token 0: Receiver (focal cell)
- Tokens 1-4: Spatial rings (increasing radii)
- Token 5-6: HLCA/LuCA reference embeddings
- Token 7: Pathway activity (PROGENy)
- Token 8: Neighborhood statistics

**Figure:** [YOUR FIG] Figure 1B from your paper

**Talking points:**
- "We tokenize the neighborhood into 9 structured tokens"
- "The receiver, four spatial rings at increasing distances, reference tokens, pathway activity, statistics"
- "This preserves spatial structure that pooling would lose"

---

### Slide 13: "Receiver-Centered Attention"
**Content:**
- Receiver = Q, neighbors = K,V
- Distance bias enforced positive → monotonic decay
- Empty token allows "no informative neighbor"
- Sparsity regularization

**Figure:** [CREATE] Attention equation + diagram, cite AMICI

**Talking points:**
- "The receiver queries its neighbors"
- "Attention decreases with distance - enforced by architecture, not just learned"
- "Empty token lets the model say 'nothing here is informative'"

---

### Slide 14: "Hierarchical Set Transformer"
**Content:**
- ISAB: O(nm) complexity via inducing points
- SAB: Direct token interaction
- PMA: Pools to fixed-size output

**Figure:** [YOUR FIG] Figure 1C from your paper

**Talking points:**
- "Set Transformer processes the token sequence"
- "ISAB gives linear complexity - important for scaling"
- "PMA extracts a fixed-size niche embedding"

---

### Slide 15: "OT-CFM Flow Matching"
**Content:**
- Learn v(x,t,c) that transports stage distributions
- Sinkhorn finds optimal cell pairings
- Flow matching regresses against OT-implied velocities

**Figure:** [YOUR FIG] fig_ot_dynamics.pdf
- File: `fig_ot_dynamics.pdf`

**Talking points:**
- "Given cells from different stages, how do we learn the transition?"
- "Optimal transport finds which source cells should match to which targets"
- "Flow matching learns a smooth velocity field that performs the transport"

---

### Slide 16: "The Math" (Optional - ML audience)
**Content:**
```
OT coupling:    π* = argmin ∫c(x,y)dπ + εH(π)
Interpolation:  x_t = (1-t)x₀ + tx₁  
Loss:           L = ||v_θ(x_t, t, c) - (x₁ - x₀)||²
```

**Figure:** [TEXT] Equations

**Talking points:**
- "Sinkhorn gives us the coupling with entropic regularization"
- "We interpolate along the OT path"
- "The network learns to predict the velocity at each point"

---

### Slide 17: "Why OT Coupling Matters"
**Content:**
- Random pairing → crossing trajectories → averaging
- OT pairing → non-crossing → clean velocity field

**Figure:** [CREATE] Crossing vs non-crossing diagram

**Talking points:**
- "Without OT, random pairings create crossing trajectories"
- "The velocity field averages out, losing information"
- "OT ensures cells take direct paths"

---

### Slide 18: "Evolution Branch (WES Integration)"
**Content:**
- TMB, mutational signatures (SBS1, SBS4, SBS13)
- ACMG pathogenicity annotations
- OncoKB actionability levels
- Gated fusion with niche context

**Figure:** [YOUR FIG] Panel A (driver mutation landscape)
- File: internal review PDF page 2

**Talking points:**
- "We also integrate whole-exome sequencing"
- "TMB, mutational signatures, driver mutations"
- "ACMG and OncoKB give us clinical actionability"
- "This fuses genomics with niche context via a learned gate"

---

### Slide 19: "Auxiliary Biological Heads"
**Content:**
- PathwayHead: 14 PROGENy pathways
- ProliferationHead: Ki67 prediction
- Purpose: Regularize latent, not primary outputs

**Figure:** [TEXT]

**Talking points:**
- "We add auxiliary heads to regularize the learned representations"
- "Pathway activities, proliferation markers"
- "These aren't the outputs - they're training signal to keep the latent biologically grounded"

---

## Section 5: Validation & Ablations (4 slides)

### Slide 20: "Baseline Ladder"
**Content:**

| Model | Perm. Invariant | Spatial | Receiver-Centered |
|-------|----------------|---------|-------------------|
| PoolingMLP | No | No | No |
| DeepSets | Yes | No | No |
| SetTransformer | Yes | No | No |
| GraphSAGE | Yes | Yes | No |
| **StageBridge** | **Yes** | **Yes** | **Yes** |

**Figure:** [CREATE] Table

**Talking points:**
- "We compare against a ladder of baselines"
- "Each adds one capability"
- "StageBridge is the only one with all three"

---

### Slide 21: "Ablation Results"
**Content:**

| Configuration | MSE | Wasserstein |
|--------------|-----|-------------|
| StageBridge (full) | 0.341 ± 0.02 | 0.680 ± 0.03 |
| No niche | 0.412 ± 0.03 (+21%) | 0.812 ± 0.04 (+19%) |
| Pooled niche | 0.389 ± 0.02 | 0.756 ± 0.03 |
| HLCA only | 0.367 ± 0.02 | 0.721 ± 0.03 |
| LuCA only | 0.378 ± 0.03 | 0.745 ± 0.04 |
| Deterministic | 0.395 ± 0.03 | 0.768 ± 0.04 |

**Figure:** [YOUR FIG] Table II from paper

**Talking points:**
- "Removing niche context: 21% worse MSE, 19% worse Wasserstein"
- "Pooling helps but structured tokens are better"
- "Need both HLCA and LuCA"

---

### Slide 22: "What the Ablations Prove"
**Content:**
- H1 (niche matters): Confirmed - 21% degradation
- H2 (both references): Confirmed - single reference insufficient
- H3 (stochastic > deterministic): Confirmed - better calibration

**Figure:** [TEXT]

**Talking points:**
- "Each hypothesis confirmed"
- "Niche matters, both references matter, stochasticity matters"

---

### Slide 23: "Learned Context is Interpretable"
**Content:**
- γ₄: Stage/progression (r=0.18)
- γ₂: Inflammatory (r=-0.16)
- TGFβ, TNFα/NFκB captured

**Figure:** [YOUR FIG] Panel F (gamma heatmap)
- File: `panel_gamma_interpretation.png` or internal review PDF page 7

**Talking points:**
- "The learned context dimensions aren't black boxes"
- "They correlate with known biological signals"
- "Stage, inflammation, TGFbeta - all captured"

---

## Section 6: Biological Results (7 slides)

### Slide 24: "IL1B-High Niches Mark the Progression Window"
**Content:**
- IL1B: 1.7× higher in preinvasive vs normal (p<0.001)
- Replicates Peng/Kadara independently

**Figure:** [YOUR FIG] Panel A-B (IL1B violin)
- File: internal review PDF page 2, panel B

**Talking points:**
- "IL1B signature is significantly elevated in preinvasive lesions"
- "This replicates the original Peng finding with our model"
- "These inflammatory niches are the progression-permissive environments"

---

### Slide 25: "Spatial IL1B Hotspots"
**Content:**
- IL1B localizes to specific tissue regions
- Hotspots align with preinvasive annotations

**Figure:** [YOUR FIG] P1 IL1B spatial map
- File: `panel_spatial_P4_il1b.png` or internal review PDF page 3, middle panel

**Talking points:**
- "IL1B isn't uniformly distributed"
- "It concentrates in specific regions"
- "Those regions correspond to preinvasive lesions"

---

### Slide 26: "T-Cell Depletion in Preinvasive"
**Content:**
- Normal: 15.9%
- Preinvasive: 5.7% ← **lowest**
- Invasive: 10.9%

**Figure:** [YOUR FIG] Panel E (T-cell raincloud)
- File: `panel_tcell_raincloud.png` or internal review PDF page 6

**Talking points:**
- "T-cells are most depleted in preinvasive - not invasive"
- "This matches the immune evasion window concept"
- "The immune system is suppressed precisely where intervention might work"

---

### Slide 27: "Microenvironment Remodeling"
**Content:**
- AT2 decreases, fibroblasts/macrophages increase
- Stage-specific ecosystem composition

**Figure:** [YOUR FIG] Panel C (stacked bar)
- File: internal review PDF page 2, panel C

**Talking points:**
- "The microenvironment remodels as stages progress"
- "Epithelial cells give way to fibroblasts and immune infiltrates"

---

### Slide 28: "Transition Zones"
**Content:**
- Stage boundaries show highest progression context
- Model identifies where transitions happen

**Figure:** [YOUR FIG] Panel C (transition zones)
- File: `panel_spatial_transition_zones.png` or internal review PDF page 4

**Talking points:**
- "The model identifies transition zones at stage boundaries"
- "Highest progression signal at the interfaces"

---

### Slide 29: "Proliferation in Context Space"
**Content:**
- High proliferation in high-inflammatory + high-progression quadrant

**Figure:** [YOUR FIG] Panel G (proliferation heatmap)
- File: `panel_phase_prolif.png` or internal review PDF page 8

**Talking points:**
- "The learned context space has biological meaning"
- "Proliferation is highest in the inflammatory, progression-prone quadrant"

---

### Slide 30: "OT Dynamics Summary"
**Content:**
- Velocity field shows directional flow
- Wasserstein distances per transition
- Irreversibility varies (mean flux 0.40)

**Figure:** [YOUR FIG] fig_ot_dynamics.pdf (full panel)

**Talking points:**
- "The full dynamics: velocity field, transport costs, irreversibility"
- "Transitions are mostly forward, but not entirely"

---

## Section 7: Clinical Applicability (4 slides)

### Slide 31: "From Latents to Actionable Outputs"
**Content:**
- Per-cell progression risk
- Per-region niche risk
- Stage-specific ecosystem summary
- Intervention targets

**Figure:** [TEXT] or [CREATE] output diagram

**Talking points:**
- "StageBridge produces concrete outputs, not just embeddings"
- "Risk scores, ecosystem summaries, targets"

---

### Slide 32: "Single-Biopsy Risk Stratification"
**Content:**
- Input: Visium + snRNA-seq (+ optional WES)
- Output: Spatial risk map

**Figure:** [YOUR FIG] P1 progression context map
- File: internal review PDF page 3, right panel

**Talking points:**
- "From a single biopsy, we can map progression risk spatially"
- "This identifies which regions to watch or intervene on"

---

### Slide 33: "Intervention Targets"
**Content:**
- IL1B-IL1R1 axis → anakinra
- ACMG/OncoKB actionable variants

**Figure:** [YOUR FIG] Panel A + [TEXT] drug list

**Talking points:**
- "IL1B pathway is druggable - anakinra exists"
- "Driver mutations have known targeted therapies"
- "The model tells you which niches to target"

---

### Slide 34: "The Clinical Question Answered"
**Content:**
- Q: Can a single biopsy identify high-risk lesions?
- A: Yes - niche + cell state + genomics = risk

**Figure:** [TEXT]

**Talking points:**
- "Coming back to the clinical question we started with"
- "Yes, we can stratify risk from a single biopsy"
- "By combining niche context, cell state, and genomics"

---

## Section 8: Limitations & Future (2 slides)

### Slide 35: "Limitations"
**Content:**
- Cross-sectional (no longitudinal validation)
- Spot-based Visium (deconvolution uncertainty)
- Atlas coverage (novel states)
- Concatenation fusion (not principled)

**Figure:** [TEXT]

---

### Slide 36: "Future Directions"
**Content:**
- GW fusion
- Prospective validation
- Clinical trial design
- Other cancers

**Figure:** [TEXT]

---

## Section 9: Summary (1 slide)

### Slide 37: "Take-Home Messages"
1. Niche context improves transition prediction (21% reduction)
2. Both references needed
3. IL1B-high niches mark progression window
4. T-cell depletion peaks in preinvasive
5. Single-biopsy stratification is feasible
6. Framework generalizable

**Figure:** [TEXT]

---

## Quick Reference: Your Figure Files

| Internal Review Panel | File | Use For |
|----------------------|------|---------|
| A. Multimodal summary | PDF page 2 | Slide 18, 24, 33 |
| B. Spatial tissue | PDF page 3 | Slide 3, 25, 32 |
| C. Transition zones | PDF page 4 | Slide 28 |
| D. Spatial proliferation | PDF page 5 | (optional) |
| E. T-cell raincloud | PDF page 6 | Slide 26 |
| F. Gamma interpretation | PDF page 7 | Slide 23 |
| G. Prolif in context | PDF page 8 | Slide 29 |
| H. Dual-reference | PDF page 9 | Slide 10 |

| Standalone File | Use For |
|-----------------|---------|
| fig_ot_dynamics.pdf | Slide 15, 30 |
| fig_flow_hero.png | Alternative for Slide 15 |
| fig_reference_contribution.png | Slide 10 |
| panel_gamma_interpretation.png | Slide 23 |
| panel_tcell_raincloud.png | Slide 26 |
| panel_spatial_transition_zones.png | Slide 28 |
| panel_phase_prolif.png | Slide 29 |

---

## Venue Adaptations

**Thesis Defense (~45-60 min):** Use all 37 slides

**Lab Meeting (~30 min):** Skip slides 11, 16, 17, 19, compress Section 8

**Poster:** Sections 1, 4 (condensed to 3-4 panels), 6 (key findings), 7 (clinical)

**Conference Talk (~15 min):** Slides 1, 3, 5, 8, 9, 15, 21, 24, 26, 30, 34, 37
