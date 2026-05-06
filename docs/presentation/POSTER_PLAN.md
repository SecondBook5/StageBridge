# StageBridge Poster Plan
## Tumor Immune Systems Biology Symposium

**Format:** Standard research poster (48" x 36" or similar)
**Audience:** Systems biology, immunology, tumor biology researchers

---

## Layout (6-Panel Structure)

```
┌─────────────────────────────────────────────────────────────────┐
│                         TITLE BANNER                            │
│  StageBridge: Receiver-Centered Niche Encoding Reveals          │
│  Non-Equilibrium Dynamics in Lung Adenocarcinoma Progression    │
│  AJ Book | Johns Hopkins University                             │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────┬──────────────────┬──────────────────┐
│   1. PROBLEM     │   2. APPROACH    │   3. METHOD      │
│                  │                  │                  │
│  • Clinical gap  │  • Hypothesis    │  • Architecture  │
│  • Peng data     │  • Prior art gap │  • 9-token niche │
│  • IL1B niches   │  • Niche matters │  • Set Transformer│
│                  │                  │  • OT-CFM        │
└──────────────────┴──────────────────┴──────────────────┘

┌──────────────────┬──────────────────┬──────────────────┐
│  4. RESULTS:     │  5. RESULTS:     │  6. CONCLUSIONS  │
│     ABLATIONS    │     BIOLOGY      │                  │
│                  │                  │  • Take-homes    │
│  • Niche +21%    │  • IL1B niches   │  • Clinical app  │
│  • Dual-ref      │  • T-cell depl.  │  • QR code       │
│  • Baseline comp │  • Transition    │  • Contact       │
│                  │     zones        │                  │
└──────────────────┴──────────────────┴──────────────────┘
```

---

## Panel 1: The Clinical Problem (Top Left)

**Title:** Early LUAD: The Niche-Gated Progression Window

**Content:**
- LUAD develops through defined stages: Normal → AAH → AIS → MIA → Invasive
- Key finding from Peng et al.: IL1B-high inflammatory niches are enriched in precursor lesions, not late-stage tumors
- Clinical question: Can we identify which precursor lesions will progress from a single biopsy?

**Figure:** Peng data overview
- Spatial tissue section showing coexisting stages
- IL1B expression overlay

**Key stat:** 25 donors, matched snRNA-seq + Visium + WES

---

## Panel 2: The Hypothesis & Gap (Top Middle)

**Title:** Niche Context Constrains Transition Inference

**Hypothesis box (prominent):**
> "Cross-sectional cell state transitions become more identifiable when conditioned on receiver-centered local niche context"

**Gap table:**

| Method | Level | Dynamics | Niche |
|--------|-------|----------|-------|
| OSDR (Somer 2026) | Population | Yes | No |
| AMICI (Hong 2025) | Cell | No | Yes |
| **StageBridge** | **Cell** | **Yes** | **Yes** |

**Key insight:** Neither population dynamics nor static cell-level models capture niche-conditioned cell fate.

---

## Panel 3: Architecture (Top Right)

**Title:** StageBridge: Four Integrated Components

**Figure:** Architecture schematic (simplified from paper Figure 1)

**Four components (brief):**

1. **Dual-Reference Geometry**
   - HLCA (healthy) + LuCA (disease)
   - Anchors cells in both coordinate systems

2. **9-Token Niche Representation**
   - Receiver + 4 spatial rings + 2 references + pathway + stats
   - Preserves spatial structure

3. **Hierarchical Set Transformer**
   - ISAB → ISAB → SAB → PMA
   - Permutation-invariant, linear complexity

4. **Cross-Attention Drift (OT-CFM)**
   - Niche-conditioned velocity field
   - Sinkhorn-coupled flow matching

**Key stat:** 20.5M parameters

---

## Panel 4: Ablation Results (Bottom Left)

**Title:** Niche Conditioning Is Necessary

**Table: Ablation Study**

| Configuration | Val Loss | Δ |
|--------------|----------|---|
| **StageBridge (full)** | **0.187** | — |
| No niche (receiver only) | 0.227 | +21% |
| Pooled niche (no rings) | 0.209 | +12% |
| HLCA only | 0.198 | +6% |
| LuCA only | 0.203 | +9% |

**Figure:** Bar chart of ablation results

**Baseline comparison:**

| Baseline | Val Loss |
|----------|----------|
| PoolingMLP | [TBD] |
| DeepSets | [TBD] |
| SetTransformer | [TBD] |
| GraphSAGE | [TBD] |
| **StageBridge** | **0.187** |

**Takeaway:** Removing niche context degrades transition modeling by 21%. Structured tokenization outperforms pooling. Both references needed.

---

## Panel 5: Biological Findings (Bottom Middle)

**Title:** Proinflammatory Niches Gate Early Progression

**Figure A: IL1B Expression by Stage**
- Violin plot: Normal (31%) → Preinvasive (49%) → Invasive (51%)
- Spearman ρ = 0.336, p < 10⁻¹⁶

**Figure B: T-Cell Depletion**
- Normal: 15.9%
- Preinvasive: **5.7%** (lowest)
- Invasive: 10.9%
- "Immune evasion peaks in preinvasive, not invasive"

**Figure C: Spatial Transition Zones**
- Tissue section with stage boundaries highlighted
- Progression context highest at interfaces

**Key finding:** The learned context dimensions correlate with known biology:
- γ₄ ↔ TGFβ (r = 0.18)
- γ₆ ↔ TNFα/NFκB (r = 0.12)

---

## Panel 6: Conclusions & Clinical Application (Bottom Right)

**Title:** From Snapshots to Risk Stratification

**Take-home messages:**
1. Niche conditioning improves transition modeling (+21%)
2. Dual-reference geometry outperforms single reference
3. IL1B-high niches mark the progression window
4. T-cell depletion peaks in preinvasive (not invasive)
5. Single-biopsy risk stratification is feasible

**Clinical application:**
- Input: Single biopsy (Visium + snRNA-seq ± WES)
- Output: Spatial progression risk map
- Targets: IL1B-IL1R1 axis (anakinra), driver mutations (OncoKB)

**Future directions:**
- Gromov-Wasserstein fusion for geometry-preserving alignment
- Prospective clinical validation
- Generalization to other epithelial cancers

**QR Code:** Link to preprint/code

**Contact:**
- abook3@jhu.edu
- GitHub: [repo URL]

---

## Figures Needed

| Panel | Figure | Source |
|-------|--------|--------|
| 1 | Spatial tissue + IL1B | Peng data / your P1 figure |
| 3 | Architecture schematic | Simplify from paper Fig 1 |
| 4 | Ablation bar chart | Generate from results |
| 5A | IL1B violin | Your existing figure |
| 5B | T-cell composition | Your raincloud plot |
| 5C | Transition zones | Your spatial figure |

---

## Design Notes

**Color scheme:**
- Use consistent colors for stages: Normal (blue), Preinvasive (orange), Invasive (red)
- IL1B: purple
- Architecture components: match paper figure colors

**Typography:**
- Title: Bold, 72-96pt
- Section headers: Bold, 48-60pt
- Body text: 24-32pt
- Captions: 18-24pt

**Emphasis:**
- Hypothesis in prominent box
- Key stats in bold
- Clinical takeaway highlighted

**Audience considerations:**
- Systems biology audience understands OT, transformers
- Emphasize biological findings over method details
- Highlight immune relevance for symposium theme
