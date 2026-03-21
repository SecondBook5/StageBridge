# Biological and Clinical Framing for StageBridge

## The Problem

The model story is currently: "can we learn progression-relevant representations?"

That's interesting but not biologically sharp or clinically urgent.

**What's needed:** Pin StageBridge to **one concrete biological mechanism** and **one concrete clinical use case**.

---

## Biological Question

> **Which epithelial cell states and local immune/stromal niches characterize early, progression-prone LUAD precursor lesions?**

### The Target (from Peng/Kadara LUAD precursor paper)

- **KAC / reactive pneumocyte-like alveolar progenitors** are early predecessors of LUAD
- They reside in **epithelial–proinflammatory niches** enriched for:
  - IL1B-high macrophages
  - IL1B–IL1R1 signaling
- These niches are **more common in precursor lesions (AAH, AIS) than in LUAD**
- Targeting this inflammatory niche **reduces alveolar progenitors and LUAD pathogenesis**

### What StageBridge Should Answer

Not just "interesting latent" or "attention map" or "better benchmark score."

Instead:

> **Can we recover or quantify the epithelial–proinflammatory niche associated with early LUAD progression, and can we identify which cells are most at risk of moving into that state?**

---

## Clinical Question

> **Can a single cross-sectional biopsy be used to identify precursor lesions or local regions that are most likely to progress and therefore best suited for interception?**

### Why This Matters

- Precursor lesions (AAH, AIS) are the window for interception
- Progression is heterogeneous and hard to risk-stratify
- Clinicians need to know: which lesions warrant intervention vs. watchful waiting?

### The OSDR Model

OSDR didn't stop at "we inferred interactions." They:
1. Used spatial snapshots to infer tissue-level dynamics
2. Showed early-treatment biopsies distinguish responders from non-responders
3. Tied mechanism to a clinical decision

**StageBridge should do the same:** Don't stop at mechanistic interpretation; tie the mechanism to a decision.

---

## Strengthening Biological Interpretation

Move from generic interpretability to **hypothesis-linked interpretation**.

### 1. Predefine the Biological Axes

For LUAD application:
- Alveolar progenitor / KAC-like state axis
- Proinflammatory macrophage niche axis
- Epithelial dedifferentiation / inflammatory coupling

Then ask: **Do the model's learned representations track these axes?**

### 2. Quantify Niche-Conditioned State Change

Don't just say "these cells cluster differently."

Ask:
- Do epithelial cells in inflammatory niches map further toward the disease-aware (LuCA) reference?
- Do they show stronger progression scores?
- Do they occupy a distinct receiver-centered niche representation?

### 3. Use Perturbation-Style Interpretation

Since the niche module is receiver-centered:
- Remove IL1B-high macrophage-like neighbors
- How much does the receiver's progression score or representation change?

**This is much stronger than raw attention weights.**

### 4. Frame Findings as Stage-Specific

Peng/Kadara's point: inflammatory niches are more common in precursors than LUAD.

Ask:
- Which niche patterns are enriched in AAH/AIS?
- Which patterns are lost or transformed in LUAD?
- Which receiver states appear early?

**This gives a progression story, not a static niche story.**

---

## Strengthening Clinical Relevance

### Option 1: Risk Stratification of Precursor Lesions (cleanest)

> **StageBridge identifies which precursor lesions harbor epithelial states and local niches associated with higher progression risk.**

Clinically relevant because precursor lesions are exactly where interception matters most.

### Option 2: Early Interception Target Discovery

> **StageBridge identifies stage-specific epithelial–immune niches that may be actionable before invasion.**

Stronger than generic biomarker language because Peng/Kadara show inflammatory niche targeting reduces the relevant progenitor population.

### Option 3: Spatial Ecosystem Prognosis

Define spatial ecosystems, connect them to prognosis or progression risk, don't leave them as abstract embeddings.

---

## Required Model Outputs

StageBridge should produce not just:
- ❌ Latent coordinates
- ❌ Attention maps
- ❌ Baseline metrics

But also:
- ✅ **Progression-risk score** per epithelial cell or per lesion region
- ✅ **Niche-risk score** tied to concrete neighborhoods
- ✅ **Stage-specific ecosystem summary**
- ✅ **Ranked list of biologically meaningful niches** associated with early progression
- ✅ **Clear candidate intervention axis** (e.g., IL1B-high macrophage / inflammatory niche)

---

## The Shift

**From:** "We model progression with a transformer."

**To:** "Which epithelial cells, in which local niches, appear most progression-prone, and does that reveal an interceptable early disease ecosystem?"

---

## Key References

- **Peng/Kadara**: LUAD precursor paper - KAC progenitors, IL1B-high macrophages, proinflammatory niche
- **OSDR**: Tissue dynamics from snapshot, clinical prediction from early biopsies
- **TLS paper**: Spatial ecosystem → prognosis connection
