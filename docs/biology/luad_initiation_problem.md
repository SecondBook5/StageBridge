# The LUAD Initiation Problem

## The Stage Ladder

Lung adenocarcinoma (LUAD) develops through a stereotyped morphological progression:

1. **Normal** — Normal alveolar epithelium. Type II pneumocytes maintain the alveolar surface.
2. **AAH** (Atypical Adenomatous Hyperplasia) — Focal proliferation of mildly atypical pneumocytes along alveolar walls. The earliest preneoplastic lesion.
3. **AIS** (Adenocarcinoma In Situ) — Lepidic growth of neoplastic cells without stromal invasion. Complete resection is curative.
4. **MIA** (Minimally Invasive Adenocarcinoma) — Predominantly lepidic pattern with ≤5mm invasion. Near-100% disease-free survival after resection.
5. **LUAD** (Invasive Lung Adenocarcinoma) — Tumor with invasion exceeding 5mm. Varied histological subtypes.

## Why This Ladder Is Biologically Interesting

The Normal-to-LUAD progression is one of the best-characterized solid tumor initiation sequences. Each transition has distinct molecular and microenvironmental correlates:

- **Normal to AAH** — Initiating mutations (often KRAS) drive focal hyperplasia. The tissue microenvironment is largely intact.
- **AAH to AIS** — Transition from hyperplasia to in-situ carcinoma. Spatial tissue reorganization is expected — the relationship between epithelial proliferation and surrounding stromal/immune composition likely changes.
- **AIS to MIA** — Onset of invasion. Local microenvironment (fibroblast activation, immune evasion) may gate whether invasion begins.
- **MIA to LUAD** — Established invasion. Tumor heterogeneity increases. The niche becomes tumor-shaped rather than tissue-shaped.

## What Makes This Tractable

The Peng et al. cohort (GSE308103, GSE307534, GSE307529) provides matched snRNA-seq, Visium spatial, and WES data across all five stages from the same patients. This is rare — most datasets capture only one or two stages.

Having matched modalities across the full ladder means:
- Cell-level transcriptomes can be placed in spatial context
- Evolutionary state (mutations, CNVs) can be linked to specific transitions
- Cross-sectional snapshots can be used to infer transition dynamics

## The Open Question

**Which transitions are niche-gated?**

Does the local cellular neighborhood (epithelial-stromal-immune composition) determine whether a cell population progresses to the next stage? And if so, how does the evolutionary state of the tumor modulate that gating?

## StageBridge Approach

StageBridge models this as a **cell-state transition problem**:

1. Cells are embedded in dual-reference latent space (HLCA + LuCA)
2. Local niches are encoded as context vectors
3. Flow matching learns niche-conditioned trajectories between stages
4. Evolutionary constraints from WES regularize biologically plausible paths

The question becomes: do niche-conditioned transitions differ from unconditioned transitions? If yes, the niche gates progression.

This is the core scientific question StageBridge V1 is designed to test.
