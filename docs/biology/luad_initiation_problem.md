# The LUAD Initiation Problem

## The Stage Ladder

Lung adenocarcinoma (LUAD) develops through a stereotyped morphological progression:

1. **Normal** — Normal alveolar epithelium. Type II pneumocytes maintain the alveolar surface.
2. **AAH** (Atypical Adenomatous Hyperplasia) — Focal proliferation of mildly atypical pneumocytes along alveolar walls. Considered the earliest preneoplastic lesion.
3. **AIS** (Adenocarcinoma In Situ) — Lepidic growth of neoplastic cells without stromal invasion. Formerly called bronchioloalveolar carcinoma. Complete resection is curative.
4. **MIA** (Minimally Invasive Adenocarcinoma) — Predominantly lepidic pattern with 5mm or less of invasion. Near-100% disease-free survival after resection.
5. **LUAD** (Invasive Lung Adenocarcinoma) — Tumor with invasion exceeding 5mm. Varied histological subtypes. Prognostically heterogeneous.

## Why This Ladder Is Biologically Interesting

The Normal-to-LUAD progression is one of the best-characterized solid tumor initiation sequences. Each transition is defined histologically and has distinct molecular correlates:

- **Normal to AAH** — Initiating mutations (often KRAS) drive focal hyperplasia. The tissue microenvironment is largely intact.
- **AAH to AIS** — The transition from hyperplasia to in-situ carcinoma. This is where spatial tissue reorganization is expected to be most informative — the relationship between epithelial proliferation and surrounding stromal/immune composition likely changes.
- **AIS to MIA** — The onset of invasion. Local microenvironment composition (fibroblast activation, immune evasion) may gate whether and how invasion begins.
- **MIA to LUAD** — Established invasion. Tumor heterogeneity increases. The niche is now tumor-shaped rather than tissue-shaped.

## What Makes This Tractable

The Peng et al. cohort (GSE308103, GSE307534, GSE307529) provides matched snRNA-seq, Visium spatial, and WES data across all five stages from the same patients. This is rare — most datasets capture only one or two stages, or lack spatial resolution.

Having matched modalities across the full ladder means:
- Cell-level transcriptomes can be placed in spatial context
- Evolutionary state (mutations, CNVs) can be linked to specific transitions
- Donor-held-out validation is possible across stages

## The Open Question

Which transitions are niche-gated? Does the local cellular neighborhood (epithelial-stromal-immune composition) determine whether a cell population progresses to the next stage? And if so, how does the evolutionary state of the tumor modulate that gating?

This is the question StageBridge v1 is designed to test.
