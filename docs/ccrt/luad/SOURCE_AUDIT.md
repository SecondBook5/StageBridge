# LUAD Source Audit

**Milestone 9 — Phase A source audit.** Populated from direct inspection of the
local StageBridge LUAD data tree. Not a template — every entry was verified.

## Source root

- **Path**: `/mnt/e/StageBridge_data` (override via `CCRT_LUAD_SOURCE_ROOT`).
- **Data present locally** (unlike the PanIN reference repo). Layout:
  `raw/geo/<ACCESSION>/{downloads,extracted}`, `interim/anndata/{snrna,spatial}`,
  `processed/{features,tangram,hlca,luca,anndata}`.

## ⚠️ Dataset accuracy / version caveat

**The biological accuracy and version of this local dataset are NOT verified.**
The maintainer has flagged uncertainty about how accurate/final this particular
processed copy is, and the tree contains pipeline *smoke-check* artifacts
(`processed/{tangram,anndata,hlca}_smoke_check/`, `*_smoke.h5ad`) alongside full
outputs. Some processed features (notably the per-lesion WES/evolutionary block)
are unusually complete for premalignant lesions and may be provisional or
synthetic pipeline output rather than final validated biology.

Consequences for this adapter:
- The adapter validates the **structure/schema** of what is present. It makes NO
  claim that the values are biologically correct or final.
- All unit tests and the model-ready dry run run on **source-faithful fixtures**,
  never on this dataset — they are unaffected by its accuracy.
- The real-source integration test is **descriptive**: it audits + structurally
  validates what exists and reports it (including smoke-vs-full detection); it
  does not assert biological correctness, and it does not gate the milestone on
  the dataset being final.
- Treat every column/row count and feature list below as *observed structure of
  this copy*, pending confirmation of the authoritative dataset version.

## Accessions (verified present)

| Accession | Modality | Observation unit |
|---|---|---|
| **GSE308103_snrna** | snRNA-seq | cell/nucleus |
| **GSE307534_spatial** | Visium spatial | spot |
| GSE223499_snrna, GSE223500_tcrseq, GSE223501_slideseq, GSE223502_lpwgs, GSE277206_spatial, GSE307529_wes | auxiliary | various |

The milestone named GSE308103/GSE307534 as *candidates*; both are verified
present. The Milestone-9 adapter uses these two as the snRNA reference + Visium
spatial partitions.

## Stages / receiver states (verified from GSM filenames)

`Normal`, `AAH`, `AIS`, `MIA`, `LUAD` — e.g. `GSM9237906_P4_AAH`,
`GSM9226189_P10_MIA`. Cross-sectional (no longitudinal claim). Donor = patient
(`P3`…`P22`); sample = GSM (patient × stage).

## Modality relationships (verified)

- snRNA (GSE308103) and Visium (GSE307534) are **separate accessions**. They
  share patients but are **NOT cell-matched**. Relationship = `study_associated`
  / at most `same_donor` when a patient id genuinely appears in both — never
  `same_observation`.
- Deconvolution (Tangram) associates typed context components with Visium
  spots/niches; components are **inferred**, not observed cells.

## Processed artifacts (adapter-consumable parquets)

| Artifact | Rows | Role |
|---|---|---|
| `processed/features/niche_hlca_features.parquet` | 639,816 niches | **semantic (z_sem)** — HLCA-projected epithelial-state similarity/deviation scores |
| `processed/features/niche_luca_features.parquet` | (LuCA latent) | alternate semantic candidate |
| `processed/tangram/spatial_tangram_celltype_scores.parquet` | 639,816 | **typed context components** (backend=`tangram`): 9 cell-type scores |
| `processed/features/lesion_evo_features.parquet` | 56 lesions | **regulatory (r)** — WES/evolutionary features |
| `processed/features/niche_tokens_full.parquet` | 639,816 | niche `tok_smooth_*` composition (ontology basis) |

Giant objects NOT loaded by the adapter: `interim/anndata/snrna/snrna_full.h5ad`
(~19 GB), `spatial_full.h5ad` (~36 GB). The parquets are the model-ready inputs.

## Semantic-feature candidate (verified)

`niche_hlca_features.parquet` columns include `hlca_normal_likeness_score`,
`hlca_deviation_from_normal_score`, `hlca_lineage_fidelity_score`,
`hlca_max_state_similarity`, `hlca_topk_entropy`,
`hlca_epithelial_like_similarity`, `hlca_immune_like_similarity`,
`hlca_stromal_endothelial_like_similarity`. These are a documented,
HLCA-projected representation of epithelial semantic state → `z_sem`. LuCA latent
is an alternate. Not an arbitrary latent.

## Sender-context ontology basis (verified)

HLCA niche tokens (`tok_smooth_*`) and Tangram score columns: `AT2`, `Basal`,
`Capillary`, `Ciliated`, `Fibroblast lineage`, `Macrophages`, `Mast cells`,
`Secretory`, `T cell lineage`.

## Regulatory-feature candidate (verified, genuinely available)

`lesion_evo_features.parquet`: 34 `evo_*` features (TMB, driver burden,
KRAS/EGFR/TP53/STK11/KEAP1/SMAD4/BRAF mutation, purity, ploidy, CNA burden,
clonal-structure metrics). Per-lesion (56 lesions). This is a legitimate
regulatory-mediator space.

**Leakage caution**: `evo_progression_risk_score` and
`evo_evidence_of_progression_link` are progression-adjacent. They are treated
strictly as source-provided regulatory-mediator features (documented provenance),
and are **never** used as CCRT targets or receiver-state derivations. Configs may
exclude them from the regulatory block if strict target-hygiene is desired.

## Coordinate units

Visium spot coordinates require the Space Ranger scalefactor (pixels → microns);
the adapter fails if the unit is unresolved (never assumes pixels == microns).
snRNA cells have no spatial coordinates and get no spatial neighborhood.

## Missing / not used

- Only **one** deconvolution backend (Tangram) is present. Other backends
  (RCTD/CARD/DestVI/cell2location/SPOTlight/TACCO) are absent → not fabricated.
- No cell-level snRNA↔Visium matching exists → never asserted.

## Decisions used by the adapter

1. Three distinct partitions: snRNA cells, Visium spots, Tangram context components.
2. Receiver states: normal / aah / ais / mia / invasive_luad (adjacent edges).
3. Donor = patient; donor-aware grouping default.
4. Semantic `z_sem` = HLCA niche features; regulatory `r` = lesion evo features.
5. Backend = `tangram` (preserved; not selected/averaged).
6. Deconvolved components are `deconvolved_context_component`, never cells; one
   backend×spot×type = one component (no abundance-proportional pseudo-cells).
