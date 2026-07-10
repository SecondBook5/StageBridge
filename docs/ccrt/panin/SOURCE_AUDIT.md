# PanIN Source Audit

**Milestone 8 — Phase A source audit.** Populated from direct inspection of the
read-only reference repository. This is not a template; every entry below was
verified against the actual source.

## Source repository

- **Path**: `~/repos/PanIN_carcinogeneisis_spatial_analysis` (read-only reference)
- **Git commit**: `8802cbd941f64ac8bd1db976b1dcffc1319270dd` ("note on process for spot selection")
- **Type**: R analysis project (`.Rproj`). **Code only** — the top level contains
  `scripts/`, `vingettes/`, `README.md`, `LICENSE`, `.Rproj`. There is **no
  `data/` directory**: `.gitignore` explicitly ignores `data/`.
- **Study**: Bell, Mitchell, Kiemen et al., "PanIN and CAF Transitions in
  Pancreatic Carcinogenesis Revealed with Spatial Data Integration."

## Data availability (CRITICAL)

**No biological data are available locally.** All datasets are hosted externally
and must be downloaded per the README:

| Dataset | Modality | Source | Expected local path (per README) |
|---|---|---|---|
| PanIN Visium (paired + extended cohorts) | spot-resolved ST | GEO **GSE254829** | `./data/spaceranger/`, `./data/coda/`, `./data/MultiLesionCohort/CODA/` |
| PanIN Xenium (5 sections) | **cell-resolved** ST | GEO **GSE267680** | `./data/xenium/PanIN*/` |
| PDAC scRNA-seq atlas | single-cell (non-spatial) | monocle3 rds (Guinn et al.) | `./data/pdac_atlas_cds.rds` |
| PDAC atlas epithelial matrix | single-cell | FertigLab/PDAC_Atlas | `./data/epiMat.mtx`, `./data/geneNames.rds`, `./data/sampleNames.rds` |
| PDAC atlas CoGAPS patterns (n=8) | learned patterns | Guinn et al. | `./data/atlas_cogaps_n8.rds` |
| Excluded / regraded spot barcodes | annotation csv | this study | `./data/cloupe/excluded_spots/*`, `./data/cloupe/grade_assignment_correction/*` |

**Consequence for Milestone 8**: the adapter is implemented as a set of
unit-testable contracts against the *verified source schema*. The real-source
integration test (`tests/ccrt/test_panin_reference_integration.py`) reports a
precise missing-data blocker when `CCRT_PANIN_SOURCE_ROOT` is supplied but the
Xenium data directories are absent. **No adapter "real-data success" is claimed
without the data present.**

## Modalities and observation units (verified)

| Platform | Observation unit | Coordinate unit | Stable ID | Evidence |
|---|---|---|---|---|
| Xenium | **cell** | **microns** | `<cell_barcode>_<sample>` | `01_Load_Xenium_Data.Rmd`: `ReadXenium(type=c("centroids","segmentations"))`, `data$microns`, `CreateCentroids`; `RenameCells(new.names = paste0(colnames(seurat), "_", sample))` |
| Visium | **spot** | pixels (Space Ranger `tissue_positions_list`) → require scale to microns | spot barcode | `extended_panin_cohort/01,03,04`; `tissue_positions_list_tissue_compositions.xlsx` |
| PDAC atlas | cell (non-spatial) | n/a | atlas cell id | README; `pdac_atlas_cds.rds` |

**Primary cell-resolved CCRT path = Xenium** (§IV: cell-resolved modality with
stable cell IDs, annotations, micron coordinates, feature vectors). Visium
spot-level inventory is preserved as metadata but is **not** the Milestone-8
model-ready path (deconvolution robustness is a later milestone).

## Xenium sections and donor derivation (verified)

`01_Load_Xenium_Data.Rmd` (with the in-script mislabel correction applied) and
`03_CAF_Typing_by_moduleScore.Rmd` name exactly five Xenium sections and their
lesion `segment`:

| Sample id | Donor (derived from prefix) | `segment` | Grade |
|---|---|---|---|
| `PanIN1131_S1A` | 1131 | `PanIN-LG1-1` | low-grade |
| `PanIN1131_S1C` | 1131 | `PanIN-LG1-2` | low-grade |
| `PanIN1132` | 1132 | `PanIN-HG1-1` | high-grade |
| `PanIN1134` | 1134 | `PanIN-HG2-1` | high-grade |
| `PanIN1142` | 1142 | `PanIN-HG3-1` | high-grade |

- **Donor id** is derivable from the `PanIN<donor>_<section>` sample-name prefix
  (donor 1131 contributes two sections). This is the grouping key for
  donor-aware splits.
- **Section id** = the full sample name.
- **Modality matching status**: Visium and Xenium are **UNMATCHED** across
  platforms. Same-donor cross-platform pairing is NOT asserted. No
  single-cell↔spatial cell-level matching exists.

## Stage / receiver-state labels (verified, cross-sectional)

Pathologist-graded epithelial states (Visium `cell_type_confirmed` after
`04_Pathologist_Annotations.Rmd`; Xenium epithelial `segment` grade):

- `normal epithelium` / normal duct
- `low_grade_PanIN`
- `high_grade_PanIN`

Progression is **cross-sectional** (grade categories), never longitudinal. No
pseudotime, trajectory, or paired transition is asserted by the source.

## Annotation labels (verified)

- **CODA tissue `cell_type`** (assigned when a component is ≥70% of a spot,
  `03_Add_CODA_annotations.Rmd`): `PanIN`, `normal epithelium`, `acinar`,
  `islet`, `fat`, `collagen`, `smooth muscle`, `nerve`, `vasculature`.
- **CAF typing (Xenium, `03_CAF_Typing_by_moduleScore.Rmd`)** — module-score
  gene sets:
  - `panCAF`: FAP, LUM, DCN, COL1A1
  - `apCAF`: CD74, HLA-DRA, HLA-DPA1, HLA-DQA1, SLPI
  - `iCAF`: CXCL1, CXCL2, CCL2, LMNA, HAS1, HAS2
  - `myCAF`: TAGLN, MYL9, TPM2, MMP11, HOPX, TWIST1, SOX4
  - CD45+ immune via `PTPRC`.

## Semantic-feature candidate (verified)

- **8 CoGAPS patterns** (`Pattern_1 … Pattern_8`) learned on the PDAC atlas
  epithelial cells (`atlas_cogaps_n8.rds`) and **projected via `projectR`** onto
  Xenium epithelial cells (`02_Pattern_Projection.Rmd`:
  `projectR(seurat@assays$SCT@scale.data, loadings = cogaps@featureLoadings)`;
  saved as `Xenium_projectedPatterns.rds` /
  `ProjectR_PDACAtlasPatterns_epithelial.rds`).
- This is a **registered, documented** representation of epithelial semantic
  state — the CCRT semantic transport space (`z_sem`). It is explicitly NOT an
  arbitrary latent (rationale recorded in `ADAPTER_CONTRACT.md`).

## Regulatory-feature candidate

- No dedicated per-cell regulatory-mediator matrix (e.g. a TF/regulon activity
  matrix) is evidenced for the Xenium cell-resolved path in the source scripts.
  The CAF/subtype **module scores** are sender-context annotations, not receiver
  regulatory features. Therefore **regulatory features are treated as
  UNAVAILABLE** for the Milestone-8 PanIN path and are omitted (not fabricated).

## Missing information / blockers

- All raw/processed matrices, coordinate tables, and annotation tables are
  **absent locally** (hosted on GEO / must be regenerated). The real-source
  integration cannot run to completion until `./data/xenium/PanIN*/` exists.
- Visium coordinate **units** require the Space Ranger scalefactor to convert
  `tissue_positions_list` pixels to microns; the adapter must not assume
  pixels==microns.

## Decisions used by the adapter (from this audit)

1. Primary model-ready path = **Xenium cell-resolved** observations.
2. Donor id derived from the `PanIN<donor>_<section>` sample prefix; donor-aware
   grouping is the default split level.
3. Receiver states = pathologist grades: normal / low_grade_PanIN / high_grade_PanIN.
4. Sender-context types = source-backed CAF subtypes + immune (+ CODA tissue
   types on the Visium inventory).
5. Semantic `z_sem` = the 8 projected CoGAPS patterns.
6. Regulatory features = unavailable (omitted).
7. Transition edges = adjacent cross-sectional grades only (see `ONTOLOGY.md` /
   `ADAPTER_CONTRACT.md`).
8. Xenium coordinates are microns; Visium requires an explicit scale factor.
