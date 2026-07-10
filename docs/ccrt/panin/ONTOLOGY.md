# PanIN Ontology Mapping

Source-backed mapping from actual PanIN reference labels to canonical CCRT
grammar IDs. Biological system id: **`panin_progression`**. Every mapping cites
source evidence; unknown labels are handled explicitly (never silently bucketed).

## Receiver states (epithelial grades)

Receivers are epithelial / ductal / premalignant-epithelial observations
supported by the pathologist-confirmed grade annotation.

| Source label | Canonical CCRT id | Role | Included | Evidence |
|---|---|---|---|---|
| `normal epithelium` / normal duct | `normal_duct` | receiver_state | yes | `04_Pathologist_Annotations.Rmd` palette + `cell_type_confirmed` |
| `low_grade_PanIN` | `low_grade_panin` | receiver_state | yes | `04_Pathologist_Annotations.Rmd` (`"low_grade_PanIN"`) |
| `high_grade_PanIN` | `high_grade_panin` | receiver_state | yes | `04_Pathologist_Annotations.Rmd` (`"high_grade_PanIN"`) |

## Transition edges (adjacent, cross-sectional)

Ordering is by pathological grade progression documented in the study
(normal → low-grade → high-grade). **Adjacent edges only** — no all-pairwise
edges, no longitudinal claim.

| Canonical edge id | source_state | target_state | Ordering evidence |
|---|---|---|---|
| `normal_duct__to__low_grade_panin` | `normal_duct` | `low_grade_panin` | study grade ordering (abstract: PanIN progression) |
| `low_grade_panin__to__high_grade_panin` | `low_grade_panin` | `high_grade_panin` | study grade ordering |

## Sender-context types

Source-backed CAF subtypes + immune (Xenium module-score annotations) and CODA
tissue types (Visium inventory). Canonical ids are stable; original labels are
preserved in provenance.

| Source label | Canonical CCRT id | Role | Included | Evidence |
|---|---|---|---|---|
| panCAF (FAP,LUM,DCN,COL1A1) | `caf` | sender_context_type | yes | `03_CAF_Typing_by_moduleScore.Rmd` `CAF_modscore_genes` |
| apCAF (CD74,HLA-DRA,…) | `apcaf` | sender_context_type | yes | `03_CAF_Typing` `apCAF_modscore_genes` |
| iCAF (CXCL1,CXCL2,CCL2,…) | `icaf` | sender_context_type | yes | `03_CAF_Typing` `iCAF_modscore_genes` |
| myCAF (TAGLN,MYL9,…) | `mycaf` | sender_context_type | yes | `03_CAF_Typing` `myCAF_modscore_genes` |
| CD45+ (PTPRC) | `immune` | sender_context_type | yes | `03_CAF_Typing` `CD45_pos` |
| CODA `collagen` / `smooth muscle` (stroma) | `ecm_rich_stroma` | sender_context_type | yes | CODA tissue types (`03_Add_CODA_annotations.Rmd`) |
| CODA `acinar` | `acinar_context` | sender_context_type | yes | CODA tissue types |
| CODA `islet` | `islet_context` | sender_context_type | yes | CODA tissue types |
| CODA `nerve` | `neural_context` | sender_context_type | yes | CODA tissue types |
| CODA `vasculature` | `endothelial` | sender_context_type | yes | CODA tissue types |
| CODA `fat` | — | — | **excluded** | adipose spots explicitly excluded in `04_Pathologist_Annotations.Rmd` (`"fat"` exclusion) |

Notes:
- `caf` (panCAF) and its subtypes (`apcaf`/`icaf`/`mycaf`) coexist: panCAF is the
  gating identity, subtypes are refinements. The adapter maps whichever
  source-provided annotation column is configured; both are legitimate
  sender-context types with distinct canonical ids.
- The **Xenium primary path** uses the CAF/immune sender types (cell-resolved);
  CODA tissue types belong to the Visium spot-level inventory.

## Signal programs

The source study frames CAF signaling (inflammatory → proliferation) but does
not export a discrete per-cell `signal_program` matrix for the Xenium path.
A single generic registered program `panin_context_signal` is declared so the
grammar validates; specific programs are added only if a source matrix is
confirmed. (Not fabricated as per-cell features.)

## Regulatory mediators

**None available** for the cell-resolved path (see `SOURCE_AUDIT.md`). Omitted,
not fabricated.

## Unknown-label policy

- **Strict mode (default)**: any source annotation not in this table fails the
  ontology mapping (no silent `other`/`unknown_context`).
- **Non-strict mode**: an unmapped label may be *explicitly excluded* with a
  recorded reason; it is never silently coerced to a catch-all id.

## Semantic representation

Semantic state (`z_sem`) = the 8 projected CoGAPS patterns
(`Pattern_1 … Pattern_8`). This is a receiver-state representation, not a
sender-context ontology entry. Registered in `features.py` /
`ADAPTER_CONTRACT.md`.
