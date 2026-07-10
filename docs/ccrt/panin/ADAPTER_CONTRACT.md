# PanIN Adapter Contract

How verified PanIN source observations map into CCRT. The source repository
determines what data exist (`SOURCE_AUDIT.md`); this contract determines how they
are represented. The adapter never invents biological pairing, stage structure,
spatial units, or feature meaning.

## Canonical states and edges

- Biological system id: `panin_progression`.
- Receiver states: `normal_duct`, `low_grade_panin`, `high_grade_panin`.
- Transition edges (adjacent, cross-sectional):
  `normal_duct__to__low_grade_panin`, `low_grade_panin__to__high_grade_panin`.

## Receiver inclusion

- A cell is a receiver iff its **source pathologist-confirmed grade** maps to a
  canonical receiver state. Marker-only guessing is not used when a validated
  source annotation exists.

## Sender inclusion

- Sender-context types are source-backed annotations (`ONTOLOGY.md`): CAF /
  apCAF / iCAF / myCAF / immune for the Xenium path; CODA tissue types for the
  Visium inventory. Adipose (`fat`) is excluded (source excludes it).

## Platform partitioning

- Xenium (cell) and Visium (spot) are **separate platform partitions** and are
  never combined in one CCRT batch.
- The Milestone-8 model-ready path uses **Xenium cell-resolved** observations.
- Neighborhoods are built **within (sample, section, platform, observation-unit)**
  only — never across donors, samples, sections, or platforms.

## Continuous spatial context

- Coordinates are converted to **microns** before any distance computation
  (Xenium centroids are already microns; Visium requires an explicit scale
  factor and fails if the unit is unresolved).
- Distance is the exact continuous Euclidean distance. **No bins, rings, radial
  categories, or pre-attention averaging.** Individual sender elements are
  preserved; the model's empty-sender element handles context-poor receivers.

## Target-population construction

- For edge `source_state → target_state`, the source population is the receiver
  cells in `source_state` and the target population is the receiver cells in
  `target_state`, **within the same fold and same platform partition**.
- Target populations are population-level. **No one-to-one source↔target pairing,
  no nearest-neighbor pairing, no longitudinal matching.** Optimal transport
  performs population matching later. Target row order is irrelevant.

## Split grouping

- **Donor-aware by default.** Donor id is derived from the
  `PanIN<donor>_<section>` sample prefix. All samples/sections/platforms of a
  donor stay in one fold. Deterministic greedy assignment with approximate stage
  balance; no `hash()`, no observation-level random split.
- Sample-level grouping requires an explicit opt-in and is never called
  "donor-held-out".

## Feature-space definitions

| Feature space | Role | Source | Metric | Normalization |
|---|---|---|---|---|
| receiver features | receiver features | Xenium SCT-scaled expression (configured gene panel) | — | none (fold normalization is later) |
| sender features | sender features | Xenium SCT-scaled expression for sender cells | — | none |
| **semantic (`z_sem`)** | semantic | **8 projected CoGAPS patterns** (`Pattern_1..8`) | `squared_euclidean` | `none` | 
| regulatory | regulatory | **unavailable — omitted** | — | — |

Semantic rationale: the CoGAPS patterns are a documented, registered
representation of epithelial transcriptional state projected from the PDAC atlas
(`02_Pattern_Projection.Rmd`). They are explicitly chosen as `z_sem` because they
are the study's canonical epithelial state coordinates — not an arbitrary latent.

## Leakage prohibitions (enforced)

Numeric feature vectors must never contain: stage/grade, donor, patient, sample,
section, platform, split, target-state, transition-edge, or outcome fields; nor
spatial coordinates. These are metadata/provenance only. Continuous distance is
the sole spatial signal, applied per sender token inside attention.
