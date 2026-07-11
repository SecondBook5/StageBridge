# LUAD Adapter Contract

How verified LUAD multimodal source observations map into CCRT while preserving
the distinction between observed cells, observed spots, and inferred components.

> Accuracy caveat: this contract is source-locked to the *structure* of the local
> dataset (`SOURCE_AUDIT.md`), which is not accuracy-verified. The adapter never
> asserts the data are biologically final; it only maps verified structure.

## Governing rule

- **snRNA cells remain cells** (`snrna_cell`).
- **Visium spots remain spots** (`visium_spot`).
- **Deconvolved context components remain inferred components**
  (`deconvolved_context_component`).
CCRT may learn from all three; the adapter never pretends they are the same kind
of observation.

## Modality partitions

1. **snRNA reference** (GSE308103): molecular reference; receiver-state and
   sender-context ontology support; NO spatial neighborhoods (no coordinates;
   latent/UMAP distance is never tissue distance).
2. **Visium spatial** (GSE307534): observed spots with coordinates (microns after
   scalefactor); no cross-section neighborhoods.
3. **Tangram typed-context components**: inferred per-(spot, cell-type) components
   associated with Visium niches. One backend × spot × type = one component;
   never abundance-proportional pseudo-cells; never called "cells".

## Backend-preserving design

Only the `tangram` backend is present. It is preserved with its identity; it is
NOT selected, averaged, or combined with other backends. Every context record
and edge partition carries `backend_id`. Other backends are absent → not
fabricated. Backend comparison is a later milestone.

## Canonical states, edges, features

- States: `normal`, `aah`, `ais`, `mia`, `invasive_luad` (see `ONTOLOGY.md`).
- Edges: adjacent only (`normal→aah→ais→mia→invasive_luad`).
- Semantic `z_sem`: HLCA niche features (`niche_hlca_features.parquet`),
  `squared_euclidean`, normalization `none`. Rationale: documented HLCA-projected
  epithelial-state representation — not an arbitrary latent.
- Regulatory `r`: `lesion_evo_features.parquet` (WES/evolutionary). Genuinely
  available; progression-adjacent columns handled per `ONTOLOGY.md`.

## Continuous spatial context

Distances are exact continuous Euclidean in **microns** (Visium coordinates
scaled by the source scalefactor; fails if unresolved). Same-spot component →
distance 0; nearby-spot component → centroid distance. No bins/rings/radial
categories, no averaging of components, no cross-section/platform/backend
context. Individual typed components preserved; empty-sender handles
context-poor receivers.

## Receiver construction

Receivers are epithelial-state-resolved Visium niches/spots (observation unit
recorded). Receiver state derives from the source `stage` column, never from
target data, held-out donors, or outcome. Deconvolved epithelial components (if
used as receivers) are labeled `deconvolved_receiver_component`.

## Donor-grouped splits

Donor = patient. Donor-aware grouping default; all samples/sections/platforms/
backends of a donor stay in one fold; deterministic greedy stage balance; no
observation-level split. Sample fallback requires explicit opt-in; never called
donor-held-out.

## Target populations

Population-level target-stage niches within the same fold/platform. No one-to-one
or nearest-neighbor source↔target pairing; OT matches later. Backend affects
source-context construction, not the target-stage population identity.

## Leakage prohibitions (enforced)

Numeric feature vectors never contain: stage/lesion, donor/patient, sample,
section, platform, backend, split, target-state, transition-edge, or outcome
fields; nor coordinates. Continuous distance is the sole spatial signal.
