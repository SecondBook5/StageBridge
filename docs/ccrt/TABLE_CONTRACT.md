# CCRT Table Contract

> Architecture-lock document. This describes the *intended* on-disk table
> contract and its validation rules. No implementation modules exist yet; this
> file specifies contracts, not implemented code. Concrete column sets marked
> "(to be finalized during implementation)" are proposals, not final schemas.

StageBridge / CCRT is a grammar-conditioned neural transport framework for
estimating how typed local sender-context signals modify receiver-cell drift,
growth, and regulatory state along biological transition edges.

This document defines the **standardized on-disk table contract** that every
disease/system adapter must emit. It is the boundary between messy,
system-specific raw data and the uniform, grammar-typed representation the rest
of CCRT consumes.

## Ownership and boundaries

- **Adapters produce these artifacts.** `adapters/` (including
  `adapters/panin/` and `adapters/luad/`) translate raw external data into the
  eight standardized artifacts below plus grammar IDs. Adapters may know
  biology; they MUST NOT import model architecture, operators, or training.
- **`contracts/` validates these artifacts.** The contract layer enforces
  canonical field names, dtypes, table shapes, schema validation, forbidden
  fields, and split validity. It exists to prevent random dicts, arbitrary
  field names, disease-specific private table formats, and leakage of
  target-stage data into model inputs.
- **`data/` consumes these artifacts.** Validated tables are assembled into a
  model-ready `CCRTBatch` by `data/`, which contains no disease-specific code.
- **No disease-specific private formats are allowed.** Every system — LUAD,
  PanIN, future viral systems — emits the *same* eight artifacts with the
  *same* canonical schema. System-specific meaning lives only in the grammar
  IDs referenced from these tables and in `system_spec.yaml`, never in a
  bespoke column layout.

See also: `ARCHITECTURE_LOCK.md` (overall structure and the shared grammar),
the field/naming contract and tensor contract documents in this directory, and
the grammar specification for the authoritative definitions of the referenced
grammar IDs.

## The eight standardized adapter outputs

Every adapter run produces exactly these artifacts, by these exact names, into a
single per-system processed directory (e.g.
`$STAGEBRIDGE_DATA/processed/panin_ccrt/` or
`$STAGEBRIDGE_DATA/processed/luad_ccrt/`):

```text
receivers.parquet
sender_context.parquet
semantic_features.parquet
regulatory_features.parquet
stage_edges.parquet
samples.parquet
split_manifest.json
system_spec.yaml
```

Output roots are governed by `ARCHITECTURE_LOCK.md`. These artifacts are
generated data and MUST land under an allowed output root
(`$STAGEBRIDGE_DATA/processed/...`). They MUST NOT be written under
`stagebridge/`, `tests/`, or `~/projects/StageBridge-ccrt/data/`.

## Cross-cutting conventions

- **Format.** The six tabular artifacts are Apache Parquet. The two remaining
  artifacts are JSON (`split_manifest.json`) and YAML (`system_spec.yaml`).
- **Keys.** All identifiers are strings and are stable within a processed
  directory. `receiver_id`, `sender_id`, `sample_id`, and `edge_id` are unique
  keys in their owning tables; referential integrity across tables is a
  contract-validated invariant.
- **Grammar IDs are references, not free text.** Columns holding a
  `receiver_state`, `sender_context_type`, `signal_program`,
  `regulatory_mediator`, `transition_edge`, or `biological_system_id` value must
  contain a token that resolves against the corresponding registry in
  `system_spec.yaml` / the grammar layer. Unregistered tokens are a validation
  failure.
- **Canonical system-id column.** The biological-system identifier column is
  named `biological_system_id` (the Milestone-1 canonical name enforced by
  `stagebridge/ccrt/contracts`), never abbreviated to `system_id`.
- **Distance is continuous.** The `distance` column is a continuous scalar. It
  is NEVER discretized into rings, radial bins, or neighborhood bins. The
  tokens `ring_id`, `radial_bin`, `radius_bin`, `neighborhood_bin`, and
  `world_token` are forbidden as column names (see "Forbidden fields" below).
- **Uncertainty is first class.** Sender-context rows carry an explicit
  `uncertainty` scalar used downstream for uncertainty downweighting.
- **dtypes** below use pandas/Arrow-style names (`string`, `int32`, `int64`,
  `float32`, `float64`, `bool`, `category`). Coordinate and feature dtypes may
  be tightened during implementation.

---

## 1. `receivers.parquet`

One row per receiver cell / receiver observation. This is the anchor table:
receivers are the queries in receiver-centered attention and the units whose
behavior (drift, growth, regulatory state) CCRT estimates.

| column | dtype | required | description |
|---|---|---|---|
| `receiver_id` | `string` | yes | Unique receiver identifier (primary key). |
| `sample_id` | `string` | yes | FK into `samples.parquet`. |
| `biological_system_id` | `string` | yes | Biological system identifier; resolves against `system_spec.yaml` `biological_system_id`. |
| `receiver_state` | `string` (categorical) | yes | Receiver state grammar ID; must be in the system's `ReceiverStateOntology`. |
| `spatial_x` | `float32` | conditional | Spatial X coordinate in the sample's coordinate frame (required for spatial systems). |
| `spatial_y` | `float32` | conditional | Spatial Y coordinate. |
| `spatial_z` | `float32` | optional | Spatial Z coordinate for 3D assays; null otherwise. |
| `spatial_unit` | `string` | optional | Physical unit / scale hint for coordinates (e.g. microns); enables system-aware distance penalties. |
| `receiver_semantic_ref` | `string` | optional | Key linking this receiver to its row(s) in `semantic_features.parquet`; may be `receiver_id` itself. |
| `receiver_regulatory_ref` | `string` | optional | Key linking this receiver to `regulatory_features.parquet`. |

Additional per-receiver metadata columns (e.g. QC flags, source cell/spot IDs)
are permitted but must not collide with reserved canonical names and must not
introduce forbidden fields. **(exact optional column set to be finalized during
implementation)**

Notes:
- `receiver_state` is the receiver's *current* semantic state. The target state
  of any transition is expressed only through `stage_edges.parquet` and edge
  conditioning — never by embedding a future/target stage into this table as a
  model input (see "Forbidden fields").

---

## 2. `sender_context.parquet`

Typed local sender-context around each receiver. One row per
(receiver, sender-context element) pair. These rows populate the typed
sender-context tokens (keys/values) attended over by the receiver query, with
continuous distance modulation and uncertainty downweighting. There is no
pre-attention averaging and no binning.

| column | dtype | required | description |
|---|---|---|---|
| `receiver_id` | `string` | yes | FK into `receivers.parquet`; the receiver this context element belongs to. |
| `sender_id` | `string` | yes | Identifier of the sender-context element (cell, spot, niche summary, or deconvolution-derived context unit). |
| `sample_id` | `string` | yes | FK into `samples.parquet`; must match the receiver's `sample_id`. |
| `biological_system_id` | `string` | yes | Biological system identifier (as in `receivers.parquet`). |
| `sender_context_type` | `string` (categorical) | yes | Sender-context type grammar ID; must be in the system's `SenderContextOntology`. |
| `distance` | `float32` | yes | Continuous receiver→sender distance in the sample coordinate frame. Never binned. |
| `uncertainty` | `float32` | yes | Per-element uncertainty (e.g. deconvolution/assignment confidence), used for downweighting. Convention (higher = more/less certain) fixed during implementation. |
| `is_empty_sender` | `bool` | optional | Marks the explicit empty sender token row for receivers with no qualifying context; default `false`. |
| `sig__<signal_program>` | `float32` | conditional | One column per registered `signal_program`, holding the sender's carried signal/program intensity. Program IDs must be registered in `SignalProgramRegistry`. |

Signal/program columns:
- Signal programs are represented as a set of `sig__<signal_program>` columns
  (one per registered program) OR as a normalized long form
  `(signal_program, signal_value)` pair. The wide `sig__` form is proposed as
  canonical; **the exact encoding and column-name prefix are to be finalized
  during implementation.**
- Every signal-program token must resolve against `SignalProgramRegistry`; no
  arbitrary signal column may silently appear.

Forbidden mechanism columns in this table (hard failure): `ring_id`,
`radial_bin`, `radius_bin`, `neighborhood_bin`, `world_token`, and any
precomputed pre-attention neighborhood average masquerading as a sender row.

---

## 3. `semantic_features.parquet`

Registered semantic representation inputs used to build the semantic transport
space (`z_sem`) and, where separated, the reconstruction space (`z_rec`). This
table keeps `z_rec` and `z_sem` distinct and prevents an arbitrary latent matrix
from silently becoming the transport geometry.

| column | dtype | required | description |
|---|---|---|---|
| `receiver_id` | `string` | yes | FK into `receivers.parquet` (or `sender_id` for sender semantic features, if a `role` column is used). |
| `sample_id` | `string` | yes | FK into `samples.parquet`. |
| `biological_system_id` | `string` | yes | Biological system identifier. |
| `feature_space_id` | `string` (categorical) | yes | Which registered semantic space this row belongs to (e.g. a `z_sem` space vs. a `z_rec` space); must be registered via `SemanticFeatureRegistry`. |
| `feat__0 … feat__k` | `float32` | yes | Feature dimensions of the named semantic space. Dimensionality k is fixed by the registry entry, not hard-coded here. |

Notes:
- The set of allowed `feature_space_id` values and the mapping from raw
  representation to registered space is owned by `representations/` and
  validated against `SemanticFeatureRegistry`. Unregistered feature spaces are
  a validation failure.
- Wide `feat__i` columns vs. an array-valued column is an encoding choice
  **(to be finalized during implementation)**; either way the space must be
  registered and its dimensionality declared.

---

## 4. `regulatory_features.parquet`

Registered regulatory feature space feeding the regulatory bottleneck and the
regulatory mediator `r_i`. Kept separate from semantic features so the mediator
is an explicit, registered quantity rather than an incidental latent.

| column | dtype | required | description |
|---|---|---|---|
| `receiver_id` | `string` | yes | FK into `receivers.parquet`. |
| `sample_id` | `string` | yes | FK into `samples.parquet`. |
| `biological_system_id` | `string` | yes | Biological system identifier. |
| `regulatory_mediator` | `string` (categorical) | conditional | Registered regulatory mediator grammar ID (when features are mediator-typed); must be in `RegulatoryMediatorRegistry`. |
| `reg__0 … reg__m` | `float32` | yes | Regulatory feature dimensions. Dimensionality m fixed by the registry entry. |

Notes:
- Every regulatory mediator token must resolve against
  `RegulatoryMediatorRegistry`. The exact column layout (wide vs. array-valued,
  mediator-per-row vs. mediator-per-column) is **(to be finalized during
  implementation)**.

---

## 5. `stage_edges.parquet`

The transition edges of the system's `TransitionGraph`. One row per directed
edge. Edges are the `e` in the CCRT operator and the unit of transition-edge
conditioning; they are also the only sanctioned place to express source→target
state relationships.

| column | dtype | required | description |
|---|---|---|---|
| `edge_id` | `string` | yes | Unique transition-edge identifier (primary key); the `transition_edge` grammar ID. |
| `biological_system_id` | `string` | yes | Biological system identifier. |
| `source_state` | `string` (categorical) | yes | Source receiver state grammar ID (in `ReceiverStateOntology`). |
| `target_state` | `string` (categorical) | yes | Target receiver state grammar ID (in `ReceiverStateOntology`). |
| `edge_label` | `string` | optional | Human-readable label (e.g. `normal_lung->aah`); cosmetic only, not a model input. |
| `directed` | `bool` | optional | Whether the edge is directed; default `true`. |

Notes:
- `source_state` and `target_state` must both exist in the system's receiver
  state ontology, and every `edge_id` referenced elsewhere must exist here.
- Example edges (illustrative, live in `system_spec.yaml`, not hard-coded in the
  core): LUAD `normal_lung->aah`, `aah->ais`, `ais->invasive`; PanIN
  `normal_duct->low_grade`, `low_grade->high_grade`.

---

## 6. `samples.parquet`

One row per biological sample / section / capture. Carries the
patient/donor grouping that split validation depends on.

| column | dtype | required | description |
|---|---|---|---|
| `sample_id` | `string` | yes | Unique sample identifier (primary key). |
| `biological_system_id` | `string` | yes | Biological system identifier. |
| `patient_id` | `string` | conditional | Patient identifier; required when patient-aware splitting is used. |
| `donor_id` | `string` | conditional | Donor identifier; used when the natural grouping unit is a donor rather than a patient. |
| `assay` | `string` | optional | Assay/platform descriptor (e.g. spatial transcriptomics platform). |
| `condition` | `string` | optional | Non-outcome experimental condition / cohort descriptor. |
| `coordinate_frame` | `string` | optional | Identifier of the coordinate frame shared by receivers/senders in this sample. |

Notes:
- At least one grouping key (`patient_id` or `donor_id`) must be present so that
  splits can be patient-aware or donor-aware. `sample_id`-aware splits are
  permitted ONLY when neither patient nor donor grouping is available (see
  `split_manifest.json`).
- `condition` is a descriptive covariate, not an outcome. Outcome/response
  fields are forbidden as model inputs (see "Forbidden fields").
- Additional descriptive sample metadata is allowed **(exact optional column
  set to be finalized during implementation)**.

---

## 7. `split_manifest.json`

Declares the train / validation / test partition and, critically, the
**granularity** at which the split was performed. Split granularity must be
patient-aware, donor-aware, or (only as a fallback) sample-aware. Spot-level or
receiver-level random splits are NEVER valid for biological claims, and the
contract layer must reject a manifest whose granularity is `receiver` or `spot`.

Proposed skeleton:

```json
{
  "biological_system_id": "<biological_system_id>",
  "split_granularity": "patient",
  "grouping_key": "patient_id",
  "created_by": "<adapter name / version>",
  "seed": 0,
  "splits": {
    "train": ["<group_id>", "..."],
    "val":   ["<group_id>", "..."],
    "test":  ["<group_id>", "..."]
  },
  "sample_assignment": {
    "<sample_id>": "train",
    "<sample_id>": "val",
    "<sample_id>": "test"
  },
  "notes": "Group IDs are patient_id values when split_granularity=patient, donor_id values when split_granularity=donor, and sample_id values only when neither patient nor donor grouping is available."
}
```

Rules the contract layer enforces:

- `split_granularity` MUST be one of `patient`, `donor`, or `sample`. `sample`
  is permitted only when the system genuinely lacks patient/donor grouping.
  Values such as `receiver`, `spot`, or `random` are a hard failure.
- `grouping_key` MUST name the column in `samples.parquet` that supplies the
  group IDs listed in `splits` (`patient_id` for `patient`, `donor_id` for
  `donor`, `sample_id` for `sample`).
- The `train`, `val`, and `test` group lists MUST be disjoint (no group appears
  in two splits) — this is the leakage guard.
- Every `sample_id` in `sample_assignment` must exist in `samples.parquet`, and
  its assigned split must be consistent with its group's membership in `splits`.
- `test_split_label` MUST NOT appear as a per-receiver column in any parquet
  (see "Forbidden fields"); split membership lives here in the manifest, not
  inside model-input tables.

The exact optional key set (e.g. cross-validation folds, nested splits) is
**(to be finalized during implementation)**.

---

## 8. `system_spec.yaml`

A serialized `BiologicalSystemSpec` — the biological-meaning layer for this
system. It declares the grammar vocabularies and registries that every grammar
ID in the parquet tables must resolve against. This is what makes CCRT unified
at the grammar level while allowing each system its own vocabulary. The core
model reads only the grammar structure here; it must not hard-code any specific
disease, cell type, or axis.

Proposed skeleton (mirrors `BiologicalSystemSpec`; illustrative LUAD values
shown, PanIN/viral systems fill the same slots with their own vocabularies):

```yaml
biological_system_id: luad          # the biological_system_id used across all tables
spec_version: "0.1"                 # (versioning scheme to be finalized during implementation)

receiver_state_ontology:            # ReceiverStateOntology
  - normal_alveolar
  - reactive_epithelial
  - aah
  - ais
  - invasive_luad

transition_graph:                   # TransitionGraph (edges mirror stage_edges.parquet)
  - edge_id: normal_lung__to__aah
    source_state: normal_alveolar
    target_state: aah
  - edge_id: aah__to__ais
    source_state: aah
    target_state: ais
  - edge_id: ais__to__invasive
    source_state: ais
    target_state: invasive_luad

sender_context_ontology:            # SenderContextOntology
  - il1b_high_macrophage
  - inflammatory_myeloid
  - macrophage_general
  - fibroblast
  - endothelial
  - t_cell
  - b_cell
  - alveolar_context

signal_program_registry:            # SignalProgramRegistry
  - il1b_il1r1
  - nfkb
  - interferon
  - inflammatory_epithelial_stress
  - immune_suppression
  - angiogenesis
  - ecm_remodeling

receiver_behavior_registry:         # ReceiverBehaviorRegistry
  - semantic_transition_drift       # v_i
  - growth                          # g_i  (growth / mass / tissue-rate)

regulatory_mediator_registry:       # RegulatoryMediatorRegistry
  - []                              # system-specific mediators (to be finalized during implementation)

semantic_feature_registry:          # SemanticFeatureRegistry (registered z_sem / z_rec spaces)
  - feature_space_id: z_sem_default
    role: semantic
    dim: null                       # (to be specified during implementation)
  - feature_space_id: z_rec_default
    role: reconstruction
    dim: null                       # (to be specified during implementation)

counterfactual_perturbation_registry:   # CounterfactualPerturbationRegistry
  - []                              # allowed sender-context counterfactual perturbations (to be finalized)

hypothesis: >
  IL1B-high macrophage and inflammatory myeloid niches alter epithelial
  transition drift and growth.
```

Notes:
- The exact YAML key names and nesting are a proposal; **the finalized
  `BiologicalSystemSpec` serialization is to be finalized during
  implementation.** What is locked is that this file declares, at minimum, the
  `biological_system_id`, the receiver-state ontology, the transition graph, the
  sender-context ontology, the signal-program registry, the receiver-behavior
  registry, the regulatory-mediator registry, the semantic-feature registry, and
  the counterfactual-perturbation registry.
- Every grammar ID appearing in the six parquet tables and in
  `split_manifest.json` must resolve against a registry declared here. This is
  the feature-space / signal-program / mediator leakage guard: nothing enters
  the model as a typed quantity without being registered.

---

## Grammar slots covered by the contract

The eight artifacts jointly instantiate the ten shared grammar slots. For
reference (verbatim), the slots are:

```text
biological_system_id
receiver_state
transition_edge
sender_context_type
signal_program
distance
uncertainty
receiver_behavior
regulatory_mediator
context_residual_effect
```

Mapping (informative):

```text
biological_system_id   -> biological_system_id (all tables) / system_spec.yaml
receiver_state         -> receivers.receiver_state
transition_edge        -> stage_edges.edge_id / transition_graph
sender_context_type    -> sender_context.sender_context_type
signal_program         -> sender_context.sig__<signal_program>
distance               -> sender_context.distance (continuous, never binned)
uncertainty            -> sender_context.uncertainty
receiver_behavior      -> receiver_behavior_registry (drift v_i, growth g_i); a model output
regulatory_mediator    -> regulatory_features.regulatory_mediator / registry
context_residual_effect-> a model OUTPUT of the decomposition, not an input column
```

`receiver_behavior` and `context_residual_effect` are produced by the CCRT
operator and evaluation layers; they are not adapter-supplied input columns.

---

## Forbidden fields (model-input leakage)

The following fields MUST NEVER appear as **model inputs** in any of the tables
above. They may exist ONLY as explicitly separated training *targets*, handled
outside `CCRTBatch` model inputs, never joined into a model-input table:

```text
target_stage_expression
future_expression
outcome_label
patient_response
test_split_label
```

The contract layer must reject any parquet whose columns include one of these as
a model-input column. Split membership is expressed only in
`split_manifest.json` (not via a `test_split_label` column); target-stage
information is expressed only through `stage_edges.parquet` conditioning and
separately-held targets (not by embedding target-stage expression into
`receivers.parquet`).

## Forbidden implementation terms

These tokens MUST NOT appear as column names, keys, or mechanisms anywhere in
the standardized tables (they appear here only as the forbidden list):

```text
world_token
ring_id
radial_bin
radius_bin
neighborhood_bin
```

Also forbidden as *mechanisms* embedded in the tables: rings, radial bins, and
pre-attention neighborhood averaging. Local sender influence is expressed as
typed sender-context rows with a continuous `distance`, consumed by
receiver-centered attention — not as binned or pre-averaged neighborhoods.

---

## Validation summary (what `contracts/` checks)

- **Presence & naming:** all eight artifacts exist with exact names; required
  canonical columns present with correct dtypes; no forbidden field names.
- **Referential integrity:** every FK (`sample_id`, `receiver_id`, `edge_id`,
  `sender_id`) resolves; `sender_context.sample_id` matches the receiver's
  `sample_id`.
- **Grammar resolution:** every `receiver_state`, `sender_context_type`,
  `signal_program`, `regulatory_mediator`, `edge_id`/`transition_edge`, and
  `feature_space_id` resolves against the registries in `system_spec.yaml`.
- **Distance is continuous:** `distance` is a scalar float; no binning columns
  or binned mechanisms present.
- **Split validity:** `split_manifest.json` uses patient/donor/sample
  granularity only; train/val/test group lists are disjoint; sample assignments
  are consistent.
- **Leakage guard:** no forbidden model-input-leakage field is present as a
  model input in any table.
