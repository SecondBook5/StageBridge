# CCRT Architecture Lock

> This is the master entry point for the StageBridge / CCRT architecture-lock
> documentation set. It defines the locked design contracts and the intended
> structure of the system. It is a **design-lock** document: it describes
> contracts and intended structure, not implemented modules. No implementation
> code is described as existing.

---

## 1. Status

```
Document:  CCRT Architecture Lock (MASTER)
Version:   1.0
Date:      2026-07-08
Status:    LOCKED
Repository: /home/ajbook/projects/StageBridge-ccrt
Branch:     ccrt-full
Phase:      Architecture lock (design-lock). No implementation modules exist.
```

**LOCKED** means: the core claim, the shared grammar, the ten grammar slots, the
model operator, the source-architecture tree, the import boundaries, the output
roots, the forbidden terms, and the leakage rules stated in this document set are
fixed. Implementation must conform to them. Changes require an explicit,
versioned revision of this document set — not an ad hoc code decision.

At lock time, the only Python file that exists in the CCRT package is
`stagebridge/ccrt/__init__.py` (a package marker). No other `.py` files exist.
This document intentionally describes **contracts and intended structure**, never
"implemented modules."

---

## 2. Core claim — grammar-level unification, NOT biological equality

**CCRT is unified at the GRAMMAR level, not at the cell-type or biology level.**

State this forcefully, because it is the single most misunderstood point of the
framework:

- The claim is **NOT** that LUAD, PanIN, and viral infection share the same
  biology. They do not. IL1B-high macrophages in LUAD are not biologically
  equivalent to CAF/ECM programs in PanIN, and neither is equivalent to
  interferon-high epithelium in a viral system.
- The claim **IS** that different biological systems express their dynamics
  through the **same transition grammar**: typed sender-context signals modify
  receiver-cell behavior along defined transition edges.

This is what gives CCRT a single, unified framework **without** pretending that
distinct diseases are the same underlying process. The biology stays
system-specific; the *grammar* — the structure of the statement being made — is
shared. Every system instantiates the same slots with its own vocabulary.

One-sentence framing (verbatim):

> "StageBridge / CCRT is a grammar-conditioned neural transport framework for
> estimating how typed local sender-context signals modify receiver-cell drift,
> growth, and regulatory state along biological transition edges."

---

## 3. The shared grammar

Every biological system CCRT models is expressed as one sentence with the same
structure. The shared grammar sentence (verbatim):

> "In biological system S, sender-context type C carrying signal/program P at
> continuous distance d from receiver R modifies receiver behavior B along
> transition edge E through regulatory mediator M, with a context-residual
> effect."

That sentence decomposes into exactly ten shared grammar slots. These ten slots
are the same for every system; only the vocabulary that fills them changes.

The ten shared grammar slots (exact list, exact order):

```
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

These ten slots are the invariant contract of CCRT. See
[`GRAMMAR_CONTRACT.md`](GRAMMAR_CONTRACT.md) for how the slots are formalized in
the `grammar/` package (`BiologicalSystemSpec`, `TransitionGraph`,
`ReceiverStateOntology`, `SenderContextOntology`, `SignalProgramRegistry`,
`ReceiverBehaviorRegistry`, `RegulatoryMediatorRegistry`,
`CounterfactualPerturbationRegistry`).

---

## 4. What is shared vs. what is system-specific

### 4.1 Shared CCRT core (system-agnostic)

The core is written once and must never mention any disease, cell type, or
malignancy. It provides:

- receiver-centered attention
- typed sender-context tokens
- continuous distance modulation
- uncertainty downweighting
- empty sender token
- semantic transport geometry
- context-residual decomposition
- transition-edge conditioning
- sender-context counterfactual perturbation
- stability diagnostics

### 4.2 System-specific vocabularies (live in `BiologicalSystemSpec`, NOT the core)

The vocabularies below are **illustrative**. They live in per-system
`BiologicalSystemSpec` objects (see [`GRAMMAR_CONTRACT.md`](GRAMMAR_CONTRACT.md))
and are produced by disease-specific adapters (see the core pipeline in Section 9
and [`DIRECTORY_PURPOSES.md`](DIRECTORY_PURPOSES.md)). They must **never** be
hard-coded into the core.

**LUAD**

```
receiver_states:
  normal_alveolar, reactive_epithelial, aah, ais, invasive_luad
transition_edges:
  normal_lung->aah, aah->ais, ais->invasive
sender_context_ontology:
  il1b_high_macrophage, inflammatory_myeloid, macrophage_general,
  fibroblast, endothelial, t_cell, b_cell, alveolar_context
signal_programs:
  il1b_il1r1, nfkb, interferon, inflammatory_epithelial_stress,
  immune_suppression, angiogenesis, ecm_remodeling
hypothesis:
  IL1B-high macrophage and inflammatory myeloid niches alter epithelial
  transition drift and growth.
```

**PanIN**

```
receiver_states:
  normal_duct, low_grade_panin, high_grade_panin
transition_edges:
  normal_duct->low_grade, low_grade->high_grade
sender_context_ontology:
  caf, apcaf, icaf, mycaf, ecm_rich_stroma, ductal_epithelial_context,
  acinar_adm_context, myeloid, lymphoid, endothelial,
  coda_lesion_geometry_context
signal_programs:
  caf_inflammatory_program, ecm_remodeling, tgfb_stromal_program,
  ductal_stress, adm_panin_program, proliferation
hypothesis:
  CAF/ECM/stromal context alters ductal epithelial PanIN transition drift,
  growth, and tissue architecture.
```

**Future viral system**

```
receiver_states:
  uninfected, early_infected, interferon_high, lytic_dead,
  persistent_latent, repaired_recovered
sender_context_ontology:
  infected_neighboring_cells, interferon_high_epithelial, macrophages,
  t_cells, stromal_damage_context, viral_burden_context
signal_programs:
  viral_rna_protein_burden, interferon, cytokine_signaling,
  immune_killing, tissue_damage, repair
hypothesis:
  infected and immune contexts redirect receiver cells among infection,
  antiviral, death, and repair states.
```

The point of the three examples is not their content but their *shape*: each one
fills the same ten slots with its own biology.

---

## 5. What the model must NOT hard-code

The shared core (the `operators/`, `sender_context/`, `transport/`,
`representations/`, `data/`, `training/`, `evaluation/`, and `plotting/`
packages) must be biology-blind. It must **not** hard-code any of:

```
LUAD          PanIN         macrophage     CAF
cancer        virus         malignancy axis
```

All of these belong exclusively in system-specific `BiologicalSystemSpec`
instances and in the disease-specific `adapters/` (see
[`AGENT_BOUNDARIES.md`](AGENT_BOUNDARIES.md) for the import boundaries that
enforce this). A biology term appearing anywhere in the core is a lock violation.

Additionally, the following five terms are **forbidden implementation terms**.
They may appear in documentation **only** as forbidden terms, and must never be
used as identifiers, fields, or mechanisms:

```
world_token
ring_id
radial_bin
radius_bin
neighborhood_bin
```

Also forbidden **as mechanisms**: rings, radial bins, and pre-attention
neighborhood averaging. Local influence is expressed through receiver-centered
attention with continuous distance modulation — never binned geometry. See
[`LEAK_PREVENTION.md`](LEAK_PREVENTION.md) and
[`TENSOR_CONTRACT.md`](TENSOR_CONTRACT.md) for the forbidden-field contract, and
Section 7 (Method stack mapping) for the AMICI-inspired attention rationale.

---

## 6. The model equation (core operator)

CCRT models receiver-cell behavior as an **intrinsic** term plus a
**context-residual** term. This is the definitional structure of the framework.

Core operator:

```
b_i^(e,S) = b_self(z_i, e, S) + delta_b_ctx(z_i, C_i, e, S)
```

where:

```
S    = biological system
e    = transition edge
z_i  = receiver semantic state
C_i  = typed sender-context set around receiver i
b    = receiver behavior
```

### 6.1 The two behaviors carried by the thesis

Semantic transition drift:

```
v_i^(e,S) = v_self(z_i, e, S) + delta_v_ctx(z_i, C_i, e, S)
```

Growth / mass / tissue-rate:

```
g_i^(e,S) = g_self(z_i, e, S) + delta_g_ctx(z_i, C_i, e, S)
```

### 6.2 With a regulatory bottleneck

A shared regulatory mediator `r_i` routes context effects through a low-rank
bottleneck before they act on drift and growth:

```
v_i^(e,S) = v_self(z_i, e, S) + B_(e,S) r_i + r_theta(z_i, C_i, e, S)
g_i^(e,S) = g_self(z_i, e, S) + a_(e,S)^T r_i + rho_theta(z_i, C_i, e, S)
```

### 6.3 Key outputs are a DECOMPOSITION, not just a prediction

The scientific product of CCRT is a decomposition of behavior into its intrinsic
and context-driven parts. The outputs are:

```
self-intrinsic transition            (what the cell does on its own)
full context-conditioned transition  (what the cell does in its actual context)
context-residual drift               (delta_v_ctx)
context-residual growth              (delta_g_ctx)
regulatory mediator                  (r_i and its routing)
sender-context attribution           (which sender types/programs drive the residual)
counterfactual context perturbation  (behavior under altered/removed context)
semantic transport stability         (robustness of the semantic matching geometry)
```

All hyperparameters implied by these equations (bottleneck rank, embedding
dimensions, neighbor counts, kernel widths, layer counts) are
**(to be specified during implementation)**. This lock fixes the *structure* of
the operator, not its numeric settings. The operator formalization lives in the
`operators/` package; see its role in [`DIRECTORY_PURPOSES.md`](DIRECTORY_PURPOSES.md)
and `stagebridge/ccrt/operators/PURPOSE.md`.

---

## 7. Method stack mapping (conceptual references only)

CCRT draws conceptual precedent from a small set of read-only reference
repositories. These are **never vendored, never copied, and never added as
submodules**. They inform design; they are not dependencies. The per-package
influence is also summarized in [`AGENT_BOUNDARIES.md`](AGENT_BOUNDARIES.md)
(reference-repo policy).

| Reference | Location | Contribution to CCRT |
|-----------|----------|----------------------|
| **AMICI** | `~/repos/amici` | Receiver-centered attention over nearby sender context: receiver-as-query, sender/neighbors as keys/values, continuous distance modulation, empty sender token, sparsity, ablation-based interpretation. Shapes `sender_context/`. |
| **OSDR** | `~/repos/osdr` | Spatial snapshot to tissue-rate / growth-death dynamics; phase-portrait thinking. Shapes the growth behavior `g` in `operators/`. |
| **Semantic transport generation** | `~/repos/semantic-transport-generation` | Reconstruction space (`z_rec`) and semantic matching space (`z_sem`) must be **separated**; Sinkhorn matching in semantic space; effective rank; matching stability. Shapes `representations/` and `transport/`. |
| **Cell behavior grammar paper** | (conceptual precedent) | Biology becomes computable via a grammar of cell types, signals, behaviors, and rules. Conceptual precedent only. |
| **PanIN spatial analysis** | `~/repos/PanIN_carcinogeneisis_spatial_analysis` | PanIN data inventory and CODA/pathology metadata discovery. Informs `adapters/panin/` only. |
| **scPrisma** | `~/repos/scPrisma` | Spectral/topological signal ideas; optional later comparison. Not core. |

**Grammar paper vs. CCRT.** The cell behavior grammar paper defines an
expert-written rule: "signal increases/decreases behavior." CCRT is the
**learned / neural** version of that idea: "a typed sender-context signal
modifies receiver transition behavior." The grammar is the conceptual precedent;
CCRT learns the rule rather than authoring it by hand.

---

## 8. Final source architecture (15 packages)

The CCRT source lives under `stagebridge/ccrt/` as exactly fifteen packages, plus
two disease-specific adapter subpackages. Only `stagebridge/ccrt/__init__.py`
exists at lock time as a package marker; no other `.py` files exist yet.

```
stagebridge/ccrt/
  contracts/        low-level law: field names, tensor/table contracts, schema
                    validation, forbidden fields, split validation
  grammar/          biological meaning layer: BiologicalSystemSpec and registries
  io/               safe paths, standardized table IO, provenance manifests
  adapters/         disease-specific translation ONLY: raw data -> CCRT tables
    adapters/panin/ PanIN raw data -> BiologicalSystemSpec + standardized tables
    adapters/luad/  LUAD raw data  -> BiologicalSystemSpec + standardized tables
  representations/  z_rec, z_sem, regulatory feature space, feature registries
  sender_context/   AMICI-inspired local influence (receiver-as-query attention)
  operators/        biological CCRT operator: v/g self, full, context-residual
  transport/        semantic OT & matching: Sinkhorn, barycentric maps, stability
  data/             validated tables -> model-ready CCRTBatch (splits, collate)
  synthetic/        known-truth mechanism-recovery benchmarks
  deconvolution/    benchmarks deconvolution backends as sender-context sources
  training/         training loops, losses, checkpointing (NOT architecture)
  evaluation/       standardized metrics from model outputs / exported predictions
  plotting/         figures from evaluation outputs ONLY
  cli/              thin command-line entry points; one workflow each
```

Repo-level directories created for the CCRT effort:

```
docs/ccrt/     the architecture-lock documents themselves (this set)
configs/ccrt/  configuration files (system specs, model/training configs)
scripts/ccrt/  thin runnable scripts / entrypoints for CCRT workflows
tests/ccrt/    contract, import-boundary, forbidden-term, split, and leak tests
```

### 8.1 Directory roles (authoritative one-liners)

- **`contracts/`** — the low-level law: canonical field names, tensor contracts,
  table contracts, schema validation, forbidden fields, split validation.
  Prevents random dicts, arbitrary field names, disease-specific private table
  formats, and leakage of target-stage data into model inputs.
- **`grammar/`** — the biological meaning layer: `BiologicalSystemSpec`,
  `TransitionGraph`, `ReceiverStateOntology`, `SenderContextOntology`,
  `SignalProgramRegistry`, `ReceiverBehaviorRegistry`,
  `RegulatoryMediatorRegistry`, `CounterfactualPerturbationRegistry`. This is
  what makes CCRT unified.
- **`io/`** — safe paths, standardized table IO, provenance manifests. Prevents
  generated data entering git, random pickle outputs, and silent artifact drift.
- **`adapters/`** — disease-specific translation ONLY: raw external data to
  standardized CCRT tables + grammar IDs. May know biology. MUST NOT import model
  architecture, operators, or training.
- **`representations/`** — reconstruction space (`z_rec`), semantic transport
  space (`z_sem`), regulatory feature space, feature registries. Keeps `z_rec`
  and `z_sem` separate.
- **`sender_context/`** — AMICI-inspired local influence: receiver-as-query
  attention, typed sender-context keys/values, continuous distance kernels,
  stage/system-aware distance penalties, uncertainty downweighting, empty sender
  token, sparsity losses, signed sender effects. NEVER rings, radial bins, or
  pre-attention averaging.
- **`operators/`** — the biological CCRT operator: `v_self`, `v_full`,
  `delta_v_ctx`, `g_self`, `g_full`, `delta_g_ctx`, regulatory bottleneck,
  context-residual decomposition. MUST NOT import adapters, deconvolution
  backends, or plotting.
- **`transport/`** — semantic OT & matching: Sinkhorn loss, barycentric maps,
  semantic transport loss, effective rank, matching stability, geometric
  alignment, context-effect stability. Estimated in semantic biological space,
  not raw expression or an arbitrary latent.
- **`data/`** — validated CCRT tables to model-ready `CCRTBatch`: dataset
  loading, collate, padding, masking, patient/donor-aware splits, batch
  validation. NO disease-specific code.
- **`synthetic/`** — known-truth mechanism-recovery benchmarks: null context,
  drift-only, growth-only, mixed drift/growth, regulatory-mediated,
  distance-specific sender effects, wrong-context negative controls.
- **`deconvolution/`** — benchmarks deconvolution backends as sources of
  sender-context construction: DestVI, cell2location, RCTD, CARD, Tangram, TACCO,
  SPOTlight, marker/program scoring. MUST NOT change the CCRT
  table/tensor/grammar contract.
- **`training/`** — training loops, losses, checkpointing. Does NOT define
  architecture.
- **`evaluation/`** — standardized metrics from model outputs and exported
  predictions. Does NOT read raw disease data directly.
- **`plotting/`** — figures from evaluation outputs ONLY. Does NOT import
  adapters or training.
- **`cli/`** — thin command-line entry points. Each CLI file calls one workflow,
  with minimal logic.

### 8.2 Import boundaries (future tests must FAIL if violated)

The import graph is part of the lock. Tests under `tests/ccrt/` must **fail** if
any of these boundaries is violated:

```
FAIL if: operators imports adapters
FAIL if: sender_context imports adapters
FAIL if: transport imports adapters
FAIL if: adapters import operators/model
FAIL if: deconvolution imports operators/model
FAIL if: plotting imports adapters
FAIL if: plotting imports training
```

Allowed dependency directions:

```
adapters   -> grammar, contracts, io
operators  -> grammar, contracts, representations, sender_context
training   -> operators, data, grammar
```

Full boundary rationale in [`AGENT_BOUNDARIES.md`](AGENT_BOUNDARIES.md).

---

## 9. The core CCRT pipeline

The end-to-end flow, with **no hidden side channels** (verbatim):

```
external data
  -> adapter inventory
  -> BiologicalSystemSpec + standardized CCRT tables
  -> contract validation
  -> CCRTBatch
  -> semantic representation registry
  -> typed sender-context attention
  -> context-residual drift/growth/regulatory operator
  -> semantic transport matching
  -> counterfactual context perturbation
  -> evaluation / stability / plots
```

Each disease-specific adapter must emit exactly these eight standardized
artifacts (exact list):

```
receivers.parquet
sender_context.parquet
semantic_features.parquet
regulatory_features.parquet
stage_edges.parquet
samples.parquet
split_manifest.json
system_spec.yaml
```

### 9.1 Output roots

Generated data lives outside git. Allowed primary generated output roots:

```
$STAGEBRIDGE_DATA/raw/
$STAGEBRIDGE_DATA/interim/
$STAGEBRIDGE_DATA/processed/
  (e.g. $STAGEBRIDGE_DATA/processed/panin_ccrt/,
        $STAGEBRIDGE_DATA/processed/luad_ccrt/)
$STAGEBRIDGE_RESULTS/
```

Forbidden primary output roots:

```
stagebridge/
tests/
~/projects/StageBridge-ccrt/data/
```

### 9.2 Leakage and registration guards

These are hard rules enforced by `contracts/` and validated in `tests/ccrt/`
(details in [`LEAK_PREVENTION.md`](LEAK_PREVENTION.md), with field-level schemas
in [`TABLE_CONTRACT.md`](TABLE_CONTRACT.md) and
[`TENSOR_CONTRACT.md`](TENSOR_CONTRACT.md)):

- **Forbidden model-input leakage.** The following may **never** appear in
  `CCRTBatch` model inputs — they may exist **only** as explicitly separated
  training targets:

  ```
  target_stage_expression
  future_expression
  outcome_label
  patient_response
  test_split_label
  ```

- **Split validation.** Biological claims require patient-aware splits or
  donor-aware splits; sample-aware splits are allowed **only** when
  patient/donor identity is unavailable. **Never** spot-level or receiver-level
  random splits for biological claims.

- **Feature-space registration.** Semantic spaces, signal programs, and
  regulatory mediators must be registered through `SemanticFeatureRegistry`,
  `SignalProgramRegistry`, and `RegulatoryMediatorRegistry`. No arbitrary latent
  matrix may silently become transport geometry.

---

## 10. Positioning

**CCRT is NOT a LUAD model.** It is not a PanIN model, and it is not a virus
model. It is not a disease-classifier dressed up as dynamics.

**CCRT IS a grammar-conditioned neural transport framework.** It happens to be
demonstrated on LUAD and PanIN (with a viral system as a future target), but its
identity is the shared grammar and the shared operator, not any one disease.

The thesis result — the thing CCRT is *for* — is this:

> **Separating what cells intrinsically do from what context makes them do.**

That separation is exactly the `b_self` vs. `delta_b_ctx` decomposition of
Section 6. Every disease demonstration is a fresh instantiation of the same
grammar and the same decomposition; the framework is validated by how cleanly it
recovers that separation (see the mechanism-recovery benchmarks in `synthetic/`).

---

## 11. Implementation order

Implementation proceeds in strict order. **Do NOT implement everything at once.**
Each stage depends on the contracts locked by the previous one.

```
1. Architecture lock            (this document set) — DONE at Version 1.0
2. Grammar                      (BiologicalSystemSpec + registries)
3. Contracts                    (field/tensor/table contracts, forbidden fields)
4. Tensor / table validation    (schema + split + leakage validators)
5. Synthetic                    (known-truth mechanism-recovery benchmarks)
6. Adapters                     (PanIN and LUAD raw data -> standardized tables)
7. Training / evaluation        (loops, losses, metrics, stability diagnostics)
8. Biological runs              (the actual LUAD / PanIN scientific results)
```

Rationale: grammar and contracts must be stable before any tensors are built;
validation must exist before any batch is trusted; synthetic mechanism-recovery
must pass before adapters feed real data; training/evaluation must be standardized
before biological claims are made. Skipping ahead reintroduces exactly the
leakage, binning, and side-channel risks this lock is designed to prevent.

---

## 12. Companion documents

This master document is the entry point. The full architecture-lock set consists
of this file plus six companion documents in `docs/ccrt/`:

```
GRAMMAR_CONTRACT.md    The shared grammar formalized: the ten slots, the grammar
                       sentence, BiologicalSystemSpec, TransitionGraph,
                       ontologies, and registries; per-system vocabularies.
DIRECTORY_PURPOSES.md  The 15-package layout, directory roles, the core pipeline
                       flow, and the allowed/forbidden import summary.
TENSOR_CONTRACT.md     The CCRTBatch tensor container: canonical tensor field
                       names, receiver/sender-context tensors, masking, and the
                       forbidden tensor fields and mechanisms.
TABLE_CONTRACT.md      The eight standardized adapter outputs, their parquet/JSON/
                       YAML schemas, canonical field names, and split manifest.
AGENT_BOUNDARIES.md    The worktree map, import-boundary rules future tests must
                       enforce, forbidden terms, and the reference-repo policy
                       (read-only, no vendoring, no submodules).
LEAK_PREVENTION.md     The four leak classes (output-location, split, model-input,
                       feature-space) and the tests in tests/ccrt/ that enforce
                       each one.
```

The method-stack mapping (AMICI, OSDR, semantic transport generation, cell
behavior grammar), the core pipeline, and the context-residual operator do not
have standalone documents — they are defined inline in this master document
(Sections 7, 9, and 6 respectively) and cross-referenced from the companions.

Cross-references to these documents appear throughout this file. In any conflict,
this master document governs the core claim, the ten slots, the operator
structure, the source tree, the import boundaries, the output roots, and the
forbidden terms; the companions govern their respective details.

---

## 13. Status

```
CCRT Architecture Lock — MASTER
Version 1.0 — 2026-07-08 — Status: LOCKED
```

The design contracts above are fixed. Implementation must conform. Any change
requires a versioned revision of this document set.
