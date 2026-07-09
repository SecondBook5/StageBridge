# CCRT Leak Prevention

This document is the consolidated **leakage contract** for StageBridge / CCRT
(Context-Residual Transport). It enumerates every class of leak the architecture
lock forbids, states the rule for each, names the concrete failure it prevents,
and specifies the test that enforces it. All enforcement tests live under
`tests/ccrt/`.

StageBridge / CCRT is a grammar-conditioned neural transport framework for
estimating how typed local sender-context signals modify receiver-cell drift,
growth, and regulatory state along biological transition edges. Because the
framework is unified at the **grammar** level rather than the cell-type level,
the same leakage contract applies uniformly across every biological system
(LUAD, PanIN, future viral systems) without disease-specific exceptions.

This document is a design-lock artifact. No implementation code exists yet; the
tests described below are the specifications those future tests must satisfy.
They are written as requirements ("the test must fail if ...") rather than as
descriptions of existing modules.

Cross-references:

- Import-boundary rules are specified in detail in `AGENT_BOUNDARIES.md`.
- Contract field names, table contracts, and split-validation logic are owned by
  the `contracts/` package (see `TABLE_CONTRACT.md` and `TENSOR_CONTRACT.md`).
- Registration of feature spaces, signal programs, and regulatory mediators is
  owned by `grammar/` and `representations/` (see `GRAMMAR_CONTRACT.md`, and the
  `representations/` role in `DIRECTORY_PURPOSES.md`).

---

## Why leakage is the central risk

CCRT makes **biological claims**: that typed sender-context signals modify
receiver transition drift, growth, and regulatory state. A biological claim is
only credible if the model never saw the answer, never trained and tested on the
same patient, never silently promoted an arbitrary latent to transport geometry,
and never wrote regenerable outputs into the tracked source tree. Every leak
class below corresponds to a specific way such a claim could be quietly
invalidated. The contract is therefore enforced by tests, not by convention.

The core CCRT pipeline has **no hidden side channels**:

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

Each leak class below closes one potential side channel around this pipeline.

---

## Leak class 1 — Output-location leaks

### Rule

Primary generated outputs (processed tables, checkpoints, results, figures,
serialized artifacts) MUST be written only to the allowed output roots, which
live **outside** the git working tree. Nothing regenerable may be written into
the source tree, the test tree, or the in-repo `data/` directory.

**ALLOWED primary generated output roots (outside git):**

```
$STAGEBRIDGE_DATA/raw/
$STAGEBRIDGE_DATA/interim/
$STAGEBRIDGE_DATA/processed/      (e.g. $STAGEBRIDGE_DATA/processed/panin_ccrt/,
                                        $STAGEBRIDGE_DATA/processed/luad_ccrt/)
$STAGEBRIDGE_RESULTS/
```

**FORBIDDEN primary output roots:**

```
stagebridge/
tests/
~/projects/StageBridge-ccrt/data/
```

Path construction is centralized in `io/` (safe paths, standardized table IO,
provenance manifests). Adapters, training, evaluation, and plotting must obtain
output paths from `io/` rather than building paths ad hoc, so that the allowed
roots are the only reachable destinations.

### Failure it prevents

- Generated data (`*.parquet`, `*.h5ad`, checkpoints) entering git and bloating
  or corrupting the repository.
- Silent artifact drift, where a stale in-tree output is mistaken for a
  regenerated one.
- Random pickle / unmanaged outputs landing next to source or tests.
- Test runs polluting the tracked tree with fixtures or byproducts.

### Test that enforces it (`tests/ccrt/`)

- **Output-root test:** given the path-construction surface in `io/`, assert
  that every primary-output path resolves under one of the allowed roots
  (`$STAGEBRIDGE_DATA/{raw,interim,processed}` or `$STAGEBRIDGE_RESULTS/`). The
  test MUST **fail** if any primary output resolves under `stagebridge/`,
  `tests/`, or `~/projects/StageBridge-ccrt/data/`.
- **No-write-in-source test:** after exercising output-producing workflows in a
  sandbox, assert that no new files were created under `stagebridge/` or
  `tests/`.
- **Defense in depth:** `.gitignore` also guards these roots. It already ignores
  generated data formats (`*.parquet`, `*.h5ad`, `*.pt`, `*.ckpt`, `*.pkl`,
  ...), `data/interim/`, `outputs/`, `results/`, `runs/`, and documents that
  primary outputs live under `$STAGEBRIDGE_DATA/` and `$STAGEBRIDGE_RESULTS/`,
  never under `stagebridge/`, `tests/`, or the in-repo `data/` tree. The
  `.gitignore` is a second line of defense; the test is the first, because
  `.gitignore` cannot prevent a write, only prevent it from being tracked.

---

## Leak class 2 — Split leakage

### Rule

Train / validation / test splits used to support a biological claim MUST be
constructed at a biological grouping level so that no individual appears on both
sides of a split:

- **patient-aware splits** are required when patient identity is available;
- **donor-aware splits** are required when donor identity is the relevant unit;
- **sample-aware splits** are allowed ONLY as a fallback when patient/donor
  identity is unavailable;
- **spot-level or receiver-level random splits are NEVER acceptable** for
  biological claims.

Split assignment is materialized in the standardized adapter output
`split_manifest.json` and validated by `contracts/` before any batching. `data/`
consumes the validated manifest to build patient/donor-aware splits; it must not
re-derive splits by random spot/receiver assignment.

### Failure it prevents

- The same patient's spots appearing in both train and test, which inflates
  metrics and produces an uncredible biological claim (the model memorizes the
  patient, not the biology).
- Spatial autocorrelation leakage, where neighboring spots/receivers from one
  tissue section straddle the split boundary.

### Test that enforces it (`tests/ccrt/`)

- **Split-granularity test:** inspect `split_manifest.json` and the split logic
  in `data/`. Assert that grouping is patient-aware or donor-aware whenever the
  corresponding identifier column is present, and sample-aware only when neither
  is present. The test MUST **fail** on any spot-level or receiver-level random
  split.
- **No-overlap test:** assert that the intersection of patient/donor identifiers
  across any two of {train, val, test} is empty.
- **Fallback-justification test:** if a sample-aware split is used, assert that
  patient and donor identifiers are genuinely absent from the standardized
  tables, so the fallback cannot be selected while a stronger grouping was
  available.

---

## Leak class 3 — Model-input leakage

### Rule

The following fields MUST NEVER appear as **CCRTBatch model inputs**. They may
exist ONLY as explicitly separated training targets, never as inputs the model
conditions on:

```
target_stage_expression
future_expression
outcome_label
patient_response
test_split_label
```

`contracts/` defines the canonical CCRTBatch input schema and the forbidden-input
list; `data/` must construct CCRTBatch such that these fields are routed into a
separated target/label structure, never into the tensors the operator sees.

### Failure it prevents

- Target leakage: the model reading the answer (the target-stage or future
  expression it is supposed to predict) directly from its own inputs, producing
  trivially perfect and meaningless results.
- Label leakage: outcome / response / split labels bleeding into the feature path
  so that predictions encode the label rather than the modeled biology.

### Test that enforces it (`tests/ccrt/`)

- **Forbidden-input test:** enumerate the field/tensor names exposed as CCRTBatch
  model inputs. Assert that none of the five forbidden fields
  (`target_stage_expression`, `future_expression`, `outcome_label`,
  `patient_response`, `test_split_label`) appear among them. The test MUST
  **fail** if any forbidden field is reachable as a model input.
- **Target-separation test:** assert that where these fields exist for training,
  they are held in a structurally separate target/label container that the
  operator forward path does not consume.

---

## Leak class 4 — Feature-space leakage

### Rule

Every space or vocabulary that acquires semantic weight in the model MUST be
**registered** through the appropriate registry before use. No arbitrary latent
matrix may silently become transport geometry, a signal program, or a regulatory
mediator:

- semantic spaces are registered through **SemanticFeatureRegistry**;
- signal programs are registered through **SignalProgramRegistry**;
- regulatory mediators are registered through **RegulatoryMediatorRegistry**.

This preserves the separation the framework requires: the reconstruction space
(`z_rec`) and the semantic transport space (`z_sem`) are distinct, and transport
matching is estimated in the registered **semantic biological space**, not in raw
expression or an arbitrary latent. Registration is owned by `grammar/` /
`representations/`; `transport/` must consume only a registered semantic space.

### Failure it prevents

- An unregistered latent matrix silently being promoted to the transport geometry
  used for Sinkhorn matching, so the "semantic" transport is actually arbitrary
  and unauditable.
- `z_rec` (reconstruction) leaking in as `z_sem` (semantic matching), collapsing
  the required separation between reconstruction and semantic spaces.
- Ad hoc signal programs or regulatory mediators entering the operator without a
  registry entry, making the context-residual decomposition non-reproducible.

### Test that enforces it (`tests/ccrt/`)

- **Registration test:** assert that any semantic space consumed by `transport/`
  and any signal program or regulatory mediator consumed by `operators/`
  corresponds to an entry in `SemanticFeatureRegistry`, `SignalProgramRegistry`,
  or `RegulatoryMediatorRegistry` respectively. The test MUST **fail** if an
  unregistered space/program/mediator is used.
- **z_rec / z_sem separation test:** assert that the transport geometry is drawn
  from the registered semantic space and not from the reconstruction space, so
  `z_rec` cannot stand in for `z_sem`.

---

## Leak class 5 — Import-boundary leaks

### Rule

Package import boundaries encode the dependency law of the architecture; a
forbidden import is a structural leak (biology or backends bleeding into the
model core, or model internals bleeding into disease adapters). The following
imports MUST cause the boundary tests to **fail**:

```
FAIL if operators       imports adapters
FAIL if sender_context  imports adapters
FAIL if transport       imports adapters
FAIL if adapters        import  operators / model
FAIL if deconvolution   imports operators / model
FAIL if plotting        imports adapters
FAIL if plotting        imports training
```

Allowed dependency directions:

```
adapters       -> grammar, contracts, io
operators      -> grammar, contracts, representations, sender_context
training       -> operators, data, grammar
```

The authoritative specification of these boundaries, including rationale per
package, is in `AGENT_BOUNDARIES.md`; this section states only the leak-relevant
enforcement.

### Failure it prevents

- Disease-specific biology (LUAD/PanIN/viral vocabularies) leaking into the
  system-agnostic core (`operators/`, `sender_context/`, `transport/`), which
  would break the grammar-level unification.
- Deconvolution backends or model architecture leaking across the
  adapter/operator boundary, so that a backend choice or an operator internal
  silently changes the standardized table/tensor/grammar contract.
- Plotting reaching back into adapters or training and re-introducing raw disease
  data or training state into the figure path.

### Test that enforces it (`tests/ccrt/`)

- **Import-boundary test:** statically analyze intra-package imports and assert
  the forbidden edges above are absent and the allowed edges are the only
  cross-package dependencies. The test MUST **fail** on any forbidden import.
  See `AGENT_BOUNDARIES.md` for the full boundary matrix.

---

## Leak class 6 — Forbidden-terminology leaks

### Rule

The following implementation terms are forbidden as fields, tensor names,
functions, or realized mechanisms anywhere in the CCRT source. They may appear in
documentation ONLY as forbidden terms (as here):

```
world_token
ring_id
radial_bin
radius_bin
neighborhood_bin
```

Also forbidden as **mechanisms**: rings, radial bins, and pre-attention
neighborhood averaging. Local sender influence is expressed exclusively through
the AMICI-inspired mechanism specified in `sender_context/`: receiver-as-query
attention over typed sender-context keys/values with continuous distance
kernels, uncertainty downweighting, an empty sender token, and sparsity — NEVER
rings, radial bins, or pre-attention averaging.

### Failure it prevents

- A binned / ringed / pre-averaged spatial mechanism silently replacing the
  continuous-distance, receiver-centered attention that the framework is built
  on, which would quietly change the modeled biology and invalidate
  context-residual attribution.
- A `world_token` or similar global side channel reintroducing untyped,
  non-local context.

### Test that enforces it (`tests/ccrt/`)

- **Forbidden-term test:** scan the CCRT source for the five forbidden terms
  (`world_token`, `ring_id`, `radial_bin`, `radius_bin`, `neighborhood_bin`) as
  identifiers. The test MUST **fail** on any occurrence outside of documentation
  marked as forbidden-term references.
- **Mechanism test:** assert that `sender_context/` exposes the continuous
  distance-kernel attention surface and does not expose ring / radial-bin /
  pre-attention-averaging surfaces.

---

## Summary: leak class -> test mapping

| Leak class | Rule (short) | Enforcing test (in `tests/ccrt/`) |
| --- | --- | --- |
| 1. Output-location | Primary outputs only under `$STAGEBRIDGE_DATA/{raw,interim,processed}`, `$STAGEBRIDGE_RESULTS/`; never `stagebridge/`, `tests/`, `~/projects/StageBridge-ccrt/data/` | Output-root test + no-write-in-source test; `.gitignore` as defense in depth |
| 2. Split | patient/donor-aware required; sample-aware only as fallback; never spot/receiver-level random | Split-granularity + no-overlap + fallback-justification tests |
| 3. Model-input | 5 forbidden fields never CCRTBatch inputs, only separated targets | Forbidden-input + target-separation tests |
| 4. Feature-space | semantic spaces / signal programs / regulatory mediators must be registered; no arbitrary latent becomes transport geometry | Registration + `z_rec`/`z_sem` separation tests |
| 5. Import-boundary | forbidden cross-package imports fail | Import-boundary test (see `AGENT_BOUNDARIES.md`) |
| 6. Forbidden-terminology | 5 forbidden terms + banned mechanisms | Forbidden-term + mechanism tests |

Every rule above is a **hard contract**. A future change that violates any rule
must be caught by the corresponding test in `tests/ccrt/` before it can support a
biological claim.
