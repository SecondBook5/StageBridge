# CCRT Agent & Import Boundaries

This document is the **enforcement contract** for every human and AI agent that
works on StageBridge / CCRT across its worktrees. It is part of the
architecture-lock documentation set. No implementation code exists yet; this
document describes the boundaries that implementation **must** honor and that the
future test suite in `tests/ccrt/` **must** enforce.

StageBridge / CCRT is a grammar-conditioned neural transport framework for
estimating how typed local sender-context signals modify receiver-cell drift,
growth, and regulatory state along biological transition edges. CCRT is unified
at the **grammar** level, not the cell-type level: the same transition grammar
describes different biological systems, while system-specific biology lives only
in `BiologicalSystemSpec` and the adapters. The rules below exist to keep that
separation intact — to prevent disease biology, deconvolution backends, or
plotting concerns from leaking into the system-agnostic core.

Read this document alongside `LEAK_PREVENTION.md` (data, feature-space, and
model-input leakage) and `ARCHITECTURE_LOCK.md` (the overall locked design).
Import boundaries and leakage prevention are two halves of the same contract:
this file governs *code dependencies and terminology*; `LEAK_PREVENTION.md`
governs *what data and features may flow into the model*.

---

## 1. Worktree map (read-only reference)

The following worktree layout is authoritative. Each worktree tracks a distinct
branch and owns a distinct slice of the codebase. There are **no submodules**,
**no vendoring**, and **no copying code verbatim** between worktrees or from
reference repositories.

```
~/projects/StageBridge-ccrt (ccrt-full integration branch);
~/projects/StageBridge-ccrt-core (ccrt-core); ~/projects/StageBridge-ccrt-synthetic;
~/projects/StageBridge-ccrt-panin (ccrt-panin-adapter);
~/projects/StageBridge-ccrt-luad (ccrt-luad-adapter);
~/projects/StageBridge-ccrt-deconv (ccrt-deconv-benchmark).
No submodules. No vendoring. No copying code verbatim.
```

A per-worktree "what you may touch" guide appears in [Section 7](#7-per-worktree-what-you-may-touch-guide).

---

## 2. Import-boundary rules

The CCRT source tree lives under `stagebridge/ccrt/` and is organized into 15
packages plus disease-specific adapter subpackages:

```
stagebridge/ccrt/
  contracts/ grammar/ io/ adapters/ representations/ sender_context/ operators/
  transport/ data/ synthetic/ deconvolution/ training/ evaluation/ plotting/ cli/
Plus adapters/panin/ and adapters/luad/.
```

The dependency direction is deliberately one-way: **biology-aware translation
layers depend on the system-agnostic core; the core never depends on them.**

### 2.1 Forbidden imports

Every rule below is a hard failure. The listed importer package (and anything
under it) must **never** import the listed target.

| # | Importer package                     | MUST NOT import        | Reason |
|---|--------------------------------------|------------------------|--------|
| 1 | `operators`                          | `adapters`             | The biological operator must stay system-agnostic; adapters know disease biology. |
| 2 | `sender_context`                     | `adapters`             | Local-influence attention must be disease-agnostic; distances/tokens are typed by grammar, not disease. |
| 3 | `transport`                          | `adapters`             | Semantic OT / matching runs in registered semantic space, independent of any disease adapter. |
| 4 | `adapters` (incl. `panin`, `luad`)   | `operators` / model    | Adapters only translate raw data into standardized tables + grammar IDs; they must never reach into model architecture. |
| 5 | `deconvolution`                      | `operators` / model    | Deconvolution backends are benchmarked as *sources of sender-context construction*; they must not touch the model. |
| 6 | `plotting`                           | `adapters`             | Figures are built from evaluation outputs only, never from raw disease adapters. |
| 7 | `plotting`                           | `training`             | Plotting must not depend on the training loop; it consumes exported evaluation outputs. |

Stated as the verbatim rule set that tests encode:

```
FAIL if: operators imports adapters; sender_context imports adapters; transport
imports adapters; adapters import operators/model; deconvolution imports
operators/model; plotting imports adapters; plotting imports training.
```

### 2.2 Allowed imports

The following dependency directions are explicitly permitted. Any dependency not
implied by these (and not forbidden above) should be treated as suspect and
reviewed before it is introduced.

```
ALLOWED: adapters -> grammar, contracts, io; operators -> grammar, contracts,
representations, sender_context; training -> operators, data, grammar.
```

In words:

- `adapters` may import `grammar`, `contracts`, and `io` — enough to emit
  standardized tables tagged with grammar IDs, and nothing more.
- `operators` may import `grammar`, `contracts`, `representations`, and
  `sender_context` — enough to build the context-residual drift/growth/regulatory
  operator over registered semantic state and typed sender context.
- `training` may import `operators`, `data`, and `grammar` — it wires the model to
  model-ready batches and losses, but does **not** define architecture.

### 2.3 Test enforcement (required)

`tests/ccrt/` **must** contain import-boundary tests that **FAIL on any
violation** of Section 2.1. These tests statically inspect the import graph of
`stagebridge/ccrt/` (for example, by parsing imports or importing each package in
isolation) and assert that no forbidden edge exists. A pull request that
introduces a forbidden import must not be able to pass CI. These tests are part
of the same suite described in `tests/ccrt/`: contract validation,
import-boundary, forbidden-term, split-validation, and leak-prevention tests.

---

## 3. Forbidden implementation terms

CCRT models local influence with **receiver-centered attention over typed
sender-context tokens and continuous distance modulation** — never with discrete
spatial bins or pre-averaged neighborhoods. To keep that mechanism honest, the
following identifiers are **forbidden in implementation code**:

```
world_token, ring_id, radial_bin, radius_bin, neighborhood_bin
```

The following **mechanisms** are equally forbidden, however they are named:

```
rings, radial bins, pre-attention neighborhood averaging
```

Rules:

- These five terms (and the three mechanisms) may appear in documentation
  **only** as forbidden terms — i.e., when explicitly labeling them as
  prohibited, exactly as this section does. They must never appear as real field
  names, variable names, config keys, function names, or column names in
  implementation code.
- Continuous distance modulation and system/stage-aware distance penalties are
  the sanctioned alternative; discretizing distance into rings, radial bins, or
  neighborhood bins, or averaging neighbors before attention, is prohibited.
- `tests/ccrt/` **must** contain a **forbidden-term test** that scans
  implementation code under `stagebridge/ccrt/` (and any generated code) for
  these identifiers and mechanism names and **FAILS** if any is found in a
  non-comment, non-doc context. This test is the automated backstop for the rule
  above.

---

## 4. Which layer may know biology

There is exactly one biology-aware layer and one system-agnostic core, and the
boundary between them is load-bearing.

**Only `adapters/` (including `adapters/panin/` and `adapters/luad/`) and
`grammar/` `BiologicalSystemSpec` instances may know disease biology.** Adapters
translate raw external data into standardized CCRT tables tagged with grammar IDs;
they may reference receiver states, sender-context ontologies, signal programs,
transition edges, and hypotheses that are specific to a system (LUAD, PanIN, a
future viral system). System-specific vocabularies live in `BiologicalSystemSpec`
under `grammar/`, **not** in the core.

**Everything else is system-agnostic core** and must not hard-code disease
concepts. In particular, the model and its supporting packages —
`sender_context`, `operators`, `transport`, `representations`, `data`,
`contracts`, `io` — MUST NOT hard-code:

```
LUAD, PanIN, macrophage, CAF, cancer, virus, malignancy axis
```

The core operates purely on grammar-typed inputs. The shared grammar sentence it
implements is:

> "In biological system S, sender-context type C carrying signal/program P at
> continuous distance d from receiver R modifies receiver behavior B along
> transition edge E through regulatory mediator M, with a context-residual
> effect."

with the ten shared grammar slots:

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

If a package outside `adapters/` and `grammar/`'s `BiologicalSystemSpec` needs to
"know" that a token is an IL1B macrophage or a myCAF, that is a boundary
violation — the information must arrive as a grammar-typed ID, not a hard-coded
disease term. `tests/ccrt/` includes forbidden-term / disease-term tests that
enforce this alongside the mechanism-term test in Section 3.

---

## 5. Rules per downstream layer

The downstream layers (`training`, `evaluation`, `plotting`, `cli`) each have a
narrow, non-overlapping responsibility. Keeping them thin prevents architecture
logic from leaking sideways and keeps figures reproducible from stable artifacts.

- **`training/`** — training loops, losses, and checkpointing. It imports the
  **model (`operators`) and `data`** (and `grammar`), wiring model-ready
  `CCRTBatch` inputs to losses. It **does NOT define architecture**; the operator
  and attention definitions live in `operators/` and `sender_context/`.
- **`evaluation/`** — standardized metrics computed from **model outputs and
  exported predictions**. It **does NOT read raw disease data directly** and does
  not reach into adapters; it consumes the model's decomposition outputs
  (self-intrinsic vs. context-conditioned transition, context-residual drift and
  growth, regulatory mediator, sender-context attribution, counterfactual
  perturbations, stability diagnostics).
- **`plotting/`** — figures built from **evaluation outputs ONLY**. Per
  Section 2.1 it must **not import `adapters`** and must **not import
  `training`**. Plots are derived from exported evaluation artifacts so figures
  are reproducible without re-running training or touching raw data.
- **`cli/`** — **thin** command-line entry points. Each CLI file calls **one**
  workflow and contains minimal logic; business logic lives in the packages the
  CLI dispatches to, not in the CLI itself. The runnable scripts in
  `scripts/ccrt/` are likewise thin entrypoints.

This mirrors the locked CCRT pipeline (no hidden side channels):

```
external data -> adapter inventory -> BiologicalSystemSpec + standardized CCRT tables
-> contract validation -> CCRTBatch -> semantic representation registry -> typed
sender-context attention -> context-residual drift/growth/regulatory operator ->
semantic transport matching -> counterfactual context perturbation ->
evaluation / stability / plots. No hidden side channels.
```

---

## 6. Reference-repo policy

Reference repositories are **conceptual references only**. They are read-only and
live at:

```
~/repos/amici
~/repos/osdr
~/repos/semantic-transport-generation
~/repos/PanIN_carcinogeneisis_spatial_analysis
~/repos/scPrisma
```

Policy:

- **Read-only.** Consult them for ideas and mechanisms; never modify them from
  within CCRT work.
- **No submodules.** CCRT must not add any of these as a git submodule.
- **No vendoring.** Do not copy their source trees (whole or partial) into
  `stagebridge/ccrt/`.
- **No verbatim copying.** Do not paste their code into CCRT. Reimplement concepts
  in CCRT's own idiom against CCRT's own contracts.

What each reference contributes, conceptually:

- **AMICI** — receiver-centered attention over nearby sender context
  (receiver-as-query, senders/neighbors as keys/values), continuous distance
  modulation, empty sender token, sparsity, and ablation-based interpretation.
- **OSDR** — spatial snapshot to tissue-rate / growth-death dynamics and
  phase-portrait thinking.
- **Semantic transport generation** — separating reconstruction space (`z_rec`)
  from semantic matching space (`z_sem`), Sinkhorn matching in semantic space,
  effective rank, and matching stability.
- **Cell behavior grammar paper** — biology becomes computable via a grammar of
  cell types, signals, behaviors, and rules; conceptual precedent only. CCRT is
  the learned/neural version of that idea: a typed sender-context signal modifies
  receiver transition behavior.

---

## 7. Per-worktree "what you may touch" guide

Each worktree owns a slice of the tree. Stay inside your slice; integrate through
`ccrt-full`. Never edit legacy StageBridge code from a CCRT worktree, and never
create `__init__.py` files beyond the single existing package marker at
`stagebridge/ccrt/__init__.py`.

| Worktree | Branch | You may touch | You must NOT touch |
|----------|--------|---------------|--------------------|
| `~/projects/StageBridge-ccrt` | `ccrt-full` (integration) | Integration across packages; `docs/ccrt/`; cross-package wiring during merges. | Do not use it as a dumping ground for work that belongs in a feature worktree. |
| `~/projects/StageBridge-ccrt-core` | `ccrt-core` | System-agnostic core: `contracts/`, `grammar/`, `io/`, `representations/`, `sender_context/`, `operators/`, `transport/`, `data/`, and their tests. | `adapters/` disease code; `deconvolution/` backends. Must not hard-code disease terms (Section 4) or forbidden terms (Section 3). |
| `~/projects/StageBridge-ccrt-synthetic` | (synthetic) | `synthetic/` known-truth mechanism-recovery benchmarks and their tests. | Disease adapters; model architecture definitions. |
| `~/projects/StageBridge-ccrt-panin` | `ccrt-panin-adapter` | `adapters/panin/` — PanIN raw data to `BiologicalSystemSpec` + the 8 standardized CCRT tables. | `operators`/model, `sender_context`, `transport` (forbidden imports #4). |
| `~/projects/StageBridge-ccrt-luad` | `ccrt-luad-adapter` | `adapters/luad/` — LUAD raw data to `BiologicalSystemSpec` + the 8 standardized CCRT tables. | `operators`/model, `sender_context`, `transport` (forbidden imports #4). |
| `~/projects/StageBridge-ccrt-deconv` | `ccrt-deconv-benchmark` | `deconvolution/` — benchmarking DestVI, cell2location, RCTD, CARD, Tangram, TACCO, SPOTlight, marker/program scoring as sources of sender-context construction. | `operators`/model (forbidden import #5). Must not change the CCRT table/tensor/grammar contract. |

Every adapter worktree emits the same eight standardized artifacts, which
downstream contract validation checks:

```
receivers.parquet, sender_context.parquet, semantic_features.parquet,
regulatory_features.parquet, stage_edges.parquet, samples.parquet,
split_manifest.json, system_spec.yaml
```

Cross-cutting reminders for **all** worktrees:

- Generated outputs go only under `$STAGEBRIDGE_DATA/{raw,interim,processed}/`
  and `$STAGEBRIDGE_RESULTS/`. Never write generated data into `stagebridge/`,
  `tests/`, or `~/projects/StageBridge-ccrt/data/` — see `LEAK_PREVENTION.md`.
- Splits must be patient-, donor-, or sample-aware, never spot-level or
  receiver-level random — see `LEAK_PREVENTION.md`.
- Forbidden model-input fields (`target_stage_expression`, `future_expression`,
  `outcome_label`, `patient_response`, `test_split_label`) must never enter a
  `CCRTBatch` model input — see `LEAK_PREVENTION.md`.

---

## Cross-references

- `LEAK_PREVENTION.md` — data-output, split, feature-space, and model-input
  leakage rules that complement the import and terminology boundaries here.
- `ARCHITECTURE_LOCK.md` — the overall locked design and the 15-package layout.
