# CCRT Directory Purposes

This document is part of the StageBridge / CCRT **architecture-lock** documentation
set. It enumerates every directory in the CCRT source layout together with its
authoritative role, the intended import boundaries between packages, and the
end-to-end pipeline flow.

> **Framing.** StageBridge / CCRT is a grammar-conditioned neural transport
> framework for estimating how typed local sender-context signals modify
> receiver-cell drift, growth, and regulatory state along biological transition
> edges.

CCRT is unified at the **grammar** level, not the cell-type level. The claim is
*not* that LUAD, PanIN, and viral systems share the same biology — they do not.
The claim is that different biological systems express their dynamics through the
same transition grammar: typed sender-context signals modify receiver-cell
behaviors along defined transition edges. The directory layout below is the
structural expression of that principle: a system-agnostic core, with all
biological knowledge quarantined inside the adapters.

## Status: architecture lock, no implementation

This is the **design-lock** phase. No implementation code exists yet.

```
Only stagebridge/ccrt/__init__.py exists (a package marker).
NO other .py files exist yet.
Nothing below should be read as describing an implemented module —
these are contracts and intended structure.
```

The directory roles in this document describe *intended* responsibilities and the
*contracts* each package will satisfy, not code that is present today.

## Full directory tree

```
stagebridge/ccrt/
  contracts/          # low-level law: field names, tensor/table contracts, validation
  grammar/            # biological meaning layer: BiologicalSystemSpec and registries
  io/                 # safe paths, standardized table IO, provenance manifests
  adapters/           # disease-specific translation ONLY (may know biology)
    panin/            # PanIN raw data -> spec + standardized CCRT tables
    luad/             # LUAD raw data -> spec + standardized CCRT tables
  representations/    # z_rec, z_sem, regulatory feature space, feature registries
  sender_context/     # AMICI-inspired receiver-centered local influence attention
  operators/          # biological CCRT operator: drift/growth/regulatory decomposition
  transport/          # semantic OT & matching in semantic biological space
  data/               # validated tables -> model-ready CCRTBatch
  synthetic/          # known-truth mechanism-recovery benchmarks
  deconvolution/      # benchmarks deconvolution backends as sender-context sources
  training/           # training loops, losses, checkpointing (not architecture)
  evaluation/         # standardized metrics from model outputs / exported predictions
  plotting/           # figures from evaluation outputs ONLY
  cli/                # thin command-line entry points

# Repo-level directories (outside the stagebridge/ package)
docs/ccrt/            # the architecture-lock documents themselves
configs/ccrt/         # configuration files (system specs, model/training configs)
scripts/ccrt/         # thin runnable scripts / entrypoints for CCRT workflows
tests/ccrt/           # CCRT test suite (contracts, boundaries, leakage, splits)
```

## Directory roles

The 15 packages under `stagebridge/ccrt/`, the two disease-specific adapter
subpackages, and the four repo-level `ccrt` directories each have a single
authoritative role. They are described below.

### `stagebridge/ccrt/contracts/`

The low-level law of the framework. `contracts/` defines the canonical field
names, tensor contracts, table contracts, schema validation, the set of forbidden
fields, and split validation. Its job is to prevent random dicts from flowing
between packages, to prevent arbitrary or ad-hoc field names, to prevent
disease-specific private table formats from leaking into the shared pipeline, and
to prevent target-stage data from leaking into model inputs. Every other package
speaks through the vocabulary that `contracts/` establishes. See
`TABLE_CONTRACT.md` and `TENSOR_CONTRACT.md` for the enumerated fields and
`LEAK_PREVENTION.md` for the forbidden-input list.

### `stagebridge/ccrt/grammar/`

The biological meaning layer, and the component that actually makes CCRT
*unified*. `grammar/` houses the `BiologicalSystemSpec`, the `TransitionGraph`,
the `ReceiverStateOntology`, the `SenderContextOntology`, the
`SignalProgramRegistry`, the `ReceiverBehaviorRegistry`, the
`RegulatoryMediatorRegistry`, and the `CounterfactualPerturbationRegistry`. This
is where each biological system declares its own vocabulary while conforming to
the ten shared grammar slots. Because the core operators are conditioned on grammar
identifiers rather than hard-coded cell types, the same architecture serves LUAD,
PanIN, and future viral systems. See `GRAMMAR_CONTRACT.md` for the shared grammar
sentence and the ten slots.

### `stagebridge/ccrt/io/`

Safe input/output plumbing. `io/` provides safe path handling, standardized table
IO, and provenance manifests. Its purpose is to prevent generated data from
entering git, to prevent random pickle outputs, and to prevent silent artifact
drift. All reads and writes of standardized artifacts are routed through `io/` so
that outputs land only in allowed roots (see **Output roots** below) and every
artifact carries provenance.

### `stagebridge/ccrt/adapters/`

Disease-specific translation, and **the only place that may know biology inside
the pipeline**. An adapter takes raw external data and produces standardized CCRT
tables plus grammar identifiers. Adapters may encode biological knowledge (which
cell type is a sender context, which program a marker set corresponds to, which
transition edges exist). Adapters **must not** import model architecture,
operators, or training. This one-directional boundary is what lets the core stay
system-agnostic. See `AGENT_BOUNDARIES.md` for the enforced import rules.

#### `stagebridge/ccrt/adapters/panin/`

Translates PanIN raw data into a `BiologicalSystemSpec` plus the standardized CCRT
tables. It encodes PanIN-specific biology (e.g. ductal receiver states and
CAF/ECM/stromal sender contexts) as grammar identifiers, never as core model code.

#### `stagebridge/ccrt/adapters/luad/`

Translates LUAD raw data into a `BiologicalSystemSpec` plus the standardized CCRT
tables. It encodes LUAD-specific biology (e.g. alveolar/epithelial receiver states
and IL1B-high macrophage / inflammatory myeloid sender contexts) as grammar
identifiers, never as core model code.

### `stagebridge/ccrt/representations/`

The semantic and feature representation layer. `representations/` maintains the
reconstruction space (`z_rec`), the semantic transport space (`z_sem`), the
regulatory feature space, and the feature registries. Its defining contract is
that `z_rec` and `z_sem` are kept **separate**: the space used to reconstruct
expression is not the space in which semantic transport matching happens. Feature,
semantic, and regulatory spaces must be registered here so that no arbitrary latent
matrix silently becomes transport geometry.

### `stagebridge/ccrt/sender_context/`

AMICI-inspired local influence. `sender_context/` implements receiver-as-query
attention over typed sender-context keys/values, continuous distance kernels,
stage/system-aware distance penalties, uncertainty downweighting, an empty sender
token, sparsity losses, and signed sender effects. It computes how nearby typed
sender context modifies a receiver — as continuous, learned attention. It **never**
uses rings, radial bins, or pre-attention neighborhood averaging (see **Forbidden
implementation terms**).

### `stagebridge/ccrt/operators/`

The biological CCRT operator: the mathematical heart of the framework.
`operators/` implements `v_self`, `v_full`, `delta_v_ctx`, `g_self`, `g_full`,
`delta_g_ctx`, the regulatory bottleneck, and the context-residual decomposition.
It realizes the core operator

```
b_i^(e,S) = b_self(z_i, e, S) + delta_b_ctx(z_i, C_i, e, S)
```

for the two thesis behaviors (semantic transition drift `v` and growth/mass/tissue
-rate `g`), with a regulatory bottleneck `r`. The key outputs are a *decomposition*
(self-intrinsic transition, full context-conditioned transition, context-residual
drift and growth, regulatory mediator, sender-context attribution, counterfactual
context perturbation, and semantic transport stability), not merely a prediction.
`operators/` **must not** import adapters, deconvolution backends, or plotting.

### `stagebridge/ccrt/transport/`

Semantic optimal transport and matching. `transport/` provides the Sinkhorn loss,
barycentric maps, the semantic transport loss, effective rank, matching stability,
geometric alignment, and context-effect stability. Crucially, transport is
estimated in **semantic biological space** (`z_sem`), not in raw expression and not
in an arbitrary latent. It **must not** import adapters.

### `stagebridge/ccrt/data/`

The bridge from validated tables to model-ready tensors. `data/` performs dataset
loading, collation, padding, masking, patient/donor-aware splits, and batch
validation, producing the `CCRTBatch`. It contains **no disease-specific code**:
it operates purely on the standardized contract established by `contracts/`. Splits
must be patient- or donor-aware (see **Split validation**).

### `stagebridge/ccrt/synthetic/`

Known-truth mechanism-recovery benchmarks. `synthetic/` generates controlled
scenarios where the ground truth is known: null context, drift-only, growth-only,
mixed drift/growth, regulatory-mediated, distance-specific sender effects, and
wrong-context negative controls. These let us verify that the operator recovers the
mechanism it claims to estimate before it is trusted on real disease data.

### `stagebridge/ccrt/deconvolution/`

Benchmarks deconvolution backends as sources of sender-context construction.
`deconvolution/` evaluates DestVI, cell2location, RCTD, CARD, Tangram, TACCO,
SPOTlight, and marker/program scoring as alternative ways to build the typed
sender-context input. It **must not** change the CCRT table/tensor/grammar
contract — it feeds the same standardized inputs, so the choice of backend is a
swappable source rather than a change to the framework.

### `stagebridge/ccrt/training/`

Training loops, losses, and checkpointing. `training/` orchestrates optimization
but **does not define architecture** — the architecture lives in `operators/`,
`sender_context/`, `representations/`, and `transport/`. Training is allowed to
import `operators`, `data`, and `grammar`.

### `stagebridge/ccrt/evaluation/`

Standardized metrics. `evaluation/` computes metrics from model outputs and
exported predictions. It **does not read raw disease data directly**; it consumes
the model's outputs and the standardized artifacts, keeping evaluation reproducible
and decoupled from disease-specific ingestion.

### `stagebridge/ccrt/plotting/`

Figures. `plotting/` produces figures from **evaluation outputs only**. It **does
not import adapters** and **does not import training**. This keeps figure generation
a pure downstream consumer with no back-channel into biology-aware or
training-time code.

### `stagebridge/ccrt/cli/`

Thin command-line entry points. Each file in `cli/` calls exactly one workflow and
contains minimal logic — argument parsing and dispatch, not business logic. The
substance lives in the packages the CLI invokes.

### `docs/ccrt/`

Holds the architecture-lock documents themselves, including this file and its
siblings: `ARCHITECTURE_LOCK.md`, `GRAMMAR_CONTRACT.md`, `TENSOR_CONTRACT.md`,
`TABLE_CONTRACT.md`, `AGENT_BOUNDARIES.md`, and `LEAK_PREVENTION.md`.

### `configs/ccrt/`

Configuration files for CCRT: system specs, model configs, and training configs.
These are the declarative inputs that parameterize the workflows without embedding
values in code.

### `scripts/ccrt/`

Thin runnable scripts and entrypoints for CCRT workflows. Like `cli/`, these stay
thin — they wire together configured workflows rather than reimplementing logic.

### `tests/ccrt/`

The CCRT test suite. It contains contract-validation tests, import-boundary tests,
forbidden-term tests, split-validation tests, and leak-prevention tests. These
tests are the mechanism that keeps the architecture lock enforced over time; see
`AGENT_BOUNDARIES.md` for the specific rules the tests must enforce.

## Who may know biology

```
MAY know biology:
  adapters/         (and adapters/panin/, adapters/luad/)
  grammar/          (declares system-specific vocabularies via BiologicalSystemSpec)

STRICTLY system-agnostic (no LUAD / PanIN / virus / cell-type hard-coding):
  contracts/  io/  representations/  sender_context/  operators/
  transport/  data/  synthetic/  deconvolution/  training/
  evaluation/ plotting/ cli/
```

The core must **not** hard-code `LUAD`, `PanIN`, `macrophage`, `CAF`, `cancer`,
`virus`, or a `malignancy axis`. Those live only in the system-specific grammar
specs (`BiologicalSystemSpec`) produced by adapters. The core operator is
conditioned on the biological system `S` and transition edge `e` as grammar
identifiers, never on the biology itself.

## Allowed-vs-forbidden import summary

The following table summarizes the import boundaries. The authoritative,
enforced version — including the tests that must fail on violation — lives in
`AGENT_BOUNDARIES.md`.

| Package | ALLOWED to import | MUST NOT import |
| --- | --- | --- |
| `adapters/` (incl. `panin/`, `luad/`) | `grammar`, `contracts`, `io` | `operators`, model, `training` |
| `operators/` | `grammar`, `contracts`, `representations`, `sender_context` | `adapters`, deconvolution backends, `plotting` |
| `sender_context/` | `grammar`, `contracts`, `representations` | `adapters` |
| `transport/` | `grammar`, `contracts`, `representations` | `adapters` |
| `deconvolution/` | `grammar`, `contracts`, `io` | `operators`, model |
| `training/` | `operators`, `data`, `grammar` | — |
| `plotting/` | `evaluation` outputs | `adapters`, `training` |

Import-boundary rules restated (future tests must **FAIL** if violated):

```
FAIL if: operators imports adapters
FAIL if: sender_context imports adapters
FAIL if: transport imports adapters
FAIL if: adapters import operators/model
FAIL if: deconvolution imports operators/model
FAIL if: plotting imports adapters
FAIL if: plotting imports training

ALLOWED: adapters      -> grammar, contracts, io
ALLOWED: operators      -> grammar, contracts, representations, sender_context
ALLOWED: training       -> operators, data, grammar
```

## Core CCRT pipeline flow

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

There are **no hidden side channels**: every stage consumes the standardized
artifacts of the previous stage, and disease-specific knowledge enters only at the
adapter step.

### Standardized adapter outputs

Each adapter emits exactly these eight artifacts, which are the contract the rest
of the pipeline consumes:

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

## Output roots

Generated outputs must land only in the allowed roots, never inside the source
tree or git-tracked data directories.

```
ALLOWED primary generated outputs (outside git):
  $STAGEBRIDGE_DATA/raw/
  $STAGEBRIDGE_DATA/interim/
  $STAGEBRIDGE_DATA/processed/
    (e.g. $STAGEBRIDGE_DATA/processed/panin_ccrt/,
          $STAGEBRIDGE_DATA/processed/luad_ccrt/)
  $STAGEBRIDGE_RESULTS/

FORBIDDEN primary output roots:
  stagebridge/
  tests/
  ~/projects/StageBridge-ccrt/data/
```

## Split validation

Splits produced by `data/` and validated by `contracts/` must be:

```
patient-aware splits
donor-aware splits
sample-aware splits ONLY when patient/donor unavailable
NEVER spot-level or receiver-level random splits for biological claims
```

## Forbidden implementation terms

The following five terms may appear in documentation **only** as forbidden terms.
They must never be used as field names, identifiers, or mechanisms in the code:

```
world_token
ring_id
radial_bin
radius_bin
neighborhood_bin
```

Also forbidden as mechanisms: **rings**, **radial bins**, and **pre-attention
neighborhood averaging**. Local influence is expressed through `sender_context/`
attention with continuous distance modulation instead.

## Cross-references

- `AGENT_BOUNDARIES.md` — authoritative, enforced import-boundary rules and the
  tests that fail on violation.
- `GRAMMAR_CONTRACT.md` — the shared grammar sentence and the ten grammar slots.
- `TABLE_CONTRACT.md` — canonical field names and standardized table schemas.
- `TENSOR_CONTRACT.md` — the CCRTBatch tensor field names and forbidden fields.
- `LEAK_PREVENTION.md` — forbidden model-input leakage and feature-space
  registration rules.
