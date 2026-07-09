# CCRT Grammar Contract

> Architecture-lock document. This describes intended contracts and structure only.
> No implementation modules exist yet; nothing here should be read as describing
> implemented code. See `ARCHITECTURE_LOCK.md` for the top-level design and the
> companion contract documents (`TABLE_CONTRACT.md`, `TENSOR_CONTRACT.md`,
> `AGENT_BOUNDARIES.md`) for the low-level field/tensor law and package boundaries.

StageBridge / CCRT is a grammar-conditioned neural transport framework for
estimating how typed local sender-context signals modify receiver-cell drift,
growth, and regulatory state along biological transition edges.

This document specifies the **biological meaning layer**: the shared grammar that
makes CCRT a single framework across biologically unrelated systems, and the
grammar objects that the `grammar/` package will own.

---

## 1. The core claim: unified at the grammar level, not the cell-type level

CCRT is unified at the **grammar** level, **not** the cell-type level. The claim
is **not** that LUAD, PanIN, and viral systems share the same biology — they do
not. IL1B-high macrophages in LUAD and CAF/ECM programs in PanIN are not
biologically equivalent, and CCRT does not pretend they are.

The claim **is** that different biological systems express their dynamics through
the **same transition grammar**: typed sender-context signals modify
receiver-cell behaviors along defined transition edges. This yields a unified
framework without collapsing distinct biology into a false equivalence.

The unification mechanism, stated compactly:

```
same grammar, different vocabulary
```

The grammar (the 10 slots, the sentence, the objects below) is shared and lives
in the model core. The vocabulary (which receiver states, which sender-context
types, which signal programs) is system-specific and lives in a
`BiologicalSystemSpec`, never in the core.

---

## 2. The 10 shared grammar slots

Every CCRT-modeled system is described through the same ten slots, in this order:

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

One-line definitions:

```
biological_system_id   Identifier of the biological system S being modeled.
receiver_state         The receiver cell's semantic state z_i within its system.
transition_edge        The directed edge e between receiver states along which behavior is estimated.
sender_context_type    The typed category C of a nearby sender-context signal source.
signal_program         The signal/program P a sender-context type carries.
distance               The continuous distance d from a sender-context source to the receiver.
uncertainty            Confidence in a sender-context assignment, used to downweight its influence.
receiver_behavior      The receiver behavior B being modified (drift, growth, regulatory state).
regulatory_mediator    The latent regulatory mediator M through which context acts on behavior.
context_residual_effect  The context-attributable residual modification of behavior beyond the self term.
```

---

## 3. The shared grammar sentence

Every system instantiates the same sentence, filling the slots with its own
vocabulary:

```
In biological system S, sender-context type C carrying signal/program P at
continuous distance d from receiver R modifies receiver behavior B along
transition edge E through regulatory mediator M, with a context-residual effect.
```

The neural realization of this sentence is the CCRT operator (see
`ARCHITECTURE_LOCK.md` for the operator's placement in the pipeline):

```
b_i^(e,S) = b_self(z_i, e, S) + delta_b_ctx(z_i, C_i, e, S)
```

where `S` is the biological system, `e` the transition edge, `z_i` the receiver
semantic state, `C_i` the typed sender-context set around receiver `i`, and `b`
the receiver behavior. `b_self` is the intrinsic ("empty context") term and
`delta_b_ctx` is the context-residual modification — the neural counterpart of
"sender-context type C ... modifies receiver behavior B ... with a
context-residual effect."

---

## 4. Grammar = expert rule; CCRT = learned rule

The cell-behavior-grammar paper established that biology becomes computable via a
**grammar of cell types, signals, behaviors, and rules**. That work is a
**conceptual precedent only** — never vendored, copied, or imported.

CCRT is the **learned / neural** version of that idea. The distinction is exact:

```
Grammar (expert-written rule):  "signal increases/decreases behavior"
CCRT   (learned rule):          "typed sender-context signal modifies receiver transition behavior"
```

An expert grammar hand-writes the sign and target of each rule. CCRT instead
learns, per system and per transition edge, how a typed sender-context signal at
a continuous distance modifies receiver drift, growth, and regulatory state —
and decomposes that modification into an interpretable context-residual effect.
The grammar *structure* (typed senders, receivers, behaviors, edges) is retained;
the *rules* are learned rather than declared.

---

## 5. `BiologicalSystemSpec` structure

A `BiologicalSystemSpec` is the single place a system declares its vocabulary and
hypothesis. It exists in **two forms** with distinct roles:

1. **Authored source spec** — a human-authored, version-controlled YAML checked in
   under `configs/ccrt/`. This is the canonical, editable declaration of a system's
   vocabulary (receiver states, edges, ontologies, programs, mediators,
   counterfactuals, hypothesis).
2. **Emitted `system_spec.yaml`** — the frozen, provenance-stamped copy an adapter
   writes per run as one of the eight standardized adapter outputs (see
   `TABLE_CONTRACT.md`), landing under `$STAGEBRIDGE_DATA/processed/<system>_ccrt/`
   alongside the parquet tables and `split_manifest.json`. It is a generated
   artifact and must never be checked in.

The authored source spec is the input; the emitted `system_spec.yaml` is a
generated snapshot that travels with the processed data for reproducibility. Both
conform to the same field structure below. Adapters author/consume the spec (see
`adapters/panin/` and `adapters/luad/`); the grammar layer consumes it.

Declared fields:

```
biological_system_id         Unique identifier for the biological system S.
receiver_states              The ordered set of receiver semantic states for S.
transition_edges             Directed edges between receiver states along which behavior is estimated.
sender_context_ontology      The typed sender-context categories present in S.
signal_programs              The signal/program vocabulary carried by sender-context types.
receiver_behaviors           The receiver behaviors modeled (e.g. transition drift, growth/tissue-rate).
regulatory_mediators         The regulatory mediator vocabulary for the bottleneck.
counterfactual_perturbations Named sender-context perturbations for counterfactual queries.
hypothesis                   A one-sentence biological hypothesis stated in grammar terms.
```

YAML skeleton (field names normative; values are system-specific and illustrative):

```yaml
biological_system_id: <string>

receiver_states:
  - <receiver_state_id>
  # ...

transition_edges:
  - <source_state> -> <target_state>
  # ...

sender_context_ontology:
  - <sender_context_type_id>
  # ...

signal_programs:
  - <signal_program_id>
  # ...

receiver_behaviors:
  - semantic_transition_drift   # v_i^(e,S)
  - growth                      # g_i^(e,S) (mass / tissue-rate)
  # ...

regulatory_mediators:
  - <regulatory_mediator_id>
  # ...

counterfactual_perturbations:
  - name: <perturbation_id>
    description: <what sender-context is added / removed / silenced>
  # ...

hypothesis: >
  <one sentence, phrased in grammar terms: which sender-context type carrying
   which program modifies which receiver behavior along which transition edge.>
```

---

## 6. Grammar objects owned by `grammar/`

The `grammar/` package owns the biological meaning layer. Each object has a single
responsibility:

```
BiologicalSystemSpec              Declares one system's full vocabulary + hypothesis (the spec above).
TransitionGraph                   Holds the receiver states and the directed transition edges among them.
ReceiverStateOntology             Canonical registry of receiver semantic states for a system.
SenderContextOntology             Canonical registry of typed sender-context categories for a system.
SignalProgramRegistry             Registers the signal/program vocabulary; prevents unregistered programs.
ReceiverBehaviorRegistry          Registers the modeled receiver behaviors (drift, growth, ...).
RegulatoryMediatorRegistry        Registers regulatory mediators; prevents unregistered mediators.
CounterfactualPerturbationRegistry  Registers named sender-context counterfactual perturbations.
```

These objects are what make CCRT unified: they encode the shared grammar as a
system-agnostic API, while their *contents* are filled per system from a
`BiologicalSystemSpec`. Registration through `SignalProgramRegistry`,
`ReceiverBehaviorRegistry`, and `RegulatoryMediatorRegistry` also enforces the
feature-space leakage prevention rule — no arbitrary vocabulary may silently
enter the model without being registered (see `LEAK_PREVENTION.md`).

---

## 7. System-specific example specs (illustrative)

The following specs are **illustrative** and live in `BiologicalSystemSpec`
instances, **not** in the core. They show the same grammar filled with three
different vocabularies.

### 7.1 LUAD

```yaml
biological_system_id: luad

receiver_states:
  - normal_alveolar
  - reactive_epithelial
  - aah
  - ais
  - invasive_luad

transition_edges:
  - normal_lung -> aah
  - aah -> ais
  - ais -> invasive

sender_context_ontology:
  - il1b_high_macrophage
  - inflammatory_myeloid
  - macrophage_general
  - fibroblast
  - endothelial
  - t_cell
  - b_cell
  - alveolar_context

signal_programs:
  - il1b_il1r1
  - nfkb
  - interferon
  - inflammatory_epithelial_stress
  - immune_suppression
  - angiogenesis
  - ecm_remodeling

hypothesis: >
  IL1B-high macrophage and inflammatory myeloid niches alter epithelial
  transition drift and growth.
```

### 7.2 PanIN

```yaml
biological_system_id: panin

receiver_states:
  - normal_duct
  - low_grade_panin
  - high_grade_panin

transition_edges:
  - normal_duct -> low_grade
  - low_grade -> high_grade

sender_context_ontology:
  - caf
  - apcaf
  - icaf
  - mycaf
  - ecm_rich_stroma
  - ductal_epithelial_context
  - acinar_adm_context
  - myeloid
  - lymphoid
  - endothelial
  - coda_lesion_geometry_context

signal_programs:
  - caf_inflammatory_program
  - ecm_remodeling
  - tgfb_stromal_program
  - ductal_stress
  - adm_panin_program
  - proliferation

hypothesis: >
  CAF/ECM/stromal context alters ductal epithelial PanIN transition drift,
  growth, and tissue architecture.
```

### 7.3 Future viral system

```yaml
biological_system_id: viral   # future system

receiver_states:
  - uninfected
  - early_infected
  - interferon_high
  - lytic_dead
  - persistent_latent
  - repaired_recovered

sender_context_ontology:
  - infected_neighboring_cells
  - interferon_high_epithelial
  - macrophages
  - t_cells
  - stromal_damage_context
  - viral_burden_context

signal_programs:
  - viral_rna_protein_burden
  - interferon
  - cytokine_signaling
  - immune_killing
  - tissue_damage
  - repair

hypothesis: >
  Infected and immune contexts redirect receiver cells among infection,
  antiviral, death, and repair states.
```

---

## 8. The model core must NOT hard-code system-specific vocabulary

The CCRT model core (`operators/`, `sender_context/`, `transport/`,
`representations/`, `data/`, `training/`) **must not hard-code** any
system-specific vocabulary. In particular, the following must never be baked into
the core:

```
LUAD, PanIN, macrophage, CAF, cancer, virus, malignancy axis
```

All such vocabulary lives exclusively in `BiologicalSystemSpec` instances and is
surfaced to the core only through the grammar objects of Section 6. Adapters may
know biology and may author specs; the core may only consume the grammar API. The
import-boundary and forbidden-term tests in `tests/ccrt/` will enforce this (see
`AGENT_BOUNDARIES.md` and `LEAK_PREVENTION.md`).

This separation is the whole point: **same grammar, different vocabulary**. The
grammar is what CCRT unifies; the vocabulary is what keeps distinct biology
distinct.
