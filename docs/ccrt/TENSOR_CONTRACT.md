# CCRT Tensor Contract

> Status: **architecture lock**. This document defines the tensor-level law for
> CCRT. Concrete numeric dimensions are symbolic and marked **(to be specified
> during implementation)**.
>
> **Milestone 1 reconciliation.** The canonical `CCRTBatch` field names below are
> the names implemented and enforced by `stagebridge/ccrt/contracts` and
> `stagebridge/ccrt/data` in Core Milestone 1. Where an earlier draft of this
> document used the conceptual model-layer names `system_id`, `sender_distance`,
> or a required `z_rec` batch tensor, those have been reconciled here to the
> canonical batch names (`biological_system_id`, `distance_to_receiver`, and a
> deferred reconstruction space). The z_sem / z_rec *separation philosophy* is
> unchanged; only the batch field names and the Milestone-1 scope are clarified.

CCRT (Context-Residual Transport) is a grammar-conditioned neural transport
framework for estimating how typed local sender-context signals modify
receiver-cell drift, growth, and regulatory state along biological transition
edges. This document specifies the **tensor container** — the `CCRTBatch` — that
carries model inputs from validated tables into the operator, sender-context,
and transport layers.

This contract is **system-agnostic**. It applies identically to every
biological system: those differences live in the grammar layer
(`BiologicalSystemSpec`) and in per-system vocabularies, never in the tensor
shapes or field names defined here.

- Tables feed tensors. The upstream table-level law lives in
  [`TABLE_CONTRACT.md`](TABLE_CONTRACT.md); every tensor field below is derived
  from a validated column in one of the standardized adapter outputs.
- The leakage prohibitions below are the tensor-level projection of the rules in
  [`LEAK_PREVENTION.md`](LEAK_PREVENTION.md). The `contracts/` package enforces
  both.

---

## 1. Scope and authority

The `contracts/` package is the **low-level law**. It owns:

- canonical tensor field names (this document),
- tensor shape and dtype contracts,
- the mapping from validated tables to tensors,
- the forbidden-field list (leakage prevention),
- forbidden tensor mechanisms (rings / radial bins / world token).

Any tensor that enters the model must pass `contracts/` validation. Random
dicts, arbitrary field names, ad-hoc latent matrices, and disease-specific
private tensor layouts are prohibited. If a field is not named here, it is not a
valid `CCRTBatch` model-input field.

The `CCRTBatch` is assembled in the `data/` package (validated CCRT tables ->
model-ready `CCRTBatch`: dataset loading, collate, padding, masking,
patient/donor-aware splits, batch validation). `data/` contains **no
disease-specific code**; it only realizes the contract defined here.

---

## 2. Symbolic dimensions

All dimensions are symbolic. No embedding widths, neighbor counts, layer counts,
or thresholds are fixed at lock time.

```
B       batch size (number of receiver cells / receiver instances in the batch)
K       maximum number of typed sender-context tokens per receiver (padded)      (to be specified during implementation)
D_R     receiver feature dimension (receiver_features)                           (to be specified during implementation)
D_S     sender feature dimension (sender_features, per token)                    (to be specified during implementation)
D_Z     semantic transport-space dimension (semantic_features / z_sem)           (to be specified during implementation)
D_REG   regulatory feature-space dimension (regulatory_features)                 (to be specified during implementation)
```

Notes:

- `B` is the number of **receivers** in the batch. CCRT is receiver-centered:
  each row of the batch is a receiver cell, and its typed sender context is
  carried along the `K` axis as keys/values for receiver-as-query attention.
- `K` is a **padded** maximum. Real receivers have variable numbers of sender
  tokens; padding + masking (Section 7) reconcile this to a fixed `K`.
- The semantic transport space (`semantic_features`, conceptually `z_sem`), the
  reconstruction space (`z_rec`, deferred — see Section 4), and the regulatory
  feature space (`regulatory_features`) are **distinct registered spaces** and
  MUST NOT be conflated. See [`LEAK_PREVENTION.md`](LEAK_PREVENTION.md) for the
  registration requirement that forbids an arbitrary latent matrix from silently
  becoming transport geometry.

---

## 3. Canonical tensor field naming conventions

Field names are **canonical and closed**. The following conventions are
mandatory:

```
- snake_case, lowercase ASCII only.
- Receiver-level feature tensors are named for their role:
    receiver_features, semantic_features, regulatory_features.
- Typed sender-context tensors are prefixed  sender_*  (sender_features, sender_mask),
    with the receiver-relative distance named  distance_to_receiver .
- ID / grammar-conditioning tensors carry an explicit  *_id  suffix
    (biological_system_id, transition_edge_id, receiver_state_id).
- Mask tensors carry an explicit  *_mask  suffix and are boolean-typed.
- Continuous physical / statistical quantities are explicit and unbinned
    (distance_to_receiver, uncertainty).
- No disease terms in any field name (no  luad_*, panin_*, macrophage_*, caf_*  fields).
- No free-form or optional "extra" fields. Unknown fields fail validation.
```

The only identifiers permitted to vary by biological system are the **values**
carried in the `*_id` fields (they index per-system grammar registries); the
**names and shapes** of the fields never vary.

---

## 4. Receiver tensors

Each receiver instance carries its receiver-level feature representation(s) plus
its grammar conditioning ids (Section 6).

```
receiver_features     float   [B, D_R]      receiver feature representation (Milestone-1 generic input)
semantic_features     float   [B, D_Z]      receiver semantic transport state (z_sem); optional in Milestone 1
regulatory_features   float   [B, D_REG]    receiver regulatory features (r); optional in Milestone 1
```

Contract:

- `receiver_features` is the generic receiver feature tensor consumed by the
  Milestone-1 batch. It is the concrete, required receiver input; richer
  registered spaces layer on top of it.
- `semantic_features` is the receiver's coordinate in the **semantic transport
  space** — the batch-level realization of the conceptual `z_sem`. It is the
  space in which semantic OT / Sinkhorn matching and context-effect stability
  are estimated (see `transport/`). It is a *registered* space
  (`SemanticFeatureRegistry`), never a raw expression vector and never an
  arbitrary latent.
- `regulatory_features` is the receiver's **regulatory feature** vector
  (conceptually `r`), feeding the regulatory bottleneck of the operator.
  Regulatory mediators are registered (`RegulatoryMediatorRegistry`); it is not
  a free latent.

**Reconstruction space (`z_rec`) is deferred.** CCRT keeps the reconstruction
space (`z_rec`) and the semantic transport space (`z_sem`) conceptually
separate: matching is never performed in `z_rec`. However, **`z_rec` is NOT a
required `CCRTBatch` field in Core Milestone 1** — no reconstruction training
exists yet. It belongs to the later representation / reconstruction layer and
will be added to the batch contract only when that layer is implemented. Its
absence here does not weaken the z_sem / z_rec separation principle; it simply
scopes the batch to what Milestone 1 needs.

These tensors correspond to the semantic and regulatory feature columns defined
in [`TABLE_CONTRACT.md`](TABLE_CONTRACT.md) (`semantic_features.parquet` and
`regulatory_features.parquet`).

The operator consumes these to produce the context-residual decomposition:

```
b_i^(e,S) = b_self(z_i, e, S) + delta_b_ctx(z_i, C_i, e, S)

v_i^(e,S) = v_self(z_i, e, S) + B_(e,S) r_i + r_theta(z_i, C_i, e, S)      [drift]
g_i^(e,S) = g_self(z_i, e, S) + a_(e,S)^T r_i + rho_theta(z_i, C_i, e, S)  [growth]
```

where `z_i` denotes the receiver semantic state (`semantic_features`), `C_i` its
typed sender-context set (Section 5), `e` its transition edge, and `S` its
biological system.

---

## 5. Typed sender-context tensors

Sender context is **local, typed, and receiver-relative**. For each receiver
(query), its surrounding sender context is carried as up to `K` typed tokens
(keys/values). This is the AMICI-inspired local-influence contract:
receiver-as-query attention over typed sender-context keys/values with
continuous distance modulation and uncertainty downweighting.

```
sender_features        float   [B, K, D_S]     per-token typed sender-context features
distance_to_receiver   float   [B, K]          continuous receiver<-sender distance
uncertainty            float   [B, K]          per-token uncertainty (for downweighting); optional in Milestone 1
sender_mask            bool    [B, K]          True = real sender token, False = padding
```

Contract:

- The **typed identity** of each sender token indexes the per-system
  `SenderContextOntology`; that grammar id is the canonical field
  `sender_context_type_id` at the table level
  ([`TABLE_CONTRACT.md`](TABLE_CONTRACT.md)). In Milestone 1 it is carried into
  the batch via `sender_features` / the table join; a dedicated integer typed-id
  tensor is a later refinement. Its values are system-specific; the field name
  and shape are not.
- `distance_to_receiver` is a **continuous** scalar per token. Distance enters
  the model only through continuous distance kernels / stage/system-aware
  distance penalties applied inside attention. Distance is **never** discretized
  into rings, radial bins, radius bins, or neighborhood bins (Section 10).
- `uncertainty` is a per-token quantity used for **uncertainty downweighting**
  of sender influence. It is a first-class input, not a derived afterthought.
- `sender_features` carries the typed key/value payload per sender token.
  Sender effects may be **signed**; the sign is a property learned over these
  features, not a table field.
- Padding along `K` is governed exclusively by `sender_mask` (Section 7).

### 5.1 Empty sender token convention

Senders are attention **keys/values**; the receiver is the **query**. A receiver
may have little or no informative sender context. To represent this **without**
collapsing sender information before attention, CCRT reserves an **empty sender
token** (an escape token) as a valid key/value.

```
- Slot 0 of the K axis is reserved as the empty sender token for every receiver.
- The empty sender token is ALWAYS unmasked (sender_mask[:, 0] = True) so the
  receiver always has at least one attendable key/value.
- Its distance_to_receiver / uncertainty follow the reserved-token convention
  (to be specified during implementation); they are never treated as a real
  neighbor measurement.
- The empty sender token is the model's escape hatch: a receiver with no
  meaningful context attends to it, yielding a near-zero context residual
  (delta_b_ctx -> 0) rather than a fabricated one.
```

**Pre-attention neighborhood averaging is forbidden.** Sender context is never
summarized, averaged, or pooled into the receiver before attention. Each sender
token remains an individual key/value; the only permitted aggregation is the
attention mechanism itself, gated by `sender_mask`, `distance_to_receiver`, and
`uncertainty`.

---

## 6. Grammar conditioning tensors

Every receiver is conditioned on its transition edge and biological system.
These are the tensor handles into the grammar layer (`BiologicalSystemSpec`,
`TransitionGraph`).

```
biological_system_id   id      [B]     biological system id (indexes BiologicalSystemSpec)
transition_edge_id     id      [B]     transition edge id (indexes TransitionGraph for system S)
receiver_state_id      id      [B]     receiver state id (indexes ReceiverStateOntology); optional in Milestone 1
```

Contract:

- `biological_system_id` selects the biological system `S`. It exists so the
  shared operator can remain system-agnostic while still routing to the correct
  per-system parameters and vocabularies. The model itself MUST NOT hard-code
  any system name (no LUAD/PanIN/virus literals in tensor construction); system
  identity is data carried in `biological_system_id`. (This is the canonical
  batch field; it is **not** abbreviated to `system_id`.)
- `transition_edge_id` selects the edge `e` in `b_self`/`delta_b_ctx`,
  `v_self`/`delta_v_ctx`, `g_self`/`delta_g_ctx`, and the edge/system-specific
  regulatory maps `B_(e,S)`, `a_(e,S)`. It corresponds to a validated row in the
  `transition_edges` table ([`TABLE_CONTRACT.md`](TABLE_CONTRACT.md)). The core
  contract field is `transition_edge_id`, never `stage_edge_id`.
- `receiver_state_id` optionally accompanies these for conditioning; its values
  index the per-system `ReceiverStateOntology`. It follows the same `*_id`
  convention and never encodes disease strings.

In the Milestone-1 `CCRTBatch`, `biological_system_id` and `transition_edge_id`
are provided as a scalar string or a per-row sequence of grammar-id strings, and
`receiver_state_id` is optional. Integer-typed index tensors resolved through
the registries are a later refinement; the canonical field names are fixed now.

---

## 7. Masking and padding conventions

Variable-length sender context is reconciled to fixed `K` by padding, and all
padding is disclosed through explicit boolean masks.

```
- The K axis is padded to the batch/global maximum. Padding fills unused sender
  slots with contract-defined pad values (to be specified during implementation).
- sender_mask is the single source of truth for validity along K:
      True  -> real sender token (or the reserved empty sender token at slot 0)
      False -> padding; MUST be excluded from attention and from all losses.
- Attention MUST apply sender_mask so padded keys receive zero attention weight.
- Distance/uncertainty kernels MUST NOT read padded slots; padded values are
  never interpreted as measurements.
- The empty sender token (slot 0) is unmasked by construction and is distinct
  from padding: padding means "no token here"; the empty token means
  "explicitly no meaningful context."
- Receiver tensors (receiver_features, semantic_features, regulatory_features)
  and grammar ids are dense over B and are not padded; every batch row is a
  valid receiver.
```

Masks are the contract; there is no implicit "zero means missing" convention on
feature tensors.

---

## 8. Dtype conventions

```
- Feature / continuous tensors (receiver_features, semantic_features,
  regulatory_features, sender_features, distance_to_receiver, uncertainty):
  floating point (precision to be specified during implementation; a single
  batch-wide float dtype).
- Mask tensors (*_mask):  boolean.
- Grammar-conditioning ids (biological_system_id, transition_edge_id,
  receiver_state_id) are grammar-id values: string tokens that resolve through
  the grammar registries in Milestone 1, and integer index tensors once a later
  layer resolves them. Categorical meaning is always carried as a registered
  grammar id, never as an ad-hoc code.
- Distances are non-negative continuous floats; uncertainties are continuous
  floats interpreted per the downweighting contract (to be specified during
  implementation).
```

Free-form string labels (disease labels, cell-type names, program names) belong
in the tables and grammar registries, **not** in the tensor feature payloads.

---

## 9. Forbidden tensor fields — model-input leakage

The following five fields encode **target-stage / outcome information** and MUST
**NEVER** appear as `CCRTBatch` model inputs:

```
target_stage_expression
future_expression
outcome_label
patient_response
test_split_label
```

These may exist **only** as explicitly separated training targets (owned by
`training/`), never as model inputs. The `contracts/` package MUST fail
validation if any of these names appears in a `CCRTBatch` input (and, in
Milestone 1, even under `model_inputs`, `metadata`, or `targets`). This is the
tensor-level enforcement of [`LEAK_PREVENTION.md`](LEAK_PREVENTION.md); split
labels in particular must never leak into inputs, and splits must be
patient-/donor-/sample-aware (never receiver-level random) per that document.

---

## 10. Forbidden tensor mechanisms — no rings, radial bins, or world token

Spatial context enters CCRT **only** as continuous per-token distance
(`distance_to_receiver`) modulating receiver-as-query attention. The following
field names / mechanisms are forbidden anywhere in the tensor container and in
any tensor-construction code:

```
world_token
ring_id
radial_bin
radius_bin
neighborhood_bin
```

Also forbidden as mechanisms, regardless of field name:

```
- rings
- radial bins
- pre-attention neighborhood averaging
```

There is no global "world" summary token, no discretized distance ring/bin, and
no pre-attention pooling of neighbors. The escape hatch for context-poor
receivers is the **empty sender token** (Section 5.1), not a world token; and the
sole distance representation is the **continuous** `distance_to_receiver` scalar.

---

## 11. CCRTBatch field summary

The canonical `CCRTBatch` model-input fields for Core Milestone 1 (as
implemented in `stagebridge/ccrt/data/batch.py` and validated by
`stagebridge/ccrt/contracts`):

```
Receiver tensors
  receiver_features     float   [B, D_R]                 (required)
  semantic_features     float   [B, D_Z]                 (optional; z_sem)
  regulatory_features   float   [B, D_REG]               (optional; r)

Typed sender-context tensors
  sender_features       float      [B, K, D_S]           (required)
  sender_mask           bool       [B, K]                (required; slot 0 reserved: empty sender token)
  distance_to_receiver  float      [B, K]                (required)
  uncertainty           float      [B, K]                (optional)
  sender_context_type_ids  grammar-id/None  [B, K]       (optional; real senders carry a
                                                          grammar-id string, masked/padded
                                                          positions carry None. String ids,
                                                          NOT a numeric tensor: the training
                                                          layer maps them to integer indices
                                                          via the CCRT index registry. The
                                                          reserved empty-sender index is NEVER
                                                          used for padding.)

Grammar conditioning
  biological_system_id  grammar-id  [B] or scalar        (required)
  transition_edge_id    grammar-id  [B] or scalar        (required)
  receiver_state_id     grammar-id  [B] or scalar        (optional)

Separated (non-input) containers, validated for forbidden fields
  model_inputs   mapping     (auxiliary named model inputs)
  targets        mapping     (training targets; forbidden leakage names rejected)
  metadata       mapping     (bookkeeping; forbidden names rejected)

Deferred (NOT a Milestone-1 batch field)
  z_rec (reconstruction space) — added when the representation/reconstruction
  layer exists; kept conceptually separate from semantic_features / z_sem.

Forbidden as inputs (targets only, owned by training/)
  target_stage_expression, future_expression, outcome_label,
  patient_response, test_split_label

Forbidden field names / mechanisms (never present)
  world_token, ring_id, radial_bin, radius_bin, neighborhood_bin;
  rings, radial bins, pre-attention neighborhood averaging
```

Any batch that adds a field not listed here, renames a field, changes a dtype
class, omits a required mask, discretizes distance, introduces a world token, or
carries a forbidden leakage field is **invalid** and MUST be rejected by
`contracts/`.

---

## 12. Cross-references

- [`TABLE_CONTRACT.md`](TABLE_CONTRACT.md) — the table-level law; every tensor
  field here is derived from a validated column in the standardized adapter
  outputs (`receivers.parquet`, `sender_context.parquet`,
  `semantic_features.parquet`, `regulatory_features.parquet`,
  `transition_edges` / `stage_edges.parquet`, `samples.parquet`,
  `split_manifest.json`, `system_spec.yaml`). Tables feed tensors.
- [`LEAK_PREVENTION.md`](LEAK_PREVENTION.md) — the leakage law; Sections 9 and 10
  are its tensor-level enforcement, including target-stage exclusion,
  split-label exclusion, and the requirement that semantic / signal / regulatory
  spaces be registered rather than assembled from an arbitrary latent matrix.
```
