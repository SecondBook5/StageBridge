# CCRT Tensor Contract

> Status: **architecture lock**. This document defines the tensor-level law for
> CCRT. No implementation modules exist yet; this is a contract, not a
> description of implemented code. All concrete numeric dimensions are symbolic
> and marked **(to be specified during implementation)**.

CCRT (Context-Residual Transport) is a grammar-conditioned neural transport
framework for estimating how typed local sender-context signals modify
receiver-cell drift, growth, and regulatory state along biological transition
edges. This document specifies the **tensor container** — the `CCRTBatch` — that
carries model inputs from validated tables into the operator, sender-context,
and transport layers.

This contract is **system-agnostic**. It applies identically to LUAD, PanIN, and
future viral systems: those differences live in the grammar layer
(`BiologicalSystemSpec`) and in per-system vocabularies, never in the tensor
shapes or field names defined here.

- Tables feed tensors. The upstream table-level law lives in
  [`TABLE_CONTRACT.md`](TABLE_CONTRACT.md); every tensor field below is derived
  from a validated column in one of the standardized adapter parquet outputs.
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
D_rec   reconstruction-space dimension (z_rec)                                    (to be specified during implementation)
D_sem   semantic transport-space dimension (z_sem)                               (to be specified during implementation)
D_reg   regulatory feature-space dimension (r)                                    (to be specified during implementation)
D_ctx   typed sender-context feature dimension (per sender token)                 (to be specified during implementation)
```

Notes:

- `B` is the number of **receivers** in the batch. CCRT is receiver-centered:
  each row of the batch is a receiver cell, and its typed sender context is
  carried along the `K` axis as keys/values for receiver-as-query attention.
- `K` is a **padded** maximum. Real receivers have variable numbers of sender
  tokens; padding + masking (Section 7) reconcile this to a fixed `K`.
- `D_rec`, `D_sem`, and `D_reg` are **distinct registered spaces** and MUST NOT
  be conflated. Reconstruction (`z_rec`) and semantic matching (`z_sem`) are
  separated by design; regulatory features (`r`) form a third space. See
  [`TABLE_CONTRACT.md`](TABLE_CONTRACT.md) for the source feature registries and
  [`LEAK_PREVENTION.md`](LEAK_PREVENTION.md) for the registration requirement
  that forbids an arbitrary latent matrix from silently becoming transport
  geometry.

---

## 3. Canonical tensor field naming conventions

Field names are **canonical and closed**. The following conventions are
mandatory:

```
- snake_case, lowercase ASCII only.
- Receiver-state tensors are prefixed with the space they live in:  z_sem, z_rec, r.
- Typed sender-context tensors are prefixed  sender_* .
- ID / index tensors carry an explicit  *_id  suffix and are integer-typed.
- Mask tensors carry an explicit  *_mask  suffix and are boolean-typed.
- Continuous physical / statistical quantities carry no space prefix
  (e.g. sender_distance, sender_uncertainty) but keep their sender_* scope.
- No abbreviations outside this document. No disease terms in any field name
  (no  luad_*, panin_*, macrophage_*, caf_*  fields).
- No free-form or optional "extra" fields. Unknown fields fail validation.
```

The only identifiers permitted to vary by biological system are the **values**
carried in the `*_id` tensors (they index per-system grammar registries); the
**names and shapes** of the fields never vary.

---

## 4. Receiver tensors

Each receiver instance carries three registered state representations plus its
grammar conditioning ids (Section 6).

```
z_sem                 float   [B, D_sem]    receiver semantic transport state
z_rec                 float   [B, D_rec]    receiver reconstruction state
r                     float   [B, D_reg]    receiver regulatory features
```

Contract:

- `z_sem` is the receiver's coordinate in the **semantic transport space**. It
  is the space in which semantic OT / Sinkhorn matching and context-effect
  stability are estimated (see `transport/`). It is a *registered* space
  (`SemanticFeatureRegistry`), never a raw expression vector and never an
  arbitrary latent.
- `z_rec` is the receiver's coordinate in the **reconstruction space**. It is
  kept strictly separate from `z_sem`; matching is never performed in `z_rec`.
- `r` is the receiver's **regulatory feature** vector, feeding the regulatory
  bottleneck of the operator. Regulatory mediators are registered
  (`RegulatoryMediatorRegistry`); `r` is not a free latent.

These three tensors correspond, respectively, to the semantic, reconstruction,
and regulatory feature columns defined in
[`TABLE_CONTRACT.md`](TABLE_CONTRACT.md) (`semantic_features.parquet`,
reconstruction features, and `regulatory_features.parquet`).

The operator consumes these to produce the context-residual decomposition:

```
b_i^(e,S) = b_self(z_i, e, S) + delta_b_ctx(z_i, C_i, e, S)

v_i^(e,S) = v_self(z_i, e, S) + B_(e,S) r_i + r_theta(z_i, C_i, e, S)      [drift]
g_i^(e,S) = g_self(z_i, e, S) + a_(e,S)^T r_i + rho_theta(z_i, C_i, e, S)  [growth]
```

where `z_i` denotes the receiver semantic state, `C_i` its typed sender-context
set (Section 5), `e` its transition edge, and `S` its biological system.

---

## 5. Typed sender-context tensors

Sender context is **local, typed, and receiver-relative**. For each receiver
(query), its surrounding sender context is carried as up to `K` typed tokens
(keys/values). This is the AMICI-inspired local-influence contract:
receiver-as-query attention over typed sender-context keys/values with
continuous distance modulation and uncertainty downweighting.

```
sender_features        float   [B, K, D_ctx]   per-token typed sender-context features
sender_context_type    int     [B, K]          typed sender-context ontology id per token
sender_distance        float   [B, K]          continuous receiver->sender distance
sender_uncertainty     float   [B, K]          per-token uncertainty (for downweighting)
sender_mask            bool    [B, K]          True = real sender token, False = padding
```

Contract:

- `sender_context_type` indexes the per-system `SenderContextOntology` (a
  grammar registry). Its integer values are system-specific
  (e.g. an `il1b_high_macrophage` id in LUAD, a `caf`/`mycaf` id in PanIN); the
  field name and shape are not.
- `sender_distance` is a **continuous** scalar per token. Distance enters the
  model only through continuous distance kernels / stage/system-aware distance
  penalties applied inside attention. Distance is **never** discretized into
  rings, radial bins, radius bins, or neighborhood bins (Section 9).
- `sender_uncertainty` is a per-token quantity used for **uncertainty
  downweighting** of sender influence. It is a first-class input, not a derived
  afterthought.
- `sender_features` carries the typed key/value payload per sender token.
  Sender effects may be **signed**; the sign is a property learned over these
  features, not a table field.
- Padding along `K` is governed exclusively by `sender_mask` (Section 7).

These tensors are derived from `sender_context.parquet`; see
[`TABLE_CONTRACT.md`](TABLE_CONTRACT.md).

### 5.1 Empty sender token convention

Senders are attention **keys/values**; the receiver is the **query**. A receiver
may have little or no informative sender context. To represent this **without**
collapsing sender information before attention, CCRT reserves an **empty sender
token** (an escape token) as a valid key/value.

```
- Slot 0 of the K axis is reserved as the empty sender token for every receiver.
- The empty sender token is ALWAYS unmasked (sender_mask[:, 0] = True) so the
  receiver always has at least one attendable key/value.
- Its sender_context_type carries the reserved "empty" ontology id.
- Its sender_distance / sender_uncertainty follow the reserved-token convention
  (to be specified during implementation); they are never treated as a real
  neighbor measurement.
- The empty sender token is the model's escape hatch: a receiver with no
  meaningful context attends to it, yielding a near-zero context residual
  (delta_b_ctx -> 0) rather than a fabricated one.
```

**Pre-attention neighborhood averaging is forbidden.** Sender context is never
summarized, averaged, or pooled into the receiver before attention. Each sender
token remains an individual key/value; the only permitted aggregation is the
attention mechanism itself, gated by `sender_mask`, `sender_distance`, and
`sender_uncertainty`.

---

## 6. Grammar conditioning tensors

Every receiver is conditioned on its transition edge and biological system.
These are the tensor handles into the grammar layer (`BiologicalSystemSpec`,
`TransitionGraph`).

```
transition_edge_id     int   [B]     transition edge id (indexes TransitionGraph for system S)
system_id              int   [B]     biological system id (indexes BiologicalSystemSpec)
```

Contract:

- `transition_edge_id` selects the edge `e` in `b_self`/`delta_b_ctx`,
  `v_self`/`delta_v_ctx`, `g_self`/`delta_g_ctx`, and the edge/system-specific
  regulatory maps `B_(e,S)`, `a_(e,S)`. It corresponds to a validated row in
  `stage_edges.parquet` ([`TABLE_CONTRACT.md`](TABLE_CONTRACT.md)).
- `system_id` selects the biological system `S`. It exists so the shared
  operator can remain system-agnostic while still routing to the correct
  per-system parameters and vocabularies. The model itself MUST NOT hard-code
  any system name (no LUAD/PanIN/virus literals in tensor construction);
  system identity is data carried in `system_id`.
- Optionally, a receiver-state id may accompany these for conditioning; its
  values index the per-system `ReceiverStateOntology`. Any such id follows the
  same `*_id` integer convention and never encodes disease strings.

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
- Receiver tensors (z_sem, z_rec, r) and grammar ids are dense over B and are
  not padded; every batch row is a valid receiver.
```

Masks are the contract; there is no implicit "zero means missing" convention on
feature tensors.

---

## 8. Dtype conventions

```
- Feature / continuous tensors (z_sem, z_rec, r, sender_features,
  sender_distance, sender_uncertainty):  floating point (precision to be
  specified during implementation; a single batch-wide float dtype).
- Id / index tensors (*_id, sender_context_type):  integer.
- Mask tensors (*_mask):  boolean.
- No object / string dtypes are permitted in a CCRTBatch. All categorical
  meaning is carried as integer ids resolved through grammar registries.
- Distances are non-negative continuous floats; uncertainties are continuous
  floats interpreted per the downweighting contract (to be specified during
  implementation).
```

Any string-valued field (disease labels, cell-type names, program names) belongs
in the tables and grammar registries, **not** in the tensor container.

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
validation if any of these names appears in a `CCRTBatch` input. This is the
tensor-level enforcement of [`LEAK_PREVENTION.md`](LEAK_PREVENTION.md); split
labels in particular must never leak into inputs, and splits must be
patient-/donor-/sample-aware (never receiver-level random) per that document.

---

## 10. Forbidden tensor mechanisms — no rings, radial bins, or world token

Spatial context enters CCRT **only** as continuous per-token distance
(`sender_distance`) modulating receiver-as-query attention. The following field
names / mechanisms are forbidden anywhere in the tensor container and in any
tensor-construction code:

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
sole distance representation is the **continuous** `sender_distance` scalar.

---

## 11. CCRTBatch field summary

The complete, closed set of `CCRTBatch` model-input fields:

```
Receiver tensors
  z_sem                 float   [B, D_sem]
  z_rec                 float   [B, D_rec]
  r                     float   [B, D_reg]

Typed sender-context tensors
  sender_features       float   [B, K, D_ctx]
  sender_context_type   int     [B, K]
  sender_distance       float   [B, K]
  sender_uncertainty    float   [B, K]
  sender_mask           bool    [B, K]        (slot 0 reserved: empty sender token)

Grammar conditioning tensors
  transition_edge_id    int     [B]
  system_id             int     [B]
  (optional receiver_state_id  int  [B], indexing the per-system ReceiverStateOntology)

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
  `stage_edges.parquet`, `samples.parquet`, `split_manifest.json`,
  `system_spec.yaml`). Tables feed tensors.
- [`LEAK_PREVENTION.md`](LEAK_PREVENTION.md) — the leakage law; Sections 9 and 10
  are its tensor-level enforcement, including target-stage exclusion,
  split-label exclusion, and the requirement that semantic / signal / regulatory
  spaces be registered rather than assembled from an arbitrary latent matrix.
