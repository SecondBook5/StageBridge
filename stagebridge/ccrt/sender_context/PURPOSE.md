# `sender_context/` — AMICI-inspired Local Influence

## Purpose

`sender_context/` is the local influence layer of CCRT. It estimates how nearby
typed sender context modifies a receiver cell as continuous, learned attention:
the receiver acts as the query, typed sender-context tokens act as keys and
values, and spatial proximity enters through continuous distance kernels rather
than any discretization of space. This is the AMICI-inspired core — receiver-
centered attention with continuous distance modulation, an empty sender token,
sparsity, and ablation-friendly, signed sender effects. It produces the typed,
distance-modulated, uncertainty-aware sender-context signal that `operators/`
consumes to build the context-residual drift and growth terms. This package is
system-agnostic: it operates on grammar identifiers and standardized tensors, not
on any specific disease vocabulary. See `../operators/PURPOSE.md`,
`../representations/PURPOSE.md`, and `docs/ccrt/DIRECTORY_PURPOSES.md`.

## Owns

- Receiver-as-query attention over typed sender-context keys/values.
- Continuous distance kernels and stage/system-aware distance penalties.
- Uncertainty downweighting of low-confidence sender context.
- The empty sender token (receivers with no local context).
- Sparsity losses and signed (increase/decrease) sender effects.

## Does NOT

- MUST NOT import `adapters` (allowed: `grammar`, `contracts`, `representations`).
- MUST NOT hard-code LUAD, PanIN, macrophage, CAF, cancer, virus, or a malignancy
  axis — those live only in `BiologicalSystemSpec`.
- MUST NEVER use rings, radial bins, or pre-attention neighborhood averaging, nor
  the forbidden terms `world_token`, `ring_id`, `radial_bin`, `radius_bin`,
  `neighborhood_bin`.
