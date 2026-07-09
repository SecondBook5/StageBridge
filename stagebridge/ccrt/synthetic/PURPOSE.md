# `synthetic/` — Known-Truth Mechanism-Recovery Benchmarks

## Purpose

`synthetic/` generates controlled scenarios in which the ground-truth mechanism is
known by construction, so that CCRT can be proven to recover the mechanism it
claims to estimate *before* it is trusted on real biological data. Each scenario
plants a specific, labeled effect — or a deliberate absence of effect — and the
context-residual decomposition produced by `operators/` is then checked against
that known truth. Because these benchmarks exercise the same standardized tables,
`CCRTBatch`, and grammar identifiers as real runs, a passing synthetic result is
direct evidence that the drift, growth, regulatory, and attribution outputs mean
what they say. `synthetic/` is strictly system-agnostic: it never encodes LUAD,
PanIN, or viral biology.

## Owns

- Null-context benchmarks (no sender-context effect present).
- Drift-only, growth-only, and mixed drift/growth benchmarks.
- Regulatory-mediated benchmarks (effect routed through the regulatory bottleneck).
- Distance-specific sender-effect benchmarks (effect varies with continuous distance `d`).
- Wrong-context negative controls (effect attributed to the incorrect sender-context type must not be recovered).
- Ground-truth labels emitted through the standardized contract for scoring by `evaluation/`.

## Does NOT

- Does NOT hard-code `LUAD`, `PanIN`, `macrophage`, `CAF`, `cancer`, `virus`, or a malignancy axis (see `../../../docs/ccrt/DIRECTORY_PURPOSES.md`).
- Does NOT change the CCRT table/tensor/grammar contract; it emits standardized artifacts only.
- Does NOT use `world_token`, `ring_id`, `radial_bin`, `radius_bin`, or `neighborhood_bin`, and never rings, radial bins, or pre-attention neighborhood averaging.
- Does NOT read raw disease data.
