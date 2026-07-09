# representations/ — Semantic and Feature Representation Layer

`representations/` maintains the semantic and feature spaces CCRT operates in: the
reconstruction space (`z_rec`), the semantic transport space (`z_sem`), the
regulatory feature space, and the feature registries that govern them. Its
defining contract, inherited from the semantic-transport principle, is that
`z_rec` and `z_sem` are kept **separate** — the space used to reconstruct
expression is not the space in which semantic transport matching happens. Every
semantic, feature, and regulatory space must be registered here so that no
arbitrary latent matrix can silently become transport geometry. This is
architecture lock — no implementation exists yet; this file describes the
intended contract only.

## Owns

- The reconstruction space `z_rec`.
- The semantic transport space `z_sem`, kept distinct from `z_rec`.
- The regulatory feature space.
- The feature registries: `SemanticFeatureRegistry`, `SignalProgramRegistry`, and
  `RegulatoryMediatorRegistry` (registration enforced, not ad hoc).

## Does NOT

- MUST NOT collapse `z_rec` and `z_sem` into a single shared space.
- MUST NOT allow an unregistered latent matrix to become transport geometry.
- MUST NOT know biology or hard-code `LUAD`, `PanIN`, cell types, or a
  malignancy axis; it is strictly system-agnostic.
- MUST NOT use the forbidden terms `world_token`, `ring_id`, `radial_bin`,
  `radius_bin`, `neighborhood_bin`.

See `../../../docs/ccrt/LEAK_PREVENTION.md` and `../../../docs/ccrt/TENSOR_CONTRACT.md`.
