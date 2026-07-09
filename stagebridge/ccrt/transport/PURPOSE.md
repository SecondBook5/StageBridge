# `transport/` — Semantic Optimal Transport & Matching

## Purpose

`transport/` provides the semantic optimal-transport and matching machinery of
CCRT. Following the semantic-transport-generation precedent, matching is estimated
in **semantic biological space** (`z_sem`) — never in raw expression and never in
an arbitrary latent. Because `representations/` keeps the reconstruction space
(`z_rec`) separate from the semantic transport space (`z_sem`), transport here
operates only on the registered semantic geometry, so no arbitrary latent matrix
can silently become the transport metric. This package supplies the Sinkhorn
matching and stability diagnostics that let CCRT align receiver states across a
transition edge and quantify how stable the estimated context effects are. It is
fully system-agnostic, operating on registered representations and grammar
identifiers rather than any disease vocabulary. See `../representations/PURPOSE.md`,
`../operators/PURPOSE.md`, and `docs/ccrt/DIRECTORY_PURPOSES.md`.

## Owns

- Sinkhorn loss and barycentric maps.
- The semantic transport loss and geometric alignment in `z_sem`.
- Effective rank, matching stability, and context-effect stability diagnostics.

## Does NOT

- MUST NOT import `adapters` (allowed: `grammar`, `contracts`, `representations`).
- MUST NOT estimate transport in raw expression or an arbitrary/unregistered
  latent — only in the registered semantic space `z_sem`.
- MUST NOT hard-code LUAD, PanIN, macrophage, CAF, cancer, virus, or a malignancy
  axis — those live only in `BiologicalSystemSpec`.
