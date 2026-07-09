# `operators/` — The Biological CCRT Operator

## Purpose

`operators/` is the mathematical heart of CCRT: the biological operator that
decomposes receiver behavior into a self-intrinsic part and a context-residual
part. It realizes the core operator

```
b_i^(e,S) = b_self(z_i, e, S) + delta_b_ctx(z_i, C_i, e, S)
```

for the two thesis behaviors — semantic transition drift `v` and
growth/mass/tissue-rate `g` — with a regulatory bottleneck `r`:

```
v_i^(e,S) = v_self(z_i, e, S) + B_(e,S) r_i + r_theta(z_i, C_i, e, S)
g_i^(e,S) = g_self(z_i, e, S) + a_(e,S)^T r_i + rho_theta(z_i, C_i, e, S)
```

The key outputs are a *decomposition*, not merely a prediction: self-intrinsic
transition, full context-conditioned transition, context-residual drift and
growth, regulatory mediator, sender-context attribution, counterfactual context
perturbation, and semantic transport stability. See `../sender_context/PURPOSE.md`,
`../transport/PURPOSE.md`, and `docs/ccrt/DIRECTORY_PURPOSES.md`.

## Owns

- `v_self`, `v_full`, `delta_v_ctx` (semantic transition drift decomposition).
- `g_self`, `g_full`, `delta_g_ctx` (growth / mass / tissue-rate decomposition).
- The regulatory bottleneck `r` and the context-residual decomposition.
- Conditioning on grammar identifiers for biological system `S` and edge `e`.

## Does NOT

- MUST NOT import `adapters`, deconvolution backends, or `plotting`
  (allowed: `grammar`, `contracts`, `representations`, `sender_context`).
- MUST NOT hard-code LUAD, PanIN, macrophage, CAF, cancer, virus, or a malignancy
  axis — those live only in `BiologicalSystemSpec`.
- Does NOT define training loops, losses, or checkpointing (those live in
  `training/`).
