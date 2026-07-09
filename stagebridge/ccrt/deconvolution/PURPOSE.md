# `deconvolution/` — Deconvolution Backends as Sender-Context Sources

## Purpose

`deconvolution/` benchmarks deconvolution backends as alternative *sources* of
sender-context construction, measuring how the choice of backend changes the
inferred sender-context effects that CCRT reports. Spatial and mixed measurements
must be resolved into typed sender-context signals before they enter the operator;
this package treats that resolution step as a swappable input rather than as part
of the framework. It evaluates DestVI, cell2location, RCTD, CARD, Tangram, TACCO,
SPOTlight, and marker/program scoring against one another, feeding each backend's
output through the same standardized tables and grammar identifiers so that the
downstream context-residual decomposition is comparable across backends. The
purpose is diagnostic: to quantify sensitivity of the sender-context attribution
to the deconvolution choice, not to privilege any single method.

## Owns

- Backend adapters that turn deconvolution/scoring output into standardized `sender_context.parquet`-shaped inputs.
- Comparison harnesses across DestVI, cell2location, RCTD, CARD, Tangram, TACCO, SPOTlight, and marker/program scoring.
- Metrics on how backend choice shifts inferred sender-context effects.

## Does NOT

- MUST NOT change the CCRT table/tensor/grammar contract — the same standardized inputs are produced regardless of backend.
- MUST NOT import `operators/` or the model (see `../../../docs/ccrt/AGENT_BOUNDARIES.md`); allowed imports are `grammar`, `contracts`, `io`.
- Does NOT use `world_token`, `ring_id`, `radial_bin`, `radius_bin`, or `neighborhood_bin`, and never rings, radial bins, or pre-attention neighborhood averaging.
