# `tests/ccrt/` — CCRT Test Suite

This directory holds the CCRT test suite: contract-validation, import-boundary,
forbidden-term, split-validation, and leak-prevention tests. These tests are the
mechanism that keeps the architecture lock enforced over time — **they FAIL on
architecture violations**. They are the executable counterpart to the docs in
`docs/ccrt/`: whatever the lock documents assert as a rule, the suite must catch
when it is broken.

## Owns

- Contract-validation tests (canonical field names, tensor/table contracts).
- Import-boundary tests that FAIL when a package imports a forbidden sibling.
- Forbidden-term tests for the five banned terms and banned mechanisms.
- Split-validation tests (patient/donor-aware; never spot- or receiver-level random).
- Leak-prevention tests for forbidden model-input fields and feature-space registration.

## Import boundaries the suite must FAIL on

```
FAIL if: operators imports adapters
FAIL if: sender_context imports adapters
FAIL if: transport imports adapters
FAIL if: adapters import operators/model
FAIL if: deconvolution imports operators/model
FAIL if: plotting imports adapters
FAIL if: plotting imports training
```

## Forbidden terms the suite must reject

```
world_token
ring_id
radial_bin
radius_bin
neighborhood_bin
```

Also rejected as mechanisms: rings, radial bins, and pre-attention neighborhood
averaging.

## Does NOT

- Serve as an output root — no generated artifacts under `tests/`.
- Relax any rule in `docs/ccrt/AGENT_BOUNDARIES.md` or `LEAK_PREVENTION.md`; it
  enforces them.

See `../../docs/ccrt/AGENT_BOUNDARIES.md` and `../../docs/ccrt/LEAK_PREVENTION.md`.
