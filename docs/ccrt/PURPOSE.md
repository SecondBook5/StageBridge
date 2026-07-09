# `docs/ccrt/` — Architecture-Lock Documents

This directory holds the StageBridge / CCRT architecture-lock documents themselves.
StageBridge / CCRT is a grammar-conditioned neural transport framework for
estimating how typed local sender-context signals modify receiver-cell drift,
growth, and regulatory state along biological transition edges. This is the
**design-lock** phase: no implementation code exists yet, and these documents
describe contracts and intended structure, not implemented modules. This PURPOSE.md
serves as an **index** to the lock documents below.

## Owns (the lock document set)

- `ARCHITECTURE_LOCK.md` — the top-level architecture lock: framing, core claim,
  operator equation, package layout, and pipeline flow.
- `DIRECTORY_PURPOSES.md` — authoritative one-liner role for every directory.
- `GRAMMAR_CONTRACT.md` — the shared grammar sentence and the ten grammar slots.
- `TENSOR_CONTRACT.md` — canonical tensor shapes and the `CCRTBatch` contract.
- `TABLE_CONTRACT.md` — standardized table schemas and canonical field names.
- `AGENT_BOUNDARIES.md` — enforced import-boundary rules and the tests that FAIL
  on violation.
- `LEAK_PREVENTION.md` — forbidden model-input leakage and feature-space
  registration rules.

## Does NOT

- Contain implementation code or any `.py` files.
- Describe implemented modules as existing; only `stagebridge/ccrt/__init__.py`
  exists today.
- Invent quantitative details (dims, neighbor counts, epsilon, layer counts);
  those are (to be specified during implementation).
- Duplicate biology into the core — biology lives in the authored
  `BiologicalSystemSpec` under `configs/ccrt/` (and in the generated
  `system_spec.yaml` emitted per run by `adapters/` under `$STAGEBRIDGE_DATA/`),
  never hard-coded into the system-agnostic packages.
