# `contracts/` — CCRT Low-Level Law

This is the design-lock description of `stagebridge/ccrt/contracts/`. No
implementation code exists yet; this document states the contract this package
will satisfy, not modules that are present today.

`contracts/` is the low-level law of the framework. It defines the canonical
field names, the tensor contracts, the table contracts, schema validation, the
set of forbidden fields, and split validation. Every other package speaks
through the vocabulary that `contracts/` establishes, so that standardized CCRT
tables and the `CCRTBatch` are never ad-hoc dictionaries. Its role is
enforcement: it prevents random dicts from flowing between packages, prevents
arbitrary or disease-specific private table formats, and prevents leakage of
target-stage data into model inputs. It is strictly system-agnostic.

## Owns

- Canonical field-name definitions for standardized tables and tensors.
- Tensor contracts and table contracts (shape, dtype, required columns).
- Schema validation of the eight standardized adapter outputs.
- The forbidden-field set and forbidden model-input leakage checks.
- Split validation (patient/donor/sample-aware; never spot- or receiver-level random).

## Does NOT

- MUST NOT know biology: no `LUAD`, `PanIN`, `macrophage`, `CAF`, `cancer`,
  `virus`, or `malignancy axis` hard-coding.
- MUST NOT admit forbidden terms as fields or identifiers: `world_token`,
  `ring_id`, `radial_bin`, `radius_bin`, `neighborhood_bin`.
- MUST NOT allow `target_stage_expression`, `future_expression`,
  `outcome_label`, `patient_response`, or `test_split_label` as model inputs.

See `docs/ccrt/TABLE_CONTRACT.md`, `docs/ccrt/TENSOR_CONTRACT.md`, and
`docs/ccrt/LEAK_PREVENTION.md`.
