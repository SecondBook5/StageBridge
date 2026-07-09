# data/ — Validated Tables to Model-Ready Batches

`data/` is the bridge from validated CCRT tables to the model-ready `CCRTBatch`.
It performs dataset loading, collation, padding, masking, patient/donor-aware
splitting, and batch validation, operating purely on the standardized contract
established by `contracts/`. It contains **no disease-specific code**: it never
references `LUAD`, `PanIN`, cell types, or a malignancy axis, so the same batching
logic serves every biological system. This is architecture lock — no
implementation exists yet; this file describes the intended contract only.

## Owns

- Dataset loading from the standardized adapter tables.
- Collation, padding, and masking into the `CCRTBatch`.
- Split construction and batch validation.
- Enforcement of split validity:
  ```
  patient-aware splits
  donor-aware splits
  sample-aware splits ONLY when patient/donor unavailable
  NEVER spot-level or receiver-level random splits for biological claims
  ```

## Does NOT

- MUST NOT contain disease-specific code or know biology.
- MUST NOT admit forbidden leakage fields into `CCRTBatch` model inputs:
  `target_stage_expression`, `future_expression`, `outcome_label`,
  `patient_response`, `test_split_label`.
- MUST NOT use random spot-level or receiver-level splits for biological claims.
- MUST NOT use the forbidden terms `world_token`, `ring_id`, `radial_bin`,
  `radius_bin`, `neighborhood_bin`.

See `../../../docs/ccrt/TABLE_CONTRACT.md` and `../../../docs/ccrt/LEAK_PREVENTION.md`.
