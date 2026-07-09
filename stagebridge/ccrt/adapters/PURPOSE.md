# adapters/ — Disease-Specific Translation

`adapters/` is the disease-specific translation layer of CCRT and one of only two
places in the pipeline permitted to know biology (the other being `grammar/`). An
adapter takes raw external data for one biological system and produces a
`BiologicalSystemSpec` plus the standardized CCRT tables and grammar identifiers
that every downstream package consumes. Adapters are what let the core stay
system-agnostic: all knowledge of which cell type is a sender context, which
marker set maps to which signal program, and which transition edges exist is
quarantined here and expressed as grammar IDs, never as core model code. This is
architecture lock — no implementation exists yet; this file describes the
intended contract only.

## Owns

- Inventory and ingestion of raw external data for a biological system.
- Emission of the exact eight standardized artifacts:
  ```
  receivers.parquet
  sender_context.parquet
  semantic_features.parquet
  regulatory_features.parquet
  stage_edges.parquet
  samples.parquet
  split_manifest.json
  system_spec.yaml
  ```
- Mapping raw biology onto the ten shared grammar slots via a `BiologicalSystemSpec`.
- The two disease subpackages: `adapters/panin/` and `adapters/luad/`.

## Does NOT

- MUST NOT import model architecture, `operators/`, or `training/`.
- ALLOWED to import only `grammar`, `contracts`, and `io`.
- MUST NOT change the CCRT table/tensor/grammar contract.
- MUST NOT write outputs into `stagebridge/`, `tests/`, or the repo `data/` dir.

See `../../../docs/ccrt/AGENT_BOUNDARIES.md` and `../../../docs/ccrt/TABLE_CONTRACT.md`.
