# adapters/luad/ — LUAD Adapter

`adapters/luad/` translates LUAD (lung adenocarcinoma) raw data into a
`BiologicalSystemSpec` plus the standardized CCRT tables. It is a disease-specific
adapter and may reference LUAD biology directly, encoding that biology as grammar
identifiers rather than as core model code. The working hypothesis it serves is
that IL1B-high macrophage and inflammatory myeloid niches alter epithelial
transition drift and growth. All LUAD-specific vocabulary lives here and in the
`BiologicalSystemSpec` it produces, so the core operators never see the word
"LUAD" or "macrophage". This is architecture lock — no implementation exists yet;
this file describes the intended contract only.

## Owns

- Ingestion of LUAD raw data and emission of the eight standardized artifacts.
- A LUAD `BiologicalSystemSpec` declaring the system's vocabulary:
  - receiver states: `normal_alveolar`, `reactive_epithelial`, `aah`, `ais`,
    `invasive_luad`.
  - transition edges: `normal_lung->aah`, `aah->ais`, `ais->invasive`.
  - sender-context ontology: `il1b_high_macrophage`, `inflammatory_myeloid`,
    `macrophage_general`, fibroblast, endothelial, T cell, B cell,
    alveolar context.
  - signal programs: `il1b_il1r1`, `nfkb`, `interferon`,
    `inflammatory_epithelial_stress`, `immune_suppression`, `angiogenesis`,
    `ecm_remodeling`.

## Does NOT

- MUST NOT import model architecture, `operators/`, or `training/`.
- ALLOWED to import only `grammar`, `contracts`, and `io`.
- MUST NOT change the CCRT table/tensor/grammar contract.

See `../PURPOSE.md` and `../../../../docs/ccrt/GRAMMAR_CONTRACT.md`.
