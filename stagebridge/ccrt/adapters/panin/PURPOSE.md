# adapters/panin/ — PanIN Adapter

`adapters/panin/` translates PanIN (pancreatic intraepithelial neoplasia) raw
data into a `BiologicalSystemSpec` plus the standardized CCRT tables. It is a
disease-specific adapter and may reference PanIN biology directly, encoding that
biology as grammar identifiers rather than as core model code. The working
hypothesis it serves is that CAF/ECM/stromal context alters ductal epithelial
PanIN transition drift, growth, and tissue architecture. All PanIN-specific
vocabulary lives here and in the `BiologicalSystemSpec` it produces, so the core
operators never see the word "PanIN". This is architecture lock — no
implementation exists yet; this file describes the intended contract only.

## Owns

- Ingestion of PanIN raw data and emission of the eight standardized artifacts.
- A PanIN `BiologicalSystemSpec` declaring the system's vocabulary:
  - receiver states: `normal_duct`, `low_grade_panin`, `high_grade_panin`.
  - transition edges: `normal_duct->low_grade`, `low_grade->high_grade`.
  - sender-context ontology: CAF (`icaf`, `mycaf`, `apcaf`), ECM-rich stroma,
    ductal/acinar-ADM context, CODA lesion-geometry context, myeloid, lymphoid,
    endothelial.
  - signal programs: `caf_inflammatory_program`, `ecm_remodeling`,
    `tgfb_stromal_program`, `ductal_stress`, `adm_panin_program`, `proliferation`.

## Does NOT

- MUST NOT import model architecture, `operators/`, or `training/`.
- ALLOWED to import only `grammar`, `contracts`, and `io`.
- MUST NOT change the CCRT table/tensor/grammar contract.

See `../PURPOSE.md` and `../../../../docs/ccrt/GRAMMAR_CONTRACT.md`.
