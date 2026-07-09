# `configs/ccrt/` — CCRT Configuration Files

This directory holds the declarative configuration for CCRT: the **authored**
`BiologicalSystemSpec` YAMLs, model configs, and training configs. It is
**configuration only, no code**. Configs are the parameterized inputs that drive
CCRT workflows — they let system specs, model shapes, and training runs be defined
and versioned without embedding values inside Python modules. Because CCRT is
unified at the grammar level, each biological system (LUAD, PanIN, future viral
systems) declares its own vocabulary through an authored `BiologicalSystemSpec`
here, conforming to the ten shared grammar slots while never touching the
system-agnostic core.

The authored spec checked in here is the canonical, editable **source**. It is
distinct from the generated `system_spec.yaml` snapshot an adapter emits per run
(one of the eight standardized adapter outputs, under
`$STAGEBRIDGE_DATA/processed/<system>_ccrt/`), which is a provenance-stamped
artifact and is never checked in. See `GRAMMAR_CONTRACT.md` §5 and
`TABLE_CONTRACT.md` for that distinction.

## Owns

- Authored `BiologicalSystemSpec` YAMLs (per-system receiver states, transition
  edges, sender-context ontology, signal programs, regulatory mediators,
  counterfactuals) — the editable source, not the generated `system_spec.yaml`.
- Model configuration files (operator, sender-context attention, transport,
  representation settings).
- Training configuration files (optimization, loss weighting, checkpointing).

## Does NOT

- Contain executable code, importable modules, or any `.py` files.
- Hard-code disease biology into the core; biology lives in system specs as grammar
  identifiers, produced by `adapters/`.
- Invent quantitative details at lock time; concrete numbers are
  (to be specified during implementation).
- Serve as an output root — generated artifacts go under `$STAGEBRIDGE_DATA/` or
  `$STAGEBRIDGE_RESULTS/`, never here.

See `../../docs/ccrt/ARCHITECTURE_LOCK.md` and `../../docs/ccrt/GRAMMAR_CONTRACT.md`.
