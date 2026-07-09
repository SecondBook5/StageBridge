# `grammar/` — CCRT Biological Meaning Layer

This is the design-lock description of `stagebridge/ccrt/grammar/`. No
implementation code exists yet; this document states the contract this package
will satisfy, not modules that are present today.

`grammar/` is the biological meaning layer, and the component that actually
makes CCRT *unified*. It houses the `BiologicalSystemSpec`, `TransitionGraph`,
`ReceiverStateOntology`, `SenderContextOntology`, `SignalProgramRegistry`,
`ReceiverBehaviorRegistry`, `RegulatoryMediatorRegistry`, and
`CounterfactualPerturbationRegistry`. This is where each biological system
declares its own vocabulary while conforming to the ten shared grammar slots.
CCRT is unified at the grammar level, not the cell-type level: same grammar,
different vocabulary. Because the core operators are conditioned on grammar
identifiers (`S`, `e`) rather than hard-coded cell types, the same architecture
serves LUAD, PanIN, and future viral systems without pretending their biology is
equivalent.

## Owns

- `BiologicalSystemSpec` and `TransitionGraph` per system.
- `ReceiverStateOntology`, `SenderContextOntology`, `SignalProgramRegistry`.
- `ReceiverBehaviorRegistry`, `RegulatoryMediatorRegistry`,
  `CounterfactualPerturbationRegistry`.
- The ten shared grammar slots that every system must populate.

## Does NOT

- MUST NOT define or import model architecture, operators, or training.
- MUST NOT read or ingest raw disease data (that is `adapters/`).
- MUST NOT collapse distinct systems into one biology; it declares distinct
  vocabularies against the shared slots.

See `docs/ccrt/GRAMMAR_CONTRACT.md` for the shared grammar sentence and the ten
slots, and `docs/ccrt/DIRECTORY_PURPOSES.md`.
