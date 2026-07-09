# `training/` — Training Loops, Losses, and Checkpointing

## Purpose

`training/` orchestrates optimization of the CCRT model: it owns the training
loops, the composition and weighting of losses, and checkpointing. It is the layer
that drives learning, but it is emphatically *not* where the model is defined —
the architecture lives in `operators/`, `sender_context/`, `representations/`, and
`transport/`. `training/` consumes model-ready `CCRTBatch` objects from `data/`,
invokes the operator and transport components to produce the context-residual
decomposition and matching losses, and manages the optimization state that turns
those signals into learned parameters. Because it does not embed architectural
decisions, the same training machinery serves every biological system without
knowing which one it is running on.

## Owns

- Training and validation loops.
- Loss composition, weighting, and scheduling (e.g. transport, sparsity, reconstruction terms defined by their owning packages).
- Optimizer and learning-rate management (concrete values to be specified during implementation).
- Checkpoint saving/loading and resume state.

## Does NOT

- Does NOT define model architecture — that belongs to `operators/`, `sender_context/`, `representations/`, and `transport/`.
- Does NOT hard-code `LUAD`, `PanIN`, `macrophage`, `CAF`, `cancer`, `virus`, or a malignancy axis.
- Allowed imports are `operators`, `data`, and `grammar` (see `../../../docs/ccrt/AGENT_BOUNDARIES.md`); it does not reach into `adapters/`.
- Does NOT write outputs into the source tree; checkpoints and logs land only in allowed output roots (`$STAGEBRIDGE_RESULTS/`, `$STAGEBRIDGE_DATA/`).
