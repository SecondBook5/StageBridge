# configs/

Hydra configuration groups for StageBridge.

## Core Entrypoints

- `config.yaml`: master defaults for pipeline-related scripts
- `train.yaml`: training-focused defaults
- `eval.yaml`: evaluation-focused defaults

## Config Groups

- `data/`: data roots, file paths, and data options
- `model/`: model architecture variants
- `training/`: optimization/runtime defaults
- `experiment/`: smoke/full experiment presets
- `splits/`: donor split strategy
- `hlca/`: HLCA mapping options
- `tangram/`: Tangram mapping options

Use overrides from notebook or CLI, e.g.:
- `data=local`
- `experiment=smoke`
- `model=stagebridge`
- `training=default`
