# Notebook Workflow API

`StageBridge.ipynb` is the top-level graded entrypoint.

To keep orchestration package-first, notebook control cells should call
`stagebridge.notebook_api` functions instead of embedding shell logic.

## Recommended pattern

```python
from stagebridge.notebook_api import compose_config, run_step, run_pipeline

cfg = compose_config("config", overrides=["data=local", "experiment=smoke"])
run_step("build_snrna", cfg)
run_step("build_spatial", cfg)
run_step("map_hlca", cfg)
run_step("run_tangram", cfg)

train_cfg = compose_config("train", overrides=["data=local", "training.max_epochs=1"])
run_step("train", train_cfg)
```

## Full pipeline helper

```python
cfg = compose_config("config", overrides=["data=local"])
run_pipeline(cfg, steps=["build_snrna", "build_spatial", "map_hlca", "run_tangram", "train"])
```

## Load run artifacts

```python
from stagebridge.notebook_api import load_run
payload = load_run("outputs/post_tangram_smoke")
```
