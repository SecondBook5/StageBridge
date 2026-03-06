import json
from pathlib import Path

import numpy as np
import pytest

anndata = pytest.importorskip("anndata")

from stagebridge.notebook_api import compose_config, run_pipeline, run_step  # noqa: E402


def _write_synthetic_snrna(path: Path, n: int = 300, d: int = 12) -> Path:
    rng = np.random.default_rng(7)
    adata = anndata.AnnData(X=np.zeros((n, 1), dtype=np.float32))
    adata.obsm["X_pca"] = rng.normal(size=(n, d)).astype(np.float32)
    adata.obs["stage"] = np.array(["Normal", "AAH", "AIS", "MIA", "LUAD"] * (n // 5), dtype=object)
    adata.obs["patient_id"] = np.array([f"D{i % 6}" for i in range(n)], dtype=object)
    adata.write_h5ad(path)
    return path


def test_notebook_api_pipeline_dispatch_order():
    cfg = compose_config("config")

    calls: list[str] = []

    def _mk(name: str):
        def _fn(_cfg):
            calls.append(name)
            return {"ok": True, "step": name}

        return _fn

    import stagebridge.notebook_api as api

    original = dict(api._STEP_REGISTRY)
    try:
        api._STEP_REGISTRY = {
            "build_snrna": _mk("build_snrna"),
            "build_spatial": _mk("build_spatial"),
        }
        out = run_pipeline(cfg, steps=["build_snrna", "build_spatial"])
    finally:
        api._STEP_REGISTRY = original

    assert calls == ["build_snrna", "build_spatial"]
    assert out["build_snrna"]["ok"] is True
    assert out["build_spatial"]["ok"] is True


def test_notebook_api_train_and_eval_steps(tmp_path: Path):
    snrna_path = _write_synthetic_snrna(tmp_path / "snrna_merged.h5ad")
    train_out = tmp_path / "train_out"

    train_cfg = compose_config(
        "train",
        overrides=[
            "run_name=workflow_contract",
            f"output_dir={train_out}",
            f"data.snrna_h5ad={snrna_path}",
            "data.use_hlca_reference=false",
            "training.max_epochs=1",
            "training.steps_per_epoch=1",
            "training.val_steps=1",
            "training.batch_cells=48",
            "training.num_ot_pairs=48",
            "training.device=cpu",
            "training.mixed_precision=false",
            "splits.n_folds=2",
            "experiment.baseline_models=[deepsets]",
            "experiment.ablations=[]",
            "experiment.variants=[]",
        ],
    )
    train_summary = run_step("train", train_cfg)
    assert train_summary["ok"] is True

    metrics_path = Path(str(train_summary["metrics_json"]))
    assert metrics_path.exists()
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    checkpoint = payload["results"]["stagebridge"]["checkpoints"][0]

    eval_cfg = compose_config(
        "eval",
        overrides=[
            "run_name=workflow_contract_eval",
            f"output_dir={train_out}",
            f"checkpoint={checkpoint}",
            f"data.snrna_h5ad={snrna_path}",
            "training.device=cpu",
            "training.mixed_precision=false",
            "splits.n_folds=2",
        ],
    )
    eval_summary = run_step("evaluate", eval_cfg)

    assert eval_summary["ok"] is True
    assert Path(str(eval_summary["eval_json"])).exists()
    assert Path(str(eval_summary["run_manifest_json"])).exists()
