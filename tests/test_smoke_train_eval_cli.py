import json
import subprocess
import sys

import numpy as np
import pytest

hydra = pytest.importorskip("hydra")
anndata = pytest.importorskip("anndata")



def test_train_and_eval_cli_smoke(tmp_path):
    n = 400
    d = 16
    rng = np.random.default_rng(1)

    adata = anndata.AnnData(X=np.zeros((n, 1), dtype=np.float32))
    adata.obsm["X_pca"] = rng.normal(size=(n, d)).astype(np.float32)
    adata.obs["stage"] = np.array(["Normal", "AAH", "AIS", "MIA", "LUAD"] * (n // 5), dtype=object)
    adata.obs["patient_id"] = np.array([f"D{i%8}" for i in range(n)], dtype=object)

    snrna_path = tmp_path / "snrna_merged.h5ad"
    adata.write_h5ad(snrna_path)

    run_name = "cli_smoke"
    cmd_train = [
        sys.executable,
        "scripts/train_stagebridge.py",
        f"run_name={run_name}",
        f"output_dir={tmp_path}",
        f"data.snrna_h5ad={snrna_path}",
        "data.use_hlca_reference=false",
        "training.max_epochs=1",
        "training.steps_per_epoch=1",
        "training.val_steps=1",
        "training.batch_cells=64",
        "training.num_ot_pairs=64",
        "training.device=cpu",
        "training.mixed_precision=false",
        "splits.n_folds=2",
        "experiment.baseline_models=[]",
        "experiment.ablations=[]",
    ]
    subprocess.run(cmd_train, check=True)

    metrics_path = tmp_path / "tables" / f"metrics_{run_name}.json"
    assert metrics_path.exists()
    run_manifest_path = tmp_path / "tables" / f"run_manifest_{run_name}.json"
    assert run_manifest_path.exists()
    payload = json.loads(metrics_path.read_text())
    ckpt = payload["results"]["stagebridge"]["checkpoints"][0]

    cmd_eval = [
        sys.executable,
        "scripts/eval_stagebridge.py",
        f"run_name={run_name}",
        f"output_dir={tmp_path}",
        f"data.snrna_h5ad={snrna_path}",
        "training.device=cpu",
        "training.mixed_precision=false",
        f"checkpoint={ckpt}",
    ]
    subprocess.run(cmd_eval, check=True)

    eval_path = tmp_path / "tables" / f"eval_{run_name}.json"
    assert eval_path.exists()
