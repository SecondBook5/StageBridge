import json
import subprocess
import sys

import numpy as np
import pytest

hydra = pytest.importorskip("hydra")
anndata = pytest.importorskip("anndata")


def test_ablation_flags_execute_and_report(tmp_path):
    n = 300
    d = 12
    rng = np.random.default_rng(3)

    adata = anndata.AnnData(X=np.zeros((n, 1), dtype=np.float32))
    adata.obsm["X_pca"] = rng.normal(size=(n, d)).astype(np.float32)
    adata.obs["stage"] = np.array(["Normal", "AAH", "AIS", "MIA", "LUAD"] * (n // 5), dtype=object)
    adata.obs["patient_id"] = np.array([f"D{i%6}" for i in range(n)], dtype=object)

    snrna_path = tmp_path / "snrna_merged.h5ad"
    adata.write_h5ad(snrna_path)

    run_name = "ablation_smoke"
    cmd = [
        sys.executable,
        "scripts/train_stagebridge.py",
        f"run_name={run_name}",
        f"output_dir={tmp_path}",
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
        "experiment.ablations=[no_ot,no_stage_embedding]",
    ]
    subprocess.run(cmd, check=True)

    metrics_path = tmp_path / "tables" / f"metrics_{run_name}.json"
    payload = json.loads(metrics_path.read_text())
    keys = set(payload["results"].keys())

    assert "stagebridge" in keys
    assert "deepsets" in keys
    assert "stagebridge__ablation_no_ot" in keys
    assert "stagebridge__ablation_no_stage_embedding" in keys
