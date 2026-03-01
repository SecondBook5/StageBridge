import numpy as np
import pytest

anndata = pytest.importorskip("anndata")

from stagebridge.models.stagebridge import StageBridgeModel
from stagebridge.training.trainer import (
    StageBridgeTrainer,
    build_donor_holdout_splits,
    build_samplers_from_anndata,
)
from stagebridge.utils.types import StageBridgeConfig



def test_mini_training_epoch_writes_checkpoint(tmp_path):
    n = 600
    d = 16
    rng = np.random.default_rng(0)

    X = rng.normal(size=(n, d)).astype(np.float32)
    stages = np.array(["Normal", "AAH", "AIS", "MIA", "LUAD"] * (n // 5), dtype=object)
    donors = np.array([f"D{i%6}" for i in range(n)], dtype=object)

    adata = anndata.AnnData(X=np.zeros((n, 1), dtype=np.float32))
    adata.obsm["X_hlca"] = X
    adata.obs["stage"] = stages
    adata.obs["patient_id"] = donors

    splits = build_donor_holdout_splits(donor_ids=sorted(set(donors)), n_folds=3, seed=42)
    train_sampler, val_sampler, test_sampler = build_samplers_from_anndata(
        adata=adata,
        split=splits[0],
        latent_key="X_hlca",
        stage_col="stage",
        donor_col="patient_id",
        batch_cells=64,
        device="cpu",
    )

    cfg = StageBridgeConfig(
        input_dim=d,
        hidden_dim=32,
        vector_field_hidden_dim=64,
        num_heads=4,
        num_inducing_points=8,
        max_epochs=2,
        steps_per_epoch=2,
        val_steps=1,
        patience=2,
        num_ot_pairs=64,
        mixed_precision=False,
        device="cpu",
    )

    model = StageBridgeModel(cfg)
    trainer = StageBridgeTrainer(model=model, config=cfg)

    out = trainer.fit(
        train_sampler=train_sampler,
        val_sampler=val_sampler,
        test_sampler=test_sampler,
        output_dir=tmp_path,
        run_name="smoke",
    )

    assert out.best_checkpoint.exists()
    assert len(out.history) >= 1
