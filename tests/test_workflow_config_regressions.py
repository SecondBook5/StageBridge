from pathlib import Path

import pytest

from stagebridge.notebook_api import compose_config
from stagebridge.utils.config_helpers import build_stagebridge_config_from_cfg
from stagebridge.workflows.eval import _build_eval_model_cfg, _fold_index_from_checkpoint_name
from stagebridge.workflows.train import _build_variant_specs


def test_full_benchmark_variant_and_ablation_labels_present() -> None:
    cfg = compose_config("train", overrides=["experiment=full_benchmark"])
    specs = _build_variant_specs(cfg)
    labels = {str(spec["label"]) for spec in specs}

    expected = {
        "stagebridge_lr",
        "stagebridge_spatial",
        "stagebridge_dirichlet",
        "stagebridge_multihop",
        "stagebridge_full_t3",
        "stagebridge__ablation_no_lr_features",
        "stagebridge__ablation_no_spatial_niche",
        "stagebridge__ablation_no_dirichlet",
        "stagebridge__ablation_no_multihop",
    }
    assert expected.issubset(labels)


def test_unknown_variant_raises_explicit_error() -> None:
    cfg = compose_config(
        "train",
        overrides=[
            "experiment.baseline_models=[]",
            "experiment.ablations=[]",
            "experiment.variants=[totally_unknown_variant]",
        ],
    )
    with pytest.raises(ValueError, match="Unsupported experiment variant"):
        _build_variant_specs(cfg)


def test_eval_model_cfg_restores_checkpoint_architecture_flags() -> None:
    cfg = compose_config("eval", overrides=["training.device=cpu"])
    ckpt_cfg = {
        "use_cross_attn_drift": True,
        "use_wes_features": True,
        "use_lr_features": True,
        "use_spatial_niche": True,
        "use_dirichlet_head": True,
        "use_multihop_consistency": True,
        "num_seed_vectors": 4,
    }
    model_cfg = _build_eval_model_cfg(cfg, ckpt_cfg)

    assert model_cfg.use_cross_attn_drift is True
    assert model_cfg.use_wes_features is True
    assert model_cfg.use_lr_features is True
    assert model_cfg.use_spatial_niche is True
    assert model_cfg.use_dirichlet_head is True
    assert model_cfg.use_multihop_consistency is True
    assert model_cfg.num_seed_vectors == 4
    assert model_cfg.resolved_device() == "cpu"


def test_fold_index_from_checkpoint_name() -> None:
    assert _fold_index_from_checkpoint_name(Path("run_stagebridge_fold1_best.pt"), n_folds=3) == 1
    assert _fold_index_from_checkpoint_name(Path("run_stagebridge_best.pt"), n_folds=3) == 0
    assert _fold_index_from_checkpoint_name(Path("run_stagebridge_fold8_best.pt"), n_folds=3) == 0


def test_build_stagebridge_config_from_cfg_respects_input_dim_override() -> None:
    cfg = compose_config("train", overrides=["training.device=cpu"])
    sb_cfg = build_stagebridge_config_from_cfg(cfg, input_dim=17)
    assert sb_cfg.input_dim == 17
    assert sb_cfg.resolved_device() == "cpu"
