from __future__ import annotations

from pathlib import Path
import importlib

import numpy as np
from omegaconf import OmegaConf

from stagebridge.data.luad_evo.neighborhood_builder import NeighborhoodBuildResult
from stagebridge.pipelines.evaluate_lesion import run_evaluate_lesion
from stagebridge.pipelines.pretrain_local import run_pretrain_local
from stagebridge.pipelines.run_eamist_reporting import run_eamist_reporting
from stagebridge.pipelines.train_lesion import run_train_lesion
from stagebridge.utils.types import LesionBag, LocalNicheExample


def _make_bag(sample_id: str, donor_id: str, label: float, shift: float) -> LesionBag:
    stage = "AIS" if label >= 0.5 else "AAH"
    stage_index = 2 if stage == "AIS" else 1
    neighborhoods = []
    for idx in range(3):
        receiver = np.asarray([1.0 + shift, 0.5 + idx * 0.1, label], dtype=np.float32)
        rings = np.asarray(
            [
                [0.7 if label else 0.3, 0.2 + shift, 0.1],
                [0.2, 0.6 if label else 0.3, 0.2],
                [0.1, 0.2, 0.7 + shift],
                [0.2 + shift, 0.3, 0.5 if label else 0.2],
            ],
            dtype=np.float32,
        )
        hlca = np.asarray([0.9 - 0.2 * label, 0.2 + shift], dtype=np.float32)
        luca = np.asarray([0.2 + 0.5 * label, 0.4 + shift, 0.3], dtype=np.float32)
        lr = np.asarray([0.8 * label + 0.1, 0.3 + shift, 0.2, 0.1], dtype=np.float32)
        stats = np.asarray([3.0 + idx, 0.2, 0.3, 0.5, 0.9, 1.0], dtype=np.float32)
        neighborhoods.append(
            LocalNicheExample(
                lesion_id=sample_id,
                sample_id=sample_id,
                donor_id=donor_id,
                patient_id=donor_id,
                stage=stage,
                edge_label="AIS->MIA" if stage == "AIS" else "AAH->AIS",
                receiver_index=idx,
                receiver_embedding=receiver,
                receiver_state_id=idx % 2,
                ring_compositions=rings,
                lr_pathway_summary=lr,
                neighborhood_stats=stats,
                flat_features=np.concatenate(
                    [receiver, rings.reshape(-1), hlca, luca, lr, stats]
                ).astype(np.float32),
                center_coord=np.asarray([float(idx), float(idx + 1)], dtype=np.float32),
                hlca_features=hlca,
                luca_features=luca,
                receiver_confidence=0.9,
            )
        )
    return LesionBag(
        lesion_id=sample_id,
        sample_id=sample_id,
        donor_id=donor_id,
        patient_id=donor_id,
        stage=stage,
        edge_id=1 if stage == "AIS" else 0,
        edge_label="AIS->MIA" if stage == "AIS" else "AAH->AIS",
        label=float(label),
        label_weight=1.0,
        label_source="synthetic",
        neighborhoods=neighborhoods,
        evolution_features=np.asarray([0.1 + shift, label], dtype=np.float32),
        stage_index=stage_index,
        displacement_target=float(stage_index) / 4.0,
        edge_targets=np.asarray([float(1.0 - label), float(label)], dtype=np.float32),
        edge_target_mask=np.asarray([True, True], dtype=bool),
        edge_target_labels=("AAH->AIS", "AIS->MIA"),
    )


def _synthetic_build_result() -> NeighborhoodBuildResult:
    bags = [
        _make_bag("S1", "P1", 1.0, 0.0),
        _make_bag("S2", "P2", 0.0, 0.1),
        _make_bag("S3", "P3", 1.0, 0.2),
        _make_bag("S4", "P4", 0.0, 0.3),
    ]
    summary = {
        "lesion_id": [bag.lesion_id for bag in bags],
        "sample_id": [bag.sample_id for bag in bags],
        "donor_id": [bag.donor_id for bag in bags],
        "patient_id": [bag.patient_id for bag in bags],
        "stage": [bag.stage for bag in bags],
        "stage_index": [bag.stage_index for bag in bags],
        "displacement_target": [bag.displacement_target for bag in bags],
        "edge_label": [bag.edge_label for bag in bags],
        "label": [bag.label for bag in bags],
        "label_weight": [bag.label_weight for bag in bags],
        "label_source": [bag.label_source for bag in bags],
        "num_neighborhoods": [bag.num_neighborhoods for bag in bags],
        "evolution_feature_dim": [2 for _ in bags],
    }
    label_table = {
        "sample_id": [bag.sample_id for bag in bags],
        "donor_id": [bag.donor_id for bag in bags],
        "patient_id": [bag.patient_id for bag in bags],
        "stage": [bag.stage for bag in bags],
        "edge_label": [bag.edge_label for bag in bags],
        "label": [bag.label for bag in bags],
        "label_weight": [bag.label_weight for bag in bags],
        "label_source": [bag.label_source for bag in bags],
        "notes": ["synthetic" for _ in bags],
    }
    return NeighborhoodBuildResult(
        bags=bags,
        summary=__import__("pandas").DataFrame(summary),
        label_table=__import__("pandas").DataFrame(label_table),
        diagnostics={"num_bags": 4, "num_instances": 12},
    )


def test_eamist_pretrain_train_evaluate_and_report(tmp_path: Path, monkeypatch) -> None:
    build_result = _synthetic_build_result()
    pretrain_module = importlib.import_module("stagebridge.pipelines.pretrain_local")
    train_module = importlib.import_module("stagebridge.pipelines.train_lesion")
    evaluate_module = importlib.import_module("stagebridge.pipelines.evaluate_lesion")
    report_module = importlib.import_module("stagebridge.pipelines.run_eamist_reporting")
    monkeypatch.setattr(pretrain_module, "build_lesion_bags_from_config", lambda cfg: build_result)
    monkeypatch.setattr(train_module, "build_lesion_bags_from_config", lambda cfg: build_result)
    monkeypatch.setattr(evaluate_module, "build_lesion_bags_from_config", lambda cfg: build_result)
    monkeypatch.setattr(report_module, "build_lesion_bags_from_config", lambda cfg: build_result)

    cfg = OmegaConf.create(
        {
            "run_name": "eamist_smoke",
            "output_dir": str(tmp_path / "outputs"),
            "reports_root": str(tmp_path / "reports"),
            "seed": 7,
            "context_model": {
                "eamist": {
                    "model_families": ["pooled", "eamist"],
                    "reference_feature_modes": ["hlca_luca", "hlca_only"],
                    "outer_folds": 2,
                    "seeds": [7],
                    "batch_size_bags": 2,
                    "max_epochs": 1,
                    "patience": 1,
                    "device": "cpu",
                    "min_lesions_per_class": 0,
                    "hidden_dim": 16,
                    "num_heads": 4,
                    "num_layers": 1,
                    "dropout": 0.0,
                    "use_prototypes": True,
                    "num_prototypes": 8,
                    "use_evolution_branch": True,
                    "local_encoder_training_mode": "full",
                    "pretrained_local_checkpoint": None,
                    "hpo": {
                        "enabled": True,
                        "backend": "optuna",
                        "num_trials": 2,
                        "seed": 11,
                        "sampler": "tpe",
                        "n_startup_trials": 1,
                        "n_warmup_steps": 1,
                        "search_space": {
                            "shared": {
                                "learning_rate": [1e-3, 5e-4],
                                "dropout": [0.0, 0.1],
                            },
                            "eamist": {
                                "num_prototypes": [8, 12],
                            },
                        },
                    },
                    "local_pretraining": {
                        "encoder_type": "transformer",
                        "hidden_dim": 16,
                        "num_heads": 4,
                        "dropout": 0.0,
                        "batch_size": 4,
                        "max_epochs": 1,
                        "learning_rate": 1e-3,
                        "weight_decay": 0.0,
                    },
                }
            },
            "eamist_report": {
                "reports_root": str(tmp_path / "reports"),
                "benchmark_root": str(tmp_path / "outputs" / "eamist_smoke" / "eamist_benchmark"),
            },
        }
    )

    pretrain = run_pretrain_local(cfg)
    assert pretrain["ok"] is True
    assert Path(pretrain["best_checkpoint"]).exists()

    cfg.context_model.eamist.pretrained_local_checkpoint = pretrain["best_checkpoint"]
    train = run_train_lesion(cfg)
    assert train["ok"] is True
    benchmark_root = Path(train["artifact_root"])
    assert (benchmark_root / "benchmark_summary.csv").exists()
    assert next(benchmark_root.glob("*/*/fold_*/hpo_trial_summary.csv")).exists()
    assert next(benchmark_root.glob("*/*/fold_*/best_hpo_config.json")).exists()
    assert next(benchmark_root.glob("*/*/fold_*/seed_*/selected_hyperparameters.json")).exists()

    checkpoint = next(benchmark_root.glob("*/*/fold_*/seed_*/best_checkpoint.pt"))
    evaluate = run_evaluate_lesion(cfg, checkpoint_path=checkpoint)
    assert evaluate["ok"] is True
    assert (checkpoint.parent / "evaluation_metrics.json").exists()

    report = run_eamist_reporting(cfg)
    assert report["ok"] is True
    assert (Path(report["tables_root"]) / "table1_dataset_composition.csv").exists()
    assert (Path(report["figures_root"]) / "figure3_benchmark_comparison.png").exists()
