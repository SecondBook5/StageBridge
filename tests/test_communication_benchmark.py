from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from stagebridge.transition_model.disease_edges import edge_id_map
from stagebridge.utils.types import CommunicationBag, CommunicationNeighborhoodExample

communication_benchmark_module = importlib.import_module("stagebridge.pipelines.run_communication_benchmark")


def _make_bag(sample_id: str, donor_id: str, edge_label: str, weak_label: float, shift: float) -> CommunicationBag:
    edge_lookup = edge_id_map()
    example = CommunicationNeighborhoodExample(
        receiver_embedding=np.asarray([2.0 * weak_label + shift, 1.0 - weak_label, 0.2], dtype=np.float32),
        receiver_programs=np.asarray([1.5 * weak_label + 0.1, 0.3 + shift, 1.0 - weak_label], dtype=np.float32),
        sender_embeddings=np.asarray([[1.0 + shift, 0.1], [0.8 + shift, 0.2]], dtype=np.float32),
        sender_types=np.asarray([0, 1], dtype=np.int64),
        sender_offsets=np.asarray([[0.0, 0.0], [0.2, 0.1]], dtype=np.float32),
        ring_ids=np.asarray([0, 1], dtype=np.int64),
        lr_token_features=np.asarray([[0.9 * weak_label + 0.1, 0.7, 0.8 * weak_label + 0.1, 1.0, 0.1, 0.0, 0.8, 0.0, 0.0, 0.6]], dtype=np.float32),
        response_token_features=np.asarray([[0.8 * weak_label + 0.1, 0.7, 0.0, 4.0, float(edge_lookup[edge_label])]], dtype=np.float32),
        relay_token_features=np.asarray([[0.7 * weak_label + 0.1, 0.8, 0.56, 0.6, 0.0, 0.0]], dtype=np.float32),
        edge_id=edge_lookup[edge_label],
        sample_id=sample_id,
        donor_id=donor_id,
        weak_label=weak_label,
        receiver_cell_id=f"{sample_id}_c0",
        lr_token_names=["IL1B->IL1R1|inflammatory|sender_0"],
        response_token_names=["inflammatory_response|inflammatory"],
        relay_token_names=["inflammatory_relay|inflammatory_response"],
    )
    return CommunicationBag(
        sample_id=sample_id,
        donor_id=donor_id,
        edge_id=edge_lookup[edge_label],
        edge_label=edge_label,
        weak_label=weak_label,
        examples=[example],
        label_source="synthetic",
    )


def test_run_communication_benchmark_writes_artifacts(tmp_path: Path, monkeypatch) -> None:
    bags = [
        _make_bag("S1", "P1", "AAH->AIS", 1.0, 0.0),
        _make_bag("S2", "P2", "AAH->AIS", 0.0, 0.1),
        _make_bag("S3", "P3", "AAH->AIS", 1.0, 0.2),
        _make_bag("S4", "P4", "AIS->MIA", 0.0, 0.3),
        _make_bag("S5", "P5", "AIS->MIA", 1.0, 0.4),
        _make_bag("S6", "P6", "AIS->MIA", 0.0, 0.5),
    ]
    bag_summary = pd.DataFrame(
        {
            "sample_id": [bag.sample_id for bag in bags],
            "donor_id": [bag.donor_id for bag in bags],
            "edge_label": [bag.edge_label for bag in bags],
            "weak_label": [bag.weak_label for bag in bags],
            "label_source": [bag.label_source for bag in bags],
        }
    )

    monkeypatch.setattr(communication_benchmark_module, "load_luad_evo_snrna_latent", lambda *args, **kwargs: object())
    monkeypatch.setattr(communication_benchmark_module, "load_luad_evo_spatial_mapping", lambda *args, **kwargs: object())
    monkeypatch.setattr(communication_benchmark_module, "load_luad_evo_wes_features", lambda *args, **kwargs: object())
    monkeypatch.setattr(communication_benchmark_module, "build_communication_bags", lambda *args, **kwargs: (bags, bag_summary))

    cfg = OmegaConf.create(
        {
            "run_name": "comm_smoke",
            "output_dir": str(tmp_path),
            "seed": 13,
            "context_model": {
                "communication_relay": {
                    "active_edges": ["AAH->AIS", "AIS->MIA"],
                    "model_families": ["focal_only"],
                    "outer_folds": 3,
                    "seeds": [13],
                    "num_trials": 1,
                    "batch_size_bags": 2,
                    "max_epochs": 1,
                    "patience": 1,
                    "hidden_dim": 32,
                    "dropout": 0.0,
                    "learning_rate": 1e-3,
                    "weight_decay": 0.0,
                    "grad_clip_norm": 1.0,
                }
            },
        }
    )

    result = communication_benchmark_module.run_communication_benchmark(cfg)

    assert result["ok"] is True
    assert result["status"] == "complete"
    artifact_root = Path(result["artifact_root"])
    assert artifact_root.exists()
    assert (artifact_root / "benchmark_summary.csv").exists()
    fold_dir = artifact_root / "focal_only" / "fold_00" / "seed_013"
    assert (fold_dir / "train_history.csv").exists()
    assert (fold_dir / "val_history.csv").exists()
    assert (fold_dir / "metrics.json").exists()
    assert (fold_dir / "test_predictions.parquet").exists()
    assert (fold_dir / "roc_curve.csv").exists()
    assert (fold_dir / "pr_curve.csv").exists()
    assert (fold_dir / "calibration_curve.csv").exists()
    assert (fold_dir / "confusion_matrix.json").exists()
