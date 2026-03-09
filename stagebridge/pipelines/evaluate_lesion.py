"""Held-out evaluation entrypoint for one trained EA-MIST lesion run."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Subset

from stagebridge.data.luad_evo.bag_dataset import LesionBagDataset, NeighborhoodPretrainDataset, collate_lesion_bags
from stagebridge.data.luad_evo.neighborhood_builder import build_lesion_bags_from_config
from stagebridge.data.luad_evo.splits import build_lesion_folds
from stagebridge.evaluation.eamist_metrics import (
    bootstrap_confidence_intervals,
    build_curve_frames,
    build_per_donor_metrics,
    compute_binary_metrics,
    confusion_matrix_payload,
)
from stagebridge.pipelines.train_lesion import _cfg_select, _model_forward, build_model_family, load_pretrained_local_encoder
from stagebridge.pipelines.pretrain_local import infer_local_feature_dims
from stagebridge.utils.seeds import seed_everything


def run_evaluate_lesion(
    cfg: DictConfig | dict[str, Any],
    *,
    checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate a trained lesion model on its held-out donor fold."""
    resolved_checkpoint = Path(str(checkpoint_path or _cfg_select(cfg, "context_model.eamist.checkpoint_path", "")))
    if not resolved_checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {resolved_checkpoint}")
    fold_root = resolved_checkpoint.parent
    checkpoint = torch.load(resolved_checkpoint, map_location="cpu")
    checkpoint_cfg = checkpoint.get("config", cfg)
    split_summary_path = fold_root / "split_summary.json"
    model_spec_path = fold_root / "model_spec.json"
    if not split_summary_path.exists() or not model_spec_path.exists():
        raise FileNotFoundError("Evaluation requires split_summary.json and model_spec.json alongside the checkpoint.")

    split_summary = json.loads(split_summary_path.read_text(encoding="utf-8"))
    model_spec = json.loads(model_spec_path.read_text(encoding="utf-8"))
    seed_everything(int(_cfg_select(checkpoint_cfg, "seed", _cfg_select(cfg, "seed", 42))))

    build_result = build_lesion_bags_from_config(checkpoint_cfg)
    edge_bags = [bag for bag in build_result.bags if bag.edge_label == str(model_spec["edge_label"])]
    dataset = LesionBagDataset(edge_bags)
    dims = infer_local_feature_dims(NeighborhoodPretrainDataset(edge_bags))
    evolution_dim = int(model_spec.get("evolution_dim") or 0)
    folds = build_lesion_folds(
        edge_bags,
        holdout_key=str(_cfg_select(checkpoint_cfg, "context_model.eamist.holdout_key", "donor_id")),
        num_folds=int(_cfg_select(checkpoint_cfg, "context_model.eamist.outer_folds", 3)),
        seed=int(_cfg_select(checkpoint_cfg, "seed", _cfg_select(cfg, "seed", 42))),
        min_lesions_per_class=int(_cfg_select(checkpoint_cfg, "context_model.eamist.min_lesions_per_class", 1)),
    )
    fold_index = int(split_summary["fold"]["fold_index"])
    fold = folds[fold_index]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model_family(str(model_spec["model_family"]), dims, cfg=checkpoint_cfg, evolution_dim=evolution_dim if evolution_dim > 0 else None).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    if hasattr(model, "local_encoder"):
        load_pretrained_local_encoder(model, _cfg_select(checkpoint_cfg, "context_model.eamist.pretrained_local_checkpoint", None))
    model.eval()

    test_loader = DataLoader(Subset(dataset, list(fold.test_indices)), batch_size=int(_cfg_select(checkpoint_cfg, "context_model.eamist.batch_size_bags", 8)), shuffle=False, collate_fn=collate_lesion_bags)
    logits: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for batch in test_loader:
        batch = batch.to(device)
        batch_logits, _ = _model_forward(model, batch)
        logits.append(batch_logits.detach().cpu().numpy())
        labels.append(batch.labels.detach().cpu().numpy())
    logits_np = np.concatenate(logits, axis=0)
    labels_np = np.concatenate(labels, axis=0)
    temperature = float(checkpoint["val_metrics"]["temperature"])
    threshold = float(checkpoint["val_metrics"]["threshold"])
    probabilities = 1.0 / (1.0 + np.exp(-logits_np / temperature))
    metrics = compute_binary_metrics(labels_np, probabilities, threshold=threshold)
    intervals = bootstrap_confidence_intervals(labels_np, probabilities, seed=int(_cfg_select(checkpoint_cfg, "seed", _cfg_select(cfg, "seed", 42))))
    prediction_rows = []
    for local_idx, bag_index in enumerate(fold.test_indices):
        bag = edge_bags[int(bag_index)]
        prediction_rows.append(
            {
                "lesion_id": bag.lesion_id,
                "sample_id": bag.sample_id,
                "donor_id": bag.donor_id,
                "patient_id": bag.patient_id,
                "stage": bag.stage,
                "edge_label": bag.edge_label,
                "label": float(labels_np[local_idx]),
                "probability": float(probabilities[local_idx]),
                "label_source": bag.label_source,
            }
        )
    prediction_frame = pd.DataFrame(prediction_rows)
    roc_df, pr_df, cal_df = build_curve_frames(labels_np, probabilities)
    per_donor = build_per_donor_metrics(prediction_frame, threshold=threshold)
    confusion = confusion_matrix_payload(labels_np, probabilities, threshold=threshold)

    prediction_frame.to_parquet(fold_root / "evaluation_predictions.parquet", index=False)
    roc_df.to_csv(fold_root / "evaluation_roc_curve.csv", index=False)
    pr_df.to_csv(fold_root / "evaluation_pr_curve.csv", index=False)
    cal_df.to_csv(fold_root / "evaluation_calibration_curve.csv", index=False)
    per_donor.to_csv(fold_root / "evaluation_per_donor_metrics.csv", index=False)
    (fold_root / "evaluation_confusion_matrix.json").write_text(json.dumps(confusion, indent=2), encoding="utf-8")
    (fold_root / "evaluation_metrics.json").write_text(json.dumps({**metrics, **intervals}, indent=2), encoding="utf-8")
    return {
        "ok": True,
        "pipeline": "evaluate_lesion",
        "status": "complete",
        "artifact_root": str(fold_root),
        "metrics": {**metrics, **intervals},
    }


__all__ = ["run_evaluate_lesion"]
