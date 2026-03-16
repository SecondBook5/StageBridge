"""Held-out evaluation entrypoint for one trained EA-MIST lesion run."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Subset

from stagebridge.data.luad_evo.bag_dataset import LesionBagDataset, NeighborhoodPretrainDataset, collate_lesion_bags
from stagebridge.data.luad_evo.neighborhood_builder import build_lesion_bags_from_config
from stagebridge.data.luad_evo.splits import build_multitask_lesion_folds
from stagebridge.evaluation.eamist_metrics import (
    CANONICAL_STAGE_LABELS,
    compute_masked_edge_metrics,
    stage_confusion_matrix_payload,
    stage_support_payload,
)
from stagebridge.pipelines.pretrain_local import infer_local_feature_dims
from stagebridge.pipelines.train_lesion import (
    _cfg_select,
    _compute_stage_class_weights,
    _epoch_metrics,
    _prediction_frame,
    _run_epoch,
    build_model_family,
)
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
    dataset = LesionBagDataset(build_result.bags)
    dims = infer_local_feature_dims(NeighborhoodPretrainDataset(build_result.bags))
    evolution_dim = int(model_spec.get("evolution_dim") or 0)
    folds = build_multitask_lesion_folds(
        build_result.bags,
        holdout_key=str(_cfg_select(checkpoint_cfg, "context_model.eamist.holdout_key", "donor_id")),
        num_folds=int(_cfg_select(checkpoint_cfg, "context_model.eamist.outer_folds", 3)),
        seed=int(_cfg_select(checkpoint_cfg, "seed", _cfg_select(cfg, "seed", 42))),
    )
    fold_index = int(split_summary["fold"]["fold_index"])
    fold = folds[fold_index]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model_family(
        str(model_spec["model_family"]),
        dims,
        cfg=checkpoint_cfg,
        evolution_dim=evolution_dim if evolution_dim > 0 else None,
        num_edge_heads=int(model_spec.get("num_edge_heads", 0)),
        reference_feature_mode=str(model_spec.get("reference_feature_mode", "hlca_luca")),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.eval()

    train_bags = [build_result.bags[idx] for idx in fold.train_indices]
    stage_class_weights = _compute_stage_class_weights(train_bags, num_stage_classes=len(CANONICAL_STAGE_LABELS)).to(device)
    test_loader = DataLoader(
        Subset(dataset, list(fold.test_indices)),
        batch_size=int(_cfg_select(checkpoint_cfg, "context_model.eamist.batch_size_bags", 8)),
        shuffle=False,
        collate_fn=collate_lesion_bags,
    )
    test_epoch = _run_epoch(model, test_loader, device=device, optimizer=None, cfg=checkpoint_cfg, stage_class_weights=stage_class_weights)
    edge_target_labels = tuple(str(label) for label in model_spec.get("edge_target_labels", []))
    metrics = _epoch_metrics(test_epoch, edge_target_labels=edge_target_labels)
    prediction_frame = _prediction_frame(build_result.bags, fold.test_indices, test_epoch)
    auxiliary_edge_metrics = compute_masked_edge_metrics(
        test_epoch["edge_logits"],
        test_epoch["edge_targets"],
        test_epoch["edge_masks"],
        edge_labels=edge_target_labels,
    )
    confusion = stage_confusion_matrix_payload(test_epoch["stage_targets"], test_epoch["stage_predictions"])
    support = stage_support_payload(test_epoch["stage_targets"])

    prediction_frame.to_parquet(fold_root / "evaluation_predictions.parquet", index=False)
    (fold_root / "evaluation_confusion_matrix.json").write_text(json.dumps(confusion, indent=2), encoding="utf-8")
    (fold_root / "evaluation_metrics.json").write_text(json.dumps({**metrics, "support": support}, indent=2), encoding="utf-8")
    (fold_root / "evaluation_auxiliary_edge_metrics.json").write_text(json.dumps(auxiliary_edge_metrics, indent=2), encoding="utf-8")
    return {
        "ok": True,
        "pipeline": "evaluate_lesion",
        "status": "complete",
        "artifact_root": str(fold_root),
        "metrics": {**metrics, **auxiliary_edge_metrics},
    }


__all__ = ["run_evaluate_lesion"]
