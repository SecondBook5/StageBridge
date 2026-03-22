"""Communication-relay classification benchmark for StageBridge."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from torch import Tensor

from stagebridge.context_model.communication_builder import build_communication_bags
from stagebridge.context_model.communication_relay import (
    CommunicationRelayOutput,
    StageBridgeCommunicationModel,
    build_communication_model,
    collate_communication_bags,
)
from stagebridge.data.luad_evo.snrna import load_luad_evo_snrna_latent
from stagebridge.data.luad_evo.visium import load_luad_evo_spatial_mapping
from stagebridge.data.luad_evo.wes import load_luad_evo_wes_features
from stagebridge.evaluation.classification import (
    TemperatureScaler,
    binary_classification_metrics,
    calibration_curve_table,
    choose_threshold,
    curve_tables,
    fit_temperature_scaler,
)
from stagebridge.transition_model.disease_edges import edge_id_map
from stagebridge.transition_model.train import build_donor_holdout_splits
from stagebridge.utils.types import CommunicationBag, CommunicationBatch


def _cfg(section: DictConfig, key: str, default: Any) -> Any:
    value = section.get(key, default)
    return default if value is None else value


def _communication_cfg(cfg: DictConfig) -> DictConfig:
    context_cfg = cfg.get("context_model", {})
    relay_cfg = context_cfg.get("communication_relay", {})
    if not relay_cfg:
        relay_cfg = {}
    return relay_cfg


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _bag_batches(
    bags: list[CommunicationBag],
    *,
    batch_size: int,
    seed: int,
    shuffle: bool,
) -> Iterable[list[CommunicationBag]]:
    rng = np.random.default_rng(int(seed))
    order = np.arange(len(bags))
    if shuffle and order.size > 1:
        rng.shuffle(order)
    for start in range(0, len(order), int(batch_size)):
        yield [bags[idx] for idx in order[start : start + int(batch_size)].tolist()]


def _prior_targets(batch: CommunicationBatch) -> tuple[Tensor, Tensor]:
    lr_query = (
        batch.lr_token_features[:, :, 2] * batch.lr_mask.to(batch.lr_token_features.dtype)
    ).sum(dim=1)
    lr_query = lr_query / batch.lr_mask.to(batch.lr_token_features.dtype).sum(dim=1).clamp_min(1.0)
    response_query = (
        batch.response_token_features[:, :, 0]
        * batch.response_mask.to(batch.response_token_features.dtype)
    ).sum(dim=1)
    response_query = response_query / batch.response_mask.to(
        batch.response_token_features.dtype
    ).sum(dim=1).clamp_min(1.0)
    lr_bag = []
    response_bag = []
    for bag_idx in range(len(batch.sample_ids)):
        mask = batch.bag_index == int(bag_idx)
        lr_bag.append(lr_query[mask].mean() if torch.any(mask) else lr_query.new_tensor(0.0))
        response_bag.append(
            response_query[mask].mean() if torch.any(mask) else response_query.new_tensor(0.0)
        )
    lr_target = torch.stack(lr_bag, dim=0)
    response_target = torch.stack(response_bag, dim=0)
    if lr_target.numel() > 0:
        lr_target = (lr_target - lr_target.min()) / (lr_target.max() - lr_target.min() + 1e-6)
    if response_target.numel() > 0:
        response_target = (response_target - response_target.min()) / (
            response_target.max() - response_target.min() + 1e-6
        )
    return lr_target, response_target


def _criterion(
    model: torch.nn.Module,
    batch: CommunicationBatch,
    output: CommunicationRelayOutput,
    *,
    prior_loss_weight: float,
    response_loss_weight: float,
) -> tuple[Tensor, dict[str, float]]:
    bag_logits = output.bag_logits
    label_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        bag_logits, batch.weak_labels
    )
    loss = label_loss
    aux = {
        "label_loss": float(label_loss.detach().item()),
        "lr_prior_loss": 0.0,
        "response_prior_loss": 0.0,
    }
    if isinstance(model, StageBridgeCommunicationModel):
        lr_target, response_target = _prior_targets(batch)
        prob = torch.sigmoid(bag_logits)
        lr_loss = torch.nn.functional.mse_loss(prob, lr_target)
        response_loss = torch.nn.functional.mse_loss(prob, response_target)
        loss = (
            loss + float(prior_loss_weight) * lr_loss + float(response_loss_weight) * response_loss
        )
        aux["lr_prior_loss"] = float(lr_loss.detach().item())
        aux["response_prior_loss"] = float(response_loss.detach().item())
    return loss, aux


def _predict(
    model: torch.nn.Module,
    bags: list[CommunicationBag],
    *,
    device: torch.device,
    batch_size: int,
    return_attention: bool = False,
) -> dict[str, Any]:
    outputs: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for bag_group in _bag_batches(bags, batch_size=batch_size, seed=0, shuffle=False):
            batch = collate_communication_bags(bag_group).to(str(device))
            forward = model(batch, return_attention=return_attention)
            outputs.append(
                {
                    "batch": batch,
                    "forward": forward,
                    "bags": bag_group,
                }
            )
    return {"batches": outputs}


def _history_frame(history: list[dict[str, Any]], prefix: str) -> pd.DataFrame:
    rows = []
    for item in history:
        row = {"epoch": item["epoch"]}
        for key, value in item.items():
            if key == "epoch":
                continue
            row[f"{prefix}_{key}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def _bag_logits_and_labels(
    prediction_batches: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    logits = np.concatenate(
        [item["forward"].bag_logits.detach().cpu().numpy() for item in prediction_batches], axis=0
    )
    labels = np.concatenate(
        [item["batch"].weak_labels.detach().cpu().numpy() for item in prediction_batches], axis=0
    )
    return logits, labels


def _sample_prediction_frame(
    prediction_batches: list[dict[str, Any]], scaler: TemperatureScaler
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for batch_payload in prediction_batches:
        batch = batch_payload["batch"]
        bag_logits = scaler.apply(batch_payload["forward"].bag_logits.detach().cpu().numpy())
        probs = 1.0 / (1.0 + np.exp(-bag_logits))
        for bag_idx, bag in enumerate(batch_payload["bags"]):
            rows.append(
                {
                    "bag_index": int(bag_idx),
                    "sample_id": str(bag.sample_id),
                    "donor_id": str(bag.donor_id),
                    "edge_label": str(bag.edge_label),
                    "label_source": str(bag.label_source),
                    "label": float(batch.weak_labels[bag_idx].detach().cpu().item()),
                    "bag_logit": float(bag_logits[bag_idx]),
                    "bag_probability": float(probs[bag_idx]),
                    "num_queries": int(len(bag.examples)),
                }
            )
    return pd.DataFrame(rows)


def _edge_metrics(
    sample_predictions: pd.DataFrame, *, threshold: float
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    overall = binary_classification_metrics(
        sample_predictions["bag_probability"].to_numpy(),
        sample_predictions["label"].to_numpy(),
        threshold=threshold,
    )
    by_edge: dict[str, dict[str, float]] = {}
    for edge_label, frame in sample_predictions.groupby("edge_label", sort=True):
        by_edge[str(edge_label)] = binary_classification_metrics(
            frame["bag_probability"].to_numpy(), frame["label"].to_numpy(), threshold=threshold
        )
    return overall, by_edge


def _per_donor_metrics(sample_predictions: pd.DataFrame, *, threshold: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (edge_label, donor_id), frame in sample_predictions.groupby(
        ["edge_label", "donor_id"], sort=True
    ):
        metrics = binary_classification_metrics(
            frame["bag_probability"].to_numpy(), frame["label"].to_numpy(), threshold=threshold
        )
        metrics["edge_label"] = str(edge_label)
        metrics["donor_id"] = str(donor_id)
        metrics["n_samples"] = int(frame.shape[0])
        rows.append(metrics)
    return pd.DataFrame(rows)


def _module_tables(
    prediction_batches: list[dict[str, Any]], scaler: TemperatureScaler
) -> tuple[pd.DataFrame, pd.DataFrame]:
    lr_rows: list[dict[str, Any]] = []
    program_rows: list[dict[str, Any]] = []
    for batch_payload in prediction_batches:
        batch_payload["batch"]
        query_probs = torch.sigmoid(
            torch.tensor(
                scaler.apply(batch_payload["forward"].query_logits.detach().cpu().numpy()),
                dtype=torch.float32,
            )
        ).numpy()
        query_ptr = 0
        for bag in batch_payload["bags"]:
            for example in bag.examples:
                prob = float(query_probs[query_ptr])
                if example.lr_token_names is not None:
                    for idx, name in enumerate(example.lr_token_names):
                        score = (
                            0.0
                            if example.lr_token_features.shape[0] <= idx
                            else float(example.lr_token_features[idx, 2])
                        )
                        lr_rows.append(
                            {
                                "edge_label": bag.edge_label,
                                "sample_id": bag.sample_id,
                                "donor_id": bag.donor_id,
                                "module": str(name),
                                "importance": prob * score,
                                "probability": prob,
                            }
                        )
                if example.response_token_names is not None:
                    for idx, name in enumerate(example.response_token_names):
                        score = (
                            0.0
                            if example.response_token_features.shape[0] <= idx
                            else float(example.response_token_features[idx, 0])
                        )
                        program_rows.append(
                            {
                                "edge_label": bag.edge_label,
                                "sample_id": bag.sample_id,
                                "donor_id": bag.donor_id,
                                "program": str(name),
                                "importance": prob * score,
                                "probability": prob,
                            }
                        )
                query_ptr += 1
    lr_table = (
        (
            pd.DataFrame(lr_rows)
            .groupby(["edge_label", "module"], as_index=False)["importance"]
            .mean()
            .sort_values(["edge_label", "importance"], ascending=[True, False])
        )
        if lr_rows
        else pd.DataFrame(columns=["edge_label", "module", "importance"])
    )
    program_table = (
        (
            pd.DataFrame(program_rows)
            .groupby(["edge_label", "program"], as_index=False)["importance"]
            .mean()
            .sort_values(["edge_label", "importance"], ascending=[True, False])
        )
        if program_rows
        else pd.DataFrame(columns=["edge_label", "program", "importance"])
    )
    return lr_table, program_table


def _write_fold_artifacts(
    artifact_dir: Path,
    *,
    train_history: pd.DataFrame,
    val_history: pd.DataFrame,
    sample_predictions: pd.DataFrame,
    query_predictions: pd.DataFrame,
    metrics: dict[str, Any],
    roc_curve: pd.DataFrame,
    pr_curve: pd.DataFrame,
    calibration_curve: pd.DataFrame,
    per_donor_metrics: pd.DataFrame,
    top_lr_modules: pd.DataFrame,
    top_receiver_programs: pd.DataFrame,
) -> None:
    _ensure_dir(artifact_dir)
    train_history.to_csv(artifact_dir / "train_history.csv", index=False)
    val_history.to_csv(artifact_dir / "val_history.csv", index=False)
    sample_predictions.to_parquet(artifact_dir / "test_predictions.parquet", index=False)
    query_predictions.to_parquet(artifact_dir / "test_query_predictions.parquet", index=False)
    roc_curve.to_csv(artifact_dir / "roc_curve.csv", index=False)
    pr_curve.to_csv(artifact_dir / "pr_curve.csv", index=False)
    calibration_curve.to_csv(artifact_dir / "calibration_curve.csv", index=False)
    per_donor_metrics.to_csv(artifact_dir / "per_donor_metrics.csv", index=False)
    top_lr_modules.to_csv(artifact_dir / "top_lr_modules.csv", index=False)
    top_receiver_programs.to_csv(artifact_dir / "top_receiver_programs.csv", index=False)
    confusion_payload = {
        "tp": metrics["overall"]["tp"],
        "tn": metrics["overall"]["tn"],
        "fp": metrics["overall"]["fp"],
        "fn": metrics["overall"]["fn"],
        "threshold": metrics["overall"]["threshold"],
    }
    (artifact_dir / "confusion_matrix.json").write_text(
        json.dumps(_jsonable(confusion_payload), indent=2), encoding="utf-8"
    )
    (artifact_dir / "metrics.json").write_text(
        json.dumps(_jsonable(metrics), indent=2), encoding="utf-8"
    )


def _query_predictions_frame(
    prediction_batches: list[dict[str, Any]], scaler: TemperatureScaler
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for batch_payload in prediction_batches:
        batch_payload["batch"]
        scaled_logits = scaler.apply(batch_payload["forward"].query_logits.detach().cpu().numpy())
        probs = 1.0 / (1.0 + np.exp(-scaled_logits))
        query_ptr = 0
        for bag_idx, bag in enumerate(batch_payload["bags"]):
            for example in bag.examples:
                rows.append(
                    {
                        "sample_id": bag.sample_id,
                        "donor_id": bag.donor_id,
                        "edge_label": bag.edge_label,
                        "receiver_cell_id": example.receiver_cell_id,
                        "query_probability": float(probs[query_ptr]),
                        "query_logit": float(scaled_logits[query_ptr]),
                        "bag_label": float(bag.weak_label),
                    }
                )
                query_ptr += 1
    return pd.DataFrame(rows)


def _evaluate_predictions(
    prediction_batches: list[dict[str, Any]],
    *,
    scaler: TemperatureScaler,
    threshold: float,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    sample_predictions = _sample_prediction_frame(prediction_batches, scaler)
    query_predictions = _query_predictions_frame(prediction_batches, scaler)
    overall, by_edge = _edge_metrics(sample_predictions, threshold=threshold)
    roc_curve, pr_curve = curve_tables(
        sample_predictions["bag_probability"].to_numpy(), sample_predictions["label"].to_numpy()
    )
    calibration_curve = calibration_curve_table(
        sample_predictions["bag_probability"].to_numpy(), sample_predictions["label"].to_numpy()
    )
    per_donor = _per_donor_metrics(sample_predictions, threshold=threshold)
    top_lr_modules, top_receiver_programs = _module_tables(prediction_batches, scaler)
    metrics = {"overall": overall, "by_edge": by_edge}
    return (
        sample_predictions,
        query_predictions,
        metrics,
        roc_curve,
        pr_curve,
        calibration_curve,
        per_donor,
        top_lr_modules,
        top_receiver_programs,
    )


def _shuffle_context_batch(batch: CommunicationBatch) -> CommunicationBatch:
    perm = torch.randperm(batch.sender_embeddings.shape[0], device=batch.sender_embeddings.device)
    return CommunicationBatch(
        receiver_embedding=batch.receiver_embedding,
        receiver_programs=batch.receiver_programs,
        sender_embeddings=batch.sender_embeddings[perm],
        sender_types=batch.sender_types[perm],
        sender_offsets=batch.sender_offsets[perm],
        ring_ids=batch.ring_ids[perm],
        lr_token_features=batch.lr_token_features[perm],
        response_token_features=batch.response_token_features[perm],
        relay_token_features=batch.relay_token_features[perm],
        query_mask=batch.query_mask,
        sender_mask=batch.sender_mask[perm],
        lr_mask=batch.lr_mask[perm],
        response_mask=batch.response_mask[perm],
        relay_mask=batch.relay_mask[perm],
        edge_ids=batch.edge_ids,
        weak_labels=batch.weak_labels,
        bag_index=batch.bag_index,
        sample_ids=batch.sample_ids,
        donor_ids=batch.donor_ids,
        label_sources=batch.label_sources,
        receiver_cell_ids=batch.receiver_cell_ids,
        wes_features=batch.wes_features,
        target_latent=batch.target_latent,
    )


def _context_shuffle_metrics(
    model: torch.nn.Module,
    bags: list[CommunicationBag],
    *,
    device: torch.device,
    batch_size: int,
    scaler: TemperatureScaler,
    threshold: float,
) -> dict[str, float]:
    real_batches = _predict(
        model, bags, device=device, batch_size=batch_size, return_attention=False
    )["batches"]
    _sample_real, _, metrics_real, _, _, _, _, _, _ = _evaluate_predictions(
        real_batches, scaler=scaler, threshold=threshold
    )
    shuffled_rows: list[pd.DataFrame] = []
    model.eval()
    with torch.no_grad():
        for bag_group in _bag_batches(bags, batch_size=batch_size, seed=0, shuffle=False):
            batch = collate_communication_bags(bag_group).to(str(device))
            shuffled = _shuffle_context_batch(batch)
            output = model(shuffled, return_attention=False)
            frame = _sample_prediction_frame(
                [{"batch": batch, "forward": output, "bags": bag_group}],
                scaler,
            )
            shuffled_rows.append(frame)
    sample_shuffled = (
        pd.concat(shuffled_rows, ignore_index=True) if shuffled_rows else pd.DataFrame()
    )
    metrics_shuffled, _ = _edge_metrics(sample_shuffled, threshold=threshold)
    return {
        "real_auroc": float(metrics_real["overall"]["auroc"]),
        "shuffled_auroc": float(metrics_shuffled["auroc"]),
        "real_auprc": float(metrics_real["overall"]["auprc"]),
        "shuffled_auprc": float(metrics_shuffled["auprc"]),
        "auroc_delta": float(metrics_real["overall"]["auroc"] - metrics_shuffled["auroc"]),
        "auprc_delta": float(metrics_real["overall"]["auprc"] - metrics_shuffled["auprc"]),
    }


def _trial_hparams(
    base_hidden: int, base_dropout: float, base_lr: float, trial_idx: int
) -> dict[str, float]:
    hidden_candidates = [base_hidden, max(32, base_hidden // 2), base_hidden + 32]
    dropout_candidates = [
        base_dropout,
        min(0.3, base_dropout + 0.05),
        max(0.0, base_dropout - 0.03),
    ]
    lr_candidates = [base_lr, base_lr * 0.5, base_lr * 1.5]
    idx = int(trial_idx) % 3
    return {
        "hidden_dim": float(hidden_candidates[idx]),
        "dropout": float(dropout_candidates[idx]),
        "learning_rate": float(lr_candidates[idx]),
    }


def _instantiate_model(
    model_name: str,
    batch: CommunicationBatch,
    cfg: DictConfig,
    trial_params: dict[str, float],
    num_edges: int,
) -> torch.nn.Module:
    return build_communication_model(
        model_name,
        receiver_dim=int(batch.receiver_embedding.shape[1]),
        receiver_program_dim=int(batch.receiver_programs.shape[1]),
        sender_dim=int(batch.sender_embeddings.shape[2]),
        lr_dim=int(batch.lr_token_features.shape[2]),
        response_dim=int(batch.response_token_features.shape[2]),
        relay_dim=int(batch.relay_token_features.shape[2]),
        hidden_dim=int(trial_params["hidden_dim"]),
        num_heads=int(_cfg(cfg, "num_heads", 4)),
        dropout=float(trial_params["dropout"]),
        num_edges=int(num_edges),
        num_sender_types=int(batch.sender_types.max().item() + 2),
        num_ring_ids=int(batch.ring_ids.max().item() + 2),
        wes_dim=0 if batch.wes_features is None else int(batch.wes_features.shape[1]),
    )


def _train_one_trial(
    model_name: str,
    train_bags: list[CommunicationBag],
    val_bags: list[CommunicationBag],
    *,
    cfg: DictConfig,
    device: torch.device,
    trial_params: dict[str, float],
    seed: int,
    num_edges: int,
) -> dict[str, Any]:
    torch.manual_seed(int(seed))
    if not train_bags:
        raise ValueError("Training requires at least one bag.")
    example_batch = collate_communication_bags(train_bags[:1])
    model = _instantiate_model(model_name, example_batch, cfg, trial_params, num_edges).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(trial_params["learning_rate"]),
        weight_decay=float(_cfg(cfg, "weight_decay", 1e-4)),
    )
    batch_size = int(_cfg(cfg, "batch_size_bags", 4))
    max_epochs = int(_cfg(cfg, "max_epochs", 8))
    patience = int(_cfg(cfg, "patience", 3))
    prior_loss_weight = float(_cfg(cfg, "prior_loss_weight", 0.05))
    response_loss_weight = float(_cfg(cfg, "response_loss_weight", 0.05))
    best_metric = (-float("inf"), -float("inf"))
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    patience_left = patience
    train_history: list[dict[str, Any]] = []
    val_history: list[dict[str, Any]] = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses: list[float] = []
        train_lr_losses: list[float] = []
        train_response_losses: list[float] = []
        for bag_group in _bag_batches(
            train_bags, batch_size=batch_size, seed=seed + epoch, shuffle=True
        ):
            batch = collate_communication_bags(bag_group).to(str(device))
            optimizer.zero_grad()
            output = model(batch, return_attention=False)
            loss, aux = _criterion(
                model,
                batch,
                output,
                prior_loss_weight=prior_loss_weight,
                response_loss_weight=response_loss_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(_cfg(cfg, "grad_clip_norm", 1.0))
            )
            optimizer.step()
            train_losses.append(float(loss.detach().item()))
            train_lr_losses.append(float(aux["lr_prior_loss"]))
            train_response_losses.append(float(aux["response_prior_loss"]))
        train_history.append(
            {
                "epoch": epoch,
                "loss": float(np.mean(train_losses)) if train_losses else float("nan"),
                "lr_prior_loss": float(np.mean(train_lr_losses)) if train_lr_losses else 0.0,
                "response_prior_loss": float(np.mean(train_response_losses))
                if train_response_losses
                else 0.0,
            }
        )
        val_pred = _predict(
            model, val_bags, device=device, batch_size=batch_size, return_attention=False
        )["batches"]
        if not val_pred:
            continue
        val_bag_logits, val_labels = _bag_logits_and_labels(val_pred)
        scaler = fit_temperature_scaler(val_bag_logits, val_labels)
        sample_predictions, _, metrics, _, _, _, _, _, _ = _evaluate_predictions(
            val_pred,
            scaler=scaler,
            threshold=choose_threshold(
                1.0 / (1.0 + np.exp(-scaler.apply(val_bag_logits))), val_labels
            ),
        )
        val_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            torch.tensor(sample_predictions["bag_logit"].to_numpy(), dtype=torch.float32),
            torch.tensor(sample_predictions["label"].to_numpy(), dtype=torch.float32),
        )
        val_history.append(
            {
                "epoch": epoch,
                "loss": float(val_loss.item()),
                "auroc": float(metrics["overall"]["auroc"]),
                "auprc": float(metrics["overall"]["auprc"]),
            }
        )
        score = (float(metrics["overall"]["auroc"]), float(metrics["overall"]["auprc"]))
        if score > best_metric:
            best_metric = score
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
    model.load_state_dict(best_state)
    return {
        "model": model,
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "train_history": train_history,
        "val_history": val_history,
    }


def run_communication_benchmark(cfg: DictConfig) -> dict[str, Any]:
    """Train and evaluate communication-relay baselines with donor holdout CV."""
    relay_cfg = _communication_cfg(cfg)
    active_edges = list(_cfg(relay_cfg, "active_edges", ["AAH->AIS", "AIS->MIA"]))
    stages = sorted({stage for label in active_edges for stage in str(label).split("->")})
    snrna = load_luad_evo_snrna_latent(
        cfg,
        stages=stages,
        max_cells_per_stage=int(_cfg(relay_cfg, "max_cells_per_stage", 4096)),
        seed=int(cfg.get("seed", 42)),
    )
    spatial = load_luad_evo_spatial_mapping(
        cfg,
        stages=stages,
        max_spots_per_stage=int(_cfg(relay_cfg, "max_spots_per_stage", 4096)),
        seed=int(cfg.get("seed", 42)),
    )
    wes = load_luad_evo_wes_features(cfg, stages=stages)
    label_manifest_path = Path(
        str(
            _cfg(
                relay_cfg,
                "curated_manifest_path",
                "stagebridge/data/luad_evo/curated_progression_labels.csv",
            )
        )
    )
    bags, bag_table = build_communication_bags(
        snrna,
        spatial,
        wes=wes,
        cfg=cfg,
        active_edges=active_edges,
        curated_manifest_path=label_manifest_path,
        max_receiver_cells_per_sample=int(_cfg(relay_cfg, "max_receiver_cells_per_sample", 16)),
        max_anchor_spots=int(_cfg(relay_cfg, "max_anchor_spots", 4)),
        max_sender_spots=int(_cfg(relay_cfg, "max_sender_spots", 24)),
        max_lr_tokens=int(_cfg(relay_cfg, "max_lr_tokens", 12)),
        num_distance_rings=int(_cfg(relay_cfg, "num_distance_rings", 3)),
        seed=int(cfg.get("seed", 42)),
    )
    if not bags:
        raise RuntimeError("Communication benchmark produced no training bags.")

    donors = sorted({bag.donor_id for bag in bags})
    splits = build_donor_holdout_splits(
        donors,
        n_folds=int(_cfg(relay_cfg, "outer_folds", 3)),
        seed=int(cfg.get("seed", 42)),
    )
    model_families = list(
        _cfg(
            relay_cfg,
            "model_families",
            [
                "focal_only",
                "pooled",
                "deep_sets",
                "graphsage",
                "transformer_no_priors",
                "transformer_no_relay",
                "stagebridge",
            ],
        )
    )
    seeds = [int(item) for item in _cfg(relay_cfg, "seeds", [int(cfg.get("seed", 42))])]
    num_trials = int(_cfg(relay_cfg, "num_trials", 1))
    output_root = _ensure_dir(
        Path(str(cfg.get("output_dir", "outputs/scratch")))
        / str(cfg.get("run_name", "stagebridge_v1"))
        / str(_cfg(relay_cfg, "output_subdir", "communication_relay"))
    )
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        and str(_cfg(relay_cfg, "device", cfg.get("device", "cuda"))).startswith("cuda")
        else "cpu"
    )

    fold_results: list[dict[str, Any]] = []
    for model_name in model_families:
        for fold_idx, split in enumerate(splits):
            train_bags = [bag for bag in bags if bag.donor_id in set(split.train_donors)]
            val_bags = [bag for bag in bags if bag.donor_id in set(split.val_donors)]
            test_bags = [bag for bag in bags if bag.donor_id in set(split.test_donors)]
            if not train_bags or not val_bags or not test_bags:
                continue
            for seed in seeds:
                best_trial: dict[str, Any] | None = None
                for trial_idx in range(num_trials):
                    trial_params = _trial_hparams(
                        int(_cfg(relay_cfg, "hidden_dim", 128)),
                        float(_cfg(relay_cfg, "dropout", 0.1)),
                        float(_cfg(relay_cfg, "learning_rate", 5e-4)),
                        trial_idx,
                    )
                    trained = _train_one_trial(
                        model_name,
                        train_bags,
                        val_bags,
                        cfg=relay_cfg,
                        device=device,
                        trial_params=trial_params,
                        seed=seed + trial_idx,
                        num_edges=max(edge_id_map().values()) + 1,
                    )
                    if best_trial is None or trained["best_metric"] > best_trial["best_metric"]:
                        best_trial = {
                            **trained,
                            "trial_params": trial_params,
                            "trial_idx": trial_idx,
                        }
                assert best_trial is not None
                model = best_trial["model"]
                batch_size = int(_cfg(relay_cfg, "batch_size_bags", 4))
                val_pred = _predict(
                    model, val_bags, device=device, batch_size=batch_size, return_attention=False
                )["batches"]
                val_bag_logits, val_labels = _bag_logits_and_labels(val_pred)
                scaler = fit_temperature_scaler(val_bag_logits, val_labels)
                val_probs = 1.0 / (1.0 + np.exp(-scaler.apply(val_bag_logits)))
                threshold = choose_threshold(val_probs, val_labels)
                test_pred = _predict(
                    model, test_bags, device=device, batch_size=batch_size, return_attention=True
                )["batches"]
                (
                    sample_predictions,
                    query_predictions,
                    metrics,
                    roc_curve,
                    pr_curve,
                    calibration_curve,
                    per_donor,
                    top_lr_modules,
                    top_receiver_programs,
                ) = _evaluate_predictions(
                    test_pred,
                    scaler=scaler,
                    threshold=threshold,
                )
                shuffle_metrics = _context_shuffle_metrics(
                    model,
                    test_bags,
                    device=device,
                    batch_size=batch_size,
                    scaler=scaler,
                    threshold=threshold,
                )
                metrics["context_shuffle"] = shuffle_metrics
                artifact_dir = (
                    output_root / model_name / f"fold_{fold_idx:02d}" / f"seed_{seed:03d}"
                )
                _write_fold_artifacts(
                    artifact_dir,
                    train_history=_history_frame(best_trial["train_history"], "train"),
                    val_history=_history_frame(best_trial["val_history"], "val"),
                    sample_predictions=sample_predictions,
                    query_predictions=query_predictions,
                    metrics=metrics,
                    roc_curve=roc_curve,
                    pr_curve=pr_curve,
                    calibration_curve=calibration_curve,
                    per_donor_metrics=per_donor,
                    top_lr_modules=top_lr_modules,
                    top_receiver_programs=top_receiver_programs,
                )
                fold_results.append(
                    {
                        "model_name": model_name,
                        "fold": int(fold_idx),
                        "seed": int(seed),
                        "artifact_dir": str(artifact_dir),
                        "metrics": metrics,
                        "trial_params": best_trial["trial_params"],
                        "split": {
                            "train_donors": list(split.train_donors),
                            "val_donors": list(split.val_donors),
                            "test_donors": list(split.test_donors),
                        },
                    }
                )

    summary_rows: list[dict[str, Any]] = []
    for row in fold_results:
        summary_rows.append(
            {
                "model_name": row["model_name"],
                "fold": row["fold"],
                "seed": row["seed"],
                "auroc": row["metrics"]["overall"]["auroc"],
                "auprc": row["metrics"]["overall"]["auprc"],
                "balanced_accuracy": row["metrics"]["overall"]["balanced_accuracy"],
                "macro_f1": row["metrics"]["overall"]["macro_f1"],
                "ece": row["metrics"]["overall"]["ece"],
                "context_shuffle_auroc_delta": row["metrics"]["context_shuffle"]["auroc_delta"],
                "context_shuffle_auprc_delta": row["metrics"]["context_shuffle"]["auprc_delta"],
                "artifact_dir": row["artifact_dir"],
            }
        )
    if summary_rows:
        summary_table = (
            pd.DataFrame(summary_rows)
            .sort_values(["model_name", "fold", "seed"])
            .reset_index(drop=True)
        )
    else:
        summary_table = pd.DataFrame(
            columns=[
                "model_name",
                "fold",
                "seed",
                "auroc",
                "auprc",
                "balanced_accuracy",
                "macro_f1",
                "ece",
                "context_shuffle_auroc_delta",
                "context_shuffle_auprc_delta",
                "artifact_dir",
            ]
        )
    summary_path = output_root / "benchmark_summary.csv"
    summary_table.to_csv(summary_path, index=False)
    payload = {
        "ok": True,
        "pipeline": "communication_benchmark",
        "status": "complete",
        "artifact_root": str(output_root),
        "bag_summary": bag_table.to_dict(orient="records"),
        "summary_table": summary_table.to_dict(orient="records"),
        "fold_results": fold_results,
        "summary_path": str(summary_path),
    }
    (output_root / "benchmark_summary.json").write_text(
        json.dumps(_jsonable(payload), indent=2), encoding="utf-8"
    )
    return payload


__all__ = ["run_communication_benchmark"]
