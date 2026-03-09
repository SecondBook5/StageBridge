"""Lesion-level EA-MIST training and benchmarking."""
from __future__ import annotations

import copy
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset

from stagebridge.context_model.baselines_lesion import (
    DeepSetsLesionBaseline,
    LesionModelOutput,
    LesionSetTransformerBaseline,
    PooledLesionBaseline,
)
from stagebridge.context_model.lesion_set_transformer import EAMISTModel
from stagebridge.context_model.local_niche_encoder import LocalNicheMLPEncoder, LocalNicheTransformerEncoder
from stagebridge.context_model.losses import lesion_subsampling_consistency_loss, weighted_binary_classification_loss
from stagebridge.data.luad_evo.bag_dataset import LesionBagDataset, NeighborhoodPretrainDataset, collate_lesion_bags
from stagebridge.data.luad_evo.neighborhood_builder import build_lesion_bags_from_config
from stagebridge.data.luad_evo.splits import (
    assert_no_split_leakage,
    build_lesion_folds,
    summarize_fold_class_balance,
)
from stagebridge.evaluation.eamist_metrics import (
    bootstrap_confidence_intervals,
    build_curve_frames,
    build_per_donor_metrics,
    compute_binary_metrics,
    confusion_matrix_payload,
    temperature_scale_logits,
    threshold_from_validation,
)
from stagebridge.logging_utils import get_logger
from stagebridge.pipelines.pretrain_local import LocalFeatureDims, infer_local_feature_dims
from stagebridge.utils.seeds import seed_everything
from stagebridge.utils.types import LesionBagBatch

log = get_logger(__name__)


def _cfg_select(cfg: DictConfig | dict[str, Any], dotted: str, default: Any) -> Any:
    """Safely read a dotted config path from OmegaConf or dict payloads."""
    if isinstance(cfg, DictConfig):
        current = cfg
        for part in dotted.split("."):
            current = current.get(part, None)
            if current is None:
                return default
        return current
    current: Any = cfg
    for part in dotted.split("."):
        if not isinstance(current, dict):
            return default
        current = current.get(part)
        if current is None:
            return default
    return current


def _ensure_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _resolve_device(cfg: DictConfig | dict[str, Any]) -> str:
    """Resolve the requested training device with clear CUDA diagnostics."""
    requested = str(_cfg_select(cfg, "context_model.eamist.device", "auto")).lower()
    require_cuda = bool(_cfg_select(cfg, "context_model.eamist.require_cuda", False))
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unsupported device setting '{requested}'. Expected one of: auto, cpu, cuda.")
    if requested == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if requested == "cuda" or require_cuda:
        raise RuntimeError("EA-MIST training requested CUDA, but no CUDA device is available.")
    return "cpu"


def _cfg_to_plain_dict(cfg: DictConfig | dict[str, Any]) -> dict[str, Any]:
    """Convert an OmegaConf payload into a mutable plain dictionary."""
    if isinstance(cfg, DictConfig):
        converted = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(converted, dict):
            raise TypeError("Expected training config to resolve to a dictionary.")
        return converted
    return copy.deepcopy(cfg)


def _cfg_with_eamist_overrides(cfg: DictConfig | dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Clone the config and apply overrides only within the EA-MIST config block."""
    cloned = _cfg_to_plain_dict(cfg)
    context_model = cloned.setdefault("context_model", {})
    if not isinstance(context_model, dict):
        raise TypeError("Config field 'context_model' must be a dictionary-like mapping.")
    eamist_cfg = context_model.setdefault("eamist", {})
    if not isinstance(eamist_cfg, dict):
        raise TypeError("Config field 'context_model.eamist' must be a dictionary-like mapping.")
    eamist_cfg.update(copy.deepcopy(overrides))
    return cloned


def _normalize_hpo_search_space(cfg: DictConfig | dict[str, Any], model_family: str) -> dict[str, list[Any]]:
    """Extract the shared and model-family-specific HPO search space."""
    hpo_cfg = _cfg_select(cfg, "context_model.eamist.hpo", {}) or {}
    if isinstance(hpo_cfg, DictConfig):
        converted = OmegaConf.to_container(hpo_cfg, resolve=True)
        hpo_cfg = converted if isinstance(converted, dict) else {}
    if not isinstance(hpo_cfg, dict):
        return {}
    search_space = hpo_cfg.get("search_space", {}) or {}
    if not isinstance(search_space, dict):
        raise TypeError("context_model.eamist.hpo.search_space must be a dictionary.")
    merged: dict[str, list[Any]] = {}
    for key, values in (search_space.get("shared", {}) or {}).items():
        merged[str(key)] = list(values)
    direct_model_space = search_space.get(model_family, {})
    if isinstance(direct_model_space, dict):
        for key, values in direct_model_space.items():
            merged[str(key)] = list(values)
    nested_model_space = (search_space.get("model_families", {}) or {}).get(model_family, {})
    if isinstance(nested_model_space, dict):
        for key, values in nested_model_space.items():
            merged[str(key)] = list(values)
    return merged


def _objective_from_validation_metrics(metrics: dict[str, float]) -> float:
    """Collapse validation metrics into one Optuna objective."""
    auroc = float(metrics.get("auroc", float("nan")))
    auprc = float(metrics.get("auprc", float("nan")))
    if np.isnan(auroc):
        auroc = -1.0
    if np.isnan(auprc):
        auprc = -1.0
    return float(auroc + 1e-3 * auprc)


def _suggest_optuna_overrides(
    trial: optuna.trial.Trial,
    *,
    search_space: dict[str, list[Any]],
) -> dict[str, Any]:
    """Suggest one EA-MIST override dict from the configured search space."""
    overrides: dict[str, Any] = {}
    for key, candidates in search_space.items():
        if not candidates:
            continue
        selected = trial.suggest_categorical(str(key), list(candidates))
        if isinstance(selected, np.generic):
            overrides[str(key)] = selected.item()
        else:
            overrides[str(key)] = selected
    return overrides


def _resolve_hpo_config(cfg: DictConfig | dict[str, Any]) -> dict[str, Any]:
    """Return the EA-MIST HPO config as a plain dictionary."""
    hpo_cfg = _cfg_select(cfg, "context_model.eamist.hpo", {}) or {}
    if isinstance(hpo_cfg, DictConfig):
        converted = OmegaConf.to_container(hpo_cfg, resolve=True)
        hpo_cfg = converted if isinstance(converted, dict) else {}
    if not isinstance(hpo_cfg, dict):
        return {}
    return hpo_cfg


def _build_optuna_trial_table(study: optuna.study.Study) -> pd.DataFrame:
    """Convert an Optuna study into the persisted EA-MIST trial summary table."""
    rows: list[dict[str, Any]] = []
    for trial in study.trials:
        row: dict[str, Any] = {
            "trial_index": int(trial.number),
            "state": str(trial.state),
            "objective": None if trial.value is None else float(trial.value),
            **trial.params,
        }
        for key, value in trial.user_attrs.items():
            if key in {"best_payload", "artifact_dir", "overrides"}:
                if key == "best_payload" and isinstance(value, dict):
                    row.update({f"val_{metric_name}": metric_value for metric_name, metric_value in value.items()})
                else:
                    row[key] = json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["trial_index", "state", "objective"])
    return pd.DataFrame(rows).sort_values(["state", "objective"], ascending=[True, False], na_position="last").reset_index(drop=True)


class LesionAggregatorModel(nn.Module):
    """Shared local encoder plus lesion-level baseline aggregator."""

    def __init__(
        self,
        dims: LocalFeatureDims,
        *,
        model_family: str,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.model_family = str(model_family)
        self.local_encoder = LocalNicheTransformerEncoder(
            receiver_dim=dims.receiver_dim,
            sender_feature_dim=dims.sender_feature_dim,
            lr_summary_dim=dims.lr_summary_dim,
            stats_dim=dims.stats_dim,
            model_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=2,
            num_receiver_states=dims.num_receiver_states + 1,
            num_rings=dims.num_rings,
            dropout=dropout,
        )
        if self.model_family == "pooled":
            self.aggregator: nn.Module = PooledLesionBaseline(hidden_dim, hidden_dim=hidden_dim, dropout=dropout)
        elif self.model_family == "deep_sets":
            self.aggregator = DeepSetsLesionBaseline(hidden_dim, hidden_dim=hidden_dim, dropout=dropout)
        elif self.model_family == "lesion_set_transformer":
            self.aggregator = LesionSetTransformerBaseline(hidden_dim, hidden_dim=hidden_dim, num_heads=num_heads, num_layers=num_layers, dropout=dropout)
        else:
            raise ValueError(f"Unsupported lesion aggregator family '{model_family}'.")

    def encode_local(self, batch: LesionBagBatch) -> Tensor:
        """Encode local neighborhoods before lesion-level aggregation."""
        bsz, num_instances = batch.receiver_embeddings.shape[:2]
        output = self.local_encoder(
            receiver_embeddings=batch.receiver_embeddings.reshape(bsz * num_instances, -1),
            receiver_state_ids=batch.receiver_state_ids.reshape(bsz * num_instances),
            ring_compositions=batch.ring_compositions.reshape(bsz * num_instances, batch.ring_compositions.shape[2], batch.ring_compositions.shape[3]),
            lr_pathway_summary=batch.lr_pathway_summary.reshape(bsz * num_instances, -1),
            neighborhood_stats=batch.neighborhood_stats.reshape(bsz * num_instances, -1),
            return_attention=False,
        )
        embeddings = output.neighborhood_embedding.reshape(bsz, num_instances, -1)
        return embeddings * batch.neighborhood_mask.unsqueeze(-1).to(embeddings.dtype)

    def forward(self, batch: LesionBagBatch) -> LesionModelOutput:
        """Run the lesion baseline."""
        embeddings = self.encode_local(batch)
        return self.aggregator(embeddings, batch.neighborhood_mask, batch.edge_ids)


def set_local_encoder_trainability(module: nn.Module, mode: str) -> None:
    """Apply frozen / partial / full fine-tuning policy to the local encoder."""
    mode = str(mode)
    if mode not in {"frozen", "partial", "full"}:
        raise ValueError(f"Unsupported local encoder trainability mode '{mode}'.")
    for parameter in module.parameters():
        parameter.requires_grad = mode != "frozen"
    if mode != "partial":
        return
    for parameter in module.parameters():
        parameter.requires_grad = False
    if hasattr(module, "blocks"):
        for block in list(getattr(module, "blocks"))[-1:]:
            for parameter in block.parameters():
                parameter.requires_grad = True
    if hasattr(module, "pool"):
        for parameter in getattr(module, "pool").parameters():
            parameter.requires_grad = True
    if hasattr(module, "tokenizer"):
        for parameter in getattr(module, "tokenizer").parameters():
            parameter.requires_grad = True
    if hasattr(module, "net"):
        for parameter in getattr(module, "net").parameters():
            parameter.requires_grad = True


def load_pretrained_local_encoder(model: nn.Module, checkpoint_path: str | Path | None) -> None:
    """Load local encoder weights from a local SSL checkpoint when available."""
    if checkpoint_path is None:
        return
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Requested pretrained encoder checkpoint does not exist: {path}")
    payload = torch.load(path, map_location="cpu")
    state_dict = payload.get("state_dict", payload)
    encoder_state = {
        key.split("encoder.", 1)[1]: value
        for key, value in state_dict.items()
        if key.startswith("encoder.")
    }
    target = getattr(model, "local_encoder", None)
    if target is None:
        raise AttributeError("Target model has no 'local_encoder' attribute for pretrained weight loading.")
    missing, unexpected = target.load_state_dict(encoder_state, strict=False)
    if unexpected:
        log.warning("Ignoring unexpected pretrained local-encoder keys: %s", unexpected)
    if missing:
        log.info("Pretrained local-encoder checkpoint is missing keys: %s", missing)


def build_model_family(
    model_family: str,
    dims: LocalFeatureDims,
    *,
    cfg: DictConfig | dict[str, Any],
    evolution_dim: int | None,
) -> nn.Module:
    """Instantiate one lesion-level model family."""
    hidden_dim = int(_cfg_select(cfg, "context_model.eamist.hidden_dim", 128))
    num_heads = int(_cfg_select(cfg, "context_model.eamist.num_heads", 4))
    num_layers = int(_cfg_select(cfg, "context_model.eamist.num_layers", 2))
    dropout = float(_cfg_select(cfg, "context_model.eamist.dropout", 0.1))
    if model_family in {"pooled", "deep_sets", "lesion_set_transformer"}:
        return LesionAggregatorModel(
            dims,
            model_family=model_family,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )
    if model_family == "eamist":
        return EAMISTModel(
            receiver_dim=dims.receiver_dim,
            sender_feature_dim=dims.sender_feature_dim,
            lr_summary_dim=dims.lr_summary_dim,
            stats_dim=dims.stats_dim,
            flat_feature_dim=dims.flat_feature_dim,
            num_receiver_states=dims.num_receiver_states + 1,
            num_rings=dims.num_rings,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            num_inducing_points=int(_cfg_select(cfg, "context_model.eamist.num_inducing_points", 16)),
            num_pma_seeds=int(_cfg_select(cfg, "context_model.eamist.num_pma_seeds", 1)),
            dropout=dropout,
            local_encoder_type=str(_cfg_select(cfg, "context_model.eamist.local_encoder_type", "transformer")),
            use_prototypes=bool(_cfg_select(cfg, "context_model.eamist.use_prototypes", True)),
            num_prototypes=int(_cfg_select(cfg, "context_model.eamist.num_prototypes", 16)),
            sparse_assignments=bool(_cfg_select(cfg, "context_model.eamist.sparse_assignments", False)),
            evolution_dim=evolution_dim if bool(_cfg_select(cfg, "context_model.eamist.use_evolution_branch", True)) else None,
            evolution_mode=str(_cfg_select(cfg, "context_model.eamist.evolution_mode", "gated")),
        )
    raise ValueError(f"Unsupported model_family '{model_family}'.")


def _fit_trial(
    *,
    cfg: DictConfig | dict[str, Any],
    model_family: str,
    dims: LocalFeatureDims,
    evolution_dim: int | None,
    device: str,
    train_loader: DataLoader[LesionBagBatch],
    val_loader: DataLoader[LesionBagBatch],
    trial_root: Path,
    local_mode: str,
    pretrained_checkpoint: str | Path | None,
    optuna_trial: optuna.trial.Trial | None = None,
) -> dict[str, Any]:
    """Train one validation-scored trial and persist its artifacts."""
    model = build_model_family(model_family, dims, cfg=cfg, evolution_dim=evolution_dim).to(device)
    if hasattr(model, "local_encoder"):
        load_pretrained_local_encoder(model, pretrained_checkpoint)
        set_local_encoder_trainability(getattr(model, "local_encoder"), local_mode)

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(_cfg_select(cfg, "context_model.eamist.learning_rate", 1e-3)),
        weight_decay=float(_cfg_select(cfg, "context_model.eamist.weight_decay", 1e-4)),
    )

    max_epochs = int(_cfg_select(cfg, "context_model.eamist.max_epochs", 150))
    patience = int(_cfg_select(cfg, "context_model.eamist.patience", 20))
    train_history: list[dict[str, float | int]] = []
    val_history: list[dict[str, float | int]] = []
    best_payload: dict[str, float] | None = None
    best_score = (-np.inf, -np.inf)
    best_checkpoint = trial_root / "best_checkpoint.pt"
    wait = 0

    for epoch in range(max_epochs):
        train_epoch = _run_epoch(model, train_loader, device=device, optimizer=optimizer, cfg=cfg)
        val_epoch = _run_epoch(model, val_loader, device=device, optimizer=None, cfg=cfg)
        if val_epoch["labels"].shape[0] == 0:
            raise ValueError("Validation split is empty for this lesion-level training trial.")
        val_temp = temperature_scale_logits(val_epoch["logits"], val_epoch["labels"])
        val_probs = 1.0 / (1.0 + np.exp(-val_epoch["logits"] / val_temp))
        val_threshold = threshold_from_validation(val_epoch["labels"], val_probs)
        val_metrics = compute_binary_metrics(val_epoch["labels"], val_probs, threshold=val_threshold)

        train_history.append({"epoch": epoch, "loss": float(train_epoch["loss"])})
        val_history.append({"epoch": epoch, "loss": float(val_epoch["loss"]), **val_metrics})

        primary_score = float(val_metrics["auroc"]) if not np.isnan(val_metrics["auroc"]) else -float(val_epoch["loss"])
        secondary_score = float(val_metrics["auprc"]) if not np.isnan(val_metrics["auprc"]) else -float(val_epoch["loss"])
        score = (primary_score, secondary_score)
        if optuna_trial is not None:
            optuna_trial.report(_objective_from_validation_metrics(val_metrics), step=int(epoch))
            if optuna_trial.should_prune():
                raise optuna.TrialPruned()
        if score > best_score:
            best_score = score
            best_payload = {
                "temperature": float(val_temp),
                "threshold": float(val_threshold),
                "best_epoch": float(epoch),
                **val_metrics,
            }
            _save_checkpoint(
                best_checkpoint,
                model,
                cfg,
                epoch=epoch,
                val_metrics=best_payload,
                model_family=model_family,
                dims=dims,
                evolution_dim=evolution_dim,
            )
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            break

    if best_payload is None:
        raise RuntimeError(f"No valid checkpoint was selected for lesion trial in {trial_root}.")

    val_history_path = trial_root / "val_history.csv"
    train_history_path = trial_root / "train_history.csv"
    pd.DataFrame(train_history).to_csv(train_history_path, index=False)
    pd.DataFrame(val_history).to_csv(val_history_path, index=False)
    (trial_root / "trial_metrics.json").write_text(
        json.dumps(
            {
                "best_validation": best_payload,
                "selection_score": {
                    "primary": best_score[0],
                    "secondary": best_score[1],
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "best_checkpoint": best_checkpoint,
        "best_payload": best_payload,
        "best_score": best_score,
        "train_history_path": train_history_path,
        "val_history_path": val_history_path,
    }


def _run_optuna_hpo(
    *,
    cfg: DictConfig | dict[str, Any],
    model_family: str,
    fold_index: int,
    dims: LocalFeatureDims,
    evolution_dim: int | None,
    device: str,
    train_loader: DataLoader[LesionBagBatch],
    val_loader: DataLoader[LesionBagBatch],
    fold_root: Path,
    local_mode: str,
    pretrained_checkpoint: str | Path | None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Run Optuna HPO for one model family and fold and return best overrides plus trial table."""
    hpo_cfg = _resolve_hpo_config(cfg)
    backend = str(hpo_cfg.get("backend", "optuna")).lower()
    if backend != "optuna":
        raise ValueError(f"Unsupported EA-MIST HPO backend '{backend}'. Only 'optuna' is supported.")
    enabled = bool(hpo_cfg.get("enabled", False))
    num_trials = max(1, int(hpo_cfg.get("num_trials", 1)))
    if not enabled or num_trials == 1:
        return {}, pd.DataFrame([{"trial_index": 0, "state": "COMPLETE", "objective": None, "overrides": "{}"}])

    search_space = _normalize_hpo_search_space(cfg, model_family)
    if not search_space:
        return {}, pd.DataFrame([{"trial_index": 0, "state": "COMPLETE", "objective": None, "overrides": "{}"}])

    sampler_name = str(hpo_cfg.get("sampler", "tpe")).lower()
    if sampler_name != "tpe":
        raise ValueError(f"Unsupported Optuna sampler '{sampler_name}'. Only 'tpe' is currently supported.")
    sampler = optuna.samplers.TPESampler(seed=int(hpo_cfg.get("seed", 17)) + 1009 * int(fold_index))
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=int(hpo_cfg.get("n_startup_trials", min(3, num_trials))),
        n_warmup_steps=int(hpo_cfg.get("n_warmup_steps", 3)),
    )
    study = optuna.create_study(
        study_name=f"eamist_{model_family}_fold_{fold_index:02d}",
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )

    def objective(trial: optuna.trial.Trial) -> float:
        overrides = _suggest_optuna_overrides(trial, search_space=search_space)
        trial_cfg = _cfg_with_eamist_overrides(cfg, overrides)
        trial_root = _ensure_dir(fold_root / "hpo_trials" / f"trial_{trial.number:03d}")
        fit_result = _fit_trial(
            cfg=trial_cfg,
            model_family=model_family,
            dims=dims,
            evolution_dim=evolution_dim,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            trial_root=trial_root,
            local_mode=str(_cfg_select(trial_cfg, "context_model.eamist.local_encoder_training_mode", local_mode)),
            pretrained_checkpoint=pretrained_checkpoint,
            optuna_trial=trial,
        )
        trial.set_user_attr("overrides", overrides)
        trial.set_user_attr("best_payload", fit_result["best_payload"])
        trial.set_user_attr("artifact_dir", str(trial_root))
        return _objective_from_validation_metrics(fit_result["best_payload"])

    study.optimize(objective, n_trials=num_trials, gc_after_trial=True)
    trial_table = _build_optuna_trial_table(study)
    complete_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if not complete_trials:
        log.warning(
            "Optuna produced no completed trials for model=%s fold=%d. Falling back to base config.",
            model_family,
            fold_index,
        )
        return {}, trial_table
    best_overrides = dict(study.best_trial.user_attrs.get("overrides", {}) or {})
    return best_overrides, trial_table


def _model_forward(model: nn.Module, batch: LesionBagBatch) -> tuple[Tensor, Tensor | None]:
    """Run one lesion model and return selected logits plus optional regularizers."""
    if isinstance(model, EAMISTModel):
        output = model(batch, return_attention=False)
        reg = None
        if output.prototype_output is not None and model.prototype_bottleneck is not None:
            from stagebridge.context_model.prototype_bottleneck import assignment_entropy_loss, prototype_diversity_loss

            reg = (
                float(0.01) * prototype_diversity_loss(model.prototype_bottleneck.prototypes)
                + float(0.001) * assignment_entropy_loss(output.prototype_output.assignment_weights)
            )
        return output.selected_logits, reg
    output = model(batch)
    return output.selected_logits, None


def _run_epoch(
    model: nn.Module,
    loader: DataLoader[LesionBagBatch],
    *,
    device: str,
    optimizer: torch.optim.Optimizer | None,
    cfg: DictConfig | dict[str, Any],
) -> dict[str, Any]:
    """Run one train or eval epoch and collect lesion-level outputs."""
    train_mode = optimizer is not None
    model.train(train_mode)
    all_logits: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_donors: list[str] = []
    losses: list[float] = []
    for batch in loader:
        batch = batch.to(device)
        logits, reg = _model_forward(model, batch)
        loss = weighted_binary_classification_loss(
            logits,
            batch.labels,
            weights=batch.label_weights,
            loss_name=str(_cfg_select(cfg, "context_model.eamist.loss_name", "weighted_bce")),
            focal_gamma=float(_cfg_select(cfg, "context_model.eamist.focal_gamma", 2.0)),
            label_smoothing=float(_cfg_select(cfg, "context_model.eamist.label_smoothing", 0.0)),
        )
        if reg is not None:
            loss = loss + reg
        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(_cfg_select(cfg, "context_model.eamist.grad_clip_norm", 1.0)))
            optimizer.step()
        losses.append(float(loss.item()))
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_logits.append(logits.detach().cpu().numpy())
        all_probs.append(probs)
        all_labels.append(batch.labels.detach().cpu().numpy())
        all_donors.extend(list(batch.donor_ids))
    logits_np = np.concatenate(all_logits, axis=0) if all_logits else np.zeros((0,), dtype=np.float32)
    probs_np = np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0,), dtype=np.float32)
    labels_np = np.concatenate(all_labels, axis=0) if all_labels else np.zeros((0,), dtype=np.float32)
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "logits": logits_np,
        "probabilities": probs_np,
        "labels": labels_np,
        "donor_ids": all_donors,
    }


def _prediction_frame(
    bags: list[Any],
    indices: list[int] | tuple[int, ...],
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    edge_label: str,
) -> pd.DataFrame:
    """Build a lesion-level prediction table."""
    rows: list[dict[str, object]] = []
    for local_idx, bag_index in enumerate(indices):
        bag = bags[int(bag_index)]
        rows.append(
            {
                "lesion_id": bag.lesion_id,
                "sample_id": bag.sample_id,
                "donor_id": bag.donor_id,
                "patient_id": bag.patient_id,
                "stage": bag.stage,
                "edge_label": edge_label,
                "label": float(labels[local_idx]),
                "probability": float(probabilities[local_idx]),
                "label_source": bag.label_source,
                "label_weight": float(bag.label_weight),
            }
        )
    return pd.DataFrame(rows)


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    cfg: DictConfig | dict[str, Any],
    *,
    epoch: int,
    val_metrics: dict[str, float],
    model_family: str,
    dims: LocalFeatureDims,
    evolution_dim: int | None,
) -> None:
    """Persist a best-checkpoint payload."""
    torch.save(
        {
            "state_dict": model.state_dict(),
            "epoch": int(epoch),
            "val_metrics": val_metrics,
            "model_family": str(model_family),
            "dims": asdict(dims),
            "evolution_dim": None if evolution_dim is None else int(evolution_dim),
            "config": cfg if isinstance(cfg, dict) else OmegaConf.to_container(cfg, resolve=True),
        },
        path,
    )


def _export_eamist_interpretability(
    model: EAMISTModel,
    loader: DataLoader[LesionBagBatch],
    *,
    device: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Export prototype composition and lesion attention summaries."""
    model.eval()
    prototype_rows: list[dict[str, object]] = []
    attention_rows: list[dict[str, object]] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(batch, return_attention=True)
            if output.prototype_output is not None:
                composition = output.prototype_output.prototype_composition.detach().cpu().numpy()
                for bag_idx, sample_id in enumerate(batch.sample_ids):
                    for proto_idx, value in enumerate(composition[bag_idx]):
                        prototype_rows.append(
                            {
                                "sample_id": sample_id,
                                "lesion_id": batch.lesion_ids[bag_idx],
                                "donor_id": batch.donor_ids[bag_idx],
                                "stage": batch.stages[bag_idx],
                                "prototype": int(proto_idx),
                                "occupancy": float(value),
                            }
                        )
            if output.lesion_attention is not None:
                attention = output.lesion_attention.detach().cpu().numpy().mean(axis=1).mean(axis=1)
                for bag_idx, sample_id in enumerate(batch.sample_ids):
                    valid = int(batch.neighborhood_mask[bag_idx].sum().item())
                    for niche_idx in range(valid):
                        attention_rows.append(
                            {
                                "sample_id": sample_id,
                                "lesion_id": batch.lesion_ids[bag_idx],
                                "donor_id": batch.donor_ids[bag_idx],
                                "stage": batch.stages[bag_idx],
                                "niche_index": int(niche_idx),
                                "attention_weight": float(attention[bag_idx, niche_idx]),
                            }
                        )
    return pd.DataFrame(prototype_rows), pd.DataFrame(attention_rows)


def run_train_lesion(cfg: DictConfig | dict[str, Any]) -> dict[str, Any]:
    """Run lesion-level benchmarking for pooled, Deep Sets, Set Transformer, and EA-MIST."""
    seed = int(_cfg_select(cfg, "seed", 42))
    seed_everything(seed)
    build_result = build_lesion_bags_from_config(cfg)
    output_root = _ensure_dir(
        Path(str(_cfg_select(cfg, "output_dir", "outputs/scratch")))
        / str(_cfg_select(cfg, "run_name", "stagebridge_v1"))
        / "eamist_benchmark"
    )
    summary_rows: list[dict[str, object]] = []
    active_edges = [str(edge) for edge in _cfg_select(cfg, "context_model.eamist.active_edges", ["AAH->AIS", "AIS->MIA"])]
    model_families = [str(name) for name in _cfg_select(cfg, "context_model.eamist.model_families", ["pooled", "deep_sets", "lesion_set_transformer", "eamist"])]
    seeds = [int(value) for value in _cfg_select(cfg, "context_model.eamist.seeds", [seed])]
    outer_folds = int(_cfg_select(cfg, "context_model.eamist.outer_folds", 3))
    batch_size = int(_cfg_select(cfg, "context_model.eamist.batch_size_bags", 8))
    local_mode = str(_cfg_select(cfg, "context_model.eamist.local_encoder_training_mode", "full"))
    pretrained_checkpoint = _cfg_select(cfg, "context_model.eamist.pretrained_local_checkpoint", None)
    device = _resolve_device(cfg)
    log.info(
        "Starting EA-MIST lesion training on device=%s with edges=%s, models=%s, folds=%d, seeds=%s",
        device,
        active_edges,
        model_families,
        outer_folds,
        seeds,
    )

    for edge_label in active_edges:
        edge_bags = [bag for bag in build_result.bags if bag.edge_label == edge_label]
        if len(edge_bags) < 3:
            log.warning("Skipping edge %s because only %d lesion bags are available.", edge_label, len(edge_bags))
            continue
        dataset = LesionBagDataset(edge_bags)
        dims = infer_local_feature_dims(NeighborhoodPretrainDataset(edge_bags))
        evolution_dim = 0
        if any(bag.evolution_features is not None for bag in edge_bags):
            evolution_dim = max(int(np.asarray(bag.evolution_features, dtype=np.float32).shape[0]) for bag in edge_bags if bag.evolution_features is not None)
        try:
            folds = build_lesion_folds(
                edge_bags,
                holdout_key=str(_cfg_select(cfg, "context_model.eamist.holdout_key", "donor_id")),
                num_folds=outer_folds,
                seed=seed,
                min_lesions_per_class=int(_cfg_select(cfg, "context_model.eamist.min_lesions_per_class", 1)),
            )
        except ValueError as exc:
            log.warning(
                "Skipping edge %s because donor-held-out folds are not statistically valid: %s",
                edge_label,
                exc,
            )
            continue

        for model_family in model_families:
            for fold in folds:
                assert_no_split_leakage(edge_bags, fold)
                fold_root = _ensure_dir(output_root / edge_label.replace("->", "_") / model_family / f"fold_{fold.fold_index:02d}")
                train_loader = DataLoader(Subset(dataset, list(fold.train_indices)), batch_size=batch_size, shuffle=True, collate_fn=collate_lesion_bags)
                val_loader = DataLoader(Subset(dataset, list(fold.val_indices)), batch_size=batch_size, shuffle=False, collate_fn=collate_lesion_bags)
                test_loader = DataLoader(Subset(dataset, list(fold.test_indices)), batch_size=batch_size, shuffle=False, collate_fn=collate_lesion_bags)

                log.info(
                    "Optuna HPO selection for edge=%s model=%s fold=%d.",
                    edge_label,
                    model_family,
                    fold.fold_index,
                )
                best_trial_overrides, hpo_trial_table = _run_optuna_hpo(
                    cfg=cfg,
                    model_family=model_family,
                    fold_index=fold.fold_index,
                    dims=dims,
                    evolution_dim=evolution_dim if evolution_dim > 0 else None,
                    device=device,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    fold_root=fold_root,
                    local_mode=local_mode,
                    pretrained_checkpoint=pretrained_checkpoint,
                )
                hpo_trial_table.to_csv(fold_root / "hpo_trial_summary.csv", index=False)
                if not hpo_trial_table.empty and "trial_index" in hpo_trial_table.columns and "objective" in hpo_trial_table.columns:
                    complete_rows = hpo_trial_table.loc[hpo_trial_table["state"] == "COMPLETE"] if "state" in hpo_trial_table.columns else hpo_trial_table
                    if complete_rows.empty:
                        best_trial_idx = 0
                        best_trial_objective = None
                    else:
                        best_row = complete_rows.sort_values("objective", ascending=False, na_position="last").iloc[0]
                        best_trial_idx = int(best_row["trial_index"])
                        best_trial_objective = None if pd.isna(best_row["objective"]) else float(best_row["objective"])
                else:
                    best_trial_idx = 0
                    best_trial_objective = None
                (fold_root / "best_hpo_config.json").write_text(
                    json.dumps(
                        {
                            "trial_index": best_trial_idx,
                            "overrides": best_trial_overrides,
                            "selection_objective": best_trial_objective,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )

                for run_seed in seeds:
                    seed_everything(run_seed)
                    run_cfg = _cfg_with_eamist_overrides(cfg, best_trial_overrides)
                    run_root = _ensure_dir(fold_root / f"seed_{run_seed:03d}")
                    fit_result = _fit_trial(
                        cfg=run_cfg,
                        model_family=model_family,
                        dims=dims,
                        evolution_dim=evolution_dim if evolution_dim > 0 else None,
                        device=device,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        trial_root=run_root,
                        local_mode=str(_cfg_select(run_cfg, "context_model.eamist.local_encoder_training_mode", local_mode)),
                        pretrained_checkpoint=pretrained_checkpoint,
                    )

                    checkpoint = torch.load(fit_result["best_checkpoint"], map_location=device)
                    model = build_model_family(model_family, dims, cfg=run_cfg, evolution_dim=evolution_dim if evolution_dim > 0 else None).to(device)
                    model.load_state_dict(checkpoint["state_dict"], strict=False)
                    model.eval()

                    val_epoch = _run_epoch(model, val_loader, device=device, optimizer=None, cfg=run_cfg)
                    test_epoch = _run_epoch(model, test_loader, device=device, optimizer=None, cfg=run_cfg)
                    val_probs = 1.0 / (1.0 + np.exp(-val_epoch["logits"] / float(fit_result["best_payload"]["temperature"])))
                    test_probs = 1.0 / (1.0 + np.exp(-test_epoch["logits"] / float(fit_result["best_payload"]["temperature"])))
                    test_metrics = compute_binary_metrics(test_epoch["labels"], test_probs, threshold=float(fit_result["best_payload"]["threshold"]))
                    intervals = bootstrap_confidence_intervals(test_epoch["labels"], test_probs, seed=run_seed)

                    test_frame = _prediction_frame(edge_bags, fold.test_indices, test_probs, test_epoch["labels"], edge_label=edge_label)
                    val_frame = _prediction_frame(edge_bags, fold.val_indices, val_probs, val_epoch["labels"], edge_label=edge_label)
                    per_donor = build_per_donor_metrics(test_frame, threshold=float(fit_result["best_payload"]["threshold"]))
                    roc_df, pr_df, cal_df = build_curve_frames(test_epoch["labels"], test_probs)
                    confusion = confusion_matrix_payload(test_epoch["labels"], test_probs, threshold=float(fit_result["best_payload"]["threshold"]))
                    split_summary = {
                        "fold": fold.summary(),
                        "class_balance": summarize_fold_class_balance(edge_bags, fold),
                        "edge_label": edge_label,
                        "model_family": model_family,
                        "selected_hpo_trial": best_trial_idx,
                    }

                    test_frame.to_parquet(run_root / "test_predictions.parquet", index=False)
                    val_frame.to_parquet(run_root / "val_predictions.parquet", index=False)
                    roc_df.to_csv(run_root / "roc_curve.csv", index=False)
                    pr_df.to_csv(run_root / "pr_curve.csv", index=False)
                    cal_df.to_csv(run_root / "calibration_curve.csv", index=False)
                    per_donor.to_csv(run_root / "per_donor_metrics.csv", index=False)
                    (run_root / "confusion_matrix.json").write_text(json.dumps(confusion, indent=2), encoding="utf-8")
                    (run_root / "metrics.json").write_text(
                        json.dumps({**test_metrics, **fit_result["best_payload"], **intervals}, indent=2),
                        encoding="utf-8",
                    )
                    (run_root / "split_summary.json").write_text(json.dumps(split_summary, indent=2), encoding="utf-8")
                    (run_root / "selected_hyperparameters.json").write_text(
                        json.dumps(best_trial_overrides, indent=2),
                        encoding="utf-8",
                    )
                    (run_root / "model_spec.json").write_text(
                        json.dumps(
                            {
                                "model_family": model_family,
                                "edge_label": edge_label,
                                "dims": asdict(dims),
                                "evolution_dim": None if evolution_dim <= 0 else int(evolution_dim),
                            },
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                    if isinstance(model, EAMISTModel):
                        prototype_frame, attention_frame = _export_eamist_interpretability(model, test_loader, device=device)
                        if not prototype_frame.empty:
                            prototype_frame.to_parquet(run_root / "prototype_composition.parquet", index=False)
                        if not attention_frame.empty:
                            attention_frame.to_parquet(run_root / "lesion_attention.parquet", index=False)

                    summary_rows.append(
                        {
                            "edge_label": edge_label,
                            "model_family": model_family,
                            "fold": int(fold.fold_index),
                            "seed": int(run_seed),
                            "selected_hpo_trial": int(best_trial_idx),
                            "selected_hpo_overrides": json.dumps(best_trial_overrides, sort_keys=True),
                            **test_metrics,
                            "auroc_ci_low": intervals["auroc_ci"][0],
                            "auroc_ci_high": intervals["auroc_ci"][1],
                            "auprc_ci_low": intervals["auprc_ci"][0],
                            "auprc_ci_high": intervals["auprc_ci"][1],
                            "artifact_dir": str(run_root),
                        }
                    )

    if not summary_rows:
        raise RuntimeError("EA-MIST lesion training produced no benchmark rows.")
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_root / "benchmark_summary.csv", index=False)
    model_family_summary = (
        summary.groupby(["edge_label", "model_family"], as_index=False)[["auroc", "auprc", "balanced_accuracy", "f1", "brier", "ece"]]
        .agg(["mean", "std"])
    )
    model_family_summary.columns = [
        "_".join([part for part in column if part]).strip("_") if isinstance(column, tuple) else str(column)
        for column in model_family_summary.columns
    ]
    model_family_summary.to_csv(output_root / "model_family_summary.csv", index=False)

    return {
        "ok": True,
        "pipeline": "train_lesion",
        "status": "complete",
        "artifact_root": str(output_root),
        "benchmark_summary": str(output_root / "benchmark_summary.csv"),
        "model_family_summary": str(output_root / "model_family_summary.csv"),
    }


__all__ = ["run_train_lesion", "build_model_family", "load_pretrained_local_encoder"]
