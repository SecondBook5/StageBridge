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
from stagebridge.context_model.local_niche_encoder import LocalNicheTransformerEncoder
from stagebridge.context_model.losses import (
    class_weighted_stage_loss,
    displacement_regression_loss,
    masked_edge_loss,
)
from stagebridge.data.luad_evo.bag_dataset import LesionBagDataset, NeighborhoodPretrainDataset, collate_lesion_bags
from stagebridge.data.luad_evo.neighborhood_builder import build_lesion_bags_from_config
from stagebridge.data.luad_evo.splits import (
    assert_no_split_leakage,
    build_multitask_lesion_folds,
    summarize_fold_stage_balance,
)
from stagebridge.evaluation.eamist_metrics import (
    CANONICAL_STAGE_LABELS,
    composite_selection_score,
    compute_displacement_metrics,
    compute_masked_edge_metrics,
    compute_stage_metrics,
    stage_confusion_matrix_payload,
    stage_support_payload,
)
from stagebridge.logging_utils import get_logger
from stagebridge.pipelines.pretrain_local import LocalFeatureDims, infer_local_feature_dims
from stagebridge.utils.seeds import seed_everything
from stagebridge.utils.types import LesionBag, LesionBagBatch

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
    """Collapse validation metrics into one guarded checkpoint-selection objective."""
    return composite_selection_score(metrics)


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
        num_stage_classes: int = 5,
        num_edge_heads: int = 0,
        reference_feature_mode: str = "hlca_luca",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.model_family = str(model_family)
        self.reference_feature_mode = str(reference_feature_mode)
        if self.reference_feature_mode not in {"hlca_luca", "hlca_only", "luca_only"}:
            raise ValueError(f"Unsupported reference_feature_mode '{reference_feature_mode}'.")
        self.local_encoder = LocalNicheTransformerEncoder(
            receiver_dim=dims.receiver_dim,
            sender_feature_dim=dims.sender_feature_dim,
            hlca_dim=dims.hlca_dim,
            luca_dim=dims.luca_dim,
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
            self.aggregator: nn.Module = PooledLesionBaseline(
                hidden_dim,
                hidden_dim=hidden_dim,
                num_stage_classes=num_stage_classes,
                num_edge_heads=num_edge_heads,
                dropout=dropout,
            )
        elif self.model_family == "deep_sets":
            self.aggregator = DeepSetsLesionBaseline(
                hidden_dim,
                hidden_dim=hidden_dim,
                num_stage_classes=num_stage_classes,
                num_edge_heads=num_edge_heads,
                dropout=dropout,
            )
        elif self.model_family == "lesion_set_transformer":
            self.aggregator = LesionSetTransformerBaseline(
                hidden_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                num_stage_classes=num_stage_classes,
                num_edge_heads=num_edge_heads,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unsupported lesion aggregator family '{model_family}'.")

    def _resolve_reference_features(self, batch: LesionBagBatch) -> tuple[Tensor, Tensor]:
        hlca = batch.hlca_features
        luca = batch.luca_features
        if hlca is None:
            hlca = torch.zeros((*batch.receiver_embeddings.shape[:2], 0), dtype=batch.receiver_embeddings.dtype, device=batch.receiver_embeddings.device)
        if luca is None:
            luca = torch.zeros((*batch.receiver_embeddings.shape[:2], 0), dtype=batch.receiver_embeddings.dtype, device=batch.receiver_embeddings.device)
        if self.reference_feature_mode == "hlca_only" and luca.shape[-1] > 0:
            luca = torch.zeros_like(luca)
        if self.reference_feature_mode == "luca_only" and hlca.shape[-1] > 0:
            hlca = torch.zeros_like(hlca)
        return hlca, luca

    def encode_local(self, batch: LesionBagBatch) -> Tensor:
        """Encode local neighborhoods before lesion-level aggregation."""
        bsz, num_instances = batch.receiver_embeddings.shape[:2]
        hlca_features, luca_features = self._resolve_reference_features(batch)
        output = self.local_encoder(
            receiver_embeddings=batch.receiver_embeddings.reshape(bsz * num_instances, -1),
            receiver_state_ids=batch.receiver_state_ids.reshape(bsz * num_instances),
            ring_compositions=batch.ring_compositions.reshape(bsz * num_instances, batch.ring_compositions.shape[2], batch.ring_compositions.shape[3]),
            hlca_features=hlca_features.reshape(bsz * num_instances, -1),
            luca_features=luca_features.reshape(bsz * num_instances, -1),
            lr_pathway_summary=batch.lr_pathway_summary.reshape(bsz * num_instances, -1),
            neighborhood_stats=batch.neighborhood_stats.reshape(bsz * num_instances, -1),
            return_attention=False,
        )
        embeddings = output.neighborhood_embedding.reshape(bsz, num_instances, -1)
        return embeddings * batch.neighborhood_mask.unsqueeze(-1).to(embeddings.dtype)

    def forward(self, batch: LesionBagBatch) -> LesionModelOutput:
        """Run the lesion baseline."""
        embeddings = self.encode_local(batch)
        return self.aggregator(embeddings, batch.neighborhood_mask)


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
    num_edge_heads: int = 0,
    reference_feature_mode: str = "hlca_luca",
) -> nn.Module:
    """Instantiate one lesion-level model family."""
    hidden_dim = int(_cfg_select(cfg, "context_model.eamist.hidden_dim", 128))
    num_heads = int(_cfg_select(cfg, "context_model.eamist.num_heads", 4))
    num_layers = int(_cfg_select(cfg, "context_model.eamist.num_layers", 2))
    dropout = float(_cfg_select(cfg, "context_model.eamist.dropout", 0.1))
    num_stage_classes = len(CANONICAL_STAGE_LABELS)
    if model_family in {"pooled", "deep_sets", "lesion_set_transformer"}:
        return LesionAggregatorModel(
            dims,
            model_family=model_family,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            num_stage_classes=num_stage_classes,
            num_edge_heads=num_edge_heads,
            reference_feature_mode=reference_feature_mode,
            dropout=dropout,
        )
    if model_family in {"eamist", "eamist_no_prototypes"}:
        use_prototypes = bool(_cfg_select(cfg, "context_model.eamist.use_prototypes", True))
        if model_family == "eamist_no_prototypes":
            use_prototypes = False
        return EAMISTModel(
            receiver_dim=dims.receiver_dim,
            sender_feature_dim=dims.sender_feature_dim,
            hlca_dim=dims.hlca_dim,
            luca_dim=dims.luca_dim,
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
            use_prototypes=use_prototypes,
            num_prototypes=int(_cfg_select(cfg, "context_model.eamist.num_prototypes", 16)),
            sparse_assignments=bool(_cfg_select(cfg, "context_model.eamist.sparse_assignments", False)),
            evolution_dim=evolution_dim if bool(_cfg_select(cfg, "context_model.eamist.use_evolution_branch", True)) else None,
            evolution_mode=str(_cfg_select(cfg, "context_model.eamist.evolution_mode", "gated")),
            num_stage_classes=num_stage_classes,
            num_edge_heads=num_edge_heads,
            reference_feature_mode=reference_feature_mode,
        )
    raise ValueError(f"Unsupported model_family '{model_family}'.")


def _compute_stage_class_weights(train_bags: list[LesionBag], *, num_stage_classes: int) -> Tensor:
    counts = np.zeros((num_stage_classes,), dtype=np.float32)
    for bag in train_bags:
        if bag.stage_index is None or int(bag.stage_index) < 0 or int(bag.stage_index) >= num_stage_classes:
            continue
        counts[int(bag.stage_index)] += 1.0
    nonzero = counts > 0
    if not np.any(nonzero):
        return torch.ones((num_stage_classes,), dtype=torch.float32)
    weights = np.zeros_like(counts)
    weights[nonzero] = counts[nonzero].sum() / (counts[nonzero] * float(np.sum(nonzero)))
    weights[~nonzero] = 0.0
    return torch.as_tensor(weights, dtype=torch.float32)


def _stage_probabilities(logits: Tensor) -> Tensor:
    return torch.softmax(logits, dim=-1)


def _run_model(model: nn.Module, batch: LesionBagBatch) -> tuple[LesionModelOutput, Tensor | None]:
    """Run one lesion model and return predictions plus optional prototype regularization."""
    if isinstance(model, EAMISTModel):
        output = model(batch, return_attention=False)
        reg = None
        if output.prototype_output is not None and model.prototype_bottleneck is not None:
            from stagebridge.context_model.prototype_bottleneck import assignment_entropy_loss, prototype_diversity_loss

            reg = (
                float(0.01) * prototype_diversity_loss(model.prototype_bottleneck.prototypes)
                + float(0.001) * assignment_entropy_loss(output.prototype_output.assignment_weights)
            )
        return (
            LesionModelOutput(
                lesion_embedding=output.lesion_embedding,
                stage_logits=output.stage_logits,
                displacement=output.displacement,
                edge_logits=output.edge_logits,
                attention_weights=output.lesion_attention,
            ),
            reg,
        )
    output = model(batch)
    return output, None


def _run_epoch(
    model: nn.Module,
    loader: DataLoader[LesionBagBatch],
    *,
    device: str,
    optimizer: torch.optim.Optimizer | None,
    cfg: DictConfig | dict[str, Any],
    stage_class_weights: Tensor,
) -> dict[str, Any]:
    """Run one train or eval epoch and collect lesion-level outputs."""
    train_mode = optimizer is not None
    model.train(train_mode)
    stage_weight = float(_cfg_select(cfg, "context_model.eamist.stage_loss_weight", 1.0))
    displacement_weight = float(_cfg_select(cfg, "context_model.eamist.displacement_loss_weight", 0.5))
    edge_weight = float(_cfg_select(cfg, "context_model.eamist.edge_loss_weight", 0.25))
    all_stage_logits: list[np.ndarray] = []
    all_stage_preds: list[np.ndarray] = []
    all_stage_targets: list[np.ndarray] = []
    all_displacement_preds: list[np.ndarray] = []
    all_displacement_targets: list[np.ndarray] = []
    all_edge_logits: list[np.ndarray] = []
    all_edge_targets: list[np.ndarray] = []
    all_edge_masks: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []
    all_donors: list[str] = []
    all_stages: list[str] = []
    all_lesions: list[str] = []
    loss_rows: list[dict[str, float]] = []
    for batch in loader:
        batch = batch.to(device)
        output, reg = _run_model(model, batch)
        stage_loss = class_weighted_stage_loss(
            output.stage_logits,
            batch.stage_indices if batch.stage_indices is not None else torch.full((output.stage_logits.shape[0],), -1, dtype=torch.long, device=output.stage_logits.device),
            class_weights=stage_class_weights.to(device),
        )
        displacement_loss = displacement_regression_loss(
            output.displacement,
            batch.displacement_targets if batch.displacement_targets is not None else torch.full((output.displacement.shape[0],), float("nan"), dtype=output.displacement.dtype, device=output.displacement.device),
        )
        edge_loss = masked_edge_loss(output.edge_logits, batch.edge_targets, batch.edge_target_mask)
        if not isinstance(edge_loss, Tensor):
            edge_loss = torch.as_tensor(edge_loss, dtype=output.stage_logits.dtype, device=output.stage_logits.device)
        else:
            edge_loss = edge_loss.to(device=output.stage_logits.device, dtype=output.stage_logits.dtype)
        total_loss = stage_weight * stage_loss + displacement_weight * displacement_loss + edge_weight * edge_loss
        if reg is not None:
            total_loss = total_loss + reg
        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(_cfg_select(cfg, "context_model.eamist.grad_clip_norm", 1.0)))
            optimizer.step()

        loss_rows.append(
            {
                "loss": float(total_loss.item()),
                "stage_loss": float(stage_loss.item()),
                "displacement_loss": float(displacement_loss.item()),
                "edge_loss": float(edge_loss.item()),
                "regularization_loss": float(0.0 if reg is None else reg.item()),
            }
        )
        stage_probs = _stage_probabilities(output.stage_logits).detach().cpu().numpy()
        all_stage_logits.append(output.stage_logits.detach().cpu().numpy())
        all_stage_preds.append(stage_probs.argmax(axis=1).astype(np.int64, copy=False))
        all_probs.append(stage_probs.astype(np.float32, copy=False))
        if batch.stage_indices is None:
            all_stage_targets.append(np.full((stage_probs.shape[0],), -1, dtype=np.int64))
        else:
            all_stage_targets.append(batch.stage_indices.detach().cpu().numpy().astype(np.int64, copy=False))
        if batch.displacement_targets is None:
            all_displacement_targets.append(np.full((stage_probs.shape[0],), np.nan, dtype=np.float32))
        else:
            all_displacement_targets.append(batch.displacement_targets.detach().cpu().numpy().astype(np.float32, copy=False))
        all_displacement_preds.append(output.displacement.detach().cpu().numpy().astype(np.float32, copy=False))
        if output.edge_logits is not None:
            all_edge_logits.append(output.edge_logits.detach().cpu().numpy().astype(np.float32, copy=False))
        if batch.edge_targets is not None:
            all_edge_targets.append(batch.edge_targets.detach().cpu().numpy().astype(np.float32, copy=False))
        if batch.edge_target_mask is not None:
            all_edge_masks.append(batch.edge_target_mask.detach().cpu().numpy().astype(bool, copy=False))
        all_donors.extend(list(batch.donor_ids))
        all_stages.extend(list(batch.stages))
        all_lesions.extend(list(batch.lesion_ids))
    return {
        "loss": float(np.mean([row["loss"] for row in loss_rows])) if loss_rows else float("nan"),
        "stage_loss": float(np.mean([row["stage_loss"] for row in loss_rows])) if loss_rows else float("nan"),
        "displacement_loss": float(np.mean([row["displacement_loss"] for row in loss_rows])) if loss_rows else float("nan"),
        "edge_loss": float(np.mean([row["edge_loss"] for row in loss_rows])) if loss_rows else float("nan"),
        "regularization_loss": float(np.mean([row["regularization_loss"] for row in loss_rows])) if loss_rows else float("nan"),
        "stage_logits": np.concatenate(all_stage_logits, axis=0) if all_stage_logits else np.zeros((0, len(CANONICAL_STAGE_LABELS)), dtype=np.float32),
        "stage_probabilities": np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, len(CANONICAL_STAGE_LABELS)), dtype=np.float32),
        "stage_predictions": np.concatenate(all_stage_preds, axis=0) if all_stage_preds else np.zeros((0,), dtype=np.int64),
        "stage_targets": np.concatenate(all_stage_targets, axis=0) if all_stage_targets else np.zeros((0,), dtype=np.int64),
        "displacement_predictions": np.concatenate(all_displacement_preds, axis=0) if all_displacement_preds else np.zeros((0,), dtype=np.float32),
        "displacement_targets": np.concatenate(all_displacement_targets, axis=0) if all_displacement_targets else np.zeros((0,), dtype=np.float32),
        "edge_logits": np.concatenate(all_edge_logits, axis=0) if all_edge_logits else None,
        "edge_targets": np.concatenate(all_edge_targets, axis=0) if all_edge_targets else None,
        "edge_masks": np.concatenate(all_edge_masks, axis=0) if all_edge_masks else None,
        "donor_ids": all_donors,
        "stages": all_stages,
        "lesion_ids": all_lesions,
    }


def _epoch_metrics(epoch_result: dict[str, Any], *, edge_target_labels: tuple[str, ...]) -> dict[str, float]:
    stage_metrics = compute_stage_metrics(epoch_result["stage_targets"], epoch_result["stage_predictions"])
    displacement_metrics = compute_displacement_metrics(
        epoch_result["displacement_targets"],
        epoch_result["displacement_predictions"],
        stage_indices=epoch_result["stage_targets"],
    )
    edge_metrics = compute_masked_edge_metrics(
        epoch_result["edge_logits"],
        epoch_result["edge_targets"],
        epoch_result["edge_masks"],
        edge_labels=edge_target_labels,
    )
    return {
        **stage_metrics,
        **displacement_metrics,
        **edge_metrics,
    }


def _prediction_frame(
    bags: list[LesionBag],
    indices: list[int] | tuple[int, ...],
    epoch_result: dict[str, Any],
) -> pd.DataFrame:
    """Build a lesion-level stage/displacement prediction table."""
    rows: list[dict[str, object]] = []
    probabilities = epoch_result["stage_probabilities"]
    predictions = epoch_result["stage_predictions"]
    displacement = epoch_result["displacement_predictions"]
    targets = epoch_result["stage_targets"]
    displacement_targets = epoch_result["displacement_targets"]
    for local_idx, bag_index in enumerate(indices):
        bag = bags[int(bag_index)]
        row = {
            "lesion_id": bag.lesion_id,
            "sample_id": bag.sample_id,
            "donor_id": bag.donor_id,
            "patient_id": bag.patient_id,
            "stage_label": bag.stage,
            "stage_index": int(targets[local_idx]),
            "pred_stage_index": int(predictions[local_idx]),
            "pred_stage_label": CANONICAL_STAGE_LABELS[int(predictions[local_idx])],
            "displacement_target": float(displacement_targets[local_idx]),
            "pred_displacement": float(displacement[local_idx]),
            "label_source": bag.label_source,
        }
        for class_idx, class_label in enumerate(CANONICAL_STAGE_LABELS):
            row[f"prob_{class_label}"] = float(probabilities[local_idx, class_idx])
        rows.append(row)
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
    num_edge_heads: int,
    reference_feature_mode: str,
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
            "num_edge_heads": int(num_edge_heads),
            "reference_feature_mode": str(reference_feature_mode),
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


def _fit_trial(
    *,
    cfg: DictConfig | dict[str, Any],
    model_family: str,
    reference_feature_mode: str,
    dims: LocalFeatureDims,
    evolution_dim: int | None,
    num_edge_heads: int,
    edge_target_labels: tuple[str, ...],
    train_bags: list[LesionBag],
    device: str,
    train_loader: DataLoader[LesionBagBatch],
    val_loader: DataLoader[LesionBagBatch],
    trial_root: Path,
    local_mode: str,
    pretrained_checkpoint: str | Path | None,
    optuna_trial: optuna.trial.Trial | None = None,
) -> dict[str, Any]:
    """Train one validation-scored trial and persist its artifacts."""
    model = build_model_family(
        model_family,
        dims,
        cfg=cfg,
        evolution_dim=evolution_dim,
        num_edge_heads=num_edge_heads,
        reference_feature_mode=reference_feature_mode,
    ).to(device)
    if hasattr(model, "local_encoder"):
        load_pretrained_local_encoder(model, pretrained_checkpoint)
        set_local_encoder_trainability(getattr(model, "local_encoder"), local_mode)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(_cfg_select(cfg, "context_model.eamist.learning_rate", 1e-3)),
        weight_decay=float(_cfg_select(cfg, "context_model.eamist.weight_decay", 1e-4)),
    )

    stage_class_weights = _compute_stage_class_weights(train_bags, num_stage_classes=len(CANONICAL_STAGE_LABELS)).to(device)
    max_epochs = int(_cfg_select(cfg, "context_model.eamist.max_epochs", 150))
    patience = int(_cfg_select(cfg, "context_model.eamist.patience", 20))
    train_history: list[dict[str, float | int]] = []
    val_history: list[dict[str, float | int]] = []
    best_payload: dict[str, float] | None = None
    best_score = -np.inf
    best_checkpoint = trial_root / "best_checkpoint.pt"
    wait = 0

    for epoch in range(max_epochs):
        train_epoch = _run_epoch(model, train_loader, device=device, optimizer=optimizer, cfg=cfg, stage_class_weights=stage_class_weights)
        val_epoch = _run_epoch(model, val_loader, device=device, optimizer=None, cfg=cfg, stage_class_weights=stage_class_weights)
        if val_epoch["stage_targets"].shape[0] == 0:
            raise ValueError("Validation split is empty for this lesion-level training trial.")
        val_metrics = _epoch_metrics(val_epoch, edge_target_labels=edge_target_labels)
        train_metrics = _epoch_metrics(train_epoch, edge_target_labels=edge_target_labels)
        val_score = _objective_from_validation_metrics(val_metrics)

        train_history.append(
            {
                "epoch": epoch,
                "loss": float(train_epoch["loss"]),
                "stage_loss": float(train_epoch["stage_loss"]),
                "displacement_loss": float(train_epoch["displacement_loss"]),
                "edge_loss": float(train_epoch["edge_loss"]),
                "regularization_loss": float(train_epoch["regularization_loss"]),
                **train_metrics,
            }
        )
        val_history.append(
            {
                "epoch": epoch,
                "loss": float(val_epoch["loss"]),
                "stage_loss": float(val_epoch["stage_loss"]),
                "displacement_loss": float(val_epoch["displacement_loss"]),
                "edge_loss": float(val_epoch["edge_loss"]),
                "regularization_loss": float(val_epoch["regularization_loss"]),
                **val_metrics,
            }
        )

        if optuna_trial is not None:
            optuna_trial.report(val_score, step=int(epoch))
            if optuna_trial.should_prune():
                raise optuna.TrialPruned()
        if val_score > best_score:
            best_score = float(val_score)
            best_payload = {
                "best_epoch": float(epoch),
                "selection_score": float(val_score),
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
                num_edge_heads=num_edge_heads,
                reference_feature_mode=reference_feature_mode,
            )
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            break

    if best_payload is None:
        raise RuntimeError(f"No valid checkpoint was selected for lesion trial in {trial_root}.")
    pd.DataFrame(train_history).to_csv(trial_root / "train_history.csv", index=False)
    pd.DataFrame(val_history).to_csv(trial_root / "val_history.csv", index=False)
    (trial_root / "trial_metrics.json").write_text(
        json.dumps(
            {
                "best_validation": best_payload,
                "selection_score": float(best_score),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "best_checkpoint": best_checkpoint,
        "best_payload": best_payload,
        "best_score": float(best_score),
        "train_history_path": trial_root / "train_history.csv",
        "val_history_path": trial_root / "val_history.csv",
    }


def _run_optuna_hpo(
    *,
    cfg: DictConfig | dict[str, Any],
    model_family: str,
    reference_feature_mode: str,
    fold_index: int,
    dims: LocalFeatureDims,
    evolution_dim: int | None,
    num_edge_heads: int,
    edge_target_labels: tuple[str, ...],
    train_bags: list[LesionBag],
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
        study_name=f"eamist_{model_family}_{reference_feature_mode}_fold_{fold_index:02d}",
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
            reference_feature_mode=reference_feature_mode,
            dims=dims,
            evolution_dim=evolution_dim,
            num_edge_heads=num_edge_heads,
            edge_target_labels=edge_target_labels,
            train_bags=train_bags,
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
            "Optuna produced no completed trials for model=%s reference_mode=%s fold=%d. Falling back to base config.",
            model_family,
            reference_feature_mode,
            fold_index,
        )
        return {}, trial_table
    best_overrides = dict(study.best_trial.user_attrs.get("overrides", {}) or {})
    return best_overrides, trial_table


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
    model_families = [str(name) for name in _cfg_select(cfg, "context_model.eamist.model_families", ["pooled", "deep_sets", "lesion_set_transformer", "eamist_no_prototypes", "eamist"])]
    reference_feature_modes = [
        str(name)
        for name in _cfg_select(cfg, "context_model.eamist.reference_feature_modes", ["hlca_luca"])
    ]
    seeds = [int(value) for value in _cfg_select(cfg, "context_model.eamist.seeds", [seed])]
    outer_folds = int(_cfg_select(cfg, "context_model.eamist.outer_folds", 3))
    batch_size = int(_cfg_select(cfg, "context_model.eamist.batch_size_bags", 8))
    local_mode = str(_cfg_select(cfg, "context_model.eamist.local_encoder_training_mode", "full"))
    pretrained_checkpoint = _cfg_select(cfg, "context_model.eamist.pretrained_local_checkpoint", None)
    device = _resolve_device(cfg)

    if not build_result.bags:
        raise RuntimeError("EA-MIST lesion training received no lesion bags.")
    dataset = LesionBagDataset(build_result.bags)
    dims = infer_local_feature_dims(NeighborhoodPretrainDataset(build_result.bags))
    evolution_dim = 0
    if any(bag.evolution_features is not None for bag in build_result.bags):
        evolution_dim = max(int(np.asarray(bag.evolution_features, dtype=np.float32).shape[0]) for bag in build_result.bags if bag.evolution_features is not None)
    edge_target_labels = tuple(build_result.bags[0].edge_target_labels or ())
    num_edge_heads = len(edge_target_labels)
    folds = build_multitask_lesion_folds(
        build_result.bags,
        holdout_key=str(_cfg_select(cfg, "context_model.eamist.holdout_key", "donor_id")),
        num_folds=outer_folds,
        seed=seed,
    )
    log.info(
        "Starting EA-MIST lesion training on device=%s with models=%s, reference_modes=%s, folds=%d, seeds=%s, lesions=%d",
        device,
        model_families,
        reference_feature_modes,
        outer_folds,
        seeds,
        len(build_result.bags),
    )

    for model_family in model_families:
        for reference_feature_mode in reference_feature_modes:
            for fold in folds:
                assert_no_split_leakage(build_result.bags, fold)
                fold_root = _ensure_dir(output_root / reference_feature_mode / model_family / f"fold_{fold.fold_index:02d}")
                train_bags = [build_result.bags[idx] for idx in fold.train_indices]
                train_loader = DataLoader(Subset(dataset, list(fold.train_indices)), batch_size=batch_size, shuffle=True, collate_fn=collate_lesion_bags)
                val_loader = DataLoader(Subset(dataset, list(fold.val_indices)), batch_size=batch_size, shuffle=False, collate_fn=collate_lesion_bags)
                test_loader = DataLoader(Subset(dataset, list(fold.test_indices)), batch_size=batch_size, shuffle=False, collate_fn=collate_lesion_bags)

                best_trial_overrides, hpo_trial_table = _run_optuna_hpo(
                    cfg=cfg,
                    model_family=model_family,
                    reference_feature_mode=reference_feature_mode,
                    fold_index=fold.fold_index,
                    dims=dims,
                    evolution_dim=evolution_dim if evolution_dim > 0 else None,
                    num_edge_heads=num_edge_heads,
                    edge_target_labels=edge_target_labels,
                    train_bags=train_bags,
                    device=device,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    fold_root=fold_root,
                    local_mode=local_mode,
                    pretrained_checkpoint=pretrained_checkpoint,
                )
                hpo_trial_table.to_csv(fold_root / "hpo_trial_summary.csv", index=False)
                complete_rows = hpo_trial_table.loc[hpo_trial_table["state"] == "COMPLETE"] if "state" in hpo_trial_table.columns else hpo_trial_table
                if complete_rows.empty or "objective" not in complete_rows.columns:
                    best_trial_idx = 0
                    best_trial_objective = None
                else:
                    best_row = complete_rows.sort_values("objective", ascending=False, na_position="last").iloc[0]
                    best_trial_idx = int(best_row["trial_index"])
                    best_trial_objective = None if pd.isna(best_row["objective"]) else float(best_row["objective"])
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
                        reference_feature_mode=reference_feature_mode,
                        dims=dims,
                        evolution_dim=evolution_dim if evolution_dim > 0 else None,
                        num_edge_heads=num_edge_heads,
                        edge_target_labels=edge_target_labels,
                        train_bags=train_bags,
                        device=device,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        trial_root=run_root,
                        local_mode=str(_cfg_select(run_cfg, "context_model.eamist.local_encoder_training_mode", local_mode)),
                        pretrained_checkpoint=pretrained_checkpoint,
                    )

                    checkpoint = torch.load(fit_result["best_checkpoint"], map_location=device)
                    model = build_model_family(
                        model_family,
                        dims,
                        cfg=run_cfg,
                        evolution_dim=evolution_dim if evolution_dim > 0 else None,
                        num_edge_heads=num_edge_heads,
                        reference_feature_mode=reference_feature_mode,
                    ).to(device)
                    model.load_state_dict(checkpoint["state_dict"], strict=False)
                    model.eval()
                    stage_class_weights = _compute_stage_class_weights(train_bags, num_stage_classes=len(CANONICAL_STAGE_LABELS)).to(device)

                    val_epoch = _run_epoch(model, val_loader, device=device, optimizer=None, cfg=run_cfg, stage_class_weights=stage_class_weights)
                    test_epoch = _run_epoch(model, test_loader, device=device, optimizer=None, cfg=run_cfg, stage_class_weights=stage_class_weights)
                    val_metrics = _epoch_metrics(val_epoch, edge_target_labels=edge_target_labels)
                    test_metrics = _epoch_metrics(test_epoch, edge_target_labels=edge_target_labels)

                    test_frame = _prediction_frame(build_result.bags, fold.test_indices, test_epoch)
                    val_frame = _prediction_frame(build_result.bags, fold.val_indices, val_epoch)
                    auxiliary_edge_metrics = compute_masked_edge_metrics(
                        test_epoch["edge_logits"],
                        test_epoch["edge_targets"],
                        test_epoch["edge_masks"],
                        edge_labels=edge_target_labels,
                    )
                    split_summary = {
                        "fold": fold.summary(),
                        "stage_balance": summarize_fold_stage_balance(build_result.bags, fold),
                        "task_name": "stage_displacement",
                        "model_family": model_family,
                        "reference_feature_mode": reference_feature_mode,
                        "selected_hpo_trial": best_trial_idx,
                    }
                    confusion = stage_confusion_matrix_payload(test_epoch["stage_targets"], test_epoch["stage_predictions"])
                    support = stage_support_payload(test_epoch["stage_targets"])

                    test_frame.to_parquet(run_root / "test_predictions.parquet", index=False)
                    val_frame.to_parquet(run_root / "val_predictions.parquet", index=False)
                    (run_root / "confusion_matrix.json").write_text(json.dumps(confusion, indent=2), encoding="utf-8")
                    (run_root / "metrics.json").write_text(
                        json.dumps(
                            {
                                **test_metrics,
                                "weak_displacement_supervision": True,
                                "support": support,
                            },
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                    (run_root / "auxiliary_edge_metrics.json").write_text(json.dumps(auxiliary_edge_metrics, indent=2), encoding="utf-8")
                    (run_root / "split_summary.json").write_text(json.dumps(split_summary, indent=2), encoding="utf-8")
                    (run_root / "selected_hyperparameters.json").write_text(
                        json.dumps(best_trial_overrides, indent=2),
                        encoding="utf-8",
                    )
                    (run_root / "model_spec.json").write_text(
                        json.dumps(
                            {
                                "model_family": model_family,
                                "task_name": "stage_displacement",
                                "reference_feature_mode": reference_feature_mode,
                                "dims": asdict(dims),
                                "evolution_dim": None if evolution_dim <= 0 else int(evolution_dim),
                                "num_edge_heads": int(num_edge_heads),
                                "edge_target_labels": list(edge_target_labels),
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
                            "task_name": "stage_displacement",
                            "model_family": model_family,
                            "reference_feature_mode": reference_feature_mode,
                            "fold": int(fold.fold_index),
                            "seed": int(run_seed),
                            "selected_hpo_trial": int(best_trial_idx),
                            "selected_hpo_overrides": json.dumps(best_trial_overrides, sort_keys=True),
                            **test_metrics,
                            "artifact_dir": str(run_root),
                        }
                    )

    if not summary_rows:
        raise RuntimeError("EA-MIST lesion training produced no benchmark rows.")
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_root / "benchmark_summary.csv", index=False)
    model_family_summary = (
        summary.groupby(["task_name", "reference_feature_mode", "model_family"], as_index=False)[
            ["stage_macro_f1", "stage_balanced_accuracy", "displacement_mae", "displacement_spearman"]
        ]
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


__all__ = ["run_train_lesion", "build_model_family", "load_pretrained_local_encoder", "_cfg_select"]
