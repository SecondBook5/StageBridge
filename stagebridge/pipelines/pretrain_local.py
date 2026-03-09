"""Local self-supervised pretraining for EA-MIST neighborhood encoders."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from torch import Tensor, nn
from torch.utils.data import DataLoader

from stagebridge.context_model.local_niche_encoder import LocalNicheMLPEncoder, LocalNicheTransformerEncoder
from stagebridge.context_model.losses import (
    masked_feature_reconstruction_loss,
    shuffled_neighborhood_discrimination_loss,
)
from stagebridge.context_model.prototype_bottleneck import PrototypeBottleneck, assignment_entropy_loss, prototype_diversity_loss
from stagebridge.data.luad_evo.bag_dataset import NeighborhoodPretrainDataset, collate_pretrain_neighborhoods
from stagebridge.data.luad_evo.neighborhood_builder import build_lesion_bags_from_config
from stagebridge.logging_utils import get_logger
from stagebridge.utils.seeds import seed_everything

log = get_logger(__name__)


@dataclass(slots=True, frozen=True)
class LocalFeatureDims:
    """Dimension summary for the local niche encoder."""

    receiver_dim: int
    sender_feature_dim: int
    lr_summary_dim: int
    stats_dim: int
    flat_feature_dim: int
    num_receiver_states: int
    num_rings: int


def infer_local_feature_dims(dataset: NeighborhoodPretrainDataset) -> LocalFeatureDims:
    """Infer structured local dimensions from the first neighborhood example."""
    first = dataset[0]
    return LocalFeatureDims(
        receiver_dim=int(first.receiver_embedding.shape[0]),
        sender_feature_dim=int(first.ring_compositions.shape[1]),
        lr_summary_dim=int(first.lr_pathway_summary.shape[0]),
        stats_dim=int(first.neighborhood_stats.shape[0]),
        flat_feature_dim=int(first.flat_features.shape[0]),
        num_receiver_states=max(int(example.receiver_state_id) for example in dataset.examples) + 1,
        num_rings=int(first.ring_compositions.shape[0]),
    )


class LocalSSLPretrainer(nn.Module):
    """Local SSL pretraining wrapper around the EA-MIST niche encoder."""

    def __init__(
        self,
        dims: LocalFeatureDims,
        *,
        encoder_type: str = "transformer",
        hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_prototypes: bool = False,
        num_prototypes: int = 16,
    ) -> None:
        super().__init__()
        self.encoder_type = str(encoder_type)
        if self.encoder_type == "transformer":
            self.encoder: nn.Module = LocalNicheTransformerEncoder(
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
        elif self.encoder_type == "mlp":
            self.encoder = LocalNicheMLPEncoder(
                input_dim=dims.flat_feature_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unsupported local SSL encoder_type '{encoder_type}'.")

        self.prototype_bottleneck = PrototypeBottleneck(hidden_dim, num_prototypes=num_prototypes) if use_prototypes else None
        self.reconstruction_head = nn.Linear(hidden_dim, dims.flat_feature_dim)
        self.shuffle_head = nn.Linear(hidden_dim, 1)

    def encode(self, batch: dict[str, Tensor | list[str]]) -> tuple[Tensor, Tensor | None]:
        """Encode one local SSL batch and optionally return prototype assignments."""
        if self.encoder_type == "transformer":
            output = self.encoder(
                receiver_embeddings=batch["receiver_embeddings"],  # type: ignore[index]
                receiver_state_ids=batch["receiver_state_ids"],  # type: ignore[index]
                ring_compositions=batch["ring_compositions"],  # type: ignore[index]
                lr_pathway_summary=batch["lr_pathway_summary"],  # type: ignore[index]
                neighborhood_stats=batch["neighborhood_stats"],  # type: ignore[index]
                return_attention=False,
            )
            embeddings = output.neighborhood_embedding
        else:
            output = self.encoder(batch["flat_features"])  # type: ignore[arg-type]
            embeddings = output.neighborhood_embedding
        if self.prototype_bottleneck is None:
            return embeddings, None
        prototype_output = self.prototype_bottleneck(embeddings)
        return prototype_output.aligned_embeddings, prototype_output.assignment_weights

    def forward(self, batch: dict[str, Tensor | list[str]], *, mask_probability: float = 0.15) -> dict[str, Tensor]:
        """Run both local SSL tasks and return loss-ready tensors."""
        flat_features = batch["flat_features"]  # type: ignore[index]
        corruption_mask = torch.rand_like(flat_features) < float(mask_probability)
        masked_flat = flat_features.clone()
        masked_flat[corruption_mask] = 0.0

        if self.encoder_type == "transformer":
            ring_compositions = batch["ring_compositions"].clone()  # type: ignore[index]
            lr_summary = batch["lr_pathway_summary"].clone()  # type: ignore[index]
            stats = batch["neighborhood_stats"].clone()  # type: ignore[index]
            ring_mask = torch.rand_like(ring_compositions) < float(mask_probability)
            lr_mask = torch.rand_like(lr_summary) < float(mask_probability)
            stats_mask = torch.rand_like(stats) < float(mask_probability)
            ring_compositions[ring_mask] = 0.0
            lr_summary[lr_mask] = 0.0
            stats[stats_mask] = 0.0
            recon_output = self.encoder(
                receiver_embeddings=batch["receiver_embeddings"],  # type: ignore[index]
                receiver_state_ids=batch["receiver_state_ids"],  # type: ignore[index]
                ring_compositions=ring_compositions,
                lr_pathway_summary=lr_summary,
                neighborhood_stats=stats,
                return_attention=False,
            )
            recon_embeddings = recon_output.neighborhood_embedding
        else:
            recon_output = self.encoder(masked_flat)
            recon_embeddings = recon_output.neighborhood_embedding

        if self.prototype_bottleneck is not None:
            proto_recon = self.prototype_bottleneck(recon_embeddings)
            recon_embeddings = proto_recon.aligned_embeddings
            prototype_assignments = proto_recon.assignment_weights
        else:
            prototype_assignments = None

        reconstructed = self.reconstruction_head(recon_embeddings)

        real_embeddings, _ = self.encode(batch)
        shuffled_batch = dict(batch)
        permutation = torch.randperm(flat_features.shape[0], device=flat_features.device)
        shuffled_batch["ring_compositions"] = batch["ring_compositions"][permutation]  # type: ignore[index]
        shuffled_batch["lr_pathway_summary"] = batch["lr_pathway_summary"][permutation]  # type: ignore[index]
        shuffled_embeddings, _ = self.encode(shuffled_batch)
        discrimination_logits = self.shuffle_head(torch.cat([real_embeddings, shuffled_embeddings], dim=0)).squeeze(-1)
        discrimination_labels = torch.cat(
            [
                torch.ones(real_embeddings.shape[0], device=flat_features.device),
                torch.zeros(shuffled_embeddings.shape[0], device=flat_features.device),
            ],
            dim=0,
        )

        return {
            "reconstructed": reconstructed,
            "target_flat": flat_features,
            "feature_mask": corruption_mask.to(flat_features.dtype),
            "shuffle_logits": discrimination_logits,
            "shuffle_labels": discrimination_labels,
            "prototype_assignments": prototype_assignments,
        }


def _ensure_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _write_history(path: Path, rows: list[dict[str, float | int]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _save_embedding_table(model: LocalSSLPretrainer, dataset: NeighborhoodPretrainDataset, output_dir: Path, device: str) -> Path:
    """Encode and save local neighborhood embeddings for inspection."""
    model.eval()
    loader = DataLoader(dataset, batch_size=256, shuffle=False, collate_fn=collate_pretrain_neighborhoods)
    rows: list[dict[str, object]] = []
    with torch.no_grad():
        for batch in loader:
            tensor_batch = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            embeddings, assignments = model.encode(tensor_batch)
            embeddings_np = embeddings.detach().cpu().numpy()
            assignment_np = None if assignments is None else assignments.detach().cpu().numpy()
            for idx in range(embeddings_np.shape[0]):
                row = {
                    "lesion_id": batch["lesion_ids"][idx],
                    "donor_id": batch["donor_ids"][idx],
                    "stage": batch["stage_labels"][idx],
                }
                for dim_idx, value in enumerate(embeddings_np[idx]):
                    row[f"emb_{dim_idx}"] = float(value)
                if assignment_np is not None:
                    for proto_idx, value in enumerate(assignment_np[idx]):
                        row[f"proto_{proto_idx}"] = float(value)
                rows.append(row)
    path = output_dir / "neighborhood_embeddings.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def run_pretrain_local(cfg: DictConfig | dict[str, Any]) -> dict[str, Any]:
    """Run local SSL pretraining for EA-MIST."""
    seed = int(_cfg_select(cfg, "seed", 42))
    seed_everything(seed)
    build_result = build_lesion_bags_from_config(cfg)
    dataset = NeighborhoodPretrainDataset(build_result.bags)
    dims = infer_local_feature_dims(dataset)

    pretrain_cfg = _cfg_select(cfg, "context_model.eamist.local_pretraining", {}) or {}
    output_root = _ensure_dir(
        Path(str(_cfg_select(cfg, "output_dir", "outputs/scratch")))
        / str(_cfg_select(cfg, "run_name", "stagebridge_v1"))
        / "eamist_pretrain"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LocalSSLPretrainer(
        dims,
        encoder_type=str(pretrain_cfg.get("encoder_type", "transformer")),
        hidden_dim=int(pretrain_cfg.get("hidden_dim", 128)),
        num_heads=int(pretrain_cfg.get("num_heads", 4)),
        dropout=float(pretrain_cfg.get("dropout", 0.1)),
        use_prototypes=bool(pretrain_cfg.get("use_prototypes", False)),
        num_prototypes=int(pretrain_cfg.get("num_prototypes", 16)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(pretrain_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(pretrain_cfg.get("weight_decay", 1e-4)),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(pretrain_cfg.get("batch_size", 128)),
        shuffle=True,
        collate_fn=collate_pretrain_neighborhoods,
    )
    epochs = int(pretrain_cfg.get("max_epochs", 10))
    history_rows: list[dict[str, float | int]] = []
    best_loss = float("inf")
    best_path = output_root / "best_local_encoder.pt"

    for epoch in range(epochs):
        model.train()
        epoch_recon = 0.0
        epoch_shuffle = 0.0
        epoch_proto = 0.0
        epoch_total = 0.0
        num_batches = 0
        for batch in loader:
            tensor_batch = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            outputs = model(tensor_batch, mask_probability=float(pretrain_cfg.get("mask_probability", 0.15)))
            recon_loss = masked_feature_reconstruction_loss(
                outputs["reconstructed"],
                outputs["target_flat"],
                outputs["feature_mask"],
            )
            shuffle_loss = shuffled_neighborhood_discrimination_loss(
                outputs["shuffle_logits"],
                outputs["shuffle_labels"],
            )
            proto_loss = torch.zeros((), device=device)
            if outputs["prototype_assignments"] is not None and model.prototype_bottleneck is not None:
                proto_loss = proto_loss + float(pretrain_cfg.get("prototype_diversity_weight", 0.01)) * prototype_diversity_loss(model.prototype_bottleneck.prototypes)
                proto_loss = proto_loss + float(pretrain_cfg.get("prototype_entropy_weight", 0.001)) * assignment_entropy_loss(outputs["prototype_assignments"])
            loss = recon_loss + shuffle_loss + proto_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(pretrain_cfg.get("grad_clip_norm", 1.0)))
            optimizer.step()

            epoch_recon += float(recon_loss.item())
            epoch_shuffle += float(shuffle_loss.item())
            epoch_proto += float(proto_loss.item())
            epoch_total += float(loss.item())
            num_batches += 1

        row = {
            "epoch": epoch,
            "reconstruction_loss": epoch_recon / max(num_batches, 1),
            "shuffle_loss": epoch_shuffle / max(num_batches, 1),
            "prototype_loss": epoch_proto / max(num_batches, 1),
            "total_loss": epoch_total / max(num_batches, 1),
        }
        history_rows.append(row)
        if row["total_loss"] < best_loss:
            best_loss = float(row["total_loss"])
            torch.save(
                {
                    "state_dict": model.state_dict(),
                        "dims": asdict(dims),
                    "encoder_type": model.encoder_type,
                },
                best_path,
            )

    _write_history(output_root / "train_history.csv", history_rows)
    embeddings_path = _save_embedding_table(model, dataset, output_root, device)
    diagnostics = {
        "num_neighborhoods": int(len(dataset)),
        "encoder_type": model.encoder_type,
        "device": device,
        "best_loss": float(best_loss),
        "feature_dims": asdict(dims),
    }
    if model.prototype_bottleneck is not None:
        with torch.no_grad():
            occupancy = model.prototype_bottleneck.get_prototype_occupancy(
                model.prototype_bottleneck.get_assignment_weights(model.prototype_bottleneck.prototypes)
            )
        diagnostics["prototype_occupancy"] = occupancy.detach().cpu().tolist()

    (output_root / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    return {
        "ok": True,
        "pipeline": "pretrain_local",
        "status": "complete",
        "artifact_root": str(output_root),
        "best_checkpoint": str(best_path),
        "embeddings_path": str(embeddings_path),
        "diagnostics": diagnostics,
    }


def _cfg_select(cfg: DictConfig | dict[str, Any], dotted: str, default: Any) -> Any:
    """Read a dotted config key from OmegaConf or dict payloads."""
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


__all__ = [
    "LocalFeatureDims",
    "LocalSSLPretrainer",
    "infer_local_feature_dims",
    "run_pretrain_local",
]
