"""Self-supervised relational pretraining for the hierarchical transformer."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from stagebridge.context_model.set_encoder import SetContextSummary


@dataclass(slots=True, frozen=True)
class RelationalPretrainingConfig:
    """Configuration for the self-supervised pretraining stage."""

    mask_fraction: float = 0.15
    masked_token_weight: float = 0.35
    ranking_weight: float = 0.35
    provider_consistency_weight: float = 0.15
    coordinate_corruption_weight: float = 0.10
    group_relation_weight: float = 0.05
    ranking_margin: float = 0.2
    max_epochs: int = 3
    steps_per_epoch: int = 4
    learning_rate: float = 8e-4
    weight_decay: float = 1e-4
    seed: int = 42


class RelationalPretrainingHeads(nn.Module):
    """Heads used during transformer pretraining and reduced fine-tuning."""

    def __init__(
        self,
        *,
        context_dim: int,
        token_dim: int,
        num_token_types: int,
        num_datasets: int = 4,
        num_edges: int = 8,
        projection_dim: int = 64,
    ) -> None:
        super().__init__()
        aux_dim = max(16, context_dim // 4)
        self.token_type_embedding = nn.Embedding(int(num_token_types), aux_dim)
        self.dataset_embedding = nn.Embedding(int(num_datasets), aux_dim)
        self.edge_embedding = nn.Embedding(int(num_edges), aux_dim)
        self.coord_projection = nn.Sequential(
            nn.Linear(2, aux_dim),
            nn.GELU(),
            nn.LayerNorm(aux_dim),
        )
        self.confidence_projection = nn.Sequential(
            nn.Linear(1, aux_dim),
            nn.GELU(),
            nn.LayerNorm(aux_dim),
        )
        decoder_in = int(context_dim) + aux_dim * 5
        self.masked_decoder = nn.Sequential(
            nn.Linear(decoder_in, int(context_dim)),
            nn.GELU(),
            nn.LayerNorm(int(context_dim)),
            nn.Linear(int(context_dim), int(token_dim)),
        )
        self.ranking_head = nn.Sequential(
            nn.Linear(int(context_dim), int(context_dim)),
            nn.GELU(),
            nn.LayerNorm(int(context_dim)),
            nn.Linear(int(context_dim), 1),
        )
        self.provider_projector = nn.Sequential(
            nn.Linear(int(context_dim), int(context_dim)),
            nn.GELU(),
            nn.Linear(int(context_dim), int(projection_dim)),
        )
        self.coord_corruption_head = nn.Sequential(
            nn.Linear(int(context_dim), max(16, int(context_dim) // 2)),
            nn.GELU(),
            nn.Linear(max(16, int(context_dim) // 2), 1),
        )
        self.group_relation_head = nn.Sequential(
            nn.Linear(int(context_dim) * 2, int(context_dim)),
            nn.GELU(),
            nn.LayerNorm(int(context_dim)),
            nn.Linear(int(context_dim), 1),
        )


def _clone_trainable_parameters(module: nn.Module) -> dict[str, Tensor]:
    return {
        name: param.detach().clone()
        for name, param in module.named_parameters()
        if param.requires_grad
    }


def _mean_parameter_delta(before: Mapping[str, Tensor], module: nn.Module) -> float:
    deltas: list[float] = []
    for name, param in module.named_parameters():
        if not param.requires_grad or name not in before:
            continue
        deltas.append(float(torch.mean(torch.abs(param.detach() - before[name])).item()))
    if not deltas:
        return 0.0
    return float(np.mean(deltas))


def _forward_context_encoder(
    context_encoder: nn.Module,
    context_tokens: Tensor,
    *,
    token_type_ids: Tensor | None,
    token_coords: Tensor | None,
    token_confidence: Tensor | None,
    dataset_ids: Tensor | None,
    edge_ids: Tensor | None,
    return_attention: bool = False,
) -> SetContextSummary:
    kwargs: dict[str, Any] = {
        "token_type_ids": token_type_ids,
        "token_coords": token_coords,
        "token_confidence": token_confidence,
        "dataset_ids": dataset_ids,
        "edge_ids": edge_ids,
    }
    if return_attention:
        kwargs["return_attention"] = True
    try:
        return context_encoder(context_tokens, **kwargs)
    except TypeError:
        reduced = dict(kwargs)
        reduced.pop("dataset_ids", None)
        reduced.pop("edge_ids", None)
        if return_attention:
            reduced["return_attention"] = True
        return context_encoder(context_tokens, **reduced)


def stratified_mask_token_indices(
    token_type_ids: Tensor,
    *,
    mask_fraction: float,
    seed: int,
) -> Tensor:
    """Choose masked token indices while preserving biological groups when possible."""
    if token_type_ids.ndim != 1:
        raise ValueError(f"token_type_ids must be 1D, got shape {tuple(token_type_ids.shape)}.")
    rng = np.random.default_rng(int(seed))
    n_tokens = int(token_type_ids.shape[0])
    target = max(1, int(round(float(mask_fraction) * n_tokens)))
    selected: set[int] = set()
    type_np = token_type_ids.detach().cpu().numpy().astype(np.int64, copy=False)
    for group_idx in sorted(set(type_np.tolist())):
        group_rows = np.flatnonzero(type_np == int(group_idx))
        if group_rows.size == 0:
            continue
        selected.add(int(rng.choice(group_rows)))
        if len(selected) >= target:
            break
    remaining = [idx for idx in range(n_tokens) if idx not in selected]
    rng.shuffle(remaining)
    for idx in remaining:
        if len(selected) >= target:
            break
        selected.add(int(idx))
    return torch.tensor(sorted(selected), dtype=torch.long, device=token_type_ids.device)


def _build_masked_view(
    *,
    context_tokens: Tensor,
    token_type_ids: Tensor,
    token_coords: Tensor | None,
    token_confidence: Tensor | None,
    mask_fraction: float,
    seed: int,
) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None, Tensor]:
    mask_idx = stratified_mask_token_indices(token_type_ids, mask_fraction=mask_fraction, seed=seed)
    masked_tokens = context_tokens.clone()
    masked_tokens.index_fill_(0, mask_idx, 0.0)
    masked_confidence = None
    if token_confidence is not None:
        masked_confidence = token_confidence.clone()
        masked_confidence.index_fill_(0, mask_idx, 0.0)
    masked_coords = None if token_coords is None else token_coords.clone()
    return masked_tokens, token_type_ids, masked_coords, masked_confidence, mask_idx


def _ensure_2d(x: Tensor) -> Tensor:
    return x.unsqueeze(0) if x.ndim == 1 else x


def _group_means(summary: SetContextSummary, *, num_groups: int) -> Tensor | None:
    group_tokens = summary.group_summary_tokens
    if group_tokens is None:
        return None
    if group_tokens.ndim == 2:
        group_tokens = group_tokens.unsqueeze(0)
    per_group = int(group_tokens.shape[1] // max(num_groups, 1))
    if per_group <= 0:
        return None
    reshaped = group_tokens[:, : per_group * num_groups, :].reshape(group_tokens.shape[0], num_groups, per_group, group_tokens.shape[-1])
    return reshaped.mean(dim=2).squeeze(0)


def _coordinate_corruption(coords: Tensor, *, seed: int) -> Tensor:
    rng = np.random.default_rng(int(seed))
    perm = torch.tensor(rng.permutation(coords.shape[0]), dtype=torch.long, device=coords.device)
    corrupted = coords.index_select(0, perm)
    if coords.shape[0] > 1:
        corrupted = corrupted + 0.05 * torch.randn_like(corrupted)
    return corrupted


def compute_relational_auxiliary_losses(
    *,
    context_encoder: nn.Module,
    heads: RelationalPretrainingHeads,
    context_tokens: Tensor,
    token_type_ids: Tensor,
    token_coords: Tensor | None,
    token_confidence: Tensor | None,
    dataset_ids: Tensor | None,
    edge_ids: Tensor | None,
    negative_controls: list[dict[str, Any]] | None,
    provider_views: list[dict[str, Any]] | None,
    config: RelationalPretrainingConfig,
    seed: int,
    include_masked_token: bool,
    include_provider_consistency: bool,
    include_coordinate_corruption: bool,
    include_group_relation: bool,
    return_attention: bool = False,
) -> tuple[Tensor, dict[str, Tensor], dict[str, Any], SetContextSummary]:
    summary = _forward_context_encoder(
        context_encoder,
        context_tokens,
        token_type_ids=token_type_ids,
        token_coords=token_coords,
        token_confidence=token_confidence,
        dataset_ids=dataset_ids,
        edge_ids=edge_ids,
        return_attention=return_attention,
    )
    pooled = _ensure_2d(summary.pooled_context)

    losses: dict[str, Tensor] = {}
    metrics: dict[str, Any] = {}
    device = context_tokens.device
    dtype = context_tokens.dtype

    if include_masked_token and context_tokens.shape[0] > 1:
        masked_tokens, masked_type_ids, masked_coords, masked_confidence, mask_idx = _build_masked_view(
            context_tokens=context_tokens,
            token_type_ids=token_type_ids,
            token_coords=token_coords,
            token_confidence=token_confidence,
            mask_fraction=config.mask_fraction,
            seed=seed,
        )
        masked_summary = _forward_context_encoder(
            context_encoder,
            masked_tokens,
            token_type_ids=masked_type_ids,
            token_coords=masked_coords,
            token_confidence=masked_confidence,
            dataset_ids=dataset_ids,
            edge_ids=edge_ids,
            return_attention=False,
        )
        masked_context = _ensure_2d(masked_summary.pooled_context).expand(mask_idx.shape[0], -1)
        decoder_parts = [
            masked_context,
            heads.token_type_embedding(token_type_ids.index_select(0, mask_idx)),
            heads.coord_projection(token_coords.index_select(0, mask_idx)) if token_coords is not None else torch.zeros(mask_idx.shape[0], heads.token_type_embedding.embedding_dim, device=device, dtype=dtype),
            heads.confidence_projection(token_confidence.index_select(0, mask_idx).unsqueeze(-1)) if token_confidence is not None else torch.zeros(mask_idx.shape[0], heads.token_type_embedding.embedding_dim, device=device, dtype=dtype),
            heads.dataset_embedding(dataset_ids[:1].long()).expand(mask_idx.shape[0], -1) if dataset_ids is not None else torch.zeros(mask_idx.shape[0], heads.token_type_embedding.embedding_dim, device=device, dtype=dtype),
            heads.edge_embedding(edge_ids[:1].long()).expand(mask_idx.shape[0], -1) if edge_ids is not None else torch.zeros(mask_idx.shape[0], heads.token_type_embedding.embedding_dim, device=device, dtype=dtype),
        ]
        masked_pred = heads.masked_decoder(torch.cat(decoder_parts, dim=-1))
        masked_target = context_tokens.index_select(0, mask_idx)
        losses["masked_token"] = F.mse_loss(masked_pred, masked_target)
        metrics["masked_token_count"] = int(mask_idx.shape[0])
    else:
        losses["masked_token"] = torch.zeros((), device=device, dtype=dtype)
        metrics["masked_token_count"] = 0

    negative_controls = list(negative_controls or [])
    negative_summaries: list[tuple[str, SetContextSummary]] = []
    for idx, control in enumerate(negative_controls):
        label = str(control.get("label", f"negative_{idx}"))
        negative_summary = _forward_context_encoder(
            context_encoder,
            control["tokens"],
            token_type_ids=control.get("token_type_ids"),
            token_coords=control.get("coords"),
            token_confidence=control.get("confidence"),
            dataset_ids=control.get("dataset_ids"),
            edge_ids=edge_ids,
            return_attention=False,
        )
        negative_summaries.append((label, negative_summary))
    positive_score = heads.ranking_head(pooled)
    if negative_summaries:
        negative_for_head = torch.cat([_ensure_2d(summary_item.pooled_context) for _, summary_item in negative_summaries], dim=0)
        negative_scores = heads.ranking_head(negative_for_head)
        margin = torch.tensor(float(config.ranking_margin), device=device, dtype=dtype)
        losses["ranking"] = torch.relu(margin - positive_score + negative_scores).mean()
        metrics["ranking_accuracy"] = float((positive_score.detach() > negative_scores.detach()).float().mean().item())
        metrics["negative_control_scores"] = {
            label: float(negative_scores[idx].mean().item())
            for idx, (label, _) in enumerate(negative_summaries)
        }
        metrics["positive_score"] = float(positive_score.mean().item())
    else:
        losses["ranking"] = torch.zeros((), device=device, dtype=dtype)
        metrics["ranking_accuracy"] = float("nan")
        metrics["negative_control_scores"] = {}
        metrics["positive_score"] = float(positive_score.mean().item())

    provider_views = list(provider_views or [])
    if include_provider_consistency and provider_views:
        anchor_proj = F.normalize(heads.provider_projector(pooled), dim=-1)
        provider_losses: list[Tensor] = []
        provider_cosines: list[float] = []
        for view in provider_views:
            alt_summary = _forward_context_encoder(
                context_encoder,
                view["tokens"],
                token_type_ids=view.get("token_type_ids"),
                token_coords=view.get("coords"),
                token_confidence=view.get("confidence"),
                dataset_ids=view.get("dataset_ids"),
                edge_ids=edge_ids,
                return_attention=False,
            )
            alt_proj = F.normalize(heads.provider_projector(_ensure_2d(alt_summary.pooled_context)), dim=-1)
            cosine = F.cosine_similarity(anchor_proj, alt_proj, dim=-1)
            provider_losses.append(1.0 - cosine.mean())
            provider_cosines.append(float(cosine.mean().item()))
        losses["provider_consistency"] = torch.stack(provider_losses).mean()
        metrics["provider_consistency_cosine"] = float(np.mean(provider_cosines))
        metrics["provider_views_used"] = len(provider_views)
    else:
        losses["provider_consistency"] = torch.zeros((), device=device, dtype=dtype)
        metrics["provider_consistency_cosine"] = float("nan")
        metrics["provider_views_used"] = 0

    if include_coordinate_corruption and token_coords is not None and token_coords.shape[0] > 1:
        corrupt_coords = _coordinate_corruption(token_coords, seed=seed + 7_003)
        corrupt_summary = _forward_context_encoder(
            context_encoder,
            context_tokens,
            token_type_ids=token_type_ids,
            token_coords=corrupt_coords,
            token_confidence=token_confidence,
            dataset_ids=dataset_ids,
            edge_ids=edge_ids,
            return_attention=False,
        )
        real_logit = heads.coord_corruption_head(pooled)
        corrupt_logit = heads.coord_corruption_head(_ensure_2d(corrupt_summary.pooled_context))
        real_target = torch.ones_like(real_logit)
        corrupt_target = torch.zeros_like(corrupt_logit)
        losses["coordinate_corruption"] = 0.5 * (
            F.binary_cross_entropy_with_logits(real_logit, real_target)
            + F.binary_cross_entropy_with_logits(corrupt_logit, corrupt_target)
        )
        real_ok = (torch.sigmoid(real_logit.detach()) > 0.5).float().mean()
        corrupt_ok = (torch.sigmoid(corrupt_logit.detach()) < 0.5).float().mean()
        metrics["coordinate_corruption_accuracy"] = float(0.5 * (real_ok + corrupt_ok).item())
    else:
        losses["coordinate_corruption"] = torch.zeros((), device=device, dtype=dtype)
        metrics["coordinate_corruption_accuracy"] = float("nan")

    if include_group_relation:
        group_means = _group_means(summary, num_groups=heads.token_type_embedding.num_embeddings)
        if group_means is not None and group_means.shape[0] >= 2:
            positive_pairs: list[Tensor] = []
            negative_pairs: list[Tensor] = []
            if negative_summaries:
                mismatch_means = _group_means(negative_summaries[0][1], num_groups=heads.token_type_embedding.num_embeddings)
            else:
                mismatch_means = None
            if mismatch_means is None and group_means.shape[0] > 1:
                mismatch_means = torch.roll(group_means, shifts=1, dims=0)
            if mismatch_means is not None:
                for left_idx in range(group_means.shape[0]):
                    for right_idx in range(left_idx + 1, group_means.shape[0]):
                        positive_pairs.append(torch.cat([group_means[left_idx], group_means[right_idx]], dim=-1))
                        negative_pairs.append(torch.cat([group_means[left_idx], mismatch_means[right_idx]], dim=-1))
            if positive_pairs and negative_pairs:
                positive_logits = heads.group_relation_head(torch.stack(positive_pairs, dim=0))
                negative_logits = heads.group_relation_head(torch.stack(negative_pairs, dim=0))
                losses["group_relation"] = 0.5 * (
                    F.binary_cross_entropy_with_logits(positive_logits, torch.ones_like(positive_logits))
                    + F.binary_cross_entropy_with_logits(negative_logits, torch.zeros_like(negative_logits))
                )
                pos_acc = (torch.sigmoid(positive_logits.detach()) > 0.5).float().mean()
                neg_acc = (torch.sigmoid(negative_logits.detach()) < 0.5).float().mean()
                metrics["group_relation_accuracy"] = float(0.5 * (pos_acc + neg_acc).item())
            else:
                losses["group_relation"] = torch.zeros((), device=device, dtype=dtype)
                metrics["group_relation_accuracy"] = float("nan")
        else:
            losses["group_relation"] = torch.zeros((), device=device, dtype=dtype)
            metrics["group_relation_accuracy"] = float("nan")
    else:
        losses["group_relation"] = torch.zeros((), device=device, dtype=dtype)
        metrics["group_relation_accuracy"] = float("nan")

    total = (
        float(config.masked_token_weight) * losses["masked_token"]
        + float(config.ranking_weight) * losses["ranking"]
        + float(config.provider_consistency_weight) * losses["provider_consistency"]
        + float(config.coordinate_corruption_weight) * losses["coordinate_corruption"]
        + float(config.group_relation_weight) * losses["group_relation"]
    )
    metrics["loss_total"] = float(total.detach().item())
    metrics["loss_masked_token"] = float(losses["masked_token"].detach().item())
    metrics["loss_ranking"] = float(losses["ranking"].detach().item())
    metrics["loss_provider_consistency"] = float(losses["provider_consistency"].detach().item())
    metrics["loss_coordinate_corruption"] = float(losses["coordinate_corruption"].detach().item())
    metrics["loss_group_relation"] = float(losses["group_relation"].detach().item())
    return total, losses, metrics, summary


def pretrain_relational_transformer(
    *,
    context_encoder: nn.Module,
    context_tokens: Tensor,
    token_type_ids: Tensor,
    token_coords: Tensor | None,
    token_confidence: Tensor | None,
    dataset_ids: Tensor | None,
    edge_ids: Tensor | None,
    negative_controls: list[dict[str, Any]] | None,
    provider_views: list[dict[str, Any]] | None,
    config: RelationalPretrainingConfig,
) -> dict[str, Any]:
    """Run self-supervised relational pretraining on the hierarchical transformer."""
    probe_summary = _forward_context_encoder(
        context_encoder,
        context_tokens,
        token_type_ids=token_type_ids,
        token_coords=token_coords,
        token_confidence=token_confidence,
        dataset_ids=dataset_ids,
        edge_ids=edge_ids,
        return_attention=False,
    )
    context_dim = int(_ensure_2d(probe_summary.pooled_context).shape[-1])
    heads = RelationalPretrainingHeads(
        context_dim=context_dim,
        token_dim=int(context_tokens.shape[-1]),
        num_token_types=int(token_type_ids.max().item()) + 1 if token_type_ids.numel() > 0 else 1,
        num_datasets=4,
        num_edges=8,
    ).to(context_tokens.device)

    before = _clone_trainable_parameters(context_encoder)
    params = list(context_encoder.parameters()) + list(heads.parameters())
    optimizer = torch.optim.Adam(
        params,
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    history: list[dict[str, float]] = []
    for epoch in range(int(config.max_epochs)):
        epoch_metrics: dict[str, list[float]] = {}
        for step in range(int(config.steps_per_epoch)):
            total, _, metrics, _ = compute_relational_auxiliary_losses(
                context_encoder=context_encoder,
                heads=heads,
                context_tokens=context_tokens,
                token_type_ids=token_type_ids,
                token_coords=token_coords,
                token_confidence=token_confidence,
                dataset_ids=dataset_ids,
                edge_ids=edge_ids,
                negative_controls=negative_controls,
                provider_views=provider_views,
                config=config,
                seed=int(config.seed) + epoch * 1_000 + step,
                include_masked_token=True,
                include_provider_consistency=True,
                include_coordinate_corruption=True,
                include_group_relation=True,
                return_attention=False,
            )
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            for key, value in metrics.items():
                if isinstance(value, (float, int)) and np.isfinite(float(value)):
                    epoch_metrics.setdefault(key, []).append(float(value))
        history.append(
            {
                "epoch": float(epoch + 1),
                **{
                    key: float(np.mean(values))
                    for key, values in epoch_metrics.items()
                    if values
                },
            }
        )

    with torch.no_grad():
        _, _, final_metrics, final_summary = compute_relational_auxiliary_losses(
            context_encoder=context_encoder,
            heads=heads,
            context_tokens=context_tokens,
            token_type_ids=token_type_ids,
            token_coords=token_coords,
            token_confidence=token_confidence,
            dataset_ids=dataset_ids,
            edge_ids=edge_ids,
            negative_controls=negative_controls,
            provider_views=provider_views,
            config=config,
            seed=int(config.seed) + 99_999,
            include_masked_token=True,
            include_provider_consistency=True,
            include_coordinate_corruption=True,
            include_group_relation=True,
            return_attention=True,
        )

    return {
        "history": history,
        "metrics": final_metrics,
        "context_summary": final_summary,
        "heads": heads,
        "encoder_parameter_delta": _mean_parameter_delta(before, context_encoder),
    }
