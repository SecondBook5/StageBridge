"""Lesion-level Set Transformer and full EA-MIST model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from stagebridge.context_model.baselines_lesion import LesionModelOutput
from stagebridge.context_model.evolution_branch import EvolutionBranch
from stagebridge.context_model.heads import LesionMultitaskHeads
from stagebridge.context_model.local_niche_encoder import (
    LocalNicheMLPEncoder,
    LocalNicheTransformerEncoder,
)
from stagebridge.context_model.prototype_bottleneck import (
    PrototypeBottleneck,
    PrototypeBottleneckOutput,
)
from stagebridge.context_model.set_encoder import PMA, ISAB, SAB
from stagebridge.utils.types import LesionBagBatch


@dataclass(slots=True, frozen=True)
class EAMISTOutput:
    """Structured output from the full EA-MIST model."""

    local_embeddings: Tensor
    lesion_embedding: Tensor
    stage_logits: Tensor
    displacement: Tensor
    edge_logits: Tensor | None = None
    prototype_output: PrototypeBottleneckOutput | None = None
    lesion_attention: Tensor | None = None
    local_attention: Tensor | None = None
    evolution_embedding: Tensor | None = None
    niche_transition_scores: Tensor | None = None


class NicheTransitionScoreHead(nn.Module):
    """Per-niche scalar transition score from local niche embeddings."""

    def __init__(self, model_dim: int, *, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(model_dim), int(model_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(model_dim), 1),
        )

    def forward(self, niche_embeddings: Tensor, mask: Tensor) -> Tensor:
        """Return per-niche transition scores (B, N). Masked niches get -inf."""
        scores = self.net(niche_embeddings).squeeze(-1)  # (B, N)
        scores = scores.masked_fill(~mask, float("-inf"))
        return scores


class LesionSetTransformerBackbone(nn.Module):
    """Permutation-invariant lesion-level Set Transformer backbone."""

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        num_inducing_points: int = 16,
        num_pma_seeds: int = 1,
        dropout: float = 0.1,
        use_isab: bool = True,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(int(input_dim), int(hidden_dim))
        if use_isab:
            self.blocks = nn.ModuleList(
                [
                    ISAB(
                        dim=int(hidden_dim),
                        num_heads=int(num_heads),
                        num_inducing_points=int(num_inducing_points),
                        dropout=float(dropout),
                    )
                    for _ in range(int(num_layers))
                ]
            )
        else:
            self.blocks = nn.ModuleList(
                [
                    SAB(dim=int(hidden_dim), num_heads=int(num_heads), dropout=float(dropout))
                    for _ in range(int(num_layers))
                ]
            )
        self.pool = PMA(
            dim=int(hidden_dim),
            num_heads=int(num_heads),
            num_seed_vectors=int(num_pma_seeds),
            dropout=float(dropout),
        )
        self.norm = nn.LayerNorm(int(hidden_dim))

    def forward(
        self, tokens: Tensor, mask: Tensor, *, return_attention: bool = False
    ) -> tuple[Tensor, Tensor | None]:
        """Encode a lesion bag into one lesion embedding."""
        hidden = self.input_proj(tokens)
        attention = None
        for layer_idx, block in enumerate(self.blocks):
            if return_attention and layer_idx == len(self.blocks) - 1 and isinstance(block, SAB):
                hidden, attention = block(hidden, mask=mask, return_attention=True)
            else:
                if isinstance(block, SAB):
                    hidden = block(hidden, mask=mask)
                else:
                    hidden = block(hidden, mask=mask)
        pooled = self.pool(hidden, mask=mask)
        lesion_embedding = self.norm(pooled[:, 0, :])
        return lesion_embedding, attention


class EAMISTModel(nn.Module):
    """Evolution-Aware Multiple-Instance Set Transformer."""

    def __init__(
        self,
        *,
        receiver_dim: int,
        sender_feature_dim: int,
        hlca_dim: int,
        luca_dim: int,
        lr_summary_dim: int,
        stats_dim: int,
        flat_feature_dim: int,
        num_receiver_states: int,
        num_rings: int = 4,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        num_inducing_points: int = 16,
        num_pma_seeds: int = 1,
        dropout: float = 0.1,
        local_encoder_type: str = "transformer",
        use_prototypes: bool = True,
        num_prototypes: int = 16,
        sparse_assignments: bool = False,
        evolution_dim: int | None = None,
        evolution_mode: str = "gated",
        num_stage_classes: int = 5,
        num_edge_heads: int = 0,
        reference_feature_mode: str = "hlca_luca",
        use_distribution_summary: bool = False,
        use_atlas_contrast_token: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.local_encoder_type = str(local_encoder_type)
        self.use_prototypes = bool(use_prototypes)
        self.reference_feature_mode = str(reference_feature_mode)
        self.use_distribution_summary = bool(use_distribution_summary)
        self.use_atlas_contrast_token = bool(use_atlas_contrast_token)
        if self.reference_feature_mode not in {"hlca_luca", "hlca_only", "luca_only", "no_atlas"}:
            raise ValueError(f"Unsupported reference_feature_mode '{reference_feature_mode}'.")
        if self.local_encoder_type == "transformer":
            self.local_encoder: nn.Module = LocalNicheTransformerEncoder(
                receiver_dim=receiver_dim,
                sender_feature_dim=sender_feature_dim,
                hlca_dim=hlca_dim,
                luca_dim=luca_dim,
                lr_summary_dim=lr_summary_dim,
                stats_dim=stats_dim,
                model_dim=self.hidden_dim,
                num_heads=num_heads,
                num_layers=2,
                num_receiver_states=num_receiver_states,
                num_rings=num_rings,
                dropout=dropout,
                use_atlas_contrast_token=self.use_atlas_contrast_token,
            )
        elif self.local_encoder_type == "mlp":
            self.local_encoder = LocalNicheMLPEncoder(
                input_dim=flat_feature_dim, hidden_dim=self.hidden_dim, dropout=dropout
            )
        else:
            raise ValueError(f"Unsupported local_encoder_type '{local_encoder_type}'.")

        self.prototype_bottleneck = (
            PrototypeBottleneck(
                self.hidden_dim,
                num_prototypes=num_prototypes,
                sparse_assignment=sparse_assignments,
            )
            if self.use_prototypes
            else None
        )
        self.lesion_backbone = LesionSetTransformerBackbone(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            num_inducing_points=num_inducing_points,
            num_pma_seeds=num_pma_seeds,
            dropout=dropout,
            use_isab=True,
        )
        self.evolution_branch = (
            None
            if evolution_dim is None or evolution_dim <= 0
            else EvolutionBranch(
                evolution_dim, self.hidden_dim, mode=evolution_mode, dropout=dropout
            )
        )
        # Distribution-aware pooling: per-niche transition score → summary stats
        _num_dist_stats = 7  # mean, std, min, max, q25, median, q75
        self.niche_transition_head = (
            NicheTransitionScoreHead(self.hidden_dim, dropout=dropout)
            if self.use_distribution_summary
            else None
        )
        head_input_dim = self.hidden_dim + (
            _num_dist_stats if self.use_distribution_summary else 0
        )
        self.heads = LesionMultitaskHeads(
            head_input_dim,
            num_stage_classes=num_stage_classes,
            num_edge_heads=num_edge_heads,
            dropout=dropout,
        )

    def _resolve_reference_features(self, batch: LesionBagBatch) -> tuple[Tensor, Tensor]:
        hlca = batch.hlca_features
        luca = batch.luca_features
        if hlca is None:
            hlca = torch.zeros(
                (*batch.receiver_embeddings.shape[:2], 0),
                dtype=batch.receiver_embeddings.dtype,
                device=batch.receiver_embeddings.device,
            )
        if luca is None:
            luca = torch.zeros(
                (*batch.receiver_embeddings.shape[:2], 0),
                dtype=batch.receiver_embeddings.dtype,
                device=batch.receiver_embeddings.device,
            )
        if self.reference_feature_mode == "hlca_only" and luca.shape[-1] > 0:
            luca = torch.zeros_like(luca)
        if self.reference_feature_mode == "luca_only" and hlca.shape[-1] > 0:
            hlca = torch.zeros_like(hlca)
        if self.reference_feature_mode == "no_atlas":
            if hlca.shape[-1] > 0:
                hlca = torch.zeros_like(hlca)
            if luca.shape[-1] > 0:
                luca = torch.zeros_like(luca)
        return hlca, luca

    def encode_local(
        self, batch: LesionBagBatch, *, return_attention: bool = False
    ) -> tuple[Tensor, Tensor | None]:
        """Encode each local niche in the batch into one embedding."""
        batch_size, num_instances = batch.receiver_embeddings.shape[:2]
        mask = batch.neighborhood_mask.reshape(batch_size * num_instances)
        hlca_features, luca_features = self._resolve_reference_features(batch)
        if self.local_encoder_type == "transformer":
            total = batch_size * num_instances
            flat_receiver = batch.receiver_embeddings.reshape(total, -1)
            flat_state_ids = batch.receiver_state_ids.reshape(total)
            flat_rings = batch.ring_compositions.reshape(
                total, batch.ring_compositions.shape[2], batch.ring_compositions.shape[3]
            )
            flat_hlca = hlca_features.reshape(total, -1)
            flat_luca = luca_features.reshape(total, -1)
            flat_lr = batch.lr_pathway_summary.reshape(total, -1)
            flat_stats = batch.neighborhood_stats.reshape(total, -1)
            # Chunk to avoid exceeding PyTorch SDPA batch limit (65535)
            max_chunk = 32768
            if total <= max_chunk:
                output = self.local_encoder(
                    receiver_embeddings=flat_receiver,
                    receiver_state_ids=flat_state_ids,
                    ring_compositions=flat_rings,
                    hlca_features=flat_hlca,
                    luca_features=flat_luca,
                    lr_pathway_summary=flat_lr,
                    neighborhood_stats=flat_stats,
                    return_attention=return_attention,
                )
                all_embeddings = output.neighborhood_embedding
                local_attention = output.attention_weights
            else:
                chunks = []
                local_attention = None
                for start in range(0, total, max_chunk):
                    end = min(start + max_chunk, total)
                    chunk_output = self.local_encoder(
                        receiver_embeddings=flat_receiver[start:end],
                        receiver_state_ids=flat_state_ids[start:end],
                        ring_compositions=flat_rings[start:end],
                        hlca_features=flat_hlca[start:end],
                        luca_features=flat_luca[start:end],
                        lr_pathway_summary=flat_lr[start:end],
                        neighborhood_stats=flat_stats[start:end],
                        return_attention=False,
                    )
                    chunks.append(chunk_output.neighborhood_embedding)
                all_embeddings = torch.cat(chunks, dim=0)
            embeddings = all_embeddings.reshape(batch_size, num_instances, -1)
        else:
            output = self.local_encoder(
                batch.flat_features.reshape(batch_size * num_instances, -1)
            )
            embeddings = output.neighborhood_embedding.reshape(batch_size, num_instances, -1)
            local_attention = None
        embeddings = embeddings * batch.neighborhood_mask.unsqueeze(-1).to(embeddings.dtype)
        if mask.sum().item() == 0:
            raise ValueError("All neighborhoods in the batch are masked out.")
        return embeddings, local_attention

    def forward(self, batch: LesionBagBatch, *, return_attention: bool = False) -> EAMISTOutput:
        """Run the full EA-MIST forward pass over one lesion batch."""
        local_embeddings, local_attention = self.encode_local(
            batch, return_attention=return_attention
        )
        if self.prototype_bottleneck is not None:
            prototype_output = self.prototype_bottleneck(
                local_embeddings, mask=batch.neighborhood_mask
            )
            lesion_tokens = prototype_output.aligned_embeddings
        else:
            prototype_output = None
            lesion_tokens = local_embeddings
        lesion_embedding, lesion_attention = self.lesion_backbone(
            lesion_tokens, batch.neighborhood_mask, return_attention=return_attention
        )
        fused_lesion, evolution_embedding = (
            (lesion_embedding, None)
            if self.evolution_branch is None
            else self.evolution_branch(lesion_embedding, batch.evolution_features)
        )
        # Distribution-aware pooling: compute per-niche transition scores and summary stats
        niche_transition_scores = None
        head_input = fused_lesion
        if self.niche_transition_head is not None:
            niche_transition_scores = self.niche_transition_head(
                local_embeddings, batch.neighborhood_mask
            )
            # Compute summary statistics over valid niches
            valid_scores = niche_transition_scores.masked_fill(
                ~batch.neighborhood_mask, float("nan")
            )
            s_mean = torch.nanmean(valid_scores, dim=-1, keepdim=True)
            # std, min, max, quantiles via sorting valid entries
            # Replace nan with large value for min/sort, small for max
            big = torch.finfo(valid_scores.dtype).max
            scores_for_min = valid_scores.masked_fill(~batch.neighborhood_mask, big)
            scores_for_max = valid_scores.masked_fill(~batch.neighborhood_mask, -big)
            s_min = scores_for_min.min(dim=-1, keepdim=True).values
            s_max = scores_for_max.max(dim=-1, keepdim=True).values
            # std: manual to handle masking
            counts = (
                batch.neighborhood_mask.sum(dim=-1, keepdim=True)
                .clamp_min(1)
                .to(valid_scores.dtype)
            )
            diffs = (valid_scores - s_mean).masked_fill(~batch.neighborhood_mask, 0.0)
            s_std = (diffs.pow(2).sum(dim=-1, keepdim=True) / counts.clamp_min(2)).sqrt()
            # quantiles via sorted valid scores
            sorted_scores, _ = scores_for_min.sort(dim=-1)
            N = counts.squeeze(-1).long()  # (B,)
            batch_idx = torch.arange(sorted_scores.shape[0], device=sorted_scores.device)
            q25_idx = ((N.float() - 1) * 0.25).clamp_min(0).long()
            q50_idx = ((N.float() - 1) * 0.50).clamp_min(0).long()
            q75_idx = ((N.float() - 1) * 0.75).clamp_min(0).long()
            s_q25 = sorted_scores[batch_idx, q25_idx].unsqueeze(-1)
            s_q50 = sorted_scores[batch_idx, q50_idx].unsqueeze(-1)
            s_q75 = sorted_scores[batch_idx, q75_idx].unsqueeze(-1)
            dist_stats = torch.cat([s_mean, s_std, s_min, s_max, s_q25, s_q50, s_q75], dim=-1)
            head_input = torch.cat([fused_lesion, dist_stats], dim=-1)
        task_output = self.heads(head_input)
        return EAMISTOutput(
            local_embeddings=local_embeddings,
            lesion_embedding=fused_lesion,
            stage_logits=task_output.stage_logits,
            displacement=task_output.displacement,
            edge_logits=task_output.edge_logits,
            prototype_output=prototype_output,
            lesion_attention=lesion_attention,
            local_attention=local_attention,
            evolution_embedding=evolution_embedding,
            niche_transition_scores=niche_transition_scores,
        )


def lesion_output_from_eamist(output: EAMISTOutput) -> LesionModelOutput:
    """Convert an EA-MIST output into the common lesion baseline contract."""
    return LesionModelOutput(
        lesion_embedding=output.lesion_embedding,
        stage_logits=output.stage_logits,
        displacement=output.displacement,
        edge_logits=output.edge_logits,
        attention_weights=output.lesion_attention,
    )
