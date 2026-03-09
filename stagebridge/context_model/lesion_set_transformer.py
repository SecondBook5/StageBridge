"""Lesion-level Set Transformer and full EA-MIST model."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from stagebridge.context_model.baselines_lesion import LesionModelOutput
from stagebridge.context_model.evolution_branch import EvolutionBranch
from stagebridge.context_model.heads import LesionMultitaskHeads
from stagebridge.context_model.local_niche_encoder import LocalNicheMLPEncoder, LocalNicheTransformerEncoder
from stagebridge.context_model.prototype_bottleneck import PrototypeBottleneck, PrototypeBottleneckOutput
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
                [ISAB(dim=int(hidden_dim), num_heads=int(num_heads), num_inducing_points=int(num_inducing_points), dropout=float(dropout)) for _ in range(int(num_layers))]
            )
        else:
            self.blocks = nn.ModuleList(
                [SAB(dim=int(hidden_dim), num_heads=int(num_heads), dropout=float(dropout)) for _ in range(int(num_layers))]
            )
        self.pool = PMA(dim=int(hidden_dim), num_heads=int(num_heads), num_seed_vectors=int(num_pma_seeds), dropout=float(dropout))
        self.norm = nn.LayerNorm(int(hidden_dim))

    def forward(self, tokens: Tensor, mask: Tensor, *, return_attention: bool = False) -> tuple[Tensor, Tensor | None]:
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
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.local_encoder_type = str(local_encoder_type)
        self.use_prototypes = bool(use_prototypes)
        self.reference_feature_mode = str(reference_feature_mode)
        if self.reference_feature_mode not in {"hlca_luca", "hlca_only", "luca_only"}:
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
            )
        elif self.local_encoder_type == "mlp":
            self.local_encoder = LocalNicheMLPEncoder(input_dim=flat_feature_dim, hidden_dim=self.hidden_dim, dropout=dropout)
        else:
            raise ValueError(f"Unsupported local_encoder_type '{local_encoder_type}'.")

        self.prototype_bottleneck = (
            PrototypeBottleneck(self.hidden_dim, num_prototypes=num_prototypes, sparse_assignment=sparse_assignments)
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
        self.evolution_branch = None if evolution_dim is None or evolution_dim <= 0 else EvolutionBranch(evolution_dim, self.hidden_dim, mode=evolution_mode, dropout=dropout)
        self.heads = LesionMultitaskHeads(self.hidden_dim, num_stage_classes=num_stage_classes, num_edge_heads=num_edge_heads, dropout=dropout)

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

    def encode_local(self, batch: LesionBagBatch, *, return_attention: bool = False) -> tuple[Tensor, Tensor | None]:
        """Encode each local niche in the batch into one embedding."""
        batch_size, num_instances = batch.receiver_embeddings.shape[:2]
        mask = batch.neighborhood_mask.reshape(batch_size * num_instances)
        hlca_features, luca_features = self._resolve_reference_features(batch)
        if self.local_encoder_type == "transformer":
            output = self.local_encoder(
                receiver_embeddings=batch.receiver_embeddings.reshape(batch_size * num_instances, -1),
                receiver_state_ids=batch.receiver_state_ids.reshape(batch_size * num_instances),
                ring_compositions=batch.ring_compositions.reshape(batch_size * num_instances, batch.ring_compositions.shape[2], batch.ring_compositions.shape[3]),
                hlca_features=hlca_features.reshape(batch_size * num_instances, -1),
                luca_features=luca_features.reshape(batch_size * num_instances, -1),
                lr_pathway_summary=batch.lr_pathway_summary.reshape(batch_size * num_instances, -1),
                neighborhood_stats=batch.neighborhood_stats.reshape(batch_size * num_instances, -1),
                return_attention=return_attention,
            )
            embeddings = output.neighborhood_embedding.reshape(batch_size, num_instances, -1)
            local_attention = output.attention_weights
        else:
            output = self.local_encoder(batch.flat_features.reshape(batch_size * num_instances, -1))
            embeddings = output.neighborhood_embedding.reshape(batch_size, num_instances, -1)
            local_attention = None
        embeddings = embeddings * batch.neighborhood_mask.unsqueeze(-1).to(embeddings.dtype)
        if mask.sum().item() == 0:
            raise ValueError("All neighborhoods in the batch are masked out.")
        return embeddings, local_attention

    def forward(self, batch: LesionBagBatch, *, return_attention: bool = False) -> EAMISTOutput:
        """Run the full EA-MIST forward pass over one lesion batch."""
        local_embeddings, local_attention = self.encode_local(batch, return_attention=return_attention)
        if self.prototype_bottleneck is not None:
            prototype_output = self.prototype_bottleneck(local_embeddings, mask=batch.neighborhood_mask)
            lesion_tokens = prototype_output.aligned_embeddings
        else:
            prototype_output = None
            lesion_tokens = local_embeddings
        lesion_embedding, lesion_attention = self.lesion_backbone(lesion_tokens, batch.neighborhood_mask, return_attention=return_attention)
        fused_lesion, evolution_embedding = (
            (lesion_embedding, None)
            if self.evolution_branch is None
            else self.evolution_branch(lesion_embedding, batch.evolution_features)
        )
        task_output = self.heads(fused_lesion)
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
