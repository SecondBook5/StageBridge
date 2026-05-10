"""StageBridge: Receiver-centered niche encoding for stage transition modeling.

The canonical StageBridge model combines:
1. Receiver-centered niche encoding (receiver as query, neighbors as keys/values)
2. Context refinement via Set Transformer (SAB layers)
3. Hierarchical aggregation (ISAB + PMA) for sample-level embeddings
4. Cross-attention drift head for OT-CFM with gated context/latent mixing
5. Optional WES/genomic conditioning via evolution branch
6. Sample-level prediction heads (stage, displacement)

The key scientific principle: information flows TO the receiver cell from its
spatial neighbors. The receiver is the query, neighbors are keys/values.

Computational Complexity
------------------------
StageBridge processes cells independently with local context:

- **Time**: O(N × K²) where N=cells, K=context tokens (9 fixed)
  - Each cell: self-attention over 9 tokens = O(K²) = O(81) = O(1)
  - Total: O(N) linear in cell count

- **Memory**: O(B × K × D) where B=batch, K=tokens, D=hidden_dim
  - Batched: 64 × 9 × 128 = ~74K parameters per batch
  - No global pairwise matrices

This contrasts with global optimal transport methods (e.g., moscot):
- **Time**: O(N² × iterations) for Sinkhorn
- **Memory**: O(N × M) for full transport matrix
  - 331K × 296K cells = 98B entries = 392GB at float32

StageBridge scales to millions of cells; OT methods require subsampling.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from stagebridge.context.encoder import ReceiverNicheOutput, ReceiverCenteredNicheEncoder
from stagebridge.context.layers import SAB, ISAB, PMA, SinusoidalTimeEmbedding
from stagebridge.context.tokenizer import NicheTokenizer
from stagebridge.context.aggregation import HierarchicalAggregator, SampleLevelHeads, PrototypeBottleneck
from stagebridge.context.evolution import EvolutionBranch
from stagebridge.models.heads import PathwayHead, ProliferationHead
from stagebridge.transition.drift import (
    CrossAttentionDrift,
    FiLMConditioner,
)
from stagebridge.reference.gw_precompute import PretrainedGWFusion
from stagebridge.reference.learned_gw_fusion import LearnedGWFusion, LearnedGWConfig
from stagebridge.contracts import EVOLUTION_DIM, STATS_TOKEN_DIM


@dataclass(slots=True)
class StageBridgeConfig:
    """Configuration for the StageBridge model.

    Attributes:
        input_dim: Cell embedding dimension (40d fused = 30d HLCA + 10d LuCA)
        hidden_dim: Internal representation dimension
        num_heads: Attention heads
        num_encoder_layers: Attention layers in niche encoder
        max_neighbors: Maximum neighbors in niche (8 for 9-token structure)
        num_stages: Number of disease stages (3, 4, or 5)
        time_dim: Time embedding dimension
        stage_dim: Stage embedding dimension
        dropout: Dropout rate

        # Context refinement (Hierarchical Set Transformer)
        use_context_refiner: Use ISAB→ISAB→SAB→PMA to refine context tokens
        refiner_num_inducing: Inducing points for ISAB layers
        refiner_use_spatial_rpe: Enable spatial RPE in second ISAB

        # Sample-level hierarchical aggregation (for multi-cell → sample)
        use_hierarchical: Enable sample-level hierarchical aggregation
        hierarchical_num_layers: ISAB layers for aggregation
        hierarchical_num_inducing: Inducing points for ISAB

        # Drift head
        use_cross_attn_drift: Use cross-attention drift (vs MLP)
        use_gated_baseline: Gated baseline mode (simple baseline + learned correction)

        # Evolution branch
        use_evolution_branch: Use WES feature conditioning
        evolution_dim: WES feature dimension (if used)
        evolution_mode: Evolution conditioning mode (gated/film)

        # Sample-level heads
        use_sample_heads: Enable sample-level prediction heads

        # Auxiliary biological heads
        use_pathway_head: Predict PROGENy pathway activities
        n_pathways: Number of pathways (14 for PROGENy)
        use_proliferation_head: Predict Ki67 proliferation
    """

    input_dim: int = 40
    hidden_dim: int = 256
    num_heads: int = 8
    num_encoder_layers: int = 2
    max_neighbors: int = 8
    num_stages: int = 3
    time_dim: int = 32
    stage_dim: int = 32
    dropout: float = 0.1

    # Context refinement (Hierarchical Set Transformer: ISAB→ISAB→SAB→PMA)
    use_context_refiner: bool = True
    refiner_num_inducing: int = 8
    refiner_use_spatial_rpe: bool = True

    # Hierarchical aggregation
    use_hierarchical: bool = True
    hierarchical_num_layers: int = 2
    hierarchical_num_inducing: int = 16
    hierarchical_use_prototypes: bool = False  # Sample-level prototype bottleneck
    hierarchical_num_prototypes: int = 8  # Patient niche composition archetypes

    # Neighborhood-level prototype bottleneck (after context refiner)
    use_niche_prototypes: bool = False
    num_niche_prototypes: int = 16  # Local niche archetypes (IL1B-high, fibrotic, etc.)

    # Drift head
    use_cross_attn_drift: bool = True

    # Evolution branch (WES somatic mutations + clonal features)
    use_evolution_branch: bool = False
    evolution_dim: int = EVOLUTION_DIM  # WES + clonal from contracts
    evolution_mode: str = "gated"

    # Sample-level heads
    use_sample_heads: bool = True

    # Auxiliary biological heads
    use_pathway_head: bool = True
    n_pathways: int = 14  # PROGENy pathways
    use_proliferation_head: bool = True

    # Biological conditioning (stats and pathway token features)
    # Dimensions detected from data - set to None to use input_dim as fallback
    use_stats_conditioning: bool = True
    stats_dim: int = STATS_TOKEN_DIM  # from contracts (5 features)
    pathway_dim: int | None = None  # Auto-detect from data, or use input_dim if None

    # Learned ring pooling (individual cells per ring with ISAB+PMA)
    # DEPRECATED: Use AMICI-style continuous attention instead
    use_learned_ring_pooling: bool = True
    ring_pooler_num_heads: int = 4
    ring_pooler_num_inducing: int = 4
    max_cells_per_ring: int = 50

    # AMICI-style continuous attention (replaces ring binning)
    use_amici_attention: bool = False  # Use ReceiverCenteredNicheEncoder
    amici_num_heads: int = 4  # Each head learns different distance decay
    amici_num_layers: int = 2  # Attention layers
    amici_max_neighbors: int = 100  # Max neighbors (sorted by distance)
    amici_distance_scale: float = 100.0  # Distance normalization (microns)
    amici_use_empty_token: bool = True  # Allow "no neighbor is informative"
    amici_empty_token_score: float = 3.0  # Fixed score for empty token
    amici_sparsity_weight: float = 0.01  # Entropy regularization
    amici_value_l1_weight: float = 0.01  # L1 on values for sparsity

    # Gromov-Wasserstein fusion (replaces concat for HLCA/LuCA)
    # Options:
    #   "concat" - Simple concatenation [HLCA; LuCA] (baseline, default)
    #   "learned_gw" - Learned GW fusion with differentiable coupling (recommended)
    #   "precompute_gw" - Precomputed GW with barycentric projection (moscot-style)
    use_gw_fusion: bool = False
    gw_fusion_type: str = "concat"  # "concat", "learned_gw", "precompute_gw"
    gw_checkpoint_dir: str | None = None  # Path to precomputed GW alignment (for precompute_gw)
    gw_output_dim: int = 40  # Output dim of fused representation
    gw_sinkhorn_iters: int = 20  # Sinkhorn iterations for learned_gw
    gw_sinkhorn_reg: float = 0.1  # Entropic regularization for learned_gw

    # Reference ablations (for hlca_only / luca_only experiments)
    use_hlca_reference: bool = True  # Include HLCA token
    use_luca_reference: bool = True  # Include LuCA token

    # Niche ablation (for no_niche experiment)
    use_niche_context: bool = True  # Include ring tokens (False = receiver only)

    @property
    def num_edges(self) -> int:
        """Number of possible stage transitions."""
        return self.num_stages * self.num_stages

    @classmethod
    def from_checkpoint(cls, checkpoint: dict) -> "StageBridgeConfig":
        """Create config from checkpoint, inferring settings from state_dict.

        Handles cases where config was not fully saved or is missing new fields
        by inspecting the actual model weights to infer architecture settings.

        Args:
            checkpoint: Dict with 'model_state_dict' and optional 'config'

        Returns:
            StageBridgeConfig with settings inferred from checkpoint
        """
        # Extract base config dict
        config_data = checkpoint.get("config", {})
        if isinstance(config_data, dict) and "model_config" in config_data:
            config_dict = config_data["model_config"].copy()
        elif isinstance(config_data, dict):
            config_dict = config_data.copy()
        else:
            config_dict = {}

        state_dict = checkpoint["model_state_dict"]

        # Infer evolution branch settings
        has_evolution_branch = any(k.startswith("evolution_branch.") for k in state_dict)
        if has_evolution_branch and not config_dict.get("use_evolution_branch"):
            config_dict["use_evolution_branch"] = True
            for k, v in state_dict.items():
                if k == "evolution_branch.proj.0.weight":
                    config_dict["evolution_dim"] = v.shape[1]
                    break

        # Infer learned ring pooling vs simple projections
        has_niche_tokenizer = any(k.startswith("niche_tokenizer.") for k in state_dict)
        has_simple_proj = any(k.startswith("simple_") for k in state_dict)
        if has_simple_proj and not has_niche_tokenizer:
            config_dict["use_learned_ring_pooling"] = False
        elif has_niche_tokenizer and not has_simple_proj:
            config_dict["use_learned_ring_pooling"] = True

        # Infer GW fusion
        has_gw_fusion = any(k.startswith("gw_fusion.") for k in state_dict)
        if has_gw_fusion and not config_dict.get("use_gw_fusion"):
            config_dict["use_gw_fusion"] = True

        return cls(**config_dict)


@dataclass(slots=True, frozen=True)
class StageBridgeOutput:
    """Output from StageBridge forward pass.

    Attributes:
        context: Receiver-centered context embedding [B, C]
        context_tokens: Context tokens for cross-attention drift [B, K, C]
        prediction: Drift velocity prediction [B, D]
        attention_weights: Neighbor importance weights for interpretability
        entropy_loss: Attention entropy for regularization
        value_l1_loss: L1 penalty on value vectors (AMICI-style sparsity)
        empty_attention: Attention to empty token [B] (interpretability)
        sample_embedding: Sample-level embedding (if hierarchical) [B, C]
        stage_logits: Stage prediction logits (if sample_heads) [B, num_stages]
        displacement: Predicted displacement (if sample_heads) [B, D]
    """

    context: Tensor
    context_tokens: Tensor | None
    prediction: Tensor
    attention_weights: Tensor | None = None
    entropy_loss: Tensor | None = None
    value_l1_loss: Tensor | None = None
    empty_attention: Tensor | None = None
    sample_embedding: Tensor | None = None
    stage_logits: Tensor | None = None
    displacement: Tensor | None = None
    pathway_logits: Tensor | None = None
    proliferation_logit: Tensor | None = None


class HierarchicalSetTransformer(nn.Module):
    """Hierarchical Set Transformer for niche context encoding.

    Architecture: ISAB → ISAB(spatial_rpe) → SAB → PMA

    This is the proper Set Transformer architecture:
    - ISAB: Inducing points create hierarchical bottleneck
    - Spatial RPE: Distance-aware attention in second ISAB
    - SAB: Token-token refinement
    - PMA: Learned pooling (not mean) to single context vector

    Args:
        dim: Token dimension
        num_heads: Attention heads
        num_inducing_points: ISAB inducing points (controls capacity)
        dropout: Dropout rate
        use_spatial_rpe: Enable spatial relative position encoding
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        num_inducing_points: int = 8,
        dropout: float = 0.1,
        use_spatial_rpe: bool = True,
    ) -> None:
        super().__init__()

        # ISAB → ISAB(spatial_rpe) → SAB → PMA
        self.isab1 = ISAB(
            dim=dim,
            num_heads=num_heads,
            num_inducing_points=num_inducing_points,
            dropout=dropout,
            use_spatial_rpe=False,
        )
        self.isab2 = ISAB(
            dim=dim,
            num_heads=num_heads,
            num_inducing_points=num_inducing_points,
            dropout=dropout,
            use_spatial_rpe=use_spatial_rpe,
        )
        self.sab = SAB(dim=dim, num_heads=num_heads, dropout=dropout)
        self.pma = PMA(
            dim=dim,
            num_heads=num_heads,
            num_seed_vectors=1,
            dropout=dropout,
        )

    def forward(
        self,
        tokens: Tensor,
        mask: Tensor | None = None,
        coords: Tensor | None = None,
        return_entropy: bool = False,
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
        """Encode tokens via hierarchical set transformer.

        Args:
            tokens: [B, N, D] context tokens
            mask: [B, N] attention mask
            coords: [B, N, 2] spatial coordinates (for RPE)
            return_entropy: If True, compute and return attention entropy loss

        Returns:
            context: [B, D] pooled context vector
            refined_tokens: [B, N, D] refined tokens (for cross-attention drift)
            entropy_loss: (optional) Scalar entropy loss for regularization
        """
        # ISAB layers (hierarchy via inducing points)
        h = self.isab1(tokens, mask=mask)
        h = self.isab2(h, mask=mask, coords=coords)

        # SAB for token-token refinement
        h = self.sab(h, mask=mask)

        # PMA for learned pooling - get attention weights for entropy
        if return_entropy:
            context, pma_attn = self.pma(h, mask=mask, return_attention=True)
            context = context.squeeze(1)  # [B, D]

            # Compute attention entropy: -sum(p * log(p + eps))
            # pma_attn shape: [B, num_heads, num_seeds, N]
            # Higher entropy = more uniform attention (bad for interpretability)
            # We want peaked attention, so minimize entropy
            eps = 1e-6
            # Compute in float32 to avoid mixed precision issues
            pma_attn_f32 = pma_attn.float()
            # Clamp attention to valid probability range
            pma_attn_clamped = torch.clamp(pma_attn_f32, min=eps, max=1.0 - eps)
            # Entropy is always non-negative: -sum(p * log(p)) >= 0
            entropy = -torch.sum(pma_attn_clamped * torch.log(pma_attn_clamped), dim=-1)  # [B, H, S]
            entropy_loss = torch.abs(entropy.mean())  # Ensure positive (defensive)

            return context, h, entropy_loss
        else:
            context = self.pma(h, mask=mask)  # [B, 1, D]
            context = context.squeeze(1)  # [B, D]
            return context, h


# Backward compatibility alias
SetTransformerRefiner = HierarchicalSetTransformer


class StageBridge(nn.Module):
    """StageBridge: Receiver-centered niche encoding for stage transitions.

    The core principle is RECEIVER-CENTERING: the focal cell (receiver) is the
    attention query, neighbors are keys/values, and information flows TO the
    receiver. This models "what does this cell receive from its neighborhood?"

    Architecture:
    1. ReceiverCenteredNicheEncoder: Encode receiver + neighbors into context
    2. SetTransformerRefiner: Refine context tokens via self-attention (SAB)
    3. HierarchicalAggregator: Aggregate niches to sample-level (ISAB + PMA)
    4. CrossAttentionDrift: Predict velocity conditioned on context
    5. SampleLevelHeads: Predict stage and displacement

    Args:
        config: Model configuration
    """

    def __init__(self, config: StageBridgeConfig) -> None:
        super().__init__()
        self.config = config

        # Resolve pathway_dim: if None, use input_dim as fallback
        if config.pathway_dim is None:
            config.pathway_dim = config.input_dim

        # Hierarchical Set Transformer: ISAB → ISAB(spatial_rpe) → SAB → PMA
        self.context_refiner: HierarchicalSetTransformer | None = None
        if config.use_context_refiner:
            self.context_refiner = HierarchicalSetTransformer(
                dim=config.hidden_dim,
                num_heads=config.num_heads,
                num_inducing_points=config.refiner_num_inducing,
                dropout=config.dropout,
                use_spatial_rpe=config.refiner_use_spatial_rpe,
            )

        # Neighborhood-level prototype bottleneck (interpretable niche archetypes)
        self.niche_prototype_bottleneck: PrototypeBottleneck | None = None
        if config.use_niche_prototypes:
            self.niche_prototype_bottleneck = PrototypeBottleneck(
                model_dim=config.hidden_dim,
                num_prototypes=config.num_niche_prototypes,
                sparse_assignment=False,
            )

        # Time and stage embeddings
        self.time_embedding = SinusoidalTimeEmbedding(config.time_dim)
        self.stage_embedding = nn.Embedding(config.num_edges, config.stage_dim)

        # Drift head
        if config.use_cross_attn_drift:
            self.drift_head = CrossAttentionDrift(
                input_dim=config.input_dim,
                context_dim=config.hidden_dim,
                time_dim=config.time_dim,
                stage_dim=config.stage_dim,
                num_heads=config.num_heads,
                dropout=config.dropout,
            )
        else:
            cond_dim = config.hidden_dim + config.stage_dim
            self.film = FiLMConditioner(config.input_dim, cond_dim)
            vf_input_dim = (
                config.input_dim
                + config.time_dim
                + config.hidden_dim
                + config.stage_dim
            )
            self.drift_head = nn.Sequential(
                nn.Linear(vf_input_dim, config.hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim * 2, config.input_dim),
            )

        # Evolution branch for WES features
        self.evolution_branch: EvolutionBranch | None = None
        if config.use_evolution_branch and config.evolution_dim > 0:
            self.evolution_branch = EvolutionBranch(
                evolution_dim=config.evolution_dim,
                model_dim=config.hidden_dim,
                mode=config.evolution_mode,
                dropout=config.dropout,
            )

        # Hierarchical aggregation for sample-level representations
        self.hierarchical_aggregator: HierarchicalAggregator | None = None
        if config.use_hierarchical:
            self.hierarchical_aggregator = HierarchicalAggregator(
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                num_layers=config.hierarchical_num_layers,
                num_inducing_points=config.hierarchical_num_inducing,
                dropout=config.dropout,
                use_prototypes=config.hierarchical_use_prototypes,
                num_prototypes=config.hierarchical_num_prototypes,
            )

        # Sample-level prediction heads
        self.sample_heads: SampleLevelHeads | None = None
        if config.use_sample_heads:
            self.sample_heads = SampleLevelHeads(
                input_dim=config.hidden_dim,
                num_stage_classes=config.num_stages,
                dropout=config.dropout,
            )

        # Auxiliary biological heads for regularization
        self.pathway_head: PathwayHead | None = None
        if config.use_pathway_head:
            self.pathway_head = PathwayHead(
                input_dim=config.hidden_dim,
                n_pathways=config.n_pathways,
            )

        self.proliferation_head: ProliferationHead | None = None
        if config.use_proliferation_head:
            self.proliferation_head = ProliferationHead(
                input_dim=config.hidden_dim,
            )

        # Stats conditioning (cell cycle, CAF, plasticity - as inputs, not targets)
        self.stats_conditioner: FiLMConditioner | None = None
        if config.use_stats_conditioning and config.stats_dim > 0:
            self.stats_conditioner = FiLMConditioner(
                feature_dim=config.hidden_dim,
                condition_dim=config.stats_dim,
            )

        # Gromov-Wasserstein fusion for HLCA-LuCA alignment
        # Must be initialized before tokenizer to know output dimension
        self.gw_fusion: LearnedGWFusion | PretrainedGWFusion | None = None
        if config.use_gw_fusion:
            if config.gw_fusion_type == "concat":
                # Baseline: no fusion module, just concatenate in forward pass
                self.gw_fusion = None
            elif config.gw_fusion_type == "learned_gw":
                # Learned GW fusion with differentiable coupling
                gw_config = LearnedGWConfig(
                    hlca_dim=30,
                    luca_dim=10,
                    metric_dim=32,
                    output_dim=config.gw_output_dim,
                    sinkhorn_reg=config.gw_sinkhorn_reg,
                    sinkhorn_iters=config.gw_sinkhorn_iters,
                    gw_iters=10,
                    dropout=config.dropout,
                )
                self.gw_fusion = LearnedGWFusion(gw_config)
            elif config.gw_fusion_type == "precompute_gw":
                # Precomputed GW with barycentric projection (moscot-style)
                if config.gw_checkpoint_dir is None:
                    raise ValueError("gw_checkpoint_dir required for precompute_gw fusion")
                self.gw_fusion = PretrainedGWFusion(config.gw_checkpoint_dir)
            else:
                raise ValueError(
                    f"Unknown gw_fusion_type: {config.gw_fusion_type}. "
                    f"Valid options: concat, learned_gw, precompute_gw"
                )

        # AMICI-style continuous attention (preferred)
        self.amici_encoder: ReceiverCenteredNicheEncoder | None = None
        if config.use_amici_attention:
            self.amici_encoder = ReceiverCenteredNicheEncoder(
                input_dim=config.input_dim,
                hidden_dim=config.hidden_dim,
                num_heads=config.amici_num_heads,
                num_layers=config.amici_num_layers,
                max_neighbors=config.amici_max_neighbors,
                sparsity_weight=config.amici_sparsity_weight,
                value_l1_weight=config.amici_value_l1_weight,
                dropout=config.dropout,
                use_reconstruction_head=True,
                use_empty_token=config.amici_use_empty_token,
                empty_token_score=config.amici_empty_token_score,
                distance_scale=config.amici_distance_scale,
            )
            # Projections for reference tokens (not part of AMICI attention)
            self.amici_hlca_proj = nn.Linear(30, config.hidden_dim)
            self.amici_luca_proj = nn.Linear(10, config.hidden_dim)
            self.amici_stats_proj = nn.Linear(config.stats_dim, config.hidden_dim)
            self.amici_pathway_proj = nn.Linear(config.pathway_dim, config.hidden_dim)
            # Fallback reconstruction head for no_niche ablation (when AMICI encoder is skipped)
            # This allows SSL training to proceed, but the model will perform poorly
            # without niche context, demonstrating the value of spatial information
            self.no_niche_reconstruction_head = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.GELU(),
                nn.Linear(config.hidden_dim, config.input_dim),
            )

        # Learned ring pooling (DEPRECATED - use AMICI instead)
        self.niche_tokenizer: NicheTokenizer | None = None
        if config.use_learned_ring_pooling and not config.use_amici_attention:
            self.niche_tokenizer = NicheTokenizer(
                input_dim=config.input_dim,
                hidden_dim=config.hidden_dim,
                num_rings=4,
                num_heads=config.ring_pooler_num_heads,
                num_inducing=config.ring_pooler_num_inducing,
                dropout=config.dropout,
                stats_dim=config.stats_dim,
                pathway_dim=config.pathway_dim,
                use_fused_reference=config.use_gw_fusion,
                fused_ref_dim=config.gw_output_dim if config.use_gw_fusion else None,
            )
        elif not config.use_amici_attention:
            # Simple projections for fallback mean-pooling tokenization
            self.simple_token_proj = nn.Linear(config.input_dim, config.hidden_dim)
            self.simple_hlca_proj = nn.Linear(30, config.hidden_dim)  # HLCA dim
            self.simple_luca_proj = nn.Linear(10, config.hidden_dim)  # LuCA dim
            self.simple_stats_proj = nn.Linear(config.stats_dim, config.hidden_dim)
            self.simple_pathway_proj = nn.Linear(config.pathway_dim, config.hidden_dim)
            # Simple reconstruction head for SSL (when NicheTokenizer not used)
            self.simple_reconstruction_head = nn.Linear(config.hidden_dim, config.input_dim)

    def encode_stage_pair(self, stage_src: int, stage_tgt: int) -> int:
        """Encode a stage transition as a single integer."""
        return int(stage_src * self.config.num_stages + stage_tgt)

    def encode_stage_pair_tensor(
        self,
        stage_src: int,
        stage_tgt: int,
        n: int,
        device: torch.device,
    ) -> Tensor:
        """Create batch of stage pair indices."""
        idx = self.encode_stage_pair(stage_src, stage_tgt)
        return torch.full((n,), idx, dtype=torch.long, device=device)

    def encode_niche(
        self,
        receiver: Tensor,
        ring_cells: list[Tensor],
        ring_masks: list[Tensor],
        hlca: Tensor,
        luca: Tensor,
        pathway: Tensor | None = None,
        stats: Tensor | None = None,
        evolution_features: Tensor | None = None,
        return_reconstruction: bool = False,
        return_gw_coupling: bool = False,
    ) -> ReceiverNicheOutput:
        """Encode niche with learned ring pooling.

        Uses NicheTokenizer to pool individual cells per ring via learned
        ISAB+PMA attention, then passes the 9-token structure through
        the hierarchical set transformer.

        If use_gw_fusion is enabled, HLCA and LuCA embeddings are first
        aligned via differentiable Gromov-Wasserstein before tokenization.

        Args:
            receiver: [B, D] receiver cell embedding
            ring_cells: List of 4 tensors, each [B, max_cells, D]
            ring_masks: List of 4 tensors, each [B, max_cells] (True = valid)
            hlca: [B, 30] HLCA reference embedding
            luca: [B, 10] LuCA reference embedding
            pathway: [B, D] pathway features (optional)
            stats: [B, D] stats features (optional)
            evolution_features: [B, E] WES/genomic features (optional)
            return_reconstruction: Return receiver reconstruction for SSL
            return_gw_coupling: Return GW transport plan (for visualization)

        Returns:
            ReceiverNicheOutput with context and attention weights
        """
        # Optional: Gromov-Wasserstein fusion before tokenization
        gw_coupling = None
        gw_cost = None
        fused_ref = None
        if self.gw_fusion is not None:
            if return_gw_coupling:
                fused_ref, gw_coupling, gw_cost = self.gw_fusion(
                    hlca, luca, return_coupling=True
                )
            else:
                fused_ref = self.gw_fusion(hlca, luca)

        receiver_reconstruction = None
        ring_attention = None

        # Apply ablation masking to inputs
        B = receiver.shape[0]
        device = receiver.device

        # hlca_only / luca_only: zero out unused reference
        if not self.config.use_hlca_reference:
            hlca = torch.zeros_like(hlca)
        if not self.config.use_luca_reference:
            luca = torch.zeros_like(luca)

        # no_niche ablation: SKIP niche tokenizer entirely
        # This is a clean ablation - no learnable niche parameters
        if not self.config.use_niche_context:
            B = receiver.shape[0]
            # Build minimal token sequence without niche processing
            token_list = []
            # Receiver still gets projected (it's not "niche", it's the cell itself)
            if self.niche_tokenizer is not None:
                token_list.append(self.niche_tokenizer.token_proj(receiver))
            else:
                token_list.append(self.simple_token_proj(receiver))
            # Zero ring tokens (no niche information)
            for _ in range(4):
                token_list.append(torch.zeros(B, self.config.hidden_dim, device=device))
            # Reference tokens
            if self.niche_tokenizer is not None:
                token_list.append(self.niche_tokenizer.hlca_proj(hlca) if self.config.use_hlca_reference else torch.zeros(B, self.config.hidden_dim, device=device))
                token_list.append(self.niche_tokenizer.luca_proj(luca) if self.config.use_luca_reference else torch.zeros(B, self.config.hidden_dim, device=device))
                token_list.append(self.niche_tokenizer.pathway_proj(pathway) if pathway is not None else torch.zeros(B, self.config.hidden_dim, device=device))
                # Zero stats in no_niche ablation - stats contains niche composition (leakage)
                token_list.append(torch.zeros(B, self.config.hidden_dim, device=device))
            else:
                token_list.append(self.simple_hlca_proj(hlca) if self.config.use_hlca_reference else torch.zeros(B, self.config.hidden_dim, device=device))
                token_list.append(self.simple_luca_proj(luca) if self.config.use_luca_reference else torch.zeros(B, self.config.hidden_dim, device=device))
                token_list.append(self.simple_pathway_proj(pathway) if pathway is not None else torch.zeros(B, self.config.hidden_dim, device=device))
                # Zero stats in no_niche ablation - stats contains niche composition (leakage)
                token_list.append(torch.zeros(B, self.config.hidden_dim, device=device))

            tokens = torch.stack(token_list, dim=1)  # [B, 9, hidden_dim]
            receiver_reconstruction = None
            ring_attention = None

        elif self.niche_tokenizer is not None:
            # NicheTokenizer: raw cells per ring -> 8 or 9-token structure
            tokens, receiver_reconstruction, ring_attention = self.niche_tokenizer(
                receiver=receiver,
                ring_cells=ring_cells,
                ring_masks=ring_masks,
                hlca=hlca,
                luca=luca,
                pathway=pathway,
                stats=stats,
                fused_ref=fused_ref,  # If GW enabled, uses this; otherwise uses hlca/luca
            )
        else:
            # Fallback: simple mean pooling for each ring (no learned attention)
            # Only reaches here if use_niche_context=True and niche_tokenizer=None
            B = receiver.shape[0]
            token_list = []

            # Receiver token
            token_list.append(self.simple_token_proj(receiver))

            # Ring tokens via mean pooling
            for i, (cells, mask) in enumerate(zip(ring_cells, ring_masks)):
                # cells: [B, max_cells, D], mask: [B, max_cells]
                if mask is not None:
                    mask_expanded = mask.unsqueeze(-1).float()  # [B, max_cells, 1]
                    masked_cells = cells * mask_expanded
                    cell_sum = masked_cells.sum(dim=1)  # [B, D]
                    cell_count = mask_expanded.sum(dim=1).clamp(min=1)  # [B, 1]
                    ring_mean = cell_sum / cell_count  # [B, D]
                else:
                    ring_mean = cells.mean(dim=1)
                token_list.append(self.simple_token_proj(ring_mean))

            # Reference tokens (respect hlca_only / luca_only ablations)
            if self.config.use_hlca_reference:
                token_list.append(self.simple_hlca_proj(hlca))
            else:
                token_list.append(torch.zeros(B, self.config.hidden_dim, device=receiver.device))

            if self.config.use_luca_reference:
                token_list.append(self.simple_luca_proj(luca))
            else:
                token_list.append(torch.zeros(B, self.config.hidden_dim, device=receiver.device))

            # Pathway token (use projection if provided, else zeros)
            if pathway is not None:
                token_list.append(self.simple_pathway_proj(pathway))
            else:
                token_list.append(torch.zeros(B, self.config.hidden_dim, device=receiver.device))

            # Stats token
            if stats is not None:
                stats_input = stats[:, :self.config.stats_dim] if stats.shape[-1] > self.config.stats_dim else stats
                token_list.append(self.simple_stats_proj(stats_input))
            else:
                token_list.append(torch.zeros(B, self.config.hidden_dim, device=receiver.device))

            tokens = torch.stack(token_list, dim=1)  # [B, 9, hidden_dim]

            # SSL reconstruction from context (skip receiver token at index 0)
            context_tokens = tokens[:, 1:, :]  # [B, 8, hidden_dim]
            context_pooled = context_tokens.mean(dim=1)  # [B, hidden_dim]
            receiver_reconstruction = self.simple_reconstruction_head(context_pooled)
            ring_attention = None

        # tokens: [B, 9, hidden_dim]
        # Pass through hierarchical set transformer: ISAB -> ISAB(rpe) -> SAB -> PMA
        entropy_loss = None
        if self.context_refiner is not None:
            # Compute entropy loss during training for attention regularization
            refiner_output = self.context_refiner(tokens, return_entropy=self.training)
            if self.training and len(refiner_output) == 3:
                context, context_tokens, entropy_loss = refiner_output
            else:
                context, context_tokens = refiner_output[:2]
        else:
            context = tokens.mean(dim=1)
            context_tokens = tokens

        # Neighborhood-level prototype bottleneck (interpretable niche archetypes)
        niche_prototype_output = None
        if self.niche_prototype_bottleneck is not None:
            niche_prototype_output = self.niche_prototype_bottleneck(context.unsqueeze(1))
            context = niche_prototype_output.aligned_embeddings.squeeze(1)

        # Stats conditioning (if enabled and provided)
        if self.stats_conditioner is not None and stats is not None:
            if stats.shape[-1] != self.config.stats_dim:
                stats_cond = stats[:, :self.config.stats_dim]
            else:
                stats_cond = stats
            context = self.stats_conditioner(context, stats_cond)

        # Evolution branch: WES/genomic feature conditioning
        if self.evolution_branch is not None and evolution_features is not None:
            context, _ = self.evolution_branch(context, evolution_features)

        return ReceiverNicheOutput(
            context=context,
            context_tokens=context_tokens,
            attention_weights=None,
            entropy_loss=entropy_loss,
            value_l1_loss=None,
            empty_attention=None,
            receiver_reconstruction=receiver_reconstruction if return_reconstruction else None,
            niche_prototype_composition=niche_prototype_output.prototype_composition if niche_prototype_output else None,
        )

    def encode_niche_amici(
        self,
        receiver: Tensor,
        neighbors: Tensor,
        distances: Tensor,
        neighbor_mask: Tensor,
        hlca: Tensor,
        luca: Tensor,
        pathway: Tensor | None = None,
        stats: Tensor | None = None,
        evolution_features: Tensor | None = None,
        return_reconstruction: bool = False,
    ) -> ReceiverNicheOutput:
        """Encode niche with AMICI-style continuous distance attention.

        Uses ReceiverCenteredNicheEncoder with learned distance decay per head.
        Each head learns its own effective interaction range from data.

        Args:
            receiver: [B, D] receiver cell embedding
            neighbors: [B, K, D] neighbor embeddings (sorted by distance)
            distances: [B, K] distances in microns
            neighbor_mask: [B, K] True = valid neighbor
            hlca: [B, 30] HLCA reference embedding
            luca: [B, 10] LuCA reference embedding
            pathway: [B, D] pathway features (optional)
            stats: [B, stats_dim] stats features (optional)
            evolution_features: [B, E] WES/genomic features (optional)
            return_reconstruction: Return receiver reconstruction for SSL

        Returns:
            ReceiverNicheOutput with context, attention, and AMICI-specific losses
        """
        if self.amici_encoder is None:
            raise RuntimeError("AMICI encoder not enabled. Set use_amici_attention=True")

        B = receiver.shape[0]
        device = receiver.device

        # hlca_only / luca_only ablations
        if not self.config.use_hlca_reference:
            hlca = torch.zeros_like(hlca)
        if not self.config.use_luca_reference:
            luca = torch.zeros_like(luca)

        # no_niche ablation: SKIP encoder entirely, use zero niche context
        # This is a clean ablation - no learnable parameters from niche processing
        # We still provide a reconstruction (from zero context) so SSL can run,
        # but it will perform poorly, demonstrating the value of niche info
        if not self.config.use_niche_context:
            niche_context = torch.zeros(B, self.config.hidden_dim, device=device)
            # Reconstruction from zero context (will be bad - proves niche is needed)
            reconstruction = None
            if return_reconstruction and self.no_niche_reconstruction_head is not None:
                reconstruction = self.no_niche_reconstruction_head(niche_context)
            amici_output = ReceiverNicheOutput(
                context=niche_context,
                context_tokens=None,
                attention_weights=None,
                entropy_loss=None,
                value_l1_loss=None,
                empty_attention=None,
                receiver_reconstruction=reconstruction,
            )
        else:
            # Run AMICI encoder (receiver-centered attention with distance decay)
            amici_output = self.amici_encoder(
                receiver=receiver,
                neighbors=neighbors,
                distances=distances,
                neighbor_mask=neighbor_mask,
                return_reconstruction=return_reconstruction,
            )
            niche_context = amici_output.context  # [B, hidden_dim]

        # Build token sequence for context refiner (keep compatible with downstream)
        # Tokens: [niche_context, hlca, luca, pathway, stats]
        token_list = [niche_context]

        # Reference tokens
        if self.config.use_hlca_reference:
            token_list.append(self.amici_hlca_proj(hlca))
        else:
            token_list.append(torch.zeros(B, self.config.hidden_dim, device=device))

        if self.config.use_luca_reference:
            token_list.append(self.amici_luca_proj(luca))
        else:
            token_list.append(torch.zeros(B, self.config.hidden_dim, device=device))

        # Pathway token
        if pathway is not None:
            token_list.append(self.amici_pathway_proj(pathway))
        else:
            token_list.append(torch.zeros(B, self.config.hidden_dim, device=device))

        # Stats token
        # NOTE: stats contains niche composition (caf_fraction, immune_fraction, diversity)
        # For no_niche ablation, zero out stats to prevent leakage of niche info
        if stats is not None and self.config.use_niche_context:
            stats_input = stats[:, :self.config.stats_dim] if stats.shape[-1] > self.config.stats_dim else stats
            token_list.append(self.amici_stats_proj(stats_input))
        else:
            token_list.append(torch.zeros(B, self.config.hidden_dim, device=device))

        tokens = torch.stack(token_list, dim=1)  # [B, 5, hidden_dim]

        # Context refinement (optional)
        entropy_loss = None
        if self.context_refiner is not None:
            refiner_output = self.context_refiner(tokens, return_entropy=self.training)
            if self.training and len(refiner_output) == 3:
                context, context_tokens, entropy_loss = refiner_output
            else:
                context, context_tokens = refiner_output[:2]
        else:
            context = tokens.mean(dim=1)
            context_tokens = tokens

        # Prototype bottleneck (optional)
        niche_prototype_output = None
        if self.niche_prototype_bottleneck is not None:
            niche_prototype_output = self.niche_prototype_bottleneck(context.unsqueeze(1))
            context = niche_prototype_output.aligned_embeddings.squeeze(1)

        # Stats conditioning (optional)
        if self.stats_conditioner is not None and stats is not None:
            if stats.shape[-1] != self.config.stats_dim:
                stats_cond = stats[:, :self.config.stats_dim]
            else:
                stats_cond = stats
            context = self.stats_conditioner(context, stats_cond)

        # Evolution branch (optional)
        if self.evolution_branch is not None and evolution_features is not None:
            context, _ = self.evolution_branch(context, evolution_features)

        return ReceiverNicheOutput(
            context=context,
            context_tokens=context_tokens,
            attention_weights=amici_output.attention_weights,
            entropy_loss=entropy_loss,
            value_l1_loss=amici_output.value_l1_loss,
            empty_attention=amici_output.empty_attention,
            receiver_reconstruction=amici_output.receiver_reconstruction if return_reconstruction else None,
            niche_prototype_composition=niche_prototype_output.prototype_composition if niche_prototype_output else None,
        )

    def aggregate_niches(
        self,
        niche_embeddings: Tensor,
        mask: Tensor | None = None,
        return_attention: bool = False,
    ) -> dict:
        """Aggregate multiple niche embeddings to sample-level.

        Args:
            niche_embeddings: [B, N, D] batch of N niche contexts per sample
            mask: [B, N] boolean mask (True = valid niche)
            return_attention: Return attention weights

        Returns:
            dict with sample_embedding and optional attention_weights
        """
        if self.hierarchical_aggregator is None:
            raise RuntimeError("Hierarchical aggregation not enabled in config")
        return self.hierarchical_aggregator(
            niche_embeddings=niche_embeddings,
            mask=mask,
            return_attention=return_attention,
        )

    def predict_sample_outputs(self, sample_embedding: Tensor) -> dict:
        """Predict sample-level outputs.

        Args:
            sample_embedding: [B, D] aggregated sample representation

        Returns:
            dict with stage_logits and displacement
        """
        if self.sample_heads is None:
            raise RuntimeError("Sample heads not enabled in config")
        return self.sample_heads(sample_embedding)

    def forward_vector_field(
        self,
        x_t: Tensor,
        t: Tensor,
        context: Tensor,
        stage_pair_id: Tensor,
        context_tokens: Tensor | None = None,
        **kwargs: object,
    ) -> Tensor:
        """Predict drift velocity at state x_t and time t.

        Args:
            x_t: [B, D] current state
            t: [B] time in [0, 1]
            context: [B, C] niche context vector
            stage_pair_id: [B] stage transition indices
            context_tokens: [B, K, C] context tokens (for cross-attention)

        Returns:
            [B, D] drift velocity
        """
        if context.ndim == 1:
            context = context.unsqueeze(0)
        if context.shape[0] == 1 and x_t.shape[0] > 1:
            context = context.expand(x_t.shape[0], -1)
        if stage_pair_id.ndim == 0:
            stage_pair_id = stage_pair_id.repeat(x_t.shape[0])

        stage_emb = self.stage_embedding(stage_pair_id)
        time_emb = self.time_embedding(t)

        # Compute learned drift
        if self.config.use_cross_attn_drift:
            if context_tokens is None:
                context_tokens = context.unsqueeze(1)
            learned_drift = self.drift_head(x_t, time_emb, context_tokens, stage_emb)
        else:
            cond = torch.cat([context, stage_emb], dim=-1)
            x_mod = self.film(x_t, cond)
            inp = torch.cat([x_mod, time_emb, context, stage_emb], dim=-1)
            learned_drift = self.drift_head(inp)

        return learned_drift

    def integrate_euler(
        self,
        x0: Tensor,
        context: Tensor,
        stage_pair_id: Tensor,
        num_steps: int = 8,
        context_tokens: Tensor | None = None,
    ) -> Tensor:
        """Integrate velocity field from t=0 to t=1 via Euler method.

        Args:
            x0: [B, D] initial state
            context: [B, C] niche context vector
            stage_pair_id: [B] stage transition indices
            num_steps: Number of Euler steps
            context_tokens: [B, K, C] context tokens

        Returns:
            [B, D] final state at t=1
        """
        x = x0
        dt = 1.0 / float(num_steps)
        for k in range(num_steps):
            t = torch.full((x.shape[0],), (k + 0.5) * dt, device=x.device, dtype=x.dtype)
            v = self.forward_vector_field(
                x_t=x,
                t=t,
                context=context,
                stage_pair_id=stage_pair_id,
                context_tokens=context_tokens,
            )
            x = x + dt * v
        return x

    def integrate_euler_maruyama(
        self,
        x0: Tensor,
        context: Tensor,
        stage_pair_id: Tensor,
        num_steps: int = 8,
        sigma: float = 0.0,
        context_tokens: Tensor | None = None,
    ) -> Tensor:
        """Euler-Maruyama integration for stochastic dynamics.

        Args:
            x0: [B, D] initial state
            context: [B, C] niche context vector
            stage_pair_id: [B] stage transition indices
            num_steps: Number of integration steps
            sigma: Noise level (0 = deterministic Euler)
            context_tokens: [B, K, C] context tokens

        Returns:
            [B, D] final state
        """
        x = x0
        dt = 1.0 / float(num_steps)
        sqrt_dt = dt**0.5
        for k in range(num_steps):
            t = torch.full((x.shape[0],), (k + 0.5) * dt, device=x.device, dtype=x.dtype)
            v = self.forward_vector_field(
                x_t=x,
                t=t,
                context=context,
                stage_pair_id=stage_pair_id,
                context_tokens=context_tokens,
            )
            x = x + dt * v
            if sigma > 0.0:
                x = x + sigma * sqrt_dt * torch.randn_like(x)
        return x

    def integrate_rk4(
        self,
        x0: Tensor,
        context: Tensor,
        stage_pair_id: Tensor,
        num_steps: int = 8,
        context_tokens: Tensor | None = None,
    ) -> Tensor:
        """Integrate velocity field via 4th-order Runge-Kutta.

        More accurate than Euler for smooth vector fields.

        Args:
            x0: [B, D] initial state
            context: [B, C] niche context vector
            stage_pair_id: [B] stage transition indices
            num_steps: Number of RK4 steps
            context_tokens: [B, K, C] context tokens

        Returns:
            [B, D] final state at t=1
        """
        x = x0
        dt = 1.0 / float(num_steps)

        def vf(x_t: Tensor, t_val: float) -> Tensor:
            t = torch.full((x_t.shape[0],), t_val, device=x_t.device, dtype=x_t.dtype)
            return self.forward_vector_field(
                x_t=x_t,
                t=t,
                context=context,
                stage_pair_id=stage_pair_id,
                context_tokens=context_tokens,
            )

        for k in range(num_steps):
            t_k = k * dt
            k1 = vf(x, t_k)
            k2 = vf(x + 0.5 * dt * k1, t_k + 0.5 * dt)
            k3 = vf(x + 0.5 * dt * k2, t_k + 0.5 * dt)
            k4 = vf(x + dt * k3, t_k + dt)
            x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        return x

    def sample_trajectory(
        self,
        x0: Tensor,
        context: Tensor,
        stage_pair_id: Tensor,
        num_steps: int = 8,
        context_tokens: Tensor | None = None,
        sigma: float = 0.0,
    ) -> Tensor:
        """Return full trajectory [B, num_steps+1, D] from t=0 to t=1."""
        trajectory = [x0]
        x = x0
        dt = 1.0 / float(num_steps)
        sqrt_dt = dt**0.5

        for k in range(num_steps):
            t = torch.full((x.shape[0],), (k + 0.5) * dt, device=x.device, dtype=x.dtype)
            v = self.forward_vector_field(
                x_t=x,
                t=t,
                context=context,
                stage_pair_id=stage_pair_id,
                context_tokens=context_tokens,
            )
            x = x + dt * v
            if sigma > 0.0:
                x = x + sigma * sqrt_dt * torch.randn_like(x)
            trajectory.append(x)

        return torch.stack(trajectory, dim=1)

    def sample_forward(
        self,
        niche_embeddings: Tensor,
        mask: Tensor | None = None,
        return_attention: bool = False,
    ) -> dict:
        """Sample-level forward pass for batch of niches.

        Args:
            niche_embeddings: [B, N, D] batch of N niche contexts per sample
            mask: [B, N] boolean mask
            return_attention: Return attention weights

        Returns:
            dict with sample_embedding, stage_logits, displacement, attention_weights
        """
        agg_output = self.aggregate_niches(
            niche_embeddings=niche_embeddings,
            mask=mask,
            return_attention=return_attention,
        )

        sample_embedding = agg_output["sample_embedding"]

        if self.sample_heads is not None:
            head_output = self.sample_heads(sample_embedding)
            return {
                "sample_embedding": sample_embedding,
                "stage_logits": head_output["stage_logits"],
                "displacement": head_output["displacement"],
                "attention_weights": agg_output.get("attention_weights"),
            }

        return {
            "sample_embedding": sample_embedding,
            "stage_logits": None,
            "displacement": None,
            "attention_weights": agg_output.get("attention_weights"),
        }
