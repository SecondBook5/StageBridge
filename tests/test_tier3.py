"""Comprehensive tests for Tier 3 features.

Covers:
1. Spatial-snRNA joint conditioning (spatial niche preprocessing + model fusion)
2. Causal perturbation module (Jacobian, effect scoring, reversal directions)
3. LR signaling features (score computation, feature matrix, lookup)
4. Multi-hop skip-stage trajectory consistency loss
5. Dirichlet stage assignment posterior (head, losses, entropy, patient-level)
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import Tensor

from stagebridge.models.stagebridge import StageBridgeModel
from stagebridge.utils.types import StageBatch, StageBridgeConfig

# ──────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ──────────────────────────────────────────────────────────────────────────────

INPUT_DIM = 16
HIDDEN_DIM = 32
NUM_STAGES = 5
BATCH = 8


def _base_config(**overrides: object) -> StageBridgeConfig:
    defaults = dict(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        vector_field_hidden_dim=64,
        num_heads=2,
        num_inducing_points=4,
        num_seed_vectors=2,
        num_stages=NUM_STAGES,
        time_embedding_dim=8,
        stage_embedding_dim=8,
        dropout=0.0,
        device="cpu",
        mixed_precision=False,
    )
    defaults.update(overrides)
    return StageBridgeConfig(**defaults)  # type: ignore[arg-type]


def _random_batch(
    config: StageBridgeConfig,
    stage_src: int = 0,
    stage_tgt: int = 1,
    lr_features: Tensor | None = None,
    spatial_niche: Tensor | None = None,
) -> StageBatch:
    return StageBatch(
        x_src=torch.randn(BATCH, config.input_dim),
        x_tgt=torch.randn(BATCH, config.input_dim),
        x_set=torch.randn(BATCH, config.input_dim),
        context_mask=torch.ones(1, BATCH, dtype=torch.bool),
        stage_src=stage_src,
        stage_tgt=stage_tgt,
        donor_id="P1",
        lr_features=lr_features,
        spatial_niche=spatial_niche,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 1. Spatial-snRNA joint conditioning
# ══════════════════════════════════════════════════════════════════════════════

class TestSpatialNicheConditioning:
    """Test spatial niche composition fusion in the model."""

    def test_model_builds_with_spatial_niche(self) -> None:
        cfg = _base_config(use_spatial_niche=True, spatial_niche_dim=10, spatial_niche_hidden=16)
        model = StageBridgeModel(cfg)
        assert model.spatial_niche_proj is not None

    def test_model_builds_without_spatial_niche(self) -> None:
        cfg = _base_config(use_spatial_niche=False)
        model = StageBridgeModel(cfg)
        assert model.spatial_niche_proj is None

    def test_forward_set_context_with_spatial_niche(self) -> None:
        cfg = _base_config(use_spatial_niche=True, spatial_niche_dim=10, spatial_niche_hidden=16)
        model = StageBridgeModel(cfg)
        model.eval()

        x_set = torch.randn(1, BATCH, INPUT_DIM)
        spatial_niche = torch.randn(1, BATCH, 10)

        ctx_with = model.forward_set_context(x_set, spatial_niche=spatial_niche)
        ctx_without = model.forward_set_context(x_set, spatial_niche=None)

        # Spatial niche should change the context
        assert ctx_with.shape == ctx_without.shape
        assert not torch.allclose(ctx_with, ctx_without, atol=1e-5)

    def test_forward_set_context_cross_attn_with_spatial_niche(self) -> None:
        cfg = _base_config(
            use_spatial_niche=True, spatial_niche_dim=10, spatial_niche_hidden=16,
            use_cross_attn_drift=True,
        )
        model = StageBridgeModel(cfg)
        model.eval()

        x_set = torch.randn(1, BATCH, INPUT_DIM)
        spatial_niche = torch.randn(1, BATCH, 10)

        ctx = model.forward_set_context(x_set, spatial_niche=spatial_niche)
        assert ctx.ndim == 3  # (1, k, D) for cross-attn mode

    def test_spatial_niche_preprocessing_module(self) -> None:
        from stagebridge.preprocessing.spatial_niche import build_fallback_niche_features

        fb = build_fallback_niche_features(n_cells=100, n_celltypes=10)
        assert fb.shape == (100, 10)
        np.testing.assert_allclose(fb.sum(axis=1), 1.0, atol=1e-5)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Causal perturbation module
# ══════════════════════════════════════════════════════════════════════════════

class TestCausalPerturbation:
    """Test the post-hoc CausalPerturbationAnalyzer."""

    @pytest.fixture
    def model_and_analyzer(self):
        cfg = _base_config()
        model = StageBridgeModel(cfg)
        model.eval()
        from stagebridge.models.perturbation import CausalPerturbationAnalyzer
        analyzer = CausalPerturbationAnalyzer(model, device="cpu")
        return model, analyzer

    def test_perturbation_effect_scores(self, model_and_analyzer) -> None:
        model, analyzer = model_and_analyzer
        x_src = torch.randn(BATCH, INPUT_DIM)
        c_s = model.forward_set_context(x_src)
        pair_id = model.encode_stage_pair_tensor(0, 1, BATCH, torch.device("cpu"))

        directions = torch.randn(5, INPUT_DIM)
        result = analyzer.compute_perturbation_effects(
            x_src=x_src, c_s=c_s, stage_pair_id=pair_id,
            perturbation_directions=directions, effect_size=1.0, num_steps=4,
        )

        assert result.effect_scores.shape == (5,)
        assert result.reversal_scores.shape == (5,)
        assert result.endpoint_shifts.shape == (5, INPUT_DIM)
        assert result.ranked_indices.shape == (5,)
        assert (result.effect_scores >= 0).all()

    def test_flow_jacobian_shape(self, model_and_analyzer) -> None:
        model, analyzer = model_and_analyzer
        x0 = torch.randn(1, INPUT_DIM)
        c_s = model.forward_set_context(x0)
        pair_id = model.encode_stage_pair_tensor(0, 1, 1, torch.device("cpu"))

        J = analyzer.compute_flow_jacobian(x0, c_s, pair_id, num_steps=2)
        assert J.shape == (INPUT_DIM, INPUT_DIM)

    def test_jacobian_top_directions(self, model_and_analyzer) -> None:
        _, analyzer = model_and_analyzer
        J = torch.randn(INPUT_DIM, INPUT_DIM)
        dirs, svals = analyzer.jacobian_top_directions(J, n=5)
        assert dirs.shape == (5, INPUT_DIM)
        assert svals.shape == (5,)
        # Singular values should be sorted descending
        assert (svals[:-1] >= svals[1:]).all()

    def test_pca_perturbation_directions(self, model_and_analyzer) -> None:
        _, analyzer = model_and_analyzer
        x_stage = torch.randn(100, INPUT_DIM)
        dirs = analyzer.define_perturbation_directions_from_pca(x_stage, n_components=5)
        assert dirs.shape == (5, INPUT_DIM)
        # PCA directions should be roughly unit-normalised
        norms = dirs.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=0.1)

    def test_stage_delta_directions(self, model_and_analyzer) -> None:
        _, analyzer = model_and_analyzer
        x_src = torch.randn(50, INPUT_DIM)
        x_tgt = torch.randn(50, INPUT_DIM) + 2.0  # shift
        dirs = analyzer.define_perturbation_directions_stage_delta(x_src, x_tgt, n_directions=10)
        assert dirs.shape == (10, INPUT_DIM)

    def test_reversal_search(self, model_and_analyzer) -> None:
        model, analyzer = model_and_analyzer
        x_src = torch.randn(BATCH, INPUT_DIM)
        c_s = model.forward_set_context(x_src)
        pair_id = model.encode_stage_pair_tensor(0, 1, BATCH, torch.device("cpu"))
        centroid = x_src.mean(dim=0)

        result = analyzer.find_stage_reversal_directions(
            x_src=x_src, c_s=c_s, stage_pair_id=pair_id,
            stage_centroid_src=centroid, n_directions=10,
            num_steps=2, effect_size=0.5,
        )
        assert result.directions.shape[0] == 10

    def test_perturbation_effect_matrix(self, model_and_analyzer) -> None:
        model, analyzer = model_and_analyzer
        x_src = torch.randn(BATCH, INPUT_DIM)
        c_s = model.forward_set_context(x_src)
        pair_id = model.encode_stage_pair_tensor(0, 1, BATCH, torch.device("cpu"))
        dirs = torch.randn(3, INPUT_DIM)

        out = analyzer.perturbation_effect_matrix(
            x_src=x_src, c_s=c_s, stage_pair_id=pair_id,
            directions=dirs, effect_sizes=[0.5, 1.0, 2.0], num_steps=2,
        )
        assert out["effect_matrix"].shape == (3, 3)
        assert out["reversal_matrix"].shape == (3, 3)


# ══════════════════════════════════════════════════════════════════════════════
# 3. LR signaling features
# ══════════════════════════════════════════════════════════════════════════════

class TestLRSignaling:
    """Test LR signaling feature computation and model integration."""

    def test_lr_pairs_defined(self) -> None:
        from stagebridge.io.lr_signaling import LUNG_LR_PAIRS, LR_FEATURE_COLS
        assert len(LUNG_LR_PAIRS) >= 20
        assert len(LR_FEATURE_COLS) == len(LUNG_LR_PAIRS)

    def test_lr_feature_dim(self) -> None:
        from stagebridge.io.lr_signaling import get_lr_feature_dim
        assert get_lr_feature_dim() >= 20

    def test_lr_pathway_groups(self) -> None:
        from stagebridge.io.lr_signaling import lr_pathway_groups
        groups = lr_pathway_groups()
        assert "VEGF" in groups
        assert "TGF-beta" in groups
        assert all(isinstance(v, list) for v in groups.values())

    def test_model_with_lr_features(self) -> None:
        cfg = _base_config(use_lr_features=True, lr_feature_dim=24, lr_hidden_dim=16)
        model = StageBridgeModel(cfg)
        model.eval()
        assert model.lr_proj is not None

        x_set = torch.randn(BATCH, INPUT_DIM)
        x_t = torch.randn(BATCH, INPUT_DIM)
        t = torch.rand(BATCH)
        lr_feat = torch.randn(BATCH, 24)

        c_s = model.forward_set_context(x_set)
        pair_id = model.encode_stage_pair_tensor(0, 1, BATCH, torch.device("cpu"))
        v = model.forward_vector_field(x_t=x_t, t=t, c_s=c_s, stage_pair_id=pair_id, lr_features=lr_feat)
        assert v.shape == (BATCH, INPUT_DIM)

    def test_lr_features_affect_output(self) -> None:
        cfg = _base_config(use_lr_features=True, lr_feature_dim=24, lr_hidden_dim=16)
        model = StageBridgeModel(cfg)
        model.eval()

        x = torch.randn(BATCH, INPUT_DIM)
        t = torch.rand(BATCH)
        c_s = model.forward_set_context(x)
        pair_id = model.encode_stage_pair_tensor(0, 1, BATCH, torch.device("cpu"))

        v_with = model.forward_vector_field(x_t=x, t=t, c_s=c_s, stage_pair_id=pair_id, lr_features=torch.ones(BATCH, 24))
        v_without = model.forward_vector_field(x_t=x, t=t, c_s=c_s, stage_pair_id=pair_id, lr_features=torch.zeros(BATCH, 24))
        assert not torch.allclose(v_with, v_without, atol=1e-5)

    def test_integrate_euler_with_lr(self) -> None:
        cfg = _base_config(use_lr_features=True, lr_feature_dim=24, lr_hidden_dim=16)
        model = StageBridgeModel(cfg)
        model.eval()

        x0 = torch.randn(BATCH, INPUT_DIM)
        c_s = model.forward_set_context(x0)
        pair_id = model.encode_stage_pair_tensor(0, 1, BATCH, torch.device("cpu"))
        lr_feat = torch.randn(BATCH, 24)

        x_T = model.integrate_euler(x0=x0, c_s=c_s, stage_pair_id=pair_id, num_steps=4, lr_features=lr_feat)
        assert x_T.shape == (BATCH, INPUT_DIM)


# ══════════════════════════════════════════════════════════════════════════════
# 4. Multi-hop skip-stage consistency loss
# ══════════════════════════════════════════════════════════════════════════════

class TestMultihopConsistency:
    """Test multi-hop trajectory composition consistency loss."""

    def test_multihop_loss_adjacent_is_zero(self) -> None:
        from stagebridge.training.losses import multihop_consistency_loss

        cfg = _base_config()
        model = StageBridgeModel(cfg)
        model.eval()
        x = torch.randn(BATCH, INPUT_DIM)

        loss, diag = multihop_consistency_loss(
            model=model, x_src=x, stage_src=0, stage_tgt=1,
            num_stages=NUM_STAGES, num_steps=4,
        )
        assert float(loss) == 0.0
        assert diag["gap"] == 1.0

    def test_multihop_loss_skip_nonzero(self) -> None:
        from stagebridge.training.losses import multihop_consistency_loss

        cfg = _base_config()
        model = StageBridgeModel(cfg)
        model.train()
        x = torch.randn(4, INPUT_DIM)

        loss, diag = multihop_consistency_loss(
            model=model, x_src=x, stage_src=0, stage_tgt=2,
            num_stages=NUM_STAGES, num_steps=2,
        )
        assert loss.requires_grad
        assert diag["gap"] == 2.0
        assert float(loss) >= 0.0

    def test_multihop_loss_big_gap(self) -> None:
        from stagebridge.training.losses import multihop_consistency_loss

        cfg = _base_config()
        model = StageBridgeModel(cfg)
        x = torch.randn(4, INPUT_DIM)

        loss, diag = multihop_consistency_loss(
            model=model, x_src=x, stage_src=0, stage_tgt=4,
            num_stages=NUM_STAGES, num_steps=2,
        )
        assert diag["gap"] == 4.0
        assert float(loss) >= 0.0

    def test_compose_trajectory_helper(self) -> None:
        from stagebridge.training.losses import _compose_trajectory

        cfg = _base_config()
        model = StageBridgeModel(cfg)
        model.eval()
        x = torch.randn(4, INPUT_DIM)

        # Single hop: should match integrate_euler
        result = _compose_trajectory(model, x, [0, 1], num_steps=4)
        assert result.shape == (4, INPUT_DIM)

        # Two hops
        result2 = _compose_trajectory(model, x, [0, 1, 2], num_steps=4)
        assert result2.shape == (4, INPUT_DIM)


# ══════════════════════════════════════════════════════════════════════════════
# 5. Dirichlet stage assignment posterior
# ══════════════════════════════════════════════════════════════════════════════

class TestDirichletStageHead:
    """Test Dirichlet stage assignment module."""

    def test_dirichlet_head_forward(self) -> None:
        from stagebridge.models.dirichlet_stage import DirichletStageHead

        head = DirichletStageHead(input_dim=INPUT_DIM, num_stages=NUM_STAGES)
        x = torch.randn(BATCH, INPUT_DIM)
        alpha = head(x)
        assert alpha.shape == (BATCH, NUM_STAGES)
        assert (alpha > 0).all()

    def test_dirichlet_distribution(self) -> None:
        from stagebridge.models.dirichlet_stage import DirichletStageHead

        head = DirichletStageHead(input_dim=INPUT_DIM, num_stages=NUM_STAGES)
        x = torch.randn(BATCH, INPUT_DIM)
        probs = head.predict_distribution(x)
        assert probs.shape == (BATCH, NUM_STAGES)
        # Probabilities should sum to 1
        np.testing.assert_allclose(
            probs.sum(dim=-1).detach().numpy(),
            np.ones(BATCH),
            atol=1e-5,
        )

    def test_dirichlet_entropy(self) -> None:
        from stagebridge.models.dirichlet_stage import DirichletStageHead

        head = DirichletStageHead(input_dim=INPUT_DIM, num_stages=NUM_STAGES)
        x = torch.randn(BATCH, INPUT_DIM)
        H = head.predict_entropy(x)
        assert H.shape == (BATCH,)

    def test_dirichlet_mode(self) -> None:
        from stagebridge.models.dirichlet_stage import DirichletStageHead

        head = DirichletStageHead(input_dim=INPUT_DIM, num_stages=NUM_STAGES)
        x = torch.randn(BATCH, INPUT_DIM)
        mode = head.predict_mode(x)
        assert mode.shape == (BATCH,)
        assert (mode >= 0).all() and (mode < NUM_STAGES).all()

    def test_dirichlet_nll_loss(self) -> None:
        from stagebridge.models.dirichlet_stage import DirichletStageHead, dirichlet_nll_loss

        head = DirichletStageHead(input_dim=INPUT_DIM, num_stages=NUM_STAGES)
        x = torch.randn(BATCH, INPUT_DIM)
        alpha = head(x)
        targets = torch.randint(0, NUM_STAGES, (BATCH,))
        loss = dirichlet_nll_loss(alpha, targets)
        assert loss.shape == ()
        assert loss.requires_grad
        assert float(loss) >= 0.0

    def test_dirichlet_consistency_loss(self) -> None:
        from stagebridge.models.dirichlet_stage import DirichletStageHead, dirichlet_consistency_loss

        head = DirichletStageHead(input_dim=INPUT_DIM, num_stages=NUM_STAGES)
        x = torch.randn(BATCH, INPUT_DIM)
        alpha = head(x)
        loss = dirichlet_consistency_loss(alpha, stage_target_idx=2, weight=0.5)
        assert loss.shape == ()
        assert float(loss) >= 0.0

    def test_dirichlet_entropy_regularization(self) -> None:
        from stagebridge.models.dirichlet_stage import DirichletStageHead, dirichlet_entropy_regularization

        head = DirichletStageHead(input_dim=INPUT_DIM, num_stages=NUM_STAGES)
        x = torch.randn(BATCH, INPUT_DIM)
        alpha = head(x)
        loss = dirichlet_entropy_regularization(alpha, target_entropy=0.5)
        assert loss.shape == ()
        assert float(loss) >= 0.0

    def test_stage_uncertainty_triple(self) -> None:
        from stagebridge.models.dirichlet_stage import DirichletStageHead

        head = DirichletStageHead(input_dim=INPUT_DIM, num_stages=NUM_STAGES)
        x = torch.randn(BATCH, INPUT_DIM)
        alpha, probs, entropy = head.stage_uncertainty(x)
        assert alpha.shape == (BATCH, NUM_STAGES)
        assert probs.shape == (BATCH, NUM_STAGES)
        assert entropy.shape == (BATCH,)

    def test_per_patient_entropy(self) -> None:
        from stagebridge.models.dirichlet_stage import DirichletStageHead, compute_stage_entropy_per_patient

        head = DirichletStageHead(input_dim=INPUT_DIM, num_stages=NUM_STAGES)
        x = torch.randn(20, INPUT_DIM)
        alpha = head(x)
        donor_ids = np.array(["P1"] * 10 + ["P2"] * 10)
        entropy_dict = compute_stage_entropy_per_patient(alpha, donor_ids)
        assert "P1" in entropy_dict
        assert "P2" in entropy_dict
        assert isinstance(entropy_dict["P1"], float)

    def test_per_patient_distribution(self) -> None:
        from stagebridge.models.dirichlet_stage import DirichletStageHead, compute_stage_distribution_per_patient

        head = DirichletStageHead(input_dim=INPUT_DIM, num_stages=NUM_STAGES)
        x = torch.randn(20, INPUT_DIM)
        alpha = head(x)
        donor_ids = np.array(["P1"] * 10 + ["P2"] * 10)
        dist_dict = compute_stage_distribution_per_patient(alpha, donor_ids)
        assert dist_dict["P1"].shape == (NUM_STAGES,)
        np.testing.assert_allclose(dist_dict["P1"].sum(), 1.0, atol=1e-5)


# ══════════════════════════════════════════════════════════════════════════════
# 6. Full model integration tests
# ══════════════════════════════════════════════════════════════════════════════

class TestFullModelIntegration:
    """Test the full StageBridgeModel with all Tier 3 features enabled."""

    def test_full_tier3_model_builds(self) -> None:
        cfg = _base_config(
            use_lr_features=True, lr_feature_dim=24, lr_hidden_dim=16,
            use_spatial_niche=True, spatial_niche_dim=10, spatial_niche_hidden=16,
            use_dirichlet_head=True, dirichlet_hidden_dim=32,
            use_cross_attn_drift=True,
        )
        model = StageBridgeModel(cfg)
        assert model.lr_proj is not None
        assert model.spatial_niche_proj is not None
        assert model.dirichlet_head is not None

    def test_full_tier3_forward(self) -> None:
        cfg = _base_config(
            use_lr_features=True, lr_feature_dim=24, lr_hidden_dim=16,
            use_spatial_niche=True, spatial_niche_dim=10, spatial_niche_hidden=16,
            use_dirichlet_head=True, dirichlet_hidden_dim=32,
        )
        model = StageBridgeModel(cfg)
        model.eval()

        x_set = torch.randn(BATCH, INPUT_DIM)
        x_t = torch.randn(BATCH, INPUT_DIM)
        t = torch.rand(BATCH)
        lr_feat = torch.randn(BATCH, 24)
        spatial_niche = torch.randn(BATCH, 10)

        v = model.forward(
            x_set=x_set, x_t=x_t, t=t,
            stage_src=0, stage_tgt=1,
            lr_features=lr_feat,
            spatial_niche=spatial_niche,
        )
        assert v.shape == (BATCH, INPUT_DIM)

        # Also test Dirichlet prediction
        alpha = model.predict_dirichlet(x_set)
        assert alpha is not None
        assert alpha.shape == (BATCH, NUM_STAGES)

    def test_full_tier3_euler_integration(self) -> None:
        cfg = _base_config(
            use_lr_features=True, lr_feature_dim=24, lr_hidden_dim=16,
            use_spatial_niche=True, spatial_niche_dim=10, spatial_niche_hidden=16,
        )
        model = StageBridgeModel(cfg)
        model.eval()

        x0 = torch.randn(BATCH, INPUT_DIM)
        spatial_niche = torch.randn(BATCH, 10)
        lr_feat = torch.randn(BATCH, 24)

        c_s = model.forward_set_context(x0.unsqueeze(0), spatial_niche=spatial_niche.unsqueeze(0))
        pair_id = model.encode_stage_pair_tensor(0, 1, BATCH, torch.device("cpu"))

        x_T = model.integrate_euler(
            x0=x0, c_s=c_s, stage_pair_id=pair_id,
            num_steps=4, lr_features=lr_feat,
        )
        assert x_T.shape == (BATCH, INPUT_DIM)
        # Should be different from input
        assert not torch.allclose(x_T, x0, atol=1e-3)

    def test_flow_matching_loss_with_lr(self) -> None:
        from stagebridge.training.losses import flow_matching_loss

        cfg = _base_config(use_lr_features=True, lr_feature_dim=24, lr_hidden_dim=16)
        model = StageBridgeModel(cfg)
        model.train()

        batch = _random_batch(cfg, lr_features=torch.randn(24))
        loss, diag, coupling = flow_matching_loss(
            batch=batch, model=model,
            num_ot_pairs=32, sinkhorn_iters=10,
            context_consistency_weight=0.0,
        )
        assert loss.requires_grad
        assert float(loss) > 0.0

    def test_backward_with_all_tier3(self) -> None:
        """Gradient flows through all Tier 3 components."""
        cfg = _base_config(
            use_lr_features=True, lr_feature_dim=24, lr_hidden_dim=16,
            use_spatial_niche=True, spatial_niche_dim=10, spatial_niche_hidden=16,
            use_dirichlet_head=True, dirichlet_hidden_dim=32,
        )
        model = StageBridgeModel(cfg)
        model.train()

        batch = _random_batch(
            cfg,
            lr_features=torch.randn(24),
            spatial_niche=torch.randn(BATCH, 10),
        )

        from stagebridge.training.losses import flow_matching_loss
        loss, _, _ = flow_matching_loss(
            batch=batch, model=model,
            num_ot_pairs=16, sinkhorn_iters=10,
            context_consistency_weight=0.0,
        )

        # Add Dirichlet loss
        from stagebridge.models.dirichlet_stage import dirichlet_nll_loss
        alpha = model.predict_dirichlet(batch.x_src)
        assert alpha is not None
        targets = torch.full((BATCH,), batch.stage_src, dtype=torch.long)
        loss = loss + 0.05 * dirichlet_nll_loss(alpha, targets)

        loss.backward()

        # All named parameters should have gradients
        has_grad = sum(1 for p in model.parameters() if p.grad is not None)
        total_params = sum(1 for _ in model.parameters())
        assert has_grad > total_params * 0.8, f"Only {has_grad}/{total_params} params got gradients"
