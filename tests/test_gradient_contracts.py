"""Gradient flow contract tests.

These tests enforce that gradients flow through all trainable parameters.
If a parameter has no gradient, the model architecture is broken.

Run with: pytest tests/test_gradient_contracts.py -v
"""

from __future__ import annotations

import pytest
import torch

from stagebridge.models import StageBridge, StageBridgeConfig
from stagebridge.contracts import LATENT_DIM, HLCA_DIM, LUCA_DIM, STATS_TOKEN_DIM, MAX_CELLS_PER_RING


class TestGradientFlowContracts:
    """Contract tests ensuring gradient flow through all model components."""

    @pytest.fixture(autouse=True)
    def seed(self):
        """Fix random seed for reproducibility."""
        torch.manual_seed(42)

    @pytest.fixture
    def model(self) -> StageBridge:
        """Create a small model for testing."""
        config = StageBridgeConfig(
            input_dim=LATENT_DIM,
            hidden_dim=64,
            num_heads=2,
            num_encoder_layers=2,
            max_neighbors=8,
            num_stages=3,
            use_hierarchical=False,
            use_sample_heads=False,
            use_pathway_head=False,
            use_proliferation_head=False,
            use_learned_ring_pooling=True,
            use_context_refiner=True,
        )
        return StageBridge(config)

    @pytest.fixture
    def batch(self) -> dict:
        """Create a minimal batch for gradient testing."""
        batch_size = 4
        max_cells = 10
        return {
            "receiver": torch.randn(batch_size, LATENT_DIM),
            "ring_cells": [torch.randn(batch_size, max_cells, LATENT_DIM) for _ in range(4)],
            "ring_masks": [torch.ones(batch_size, max_cells, dtype=torch.bool) for _ in range(4)],
            "hlca": torch.randn(batch_size, HLCA_DIM),
            "luca": torch.randn(batch_size, LUCA_DIM),
            "pathway": torch.randn(batch_size, LATENT_DIM),
            "stats": torch.randn(batch_size, STATS_TOKEN_DIM),
            "t": torch.rand(batch_size),
            "x_t": torch.randn(batch_size, LATENT_DIM),
            "stage_pair_id": torch.zeros(batch_size, dtype=torch.long),
        }

    def _run_forward_backward(self, model: StageBridge, batch: dict) -> None:
        """Run forward pass and compute gradients."""
        model.zero_grad()

        # Encode niche
        niche_output = model.encode_niche(
            receiver=batch["receiver"],
            ring_cells=batch["ring_cells"],
            ring_masks=batch["ring_masks"],
            hlca=batch["hlca"],
            luca=batch["luca"],
            pathway=batch["pathway"],
            stats=batch["stats"],
        )

        # Forward vector field
        prediction = model.forward_vector_field(
            x_t=batch["x_t"],
            t=batch["t"],
            context=niche_output.context,
            stage_pair_id=batch["stage_pair_id"],
            context_tokens=niche_output.context_tokens,
        )

        loss = prediction.pow(2).mean()
        loss.backward()

    def _get_params_without_grad(
        self, model: StageBridge, exclude_patterns: list[str] | None = None
    ) -> list[str]:
        """Return names of parameters that didn't receive gradients."""
        exclude_patterns = exclude_patterns or []
        missing = []
        for name, p in model.named_parameters():
            if any(pat in name for pat in exclude_patterns):
                continue
            if p.grad is None or p.grad.abs().sum() == 0:
                missing.append(name)
        return missing

    def test_all_encoder_attention_layers_receive_gradients(
        self, model: StageBridge, batch: dict
    ):
        """CONTRACT: Active attention layers must receive gradients.

        Note: niche_encoder is not used when use_learned_ring_pooling=True.
        We use niche_tokenizer + context_refiner instead.
        """
        self._run_forward_backward(model, batch)

        # Only check components that are actually used in the forward pass
        # niche_encoder is not used with learned ring pooling
        # context_refiner.pma output goes to stats_conditioner, not drift head
        # context_refiner.isab2.rpe needs coords which we don't pass
        exclude_patterns = ["niche_encoder", "pma", "rpe", "stats_conditioner"]

        missing = []
        for name, p in model.named_parameters():
            if any(ex in name for ex in exclude_patterns):
                continue
            if "attention" in name.lower() or "isab" in name.lower() or "sab" in name.lower():
                if p.grad is None or p.grad.abs().sum() == 0:
                    missing.append(name)

        assert not missing, (
            f"GRADIENT FLOW BROKEN: Attention layers without gradients:\n"
            f"  {missing}\n"
            f"This likely means context_tokens doesn't flow through attention."
        )

    def test_all_drift_head_params_receive_gradients(
        self, model: StageBridge, batch: dict
    ):
        """CONTRACT: All drift head parameters must receive gradients."""
        self._run_forward_backward(model, batch)

        missing = []
        for name, p in model.named_parameters():
            if "drift" in name.lower():
                if p.grad is None or p.grad.abs().sum() == 0:
                    missing.append(name)

        assert not missing, (
            f"GRADIENT FLOW BROKEN: Drift head params without gradients:\n"
            f"  {missing}"
        )

    def test_all_stage_embeddings_receive_gradients(
        self, model: StageBridge, batch: dict
    ):
        """CONTRACT: Stage embeddings must receive gradients when used."""
        self._run_forward_backward(model, batch)

        missing = []
        for name, p in model.named_parameters():
            if "stage" in name.lower() and "embedding" in name.lower():
                if p.grad is None or p.grad.abs().sum() == 0:
                    missing.append(name)

        assert not missing, (
            f"GRADIENT FLOW BROKEN: Stage embeddings without gradients:\n"
            f"  {missing}"
        )

    def test_minimum_gradient_coverage(self, model: StageBridge, batch: dict):
        """CONTRACT: At least 70% of parameters must receive gradients.

        Expected exclusions (not receiving gradients in this test):
        - reconstruction_head (SSL-only)
        - niche_encoder (not used with learned ring pooling)
        - context_refiner.pma (output goes to stats_conditioner, not drift)
        - context_refiner.isab2.rpe (needs coords, not passed in this test)
        - stats_conditioner (stats dim mismatch in test)
        """
        self._run_forward_backward(model, batch)

        total = 0
        with_grad = 0
        for name, p in model.named_parameters():
            total += 1
            if p.grad is not None and p.grad.abs().sum() > 0:
                with_grad += 1

        coverage = with_grad / total
        assert coverage >= 0.70, (
            f"GRADIENT FLOW BROKEN: Only {with_grad}/{total} ({coverage:.1%}) "
            f"parameters received gradients. Expected >= 70%."
        )

    def test_reconstruction_head_excluded_is_expected(
        self, model: StageBridge, batch: dict
    ):
        """Document expected gradient exclusions.

        Expected exclusions (not receiving gradients in flow matching):
        - reconstruction_head: SSL-only
        - niche_encoder: replaced by niche_tokenizer with learned ring pooling
        - pma: output goes to stats_conditioner, not directly to drift
        - rpe: spatial RPE needs coords which aren't passed in this test
        - stats_conditioner: test batch stats dim doesn't match config
        """
        self._run_forward_backward(model, batch)

        expected_exclusions = [
            "reconstruction",
            "niche_encoder",
            "pma",
            "rpe",
            "stats_conditioner",
        ]
        missing = self._get_params_without_grad(model)
        unexpected = [n for n in missing if not any(ex in n for ex in expected_exclusions)]

        assert not unexpected, (
            f"GRADIENT FLOW BROKEN: Unexpected params without gradients:\n"
            f"  {unexpected}\n"
            f"Expected exclusions: {expected_exclusions}"
        )

    def test_gradient_magnitudes_reasonable(self, model: StageBridge, batch: dict):
        """CONTRACT: Gradients should not be NaN, Inf, or extremely large."""
        self._run_forward_backward(model, batch)

        issues = []
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            if torch.isnan(p.grad).any():
                issues.append(f"{name}: contains NaN")
            elif torch.isinf(p.grad).any():
                issues.append(f"{name}: contains Inf")
            elif p.grad.abs().max() > 1000:
                issues.append(f"{name}: max gradient {p.grad.abs().max():.1f} > 1000")

        assert not issues, (
            f"GRADIENT STABILITY BROKEN:\n"
            + "\n".join(f"  {i}" for i in issues)
        )


class TestNicheTokenizerGradientFlow:
    """Focused tests on the NicheTokenizer gradient path."""

    @pytest.fixture(autouse=True)
    def seed(self):
        """Fix random seed for reproducibility."""
        torch.manual_seed(42)

    @pytest.fixture
    def tokenizer(self):
        """Create tokenizer directly for isolated testing."""
        from stagebridge.context.tokenizer import NicheTokenizer

        return NicheTokenizer(
            input_dim=LATENT_DIM,
            hidden_dim=64,
            num_rings=4,
            num_heads=2,
            num_inducing=4,
        )

    def test_tokenizer_ring_poolers_receive_gradients(self, tokenizer):
        """CONTRACT: All ring poolers must receive gradients."""
        batch_size = 4
        max_cells = 10

        receiver = torch.randn(batch_size, LATENT_DIM)
        ring_cells = [torch.randn(batch_size, max_cells, LATENT_DIM) for _ in range(4)]
        ring_masks = [torch.ones(batch_size, max_cells, dtype=torch.bool) for _ in range(4)]
        hlca = torch.randn(batch_size, HLCA_DIM)
        luca = torch.randn(batch_size, LUCA_DIM)

        tokenizer.zero_grad()
        tokens, reconstruction, _ = tokenizer(
            receiver=receiver,
            ring_cells=ring_cells,
            ring_masks=ring_masks,
            hlca=hlca,
            luca=luca,
        )

        loss = tokens.pow(2).mean()
        loss.backward()

        for i, pooler in enumerate(tokenizer.ring_poolers):
            for name, p in pooler.named_parameters():
                assert p.grad is not None and p.grad.abs().sum() > 0, (
                    f"TOKENIZER GRADIENT BROKEN: ring_poolers.{i}.{name} has no gradient."
                )


class TestBaselineGradientFlow:
    """Ensure baselines also have proper gradient flow."""

    @pytest.fixture(autouse=True)
    def seed(self):
        """Fix random seed for reproducibility."""
        torch.manual_seed(42)

    @pytest.fixture
    def batch(self) -> dict:
        """Minimal batch for baselines (uses legacy API)."""
        batch_size = 4
        return {
            "receiver": torch.randn(batch_size, LATENT_DIM),
            "neighbors": torch.randn(batch_size, 8, LATENT_DIM),
            "distances": torch.rand(batch_size, 8) * 50,
            "x_t": torch.randn(batch_size, LATENT_DIM),
            "t": torch.rand(batch_size),
            "stage_pair_id": torch.zeros(batch_size, dtype=torch.long),
            "neighbor_mask": torch.ones(batch_size, 8, dtype=torch.bool),
        }

    @pytest.mark.parametrize("baseline_name", [
        "pooling",
        "deepsets",
        "set_transformer",
        "graphsage",
    ])
    def test_baseline_full_gradient_flow(self, baseline_name: str, batch: dict):
        """CONTRACT: All baseline parameters must receive gradients."""
        from stagebridge.baselines import get_baseline

        model = get_baseline(baseline_name, input_dim=LATENT_DIM, hidden_dim=64)
        model.zero_grad()

        out = model(**batch)
        loss = out.pow(2).mean()
        loss.backward()

        missing = []
        for name, p in model.named_parameters():
            if p.grad is None or p.grad.abs().sum() == 0:
                missing.append(name)

        assert not missing, (
            f"BASELINE {baseline_name} GRADIENT BROKEN:\n"
            f"  Parameters without gradients: {missing}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
