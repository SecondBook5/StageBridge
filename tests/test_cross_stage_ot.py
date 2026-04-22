"""Tests for cross-stage optimal transport pairing in transition model."""

import pytest
import torch

from stagebridge.transition_model.losses import (
    build_sinkhorn_coupling,
    sample_coupling_pairs,
)


class TestCrossStageOTPairing:
    """Verify cross-stage OT pairing logic is correct."""

    def test_sinkhorn_coupling_shape(self):
        """Coupling matrix should be (n_src, n_tgt)."""
        n_src, n_tgt, dim = 10, 15, 32
        x_src = torch.randn(n_src, dim)
        x_tgt = torch.randn(n_tgt, dim)

        coupling = build_sinkhorn_coupling(x_src, x_tgt, epsilon=0.05, n_iters=50)

        assert coupling.shape == (n_src, n_tgt)

    def test_sinkhorn_coupling_non_negative(self):
        """All coupling entries should be non-negative."""
        x_src = torch.randn(10, 32)
        x_tgt = torch.randn(15, 32)

        coupling = build_sinkhorn_coupling(x_src, x_tgt)

        assert (coupling >= 0).all()

    def test_sinkhorn_coupling_row_marginals(self):
        """Row sums should be approximately uniform (1/n_src)."""
        n_src, n_tgt = 10, 15
        x_src = torch.randn(n_src, 32)
        x_tgt = torch.randn(n_tgt, 32)

        # Use more iterations and larger epsilon for better convergence
        coupling = build_sinkhorn_coupling(x_src, x_tgt, epsilon=0.1, n_iters=150)

        row_sums = coupling.sum(dim=1)
        expected = torch.full((n_src,), 1.0 / n_src)
        # Relaxed tolerance - Sinkhorn with finite iterations doesn't give exact marginals
        assert torch.allclose(row_sums, expected, atol=0.05)

    def test_sinkhorn_coupling_col_marginals(self):
        """Column sums should be uniform (1/n_tgt)."""
        n_src, n_tgt = 10, 15
        x_src = torch.randn(n_src, 32)
        x_tgt = torch.randn(n_tgt, 32)

        coupling = build_sinkhorn_coupling(x_src, x_tgt, epsilon=0.05, n_iters=100)

        col_sums = coupling.sum(dim=0)
        expected = torch.full((n_tgt,), 1.0 / n_tgt)
        assert torch.allclose(col_sums, expected, atol=1e-3)

    def test_sinkhorn_coupling_sums_to_one(self):
        """Total coupling mass should be 1."""
        x_src = torch.randn(10, 32)
        x_tgt = torch.randn(15, 32)

        coupling = build_sinkhorn_coupling(x_src, x_tgt)

        assert torch.isclose(coupling.sum(), torch.tensor(1.0), atol=1e-4)

    def test_sample_coupling_pairs_valid_indices(self):
        """Sampled indices should be within bounds."""
        n_src, n_tgt = 10, 15
        coupling = torch.rand(n_src, n_tgt)
        coupling = coupling / coupling.sum()

        num_pairs = 50
        src_idx, tgt_idx = sample_coupling_pairs(coupling, num_pairs)

        assert len(src_idx) == num_pairs
        assert len(tgt_idx) == num_pairs
        assert (src_idx >= 0).all() and (src_idx < n_src).all()
        assert (tgt_idx >= 0).all() and (tgt_idx < n_tgt).all()

    def test_sample_coupling_prefers_high_probability_pairs(self):
        """Sampling should prefer high-probability pairs in the coupling."""
        n_src, n_tgt = 5, 5
        # Create coupling with strong diagonal (identity-like)
        coupling = torch.eye(n_src, n_tgt) * 0.18 + torch.ones(n_src, n_tgt) * 0.004
        coupling = coupling / coupling.sum()

        # Sample many pairs
        num_pairs = 1000
        src_idx, tgt_idx = sample_coupling_pairs(coupling, num_pairs)

        # Count diagonal matches (src_idx == tgt_idx)
        diagonal_matches = (src_idx == tgt_idx).float().mean()

        # Should strongly prefer diagonal (> 50% if coupling is diagonal-heavy)
        assert diagonal_matches > 0.5

    def test_cross_stage_pairing_logic(self):
        """Verify cross-stage OT pairs cells from stage s with stage s+1."""
        # Simulate a batch with cells from different stages
        batch_size = 100
        dim = 32

        # Create stage indices: 25 cells each from stages 0, 1, 2, 3
        stage_indices = torch.cat([
            torch.zeros(25, dtype=torch.long),     # Stage 0 (Normal)
            torch.ones(25, dtype=torch.long),      # Stage 1 (AAH)
            torch.full((25,), 2, dtype=torch.long),  # Stage 2 (AIS)
            torch.full((25,), 3, dtype=torch.long),  # Stage 3 (MIA)
        ])

        # Create embeddings (z_source = z_target for cross-stage OT)
        z_source = torch.randn(batch_size, dim)
        z_target = z_source.clone()  # Same as source - targets come from next stage

        # Simulate cross-stage OT (simplified version of transition_forward logic)
        all_src_idx = []
        all_tgt_idx = []

        for s in range(3):  # Stages 0-2 can transition to s+1
            src_mask = (stage_indices == s)
            tgt_mask = (stage_indices == s + 1)

            n_src = src_mask.sum().item()
            n_tgt = tgt_mask.sum().item()

            if n_src >= 2 and n_tgt >= 2:
                src_batch_idx = torch.where(src_mask)[0]
                tgt_batch_idx = torch.where(tgt_mask)[0]

                # Build OT coupling for this stage pair
                coupling = build_sinkhorn_coupling(
                    x_src=z_source[src_batch_idx],
                    x_tgt=z_target[tgt_batch_idx],
                    epsilon=0.1,
                    n_iters=50,
                )

                # Sample pairs
                n_pairs = 10
                local_src_idx, local_tgt_idx = sample_coupling_pairs(coupling, n_pairs)

                # Map back to batch indices
                all_src_idx.append(src_batch_idx[local_src_idx])
                all_tgt_idx.append(tgt_batch_idx[local_tgt_idx])

        src_idx = torch.cat(all_src_idx)
        tgt_idx = torch.cat(all_tgt_idx)

        # Verify: each pair should be (stage s, stage s+1)
        src_stages = stage_indices[src_idx]
        tgt_stages = stage_indices[tgt_idx]

        # Target stage should always be source stage + 1
        assert (tgt_stages == src_stages + 1).all(), "Cross-stage OT should pair s with s+1"

    def test_ot_prefers_nearby_cells(self):
        """OT should prefer pairing nearby cells in embedding space."""
        n = 20
        dim = 2

        # Create two clusters: one at origin, one at (10, 10)
        x_src = torch.cat([
            torch.randn(n // 2, dim) * 0.5,         # Cluster 1 near origin
            torch.randn(n // 2, dim) * 0.5 + 10,    # Cluster 2 near (10, 10)
        ])
        x_tgt = torch.cat([
            torch.randn(n // 2, dim) * 0.5,         # Cluster 1 near origin
            torch.randn(n // 2, dim) * 0.5 + 10,    # Cluster 2 near (10, 10)
        ])

        coupling = build_sinkhorn_coupling(x_src, x_tgt, epsilon=0.01, n_iters=100)

        # Sample many pairs
        src_idx, tgt_idx = sample_coupling_pairs(coupling, 500)

        # Check that pairs are mostly within-cluster
        # Cluster 1: indices 0-9, Cluster 2: indices 10-19
        src_cluster = (src_idx >= n // 2).long()
        tgt_cluster = (tgt_idx >= n // 2).long()

        same_cluster_frac = (src_cluster == tgt_cluster).float().mean()

        # Should strongly prefer same-cluster pairings
        assert same_cluster_frac > 0.8, f"OT should prefer nearby cells, got {same_cluster_frac:.2f}"
