"""Tests for pathway_targets.py - IL1B, KAC, and pathway target computation."""

import pytest
import torch
from stagebridge.biology.pathway_targets import (
    compute_pathway_targets,
    compute_pathway_reference_stats,
    compute_proliferation_targets,
    compute_il1b_targets,
    compute_il1b_reference_stats,
    compute_kac_targets,
    compute_kac_reference_stats,
    compute_pathway_raw,
    compute_il1b_raw,
    compute_kac_raw,
    zscore_from_train_stats,
    get_pathway_gene_list,
    PROGENY_PATHWAYS,
    IL1B_SIGNALING_GENES,
    KAC_MARKERS,
    PROLIFERATION_MARKERS,
)


class TestPathwayTargets:
    """Tests for PROGENy pathway scoring."""

    def test_returns_correct_shape(self):
        n_cells, n_genes = 100, 50
        gene_names = list(PROGENY_PATHWAYS["Hypoxia"]) + ["FAKE_" + str(i) for i in range(n_genes - 5)]
        expr = torch.randn(n_cells, n_genes)

        result = compute_pathway_targets(expr, gene_names, torch.device("cpu"))

        assert result is not None
        assert result.shape == (n_cells, len(PROGENY_PATHWAYS))

    def test_returns_none_for_empty_genes(self):
        expr = torch.randn(10, 5)
        result = compute_pathway_targets(expr, [], torch.device("cpu"))
        assert result is None

    def test_returns_none_for_no_matching_genes(self):
        expr = torch.randn(10, 5)
        gene_names = ["FAKE1", "FAKE2", "FAKE3", "FAKE4", "FAKE5"]
        result = compute_pathway_targets(expr, gene_names, torch.device("cpu"))
        assert result is None

    def test_reference_stats_prevent_leakage(self):
        """Verify that reference stats from train set are used for val set."""
        n_train, n_val, n_genes = 80, 20, 10
        gene_names = list(PROGENY_PATHWAYS["Hypoxia"])[:5] + ["FAKE_" + str(i) for i in range(5)]

        train_expr = torch.randn(n_train, n_genes) * 2 + 5  # Different distribution
        val_expr = torch.randn(n_val, n_genes)

        # Compute reference stats from train only
        ref_stats = compute_pathway_reference_stats(train_expr, gene_names)

        # Val targets with and without reference stats
        val_with_ref = compute_pathway_targets(val_expr, gene_names, torch.device("cpu"), ref_stats)
        val_without_ref = compute_pathway_targets(val_expr, gene_names, torch.device("cpu"))

        # They should be different (val_without_ref uses val's own stats)
        assert not torch.allclose(val_with_ref, val_without_ref, atol=0.1)


class TestIL1BTargets:
    """Tests for IL1B signaling target computation."""

    def test_returns_correct_shape(self):
        n_cells, n_genes = 100, 20
        gene_names = IL1B_SIGNALING_GENES + ["FAKE_" + str(i) for i in range(n_genes - len(IL1B_SIGNALING_GENES))]
        expr = torch.randn(n_cells, n_genes)

        result = compute_il1b_targets(expr, gene_names, torch.device("cpu"))

        assert result is not None
        assert result.shape == (n_cells, 1)

    def test_uses_correct_genes(self):
        """Verify IL1B targets use IL1B signaling genes, not JAK-STAT."""
        n_cells = 50
        # Create expression with variance - half cells high IL1B, half low
        gene_names = ["IL1B", "IL1R1", "JAK1", "STAT1", "OTHER"]
        expr = torch.zeros(n_cells, 5)
        expr[:25, 0] = 10.0  # First half: IL1B high
        expr[25:, 0] = 0.0   # Second half: IL1B low
        expr[:25, 1] = 8.0   # IL1R1 also high in first half
        expr[:, 2:4] = 1.0   # JAK-STAT genes constant (not used)

        result = compute_il1b_targets(expr, gene_names, torch.device("cpu"))

        assert result is not None
        # First half should have higher scores than second half
        assert result[:25].mean() > result[25:].mean()

    def test_returns_none_without_il1b_genes(self):
        expr = torch.randn(10, 5)
        gene_names = ["FAKE1", "FAKE2", "FAKE3", "FAKE4", "FAKE5"]
        result = compute_il1b_targets(expr, gene_names, torch.device("cpu"))
        assert result is None

    def test_reference_stats_prevent_leakage(self):
        """Verify reference stats from train set used for val."""
        n_train, n_val = 80, 20
        gene_names = IL1B_SIGNALING_GENES[:3] + ["FAKE1", "FAKE2"]
        n_genes = len(gene_names)

        train_expr = torch.randn(n_train, n_genes) * 3 + 10
        val_expr = torch.randn(n_val, n_genes)

        ref_stats = compute_il1b_reference_stats(train_expr, gene_names)

        val_with_ref = compute_il1b_targets(val_expr, gene_names, torch.device("cpu"), ref_stats)
        val_without_ref = compute_il1b_targets(val_expr, gene_names, torch.device("cpu"))

        assert not torch.allclose(val_with_ref, val_without_ref, atol=0.1)


class TestKACTargets:
    """Tests for KAC (KRT8+ Alveolar Intermediate Cell) target computation."""

    def test_returns_correct_shape(self):
        n_cells, n_genes = 100, 20
        gene_names = KAC_MARKERS[:5] + ["FAKE_" + str(i) for i in range(n_genes - 5)]
        expr = torch.randn(n_cells, n_genes)

        result = compute_kac_targets(expr, gene_names, torch.device("cpu"))

        assert result is not None
        assert result.shape == (n_cells, 1)

    def test_uses_correct_markers(self):
        """Verify KAC targets use KAC markers, not p53 pathway."""
        n_cells = 50
        # Create expression with variance - half cells high KAC, half low
        gene_names = ["KRT8", "CLDN4", "CDKN1A", "BAX", "MDM2", "PMAIP1"]  # First 3 are KAC, last 3 are p53
        expr = torch.zeros(n_cells, 6)
        expr[:25, :3] = 10.0  # First half: KAC markers high
        expr[25:, :3] = 0.0   # Second half: KAC markers low
        expr[:, 3:] = 1.0     # p53 genes constant (not used for KAC)

        result = compute_kac_targets(expr, gene_names, torch.device("cpu"))

        assert result is not None
        # First half should have higher scores than second half
        assert result[:25].mean() > result[25:].mean()

    def test_requires_minimum_markers(self):
        """KAC needs at least 3 markers."""
        n_cells = 50
        gene_names = ["KRT8", "CLDN4", "FAKE1", "FAKE2"]  # Only 2 KAC markers
        expr = torch.randn(n_cells, 4)

        result = compute_kac_targets(expr, gene_names, torch.device("cpu"))

        # Should return None - not enough markers
        assert result is None

    def test_reference_stats_prevent_leakage(self):
        """Verify reference stats from train set used for val."""
        n_train, n_val = 80, 20
        gene_names = KAC_MARKERS[:4] + ["FAKE1"]
        n_genes = len(gene_names)

        train_expr = torch.randn(n_train, n_genes) * 2 + 5
        val_expr = torch.randn(n_val, n_genes)

        ref_stats = compute_kac_reference_stats(train_expr, gene_names)

        val_with_ref = compute_kac_targets(val_expr, gene_names, torch.device("cpu"), ref_stats)
        val_without_ref = compute_kac_targets(val_expr, gene_names, torch.device("cpu"))

        assert not torch.allclose(val_with_ref, val_without_ref, atol=0.1)


class TestProliferationTargets:
    """Tests for Ki67/proliferation target computation."""

    def test_returns_binary_labels(self):
        n_cells = 100
        gene_names = ["MKI67", "OTHER1", "OTHER2"]
        expr = torch.randn(n_cells, 3)

        result = compute_proliferation_targets(expr, gene_names, torch.device("cpu"))

        assert result is not None
        assert result.shape == (n_cells, 1)
        # Should be binary (0 or 1)
        assert torch.all((result == 0) | (result == 1))

    def test_finds_ki67_case_insensitive(self):
        """Should find Ki67 regardless of case."""
        n_cells = 50
        for ki67_name in ["MKI67", "KI67", "Ki67", "mki67"]:
            gene_names = [ki67_name, "OTHER"]
            expr = torch.randn(n_cells, 2)
            result = compute_proliferation_targets(expr, gene_names, torch.device("cpu"))
            assert result is not None, f"Failed to find {ki67_name}"

    def test_returns_none_without_ki67(self):
        expr = torch.randn(10, 5)
        gene_names = ["FAKE1", "FAKE2", "FAKE3", "FAKE4", "FAKE5"]
        result = compute_proliferation_targets(expr, gene_names, torch.device("cpu"))
        assert result is None


class TestGeneList:
    """Tests for gene list utility."""

    def test_includes_all_marker_types(self):
        genes = get_pathway_gene_list()

        # Should include pathway genes
        assert "VEGFA" in genes  # Hypoxia

        # Should include proliferation markers
        assert "MKI67" in genes

        # Should include KAC markers
        assert "KRT8" in genes

        # Should include IL1B signaling genes
        assert "IL1B" in genes
        assert "IL1R1" in genes

    def test_returns_sorted_unique(self):
        genes = get_pathway_gene_list()
        assert genes == sorted(set(genes))


class TestRawComputation:
    """Tests for raw (un-normalized) target computation functions."""

    def test_pathway_raw_no_zscore(self):
        """Verify pathway_raw returns raw means, not z-scored values."""
        n_cells = 100
        gene_names = list(PROGENY_PATHWAYS["Hypoxia"])[:3] + ["FAKE1", "FAKE2"]
        # Create expression with known mean
        expr = torch.ones(n_cells, len(gene_names)) * 5.0  # All values = 5

        result = compute_pathway_raw(expr, gene_names)

        assert result is not None
        # Raw mean of [5, 5, 5] should be 5, not 0 (which z-score would give)
        assert torch.allclose(result[:, 3], torch.tensor(5.0), atol=0.1)  # Hypoxia is index 3

    def test_il1b_raw_no_zscore(self):
        """Verify il1b_raw returns raw means, not z-scored values."""
        n_cells = 50
        gene_names = IL1B_SIGNALING_GENES[:2] + ["FAKE1"]
        expr = torch.ones(n_cells, len(gene_names)) * 10.0

        result = compute_il1b_raw(expr, gene_names)

        assert result is not None
        # Raw mean should be 10, not 0
        assert torch.allclose(result, torch.tensor(10.0), atol=0.1)

    def test_kac_raw_no_zscore(self):
        """Verify kac_raw returns raw means, not z-scored values."""
        n_cells = 50
        gene_names = KAC_MARKERS[:4] + ["FAKE1"]
        expr = torch.ones(n_cells, len(gene_names)) * 7.0

        result = compute_kac_raw(expr, gene_names)

        assert result is not None
        # Raw mean should be 7, not 0
        assert torch.allclose(result, torch.tensor(7.0), atol=0.1)


class TestZscoreFromTrainStats:
    """Tests for the leakage-free z-scoring function."""

    def test_uses_train_only_stats(self):
        """Verify z-scoring uses only train indices for statistics."""
        n_total = 100
        raw_values = torch.randn(n_total, 3)
        raw_values[:50] = raw_values[:50] * 2 + 10  # Train: different distribution
        raw_values[50:] = raw_values[50:] * 0.5     # Val: different distribution

        train_idx = torch.arange(50)

        z_scored, (train_mean, train_std) = zscore_from_train_stats(raw_values, train_idx)

        # Train z-scores should have mean ~0, std ~1
        train_z = z_scored[:50]
        assert torch.abs(train_z.mean()) < 0.2  # Approximately 0
        assert torch.abs(train_z.std() - 1.0) < 0.2  # Approximately 1

        # Val z-scores should NOT have mean 0 (using train stats on different distribution)
        val_z = z_scored[50:]
        # Val has different distribution, so its z-score mean should be far from 0
        assert torch.abs(val_z.mean()) > 0.5  # Definitely not 0

    def test_returns_train_stats_for_reproducibility(self):
        """Verify function returns train stats for logging/reproducibility."""
        n_total = 100
        raw_values = torch.randn(n_total, 2)
        train_idx = torch.arange(80)

        z_scored, (train_mean, train_std) = zscore_from_train_stats(raw_values, train_idx)

        # Should return proper shapes
        assert train_mean.shape == (2,)
        assert train_std.shape == (2,)
        # Std should be positive
        assert (train_std > 0).all()

    def test_no_leakage_different_results(self):
        """Verify that train-only z-scoring gives different results than global."""
        n_total = 100
        raw_values = torch.randn(n_total, 1)
        raw_values[:80] = raw_values[:80] + 5  # Train has offset
        train_idx = torch.arange(80)

        # Z-score with train-only stats
        z_train_only, _ = zscore_from_train_stats(raw_values, train_idx)

        # Z-score with global stats (what we want to avoid)
        global_mean = raw_values.mean(dim=0, keepdim=True)
        global_std = raw_values.std(dim=0, keepdim=True) + 1e-8
        z_global = (raw_values - global_mean) / global_std

        # Val set z-scores should be different
        val_z_train_only = z_train_only[80:]
        val_z_global = z_global[80:]

        # They should NOT be the same (proves we're using train-only stats)
        assert not torch.allclose(val_z_train_only, val_z_global, atol=0.1)
