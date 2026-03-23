"""Tests for attention-weighted L-R interaction scoring."""

import numpy as np
import pandas as pd
import pytest

from stagebridge.biology.attention_lr_scoring import (
    LR_PRIORS,
    SENDER_TYPE_CATEGORIES,
    compute_attention_weighted_lr_scores,
    aggregate_lr_scores_by_stage,
    identify_stage_specific_interactions,
    compute_il1b_axis_score,
    generate_niche_ecosystem_summary,
    create_lr_interaction_report,
    export_lr_scores_for_visualization,
)


class TestLRPriors:
    """Tests for L-R prior definitions."""

    def test_il1b_il1r1_present(self):
        """IL1B-IL1R1 axis should be in priors with high support."""
        assert ("IL1B", "IL1R1") in LR_PRIORS
        assert LR_PRIORS[("IL1B", "IL1R1")]["support"] == 1.00
        assert LR_PRIORS[("IL1B", "IL1R1")]["family"] == "inflammatory"

    def test_all_priors_have_required_fields(self):
        """All L-R priors should have family, support, mechanism."""
        for (ligand, receptor), info in LR_PRIORS.items():
            assert "family" in info, f"Missing family for {ligand}-{receptor}"
            assert "support" in info, f"Missing support for {ligand}-{receptor}"
            assert "mechanism" in info, f"Missing mechanism for {ligand}-{receptor}"
            assert 0 <= info["support"] <= 1, f"Invalid support for {ligand}-{receptor}"

    def test_families_are_consistent(self):
        """Check that families are from expected set."""
        expected_families = {
            "inflammatory",
            "chemokine",
            "tgfb",
            "growth_factor",
            "notch",
            "ecm",
            "vascular",
            "immune_modulatory",
            "developmental",
        }
        for (ligand, receptor), info in LR_PRIORS.items():
            assert info["family"] in expected_families, f"Unknown family: {info['family']}"


class TestComputeAttentionWeightedLRScores:
    """Tests for attention-weighted L-R scoring."""

    def test_basic_scoring(self):
        """Test basic L-R scoring with attention."""
        n_senders = 10

        # Attention weights (normalized)
        attention = np.ones(n_senders) / n_senders

        # Sender types
        sender_types = np.zeros(n_senders, dtype=int)

        # Ligand expression - IL1B expressed in first 5 senders
        ligand_df = pd.DataFrame(
            {
                "IL1B": [1.0] * 5 + [0.0] * 5,
                "IL6": [0.5] * 10,
            }
        )

        # Receptor expression in receiver
        receptor_expr = pd.Series({"IL1R1": 0.8, "IL6ST": 0.3})

        scores = compute_attention_weighted_lr_scores(
            attention_weights=attention,
            sender_types=sender_types,
            ligand_expression=ligand_df,
            receptor_expression=receptor_expr,
        )

        assert len(scores) > 0
        assert all(hasattr(s, "ligand") and hasattr(s, "receptor") for s in scores)

        # IL1B-IL1R1 should have a score
        il1b_scores = [s for s in scores if s.ligand == "IL1B"]
        assert len(il1b_scores) > 0
        assert il1b_scores[0].interaction_score > 0

    def test_attention_weighting_matters(self):
        """Higher attention to IL1B expressors should increase score."""
        n_senders = 10

        # Ligand expression - IL1B in first 5
        ligand_df = pd.DataFrame(
            {
                "IL1B": [1.0] * 5 + [0.0] * 5,
            }
        )
        receptor_expr = pd.Series({"IL1R1": 1.0})
        sender_types = np.zeros(n_senders, dtype=int)

        # Attention focused on IL1B expressors
        attention_focused = np.array([0.15] * 5 + [0.05] * 5)
        scores_focused = compute_attention_weighted_lr_scores(
            attention_focused, sender_types, ligand_df, receptor_expr
        )

        # Attention focused away from IL1B expressors
        attention_away = np.array([0.05] * 5 + [0.15] * 5)
        scores_away = compute_attention_weighted_lr_scores(
            attention_away, sender_types, ligand_df, receptor_expr
        )

        il1b_focused = next(s for s in scores_focused if s.ligand == "IL1B")
        il1b_away = next(s for s in scores_away if s.ligand == "IL1B")

        # Focused attention should yield higher score
        assert il1b_focused.interaction_score > il1b_away.interaction_score

    def test_multihead_attention_averaging(self):
        """Multi-head attention should be averaged."""
        n_senders = 10
        n_heads = 4

        # Multi-head attention (n_heads, n_senders)
        attention = np.random.rand(n_heads, n_senders)

        ligand_df = pd.DataFrame({"IL1B": np.random.rand(n_senders)})
        receptor_expr = pd.Series({"IL1R1": 0.5})
        sender_types = np.zeros(n_senders, dtype=int)

        scores = compute_attention_weighted_lr_scores(
            attention, sender_types, ligand_df, receptor_expr
        )

        # Should not fail and return results
        assert len(scores) > 0

    def test_missing_ligand_skipped(self):
        """Ligands not in expression matrix should be skipped."""
        ligand_df = pd.DataFrame({"FAKE_GENE": [1.0, 1.0]})
        receptor_expr = pd.Series({"IL1R1": 0.5})
        attention = np.array([0.5, 0.5])
        sender_types = np.zeros(2, dtype=int)

        scores = compute_attention_weighted_lr_scores(
            attention, sender_types, ligand_df, receptor_expr
        )

        # IL1B-IL1R1 should not appear (no IL1B in expression)
        il1b_scores = [s for s in scores if s.ligand == "IL1B"]
        assert len(il1b_scores) == 0

    def test_type_breakdown(self):
        """Sender type breakdown should be computed when type_names provided."""
        n_senders = 10
        attention = np.ones(n_senders) / n_senders
        sender_types = np.array([0] * 5 + [1] * 5)  # Two types
        type_names = ["Macrophage", "Fibroblast"]

        ligand_df = pd.DataFrame(
            {
                "IL1B": [1.0] * 5 + [0.0] * 5,  # Only macrophages express IL1B
            }
        )
        receptor_expr = pd.Series({"IL1R1": 1.0})

        scores = compute_attention_weighted_lr_scores(
            attention, sender_types, ligand_df, receptor_expr, type_names=type_names
        )

        il1b = next(s for s in scores if s.ligand == "IL1B")
        # Macrophage contribution should be positive
        assert "Macrophage" in il1b.sender_type_breakdown


class TestAggregateLRScoresByStage:
    """Tests for stage-level aggregation."""

    def test_basic_aggregation(self):
        """Test aggregation of cell-level scores by stage."""
        from stagebridge.biology.attention_lr_scoring import LRInteractionScore

        # Create mock cell scores
        cell_scores = [
            (
                "AAH",
                [
                    LRInteractionScore(
                        ligand="IL1B",
                        receptor="IL1R1",
                        family="inflammatory",
                        mechanism="test",
                        prior_support=1.0,
                        attention_weight=0.5,
                        ligand_expression=0.8,
                        receptor_expression=0.7,
                        interaction_score=0.56,
                        confidence="high",
                    )
                ],
            ),
            (
                "AAH",
                [
                    LRInteractionScore(
                        ligand="IL1B",
                        receptor="IL1R1",
                        family="inflammatory",
                        mechanism="test",
                        prior_support=1.0,
                        attention_weight=0.6,
                        ligand_expression=0.9,
                        receptor_expression=0.8,
                        interaction_score=0.72,
                        confidence="high",
                    )
                ],
            ),
            (
                "AIS",
                [
                    LRInteractionScore(
                        ligand="IL1B",
                        receptor="IL1R1",
                        family="inflammatory",
                        mechanism="test",
                        prior_support=1.0,
                        attention_weight=0.3,
                        ligand_expression=0.4,
                        receptor_expression=0.5,
                        interaction_score=0.20,
                        confidence="medium",
                    )
                ],
            ),
        ]

        result = aggregate_lr_scores_by_stage(cell_scores)

        assert "AAH" in result
        assert "AIS" in result
        assert len(result["AAH"]) > 0
        assert "mean_score" in result["AAH"].columns


class TestComputeIL1BAxisScore:
    """Tests for IL1B axis focused scoring."""

    def test_il1b_detected(self):
        """Test IL1B axis detection."""
        from stagebridge.biology.attention_lr_scoring import LRInteractionScore

        scores = [
            LRInteractionScore(
                ligand="IL1B",
                receptor="IL1R1",
                family="inflammatory",
                mechanism="test",
                prior_support=1.0,
                attention_weight=0.5,
                ligand_expression=0.8,
                receptor_expression=0.7,
                interaction_score=0.56,
                sender_type_breakdown={"Macrophage": 0.4},
                confidence="high",
            ),
        ]

        result = compute_il1b_axis_score(scores)

        assert result["detected"] is True
        assert result["score"] == 0.56
        assert result["macrophage_contribution"] == 0.4
        assert "IL1B" in result["interpretation"]

    def test_il1b_not_detected(self):
        """Test when IL1B axis is not present."""
        from stagebridge.biology.attention_lr_scoring import LRInteractionScore

        scores = [
            LRInteractionScore(
                ligand="IL6",
                receptor="IL6ST",
                family="inflammatory",
                mechanism="test",
                prior_support=0.95,
                attention_weight=0.5,
                ligand_expression=0.8,
                receptor_expression=0.7,
                interaction_score=0.56,
                confidence="high",
            ),
        ]

        result = compute_il1b_axis_score(scores)

        assert result["detected"] is False
        assert result["score"] == 0.0


class TestGenerateNicheEcosystemSummary:
    """Tests for niche ecosystem summary generation."""

    def test_basic_summary_generation(self):
        """Test summary generation from stage scores."""
        stage_scores = {
            "AAH": pd.DataFrame(
                {
                    "ligand": ["IL1B", "IL6"],
                    "receptor": ["IL1R1", "IL6ST"],
                    "family": ["inflammatory", "inflammatory"],
                    "mean_score": [0.8, 0.3],
                    "std_score": [0.1, 0.05],
                    "n_cells": [100, 100],
                    "mean_attention": [0.4, 0.2],
                }
            ),
            "AIS": pd.DataFrame(
                {
                    "ligand": ["IL1B", "TGFB1"],
                    "receptor": ["IL1R1", "TGFBR2"],
                    "family": ["inflammatory", "tgfb"],
                    "mean_score": [0.9, 0.5],
                    "std_score": [0.15, 0.1],
                    "n_cells": [150, 150],
                    "mean_attention": [0.5, 0.3],
                }
            ),
        }

        summaries = generate_niche_ecosystem_summary(stage_scores)

        assert "AAH" in summaries
        assert "AIS" in summaries
        assert summaries["AAH"].stage == "AAH"
        assert len(summaries["AAH"].dominant_lr_interactions) > 0
        assert summaries["AAH"].biological_interpretation != ""


class TestCreateLRInteractionReport:
    """Tests for report generation."""

    def test_report_structure(self):
        """Test report has expected structure."""
        from stagebridge.biology.attention_lr_scoring import (
            NicheEcosystemSummary,
            LRInteractionScore,
        )

        summaries = {
            "AAH": NicheEcosystemSummary(
                stage="AAH",
                n_cells=100,
                dominant_lr_interactions=[
                    LRInteractionScore(
                        ligand="IL1B",
                        receptor="IL1R1",
                        family="inflammatory",
                        mechanism="test",
                        prior_support=1.0,
                        attention_weight=0.5,
                        ligand_expression=0.8,
                        receptor_expression=0.7,
                        interaction_score=0.8,
                        stage="AAH",
                        confidence="high",
                    )
                ],
                dominant_sender_types={"inflammatory": 0.8},
                pathway_activity={},
                risk_level="medium",
                biological_interpretation="Test interpretation",
                key_findings=["Finding 1"],
            ),
        }

        report = create_lr_interaction_report(summaries)

        assert "title" in report
        assert "stages" in report
        assert "key_mechanism" in report
        assert "AAH" in report["stages"]


class TestExportLRScores:
    """Tests for L-R score export."""

    def test_export_combines_stages(self):
        """Test export combines all stages into one DataFrame."""
        stage_scores = {
            "AAH": pd.DataFrame(
                {
                    "ligand": ["IL1B"],
                    "receptor": ["IL1R1"],
                    "family": ["inflammatory"],
                    "mean_score": [0.8],
                    "std_score": [0.1],
                    "n_cells": [100],
                    "mean_attention": [0.4],
                }
            ),
            "AIS": pd.DataFrame(
                {
                    "ligand": ["IL6"],
                    "receptor": ["IL6ST"],
                    "family": ["inflammatory"],
                    "mean_score": [0.5],
                    "std_score": [0.1],
                    "n_cells": [100],
                    "mean_attention": [0.3],
                }
            ),
        }

        result = export_lr_scores_for_visualization(stage_scores)

        assert "stage" in result.columns
        assert "lr_pair" in result.columns
        assert len(result) == 2
        assert set(result["stage"]) == {"AAH", "AIS"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
