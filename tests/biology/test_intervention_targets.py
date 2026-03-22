"""Tests for intervention target prioritization and niche risk scoring."""

import numpy as np
import pandas as pd
import pytest

from stagebridge.biology.intervention_targets import (
    DRUGGABILITY_DATABASE,
    InterventionTarget,
    NicheRiskScore,
    InterventionPlan,
    prioritize_intervention_targets,
    compute_niche_level_risk,
    aggregate_niche_risks_by_region,
    generate_intervention_plan,
    export_intervention_report,
)


class TestDruggabilityDatabase:
    """Tests for druggability information."""

    def test_il1b_is_approved(self):
        """IL1B should be marked as approved with known drugs."""
        assert "IL1B" in DRUGGABILITY_DATABASE
        assert DRUGGABILITY_DATABASE["IL1B"]["status"] == "approved"
        assert "Canakinumab" in DRUGGABILITY_DATABASE["IL1B"]["drugs"]

    def test_egfr_is_approved(self):
        """EGFR should have approved drugs."""
        assert "EGFR" in DRUGGABILITY_DATABASE
        assert DRUGGABILITY_DATABASE["EGFR"]["status"] == "approved"
        assert any("Erlotinib" in d for d in DRUGGABILITY_DATABASE["EGFR"]["drugs"])

    def test_all_entries_have_required_fields(self):
        """All database entries should have status and drugs."""
        for gene, info in DRUGGABILITY_DATABASE.items():
            assert "status" in info, f"Missing status for {gene}"
            assert "drugs" in info, f"Missing drugs for {gene}"
            assert info["status"] in ["approved", "clinical", "preclinical", "undrugged"]


class TestPrioritizeInterventionTargets:
    """Tests for intervention target prioritization."""

    def test_prioritizes_enriched_targets(self):
        """Should prioritize targets enriched in early stages."""
        stage_scores = {
            "AAH": pd.DataFrame({
                "ligand": ["IL1B", "IL6"],
                "receptor": ["IL1R1", "IL6ST"],
                "family": ["inflammatory", "inflammatory"],
                "mean_score": [0.8, 0.3],
                "std_score": [0.1, 0.05],
                "n_cells": [100, 100],
                "mean_attention": [0.4, 0.2],
            }),
        }

        stage_specific = pd.DataFrame({
            "stage": ["AAH", "AAH"],
            "ligand": ["IL1B", "IL6"],
            "receptor": ["IL1R1", "IL6ST"],
            "family": ["inflammatory", "inflammatory"],
            "mechanism": ["IL1B+ macrophage niche", "Inflammatory cytokine"],
            "stage_score": [0.8, 0.3],
            "other_score": [0.2, 0.1],
            "fold_change": [4.0, 3.0],
            "n_cells": [100, 100],
            "mean_attention": [0.4, 0.2],
        })

        targets = prioritize_intervention_targets(
            stage_scores,
            stage_specific_df=stage_specific,
            target_stages=["AAH"],
        )

        assert len(targets) > 0
        # IL1B should be prioritized (higher score, approved drug)
        il1b_targets = [t for t in targets if t.ligand == "IL1B"]
        assert len(il1b_targets) > 0

    def test_druggability_affects_priority(self):
        """Druggable targets should have higher priority."""
        stage_specific = pd.DataFrame({
            "stage": ["AAH", "AAH"],
            "ligand": ["IL1B", "FN1"],  # IL1B approved, FN1 undrugged
            "receptor": ["IL1R1", "ITGB1"],
            "family": ["inflammatory", "ecm"],
            "mechanism": ["test", "test"],
            "stage_score": [0.5, 0.5],  # Same score
            "other_score": [0.1, 0.1],
            "fold_change": [5.0, 5.0],  # Same enrichment
            "n_cells": [100, 100],
            "mean_attention": [0.4, 0.4],
        })

        targets = prioritize_intervention_targets(
            {},
            stage_specific_df=stage_specific,
            target_stages=["AAH"],
        )

        # IL1B should rank higher due to druggability
        if len(targets) >= 2:
            il1b_rank = next(i for i, t in enumerate(targets) if t.ligand == "IL1B")
            fn1_rank = next((i for i, t in enumerate(targets) if t.ligand == "FN1"), len(targets))
            assert il1b_rank < fn1_rank

    def test_returns_intervention_target_objects(self):
        """Should return properly structured InterventionTarget objects."""
        stage_specific = pd.DataFrame({
            "stage": ["AAH"],
            "ligand": ["IL1B"],
            "receptor": ["IL1R1"],
            "family": ["inflammatory"],
            "mechanism": ["test"],
            "stage_score": [0.8],
            "other_score": [0.2],
            "fold_change": [4.0],
            "n_cells": [100],
            "mean_attention": [0.4],
        })

        targets = prioritize_intervention_targets(
            {},
            stage_specific_df=stage_specific,
        )

        assert len(targets) > 0
        target = targets[0]
        assert isinstance(target, InterventionTarget)
        assert target.ligand == "IL1B"
        assert target.receptor == "IL1R1"
        assert target.priority_score > 0
        assert target.rationale != ""
        assert len(target.evidence) > 0


class TestComputeNicheLevelRisk:
    """Tests for niche-level risk computation."""

    def test_basic_niche_risk(self):
        """Test basic niche risk calculation."""
        cell_risks = pd.DataFrame({
            "cell_id": [f"cell_{i}" for i in range(20)],
            "risk_score": np.random.rand(20),
        })

        # Random spatial coordinates in a small area
        coords = np.random.rand(20, 2) * 100

        niche_scores = compute_niche_level_risk(
            cell_risks=cell_risks,
            spatial_coords=coords,
            niche_radius=50.0,  # Should find neighbors
            min_cells=2,
        )

        assert len(niche_scores) > 0
        for niche in niche_scores:
            assert isinstance(niche, NicheRiskScore)
            assert 0 <= niche.intrinsic_risk <= 1
            assert niche.n_cells >= 2

    def test_attention_weighted_niche_risk(self):
        """Test that attention weighting affects niche risk."""
        cell_risks = pd.DataFrame({
            "cell_id": ["center", "high_risk", "low_risk"],
            "risk_score": [0.5, 0.9, 0.1],
        })

        # Place cells close together
        coords = np.array([
            [50, 50],   # center
            [51, 50],   # high_risk neighbor
            [49, 50],   # low_risk neighbor
        ])

        # Attention focused on high-risk neighbor
        attention = np.array([[0.0, 0.9, 0.1]])  # From center to others

        niche_scores = compute_niche_level_risk(
            cell_risks=cell_risks,
            spatial_coords=coords,
            attention_weights=attention,
            niche_radius=10.0,
            min_cells=2,
        )

        # Find center's niche score
        center_niche = next((n for n in niche_scores if n.center_cell_id == "center"), None)
        assert center_niche is not None
        # Niche risk should be biased toward high-risk neighbor due to attention
        assert center_niche.niche_risk > 0.5

    def test_risk_categories(self):
        """Test risk category assignment."""
        # Very high risk cells
        cell_risks = pd.DataFrame({
            "cell_id": [f"cell_{i}" for i in range(10)],
            "risk_score": [0.9] * 10,  # All high risk
        })
        coords = np.random.rand(10, 2) * 10  # Clustered

        niche_scores = compute_niche_level_risk(
            cell_risks, coords, niche_radius=20.0, min_cells=2
        )

        # Should have high/very_high risk categories
        categories = {n.risk_category for n in niche_scores}
        assert "very_high" in categories or "high" in categories


class TestAggregateNicheRisksByRegion:
    """Tests for region-level aggregation."""

    def test_basic_aggregation(self):
        """Test aggregation of niche scores."""
        niche_scores = [
            NicheRiskScore(
                niche_id="n1", center_cell_id="c1", n_cells=5,
                intrinsic_risk=0.6, niche_risk=0.7, niche_contribution=0.1,
                dominant_risk_pathway="inflammatory", dominant_sender_type="macrophage",
                il1b_axis_active=True, risk_category="high", confidence="medium"
            ),
            NicheRiskScore(
                niche_id="n2", center_cell_id="c2", n_cells=8,
                intrinsic_risk=0.3, niche_risk=0.4, niche_contribution=0.1,
                dominant_risk_pathway="tgfb", dominant_sender_type="fibroblast",
                il1b_axis_active=False, risk_category="intermediate", confidence="high"
            ),
        ]

        df, summary = aggregate_niche_risks_by_region(niche_scores)

        assert len(df) == 2
        assert "n_niches" in summary
        assert summary["n_niches"] == 2
        assert "pct_il1b_active" in summary


class TestGenerateInterventionPlan:
    """Tests for intervention plan generation."""

    def test_plan_generation(self):
        """Test complete intervention plan generation."""
        targets = [
            InterventionTarget(
                ligand="IL1B", receptor="IL1R1", target_gene="IL1B",
                priority_score=10.0, rationale="Test rationale",
                evidence=["Evidence 1"], stage_enrichment={"AAH": 4.0},
                druggability="approved", safety_considerations=["Test safety"],
                expected_effect="Test effect"
            ),
        ]

        niche_scores = [
            NicheRiskScore(
                niche_id="n1", center_cell_id="c1", n_cells=5,
                intrinsic_risk=0.6, niche_risk=0.8, niche_contribution=0.2,
                dominant_risk_pathway="inflammatory", dominant_sender_type="macrophage",
                il1b_axis_active=True, risk_category="high", confidence="medium"
            ),
        ]

        plan = generate_intervention_plan(
            sample_id="test_sample",
            stage="AAH",
            targets=targets,
            niche_scores=niche_scores,
            overall_risk=0.7,
        )

        assert isinstance(plan, InterventionPlan)
        assert plan.sample_id == "test_sample"
        assert plan.stage == "AAH"
        assert plan.overall_risk == "high"
        assert plan.primary_target is not None
        assert plan.primary_target.ligand == "IL1B"
        assert len(plan.high_risk_niches) > 0
        assert len(plan.caveats) > 0

    def test_plan_includes_caveats(self):
        """Plan should include appropriate caveats."""
        plan = generate_intervention_plan(
            sample_id="test",
            stage="AAH",
            targets=[],
            niche_scores=[],
            overall_risk=0.5,
        )

        assert "MODEL-GENERATED HYPOTHESIS" in plan.caveats[0]
        assert any("validation" in c.lower() for c in plan.caveats)


class TestExportInterventionReport:
    """Tests for report export."""

    def test_export_structure(self):
        """Test exported report has expected structure."""
        plan = InterventionPlan(
            sample_id="test_sample",
            stage="AAH",
            overall_risk="high",
            primary_target=InterventionTarget(
                ligand="IL1B", receptor="IL1R1", target_gene="IL1B",
                priority_score=10.0, rationale="Test",
                evidence=["Evidence"], stage_enrichment={"AAH": 4.0},
                druggability="approved", safety_considerations=["Safety"],
                expected_effect="Effect"
            ),
            secondary_targets=[],
            high_risk_niches=[],
            clinical_recommendation="Test recommendation",
            monitoring_strategy="Test monitoring",
            caveats=["Caveat 1"],
        )

        report = export_intervention_report(plan)

        assert "sample_id" in report
        assert "primary_target" in report
        assert report["sample_id"] == "test_sample"
        assert report["primary_target"]["pair"] == "IL1B-IL1R1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
