"""
Tests for spatial backend selection module.
"""

import json
import pytest

from stagebridge.spatial_mapping.selection import (
    BackendSelection,
    select_canonical_backend,
    generate_selection_report,
    save_canonical_decision,
    load_canonical_decision,
)
from stagebridge.spatial_mapping.comparison import ComparisonResult


class TestBackendSelection:
    """Tests for BackendSelection dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        selection = BackendSelection(
            canonical_backend="tangram",
            selection_score=0.75,
            justification="Test justification",
            category_scores={"upstream": 0.7, "downstream": 0.8},
            alternatives=["destvi", "tacco"],
            alternative_scores={"destvi": 0.65, "tacco": 0.60},
        )

        d = selection.to_dict()

        assert d["canonical_backend"] == "tangram"
        assert d["selection_score"] == 0.75
        assert "upstream" in d["category_scores"]
        assert "destvi" in d["alternatives"]


class TestSelectCanonicalBackend:
    """Tests for canonical backend selection."""

    def test_basic_selection(self, synthetic_comparison_table):
        """Test basic canonical backend selection."""
        comparison = ComparisonResult(
            comparison_table=synthetic_comparison_table,
        )

        selection = select_canonical_backend(comparison)

        assert isinstance(selection, BackendSelection)
        assert selection.canonical_backend in ["tangram", "destvi", "tacco"]
        assert 0 <= selection.selection_score <= 1
        assert len(selection.justification) > 0

    def test_selection_with_custom_weights(self, synthetic_comparison_table):
        """Test selection with custom weights."""
        comparison = ComparisonResult(
            comparison_table=synthetic_comparison_table,
        )

        # Weight heavily toward downstream
        weights = {
            "upstream": 0.0,
            "downstream": 1.0,
            "spatial": 0.0,
            "robustness": 0.0,
            "runtime": 0.0,
        }

        selection = select_canonical_backend(comparison, weights=weights)

        # tacco has highest downstream utility in synthetic data
        assert selection.canonical_backend == "tacco"

    def test_selection_alternatives(self, synthetic_comparison_table):
        """Test that alternatives are populated."""
        comparison = ComparisonResult(
            comparison_table=synthetic_comparison_table,
        )

        selection = select_canonical_backend(comparison)

        # Should have 2 alternatives (3 backends - 1 canonical)
        assert len(selection.alternatives) == 2

        # Canonical should not be in alternatives
        assert selection.canonical_backend not in selection.alternatives

    def test_no_successful_backends(self):
        """Test error when no backends succeeded."""
        import pandas as pd

        comparison = ComparisonResult(
            comparison_table=pd.DataFrame(
                {
                    "backend": ["tangram"],
                    "success": [False],
                    "runtime_seconds": [0.0],
                }
            ),
        )

        with pytest.raises(ValueError, match="No backends completed"):
            select_canonical_backend(comparison)

    def test_selection_metadata(self, synthetic_comparison_table):
        """Test selection metadata."""
        comparison = ComparisonResult(
            comparison_table=synthetic_comparison_table,
        )

        selection = select_canonical_backend(comparison)

        assert "selection_weights" in selection.metadata
        assert "selection_timestamp" in selection.metadata
        assert "n_successful_backends" in selection.metadata


class TestGenerateSelectionReport:
    """Tests for selection report generation."""

    def test_report_generation(self, synthetic_comparison_table):
        """Test markdown report generation."""
        comparison = ComparisonResult(
            comparison_table=synthetic_comparison_table,
            rankings={"overall": ["tacco", "tangram", "destvi"]},
        )

        selection = BackendSelection(
            canonical_backend="tacco",
            selection_score=0.75,
            justification="Selected based on downstream utility.",
            category_scores={"downstream": 0.8, "upstream": 0.7},
            alternatives=["tangram", "destvi"],
        )

        report = generate_selection_report(comparison, selection)

        assert isinstance(report, str)
        assert "TACCO" in report
        assert "Canonical Backend" in report
        assert "Rankings" in report

    def test_report_save_to_file(self, synthetic_comparison_table, tmp_output_dir):
        """Test saving report to file."""
        comparison = ComparisonResult(
            comparison_table=synthetic_comparison_table,
            rankings={"overall": ["tacco", "tangram", "destvi"]},
        )

        selection = BackendSelection(
            canonical_backend="tacco",
            selection_score=0.75,
            justification="Test justification",
            alternatives=["tangram"],
        )

        output_path = tmp_output_dir / "test_report.md"
        generate_selection_report(comparison, selection, output_path=output_path)

        assert output_path.exists()
        with open(output_path) as f:
            content = f.read()
        assert "TACCO" in content


class TestSaveLoadCanonicalDecision:
    """Tests for saving and loading canonical decision."""

    def test_save_canonical_decision(self, tmp_output_dir):
        """Test saving canonical decision to JSON."""
        selection = BackendSelection(
            canonical_backend="tangram",
            selection_score=0.78,
            justification="# Canonical Selection\nSelected Tangram.",
            category_scores={"upstream": 0.7, "downstream": 0.8},
            alternatives=["destvi"],
            alternative_scores={"destvi": 0.65},
            metadata={"test": "value"},
        )

        json_path = save_canonical_decision(selection, tmp_output_dir)

        assert json_path.exists()
        assert (tmp_output_dir / "canonical_backend.json").exists()
        assert (tmp_output_dir / "backend_selection_report.md").exists()

        # Verify JSON content
        with open(json_path) as f:
            data = json.load(f)

        assert data["canonical_backend"] == "tangram"
        assert data["selection_score"] == 0.78

    def test_load_canonical_decision(self, tmp_output_dir):
        """Test loading canonical decision from JSON."""
        # First save
        selection = BackendSelection(
            canonical_backend="destvi",
            selection_score=0.72,
            justification="Test justification",
            category_scores={"upstream": 0.65},
            alternatives=["tangram", "tacco"],
            alternative_scores={"tangram": 0.60, "tacco": 0.55},
        )

        save_canonical_decision(selection, tmp_output_dir)

        # Then load
        loaded = load_canonical_decision(tmp_output_dir)

        assert loaded.canonical_backend == "destvi"
        assert loaded.selection_score == 0.72
        assert loaded.category_scores["upstream"] == 0.65
        assert "tangram" in loaded.alternatives
