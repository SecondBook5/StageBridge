"""Tests for stagebridge.validation.splits module."""
import json
import pandas as pd
import pytest
from pathlib import Path
import tempfile

from stagebridge.validation.splits import (
    validate_splits,
    validate_splits_from_files,
    check_paired_sample_leakage,
)


@pytest.fixture
def cells_df():
    """Create test cells DataFrame."""
    return pd.DataFrame({
        "cell_id": [f"cell_{i}" for i in range(100)],
        "donor_id": ["D1"] * 30 + ["D2"] * 30 + ["D3"] * 20 + ["D4"] * 20,
        "sample_id": ["S1"] * 15 + ["S2"] * 15 + ["S3"] * 15 + ["S4"] * 15 +
                     ["S5"] * 20 + ["S6"] * 20,
    })


@pytest.fixture
def valid_splits():
    """Create valid split manifest (no leakage)."""
    return {
        "folds": [
            {
                "train_donors": ["D1", "D2"],
                "val_donors": ["D3"],
                "test_donors": ["D4"],
            },
            {
                "train_donors": ["D1", "D3"],
                "val_donors": ["D2"],
                "test_donors": ["D4"],
            },
        ]
    }


@pytest.fixture
def leaky_splits():
    """Create split manifest with train-test leakage."""
    return {
        "folds": [
            {
                "train_donors": ["D1", "D2"],
                "val_donors": ["D3"],
                "test_donors": ["D2"],  # D2 in both train and test!
            },
        ]
    }


class TestValidateSplits:
    def test_valid_splits_pass(self, cells_df, valid_splits):
        result = validate_splits(cells_df, valid_splits)
        assert result["valid"] is True
        assert len(result["issues"]) == 0

    def test_detects_train_test_leakage(self, cells_df, leaky_splits):
        result = validate_splits(cells_df, leaky_splits)
        assert result["valid"] is False
        assert any("Train-Test" in issue for issue in result["issues"])

    def test_detects_train_val_leakage(self, cells_df):
        leaky = {
            "folds": [{
                "train_donors": ["D1", "D2"],
                "val_donors": ["D2"],  # D2 in both!
                "test_donors": ["D3"],
            }]
        }
        result = validate_splits(cells_df, leaky)
        assert result["valid"] is False
        assert any("Train-Val" in issue for issue in result["issues"])

    def test_detects_val_test_leakage(self, cells_df):
        leaky = {
            "folds": [{
                "train_donors": ["D1"],
                "val_donors": ["D2"],
                "test_donors": ["D2"],  # D2 in both!
            }]
        }
        result = validate_splits(cells_df, leaky)
        assert result["valid"] is False
        assert any("Val-Test" in issue for issue in result["issues"])

    def test_warns_on_missing_donors(self, cells_df, valid_splits):
        # Modify splits to miss D4
        valid_splits["folds"][0]["test_donors"] = []
        result = validate_splits(cells_df, valid_splits)
        # Should still be valid but have warnings
        assert len(result["warnings"]) > 0

    def test_computes_cell_counts(self, cells_df, valid_splits):
        result = validate_splits(cells_df, valid_splits)
        fold_0 = result["summary"]["fold_0"]

        assert "train_cells" in fold_0
        assert "val_cells" in fold_0
        assert "test_cells" in fold_0
        assert fold_0["train_cells"] == 60  # D1 + D2 = 30 + 30

    def test_alternate_key_names(self, cells_df):
        splits = {
            "folds": [{
                "train": ["D1", "D2"],  # Using 'train' instead of 'train_donors'
                "val": ["D3"],
                "test": ["D4"],
            }]
        }
        result = validate_splits(cells_df, splits)
        assert result["valid"] is True


class TestValidateSplitsFromFiles:
    def test_loads_and_validates(self, cells_df, valid_splits):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            cells_df.to_parquet(tmpdir / "cells.parquet")
            with open(tmpdir / "split_manifest.json", "w") as f:
                json.dump(valid_splits, f)

            result = validate_splits_from_files(
                tmpdir / "cells.parquet",
                tmpdir / "split_manifest.json"
            )

            assert result["valid"] is True


class TestCheckPairedSampleLeakage:
    def test_no_leakage_when_donors_separated(self, cells_df, valid_splits):
        result = check_paired_sample_leakage(cells_df, valid_splits)
        assert result["has_paired_leakage"] is False

    def test_detects_paired_sample_leakage(self, cells_df):
        # Create scenario where same donor has samples in train and test
        splits = {
            "folds": [{
                "train_donors": ["D1"],
                "val_donors": ["D2"],
                "test_donors": ["D3", "D4"],
            }]
        }
        result = check_paired_sample_leakage(cells_df, splits)
        # No leakage here since donors are fully separated
        assert result["has_paired_leakage"] is False

    def test_handles_missing_sample_column(self, cells_df, valid_splits):
        cells_no_sample = cells_df.drop(columns=["sample_id"])
        result = check_paired_sample_leakage(cells_no_sample, valid_splits)
        assert "error" in result["summary"]
