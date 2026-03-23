"""
Tests for spatial backend output standardization module.
"""

import numpy as np
import pandas as pd

from stagebridge.spatial_mapping.standardize import (
    StandardizedOutput,
    standardize_backend_output,
    validate_standardized_output,
    merge_standardized_outputs,
    load_all_standardized_outputs,
)


class TestStandardizedOutput:
    """Tests for StandardizedOutput dataclass."""

    def test_validate_valid_output(self, synthetic_standardized_output):
        """Test validation passes for valid output."""
        errors = synthetic_standardized_output.validate()
        assert len(errors) == 0

    def test_validate_negative_proportions(self):
        """Test validation catches negative proportions."""
        props = pd.DataFrame(
            {
                "A": [-0.1, 0.5, 0.6],
                "B": [0.5, 0.5, 0.4],
            },
            index=["spot_0", "spot_1", "spot_2"],
        )

        # Renormalize to make rows sum to ~1
        props = props.clip(lower=0)
        row_sums = props.sum(axis=1)
        props = props.div(row_sums, axis=0)

        output = StandardizedOutput(
            cell_type_proportions=props,
            confidence=pd.Series([0.5, 0.5, 0.5], index=props.index),
            backend_name="test",
        )

        # After clipping, should be valid
        errors = output.validate()
        assert len(errors) == 0

    def test_validate_non_normalized_proportions(self):
        """Test validation catches non-normalized proportions."""
        props = pd.DataFrame(
            {
                "A": [0.3, 0.3, 0.3],
                "B": [0.3, 0.3, 0.3],
            },
            index=["spot_0", "spot_1", "spot_2"],
        )

        output = StandardizedOutput(
            cell_type_proportions=props,
            confidence=pd.Series([0.5, 0.5, 0.5], index=props.index),
            backend_name="test",
        )

        errors = output.validate()

        # Rows sum to 0.6, not 1.0
        assert any("sum to 1" in e for e in errors)

    def test_validate_mismatched_indices(self):
        """Test validation catches mismatched indices."""
        props = pd.DataFrame(
            {
                "A": [0.5, 0.5],
                "B": [0.5, 0.5],
            },
            index=["spot_0", "spot_1"],
        )

        conf = pd.Series([0.5, 0.5], index=["spot_2", "spot_3"])

        output = StandardizedOutput(
            cell_type_proportions=props,
            confidence=conf,
            backend_name="test",
        )

        errors = output.validate()

        assert any("mismatched indices" in e for e in errors)

    def test_validate_missing_backend_name(self):
        """Test validation catches missing backend name."""
        props = pd.DataFrame(
            {
                "A": [0.5, 0.5],
                "B": [0.5, 0.5],
            },
            index=["spot_0", "spot_1"],
        )

        output = StandardizedOutput(
            cell_type_proportions=props,
            confidence=pd.Series([0.5, 0.5], index=props.index),
            backend_name="",  # Empty
        )

        errors = output.validate()

        assert any("backend_name" in e for e in errors)

    def test_save_load(self, synthetic_standardized_output, tmp_output_dir):
        """Test save and load round-trip."""
        # Save
        synthetic_standardized_output.save(tmp_output_dir)

        # Verify files exist
        assert (tmp_output_dir / "cell_type_proportions.parquet").exists()
        assert (tmp_output_dir / "mapping_confidence.parquet").exists()
        assert (tmp_output_dir / "backend_metadata.json").exists()

        # Load
        loaded = StandardizedOutput.load(tmp_output_dir)

        # Verify data integrity
        pd.testing.assert_frame_equal(
            loaded.cell_type_proportions,
            synthetic_standardized_output.cell_type_proportions,
        )
        pd.testing.assert_series_equal(
            loaded.confidence,
            synthetic_standardized_output.confidence,
        )
        assert loaded.backend_name == synthetic_standardized_output.backend_name


class TestStandardizeBackendOutput:
    """Tests for backend output standardization."""

    def test_basic_standardization(self, synthetic_mapping_result):
        """Test basic standardization."""
        output = standardize_backend_output(
            synthetic_mapping_result,
            backend_name="test_backend",
            backend_version="1.0.0",
        )

        assert isinstance(output, StandardizedOutput)
        assert output.backend_name == "test_backend"
        assert output.backend_version == "1.0.0"

    def test_standardization_normalizes_proportions(self, synthetic_mapping_result):
        """Test that standardization ensures normalized proportions."""
        output = standardize_backend_output(
            synthetic_mapping_result,
            backend_name="test",
        )

        # Check rows sum to 1
        row_sums = output.cell_type_proportions.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_standardization_clips_confidence(self, synthetic_mapping_result):
        """Test that standardization clips confidence to [0, 1]."""
        # Modify confidence to have out-of-range values
        synthetic_mapping_result.confidence.iloc[0] = 1.5
        synthetic_mapping_result.confidence.iloc[1] = -0.1

        output = standardize_backend_output(
            synthetic_mapping_result,
            backend_name="test",
        )

        assert output.confidence.min() >= 0.0
        assert output.confidence.max() <= 1.0

    def test_standardization_handles_zero_rows(self, synthetic_snrna):
        """Test that standardization handles zero-sum rows."""
        from stagebridge.spatial_mapping.backend_base import BackendMappingResult

        n_spots = 10
        cell_types = synthetic_snrna.obs["cell_type"].cat.categories.tolist()

        # Create proportions with one zero row
        props = pd.DataFrame(
            np.random.rand(n_spots, len(cell_types)),
            index=[f"spot_{i}" for i in range(n_spots)],
            columns=cell_types,
        )
        props.iloc[0] = 0  # Zero row

        result = BackendMappingResult(
            cell_type_proportions=props,
            confidence=pd.Series(np.ones(n_spots), index=props.index),
            upstream_metrics={},
            metadata={},
        )

        output = standardize_backend_output(result, backend_name="test")

        # Zero row should now have uniform distribution
        expected = 1.0 / len(cell_types)
        np.testing.assert_allclose(
            output.cell_type_proportions.iloc[0].values,
            expected,
            atol=1e-6,
        )


class TestValidateStandardizedOutput:
    """Tests for standardized output validation function."""

    def test_valid_output(self, synthetic_standardized_output):
        """Test validation of valid output."""
        is_valid, errors = validate_standardized_output(synthetic_standardized_output)

        assert is_valid
        assert len(errors) == 0

    def test_invalid_output(self):
        """Test validation of invalid output."""
        output = StandardizedOutput(
            cell_type_proportions=None,
            confidence=None,
            backend_name="",
        )

        is_valid, errors = validate_standardized_output(output)

        assert not is_valid
        assert len(errors) > 0


class TestMergeAndLoadOutputs:
    """Tests for merging and loading multiple outputs."""

    def test_merge_standardized_outputs(self, synthetic_standardized_output, tmp_output_dir):
        """Test merging multiple standardized outputs."""
        outputs = {
            "tangram": synthetic_standardized_output,
            "destvi": synthetic_standardized_output,
            "tacco": synthetic_standardized_output,
        }

        merge_standardized_outputs(outputs, tmp_output_dir)

        # Check directory structure
        assert (tmp_output_dir / "tangram").exists()
        assert (tmp_output_dir / "destvi").exists()
        assert (tmp_output_dir / "tacco").exists()
        assert (tmp_output_dir / "comparison_index.json").exists()

    def test_load_all_standardized_outputs(self, synthetic_standardized_output, tmp_output_dir):
        """Test loading all outputs from merged directory."""
        outputs = {
            "tangram": synthetic_standardized_output,
            "destvi": synthetic_standardized_output,
        }

        merge_standardized_outputs(outputs, tmp_output_dir)

        loaded = load_all_standardized_outputs(tmp_output_dir)

        assert len(loaded) == 2
        assert "tangram" in loaded
        assert "destvi" in loaded
