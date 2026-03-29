"""
Marker gene scoring backend wrapper.

Simple baseline that often outperforms complex methods for rare cell types
(Sun et al. 2026, Nature).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad

from .backend_base import (
    SpatialBackend,
    BackendMappingResult,
    compute_cell_type_entropy,
    compute_sparsity,
)
from .marker_scoring import score_markers_scanpy, get_markers_from_reference
from .lung_markers import get_markers_for_reference, ALL_LUNG_MARKERS


class MarkerScoringBackend(SpatialBackend):
    """
    Marker gene scoring spatial mapping wrapper.

    Configuration options:
    - use_reference_markers: If True, derive markers from reference scRNA; if False, use curated
    - n_markers: Number of top markers per cell type (if deriving from reference)
    - marker_dict: Optional pre-defined marker dictionary
    """

    def __init__(
        self,
        use_reference_markers: bool = True,
        n_markers: int = 50,
        marker_dict: dict[str, list[str]] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.use_reference_markers = use_reference_markers
        self.n_markers = n_markers
        self.marker_dict = marker_dict

    def map(
        self,
        snrna: ad.AnnData,
        spatial: ad.AnnData,
        output_dir: Path | None = None,
    ) -> BackendMappingResult:
        """Run marker gene scoring."""
        # Validate and preprocess
        self.validate_inputs(snrna, spatial)
        snrna, spatial = self.preprocess(snrna, spatial)

        # Get or derive markers
        if self.marker_dict is not None:
            markers = self.marker_dict
            print(f"MarkerScoring: Using provided markers for {len(markers)} cell types")
        elif self.use_reference_markers:
            print(f"MarkerScoring: Deriving markers from reference (top {self.n_markers} per type)")
            from .marker_scoring import get_markers_from_reference as derive_markers
            markers = derive_markers(
                snrna,
                cell_type_key="cell_type",
                n_markers=self.n_markers,
            )
        else:
            print("MarkerScoring: Using curated lung markers")
            # Get markers for the cell types in reference
            ref_cell_types = snrna.obs["cell_type"].unique().tolist()
            markers = get_markers_for_reference(ref_cell_types, ALL_LUNG_MARKERS)

        print(f"MarkerScoring: Scoring {len(markers)} cell types across {len(spatial)} spots")

        # Score markers
        proportions = score_markers_scanpy(
            spatial,
            marker_dict=markers,
            use_raw=False,
            normalize=True,
        )

        # Compute confidence (mean score before normalization)
        # Re-score without normalization to get raw scores
        from .marker_scoring import score_markers_scanpy as score_raw
        raw_scores = score_raw(spatial, marker_dict=markers, use_raw=False, normalize=False)
        confidence = pd.Series(raw_scores.mean(axis=1).values, index=proportions.index)

        # Normalize confidence to [0, 1]
        confidence = (confidence - confidence.min()) / (confidence.max() - confidence.min() + 1e-10)

        # Create result
        result = BackendMappingResult(
            cell_type_proportions=proportions,
            confidence=confidence,
            reconstructed_expression=None,  # Marker scoring doesn't reconstruct
            metadata={
                "backend": "marker_scoring",
                "use_reference_markers": self.use_reference_markers,
                "n_cell_types": len(markers),
                "n_spots": len(spatial),
            },
        )

        # Save if output_dir provided
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save proportions
            proportions.to_parquet(output_dir / "cell_type_proportions.parquet")

            # Save confidence
            pd.DataFrame({"confidence": confidence}).to_parquet(output_dir / "confidence.parquet")

            # Save markers used
            import json
            with open(output_dir / "markers_used.json", "w") as f:
                json.dump(markers, f, indent=2)

            print(f"MarkerScoring: Results saved to {output_dir}")

        return result
