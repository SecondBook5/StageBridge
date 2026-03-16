"""
Evaluation metrics for spatial backend comparison.

Provides both upstream (spatial quality) and downstream (StageBridge utility) metrics.
"""

from dataclasses import dataclass, field
from typing import Any
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors

from .base import BackendMappingResult, compute_cell_type_entropy, compute_sparsity


@dataclass
class MetricsReport:
    """
    Comprehensive metrics report for a spatial backend.

    Contains upstream metrics (spatial quality), downstream metrics (StageBridge utility),
    and backend metadata.
    """

    # Backend identification
    backend_name: str

    # Upstream metrics: spatial quality
    upstream_metrics: dict[str, float] = field(default_factory=dict)

    # Downstream metrics: StageBridge utility
    downstream_metrics: dict[str, float] = field(default_factory=dict)

    # Spatial coherence metrics
    spatial_metrics: dict[str, float] = field(default_factory=dict)

    # Donor robustness metrics
    robustness_metrics: dict[str, float] = field(default_factory=dict)

    # Runtime and resource metrics
    runtime_metrics: dict[str, float] = field(default_factory=dict)

    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to flat dictionary for comparison tables."""
        result = {"backend": self.backend_name}

        # Flatten all metric categories
        for prefix, metrics in [
            ("upstream", self.upstream_metrics),
            ("downstream", self.downstream_metrics),
            ("spatial", self.spatial_metrics),
            ("robustness", self.robustness_metrics),
            ("runtime", self.runtime_metrics),
        ]:
            for key, value in metrics.items():
                result[f"{prefix}_{key}"] = value

        return result

    def get_summary_score(self, weights: dict[str, float] | None = None) -> float:
        """
        Compute weighted summary score for backend comparison.

        Args:
            weights: Optional weights for different metric categories

        Returns:
            Weighted summary score (higher is better)
        """
        if weights is None:
            weights = {
                "upstream": 0.3,
                "downstream": 0.4,
                "spatial": 0.2,
                "robustness": 0.1,
            }

        scores = {
            "upstream": self._compute_category_score(self.upstream_metrics),
            "downstream": self._compute_category_score(self.downstream_metrics),
            "spatial": self._compute_category_score(self.spatial_metrics),
            "robustness": self._compute_category_score(self.robustness_metrics),
        }

        total = sum(weights.get(cat, 0) * score for cat, score in scores.items())

        return total

    def _compute_category_score(self, metrics: dict[str, float]) -> float:
        """Compute normalized score for a metric category."""
        if not metrics:
            return 0.5  # Neutral score if no metrics

        # Average of metrics (assuming they're already normalized to [0, 1])
        values = [v for v in metrics.values() if isinstance(v, (int, float)) and not np.isnan(v)]
        return np.mean(values) if values else 0.5


def compute_upstream_metrics(
    result: BackendMappingResult,
    spatial_expression: pd.DataFrame | None = None,
    held_out_genes: list[str] | None = None,
) -> dict[str, float]:
    """
    Compute upstream quality metrics for a spatial mapping result.

    Metrics computed:
    - mean_entropy: Average cell type entropy across spots (diversity)
    - std_entropy: Standard deviation of entropy (homogeneity)
    - sparsity: Fraction of zero proportions
    - coverage: Fraction of spots with confident mapping
    - gene_reconstruction_error: MSE of reconstructed vs original genes (if available)
    - max_proportion_mean: Average maximum proportion per spot
    - n_dominant_types: Number of cell types that dominate any spot

    Args:
        result: BackendMappingResult from a backend
        spatial_expression: Original spatial expression matrix (for reconstruction)
        held_out_genes: Genes to use for reconstruction error (if available)

    Returns:
        Dictionary of metric name to value
    """
    proportions = result.cell_type_proportions
    confidence = result.confidence

    # Cell type entropy
    entropy = compute_cell_type_entropy(proportions)

    # Sparsity
    sparsity = compute_sparsity(proportions)

    # Coverage (fraction with confident mapping)
    coverage = (confidence > 0.5).mean()

    # Max proportion statistics
    max_proportions = proportions.max(axis=1)

    # Dominant cell types (>50% in any spot)
    dominant_types = (proportions > 0.5).any(axis=0).sum()

    metrics = {
        "mean_entropy": float(entropy.mean()),
        "std_entropy": float(entropy.std()),
        "sparsity": float(sparsity),
        "coverage": float(coverage),
        "max_proportion_mean": float(max_proportions.mean()),
        "max_proportion_std": float(max_proportions.std()),
        "n_dominant_types": int(dominant_types),
        "n_spots": len(proportions),
        "n_celltypes": proportions.shape[1],
    }

    # Gene reconstruction error (if possible)
    if result.reconstructed_expression is not None and spatial_expression is not None:
        common_genes = result.reconstructed_expression.columns.intersection(
            spatial_expression.columns
        )
        if held_out_genes:
            common_genes = common_genes.intersection(held_out_genes)

        if len(common_genes) > 0:
            recon = result.reconstructed_expression[common_genes].values
            orig = spatial_expression[common_genes].values

            # Normalize for comparison
            recon_norm = (recon - recon.mean(axis=0)) / (recon.std(axis=0) + 1e-10)
            orig_norm = (orig - orig.mean(axis=0)) / (orig.std(axis=0) + 1e-10)

            mse = np.mean((recon_norm - orig_norm) ** 2)
            correlation = np.corrcoef(recon_norm.flatten(), orig_norm.flatten())[0, 1]

            metrics["gene_reconstruction_mse"] = float(mse)
            metrics["gene_reconstruction_corr"] = float(correlation)

    return metrics


def compute_downstream_utility(
    result: BackendMappingResult,
    transition_data: dict[str, Any] | None = None,
) -> dict[str, float]:
    """
    Compute downstream utility metrics for StageBridge.

    This is a proxy for how useful the spatial mapping will be for
    transition modeling. It evaluates:
    - Proportion stability (for stable transition inputs)
    - Cell type coverage (for diverse transition modeling)
    - Confidence distribution (for reliable assignments)
    - Entropy distribution (for mixture modeling)

    Args:
        result: BackendMappingResult from a backend
        transition_data: Optional transition data for direct utility assessment

    Returns:
        Dictionary of metric name to value
    """
    proportions = result.cell_type_proportions
    confidence = result.confidence

    metrics = {}

    # 1. Proportion stability: low variance across similar spots is good
    # Use coefficient of variation as stability measure
    cv_per_type = proportions.std(axis=0) / (proportions.mean(axis=0) + 1e-10)
    metrics["proportion_stability"] = float(1.0 - cv_per_type.mean())

    # 2. Cell type coverage: fraction of cell types with non-trivial presence
    significant_presence = proportions.mean(axis=0) > 0.01
    metrics["celltype_coverage"] = float(significant_presence.mean())

    # 3. Confidence quality: high and consistent confidence is good
    metrics["confidence_mean"] = float(confidence.mean())
    metrics["confidence_std"] = float(confidence.std())
    metrics["confidence_quality"] = float(confidence.mean() * (1 - confidence.std()))

    # 4. Entropy quality for mixtures
    # Good for transition: moderate entropy (mixtures, not extremes)
    entropy = compute_cell_type_entropy(proportions)

    # Optimal entropy is around 0.3-0.7 (not too uniform, not too sparse)
    entropy_quality = 1.0 - 2 * np.abs(entropy - 0.5)
    metrics["entropy_quality"] = float(entropy_quality.mean())

    # 5. Transition support: can we identify clear transitions?
    # High max proportion spots indicate clear identities for transition anchors
    max_props = proportions.max(axis=1)
    transition_anchors = (max_props > 0.7).mean()
    metrics["transition_anchor_fraction"] = float(transition_anchors)

    # 6. If transition data is provided, compute direct utility
    if transition_data is not None:
        metrics.update(_compute_direct_transition_utility(result, transition_data))

    # Overall downstream utility score (normalized to [0, 1])
    utility_components = [
        metrics["proportion_stability"],
        metrics["celltype_coverage"],
        metrics["confidence_quality"],
        metrics["entropy_quality"],
    ]
    metrics["overall_utility"] = float(np.mean(utility_components))

    return metrics


def _compute_direct_transition_utility(
    result: BackendMappingResult,
    transition_data: dict[str, Any],
) -> dict[str, float]:
    """
    Compute direct transition utility when transition data is available.

    Args:
        result: Spatial mapping result
        transition_data: Dictionary with transition-related data

    Returns:
        Direct utility metrics
    """
    metrics = {}

    # Expected keys in transition_data:
    # - source_types: cell types at source stage
    # - target_types: cell types at target stage
    # - known_transitions: list of (source, target) tuples

    proportions = result.cell_type_proportions

    if "source_types" in transition_data and "target_types" in transition_data:
        source_types = transition_data["source_types"]
        target_types = transition_data["target_types"]

        # Check if mapping covers transition-relevant types
        mapped_types = set(proportions.columns)
        source_coverage = len(mapped_types.intersection(source_types)) / len(source_types)
        target_coverage = len(mapped_types.intersection(target_types)) / len(target_types)

        metrics["source_type_coverage"] = float(source_coverage)
        metrics["target_type_coverage"] = float(target_coverage)

    if "known_transitions" in transition_data:
        # Check if proportions support known transitions
        # (spots with source type should have spatial neighbors with target type)
        known = transition_data["known_transitions"]
        supported = 0

        for source, target in known:
            if source in proportions.columns and target in proportions.columns:
                # Spots with high source proportion
                source_spots = proportions[source] > 0.3
                # Check if target also present (transition signal)
                target_present = proportions.loc[source_spots, target] > 0.1
                if target_present.any():
                    supported += 1

        metrics["transition_support_rate"] = float(supported / len(known) if known else 0)

    return metrics


def compute_spatial_coherence(
    result: BackendMappingResult,
    spatial_coords: np.ndarray,
    k_neighbors: int = 6,
) -> dict[str, float]:
    """
    Compute spatial coherence metrics.

    Measures how spatially smooth/coherent the mapping is.
    Good spatial coherence means nearby spots have similar compositions.

    Args:
        result: BackendMappingResult from a backend
        spatial_coords: (n_spots, 2) array of spatial coordinates
        k_neighbors: Number of neighbors for local coherence

    Returns:
        Dictionary of spatial coherence metrics
    """
    proportions = result.cell_type_proportions.values
    n_spots = len(proportions)

    if n_spots < k_neighbors + 1:
        return {"spatial_coherence": np.nan, "local_smoothness": np.nan}

    # Build k-NN graph
    nn = NearestNeighbors(n_neighbors=k_neighbors + 1, metric="euclidean")
    nn.fit(spatial_coords)
    distances, indices = nn.kneighbors(spatial_coords)

    # Exclude self (first neighbor)
    neighbor_indices = indices[:, 1:]

    # 1. Local coherence: correlation of proportions with neighbors
    local_coherences = []
    for i in range(n_spots):
        neighbors = neighbor_indices[i]
        spot_props = proportions[i]
        neighbor_props = proportions[neighbors].mean(axis=0)

        # Pearson correlation
        if spot_props.std() > 0 and neighbor_props.std() > 0:
            corr = np.corrcoef(spot_props, neighbor_props)[0, 1]
            local_coherences.append(corr)

    local_coherence = np.nanmean(local_coherences) if local_coherences else np.nan

    # 2. Spatial smoothness: low variation in local neighborhoods
    local_variations = []
    for i in range(n_spots):
        neighbors = neighbor_indices[i]
        local_group = proportions[np.concatenate([[i], neighbors])]
        variation = local_group.std(axis=0).mean()
        local_variations.append(variation)

    # Convert to smoothness (inverse of variation)
    smoothness = 1.0 - np.mean(local_variations)

    # 3. Spatial autocorrelation (Moran's I approximation)
    # Simplified: correlation of dominant cell type across neighbors
    dominant_types = np.argmax(proportions, axis=1)
    neighbor_agreement = []
    for i in range(n_spots):
        neighbors = neighbor_indices[i]
        agreement = (dominant_types[neighbors] == dominant_types[i]).mean()
        neighbor_agreement.append(agreement)

    spatial_autocorr = np.mean(neighbor_agreement)

    # 4. Niche coherence: do spots form coherent niches?
    # Measure clustering of similar compositions
    from sklearn.cluster import KMeans

    n_clusters = min(10, n_spots // 5)
    if n_clusters >= 2:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(proportions)

        # Measure spatial compactness of clusters
        cluster_compactness = []
        for c in range(n_clusters):
            cluster_mask = cluster_labels == c
            if cluster_mask.sum() > 1:
                cluster_coords = spatial_coords[cluster_mask]
                centroid = cluster_coords.mean(axis=0)
                distances_to_centroid = np.linalg.norm(cluster_coords - centroid, axis=1)
                compactness = 1.0 / (1.0 + distances_to_centroid.mean())
                cluster_compactness.append(compactness)

        niche_coherence = np.mean(cluster_compactness) if cluster_compactness else np.nan
    else:
        niche_coherence = np.nan

    return {
        "local_coherence": float(local_coherence),
        "spatial_smoothness": float(smoothness),
        "spatial_autocorrelation": float(spatial_autocorr),
        "niche_coherence": float(niche_coherence),
    }


def compute_donor_robustness(
    results_by_donor: dict[str, BackendMappingResult],
) -> dict[str, float]:
    """
    Compute cross-donor robustness metrics.

    Measures consistency of mapping results across different donors.
    High robustness means the backend produces stable results regardless of donor.

    Args:
        results_by_donor: Dictionary mapping donor ID to BackendMappingResult

    Returns:
        Dictionary of robustness metrics
    """
    if len(results_by_donor) < 2:
        return {
            "donor_consistency": np.nan,
            "celltype_stability": np.nan,
            "confidence_stability": np.nan,
        }

    # Collect statistics per donor
    donor_stats = {}
    for donor_id, result in results_by_donor.items():
        props = result.cell_type_proportions
        conf = result.confidence

        donor_stats[donor_id] = {
            "mean_proportions": props.mean(axis=0),
            "entropy_mean": compute_cell_type_entropy(props).mean(),
            "confidence_mean": conf.mean(),
            "sparsity": compute_sparsity(props),
        }

    # 1. Cell type proportion consistency across donors
    all_mean_props = pd.DataFrame({d: s["mean_proportions"] for d, s in donor_stats.items()})

    # Coefficient of variation across donors (lower is more consistent)
    prop_cv = all_mean_props.std(axis=1) / (all_mean_props.mean(axis=1) + 1e-10)
    celltype_stability = 1.0 - prop_cv.mean()

    # 2. Pairwise correlation of mean proportions
    donor_ids = list(donor_stats.keys())
    correlations = []
    for i in range(len(donor_ids)):
        for j in range(i + 1, len(donor_ids)):
            corr = np.corrcoef(all_mean_props[donor_ids[i]], all_mean_props[donor_ids[j]])[0, 1]
            correlations.append(corr)

    donor_consistency = np.mean(correlations) if correlations else np.nan

    # 3. Confidence stability across donors
    conf_means = [s["confidence_mean"] for s in donor_stats.values()]
    conf_stability = 1.0 - (np.std(conf_means) / (np.mean(conf_means) + 1e-10))

    # 4. Entropy consistency
    entropy_means = [s["entropy_mean"] for s in donor_stats.values()]
    entropy_stability = 1.0 - (np.std(entropy_means) / (np.mean(entropy_means) + 1e-10))

    return {
        "donor_consistency": float(donor_consistency),
        "celltype_stability": float(celltype_stability),
        "confidence_stability": float(conf_stability),
        "entropy_stability": float(entropy_stability),
        "n_donors": len(results_by_donor),
    }


def compute_comprehensive_metrics(
    result: BackendMappingResult,
    spatial_coords: np.ndarray | None = None,
    spatial_expression: pd.DataFrame | None = None,
    transition_data: dict[str, Any] | None = None,
    runtime_seconds: float | None = None,
    memory_mb: float | None = None,
) -> MetricsReport:
    """
    Compute comprehensive metrics report for a backend result.

    Args:
        result: BackendMappingResult from a backend
        spatial_coords: Spatial coordinates for coherence metrics
        spatial_expression: Original expression for reconstruction metrics
        transition_data: Transition data for downstream utility
        runtime_seconds: Runtime in seconds
        memory_mb: Peak memory usage in MB

    Returns:
        Complete MetricsReport
    """
    backend_name = result.metadata.get("backend", "unknown")

    # Compute upstream metrics
    upstream = compute_upstream_metrics(result, spatial_expression=spatial_expression)

    # Compute downstream metrics
    downstream = compute_downstream_utility(result, transition_data=transition_data)

    # Compute spatial metrics (if coordinates available)
    if spatial_coords is not None:
        spatial = compute_spatial_coherence(result, spatial_coords)
    else:
        spatial = {}

    # Runtime metrics
    runtime = {}
    if runtime_seconds is not None:
        runtime["runtime_seconds"] = runtime_seconds
    if memory_mb is not None:
        runtime["memory_mb"] = memory_mb

    return MetricsReport(
        backend_name=backend_name,
        upstream_metrics=upstream,
        downstream_metrics=downstream,
        spatial_metrics=spatial,
        runtime_metrics=runtime,
        metadata=result.metadata,
    )
