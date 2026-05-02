"""Dynamic trajectory analysis and visualization for StageBridge.

Implements geodesic-aware trajectory inference and visualization for
stage progression analysis. Key capabilities:
- Optimal transport-based cell fate probability estimation
- Smooth trajectory interpolation in latent space
- Dynamic driver gene identification along progression paths
- Publication-quality trajectory visualizations
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist
from tqdm import tqdm

if TYPE_CHECKING:
    from stagebridge.models import StageBridge


def _scaled_normalize(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Min-max normalize tensor, returning normalized values and bounds."""
    z_min = z.min(dim=0, keepdim=True)[0]
    z_max = z.max(dim=0, keepdim=True)[0]
    z_norm = (z - z_min) / (z_max - z_min + 1e-8)
    return z_norm, z_min, z_max


def _kde_probability_measure(data: pd.DataFrame, bandwidth: float = 1.0) -> np.ndarray:
    """Compute KDE-based probability measure for optimal transport."""
    from scipy.stats import gaussian_kde

    if len(data) < 2:
        return np.ones(len(data)) / len(data)

    try:
        kde = gaussian_kde(data.values.T, bw_method=bandwidth)
        densities = kde(data.values.T)
        densities = densities / densities.sum()
        return densities
    except Exception:
        return np.ones(len(data)) / len(data)


def _compute_ot_plan(
    source_latent: np.ndarray,
    target_latent: np.ndarray,
    reg: float = 0.02,
) -> np.ndarray:
    """Compute optimal transport plan between source and target distributions."""
    import ot

    source_df = pd.DataFrame(source_latent)
    target_df = pd.DataFrame(target_latent)

    C = cdist(source_df.values, target_df.values, metric='euclidean')

    mu = _kde_probability_measure(source_df)
    nu = _kde_probability_measure(target_df)

    P = ot.sinkhorn(mu, nu, C, reg=reg)
    P_normalized = P / P.sum(axis=1, keepdims=True)

    return P_normalized


@dataclass
class FateProbability:
    """Cell fate probability estimates.

    Attributes:
        cell_ids: Cell identifiers
        stage_probs: Dict mapping target stage -> probability array
        assigned_fate: Assigned fate label per cell
        confidence: Confidence score per cell
    """
    cell_ids: np.ndarray
    stage_probs: dict[str, np.ndarray]
    assigned_fate: np.ndarray
    confidence: np.ndarray

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame."""
        df = pd.DataFrame({
            'cell_id': self.cell_ids,
            'assigned_fate': self.assigned_fate,
            'confidence': self.confidence,
        })
        for stage, probs in self.stage_probs.items():
            df[f'prob_{stage}'] = probs
        return df


@dataclass
class DynamicDriverResult:
    """Dynamic driver gene analysis results.

    Attributes:
        gene_names: Gene names
        driver_index_matrix: Time x genes matrix of driver indices
        top_genes: Top driver genes ranked by importance
        temporal_clusters: Cluster assignments (early/mid/late)
    """
    gene_names: list[str]
    driver_index_matrix: np.ndarray
    top_genes: list[str]
    temporal_clusters: np.ndarray | None = None

    def get_top_drivers(self, n: int = 50) -> list[str]:
        """Get top n driver genes by average importance."""
        avg_importance = np.abs(self.driver_index_matrix).mean(axis=0)
        top_idx = np.argsort(avg_importance)[::-1][:n]
        return [self.gene_names[i] for i in top_idx]


class TrajectoryAnalysis:
    """Trajectory analysis for stage progression.

    Computes cell fate probabilities, interpolates trajectories,
    and identifies dynamic driver genes along progression paths.
    """

    def __init__(
        self,
        model: "StageBridge",
        device: str | torch.device = "cpu",
    ):
        self.model = model
        self.device = device
        self.model.eval()
        self.model.to(device)

        self._pca = None
        self._latent_bounds = None

    def compute_fate_probabilities(
        self,
        source_data: torch.Tensor,
        target_data: torch.Tensor,
        target_stages: list[str],
        stage_masks: dict[str, np.ndarray],
        reg: float = 0.02,
        threshold_score: float = 0.5,
        threshold_weight: float = 0.5,
    ) -> FateProbability:
        """Compute cell fate probabilities via optimal transport.

        Args:
            source_data: Source cell expression (cells to predict fate for)
            target_data: Target cell expression (terminal states)
            target_stages: List of target stage names
            stage_masks: Dict mapping stage name -> boolean mask over target_data
            reg: Sinkhorn regularization
            threshold_score: Score threshold for fate assignment
            threshold_weight: Weight threshold for confidence

        Returns:
            FateProbability with fate estimates
        """
        with torch.no_grad():
            source_latent = self._encode_to_latent(source_data)
            target_latent = self._encode_to_latent(target_data)

        stage_probs = {}
        for stage in target_stages:
            mask = stage_masks[stage]
            stage_target = target_latent[mask]

            P = _compute_ot_plan(source_latent, stage_target, reg=reg)
            stage_probs[stage] = P.sum(axis=1)

        assigned_fate, confidence = self._assign_fates(
            stage_probs, target_stages, threshold_score, threshold_weight
        )

        return FateProbability(
            cell_ids=np.arange(len(source_data)),
            stage_probs=stage_probs,
            assigned_fate=assigned_fate,
            confidence=confidence,
        )

    def _encode_to_latent(self, data: torch.Tensor) -> np.ndarray:
        """Encode data to latent space via model's niche encoder."""
        data = data.to(self.device)
        with torch.no_grad():
            output = self.model.encode(data)
            latent = output.detach().cpu().numpy()
        return latent

    def _assign_fates(
        self,
        stage_probs: dict[str, np.ndarray],
        stages: list[str],
        threshold_score: float,
        threshold_weight: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Assign fates based on probability scores."""
        n_cells = len(list(stage_probs.values())[0])

        if len(stages) == 2:
            p1, p2 = stage_probs[stages[0]], stage_probs[stages[1]]
            score = np.log2((p1 + 1e-8) / (p2 + 1e-8))
            max_weight = np.maximum(p1, p2)

            assigned = np.full(n_cells, 'Undetermined', dtype=object)
            assigned[(score > threshold_score) & (max_weight > threshold_weight)] = stages[0]
            assigned[(score < -threshold_score) & (max_weight > threshold_weight)] = stages[1]
            assigned[(np.abs(score) <= threshold_score) & (max_weight > threshold_weight)] = 'Intermediate'
            assigned[max_weight <= threshold_weight] = 'Low_Confidence'

            confidence = max_weight
        else:
            all_probs = np.stack([stage_probs[s] for s in stages], axis=1)
            max_idx = np.argmax(all_probs, axis=1)
            max_prob = np.max(all_probs, axis=1)

            assigned = np.array([stages[i] for i in max_idx])
            assigned[max_prob < threshold_weight] = 'Low_Confidence'
            confidence = max_prob

        return assigned, confidence

    def interpolate_trajectory(
        self,
        source_latent: np.ndarray,
        target_latent: np.ndarray,
        n_steps: int = 100,
        reg: float = 0.02,
    ) -> np.ndarray:
        """Interpolate trajectory between source and target in latent space.

        Uses optimal transport to match source cells to targets,
        then linear interpolation in latent space.

        Args:
            source_latent: Source latent coordinates
            target_latent: Target latent coordinates
            n_steps: Number of interpolation steps
            reg: OT regularization

        Returns:
            Array of shape (n_steps, n_cells, latent_dim)
        """
        P = _compute_ot_plan(source_latent, target_latent, reg=reg)

        target_matched = np.dot(P, target_latent)

        t_values = np.linspace(0.0, 1.0, n_steps)
        trajectory = np.stack([
            (1 - t) * source_latent + t * target_matched
            for t in t_values
        ])

        return trajectory

    def compute_dynamic_drivers(
        self,
        data: torch.Tensor,
        source_mask: np.ndarray,
        target_mask: np.ndarray,
        gene_names: list[str],
        fate_mask: np.ndarray | None = None,
        reg: float = 0.02,
        n_steps: int = 50,
    ) -> DynamicDriverResult:
        """Compute dynamic driver genes along trajectory.

        Driver index = gradient of latent w.r.t. genes * velocity.
        Identifies genes driving the transition at each time point.

        Args:
            data: Full expression data
            source_mask: Boolean mask for source cells
            target_mask: Boolean mask for target cells
            gene_names: Gene names
            fate_mask: Optional mask for specific fate subset
            reg: OT regularization
            n_steps: Number of time steps

        Returns:
            DynamicDriverResult with driver indices
        """
        data = data.to(self.device)

        with torch.no_grad():
            latent_all = self._encode_to_latent(data.cpu())
            z_min = latent_all.min(axis=0, keepdims=True)
            z_max = latent_all.max(axis=0, keepdims=True)

        source_latent = (latent_all[source_mask] - z_min) / (z_max - z_min + 1e-8)
        target_latent = (latent_all[target_mask] - z_min) / (z_max - z_min + 1e-8)

        P = _compute_ot_plan(source_latent, target_latent, reg=reg)
        target_matched = np.dot(P, target_latent)

        if fate_mask is not None:
            source_latent = source_latent[fate_mask]
            target_matched = target_matched[fate_mask]

        velocity = torch.from_numpy(target_matched - source_latent).float().to(self.device)

        t_values = np.linspace(0.0, 1.0, n_steps)[:-1]
        driver_indices = []

        source_t = torch.from_numpy(source_latent).float().to(self.device)
        target_t = torch.from_numpy(target_matched).float().to(self.device)

        for t in tqdm(t_values, desc="Computing drivers"):
            inter_latent = (1 - t) * source_t + t * target_t
            inter_latent_unnorm = inter_latent * (z_max - z_min + 1e-8) + z_min
            inter_latent_unnorm = torch.from_numpy(inter_latent_unnorm).float().to(self.device)

            with torch.no_grad():
                data_inter = self.model.decode(inter_latent_unnorm)

            data_inter.requires_grad = True
            latent_out = self.model.encode(data_inter)
            latent_scaled = (latent_out - torch.from_numpy(z_min).to(self.device)) / (
                torch.from_numpy(z_max - z_min + 1e-8).to(self.device)
            )

            grad_outputs = velocity[:len(latent_scaled)]
            gradients = torch.autograd.grad(
                outputs=latent_scaled,
                inputs=data_inter,
                grad_outputs=grad_outputs,
                create_graph=False,
                only_inputs=True,
            )[0]

            driver_index = gradients.sum(dim=0).detach().cpu().numpy()
            driver_indices.append(driver_index)

        driver_matrix = np.stack(driver_indices)

        avg_importance = np.abs(driver_matrix).mean(axis=0)
        top_idx = np.argsort(avg_importance)[::-1][:100]
        top_genes = [gene_names[i] for i in top_idx]

        return DynamicDriverResult(
            gene_names=gene_names,
            driver_index_matrix=driver_matrix,
            top_genes=top_genes,
        )


def cluster_driver_genes(
    driver_result: DynamicDriverResult,
    n_clusters: int = 3,
) -> DynamicDriverResult:
    """Cluster driver genes by temporal activity pattern.

    Identifies early, mid, and late driver gene modules.
    """
    from scipy.cluster.hierarchy import linkage, fcluster

    top_idx = [driver_result.gene_names.index(g) for g in driver_result.top_genes[:100]]
    data = driver_result.driver_index_matrix[:, top_idx].T

    data_norm = (data - data.mean(axis=1, keepdims=True)) / (data.std(axis=1, keepdims=True) + 1e-8)

    Z = linkage(data_norm, method='ward')
    clusters = fcluster(Z, t=n_clusters, criterion='maxclust')

    driver_result.temporal_clusters = clusters
    return driver_result
