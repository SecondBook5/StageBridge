"""Flow field analysis: drift, curl, divergence, irreversibility.

These metrics characterize the learned velocity field from OT-CFM:
- Drift magnitude: How fast cells are moving in latent space
- Divergence: Source/sink regions (proliferation/death)
- Curl: Rotational dynamics (cycling behavior)
- Irreversibility: Entropy production rate (deviation from equilibrium)

Based on:
- Hashimoto et al. (2016): Learning Population-Level Diffusions
- Li et al. (2020): Scalable Gradients for Stochastic Differential Equations
- Tong et al. (2023): Conditional Flow Matching
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import cKDTree


@dataclass
class FlowFieldMetrics:
    """Container for computed flow field metrics."""

    drift_magnitude: np.ndarray  # [N] magnitude of velocity
    divergence: np.ndarray  # [N] local divergence (source/sink)
    curl_magnitude: np.ndarray  # [N] magnitude of curl (rotation)
    irreversibility: np.ndarray  # [N] entropy production rate

    # Optional: full vectors
    velocity: np.ndarray | None = None  # [N, D] velocity vectors
    curl_vector: np.ndarray | None = None  # [N, D] curl vectors (3D projection)


class FlowFieldAnalyzer:
    """Analyze flow field properties from learned velocity model.

    Computes differential properties of the velocity field using
    finite differences on a local neighborhood.

    Args:
        n_neighbors: Number of neighbors for gradient estimation
        epsilon: Perturbation size for numerical derivatives
    """

    def __init__(self, n_neighbors: int = 20, epsilon: float = 0.01):
        self.n_neighbors = n_neighbors
        self.epsilon = epsilon

    def compute_metrics(
        self,
        embeddings: np.ndarray,
        velocities: np.ndarray,
        spatial_coords: np.ndarray | None = None,
    ) -> FlowFieldMetrics:
        """Compute flow field metrics from embeddings and velocities.

        Args:
            embeddings: [N, D] latent embeddings
            velocities: [N, D] velocity vectors (displacement/drift)
            spatial_coords: [N, 2] optional spatial coordinates for spatial metrics

        Returns:
            FlowFieldMetrics with drift, divergence, curl, irreversibility
        """
        N, D = embeddings.shape

        # Build KD-tree for neighbor queries
        tree = cKDTree(embeddings)

        # Compute metrics for each point
        drift_mag = np.linalg.norm(velocities, axis=1)
        divergence = np.zeros(N)
        curl_mag = np.zeros(N)
        irreversibility = np.zeros(N)

        for i in range(N):
            # Get neighbors
            dists, idx = tree.query(embeddings[i], k=self.n_neighbors + 1)
            idx = idx[1:]  # Exclude self

            # Local coordinate system centered at point i
            delta_x = embeddings[idx] - embeddings[i]  # [K, D]
            delta_v = velocities[idx] - velocities[i]  # [K, D]

            # Estimate Jacobian via least squares: delta_v ≈ J @ delta_x
            # J is [D, D], we solve for each row
            try:
                J = np.linalg.lstsq(delta_x, delta_v, rcond=None)[0].T  # [D, D]

                # Divergence = trace(J)
                divergence[i] = np.trace(J)

                # Curl magnitude (generalized to high-D as antisymmetric part)
                antisym = (J - J.T) / 2
                curl_mag[i] = np.linalg.norm(antisym, 'fro')

                # Irreversibility: based on entropy production
                # EPR ∝ (v - D∇log(ρ))² where D is diffusion, ρ is density
                # Simplified: use velocity magnitude weighted by local divergence
                sym = (J + J.T) / 2
                irreversibility[i] = np.abs(divergence[i]) * drift_mag[i]

            except np.linalg.LinAlgError:
                # Fallback if singular
                divergence[i] = 0
                curl_mag[i] = 0
                irreversibility[i] = 0

        return FlowFieldMetrics(
            drift_magnitude=drift_mag,
            divergence=divergence,
            curl_magnitude=curl_mag,
            irreversibility=irreversibility,
            velocity=velocities,
        )

    def compute_from_model(
        self,
        model: nn.Module,
        embeddings: torch.Tensor,
        t: float = 0.5,
        device: str = "cuda",
    ) -> FlowFieldMetrics:
        """Compute metrics by querying the velocity model directly.

        For OT-CFM models that can output velocity at any time t.

        Args:
            model: Model with .velocity(x, t) method
            embeddings: [N, D] latent embeddings
            t: Time point to evaluate velocity (0=start, 1=end)
            device: Device for computation

        Returns:
            FlowFieldMetrics
        """
        model.eval()
        embeddings = embeddings.to(device)

        with torch.no_grad():
            # Get velocities from model
            t_tensor = torch.full((len(embeddings), 1), t, device=device)
            velocities = model.velocity(embeddings, t_tensor)

        return self.compute_metrics(
            embeddings.cpu().numpy(),
            velocities.cpu().numpy(),
        )


def compute_spatial_gradients(
    values: np.ndarray,
    coords: np.ndarray,
    n_neighbors: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute spatial gradients of a scalar field.

    Args:
        values: [N] scalar values
        coords: [N, 2] spatial coordinates
        n_neighbors: Number of neighbors for gradient estimation

    Returns:
        grad_x, grad_y: [N] gradient components
    """
    N = len(values)
    tree = cKDTree(coords)

    grad_x = np.zeros(N)
    grad_y = np.zeros(N)

    for i in range(N):
        dists, idx = tree.query(coords[i], k=n_neighbors + 1)
        idx = idx[1:]

        delta_xy = coords[idx] - coords[i]  # [K, 2]
        delta_v = values[idx] - values[i]  # [K]

        try:
            grad = np.linalg.lstsq(delta_xy, delta_v, rcond=None)[0]
            grad_x[i] = grad[0]
            grad_y[i] = grad[1]
        except np.linalg.LinAlgError:
            pass

    return grad_x, grad_y


def compute_spatial_divergence(
    vector_field: np.ndarray,
    coords: np.ndarray,
    n_neighbors: int = 10,
) -> np.ndarray:
    """Compute divergence of a 2D vector field on spatial coordinates.

    Args:
        vector_field: [N, 2] vector field
        coords: [N, 2] spatial coordinates
        n_neighbors: Number of neighbors

    Returns:
        [N] divergence values
    """
    grad_vx_x, _ = compute_spatial_gradients(vector_field[:, 0], coords, n_neighbors)
    _, grad_vy_y = compute_spatial_gradients(vector_field[:, 1], coords, n_neighbors)

    return grad_vx_x + grad_vy_y


def compute_spatial_curl(
    vector_field: np.ndarray,
    coords: np.ndarray,
    n_neighbors: int = 10,
) -> np.ndarray:
    """Compute curl (z-component) of a 2D vector field.

    Args:
        vector_field: [N, 2] vector field
        coords: [N, 2] spatial coordinates
        n_neighbors: Number of neighbors

    Returns:
        [N] curl values (scalar, z-component)
    """
    _, grad_vx_y = compute_spatial_gradients(vector_field[:, 0], coords, n_neighbors)
    grad_vy_x, _ = compute_spatial_gradients(vector_field[:, 1], coords, n_neighbors)

    return grad_vy_x - grad_vx_y
