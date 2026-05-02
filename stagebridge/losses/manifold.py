"""Manifold-aware losses for StageBridge.

Implements geodesic-preserving losses that ensure:
1. Local isometry: Nearby cells in expression space remain nearby in latent
2. Constant-velocity linear (CVL): Trajectories become straight lines in latent
3. Velocity consistency: Progression direction is consistent across timepoints

These losses enable biologically meaningful interpolation between disease stages.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    pass


def compute_pairwise_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute pairwise Euclidean distances."""
    return torch.cdist(x, y, p=2)


def compute_mmd(x: torch.Tensor, y: torch.Tensor, kernel: str = 'linear') -> torch.Tensor:
    """Compute Maximum Mean Discrepancy between two distributions.

    Args:
        x: Samples from first distribution (n, d)
        y: Samples from second distribution (m, d)
        kernel: Kernel type ('linear', 'rbf')

    Returns:
        MMD value
    """
    if kernel == 'linear':
        xx = torch.mm(x, x.t()).mean()
        yy = torch.mm(y, y.t()).mean()
        xy = torch.mm(x, y.t()).mean()
        return xx + yy - 2 * xy
    elif kernel == 'rbf':
        gamma = 1.0 / x.shape[1]
        xx = torch.exp(-gamma * compute_pairwise_distance(x, x).pow(2)).mean()
        yy = torch.exp(-gamma * compute_pairwise_distance(y, y).pow(2)).mean()
        xy = torch.exp(-gamma * compute_pairwise_distance(x, y).pow(2)).mean()
        return xx + yy - 2 * xy
    else:
        raise ValueError(f"Unknown kernel: {kernel}")


def sinkhorn_ot_plan(
    source: torch.Tensor,
    target: torch.Tensor,
    reg: float = 0.02,
    n_iter: int = 100,
) -> torch.Tensor:
    """Compute optimal transport plan using Sinkhorn algorithm.

    Args:
        source: Source distribution (n, d)
        target: Target distribution (m, d)
        reg: Entropic regularization
        n_iter: Number of Sinkhorn iterations

    Returns:
        Transport plan matrix (n, m)
    """
    C = compute_pairwise_distance(source, target)
    n, m = C.shape

    mu = torch.ones(n, device=C.device) / n
    nu = torch.ones(m, device=C.device) / m

    K = torch.exp(-C / reg)

    u = torch.ones(n, device=C.device)
    for _ in range(n_iter):
        v = nu / (K.t() @ u + 1e-8)
        u = mu / (K @ v + 1e-8)

    P = torch.diag(u) @ K @ torch.diag(v)
    return P


class LocalIsometryLoss(nn.Module):
    """Loss enforcing local distance preservation (isometry).

    Ensures that nearest-neighbor distances in expression space
    are preserved in latent space. This is crucial for meaningful
    interpolation - geodesics in latent space map to geodesics
    in expression space.

    The loss penalizes variance in the ratio of nearest-neighbor
    distances between spaces, normalized by the mean ratio.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        x_input: torch.Tensor,
        z_latent: torch.Tensor,
        stage_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute local isometry loss.

        Args:
            x_input: Input expression data (B, D)
            z_latent: Latent representations (B, d)
            stage_idx: Optional stage labels for per-stage computation

        Returns:
            Isometry loss value
        """
        if stage_idx is None:
            return self._compute_isometry(x_input, z_latent)

        unique_stages = torch.unique(stage_idx)
        losses = []

        for stage in unique_stages:
            mask = stage_idx == stage
            if mask.sum() < 3:
                continue
            loss = self._compute_isometry(x_input[mask], z_latent[mask])
            losses.append(loss)

        if not losses:
            return torch.tensor(0.0, device=x_input.device)

        return torch.stack(losses).mean()

    def _compute_isometry(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Compute isometry for a single group."""
        n = x.shape[0]
        if n < 3:
            return torch.tensor(0.0, device=x.device)

        x_dist = compute_pairwise_distance(x, x)
        z_dist = compute_pairwise_distance(z, z)

        inf_diag = torch.eye(n, device=x.device) * 1e10
        x_dist = x_dist + inf_diag
        z_dist = z_dist + inf_diag

        x_min_dist, x_min_idx = torch.min(x_dist, dim=1)
        z_min_dist = z_dist[torch.arange(n, device=x.device), x_min_idx]

        ratio = x_min_dist / (z_min_dist + 1e-8)

        loss = ratio.var() / (ratio.mean() + 1e-8)
        return loss


class ConstantVelocityLinearLoss(nn.Module):
    """Constant-velocity linear (CVL) loss.

    Forces trajectories in latent space to follow straight-line paths
    with constant velocity. When combined with isometry, this means
    linear interpolation in latent space corresponds to geodesic
    (minimum-action) paths in expression space.

    The loss measures discrepancy between:
    1. Interpolated points in latent space (via OT matching)
    2. True observations at intermediate timepoints

    This is evaluated by mapping interpolated latent points back to
    expression space and computing MMD with true data.
    """

    def __init__(self, ot_reg: float = 0.02, min_gap: int = 2):
        """
        Args:
            ot_reg: Optimal transport regularization
            min_gap: Minimum gap between source and target timepoints
        """
        super().__init__()
        self.ot_reg = ot_reg
        self.min_gap = min_gap

    def forward(
        self,
        x_input: torch.Tensor,
        z_latent: torch.Tensor,
        stage_idx: torch.Tensor,
        decoder_fn: callable,
        z_min: torch.Tensor | None = None,
        z_max: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute CVL loss.

        Args:
            x_input: Input expression data
            z_latent: Latent representations
            stage_idx: Stage labels (ordered)
            decoder_fn: Function to decode latent -> expression
            z_min: Latent minimum for normalization
            z_max: Latent maximum for normalization

        Returns:
            CVL loss value
        """
        unique_stages, inverse_idx = torch.unique(stage_idx, sorted=True, return_inverse=True)
        n_stages = len(unique_stages)

        if n_stages < self.min_gap + 2:
            return torch.tensor(0.0, device=x_input.device)

        if z_min is None:
            z_min = z_latent.min(dim=0, keepdim=True)[0]
        if z_max is None:
            z_max = z_latent.max(dim=0, keepdim=True)[0]

        z_scaled = (z_latent - z_min) / (z_max - z_min + 1e-8)

        stage_indices = list(range(n_stages))
        valid_pairs = [
            (i, j) for i in stage_indices for j in stage_indices
            if abs(j - i) > self.min_gap
        ]

        if not valid_pairs:
            return torch.tensor(0.0, device=x_input.device)

        start_idx, end_idx = random.choice(valid_pairs)

        source_mask = inverse_idx == start_idx
        target_mask = inverse_idx == end_idx

        z_source = z_scaled[source_mask]
        z_target = z_scaled[target_mask]

        P = sinkhorn_ot_plan(z_source, z_target, reg=self.ot_reg)
        P_normalized = P / (P.sum(dim=1, keepdim=True) + 1e-8)

        z_matched = torch.mm(P_normalized, z_target)

        t_start = unique_stages[start_idx].float()
        t_end = unique_stages[end_idx].float()
        time_span = t_end - t_start

        # Guard against division by zero
        if time_span.abs() < 1e-8:
            return torch.tensor(0.0, device=x_input.device, requires_grad=True)

        mmd_values = []

        for stage_i in range(n_stages):
            if stage_i == start_idx or stage_i == end_idx:
                continue

            t_i = unique_stages[stage_i].float()
            alpha = (t_i - t_start) / time_span

            if alpha <= 0 or alpha >= 1:
                continue

            z_interp_scaled = (1 - alpha) * z_source + alpha * z_matched

            z_interp = z_interp_scaled * (z_max - z_min + 1e-8) + z_min

            with torch.no_grad():
                x_interp = decoder_fn(z_interp)

            x_true = x_input[inverse_idx == stage_i]

            mmd = compute_mmd(x_interp, x_true, kernel='linear')
            mmd_values.append(mmd)

        if not mmd_values:
            return torch.tensor(0.0, device=x_input.device)

        return torch.stack(mmd_values).mean()


class VelocityConsistencyLoss(nn.Module):
    """Velocity consistency loss.

    Ensures that the direction of progression (velocity) in latent space
    is consistent across different stage pairs. This enforces that
    progression follows a coherent trajectory rather than jumping
    around in latent space.

    Computed as the variance of pairwise velocity vectors between
    consecutive stage centroids.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        z_latent: torch.Tensor,
        stage_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Compute velocity consistency loss.

        Args:
            z_latent: Latent representations
            stage_idx: Stage labels (ordered)

        Returns:
            Velocity consistency loss
        """
        unique_stages, inverse_idx = torch.unique(stage_idx, sorted=True, return_inverse=True)
        n_stages = len(unique_stages)

        if n_stages < 3:
            return torch.tensor(0.0, device=z_latent.device)

        z_centroids = torch.zeros(n_stages, z_latent.shape[1], device=z_latent.device)
        for i in range(n_stages):
            mask = inverse_idx == i
            z_centroids[i] = z_latent[mask].mean(dim=0)

        velocity_vectors = []

        for i in range(n_stages):
            for j in range(i + 1, n_stages):
                t_i = unique_stages[i].float()
                t_j = unique_stages[j].float()
                time_diff = t_j - t_i

                # Guard against division by zero (shouldn't happen with unique stages, but be safe)
                if time_diff.abs() < 1e-8:
                    continue

                velocity = (z_centroids[j] - z_centroids[i]) / time_diff
                velocity_vectors.append(velocity)

        if len(velocity_vectors) < 2:
            return torch.tensor(0.0, device=z_latent.device)

        velocity_matrix = torch.stack(velocity_vectors)

        pairwise_dist = compute_pairwise_distance(velocity_matrix, velocity_matrix).pow(2)

        loss = pairwise_dist.mean()
        return loss


class ManifoldLoss(nn.Module):
    """Combined manifold-aware loss for geodesic-preserving representations.

    Combines:
    - Local isometry: Preserve local distances
    - CVL: Straight-line trajectories in latent space
    - Velocity consistency: Coherent progression direction

    Together, these ensure that linear interpolation in latent space
    corresponds to biologically meaningful geodesic paths in
    expression space.
    """

    def __init__(
        self,
        iso_weight: float = 1.0,
        cvl_weight: float = 1.0,
        velocity_weight: float = 0.5,
        ot_reg: float = 0.02,
    ):
        """
        Args:
            iso_weight: Weight for isometry loss
            cvl_weight: Weight for CVL loss
            velocity_weight: Weight for velocity consistency loss
            ot_reg: OT regularization for CVL
        """
        super().__init__()
        self.iso_weight = iso_weight
        self.cvl_weight = cvl_weight
        self.velocity_weight = velocity_weight

        self.isometry_loss = LocalIsometryLoss()
        self.cvl_loss = ConstantVelocityLinearLoss(ot_reg=ot_reg)
        self.velocity_loss = VelocityConsistencyLoss()

    def forward(
        self,
        x_input: torch.Tensor,
        z_latent: torch.Tensor,
        stage_idx: torch.Tensor,
        decoder_fn: callable | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute combined manifold loss.

        Args:
            x_input: Input expression data
            z_latent: Latent representations
            stage_idx: Stage labels
            decoder_fn: Optional decoder for CVL loss

        Returns:
            Dict with 'total' and individual loss components
        """
        losses = {}

        losses['isometry'] = self.isometry_loss(x_input, z_latent, stage_idx)

        if decoder_fn is not None and self.cvl_weight > 0:
            losses['cvl'] = self.cvl_loss(x_input, z_latent, stage_idx, decoder_fn)
        else:
            losses['cvl'] = torch.tensor(0.0, device=x_input.device)

        losses['velocity'] = self.velocity_loss(z_latent, stage_idx)

        losses['total'] = (
            self.iso_weight * losses['isometry'] +
            self.cvl_weight * losses['cvl'] +
            self.velocity_weight * losses['velocity']
        )

        return losses
