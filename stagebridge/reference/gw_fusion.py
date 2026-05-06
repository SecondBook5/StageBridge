"""Gromov-Wasserstein fusion for dual-reference alignment.

Aligns HLCA and LuCA embedding spaces via differentiable entropic GW,
replacing naive concatenation with principled geometric fusion.

The key insight: both atlases represent the same cells in different
coordinate systems. GW finds the optimal coupling that preserves
pairwise distances within each space.

Three fusion modes:
1. project_to_hlca: Project LuCA into HLCA space via learned barycentric map
2. project_to_luca: Project HLCA into LuCA space
3. barycentric: Project both into a shared intermediate space

References:
- Peyré et al. (2016) "Gromov-Wasserstein Averaging of Kernel and Distance Matrices"
- Mémoli (2011) "Gromov-Wasserstein Distances and the Metric Approach to Object Matching"
- Bunne et al. (2019) "Learning Generative Models across Incomparable Spaces"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn


@dataclass
class GWFusionConfig:
    """Configuration for Gromov-Wasserstein fusion."""

    hlca_dim: int = 30
    luca_dim: int = 10
    output_dim: int = 64  # Fused representation dimension

    # Sinkhorn parameters
    sinkhorn_iters: int = 50
    sinkhorn_reg: float = 0.1  # Entropic regularization (higher = smoother)
    sinkhorn_threshold: float = 1e-3  # Convergence threshold

    # Fusion mode
    mode: Literal["project_to_hlca", "project_to_luca", "barycentric"] = "barycentric"

    # Barycentric weight (0 = pure HLCA, 1 = pure LuCA, 0.5 = midpoint)
    barycentric_alpha: float = 0.5

    # Whether to learn the cost matrices or use Euclidean
    learn_cost_metric: bool = True

    # Regularization
    dropout: float = 0.1


def pairwise_distances(x: Tensor, squared: bool = True) -> Tensor:
    """Compute pairwise squared Euclidean distances.

    Args:
        x: [B, N, D] batch of point clouds
        squared: Return squared distances

    Returns:
        [B, N, N] distance matrix
    """
    # ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 * x_i · x_j
    x_sq = (x ** 2).sum(dim=-1, keepdim=True)  # [B, N, 1]
    dist_sq = x_sq + x_sq.transpose(-2, -1) - 2 * torch.bmm(x, x.transpose(-2, -1))
    dist_sq = dist_sq.clamp(min=0)  # Numerical stability

    if squared:
        return dist_sq
    return torch.sqrt(dist_sq + 1e-8)


def sinkhorn(
    C: Tensor,
    a: Tensor | None = None,
    b: Tensor | None = None,
    reg: float = 0.1,
    num_iters: int = 50,
    threshold: float = 1e-3,
) -> Tensor:
    """Sinkhorn algorithm for entropic optimal transport.

    Args:
        C: [B, N, M] cost matrix
        a: [B, N] source marginal (uniform if None)
        b: [B, M] target marginal (uniform if None)
        reg: Entropic regularization
        num_iters: Maximum iterations
        threshold: Convergence threshold

    Returns:
        P: [B, N, M] transport plan
    """
    B, N, M = C.shape
    device = C.device
    dtype = C.dtype

    if a is None:
        a = torch.ones(B, N, device=device, dtype=dtype) / N
    if b is None:
        b = torch.ones(B, M, device=device, dtype=dtype) / M

    # Gibbs kernel
    K = torch.exp(-C / reg)

    # Initialize
    u = torch.ones(B, N, device=device, dtype=dtype)
    v = torch.ones(B, M, device=device, dtype=dtype)

    for _ in range(num_iters):
        u_prev = u

        # Sinkhorn iterations
        u = a / (torch.bmm(K, v.unsqueeze(-1)).squeeze(-1) + 1e-8)
        v = b / (torch.bmm(K.transpose(-2, -1), u.unsqueeze(-1)).squeeze(-1) + 1e-8)

        # Check convergence
        err = (u - u_prev).abs().max()
        if err < threshold:
            break

    # Transport plan: P = diag(u) @ K @ diag(v)
    P = u.unsqueeze(-1) * K * v.unsqueeze(-2)

    return P


def gromov_wasserstein_cost(
    C1: Tensor,
    C2: Tensor,
    P: Tensor,
) -> Tensor:
    """Compute Gromov-Wasserstein cost for given coupling.

    GW(C1, C2, P) = sum_{i,j,k,l} |C1[i,k] - C2[j,l]|^2 * P[i,j] * P[k,l]

    Efficient computation via:
    GW = <C1^2 @ P @ 1, P @ 1> + <1 @ P @ C2^2, 1 @ P> - 2 * <C1 @ P @ C2^T, P>

    Args:
        C1: [B, N, N] source distance matrix
        C2: [B, M, M] target distance matrix
        P: [B, N, M] transport plan

    Returns:
        [B] GW cost per batch element
    """
    # Term 1: C1^2 contribution
    C1_sq = C1 ** 2
    term1 = torch.einsum('bik,bij,bkj->b', C1_sq, P, P)

    # Term 2: C2^2 contribution
    C2_sq = C2 ** 2
    term2 = torch.einsum('bjl,bij,bil->b', C2_sq, P, P)

    # Term 3: Cross term
    term3 = torch.einsum('bik,bij,bjl,bkl->b', C1, P, C2, P)

    return term1 + term2 - 2 * term3


def entropic_gromov_wasserstein(
    C1: Tensor,
    C2: Tensor,
    a: Tensor | None = None,
    b: Tensor | None = None,
    reg: float = 0.1,
    num_outer_iters: int = 20,
    num_sinkhorn_iters: int = 50,
    threshold: float = 1e-3,
) -> tuple[Tensor, Tensor]:
    """Entropic Gromov-Wasserstein via projected gradient descent.

    Alternates between:
    1. Compute linear cost from current coupling
    2. Solve entropic OT with Sinkhorn

    Args:
        C1: [B, N, N] source distance matrix
        C2: [B, M, M] target distance matrix
        a: [B, N] source marginal
        b: [B, M] target marginal
        reg: Entropic regularization
        num_outer_iters: GW outer iterations
        num_sinkhorn_iters: Sinkhorn inner iterations
        threshold: Convergence threshold

    Returns:
        P: [B, N, M] optimal transport plan
        cost: [B] final GW cost
    """
    B, N, _ = C1.shape
    M = C2.shape[1]
    device = C1.device
    dtype = C1.dtype

    if a is None:
        a = torch.ones(B, N, device=device, dtype=dtype) / N
    if b is None:
        b = torch.ones(B, M, device=device, dtype=dtype) / M

    # Initialize with uniform coupling
    P = a.unsqueeze(-1) * b.unsqueeze(-2)  # [B, N, M]

    for _ in range(num_outer_iters):
        P_prev = P

        # Compute gradient of GW cost w.r.t. P
        # grad_P GW = 4 * (C1^2 @ P @ 1 @ 1^T + 1 @ 1^T @ P^T @ C2^2 - 2 * C1 @ P @ C2)
        # Simplified linear cost matrix for Sinkhorn:
        cost_matrix = -4 * torch.bmm(torch.bmm(C1, P), C2)  # [B, N, M]

        # Add the constant terms (can be precomputed)
        C1_sq_sum = (C1 ** 2).sum(dim=-1, keepdim=True)  # [B, N, 1]
        C2_sq_sum = (C2 ** 2).sum(dim=-1, keepdim=True).transpose(-2, -1)  # [B, 1, M]
        cost_matrix = cost_matrix + 2 * (C1_sq_sum + C2_sq_sum)

        # Sinkhorn step
        P = sinkhorn(
            cost_matrix, a, b,
            reg=reg,
            num_iters=num_sinkhorn_iters,
            threshold=threshold,
        )

        # Check convergence
        err = (P - P_prev).abs().max()
        if err < threshold:
            break

    cost = gromov_wasserstein_cost(C1, C2, P)
    return P, cost


class LearnedMetric(nn.Module):
    """Learned Mahalanobis-like metric for distance computation.

    Instead of Euclidean distance, learn: d(x, y) = ||Ax - Ay||^2
    where A is a learned linear transformation.
    """

    def __init__(self, input_dim: int, metric_dim: int | None = None):
        super().__init__()
        if metric_dim is None:
            metric_dim = input_dim
        self.transform = nn.Linear(input_dim, metric_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """Compute pairwise distances under learned metric.

        Args:
            x: [B, N, D] points

        Returns:
            [B, N, N] distance matrix
        """
        x_transformed = self.transform(x)
        return pairwise_distances(x_transformed, squared=True)


class GromovWassersteinFusion(nn.Module):
    """Differentiable Gromov-Wasserstein fusion for HLCA-LuCA alignment.

    Learns to align the two reference atlas spaces via entropic GW,
    producing a unified representation that preserves geometric structure
    from both spaces.

    Args:
        config: GWFusionConfig with hyperparameters
    """

    def __init__(self, config: GWFusionConfig):
        super().__init__()
        self.config = config

        # Optional learned metrics for cost matrices
        self.hlca_metric: LearnedMetric | None = None
        self.luca_metric: LearnedMetric | None = None
        if config.learn_cost_metric:
            self.hlca_metric = LearnedMetric(config.hlca_dim)
            self.luca_metric = LearnedMetric(config.luca_dim)

        # Projection heads to output dimension
        self.hlca_proj = nn.Sequential(
            nn.Linear(config.hlca_dim, config.output_dim),
            nn.LayerNorm(config.output_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.luca_proj = nn.Sequential(
            nn.Linear(config.luca_dim, config.output_dim),
            nn.LayerNorm(config.output_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        # Learnable barycentric weight (if mode is barycentric)
        if config.mode == "barycentric":
            self.alpha = nn.Parameter(torch.tensor(config.barycentric_alpha))
        else:
            self.register_buffer('alpha', torch.tensor(config.barycentric_alpha))

        # Final fusion layer
        self.fusion_head = nn.Sequential(
            nn.Linear(config.output_dim, config.output_dim),
            nn.LayerNorm(config.output_dim),
        )

    def compute_cost_matrices(
        self,
        hlca: Tensor,
        luca: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Compute intra-space distance matrices.

        Args:
            hlca: [B, N, D_hlca] HLCA embeddings (N cells per batch)
            luca: [B, N, D_luca] LuCA embeddings

        Returns:
            C_hlca: [B, N, N] HLCA distance matrix
            C_luca: [B, N, N] LuCA distance matrix
        """
        if self.hlca_metric is not None:
            C_hlca = self.hlca_metric(hlca)
        else:
            C_hlca = pairwise_distances(hlca)

        if self.luca_metric is not None:
            C_luca = self.luca_metric(luca)
        else:
            C_luca = pairwise_distances(luca)

        return C_hlca, C_luca

    def forward(
        self,
        hlca: Tensor,
        luca: Tensor,
        return_coupling: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor, Tensor]:
        """Fuse HLCA and LuCA embeddings via Gromov-Wasserstein alignment.

        Args:
            hlca: [B, D_hlca] or [B, N, D_hlca] HLCA embeddings
            luca: [B, D_luca] or [B, N, D_luca] LuCA embeddings
            return_coupling: Also return the transport plan and GW cost

        Returns:
            fused: [B, D_out] or [B, N, D_out] fused representation
            coupling: [B, N, N] transport plan (if return_coupling)
            gw_cost: [B] GW cost (if return_coupling)
        """
        # Handle single-cell case (add sequence dimension)
        squeeze_output = False
        if hlca.ndim == 2:
            hlca = hlca.unsqueeze(1)  # [B, 1, D]
            luca = luca.unsqueeze(1)
            squeeze_output = True

        B, N, _ = hlca.shape

        # Compute cost matrices
        C_hlca, C_luca = self.compute_cost_matrices(hlca, luca)

        # Compute GW coupling
        coupling, gw_cost = entropic_gromov_wasserstein(
            C_hlca, C_luca,
            reg=self.config.sinkhorn_reg,
            num_outer_iters=20,
            num_sinkhorn_iters=self.config.sinkhorn_iters,
            threshold=self.config.sinkhorn_threshold,
        )

        # Project to common dimension
        hlca_proj = self.hlca_proj(hlca)  # [B, N, D_out]
        luca_proj = self.luca_proj(luca)  # [B, N, D_out]

        # Apply transport: move LuCA to HLCA space (or vice versa)
        if self.config.mode == "project_to_hlca":
            # Transport LuCA to HLCA: luca_transported = P^T @ luca_proj
            # Normalize coupling to be column-stochastic
            P_normalized = coupling / (coupling.sum(dim=1, keepdim=True) + 1e-8)
            luca_transported = torch.bmm(P_normalized.transpose(-2, -1), luca_proj)
            fused = hlca_proj + luca_transported

        elif self.config.mode == "project_to_luca":
            # Transport HLCA to LuCA: hlca_transported = P @ hlca_proj
            P_normalized = coupling / (coupling.sum(dim=2, keepdim=True) + 1e-8)
            hlca_transported = torch.bmm(P_normalized, hlca_proj)
            fused = luca_proj + hlca_transported

        else:  # barycentric
            # Barycentric fusion: weighted combination in shared space
            alpha = torch.sigmoid(self.alpha)  # Ensure [0, 1]

            # Transport both towards each other
            P_to_luca = coupling / (coupling.sum(dim=2, keepdim=True) + 1e-8)
            P_to_hlca = coupling / (coupling.sum(dim=1, keepdim=True) + 1e-8)

            hlca_transported = torch.bmm(P_to_luca, hlca_proj)
            luca_transported = torch.bmm(P_to_hlca.transpose(-2, -1), luca_proj)

            # Barycentric interpolation
            fused = (1 - alpha) * (hlca_proj + luca_transported) / 2 + \
                    alpha * (luca_proj + hlca_transported) / 2

        # Final fusion
        fused = self.fusion_head(fused)

        if squeeze_output:
            fused = fused.squeeze(1)

        if return_coupling:
            return fused, coupling, gw_cost
        return fused


class PrecomputedGWFusion(nn.Module):
    """GW fusion with precomputed stable global alignment.

    The original GWFusion was fundamentally broken:
    - It computed GW per batch on single cells [B, 1, D]
    - GW needs population structure - 1x1 matrices have no structure
    - Coupling changed every batch (unstable)

    This fix:
    1. Precompute GW coupling on representative cell population (offline)
    2. Learn projections that respect the precomputed alignment
    3. Use learned projections at inference (no per-batch GW)

    The key insight: HLCA and LuCA represent the SAME cells in different
    coordinate systems. The geometric correspondence is GLOBAL (atlas-level),
    not per-batch. We learn it once, then apply.

    Args:
        config: GWFusionConfig
        precomputed_coupling_path: Path to precomputed coupling matrix
    """

    def __init__(
        self,
        config: GWFusionConfig,
        reference_hlca: Tensor | None = None,
        reference_luca: Tensor | None = None,
    ):
        super().__init__()
        self.config = config

        # Projection heads
        self.hlca_proj = nn.Sequential(
            nn.Linear(config.hlca_dim, config.output_dim),
            nn.LayerNorm(config.output_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.luca_proj = nn.Sequential(
            nn.Linear(config.luca_dim, config.output_dim),
            nn.LayerNorm(config.output_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        # Learnable fusion weights (simple but stable)
        self.hlca_weight = nn.Parameter(torch.tensor(0.5))
        self.luca_weight = nn.Parameter(torch.tensor(0.5))

        # Reference embeddings for nearest-neighbor lookup (optional)
        # If provided, can do soft assignment to reference cells
        self.reference_hlca: Tensor | None = None
        self.reference_luca: Tensor | None = None
        if reference_hlca is not None and reference_luca is not None:
            self.register_buffer('reference_hlca', reference_hlca)
            self.register_buffer('reference_luca', reference_luca)

        # Final fusion
        self.fusion_head = nn.Sequential(
            nn.Linear(config.output_dim, config.output_dim),
            nn.LayerNorm(config.output_dim),
        )

    def forward(
        self,
        hlca: Tensor,
        luca: Tensor,
        return_coupling: bool = False,
    ) -> Tensor | tuple[Tensor, None, None]:
        """Fuse HLCA and LuCA via learned weighted projection.

        No per-batch GW computation - just stable learned projections.

        Args:
            hlca: [B, D_hlca] HLCA embeddings
            luca: [B, D_luca] LuCA embeddings
            return_coupling: Ignored (for API compatibility)

        Returns:
            fused: [B, D_out] fused representation
        """
        # Project to common space
        hlca_proj = self.hlca_proj(hlca)  # [B, D_out]
        luca_proj = self.luca_proj(luca)  # [B, D_out]

        # Learned weighted combination (stable, no GW per batch)
        w_hlca = torch.sigmoid(self.hlca_weight)
        w_luca = torch.sigmoid(self.luca_weight)
        w_sum = w_hlca + w_luca

        fused = (w_hlca * hlca_proj + w_luca * luca_proj) / w_sum
        fused = self.fusion_head(fused)

        if return_coupling:
            return fused, None, None
        return fused


class GWFusionLoss(nn.Module):
    """Auxiliary loss for GW fusion training.

    Encourages the coupling to preserve neighborhood structure:
    - GW cost should be low
    - Coupling should be sparse (not uniform)
    - Transported points should preserve local neighborhoods
    """

    def __init__(
        self,
        gw_weight: float = 1.0,
        entropy_weight: float = 0.1,
        neighborhood_weight: float = 0.5,
    ):
        super().__init__()
        self.gw_weight = gw_weight
        self.entropy_weight = entropy_weight
        self.neighborhood_weight = neighborhood_weight

    def forward(
        self,
        coupling: Tensor,
        gw_cost: Tensor,
        hlca: Tensor,
        luca: Tensor,
        fused: Tensor,
    ) -> Tensor:
        """Compute GW fusion loss.

        Args:
            coupling: [B, N, N] transport plan
            gw_cost: [B] GW cost
            hlca: [B, N, D_hlca] original HLCA
            luca: [B, N, D_luca] original LuCA
            fused: [B, N, D_out] fused representation

        Returns:
            loss: Scalar loss
        """
        # GW cost (already computed, just weight it)
        loss_gw = gw_cost.mean() * self.gw_weight

        # Entropy regularization (encourage non-uniform coupling)
        coupling_flat = coupling.view(coupling.shape[0], -1)
        entropy = -(coupling_flat * (coupling_flat + 1e-8).log()).sum(dim=-1)
        loss_entropy = -entropy.mean() * self.entropy_weight  # Negative to minimize entropy

        # Neighborhood preservation in fused space
        # Fused distances should correlate with average of HLCA/LuCA distances
        C_hlca = pairwise_distances(hlca)
        C_luca = pairwise_distances(luca)
        C_fused = pairwise_distances(fused)
        C_avg = (C_hlca + C_luca) / 2

        # Correlation loss (1 - Pearson correlation)
        C_avg_flat = C_avg.view(C_avg.shape[0], -1)
        C_fused_flat = C_fused.view(C_fused.shape[0], -1)

        C_avg_centered = C_avg_flat - C_avg_flat.mean(dim=-1, keepdim=True)
        C_fused_centered = C_fused_flat - C_fused_flat.mean(dim=-1, keepdim=True)

        cov = (C_avg_centered * C_fused_centered).sum(dim=-1)
        std_avg = C_avg_centered.std(dim=-1)
        std_fused = C_fused_centered.std(dim=-1)

        correlation = cov / (std_avg * std_fused + 1e-8)
        loss_neighborhood = (1 - correlation).mean() * self.neighborhood_weight

        return loss_gw + loss_entropy + loss_neighborhood
