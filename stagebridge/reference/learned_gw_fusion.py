"""Learned Gromov-Wasserstein Fusion for HLCA-LuCA alignment.

The key insight: GW finds structure-preserving alignment between heterogeneous
spaces. We want to LEARN this alignment during training, not just precompute it.

Approach:
1. Learn metric spaces for HLCA and LuCA via projection heads
2. Compute GW coupling in these learned spaces
3. Use coupling to create fused representation
4. Backprop through the whole thing

This differs from:
- Concat: No alignment at all
- Precomputed GW: Fixed coupling, can't adapt to downstream task
- ICNN: Overkill for atlas alignment (designed for perturbation prediction)

The learned metrics allow the model to discover which dimensions of HLCA/LuCA
are most important for the downstream task (niche-gated transitions).

References:
- Peyré et al. (2016) "Gromov-Wasserstein Averaging"
- Bunne et al. (2019) "Learning Generative Models across Incomparable Spaces"
- Klein et al. (2023) "moscot" for scalable OT
"""

from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class LearnedGWConfig:
    """Configuration for learned GW fusion."""
    hlca_dim: int = 30
    luca_dim: int = 10
    metric_dim: int = 32  # Shared metric space dimension
    output_dim: int = 40  # Fused output dimension

    # Sinkhorn parameters
    sinkhorn_reg: float = 0.1  # Entropic regularization
    sinkhorn_iters: int = 20   # Sinkhorn iterations
    gw_iters: int = 10         # Outer GW iterations

    # Architecture
    num_metric_layers: int = 2
    dropout: float = 0.1

    # Training stability
    stop_gradient_coupling: bool = False  # Stop gradients through coupling
    coupling_temperature: float = 1.0     # Temperature for coupling softmax


def sinkhorn_log_stabilized(
    C: Tensor,
    reg: float,
    num_iters: int,
) -> Tensor:
    """Log-stabilized Sinkhorn for numerical stability.

    Args:
        C: [B, N, M] cost matrix
        reg: Entropic regularization
        num_iters: Number of iterations

    Returns:
        P: [B, N, M] transport plan
    """
    B, N, M = C.shape
    device = C.device
    dtype = C.dtype

    # Uniform marginals in log space
    log_a = torch.full((B, N), -torch.log(torch.tensor(N, dtype=dtype)), device=device)
    log_b = torch.full((B, M), -torch.log(torch.tensor(M, dtype=dtype)), device=device)

    # Log kernel
    log_K = -C / reg

    # Initialize dual variables
    log_u = torch.zeros(B, N, device=device, dtype=dtype)
    log_v = torch.zeros(B, M, device=device, dtype=dtype)

    for _ in range(num_iters):
        # u update
        log_u = log_a - torch.logsumexp(log_K + log_v.unsqueeze(1), dim=2)
        # v update
        log_v = log_b - torch.logsumexp(log_K.transpose(1, 2) + log_u.unsqueeze(1), dim=2)

    # Coupling in log domain
    log_P = log_u.unsqueeze(2) + log_K + log_v.unsqueeze(1)
    return log_P.exp()


def gromov_wasserstein_differentiable(
    C_X: Tensor,
    C_Y: Tensor,
    reg: float = 0.1,
    num_gw_iters: int = 10,
    num_sinkhorn_iters: int = 20,
) -> tuple[Tensor, Tensor]:
    """Differentiable entropic Gromov-Wasserstein.

    Finds coupling P that minimizes:
        Σ_ijkl |C_X[i,k] - C_Y[j,l]|² P[i,j] P[k,l]

    This is differentiable w.r.t. C_X and C_Y, allowing backprop
    through the learned metrics.

    Args:
        C_X: [B, N, N] distance matrix in X space
        C_Y: [B, M, M] distance matrix in Y space
        reg: Entropic regularization
        num_gw_iters: Outer GW iterations
        num_sinkhorn_iters: Inner Sinkhorn iterations

    Returns:
        P: [B, N, M] optimal coupling
        cost: [B] GW cost
    """
    B, N, _ = C_X.shape
    M = C_Y.shape[1]
    device = C_X.device
    dtype = C_X.dtype

    # Initialize coupling as outer product of uniforms
    a = torch.ones(B, N, device=device, dtype=dtype) / N
    b = torch.ones(B, M, device=device, dtype=dtype) / M
    P = a.unsqueeze(2) * b.unsqueeze(1)  # [B, N, M]

    # Precompute squared costs
    C_X_sq = C_X ** 2
    C_Y_sq = C_Y ** 2

    for _ in range(num_gw_iters):
        # Compute linear cost matrix for current P
        # cost[i,j] = Σ_kl P[k,l] (C_X[i,k] - C_Y[j,l])²
        #           = Σ_k C_X²[i,k] Σ_l P[k,l] + Σ_l C_Y²[j,l] Σ_k P[k,l] - 2 Σ_kl C_X[i,k] P[k,l] C_Y[j,l]

        # Term 1: [B, N, 1]
        term1 = torch.bmm(C_X_sq, P.sum(dim=2, keepdim=True))
        # Term 2: [B, 1, M]
        term2 = torch.bmm(P.sum(dim=1, keepdim=True), C_Y_sq)
        # Term 3: [B, N, M]
        term3 = torch.bmm(torch.bmm(C_X, P), C_Y)

        cost_matrix = term1 + term2.transpose(1, 2) - 2 * term3

        # Sinkhorn step
        P = sinkhorn_log_stabilized(cost_matrix, reg, num_sinkhorn_iters)

    # Final GW cost
    gw_cost = (P * cost_matrix).sum(dim=(1, 2))

    return P, gw_cost


class LearnedMetricHead(nn.Module):
    """Learn a metric space via projection.

    Projects input to a learned metric space where distances
    are meaningful for GW alignment.
    """

    def __init__(
        self,
        input_dim: int,
        metric_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        layers = [nn.Linear(input_dim, metric_dim), nn.GELU(), nn.Dropout(dropout)]
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(metric_dim, metric_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
        layers.append(nn.LayerNorm(metric_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Project to metric space."""
        return self.net(x)


class LearnedGWFusion(nn.Module):
    """Learned Gromov-Wasserstein fusion for HLCA-LuCA alignment.

    Key idea: Learn metric projections for both spaces such that
    GW alignment in these spaces is optimal for downstream tasks.

    Architecture:
    1. Project HLCA → metric space via learned head
    2. Project LuCA → metric space via learned head
    3. Compute pairwise distances in metric spaces
    4. Solve GW to get coupling
    5. Use coupling to create fused representation

    The whole thing is differentiable, so the metric heads learn
    to create spaces where GW alignment helps the downstream task.

    Note: This operates on BATCHES of cells. For single-cell inference,
    use the precomputed coupling from training.
    """

    def __init__(self, config: LearnedGWConfig):
        super().__init__()
        self.config = config

        # Metric projection heads
        self.hlca_metric = LearnedMetricHead(
            config.hlca_dim,
            config.metric_dim,
            config.num_metric_layers,
            config.dropout,
        )
        self.luca_metric = LearnedMetricHead(
            config.luca_dim,
            config.metric_dim,
            config.num_metric_layers,
            config.dropout,
        )

        # Output projection (from both embeddings to fused)
        self.fusion_head = nn.Sequential(
            nn.Linear(config.hlca_dim + config.luca_dim, config.output_dim),
            nn.LayerNorm(config.output_dim),
        )

        # Learned weighting for fusion
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def compute_coupling(
        self,
        hlca: Tensor,
        luca: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Compute GW coupling in learned metric spaces.

        Args:
            hlca: [B, hlca_dim] HLCA embeddings
            luca: [B, luca_dim] LuCA embeddings

        Returns:
            coupling: [B, B] transport plan (how to align cells)
            gw_cost: [1] GW cost (for monitoring/loss)
        """
        # Project to metric spaces
        h_metric = self.hlca_metric(hlca)  # [B, metric_dim]
        l_metric = self.luca_metric(luca)  # [B, metric_dim]

        # Compute distance matrices
        # Add batch dimension for GW solver
        C_h = torch.cdist(h_metric, h_metric).unsqueeze(0)  # [1, B, B]
        C_l = torch.cdist(l_metric, l_metric).unsqueeze(0)  # [1, B, B]

        # Normalize distances for stability
        C_h = C_h / (C_h.max() + 1e-8)
        C_l = C_l / (C_l.max() + 1e-8)

        # Solve GW
        coupling, gw_cost = gromov_wasserstein_differentiable(
            C_h, C_l,
            reg=self.config.sinkhorn_reg,
            num_gw_iters=self.config.gw_iters,
            num_sinkhorn_iters=self.config.sinkhorn_iters,
        )

        # Remove batch dimension
        coupling = coupling.squeeze(0)  # [B, B]
        gw_cost = gw_cost.squeeze(0)    # scalar

        if self.config.stop_gradient_coupling:
            coupling = coupling.detach()

        return coupling, gw_cost

    def forward(
        self,
        hlca: Tensor,
        luca: Tensor,
        return_coupling: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor, Tensor]:
        """Fuse HLCA and LuCA via learned GW alignment.

        For batch of B cells:
        1. Compute GW coupling [B, B] in learned metric spaces
        2. For each cell i, its "aligned luca" is Σ_j P[i,j] * luca[j]
        3. Fuse: output = fusion_head([hlca; aligned_luca])

        This encourages the model to learn metrics where cells with
        similar structure (according to GW) should have similar fused
        representations.

        Args:
            hlca: [B, 30] HLCA embeddings
            luca: [B, 10] LuCA embeddings
            return_coupling: Return coupling and cost

        Returns:
            fused: [B, output_dim] fused representation
            coupling: [B, B] if return_coupling
            gw_cost: scalar if return_coupling
        """
        B = hlca.shape[0]

        if B == 1:
            # Single cell: can't compute meaningful GW, just concat
            fused = self.fusion_head(torch.cat([hlca, luca], dim=-1))
            if return_coupling:
                dummy_coupling = torch.ones(1, 1, device=hlca.device)
                return fused, dummy_coupling, torch.tensor(0.0, device=hlca.device)
            return fused

        # Compute coupling
        coupling, gw_cost = self.compute_coupling(hlca, luca)

        # Normalize coupling to be row-stochastic
        P = coupling / (coupling.sum(dim=1, keepdim=True) + 1e-8)

        # Transport LuCA: aligned_luca[i] = Σ_j P[i,j] * luca[j]
        aligned_luca = torch.mm(P, luca)  # [B, luca_dim]

        # Interpolate between direct and aligned
        alpha = torch.sigmoid(self.alpha)
        luca_final = alpha * aligned_luca + (1 - alpha) * luca

        # Fuse
        fused = self.fusion_head(torch.cat([hlca, luca_final], dim=-1))

        if return_coupling:
            return fused, coupling, gw_cost
        return fused

    def get_gw_loss(self, hlca: Tensor, luca: Tensor) -> Tensor:
        """Get GW cost as auxiliary loss for training.

        Lower GW cost = better alignment in metric spaces.
        Can add to main loss with small weight.
        """
        _, gw_cost = self.compute_coupling(hlca, luca)
        return gw_cost


class PretrainedLearnedGWFusion(nn.Module):
    """Use a pretrained LearnedGWFusion for inference.

    During training, we compute GW per batch. For inference on
    single cells, we use a precomputed coupling from reference data.
    """

    def __init__(
        self,
        learned_fusion: LearnedGWFusion,
        reference_hlca: Tensor,
        reference_luca: Tensor,
        k_neighbors: int = 15,
    ):
        super().__init__()
        self.learned_fusion = learned_fusion
        self.k_neighbors = k_neighbors

        # Precompute coupling on reference
        with torch.no_grad():
            coupling, _ = learned_fusion.compute_coupling(reference_hlca, reference_luca)

        # Store reference data
        self.register_buffer("reference_hlca", reference_hlca)
        self.register_buffer("reference_luca", reference_luca)
        self.register_buffer("reference_coupling", coupling)

    def forward(self, hlca: Tensor, luca: Tensor) -> Tensor:
        """Fuse using k-NN + precomputed coupling.

        For query cell:
        1. Find k nearest neighbors in reference HLCA
        2. Aggregate their coupling weights
        3. Use for barycentric projection
        """
        # Find k-NN in reference
        dists = torch.cdist(hlca, self.reference_hlca)  # [B, N_ref]
        _, knn_idx = dists.topk(self.k_neighbors, dim=1, largest=False)  # [B, k]

        # Get coupling rows for neighbors
        # knn_coupling[b, i, :] = reference_coupling[knn_idx[b, i], :]
        B, K = knn_idx.shape
        N_ref = self.reference_coupling.shape[0]

        # Gather coupling rows
        knn_coupling = self.reference_coupling[knn_idx.view(-1)].view(B, K, N_ref)  # [B, k, N_ref]

        # Average over neighbors
        soft_coupling = knn_coupling.mean(dim=1)  # [B, N_ref]

        # Normalize
        soft_coupling = soft_coupling / (soft_coupling.sum(dim=1, keepdim=True) + 1e-8)

        # Transport
        aligned_luca = torch.mm(soft_coupling, self.reference_luca)

        # Fuse (use same fusion head)
        alpha = torch.sigmoid(self.learned_fusion.alpha)
        luca_final = alpha * aligned_luca + (1 - alpha) * luca

        return self.learned_fusion.fusion_head(torch.cat([hlca, luca_final], dim=-1))
