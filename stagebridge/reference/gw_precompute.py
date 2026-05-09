"""Offline Gromov-Wasserstein precomputation for HLCA-LuCA alignment.

Proper GW requires a population of cells to find structure-preserving alignment.
This module:
1. Samples representative cells from the dataset
2. Computes intra-space distance matrices for HLCA and LuCA
3. Solves GW to find optimal coupling
4. Uses coupling-based barycentric projection for fusion (moscot-style)

The key insight from moscot (Klein et al. 2023): the GW coupling IS the principled
OT solution. For out-of-sample cells, use k-NN in source space + barycentric
projection using coupling weights. No neural network needed for valid OT.

References:
- Bunne et al. (2024) "Optimal transport for single-cell and spatial omics"
- Klein et al. (2023) "moscot: scalable optimal transport for single-cell genomics"
- Peyré et al. (2016) "Gromov-Wasserstein Averaging"
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


@dataclass
class GWPrecomputeConfig:
    """Configuration for GW precomputation."""
    n_reference_cells: int = 5000  # Cells to sample for GW computation
    hlca_dim: int = 30
    luca_dim: int = 10
    output_dim: int = 40

    # Sinkhorn parameters
    sinkhorn_iters: int = 100
    sinkhorn_reg: float = 0.05  # Lower = sparser coupling
    gw_iters: int = 50

    # Neural map training
    map_hidden_dim: int = 128
    map_num_layers: int = 3
    map_epochs: int = 100
    map_lr: float = 1e-3
    map_batch_size: int = 256

    # Stratification
    stratify_by_stage: bool = True
    stratify_by_celltype: bool = True


def pairwise_distances(x: Tensor, squared: bool = True) -> Tensor:
    """Compute pairwise squared Euclidean distances."""
    x_sq = (x ** 2).sum(dim=-1, keepdim=True)
    dist_sq = x_sq + x_sq.T - 2 * x @ x.T
    dist_sq = dist_sq.clamp(min=0)
    return dist_sq if squared else torch.sqrt(dist_sq + 1e-8)


def sinkhorn_log_domain(
    C: Tensor,
    a: Tensor,
    b: Tensor,
    reg: float,
    num_iters: int,
) -> Tensor:
    """Sinkhorn in log domain for numerical stability."""
    n, m = C.shape

    # Log marginals
    log_a = a.log()
    log_b = b.log()

    # Initialize dual variables
    f = torch.zeros(n, device=C.device, dtype=C.dtype)
    g = torch.zeros(m, device=C.device, dtype=C.dtype)

    # Log kernel
    log_K = -C / reg

    for _ in range(num_iters):
        # f update: f = log_a - logsumexp(log_K + g, dim=1)
        f = log_a - torch.logsumexp(log_K + g.unsqueeze(0), dim=1)
        # g update: g = log_b - logsumexp(log_K.T + f, dim=1)
        g = log_b - torch.logsumexp(log_K.T + f.unsqueeze(0), dim=1)

    # Coupling in log domain
    log_P = f.unsqueeze(1) + log_K + g.unsqueeze(0)
    return log_P.exp()


def gromov_wasserstein(
    D_X: Tensor,
    D_Y: Tensor,
    a: Tensor | None = None,
    b: Tensor | None = None,
    reg: float = 0.05,
    num_gw_iters: int = 50,
    num_sinkhorn_iters: int = 100,
) -> tuple[Tensor, float]:
    """Entropic Gromov-Wasserstein via alternating optimization.

    Minimizes: Σ_ijkl P_ij P_kl (D_X[i,k] - D_Y[j,l])²

    Args:
        D_X: [N, N] distance matrix in source space
        D_Y: [M, M] distance matrix in target space
        a: [N] source marginal (uniform if None)
        b: [M] target marginal (uniform if None)
        reg: Entropic regularization
        num_gw_iters: Outer GW iterations
        num_sinkhorn_iters: Inner Sinkhorn iterations

    Returns:
        P: [N, M] optimal coupling
        cost: Final GW cost
    """
    N = D_X.shape[0]
    M = D_Y.shape[0]
    device = D_X.device
    dtype = D_X.dtype

    if a is None:
        a = torch.ones(N, device=device, dtype=dtype) / N
    if b is None:
        b = torch.ones(M, device=device, dtype=dtype) / M

    # Initialize coupling as outer product of marginals
    P = a.unsqueeze(1) * b.unsqueeze(0)

    # Precompute squared distance matrices
    D_X_sq = D_X ** 2
    D_Y_sq = D_Y ** 2

    for it in range(num_gw_iters):
        P_prev = P.clone()

        # Compute linear cost matrix for this iteration
        # C[i,j] = Σ_kl P_kl (D_X[i,k] - D_Y[j,l])²
        #        = Σ_k D_X²[i,k] Σ_l P_kl + Σ_l D_Y²[j,l] Σ_k P_kl - 2 Σ_kl P_kl D_X[i,k] D_Y[j,l]
        #        = D_X² @ P @ 1 @ 1ᵀ + 1 @ 1ᵀ @ Pᵀ @ D_Y² - 2 D_X @ P @ D_Y

        term1 = (D_X_sq @ P).sum(dim=1, keepdim=True)  # [N, 1]
        term2 = (P @ D_Y_sq).sum(dim=0, keepdim=True)  # [1, M]
        term3 = D_X @ P @ D_Y  # [N, M]

        C = term1 + term2 - 2 * term3

        # Sinkhorn step
        P = sinkhorn_log_domain(C, a, b, reg, num_sinkhorn_iters)

        # Check convergence
        diff = (P - P_prev).abs().max().item()
        if diff < 1e-6:
            break

    # Compute final GW cost
    cost = (P * C).sum().item()

    return P, cost


class BarycentricFusion(nn.Module):
    """moscot-style barycentric projection using GW coupling.

    This is the principled approach: the GW coupling matrix P defines how mass
    should be transported between HLCA and LuCA embeddings. For a query cell:
    1. Find k nearest neighbors in the reference HLCA space
    2. Use their coupling weights to compute barycentric projection into LuCA space
    3. Fuse the aligned representations

    No neural network training needed - the OT plan IS the valid transport.
    """

    def __init__(
        self,
        hlca_ref: Tensor,
        luca_ref: Tensor,
        coupling: Tensor,
        k_neighbors: int = 15,
        fused_dim: int = 40,
    ):
        super().__init__()

        # Store reference data (not parameters, just buffers for inference)
        self.register_buffer("hlca_ref", hlca_ref)  # [N_ref, hlca_dim]
        self.register_buffer("luca_ref", luca_ref)  # [N_ref, luca_dim]

        # Row-normalize coupling: P[i,:] = how cell i's mass distributes to LuCA
        coupling_normalized = coupling / (coupling.sum(dim=1, keepdim=True) + 1e-8)
        self.register_buffer("coupling", coupling_normalized)  # [N_ref, N_ref]

        self.k_neighbors = k_neighbors
        self.hlca_dim = hlca_ref.shape[1]
        self.luca_dim = luca_ref.shape[1]
        self.fused_dim = fused_dim

        # Simple linear projections to shared dimension (no complex training)
        self.hlca_proj = nn.Linear(self.hlca_dim, fused_dim)
        self.luca_proj = nn.Linear(self.luca_dim, fused_dim)
        self.output_norm = nn.LayerNorm(fused_dim)

    def _find_knn(self, query_hlca: Tensor) -> tuple[Tensor, Tensor]:
        """Find k nearest neighbors in reference HLCA space.

        Returns:
            indices: [B, k] indices into reference
            weights: [B, k] softmax of negative distances (soft assignment)
        """
        # Compute distances: [B, N_ref]
        dists = torch.cdist(query_hlca, self.hlca_ref)

        # Get top-k nearest
        top_dists, top_idx = dists.topk(self.k_neighbors, dim=1, largest=False)

        # Convert to soft weights (temperature-scaled softmax)
        temperature = top_dists.mean() + 1e-8  # Adaptive temperature
        weights = torch.softmax(-top_dists / temperature, dim=1)

        return top_idx, weights

    def forward(
        self,
        hlca: Tensor,
        luca: Tensor,
        return_coupling: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor, Tensor]:
        """Fuse HLCA and LuCA using coupling-based barycentric projection.

        For each query cell:
        1. Find k-NN in HLCA reference space
        2. Aggregate their GW coupling weights to get soft assignment to LuCA ref
        3. Compute barycentric projection: weighted average of LuCA ref points
        4. Combine query's direct LuCA with its GW-transported HLCA

        Args:
            hlca: [B, hlca_dim] query HLCA embeddings
            luca: [B, luca_dim] query LuCA embeddings
            return_coupling: Whether to return the computed soft coupling

        Returns:
            fused: [B, fused_dim] aligned/fused representation
        """
        B = hlca.shape[0]

        # Step 1: Find k-NN in HLCA reference
        knn_idx, knn_weights = self._find_knn(hlca)  # [B, k], [B, k]

        # Step 2: Aggregate coupling weights from neighbors
        # For each query, get its soft assignment to LuCA reference via k-NN coupling
        # knn_coupling[b, j] = Σ_i knn_weights[b,i] * coupling[knn_idx[b,i], j]

        # Gather coupling rows for neighbors: [B, k, N_ref]
        neighbor_couplings = self.coupling[knn_idx]  # [B, k, N_ref]

        # Weighted average: [B, N_ref]
        soft_coupling = torch.einsum("bk,bkn->bn", knn_weights, neighbor_couplings)

        # Step 3: Barycentric projection into LuCA space
        # transported_hlca[b] = Σ_j soft_coupling[b,j] * luca_ref[j]
        transported_hlca = torch.einsum("bn,nd->bd", soft_coupling, self.luca_ref)  # [B, luca_dim]

        # Step 4: Fuse direct LuCA with GW-transported HLCA
        # Project both to shared dimension
        hlca_aligned = self.hlca_proj(hlca)  # Original HLCA info
        luca_direct = self.luca_proj(luca)   # Direct LuCA
        luca_transported = self.luca_proj(transported_hlca)  # GW-transported HLCA→LuCA

        # Combine: direct HLCA + (averaged: direct LuCA, transported LuCA)
        fused = hlca_aligned + 0.5 * (luca_direct + luca_transported)
        fused = self.output_norm(fused)

        if return_coupling:
            return fused, soft_coupling, knn_idx
        return fused


class NeuralTransportMap(nn.Module):
    """DEPRECATED: Neural network approach - kept for backward compatibility.

    The original approach tried to train a neural map supervised by coupling,
    but the implementation was broken (line 336: just used MSE between encodings).

    Use BarycentricFusion instead - it directly uses the coupling matrix
    as moscot does, which is the mathematically correct approach.
    """

    def __init__(
        self,
        hlca_dim: int = 30,
        luca_dim: int = 10,
        output_dim: int = 40,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        import warnings
        warnings.warn(
            "NeuralTransportMap is deprecated - use BarycentricFusion instead. "
            "The neural map training was not correctly supervised by the GW coupling.",
            DeprecationWarning
        )

        # HLCA encoder
        hlca_layers = [nn.Linear(hlca_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
        for _ in range(num_layers - 1):
            hlca_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)])
        hlca_layers.append(nn.Linear(hidden_dim, output_dim))
        self.hlca_encoder = nn.Sequential(*hlca_layers)

        # LuCA encoder (separate pathway)
        luca_layers = [nn.Linear(luca_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
        for _ in range(num_layers - 1):
            luca_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)])
        luca_layers.append(nn.Linear(hidden_dim, output_dim))
        self.luca_encoder = nn.Sequential(*luca_layers)

        # Final fusion
        self.fusion = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, hlca: Tensor, luca: Tensor) -> Tensor:
        """Fuse HLCA and LuCA using learned aligned projections."""
        h = self.hlca_encoder(hlca)
        l = self.luca_encoder(luca)
        return self.fusion(torch.cat([h, l], dim=-1))

    def encode_hlca(self, hlca: Tensor) -> Tensor:
        """Project HLCA to aligned space."""
        return self.hlca_encoder(hlca)

    def encode_luca(self, luca: Tensor) -> Tensor:
        """Project LuCA to aligned space."""
        return self.luca_encoder(luca)


def sample_reference_cells(
    hlca_embeddings: np.ndarray,
    luca_embeddings: np.ndarray,
    n_cells: int,
    stages: np.ndarray | None = None,
    cell_types: np.ndarray | None = None,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample representative cells, optionally stratified.

    Returns:
        hlca_sample: [n_cells, hlca_dim]
        luca_sample: [n_cells, luca_dim]
        indices: Original indices of sampled cells
    """
    rng = np.random.RandomState(random_state)
    N = len(hlca_embeddings)

    if stages is None and cell_types is None:
        # Simple random sampling
        indices = rng.choice(N, size=min(n_cells, N), replace=False)
    else:
        # Stratified sampling
        strata = []
        if stages is not None:
            strata.append(stages)
        if cell_types is not None:
            strata.append(cell_types)

        # Create combined strata
        if len(strata) == 1:
            combined = strata[0]
        else:
            combined = np.array([f"{s}_{c}" for s, c in zip(*strata)])

        unique_strata = np.unique(combined)
        n_per_stratum = max(1, n_cells // len(unique_strata))

        indices = []
        for stratum in unique_strata:
            stratum_idx = np.where(combined == stratum)[0]
            n_sample = min(n_per_stratum, len(stratum_idx))
            indices.extend(rng.choice(stratum_idx, size=n_sample, replace=False))

        indices = np.array(indices)
        if len(indices) > n_cells:
            indices = rng.choice(indices, size=n_cells, replace=False)

    return hlca_embeddings[indices], luca_embeddings[indices], indices


def create_barycentric_fusion(
    hlca_ref: Tensor,
    luca_ref: Tensor,
    coupling: Tensor,
    config: GWPrecomputeConfig,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> BarycentricFusion:
    """Create barycentric fusion module using GW coupling.

    No training needed - the GW coupling matrix directly defines the valid
    transport plan. We just store it and use for barycentric projection.

    This is the moscot approach: coupling IS the solution.
    """
    model = BarycentricFusion(
        hlca_ref=hlca_ref.to(device),
        luca_ref=luca_ref.to(device),
        coupling=coupling.to(device),
        k_neighbors=min(15, hlca_ref.shape[0] // 10),  # Adaptive k
        fused_dim=config.output_dim,
    ).to(device)

    # Initialize the linear projections sensibly
    # HLCA projection: preserve variance
    with torch.no_grad():
        hlca_std = hlca_ref.std()
        model.hlca_proj.weight.normal_(0, 1.0 / (config.hlca_dim ** 0.5))
        model.hlca_proj.bias.zero_()

        luca_std = luca_ref.std()
        model.luca_proj.weight.normal_(0, 1.0 / (config.luca_dim ** 0.5))
        model.luca_proj.bias.zero_()

    print(f"  Created BarycentricFusion with k={model.k_neighbors} neighbors")
    return model


def train_neural_map(
    hlca_ref: Tensor,
    luca_ref: Tensor,
    coupling: Tensor,
    config: GWPrecomputeConfig,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> BarycentricFusion:
    """DEPRECATED: Use create_barycentric_fusion instead.

    This function now just calls create_barycentric_fusion for backward compatibility.
    The original neural map training was broken - it didn't use the coupling properly.
    """
    import warnings
    warnings.warn(
        "train_neural_map is deprecated - use create_barycentric_fusion instead",
        DeprecationWarning
    )
    return create_barycentric_fusion(hlca_ref, luca_ref, coupling, config, device)


def precompute_gw_alignment(
    hlca_embeddings: np.ndarray,
    luca_embeddings: np.ndarray,
    output_dir: Path,
    config: GWPrecomputeConfig | None = None,
    stages: np.ndarray | None = None,
    cell_types: np.ndarray | None = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict:
    """Main function to precompute GW alignment and train neural map.

    Args:
        hlca_embeddings: [N, 30] full dataset HLCA embeddings
        luca_embeddings: [N, 10] full dataset LuCA embeddings
        output_dir: Where to save coupling, neural map, and metadata
        config: Precomputation config
        stages: Optional stage labels for stratification
        cell_types: Optional cell type labels for stratification
        device: Compute device

    Returns:
        dict with coupling, cost, model path, and metadata
    """
    if config is None:
        config = GWPrecomputeConfig()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Precomputing GW alignment with {config.n_reference_cells} reference cells...")

    # Sample reference cells
    print("  Sampling reference cells...")
    hlca_ref, luca_ref, ref_indices = sample_reference_cells(
        hlca_embeddings, luca_embeddings,
        n_cells=config.n_reference_cells,
        stages=stages if config.stratify_by_stage else None,
        cell_types=cell_types if config.stratify_by_celltype else None,
    )

    # Convert to tensors
    hlca_ref_t = torch.from_numpy(hlca_ref).float().to(device)
    luca_ref_t = torch.from_numpy(luca_ref).float().to(device)

    # Compute distance matrices
    print("  Computing distance matrices...")
    D_hlca = pairwise_distances(hlca_ref_t)
    D_luca = pairwise_distances(luca_ref_t)

    # Normalize distances
    D_hlca = D_hlca / D_hlca.max()
    D_luca = D_luca / D_luca.max()

    # Solve GW
    print(f"  Solving Gromov-Wasserstein (reg={config.sinkhorn_reg}, iters={config.gw_iters})...")
    coupling, gw_cost = gromov_wasserstein(
        D_hlca, D_luca,
        reg=config.sinkhorn_reg,
        num_gw_iters=config.gw_iters,
        num_sinkhorn_iters=config.sinkhorn_iters,
    )
    print(f"  GW cost: {gw_cost:.4f}")

    # Create barycentric fusion (no training - coupling IS the solution)
    print("  Creating barycentric fusion module...")
    fusion_model = create_barycentric_fusion(hlca_ref_t, luca_ref_t, coupling, config, device)

    # Save everything
    print("  Saving results...")

    # Coupling matrix (the core OT artifact)
    torch.save(coupling.cpu(), output_dir / "gw_coupling.pt")

    # Fusion model (includes reference data and linear projections)
    torch.save(fusion_model.state_dict(), output_dir / "barycentric_fusion.pt")

    # Legacy: also save as neural_transport_map.pt for backward compatibility
    torch.save(fusion_model.state_dict(), output_dir / "neural_transport_map.pt")

    # Reference data
    np.savez(
        output_dir / "reference_data.npz",
        hlca=hlca_ref,
        luca=luca_ref,
        indices=ref_indices,
    )

    # Metadata
    metadata = {
        "n_reference_cells": len(ref_indices),
        "gw_cost": gw_cost,
        "config": {
            "n_reference_cells": config.n_reference_cells,
            "hlca_dim": config.hlca_dim,
            "luca_dim": config.luca_dim,
            "output_dim": config.output_dim,
            "sinkhorn_reg": config.sinkhorn_reg,
            "gw_iters": config.gw_iters,
        }
    }
    with open(output_dir / "gw_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Done! Results saved to {output_dir}")

    return {
        "coupling": coupling.cpu(),
        "gw_cost": gw_cost,
        "fusion_model": fusion_model,
        "neural_map": fusion_model,  # Backward compatibility alias
        "reference_indices": ref_indices,
        "output_dir": output_dir,
    }


class PretrainedGWFusion(nn.Module):
    """GW fusion using precomputed alignment.

    Loads the BarycentricFusion model which uses the GW coupling for
    principled transport. No per-batch GW computation.
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        checkpoint_dir = Path(checkpoint_dir)

        # Load metadata
        with open(checkpoint_dir / "gw_metadata.json") as f:
            metadata = json.load(f)

        config = metadata["config"]

        # Load reference data (needed for BarycentricFusion)
        ref_data = np.load(checkpoint_dir / "reference_data.npz")
        hlca_ref = torch.from_numpy(ref_data["hlca"]).float()
        luca_ref = torch.from_numpy(ref_data["luca"]).float()

        # Load coupling
        coupling = torch.load(checkpoint_dir / "gw_coupling.pt", map_location="cpu")

        # Create fusion model
        self.fusion = BarycentricFusion(
            hlca_ref=hlca_ref,
            luca_ref=luca_ref,
            coupling=coupling,
            k_neighbors=min(15, len(hlca_ref) // 10),
            fused_dim=config["output_dim"],
        ).to(device)

        # Try to load trained projection weights if they exist
        fusion_path = checkpoint_dir / "barycentric_fusion.pt"
        if fusion_path.exists():
            state = torch.load(fusion_path, map_location=device)
            # Only load the projection layers (buffers are already set)
            proj_state = {k: v for k, v in state.items()
                        if "proj" in k or "norm" in k}
            if proj_state:
                self.fusion.load_state_dict(proj_state, strict=False)

        self.fusion.eval()
        self.output_dim = config["output_dim"]

    def forward(
        self,
        hlca: Tensor,
        luca: Tensor,
        return_coupling: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor, Tensor]:
        """Fuse HLCA and LuCA using coupling-based barycentric projection.

        Args:
            hlca: [B, 30] HLCA embeddings
            luca: [B, 10] LuCA embeddings
            return_coupling: Whether to return the soft coupling

        Returns:
            fused: [B, output_dim] fused representation
            (if return_coupling: also soft_coupling and knn_indices)
        """
        return self.fusion(hlca, luca, return_coupling=return_coupling)
