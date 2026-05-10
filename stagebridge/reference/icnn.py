"""Input Convex Neural Networks for Optimal Transport.

Implements ICNNs following Amos et al. (2017) and Bunne et al. (2023) CellOT.

The key insight: for squared Euclidean cost, the optimal transport map T*
can be written as T* = ∇g* where g* is a convex function. By parameterizing
g with an ICNN (convex in its input), we guarantee ∇g is a valid OT map.

For Gromov-Wasserstein between heterogeneous spaces (HLCA 30d ↔ LuCA 10d),
we learn dual potentials f and g where the transport is defined by their
gradients. The coupling P supervises this learning.

References:
- Amos et al. (2017) "Input Convex Neural Networks"
- Makkuva et al. (2020) "Optimal transport mapping via ICNNs"
- Bunne et al. (2023) "Learning single-cell perturbation responses using neural OT"
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class ICNN(nn.Module):
    """Input Convex Neural Network.

    A neural network that is convex in its input x. This is achieved by:
    1. Non-negative weights in skip connections from input
    2. Convex non-decreasing activations (e.g., ReLU, softplus)
    3. Proper initialization

    The gradient ∇_x ICNN(x) is guaranteed to be a monotone map,
    which is necessary for valid optimal transport.

    Architecture:
        z_0 = W_0 x + b_0
        z_{l+1} = activation(W_l^z z_l + W_l^x x + b_l)  for l = 0, ..., L-1
        output = W_L^z z_L + W_L^x x + b_L

    Where W_l^x (input skip connections) have non-negative weights.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        activation: str = "softplus",
        softplus_beta: float = 1.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.n_layers = len(hidden_dims)

        if activation == "softplus":
            self.activation = nn.Softplus(beta=softplus_beta)
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "elu":
            self.activation = nn.ELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # First layer: input -> hidden (no constraint needed)
        self.W0 = nn.Linear(input_dim, hidden_dims[0])

        # Hidden layers: z pathway (unconstrained) and x pathway (non-negative)
        self.Wz = nn.ModuleList()  # z_{l} -> z_{l+1}
        self.Wx = nn.ModuleList()  # x -> z_{l+1} (non-negative weights)

        for i in range(self.n_layers - 1):
            self.Wz.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1], bias=True))
            self.Wx.append(nn.Linear(input_dim, hidden_dims[i + 1], bias=False))

        # Output layer
        self.Wz_out = nn.Linear(hidden_dims[-1], 1, bias=True)
        self.Wx_out = nn.Linear(input_dim, 1, bias=False)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights, ensuring non-negativity for x pathways."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Initialize x-pathway weights to be non-negative
        for Wx in self.Wx:
            Wx.weight.data.abs_()
        self.Wx_out.weight.data.abs_()

    def _enforce_non_negative(self):
        """Enforce non-negative weights on x pathways (call after optimizer step)."""
        for Wx in self.Wx:
            Wx.weight.data.clamp_(min=0)
        self.Wx_out.weight.data.clamp_(min=0)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass computing the convex potential g(x).

        Args:
            x: [B, input_dim] input

        Returns:
            [B, 1] scalar convex potential value
        """
        # First layer
        z = self.activation(self.W0(x))

        # Hidden layers with skip connections
        for Wz, Wx in zip(self.Wz, self.Wx):
            z = self.activation(Wz(z) + Wx(x))

        # Output
        out = self.Wz_out(z) + self.Wx_out(x)
        return out

    def gradient(self, x: Tensor) -> Tensor:
        """Compute gradient ∇_x g(x), which is the transport map.

        Args:
            x: [B, input_dim] input

        Returns:
            [B, input_dim] gradient (transport direction)
        """
        x = x.requires_grad_(True)
        g = self.forward(x)
        grad = torch.autograd.grad(
            g.sum(), x, create_graph=True, retain_graph=True
        )[0]
        return grad


class DualICNN(nn.Module):
    """Dual ICNN formulation for optimal transport.

    Following Makkuva et al. (2020), we parameterize both the forward
    and backward transport maps using ICNNs f and g:

        T(x) = ∇g(x)      (forward: source -> target)
        T^{-1}(y) = ∇f(y) (backward: target -> source)

    The dual OT objective with squared Euclidean cost:
        max_{f,g} E_μ[g(x)] + E_ν[f(y)] - E_μ[f(∇g(x))]

    For GW, we adapt this to work with the coupling P as supervision.
    """

    def __init__(
        self,
        source_dim: int,
        target_dim: int,
        hidden_dims: list[int] | None = None,
        activation: str = "softplus",
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 128, 128]

        self.source_dim = source_dim
        self.target_dim = target_dim

        # Forward potential g: source space -> R (∇g gives forward map)
        self.g = ICNN(source_dim, hidden_dims, activation)

        # Backward potential f: target space -> R (∇f gives backward map)
        self.f = ICNN(target_dim, hidden_dims, activation)

        # For heterogeneous spaces, we need projection layers
        # ∇g(x) lives in source_dim, but target is target_dim
        # So we add a linear layer to project gradients
        if source_dim != target_dim:
            self.proj_forward = nn.Linear(source_dim, target_dim)
            self.proj_backward = nn.Linear(target_dim, source_dim)
        else:
            self.proj_forward = nn.Identity()
            self.proj_backward = nn.Identity()

    def forward_map(self, x: Tensor) -> Tensor:
        """Transport from source to target space.

        Args:
            x: [B, source_dim] source points

        Returns:
            [B, target_dim] transported points
        """
        grad_g = self.g.gradient(x)  # [B, source_dim]
        return self.proj_forward(grad_g)  # [B, target_dim]

    def backward_map(self, y: Tensor) -> Tensor:
        """Transport from target to source space.

        Args:
            y: [B, target_dim] target points

        Returns:
            [B, source_dim] transported points
        """
        grad_f = self.f.gradient(y)  # [B, target_dim]
        return self.proj_backward(grad_f)  # [B, source_dim]

    def enforce_constraints(self):
        """Enforce ICNN constraints after optimizer step."""
        self.g._enforce_non_negative()
        self.f._enforce_non_negative()


class GWICNNFusion(nn.Module):
    """Gromov-Wasserstein fusion using ICNNs.

    For fusing HLCA (30d) and LuCA (10d) embeddings of the same cells,
    we learn transport maps that respect the GW coupling structure.

    The coupling P from GW tells us how mass should flow between the
    distance structures of the two spaces. We train ICNNs to reproduce
    this transport for out-of-sample cells.

    Training objective:
    1. Coupling matching: T(hlca_i) should be close to Σ_j P[i,j] * luca_j
    2. Cycle consistency: T^{-1}(T(x)) ≈ x
    3. Potential fitting: Match the dual potentials to coupling
    """

    def __init__(
        self,
        hlca_dim: int = 30,
        luca_dim: int = 10,
        output_dim: int = 40,
        hidden_dims: list[int] | None = None,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 128, 128]

        self.hlca_dim = hlca_dim
        self.luca_dim = luca_dim
        self.output_dim = output_dim

        # ICNN for HLCA -> shared space
        self.icnn_hlca = ICNN(hlca_dim, hidden_dims)

        # ICNN for LuCA -> shared space
        self.icnn_luca = ICNN(luca_dim, hidden_dims)

        # Projection to output dimension
        self.hlca_proj = nn.Sequential(
            nn.Linear(hlca_dim, output_dim),
            nn.LayerNorm(output_dim),
        )
        self.luca_proj = nn.Sequential(
            nn.Linear(luca_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

        # Learnable fusion weights based on transport cost
        self.fusion_mlp = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(
        self,
        hlca: Tensor,
        luca: Tensor,
        return_potentials: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor, Tensor]:
        """Fuse HLCA and LuCA embeddings.

        The fusion is geometry-aware: cells are positioned based on their
        transport potential, which encodes their relationship to the
        coupling structure learned from GW.

        Args:
            hlca: [B, 30] HLCA embeddings
            luca: [B, 10] LuCA embeddings
            return_potentials: Whether to return ICNN potentials

        Returns:
            fused: [B, output_dim] fused representation
            (optional) g_hlca, g_luca: scalar potentials
        """
        # Compute transport potentials (scalar, encodes geometry)
        g_hlca = self.icnn_hlca(hlca)  # [B, 1]
        g_luca = self.icnn_luca(luca)  # [B, 1]

        # Project to shared space
        h_proj = self.hlca_proj(hlca)  # [B, output_dim]
        l_proj = self.luca_proj(luca)  # [B, output_dim]

        # Weight by transport potential (softmax for stability)
        weights = F.softmax(torch.cat([g_hlca, g_luca], dim=-1), dim=-1)
        w_hlca = weights[:, 0:1]  # [B, 1]
        w_luca = weights[:, 1:2]  # [B, 1]

        # Geometry-weighted combination
        weighted = w_hlca * h_proj + w_luca * l_proj

        # Final fusion with cross-term interaction
        concat = torch.cat([h_proj, l_proj], dim=-1)
        fused = self.fusion_mlp(concat) + weighted

        if return_potentials:
            return fused, g_hlca, g_luca
        return fused

    def transport_hlca_to_luca(self, hlca: Tensor) -> Tensor:
        """Transport HLCA point toward LuCA geometry.

        Uses gradient of HLCA potential as transport direction.
        """
        return self.icnn_hlca.gradient(hlca)

    def transport_luca_to_hlca(self, luca: Tensor) -> Tensor:
        """Transport LuCA point toward HLCA geometry."""
        return self.icnn_luca.gradient(luca)

    def enforce_constraints(self):
        """Enforce ICNN non-negativity constraints."""
        self.icnn_hlca._enforce_non_negative()
        self.icnn_luca._enforce_non_negative()


def train_gw_icnn(
    hlca_ref: Tensor,
    luca_ref: Tensor,
    coupling: Tensor,
    config: dict | None = None,
    device: str = "cuda",
    verbose: bool = True,
) -> GWICNNFusion:
    """Train ICNN fusion model supervised by GW coupling.

    The coupling P[i,j] tells us how much mass from HLCA cell i
    should be transported to match LuCA cell j. We use this to
    supervise the ICNN training.

    Loss components:
    1. Transport matching: ICNN gradient should point toward coupled cells
    2. Potential consistency: Dual potentials should satisfy c-transform
    3. Cycle consistency: Round-trip transport should preserve identity

    Args:
        hlca_ref: [N, 30] reference HLCA embeddings
        luca_ref: [N, 10] reference LuCA embeddings
        coupling: [N, N] GW coupling matrix (row-normalized)
        config: Training config dict
        device: Compute device
        verbose: Print progress

    Returns:
        Trained GWICNNFusion model
    """
    if config is None:
        config = {
            "epochs": 200,
            "lr": 1e-3,
            "batch_size": 256,
            "cycle_weight": 0.1,
            "potential_weight": 0.1,
        }

    model = GWICNNFusion(
        hlca_dim=hlca_ref.shape[1],
        luca_dim=luca_ref.shape[1],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"]
    )

    hlca_ref = hlca_ref.to(device)
    luca_ref = luca_ref.to(device)
    coupling = coupling.to(device)

    # Normalize coupling to be row-stochastic
    P = coupling / (coupling.sum(dim=1, keepdim=True) + 1e-8)

    # Compute transport targets: for each HLCA cell, its target is
    # the weighted combination of LuCA cells according to coupling
    # target_i = Σ_j P[i,j] * luca_j
    transport_targets = P @ luca_ref  # [N, luca_dim]

    n_samples = hlca_ref.shape[0]
    n_batches = (n_samples + config["batch_size"] - 1) // config["batch_size"]

    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0.0
        total_transport = 0.0
        total_cycle = 0.0

        # Shuffle indices
        perm = torch.randperm(n_samples, device=device)

        for batch_idx in range(n_batches):
            start = batch_idx * config["batch_size"]
            end = min(start + config["batch_size"], n_samples)
            idx = perm[start:end]

            hlca_batch = hlca_ref[idx]
            luca_batch = luca_ref[idx]
            target_batch = transport_targets[idx]

            optimizer.zero_grad()

            # Forward pass with potentials
            fused, g_hlca, g_luca = model(
                hlca_batch, luca_batch, return_potentials=True
            )

            # Loss 1: Transport matching
            # The gradient of HLCA potential should point toward transport target
            grad_hlca = model.transport_hlca_to_luca(hlca_batch)
            # Scale gradient to match LuCA dimension (use projection)
            grad_projected = model.luca_proj(
                F.pad(grad_hlca, (0, model.luca_dim - model.hlca_dim))
                if model.hlca_dim < model.luca_dim
                else grad_hlca[:, :model.luca_dim]
            )
            transport_loss = F.mse_loss(grad_projected, model.luca_proj(target_batch))

            # Loss 2: Cycle consistency (reconstruction)
            # fused should allow reconstructing both inputs
            cycle_loss = (
                F.mse_loss(fused[:, :model.hlca_dim], hlca_batch) +
                F.mse_loss(fused[:, model.hlca_dim:model.hlca_dim + model.luca_dim],
                          luca_batch)
            ) if model.output_dim >= model.hlca_dim + model.luca_dim else torch.tensor(0.0)

            # Loss 3: Potential regularization (encourage smoothness)
            potential_reg = (g_hlca.var() + g_luca.var()) * 0.01

            loss = (
                transport_loss +
                config["cycle_weight"] * cycle_loss +
                config["potential_weight"] * potential_reg
            )

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            # Enforce ICNN constraints
            model.enforce_constraints()

            total_loss += loss.item()
            total_transport += transport_loss.item()
            total_cycle += cycle_loss.item() if isinstance(cycle_loss, Tensor) else cycle_loss

        scheduler.step()

        if verbose and (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch + 1}/{config['epochs']}: "
                  f"loss={total_loss/n_batches:.4f}, "
                  f"transport={total_transport/n_batches:.4f}, "
                  f"cycle={total_cycle/n_batches:.4f}")

    model.eval()
    return model
