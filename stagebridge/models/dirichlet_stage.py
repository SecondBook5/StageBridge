"""Dirichlet stage assignment posterior for StageBridge.

Replaces hard integer stage labels with a Dirichlet distribution over the
five progression stages.  Each cell receives concentration parameters
α ∈ ℝ^5_+ from which the expected stage probability and stage assignment
entropy are derived.

Biological interpretation
-------------------------
* **Expected distribution** E[p] = α / Σα — the most likely stage mix.
* **Entropy** H[Dir(α)] — high entropy = the cell is at a transition point
  between stages (biologically ambiguous / plastic state).
* **Transition cells** = cells with the highest entropy are in the act of
  malignant transformation and are the most therapeutically relevant.

Training
--------
Two complementary supervision signals:

1. **NLL supervision**: negative log-likelihood of observed hard stage label
   under the Dirichlet-induced categorical:
       L_nll = -log(α_k / Σα)  for ground-truth stage k

2. **Trajectory consistency**: predicted endpoint distribution should "look
   like" the target stage.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor, nn

if TYPE_CHECKING:
    import numpy as np

# --------------------------------------------------------------------------
# Module
# --------------------------------------------------------------------------

class DirichletStageHead(nn.Module):
    """Predicts Dirichlet concentration parameters for stage assignment.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input cell representation (HLCA latent dim).
    num_stages : int
        Number of cancer progression stages (default 5).
    hidden_dim : int
        Hidden dimension of the two-layer MLP.
    min_alpha : float
        Minimum concentration value (added after softplus for numerical
        stability).  Values > 1 produce unimodal Dirichlet distributions;
        values in (0, 1) produce multimodal ones.
    """

    def __init__(
        self,
        input_dim: int,
        num_stages: int = 5,
        hidden_dim: int = 64,
        min_alpha: float = 1e-2,
    ) -> None:
        super().__init__()
        self.num_stages = num_stages
        self.min_alpha = min_alpha

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_stages),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Return Dirichlet concentration parameters.

        Parameters
        ----------
        x : Tensor
            Shape ``(B, input_dim)``.

        Returns
        -------
        Tensor
            Shape ``(B, num_stages)`` — Dirichlet α (all positive).
        """
        logits = self.mlp(x)
        # softplus ensures positivity; +min_alpha prevents α→0 collapse
        return F.softplus(logits) + self.min_alpha

    def predict_distribution(self, x: Tensor) -> Tensor:
        """Expected stage probabilities E[Dir(α)] = α / Σα.

        Returns
        -------
        Tensor
            Shape ``(B, num_stages)`` — probability simplex.
        """
        alpha = self(x)
        return alpha / alpha.sum(dim=-1, keepdim=True)

    def predict_entropy(self, x: Tensor) -> Tensor:
        """Differential entropy H[Dir(α)] of the Dirichlet distribution.

        Uses the closed-form expression:
            H = log B(α) + (α₀ - K) ψ(α₀) - Σ (αₖ - 1) ψ(αₖ)
        where ψ is the digamma function and B(α) is the multivariate
        beta function.

        Returns
        -------
        Tensor
            Shape ``(B,)`` — entropy value (higher = more uncertain/plastic).
        """
        alpha = self(x)
        return _dirichlet_entropy(alpha)

    def predict_mode(self, x: Tensor) -> Tensor:
        """Hard stage prediction (argmax of expected probabilities).

        Returns
        -------
        Tensor
            Shape ``(B,)`` — integer stage indices.
        """
        probs = self.predict_distribution(x)
        return probs.argmax(dim=-1)

    def stage_uncertainty(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return (alpha, expected_probs, entropy) in one forward pass.

        Returns
        -------
        alpha : (B, num_stages)
        probs : (B, num_stages)
        entropy : (B,)
        """
        alpha = self(x)
        alpha0 = alpha.sum(dim=-1, keepdim=True)
        probs = alpha / alpha0
        entropy = _dirichlet_entropy(alpha)
        return alpha, probs, entropy


# --------------------------------------------------------------------------
# Loss functions
# --------------------------------------------------------------------------

def dirichlet_nll_loss(alpha: Tensor, stage_targets: Tensor) -> Tensor:
    """Negative log-likelihood of hard stage labels under Dirichlet-categorical.

    P(stage=k | α) = α_k / Σα
    L = -log P(stage=k | α)

    Parameters
    ----------
    alpha : Tensor
        Shape ``(B, K)`` — Dirichlet concentration parameters.
    stage_targets : Tensor
        Shape ``(B,)`` — integer stage indices in [0, K).

    Returns
    -------
    Tensor
        Scalar mean NLL loss.
    """
    # log P(k) = log(α_k) - log(Σα)
    log_probs = torch.log(alpha + 1e-8) - torch.log(alpha.sum(dim=-1, keepdim=True) + 1e-8)
    return F.nll_loss(log_probs, stage_targets.long())


def dirichlet_consistency_loss(
    alpha_pred: Tensor,
    stage_target_idx: int,
    weight: float = 1.0,
) -> Tensor:
    """Encourage predicted distribution to concentrate on the target stage.

    Penalises the KL divergence from a Dirichlet(α_pred) to the degenerate
    distribution concentrated at stage_target_idx.  Implemented as
    cross-entropy from predicted distribution to one-hot target.

    Parameters
    ----------
    alpha_pred : Tensor
        Shape ``(B, K)`` — predicted Dirichlet parameters.
    stage_target_idx : int
        Index of the target stage (ground-truth).
    weight : float
        Loss weighting coefficient.

    Returns
    -------
    Tensor
        Scalar cross-entropy loss × weight.
    """
    B = alpha_pred.shape[0]
    target = torch.full((B,), stage_target_idx, dtype=torch.long, device=alpha_pred.device)
    return weight * dirichlet_nll_loss(alpha_pred, target)


def dirichlet_entropy_regularization(alpha: Tensor, target_entropy: float = 0.5) -> Tensor:
    """Regularise entropy toward a target value (prevents distribution collapse).

    If target_entropy is positive, encourages the model to maintain some
    uncertainty rather than always predicting hard stage labels.

    Parameters
    ----------
    alpha : Tensor
        Shape ``(B, K)``.
    target_entropy : float
        Target entropy in nats (0.5 is mild regularisation).

    Returns
    -------
    Tensor
        Scalar L2 penalty on deviation from target entropy.
    """
    H = _dirichlet_entropy(alpha)  # (B,)
    return F.mse_loss(H, torch.full_like(H, target_entropy))


# --------------------------------------------------------------------------
# Patient-level aggregation
# --------------------------------------------------------------------------

def compute_stage_entropy_per_patient(
    alpha: Tensor,
    donor_ids: "np.ndarray",
) -> dict[str, float]:
    """Compute mean stage assignment entropy per patient.

    This is the *progression entropy* biomarker: a patient with cells
    spread uniformly across stages has H ≈ log(K) (maximum heterogeneity),
    while a patient with homogeneous stage assignment has H ≈ 0.

    Parameters
    ----------
    alpha : Tensor
        Shape ``(N_cells, K)`` — Dirichlet parameters for all cells.
    donor_ids : np.ndarray
        Shape ``(N_cells,)`` — donor / patient identifier per cell.

    Returns
    -------
    dict[str, float]
        Maps donor_id → mean entropy over their cells.
    """
    import numpy as np
    H = _dirichlet_entropy(alpha).detach().cpu().numpy()  # (N_cells,)
    unique_donors = np.unique(donor_ids)
    return {
        donor: float(H[donor_ids == donor].mean())
        for donor in unique_donors
    }


def compute_stage_distribution_per_patient(
    alpha: Tensor,
    donor_ids: "np.ndarray",
    num_stages: int = 5,
) -> dict[str, "np.ndarray"]:
    """Compute mean expected stage distribution per patient.

    Returns
    -------
    dict[str, np.ndarray]
        Maps donor_id → (K,) expected probability vector.
    """
    import numpy as np
    probs = (alpha / alpha.sum(dim=-1, keepdim=True)).detach().cpu().numpy()  # (N, K)
    unique_donors = np.unique(donor_ids)
    return {
        donor: probs[donor_ids == donor].mean(axis=0)
        for donor in unique_donors
    }


# --------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------

def _dirichlet_entropy(alpha: Tensor) -> Tensor:
    """Closed-form differential entropy of Dir(α).

    H = log B(α) + (α₀ - K) ψ(α₀) - Σ (αₖ - 1) ψ(αₖ)

    where B(α) = Π Γ(αₖ) / Γ(α₀), ψ = digamma.

    Parameters
    ----------
    alpha : Tensor
        Shape ``(B, K)``.

    Returns
    -------
    Tensor
        Shape ``(B,)`` — entropy in nats.
    """
    alpha0 = alpha.sum(dim=-1)  # (B,)
    K = alpha.shape[-1]

    # log B(α) = Σ log Γ(αₖ) - log Γ(α₀)
    log_B = torch.lgamma(alpha).sum(dim=-1) - torch.lgamma(alpha0)

    # (α₀ - K) ψ(α₀)
    term1 = (alpha0 - K) * torch.digamma(alpha0)

    # -Σ (αₖ - 1) ψ(αₖ)
    term2 = -((alpha - 1.0) * torch.digamma(alpha)).sum(dim=-1)

    return log_B + term1 + term2
