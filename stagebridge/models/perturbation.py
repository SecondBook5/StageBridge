"""Causal perturbation analysis for StageBridge.

Implements in-silico perturbation to identify driver gene programs for
malignant stage progression.

Core idea
---------
For a cell x at source stage S, define a perturbation Δ and compute:

    x_T^pred       = integrate(x,     S→T, num_steps)   # unperturbed endpoint
    x_T^intervened = integrate(x + Δ, S→T, num_steps)   # perturbed endpoint

Three scores capture causal influence:

* **Effect score**: ||x_T^pred - x_T^intervened||₂ / ||Δ||₂
  — normalised sensitivity of the endpoint to the perturbation direction.

* **Reversal score**: cosine(x_T^pred - x_S, x_T^intervened - x_S_interv)
  — negative cosine means the perturbation *reverses* the trajectory.
  A reversal score < 0 identifies candidate therapeutic targets.

* **Flow Jacobian**: ∂x_T/∂x₀ — first-order Taylor approximation valid for
  small ||Δ||.  Efficient via ``torch.autograd.functional.jacobian``.
  For input_dim=30 (HLCA), this is a 30×30 matrix computable in <1 s.

Usage
-----
>>> analyzer = CausalPerturbationAnalyzer(model, device="cuda")
>>> directions = analyzer.define_perturbation_directions_stage_delta(
...     x_normal, x_luad, n_directions=30
... )
>>> results = analyzer.compute_perturbation_effects(
...     x_src, c_s, stage_pair_id, directions
... )
>>> # Directions ranked by reversal_score give putative therapeutic targets
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class PerturbationResult:
    """Results of a causal perturbation analysis run.

    Attributes
    ----------
    directions : (K, input_dim)
        Perturbation directions (unit-normalised).
    effect_scores : (K,)
        Normalised endpoint sensitivity for each direction.
    reversal_scores : (K,)
        Cosine of trajectory reversal; negative = therapy candidate.
    endpoint_shifts : (K, input_dim)
        Mean endpoint displacement x_T^intervened - x_T^pred.
    jacobian : (input_dim, input_dim) or None
        First-order Jacobian ∂x_T/∂x₀ (if computed).
    ranked_indices : (K,)
        Indices into ``directions`` sorted by reversal_score (most reversed first).
    """
    directions: Tensor
    effect_scores: Tensor
    reversal_scores: Tensor
    endpoint_shifts: Tensor
    jacobian: Tensor | None = None
    ranked_indices: Tensor = field(init=False)

    def __post_init__(self) -> None:
        # Sort by reversal score ascending (most reversed = best therapeutic target)
        self.ranked_indices = torch.argsort(self.reversal_scores)

    def top_k_driver_directions(self, k: int = 10) -> Tensor:
        """Return the top-k most reversing perturbation directions."""
        return self.directions[self.ranked_indices[:k]]

    def summary_dict(self, k: int = 10) -> dict[str, Any]:
        """Compact summary suitable for logging."""
        idx = self.ranked_indices
        return {
            "top_reversal_score": float(self.reversal_scores[idx[0]]),
            "mean_reversal_score": float(self.reversal_scores.mean()),
            "top_effect_score": float(self.effect_scores[idx[0]]),
            "mean_effect_score": float(self.effect_scores.mean()),
            "frac_reversing": float((self.reversal_scores < 0).float().mean()),
        }


# ---------------------------------------------------------------------------
# Main analyzer class
# ---------------------------------------------------------------------------

class CausalPerturbationAnalyzer:
    """Post-hoc causal perturbation analysis for a trained StageBridge model.

    No retraining required.  All methods run in inference mode.

    Parameters
    ----------
    model : StageBridgeModel
        Trained model with ``integrate_euler`` and ``forward_set_context``.
    device : str
        Torch device for computation.
    """

    def __init__(self, model: Any, device: str = "cuda") -> None:
        self.model = model
        self.device = device
        self.model.eval()

    # ------------------------------------------------------------------
    # Primary analysis entry point
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute_perturbation_effects(
        self,
        x_src: Tensor,
        c_s: Tensor,
        stage_pair_id: Tensor,
        perturbation_directions: Tensor,
        effect_size: float = 1.0,
        num_steps: int = 8,
        wes_features: Tensor | None = None,
    ) -> PerturbationResult:
        """Compute endpoint sensitivity to each perturbation direction.

        Parameters
        ----------
        x_src : Tensor
            Shape ``(B, input_dim)`` — source cells.
        c_s : Tensor
            Shape ``(B, context_dim)`` or ``(1, context_dim)`` — set context.
        stage_pair_id : Tensor
            Shape ``(B,)`` or scalar — stage-pair index.
        perturbation_directions : Tensor
            Shape ``(K, input_dim)`` — perturbation axes (need not be normalised).
        effect_size : float
            L2 norm of the perturbation applied along each direction.
        num_steps : int
            Euler integration steps.
        wes_features : Tensor or None
            Optional WES conditioning.

        Returns
        -------
        PerturbationResult
        """
        x_src = x_src.to(self.device)
        c_s = c_s.to(self.device)
        stage_pair_id = stage_pair_id.to(self.device)
        perturbation_directions = perturbation_directions.to(self.device)
        if wes_features is not None:
            wes_features = wes_features.to(self.device)

        # Unit-normalise directions
        dir_norms = perturbation_directions.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        dirs_unit = perturbation_directions / dir_norms  # (K, D)

        # Unperturbed endpoint — single pass
        x_T_base = self.model.integrate_euler(
            x0=x_src, c_s=c_s, stage_pair_id=stage_pair_id,
            num_steps=num_steps, wes_features=wes_features,
        )  # (B, D)

        K = dirs_unit.shape[0]
        B, D = x_src.shape

        effect_scores = torch.zeros(K, device=self.device)
        reversal_scores = torch.zeros(K, device=self.device)
        endpoint_shifts = torch.zeros(K, D, device=self.device)

        # Trajectory direction from unperturbed source to endpoint
        traj_dir = (x_T_base - x_src).mean(dim=0)  # (D,) — mean drift direction

        for ki in range(K):
            delta = effect_size * dirs_unit[ki]  # (D,)
            x_perturbed = x_src + delta.unsqueeze(0)  # (B, D)

            x_T_interv = self.model.integrate_euler(
                x0=x_perturbed, c_s=c_s, stage_pair_id=stage_pair_id,
                num_steps=num_steps, wes_features=wes_features,
            )  # (B, D)

            shift = (x_T_interv - x_T_base).mean(dim=0)  # (D,)
            endpoint_shifts[ki] = shift

            # Effect score: normalised endpoint displacement
            effect_scores[ki] = shift.norm() / max(effect_size, 1e-8)

            # Reversal score: cosine(mean trajectory, perturbed trajectory)
            traj_interv = (x_T_interv - x_perturbed).mean(dim=0)  # (D,)
            reversal_scores[ki] = F.cosine_similarity(
                traj_dir.unsqueeze(0), traj_interv.unsqueeze(0)
            ).squeeze()

        return PerturbationResult(
            directions=dirs_unit.cpu(),
            effect_scores=effect_scores.cpu(),
            reversal_scores=reversal_scores.cpu(),
            endpoint_shifts=endpoint_shifts.cpu(),
        )

    # ------------------------------------------------------------------
    # Jacobian-based first-order analysis
    # ------------------------------------------------------------------

    def compute_flow_jacobian(
        self,
        x0: Tensor,
        c_s: Tensor,
        stage_pair_id: Tensor,
        num_steps: int = 8,
        wes_features: Tensor | None = None,
    ) -> Tensor:
        """Compute the flow-map Jacobian ∂x_T/∂x₀ via autograd.

        Efficient for ``input_dim ≤ 64`` (e.g., HLCA latent dim=30).
        The Jacobian encodes first-order sensitivity: the row J[i, :] gives the
        direction in input space that most affects output dimension i.

        Parameters
        ----------
        x0 : Tensor
            Shape ``(1, input_dim)`` — single source cell.
        c_s, stage_pair_id, wes_features
            Model conditioning (passed through).
        num_steps : int
            Euler steps.

        Returns
        -------
        Tensor
            Shape ``(input_dim, input_dim)`` — Jacobian matrix.
        """
        x0 = x0.to(self.device).requires_grad_(True)
        c_s = c_s.to(self.device)
        stage_pair_id = stage_pair_id.to(self.device)

        def _flow(x: Tensor) -> Tensor:
            return self.model.integrate_euler(
                x0=x, c_s=c_s, stage_pair_id=stage_pair_id,
                num_steps=num_steps, wes_features=wes_features,
            ).squeeze(0)  # (D,)

        J = torch.autograd.functional.jacobian(_flow, x0.squeeze(0))  # (D_out, D_in)
        return J.detach().cpu()

    def jacobian_top_directions(
        self,
        jacobian: Tensor,
        n: int = 10,
    ) -> tuple[Tensor, Tensor]:
        """Extract top-n singular vectors of the Jacobian.

        The leading left singular vectors indicate the output directions most
        sensitive to input perturbation; the right singular vectors give the
        input directions with the largest amplified effect.

        Returns
        -------
        (input_directions, singular_values) : (n, D), (n,)
            Right singular vectors ranked by singular value descending.
        """
        U, S, Vh = torch.linalg.svd(jacobian, full_matrices=False)
        return Vh[:n], S[:n]

    # ------------------------------------------------------------------
    # Reversal direction search
    # ------------------------------------------------------------------

    @torch.no_grad()
    def find_stage_reversal_directions(
        self,
        x_src: Tensor,
        c_s: Tensor,
        stage_pair_id: Tensor,
        stage_centroid_src: Tensor,
        n_directions: int = 50,
        num_steps: int = 8,
        effect_size: float = 1.0,
        wes_features: Tensor | None = None,
    ) -> PerturbationResult:
        """Find perturbation directions that most reverse the trajectory.

        Generates ``n_directions`` random orthonormal directions plus the
        direction toward the source-stage centroid, then evaluates all via
        :meth:`compute_perturbation_effects`.

        Parameters
        ----------
        stage_centroid_src : Tensor
            Shape ``(input_dim,)`` — mean latent position of source-stage cells.
        n_directions : int
            Total number of directions to test (random + centroid direction).

        Returns
        -------
        PerturbationResult
            Sorted with most-reversing directions first.
        """
        D = x_src.shape[-1]

        # Generate random orthonormal basis via QR decomposition
        rand_mat = torch.randn(n_directions - 1, D)
        dirs_rand, _ = torch.linalg.qr(rand_mat.T)  # (D, min(n-1, D))
        dirs_rand = dirs_rand.T[:n_directions - 1]   # (n-1, D)

        # Add centroid-reversal direction
        centroid_dir = (stage_centroid_src.to(self.device) - x_src.mean(dim=0)).cpu()
        centroid_dir = centroid_dir / centroid_dir.norm().clamp_min(1e-8)
        directions = torch.cat([dirs_rand, centroid_dir.unsqueeze(0)], dim=0)  # (K, D)

        return self.compute_perturbation_effects(
            x_src=x_src,
            c_s=c_s,
            stage_pair_id=stage_pair_id,
            perturbation_directions=directions,
            effect_size=effect_size,
            num_steps=num_steps,
            wes_features=wes_features,
        )

    # ------------------------------------------------------------------
    # Direction constructors
    # ------------------------------------------------------------------

    @staticmethod
    def define_perturbation_directions_from_pca(
        x_stage: Tensor,
        n_components: int = 20,
    ) -> Tensor:
        """PCA-based perturbation directions from a cell population.

        The principal components capture the main axes of variation within the
        stage population.  Perturbing along these axes tests which aspects of
        within-stage variation affect downstream trajectories.

        Parameters
        ----------
        x_stage : Tensor
            Shape ``(N, D)`` — cells from the stage of interest.
        n_components : int
            Number of PCA components to return.

        Returns
        -------
        Tensor
            Shape ``(n_components, D)`` — unit-normalised PCA directions.
        """
        x_centered = x_stage - x_stage.mean(dim=0, keepdim=True)
        _, _, Vh = torch.linalg.svd(x_centered, full_matrices=False)
        return Vh[:n_components]  # (n_components, D) — already unit-normalised

    @staticmethod
    def define_perturbation_directions_stage_delta(
        x_src_stage: Tensor,
        x_tgt_stage: Tensor,
        n_directions: int = 20,
    ) -> Tensor:
        """Direction vectors derived from between-stage differences.

        Constructs a basis spanning:
        1. The mean stage transition direction (centroid delta).
        2. PCA components of within-pair differences.
        3. Random orthogonal complement if ``n_directions`` > available.

        Parameters
        ----------
        x_src_stage : Tensor
            Shape ``(N_src, D)`` — source-stage cells.
        x_tgt_stage : Tensor
            Shape ``(N_tgt, D)`` — target-stage cells.
        n_directions : int
            Number of directions to return.

        Returns
        -------
        Tensor
            Shape ``(n_directions, D)`` — unit-normalised directions.
        """
        D = x_src_stage.shape[-1]
        n_directions = min(n_directions, D)

        # 1. Mean transition direction
        mu_delta = x_tgt_stage.mean(dim=0) - x_src_stage.mean(dim=0)
        mu_delta = mu_delta / mu_delta.norm().clamp_min(1e-8)

        # 2. PCA of the combined difference space
        n_take = min(n_directions - 1, min(x_src_stage.shape[0], x_tgt_stage.shape[0]))
        if n_take > 0:
            # Sample matched pairs (min size)
            n_pairs = min(x_src_stage.shape[0], x_tgt_stage.shape[0], n_take * 4)
            idx_s = torch.randperm(x_src_stage.shape[0])[:n_pairs]
            idx_t = torch.randperm(x_tgt_stage.shape[0])[:n_pairs]
            diffs = x_tgt_stage[idx_t] - x_src_stage[idx_s]
            diffs_centered = diffs - diffs.mean(dim=0, keepdim=True)
            _, _, Vh = torch.linalg.svd(diffs_centered, full_matrices=False)
            pca_dirs = Vh[:n_take]  # (n_take, D)
        else:
            pca_dirs = torch.zeros(0, D, dtype=x_src_stage.dtype)

        directions = torch.cat([mu_delta.unsqueeze(0), pca_dirs], dim=0)  # (≤K, D)

        # Pad with random orthonormal vectors if needed
        if directions.shape[0] < n_directions:
            n_pad = n_directions - directions.shape[0]
            rand_mat = torch.randn(n_pad, D, dtype=x_src_stage.dtype)
            # Orthogonalise against existing directions via Gram-Schmidt
            for i in range(n_pad):
                v = rand_mat[i]
                for d in directions:
                    v = v - (v @ d) * d
                norm = v.norm()
                if norm > 1e-8:
                    rand_mat[i] = v / norm
                else:
                    rand_mat[i] = torch.randn(D, dtype=x_src_stage.dtype)
                    rand_mat[i] = rand_mat[i] / rand_mat[i].norm()
            directions = torch.cat([directions, rand_mat], dim=0)

        return directions[:n_directions]

    # ------------------------------------------------------------------
    # Full-batch perturbation matrix (for heatmap viz)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def perturbation_effect_matrix(
        self,
        x_src: Tensor,
        c_s: Tensor,
        stage_pair_id: Tensor,
        directions: Tensor,
        effect_sizes: list[float] | None = None,
        num_steps: int = 8,
        wes_features: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Compute effect scores across multiple effect sizes.

        Returns a dict with key ``"effect_matrix"`` of shape
        ``(K_directions, len(effect_sizes))`` and ``"reversal_matrix"``
        of the same shape.

        Useful for generating dose-response heatmaps.
        """
        if effect_sizes is None:
            effect_sizes = [0.25, 0.5, 1.0, 2.0, 4.0]

        effect_matrix = torch.zeros(directions.shape[0], len(effect_sizes))
        reversal_matrix = torch.zeros(directions.shape[0], len(effect_sizes))

        for j, es in enumerate(effect_sizes):
            result = self.compute_perturbation_effects(
                x_src=x_src, c_s=c_s, stage_pair_id=stage_pair_id,
                perturbation_directions=directions, effect_size=es,
                num_steps=num_steps, wes_features=wes_features,
            )
            effect_matrix[:, j] = result.effect_scores
            reversal_matrix[:, j] = result.reversal_scores

        return {
            "effect_matrix": effect_matrix,
            "reversal_matrix": reversal_matrix,
            "effect_sizes": torch.tensor(effect_sizes),
        }
