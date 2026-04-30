"""Token ablation analysis for StageBridge.

Adapted from AMICI's ablation module for ring-based niche structure.
Instead of ablating individual neighbor cells, we ablate:
- Spatial ring tokens (ring1, ring2, ring3, ring4)
- Reference tokens (HLCA, LuCA)
- Auxiliary tokens (pathway, stats)

This measures the contribution of each token type to receiver reconstruction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import torch
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

from stagebridge.contracts import TOKEN_NAMES, N_TOKENS

if TYPE_CHECKING:
    from stagebridge.models import StageBridge
    from stagebridge.loaders.dataset import NicheBatch


@dataclass
class AblationResult:
    """Result from token ablation analysis.

    Attributes:
        token_name: Name of ablated token
        baseline_loss: Reconstruction loss without ablation
        ablated_loss: Reconstruction loss with token ablated
        delta_loss: Change in loss (ablated - baseline)
        relative_importance: delta_loss / baseline_loss
        p_value: Statistical significance (paired test)
        n_samples: Number of samples analyzed
    """
    token_name: str
    baseline_loss: float
    ablated_loss: float
    delta_loss: float
    relative_importance: float
    p_value: float
    n_samples: int


@dataclass
class AblationModule:
    """Token ablation analysis for interpretability.

    Measures contribution of each token type by computing reconstruction
    loss with and without each token. Tokens with higher delta_loss are
    more important for the receiver's representation.

    Attributes:
        results: Dict mapping token_name -> AblationResult
        per_sample_losses: DataFrame with per-sample ablation losses
        stage_breakdown: Optional per-stage analysis
    """
    results: dict[str, AblationResult] = field(default_factory=dict)
    per_sample_losses: pd.DataFrame | None = None
    stage_breakdown: pd.DataFrame | None = None

    @classmethod
    def compute(
        cls,
        model: "StageBridge",
        dataloader: torch.utils.data.DataLoader,
        device: str | torch.device = "cpu",
        tokens_to_ablate: list[str] | None = None,
        compute_per_stage: bool = True,
        progress_bar: bool = True,
    ) -> "AblationModule":
        """Compute token ablation analysis.

        Args:
            model: Trained StageBridge model
            dataloader: DataLoader yielding NicheBatch
            device: Compute device
            tokens_to_ablate: Token names to ablate (default: all non-receiver)
            compute_per_stage: Break down by disease stage
            progress_bar: Show progress

        Returns:
            AblationModule with ablation results
        """
        model.eval()
        model.to(device)

        if tokens_to_ablate is None:
            tokens_to_ablate = TOKEN_NAMES[1:]  # All except receiver

        all_losses = []

        with torch.no_grad():
            for batch in tqdm(dataloader, disable=not progress_bar, desc="Ablation"):
                batch = batch.to(device)
                B = len(batch)

                baseline_output = model.niche_encoder(
                    receiver=batch.receiver,
                    neighbors=batch.neighbors,
                    distances=batch.distances,
                    neighbor_mask=batch.neighbor_mask,
                    token_type_ids=batch.token_type_ids,
                    return_reconstruction=True,
                )

                if baseline_output.receiver_reconstruction is None:
                    continue

                baseline_loss = (
                    (baseline_output.receiver_reconstruction - batch.receiver) ** 2
                ).mean(dim=-1).cpu().numpy()

                batch_results = {
                    "cell_id": batch.cell_ids,
                    "donor_id": batch.donor_ids,
                    "stage_idx": batch.stage_idx.cpu().numpy(),
                    "baseline_loss": baseline_loss,
                }

                for token_idx, token_name in enumerate(tokens_to_ablate):
                    actual_idx = TOKEN_NAMES.index(token_name) - 1
                    if actual_idx < 0 or actual_idx >= batch.neighbors.shape[1]:
                        continue

                    ablated_mask = batch.neighbor_mask.clone()
                    ablated_mask[:, actual_idx] = False

                    ablated_output = model.niche_encoder(
                        receiver=batch.receiver,
                        neighbors=batch.neighbors,
                        distances=batch.distances,
                        neighbor_mask=ablated_mask,
                        token_type_ids=batch.token_type_ids,
                        return_reconstruction=True,
                    )

                    if ablated_output.receiver_reconstruction is None:
                        continue

                    ablated_loss = (
                        (ablated_output.receiver_reconstruction - batch.receiver) ** 2
                    ).mean(dim=-1).cpu().numpy()

                    batch_results[f"{token_name}_loss"] = ablated_loss
                    batch_results[f"{token_name}_delta"] = ablated_loss - baseline_loss

                all_losses.append(pd.DataFrame(batch_results))

        if not all_losses:
            return cls()

        per_sample_losses = pd.concat(all_losses, ignore_index=True)

        results = {}
        for token_name in tokens_to_ablate:
            loss_col = f"{token_name}_loss"
            delta_col = f"{token_name}_delta"

            if loss_col not in per_sample_losses.columns:
                continue

            baseline = per_sample_losses["baseline_loss"].values
            ablated = per_sample_losses[loss_col].values
            delta = per_sample_losses[delta_col].values

            _, p_value = mannwhitneyu(ablated, baseline, alternative="greater")

            results[token_name] = AblationResult(
                token_name=token_name,
                baseline_loss=float(baseline.mean()),
                ablated_loss=float(ablated.mean()),
                delta_loss=float(delta.mean()),
                relative_importance=float(delta.mean() / (baseline.mean() + 1e-8)),
                p_value=float(p_value),
                n_samples=len(baseline),
            )

        stage_breakdown = None
        if compute_per_stage:
            stage_groups = []
            for stage_idx in per_sample_losses["stage_idx"].unique():
                stage_data = per_sample_losses[per_sample_losses["stage_idx"] == stage_idx]
                row = {"stage_idx": stage_idx, "n_samples": len(stage_data)}
                row["baseline_loss"] = stage_data["baseline_loss"].mean()

                for token_name in tokens_to_ablate:
                    delta_col = f"{token_name}_delta"
                    if delta_col in stage_data.columns:
                        row[f"{token_name}_importance"] = stage_data[delta_col].mean()

                stage_groups.append(row)

            stage_breakdown = pd.DataFrame(stage_groups)

        return cls(
            results=results,
            per_sample_losses=per_sample_losses,
            stage_breakdown=stage_breakdown,
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame."""
        rows = []
        for name, result in self.results.items():
            rows.append({
                "token": result.token_name,
                "baseline_loss": result.baseline_loss,
                "ablated_loss": result.ablated_loss,
                "delta_loss": result.delta_loss,
                "relative_importance": result.relative_importance,
                "p_value": result.p_value,
                "n_samples": result.n_samples,
            })
        df = pd.DataFrame(rows)
        if len(df) > 0:
            _, adj_p, _, _ = multipletests(df["p_value"].values, method="fdr_bh")
            df["p_adj"] = adj_p
        return df

    def get_token_ranking(self) -> list[str]:
        """Get tokens ranked by importance (highest first)."""
        df = self.to_dataframe()
        if len(df) == 0:
            return []
        return df.sort_values("relative_importance", ascending=False)["token"].tolist()


def compute_token_ablation(
    model: "StageBridge",
    dataloader: torch.utils.data.DataLoader,
    device: str = "cpu",
) -> pd.DataFrame:
    """Convenience function for quick ablation analysis.

    Returns DataFrame with token importance scores.
    """
    module = AblationModule.compute(model, dataloader, device)
    return module.to_dataframe()
