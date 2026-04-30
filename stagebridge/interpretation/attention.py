"""Attention pattern extraction and analysis for StageBridge.

Adapted from AMICI's attention module for ring-based niche structure.
Key differences from AMICI:
- Tokens are aggregated rings, not individual cells
- Distance is encoded in ring structure (ring1=closest, ring4=farthest)
- Empty attention indicates "no informative neighbor at this range"
- Stage context matters for attention interpretation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from stagebridge.contracts import TOKEN_NAMES, STAGES, IDX_TO_STAGE

if TYPE_CHECKING:
    from stagebridge.models import StageBridge


@dataclass
class AttentionModule:
    """Attention pattern analysis for StageBridge.

    Stores and analyzes attention patterns across the dataset.
    Key interpretations:
    - Ring attention decay: Does attention decrease with ring number?
    - Reference balance: HLCA vs LuCA attention ratio
    - Stage-specific patterns: How attention changes across stages
    - Empty attention: When does model "opt out" of neighbor info?

    Attributes:
        attention_df: Per-cell attention patterns
        empty_attention_df: Per-cell empty token attention
        summary_stats: Aggregated statistics
    """
    attention_df: pd.DataFrame | None = None
    empty_attention_df: pd.DataFrame | None = None
    summary_stats: dict = field(default_factory=dict)

    @classmethod
    def compute(
        cls,
        model: "StageBridge",
        dataloader: torch.utils.data.DataLoader,
        device: str | torch.device = "cpu",
        progress_bar: bool = True,
    ) -> "AttentionModule":
        """Extract attention patterns from trained model.

        Args:
            model: Trained StageBridge model
            dataloader: DataLoader yielding NicheBatch
            device: Compute device
            progress_bar: Show progress

        Returns:
            AttentionModule with attention patterns
        """
        model.eval()
        model.to(device)

        all_attention = []
        all_empty = []

        with torch.no_grad():
            for batch in tqdm(dataloader, disable=not progress_bar, desc="Attention"):
                batch = batch.to(device)

                output = model.niche_encoder(
                    receiver=batch.receiver,
                    neighbors=batch.neighbors,
                    distances=batch.distances,
                    neighbor_mask=batch.neighbor_mask,
                    token_type_ids=batch.token_type_ids,
                )

                if output.attention_weights is None:
                    continue

                attn = output.attention_weights.cpu().numpy()
                empty = output.empty_attention.cpu().numpy() if output.empty_attention is not None else np.zeros(len(batch))

                for i in range(len(batch)):
                    row = {
                        "cell_id": batch.cell_ids[i],
                        "donor_id": batch.donor_ids[i],
                        "stage_idx": int(batch.stage_idx[i].cpu()),
                    }

                    for j, token_name in enumerate(TOKEN_NAMES[1:]):  # Skip receiver
                        if j < attn.shape[1]:
                            row[f"attn_{token_name}"] = attn[i, j]

                    all_attention.append(row)

                    all_empty.append({
                        "cell_id": batch.cell_ids[i],
                        "donor_id": batch.donor_ids[i],
                        "stage_idx": int(batch.stage_idx[i].cpu()),
                        "empty_attention": empty[i],
                    })

        attention_df = pd.DataFrame(all_attention) if all_attention else None
        empty_attention_df = pd.DataFrame(all_empty) if all_empty else None

        summary_stats = {}
        if attention_df is not None:
            attn_cols = [c for c in attention_df.columns if c.startswith("attn_")]

            summary_stats["mean_attention"] = {
                col.replace("attn_", ""): attention_df[col].mean()
                for col in attn_cols
            }

            ring_cols = [c for c in attn_cols if "ring" in c]
            if ring_cols:
                ring_means = [attention_df[c].mean() for c in sorted(ring_cols)]
                summary_stats["ring_decay"] = ring_means
                if len(ring_means) > 1:
                    summary_stats["ring_decay_ratio"] = ring_means[0] / (ring_means[-1] + 1e-8)

            if "attn_hlca" in attn_cols and "attn_luca" in attn_cols:
                hlca_mean = attention_df["attn_hlca"].mean()
                luca_mean = attention_df["attn_luca"].mean()
                summary_stats["hlca_luca_ratio"] = hlca_mean / (luca_mean + 1e-8)

            for stage_idx in attention_df["stage_idx"].unique():
                stage_data = attention_df[attention_df["stage_idx"] == stage_idx]
                stage_name = IDX_TO_STAGE.get(stage_idx, f"stage_{stage_idx}")
                summary_stats[f"stage_{stage_name}_mean"] = {
                    col.replace("attn_", ""): stage_data[col].mean()
                    for col in attn_cols
                }

        if empty_attention_df is not None:
            summary_stats["mean_empty_attention"] = empty_attention_df["empty_attention"].mean()
            summary_stats["empty_attention_by_stage"] = (
                empty_attention_df.groupby("stage_idx")["empty_attention"].mean().to_dict()
            )

        return cls(
            attention_df=attention_df,
            empty_attention_df=empty_attention_df,
            summary_stats=summary_stats,
        )

    def get_ring_decay_profile(self) -> pd.DataFrame:
        """Get attention decay across spatial rings."""
        if self.attention_df is None:
            return pd.DataFrame()

        ring_cols = [c for c in self.attention_df.columns if c.startswith("attn_ring")]
        if not ring_cols:
            return pd.DataFrame()

        return pd.DataFrame({
            "ring": [c.replace("attn_", "") for c in sorted(ring_cols)],
            "mean_attention": [self.attention_df[c].mean() for c in sorted(ring_cols)],
            "std_attention": [self.attention_df[c].std() for c in sorted(ring_cols)],
        })

    def get_stage_comparison(self) -> pd.DataFrame:
        """Compare attention patterns across disease stages."""
        if self.attention_df is None:
            return pd.DataFrame()

        attn_cols = [c for c in self.attention_df.columns if c.startswith("attn_")]

        rows = []
        for stage_idx in sorted(self.attention_df["stage_idx"].unique()):
            stage_data = self.attention_df[self.attention_df["stage_idx"] == stage_idx]
            stage_name = IDX_TO_STAGE.get(stage_idx, f"stage_{stage_idx}") if stage_idx < len(STAGE_NAMES) else f"stage_{stage_idx}"

            row = {"stage": stage_name, "stage_idx": stage_idx, "n_cells": len(stage_data)}
            for col in attn_cols:
                row[col.replace("attn_", "")] = stage_data[col].mean()

            rows.append(row)

        return pd.DataFrame(rows)

    def get_reference_balance(self) -> pd.DataFrame:
        """Get HLCA vs LuCA attention balance per stage."""
        if self.attention_df is None:
            return pd.DataFrame()

        if "attn_hlca" not in self.attention_df.columns or "attn_luca" not in self.attention_df.columns:
            return pd.DataFrame()

        rows = []
        for stage_idx in sorted(self.attention_df["stage_idx"].unique()):
            stage_data = self.attention_df[self.attention_df["stage_idx"] == stage_idx]
            stage_name = IDX_TO_STAGE.get(stage_idx, f"stage_{stage_idx}") if stage_idx < len(STAGE_NAMES) else f"stage_{stage_idx}"

            hlca = stage_data["attn_hlca"].mean()
            luca = stage_data["attn_luca"].mean()

            rows.append({
                "stage": stage_name,
                "hlca_attention": hlca,
                "luca_attention": luca,
                "hlca_luca_ratio": hlca / (luca + 1e-8),
                "reference_preference": "HLCA" if hlca > luca else "LuCA",
            })

        return pd.DataFrame(rows)


def extract_attention_patterns(
    model: "StageBridge",
    dataloader: torch.utils.data.DataLoader,
    device: str = "cpu",
) -> pd.DataFrame:
    """Convenience function to extract attention as DataFrame."""
    module = AttentionModule.compute(model, dataloader, device)
    return module.attention_df
