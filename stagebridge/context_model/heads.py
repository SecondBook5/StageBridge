"""Prediction heads for lesion-level EA-MIST models."""
from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn


@dataclass(slots=True, frozen=True)
class LesionTaskHeadOutput:
    """Shared multitask lesion-head output."""

    stage_logits: Tensor
    displacement: Tensor
    edge_logits: Tensor | None = None


class LesionMultitaskHeads(nn.Module):
    """Shared stage, weak displacement, and optional edge heads."""

    def __init__(
        self,
        model_dim: int,
        *,
        num_stage_classes: int = 5,
        num_edge_heads: int = 0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden_dim = int(model_dim)
        self.stage_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, int(num_stage_classes)),
        )
        self.displacement_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, 1),
        )
        self.edge_head = (
            None
            if int(num_edge_heads) <= 0
            else nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(hidden_dim, int(num_edge_heads)),
            )
        )

    def forward(self, lesion_embedding: Tensor) -> LesionTaskHeadOutput:
        """Return multitask lesion predictions."""
        return LesionTaskHeadOutput(
            stage_logits=self.stage_head(lesion_embedding),
            displacement=self.displacement_head(lesion_embedding).squeeze(-1),
            edge_logits=None if self.edge_head is None else self.edge_head(lesion_embedding),
        )
