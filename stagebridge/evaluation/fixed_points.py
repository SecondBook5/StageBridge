"""Fixed-point style diagnostics for edge-wise drift fields."""

from __future__ import annotations

import torch
from torch import Tensor


def summarize_fixed_points(
    model: object, x_state: Tensor, *, context: Tensor, edge_id: int
) -> dict[str, float]:
    edge_ids = torch.full(
        (x_state.shape[0],), int(edge_id), dtype=torch.long, device=x_state.device
    )
    t = torch.full((x_state.shape[0],), 0.95, dtype=x_state.dtype, device=x_state.device)
    with torch.no_grad():
        drift = model.forward_drift(x_t=x_state, t=t, context=context, edge_ids=edge_ids)
    norms = drift.norm(dim=1)
    return {
        "mean_terminal_drift_norm": float(norms.mean().item()),
        "min_terminal_drift_norm": float(norms.min().item()),
    }
