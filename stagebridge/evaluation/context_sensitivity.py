"""Context sensitivity analysis for StageBridge.

Quantifies how much the population context (source-stage cell composition)
matters for each stage-to-stage transition by comparing:

  - Prediction with REAL source context (the actual source population)
  - Prediction with SHUFFLED source context (randomly permuted source cells)

A large gap (real vs. shuffled Sinkhorn distance to target) means the model
uses population context to improve prediction accuracy.

Biological prediction (Peng et al. 2026, Cancer Cell):
  Context sensitivity should be HIGHEST at Normal→AAH and AAH→AIS transitions,
  where the IL1B-driven proinflammatory niche is most compositionally distinct
  (peak KAC + IL1B+ macrophage diversity).  Sensitivity should decay at MIA→LUAD
  as the inflammatory niche depletes and the TME becomes more homogeneous.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from stagebridge.logging_utils import get_logger
from stagebridge.data.luad_evo.stages import (
    normalize_stage_label,
    ordered_transitions,
)
from stagebridge.transition_model.losses import sinkhorn_distance

log = get_logger(__name__)


def compare_real_vs_shuffled_context(
    model: Any,
    x_src: torch.Tensor,
    x_tgt: torch.Tensor,
    *,
    real_context: torch.Tensor,
    shuffled_context: torch.Tensor,
    edge_id: int,
    num_steps: int = 8,
    stochastic: bool = False,
    ot_epsilon: float = 0.05,
    sinkhorn_iters: int = 80,
) -> dict[str, float]:
    """Measure how much edge predictions change when context is shuffled."""
    edge_ids = torch.full((x_src.shape[0],), int(edge_id), dtype=torch.long, device=x_src.device)
    with torch.no_grad():
        pred_real, _ = model.rollout(
            x_src,
            context=real_context,
            edge_ids=edge_ids,
            num_steps=int(num_steps),
            stochastic=bool(stochastic),
        )
        pred_shuffled, _ = model.rollout(
            x_src,
            context=shuffled_context,
            edge_ids=edge_ids,
            num_steps=int(num_steps),
            stochastic=bool(stochastic),
        )

    n = min(pred_real.shape[0], pred_shuffled.shape[0], x_tgt.shape[0])
    pred_real = pred_real[:n]
    pred_shuffled = pred_shuffled[:n]
    x_tgt = x_tgt[:n]
    sink_real = float(
        sinkhorn_distance(
            x_src=pred_real,
            x_tgt=x_tgt,
            epsilon=float(ot_epsilon),
            n_iters=int(sinkhorn_iters),
        ).item()
    )
    sink_shuffled = float(
        sinkhorn_distance(
            x_src=pred_shuffled,
            x_tgt=x_tgt,
            epsilon=float(ot_epsilon),
            n_iters=int(sinkhorn_iters),
        ).item()
    )
    return {
        "sinkhorn_real_context": sink_real,
        "sinkhorn_shuffled_context": sink_shuffled,
        "context_sensitivity_delta": sink_shuffled - sink_real,
    }


def compute_context_sensitivity(
    model: Any,
    adata: Any,
    stage_pairs: list[tuple[str, str]] | None = None,
    n_subsets: int = 8,
    subset_size: int = 128,
    latent_key: str = "X_hlca",
    stage_col: str = "stage",
    device: str = "cpu",
    ot_epsilon: float = 0.05,
    sinkhorn_iters: int = 20,
    seed: int = 42,
) -> dict[str, float]:
    """Compute context sensitivity score for each stage-to-stage transition.

    For each transition (src→tgt) and each random subset of source cells,
    computes:
        sensitivity = mean_Sinkhorn(shuffled_context) - mean_Sinkhorn(real_context)

    A positive sensitivity score means the model produces *better* (lower
    Sinkhorn distance) predictions when given the real source population context
    vs. a randomly permuted context.  Higher = more context-dependent.

    Parameters
    ----------
    model : StageBridgeModel
        Trained model with ``forward_set_context`` and ``integrate_euler`` methods.
    adata : AnnData
        Must have obsm[latent_key] and obs[stage_col].
    stage_pairs : list of (src, tgt) or None
        Which transitions to evaluate.  Defaults to all ordered transitions.
    n_subsets : int
        Number of random source-cell subsets to average over per transition.
    subset_size : int
        Number of source cells per subset.
    latent_key : str
        adata.obsm key for latent coordinates (default 'X_hlca').
    stage_col : str
        obs column for stage labels.
    device : str
        Torch device.
    ot_epsilon : float
        Sinkhorn regularization for distance computation.
    sinkhorn_iters : int
        Sinkhorn iterations for distance computation.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    dict mapping "src→tgt" strings to sensitivity scores (float).
    Higher = context matters more for that transition.
    """
    if latent_key not in adata.obsm:
        raise KeyError(
            f"latent_key '{latent_key}' not in adata.obsm. "
            f"Available: {list(adata.obsm.keys())}"
        )

    if stage_pairs is None:
        stage_pairs = ordered_transitions()

    X = np.asarray(adata.obsm[latent_key], dtype=np.float32)
    stages = adata.obs[stage_col].astype(str).values

    rng = np.random.default_rng(seed)
    model = model.to(device)
    model.eval()

    results: dict[str, float] = {}

    for stage_src, stage_tgt in stage_pairs:
        key = f"{stage_src}→{stage_tgt}"

        src_mask = stages == normalize_stage_label(stage_src)
        tgt_mask = stages == normalize_stage_label(stage_tgt)

        if src_mask.sum() < 2 or tgt_mask.sum() < 2:
            log.warning("Skipping %s: insufficient cells (src=%d, tgt=%d)",
                        key, src_mask.sum(), tgt_mask.sum())
            results[key] = float("nan")
            continue

        src_idx = np.where(src_mask)[0]
        tgt_idx = np.where(tgt_mask)[0]

        # Stage pair id for model conditioning
        from stagebridge.data.luad_evo.stages import infer_stage_pair_id
        pair_id = torch.tensor([infer_stage_pair_id(stage_src, stage_tgt)], dtype=torch.long, device=device)

        real_sinks: list[float] = []
        shuffled_sinks: list[float] = []

        with torch.no_grad():
            for _ in range(n_subsets):
                # Sample source subset
                n_src = min(subset_size, len(src_idx))
                chosen_src = rng.choice(src_idx, size=n_src, replace=False)
                x_src = torch.tensor(X[chosen_src], dtype=torch.float32, device=device)

                # Sample target subset for evaluation
                n_tgt = min(subset_size, len(tgt_idx))
                chosen_tgt = rng.choice(tgt_idx, size=n_tgt, replace=False)
                x_tgt = torch.tensor(X[chosen_tgt], dtype=torch.float32, device=device)

                # --- Real context ---
                src_set = x_src.unsqueeze(0)  # (1, n_src, d)
                c_s_real = model.forward_set_context(src_set)  # (1, context_dim)

                # Integrate source cells to target using real context
                # model._broadcast_condition handles (1,D)→(n_src,D) internally
                x_pred_real = model.integrate_euler(x_src, c_s_real, pair_id, num_steps=10)
                d_real = sinkhorn_distance(
                    x_pred_real, x_tgt, epsilon=ot_epsilon, n_iter=sinkhorn_iters,
                ).item()

                # --- Shuffled context ---
                # Randomly permute which cells form the source set
                shuffle_idx = rng.choice(src_idx, size=n_src, replace=False)
                x_shuffled = torch.tensor(X[shuffle_idx], dtype=torch.float32, device=device)
                src_set_shuf = x_shuffled.unsqueeze(0)
                c_s_shuf = model.forward_set_context(src_set_shuf)  # (1, context_dim)

                x_pred_shuf = model.integrate_euler(x_src, c_s_shuf, pair_id, num_steps=10)
                d_shuffled = sinkhorn_distance(
                    x_pred_shuf, x_tgt, epsilon=ot_epsilon, n_iter=sinkhorn_iters,
                ).item()

                real_sinks.append(d_real)
                shuffled_sinks.append(d_shuffled)

        # Sensitivity = how much WORSE predictions are when context is shuffled
        # Positive = real context gives lower (better) Sinkhorn than shuffled
        sensitivity = float(np.mean(shuffled_sinks) - np.mean(real_sinks))
        results[key] = sensitivity
        log.info(
            "%s: real_sink=%.4f  shuffled_sink=%.4f  sensitivity=%.4f",
            key, np.mean(real_sinks), np.mean(shuffled_sinks), sensitivity,
        )

    return results
