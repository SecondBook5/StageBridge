"""Loss functions for EA-MIST pretraining and lesion supervision."""
from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def weighted_binary_classification_loss(
    logits: Tensor,
    labels: Tensor,
    *,
    weights: Tensor | None = None,
    loss_name: str = "weighted_bce",
    focal_gamma: float = 2.0,
    label_smoothing: float = 0.0,
) -> Tensor:
    """Compute the configured lesion-level binary classification loss."""
    if logits.ndim != 1 or labels.ndim != 1:
        raise ValueError("weighted_binary_classification_loss expects 1D logits and labels.")
    if logits.shape[0] != labels.shape[0]:
        raise ValueError("Logits and labels must have the same length.")
    target = labels.to(dtype=logits.dtype)
    if label_smoothing > 0.0:
        target = target * (1.0 - float(label_smoothing)) + 0.5 * float(label_smoothing)
    base = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    if loss_name == "focal":
        probs = torch.sigmoid(logits)
        p_t = target * probs + (1.0 - target) * (1.0 - probs)
        base = base * (1.0 - p_t).pow(float(focal_gamma))
    elif loss_name != "weighted_bce":
        raise ValueError(f"Unsupported lesion loss '{loss_name}'.")
    if weights is not None:
        base = base * weights.to(dtype=base.dtype)
        denom = weights.sum().clamp_min(1e-6)
        return base.sum() / denom
    return base.mean()


def masked_feature_reconstruction_loss(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """Compute masked feature reconstruction loss for local SSL pretraining."""
    if prediction.shape != target.shape or prediction.shape != mask.shape:
        raise ValueError("Prediction, target, and mask must share the same shape for reconstruction loss.")
    squared = (prediction - target).pow(2) * mask
    denom = mask.sum().clamp_min(1.0)
    return squared.sum() / denom


def shuffled_neighborhood_discrimination_loss(logits: Tensor, labels: Tensor) -> Tensor:
    """Binary discrimination loss for real-vs-shuffled local neighborhoods."""
    if logits.shape != labels.shape:
        raise ValueError("Shuffled-neighborhood logits and labels must share the same shape.")
    return F.binary_cross_entropy_with_logits(logits, labels.to(dtype=logits.dtype))


def lesion_subsampling_consistency_loss(
    logits_a: Tensor,
    logits_b: Tensor,
    *,
    reduction: str = "mean",
) -> Tensor:
    """Encourage stable lesion predictions under repeated bag subsampling."""
    loss = (torch.sigmoid(logits_a) - torch.sigmoid(logits_b)).pow(2)
    if reduction == "sum":
        return loss.sum()
    if reduction != "mean":
        raise ValueError(f"Unsupported consistency reduction '{reduction}'.")
    return loss.mean()
