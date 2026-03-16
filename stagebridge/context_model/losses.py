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
        raise ValueError(
            "Prediction, target, and mask must share the same shape for reconstruction loss."
        )
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


def class_weighted_stage_loss(
    logits: Tensor,
    labels: Tensor,
    *,
    class_weights: Tensor | None = None,
) -> Tensor:
    """Compute class-weighted multiclass stage loss."""
    if logits.ndim != 2:
        raise ValueError(
            f"class_weighted_stage_loss expects 2D logits, got shape={tuple(logits.shape)}"
        )
    if labels.ndim != 1:
        raise ValueError(
            f"class_weighted_stage_loss expects 1D labels, got shape={tuple(labels.shape)}"
        )
    if logits.shape[0] != labels.shape[0]:
        raise ValueError("Stage logits and labels must share the same batch length.")
    valid_mask = labels >= 0
    if not torch.any(valid_mask):
        return torch.zeros((), dtype=logits.dtype, device=logits.device)
    valid_logits = logits[valid_mask]
    valid_labels = labels[valid_mask].to(dtype=torch.long)
    weight = (
        None
        if class_weights is None
        else class_weights.to(device=logits.device, dtype=logits.dtype)
    )
    return F.cross_entropy(valid_logits, valid_labels, weight=weight)


def displacement_regression_loss(prediction: Tensor, target: Tensor) -> Tensor:
    """Compute SmoothL1 loss for weak stage-ordered displacement supervision."""
    if prediction.ndim != 1 or target.ndim != 1:
        raise ValueError("displacement_regression_loss expects 1D prediction and target tensors.")
    if prediction.shape[0] != target.shape[0]:
        raise ValueError("Displacement prediction and target must share the same batch length.")
    valid_mask = torch.isfinite(target)
    if not torch.any(valid_mask):
        return torch.zeros((), dtype=prediction.dtype, device=prediction.device)
    return F.smooth_l1_loss(prediction[valid_mask], target[valid_mask].to(dtype=prediction.dtype))


def ordinal_stage_loss(
    logits: Tensor,
    labels: Tensor,
    *,
    num_classes: int = 5,
) -> Tensor:
    """Compute ordinal EMD loss on cumulative stage distributions.

    Computes the Earth Mover's Distance between the predicted probability
    distribution (softmax of logits) and the one-hot target by comparing
    their cumulative distributions.  This penalises predictions that are
    ordinally far from the true class more than those that are close.
    """
    if logits.ndim != 2:
        raise ValueError(f"ordinal_stage_loss expects 2D logits, got shape={tuple(logits.shape)}")
    if labels.ndim != 1:
        raise ValueError(f"ordinal_stage_loss expects 1D labels, got shape={tuple(labels.shape)}")
    if logits.shape[0] != labels.shape[0]:
        raise ValueError("Stage logits and labels must share the same batch length.")
    valid_mask = labels >= 0
    if not torch.any(valid_mask):
        return torch.zeros((), dtype=logits.dtype, device=logits.device)
    valid_logits = logits[valid_mask]
    valid_labels = labels[valid_mask].to(dtype=torch.long)
    pred_cdf = torch.cumsum(F.softmax(valid_logits, dim=-1), dim=-1)
    target_onehot = F.one_hot(valid_labels, num_classes=int(num_classes)).to(dtype=pred_cdf.dtype)
    target_cdf = torch.cumsum(target_onehot, dim=-1)
    return (pred_cdf - target_cdf).abs().mean()


def transition_consistency_loss(
    displacement_pred: Tensor,
    niche_transition_scores: Tensor,
    mask: Tensor,
) -> Tensor:
    """Soft monotonic regularizer: SmoothL1 between predicted displacement and mean niche transition score.

    This encourages the lesion-level displacement prediction to be
    consistent with the average per-niche transition signal. The niche
    scores are detached so gradients only flow into the displacement head.
    """
    if displacement_pred.ndim != 1:
        raise ValueError(
            f"displacement_pred must be 1D, got shape={tuple(displacement_pred.shape)}"
        )
    if niche_transition_scores.ndim != 2 or mask.ndim != 2:
        raise ValueError("niche_transition_scores and mask must be 2D (B, N).")
    valid_scores = niche_transition_scores.masked_fill(~mask, float("nan"))
    mean_score = torch.nanmean(valid_scores, dim=-1)
    finite = torch.isfinite(mean_score) & torch.isfinite(displacement_pred)
    if not torch.any(finite):
        return torch.zeros((), dtype=displacement_pred.dtype, device=displacement_pred.device)
    return F.smooth_l1_loss(displacement_pred[finite], mean_score[finite].detach())


def masked_edge_loss(
    logits: Tensor | None,
    targets: Tensor | None,
    mask: Tensor | None,
) -> Tensor:
    """Compute masked BCE over optional auxiliary edge heads."""
    if logits is None or targets is None or mask is None:
        if logits is not None or targets is not None or mask is not None:
            raise ValueError(
                "masked_edge_loss expects logits, targets, and mask to be provided together."
            )
        return torch.zeros((), dtype=torch.float32)
    if logits.shape != targets.shape or logits.shape != mask.shape:
        raise ValueError(
            "masked_edge_loss expects logits, targets, and mask to share the same shape; "
            f"got logits={tuple(logits.shape)}, targets={tuple(targets.shape)}, mask={tuple(mask.shape)}."
        )
    valid_mask = mask.to(dtype=torch.bool)
    if not torch.any(valid_mask):
        return torch.zeros((), dtype=logits.dtype, device=logits.device)
    losses = F.binary_cross_entropy_with_logits(
        logits[valid_mask], targets[valid_mask].to(dtype=logits.dtype), reduction="none"
    )
    return losses.mean()
