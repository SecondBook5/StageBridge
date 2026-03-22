"""
Numerical stability checks for StageBridge.

Detects NaNs, infinities, exploding gradients, and other numerical issues.
"""

import logging
from typing import Any

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

log = logging.getLogger(__name__)


def check_tensor_health(
    tensor: "torch.Tensor | np.ndarray",
    name: str = "tensor",
) -> dict[str, Any]:
    """
    Check tensor for numerical issues.

    Parameters
    ----------
    tensor : Tensor or ndarray
        Tensor to check
    name : str
        Name for reporting

    Returns
    -------
    dict
        Health report
    """
    if HAS_TORCH and isinstance(tensor, torch.Tensor):
        arr = tensor.detach().cpu().numpy()
    else:
        arr = np.asarray(tensor)

    n_total = arr.size
    n_nan = np.isnan(arr).sum()
    n_inf = np.isinf(arr).sum()
    n_zero = (arr == 0).sum()

    report = {
        "name": name,
        "shape": arr.shape,
        "dtype": str(arr.dtype),
        "n_total": int(n_total),
        "n_nan": int(n_nan),
        "n_inf": int(n_inf),
        "n_zero": int(n_zero),
        "pct_nan": float(n_nan / n_total * 100) if n_total > 0 else 0,
        "pct_inf": float(n_inf / n_total * 100) if n_total > 0 else 0,
        "valid": n_nan == 0 and n_inf == 0,
    }

    if n_nan == 0 and n_inf == 0:
        report["min"] = float(arr.min())
        report["max"] = float(arr.max())
        report["mean"] = float(arr.mean())
        report["std"] = float(arr.std())

    return report


def check_gradient_health(model: "torch.nn.Module") -> dict[str, Any]:
    """
    Check model gradients for numerical issues.

    Parameters
    ----------
    model : nn.Module
        PyTorch model after backward pass

    Returns
    -------
    dict
        Gradient health report
    """
    if not HAS_TORCH:
        return {"valid": False, "issues": ["PyTorch not available"]}

    report = {
        "valid": True,
        "parameters": {},
        "issues": [],
    }

    for name, param in model.named_parameters():
        if param.grad is None:
            continue

        grad = param.grad
        grad.detach().cpu().numpy()

        param_report = {
            "shape": list(grad.shape),
            "has_nan": bool(torch.isnan(grad).any()),
            "has_inf": bool(torch.isinf(grad).any()),
            "norm": float(grad.norm().item()),
            "max_abs": float(grad.abs().max().item()),
        }

        report["parameters"][name] = param_report

        if param_report["has_nan"]:
            report["valid"] = False
            report["issues"].append(f"NaN gradient in {name}")

        if param_report["has_inf"]:
            report["valid"] = False
            report["issues"].append(f"Inf gradient in {name}")

        # Check for exploding gradients
        if param_report["norm"] > 1e6:
            report["issues"].append(f"Exploding gradient in {name}: norm={param_report['norm']:.2e}")

    return report


def check_loss_stability(
    losses: list[float],
    window: int = 10,
) -> dict[str, Any]:
    """
    Check loss trajectory for instability.

    Parameters
    ----------
    losses : list
        Loss values over training
    window : int
        Window size for moving average

    Returns
    -------
    dict
        Stability report
    """
    losses = np.array(losses)

    report = {
        "n_steps": len(losses),
        "valid": True,
        "issues": [],
    }

    # Check for NaN/Inf
    n_nan = np.isnan(losses).sum()
    n_inf = np.isinf(losses).sum()

    if n_nan > 0:
        report["valid"] = False
        report["issues"].append(f"Loss has {n_nan} NaN values")

    if n_inf > 0:
        report["valid"] = False
        report["issues"].append(f"Loss has {n_inf} Inf values")

    # Check for explosion (sudden large increase)
    if len(losses) > window:
        valid_losses = losses[~np.isnan(losses) & ~np.isinf(losses)]
        if len(valid_losses) > window:
            diffs = np.diff(valid_losses)
            max_jump = np.max(np.abs(diffs))
            mean_loss = np.mean(valid_losses)

            if max_jump > 10 * mean_loss:
                report["issues"].append(f"Loss explosion detected: max_jump={max_jump:.2e}")

            report["max_jump"] = float(max_jump)
            report["final_loss"] = float(valid_losses[-1])
            report["min_loss"] = float(valid_losses.min())

    return report


def check_embedding_quality(
    embeddings: np.ndarray,
    name: str = "embeddings",
) -> dict[str, Any]:
    """
    Check embedding quality (variance, collapse, etc.).

    Parameters
    ----------
    embeddings : ndarray
        Embedding matrix (n_samples, n_dims)
    name : str
        Name for reporting

    Returns
    -------
    dict
        Quality report
    """
    report = check_tensor_health(embeddings, name)

    if not report["valid"]:
        return report

    # Check for collapse (all embeddings nearly identical)
    per_dim_std = np.std(embeddings, axis=0)
    collapsed_dims = (per_dim_std < 1e-6).sum()

    report["collapsed_dims"] = int(collapsed_dims)
    report["pct_collapsed"] = float(collapsed_dims / embeddings.shape[1] * 100)

    if collapsed_dims > embeddings.shape[1] * 0.5:
        report["issues"] = report.get("issues", [])
        report["issues"].append(f"Embedding collapse: {collapsed_dims}/{embeddings.shape[1]} dims have near-zero variance")
        report["valid"] = False

    # Check for extreme values
    if np.abs(embeddings).max() > 1e6:
        report["issues"] = report.get("issues", [])
        report["issues"].append(f"Extreme embedding values: max_abs={np.abs(embeddings).max():.2e}")

    return report


def check_confidence_calibration(
    confidences: np.ndarray,
    name: str = "confidence",
) -> dict[str, Any]:
    """
    Check confidence scores for validity.

    Parameters
    ----------
    confidences : ndarray
        Confidence scores (should be in [0, 1])
    name : str
        Name for reporting

    Returns
    -------
    dict
        Calibration report
    """
    report = check_tensor_health(confidences, name)

    if not report["valid"]:
        return report

    # Check range
    out_of_range = ((confidences < 0) | (confidences > 1)).sum()
    report["out_of_range"] = int(out_of_range)

    if out_of_range > 0:
        report["valid"] = False
        report["issues"] = [f"{out_of_range} confidence values outside [0, 1]"]

    # Check for all-same (no discrimination)
    if np.std(confidences) < 1e-6:
        report["issues"] = report.get("issues", [])
        report["issues"].append("Confidence scores have no variance (no discrimination)")

    return report
