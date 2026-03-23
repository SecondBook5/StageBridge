"""Post-hoc calibration for confidence scores.

This module provides temperature scaling and other calibration methods
to improve the reliability of confidence estimates from reference mapping
and model predictions.

Temperature scaling learns a single scalar T such that:
    calibrated_confidence = sigmoid(logit(confidence) / T)

This is a simple but effective post-hoc calibration method that preserves
ranking while improving calibration (Guo et al., 2017).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import optimize
from scipy.special import expit, logit

from stagebridge.logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class CalibrationResult:
    """Result of calibration fitting."""

    method: str
    temperature: float
    calibrated_confidence: np.ndarray
    original_ece: float
    calibrated_ece: float
    n_samples: int
    n_bins: int


def expected_calibration_error(
    confidence: np.ndarray,
    accuracy: np.ndarray,
    *,
    n_bins: int = 15,
) -> float:
    """Compute Expected Calibration Error (ECE).

    ECE measures the difference between predicted confidence and actual accuracy
    across confidence bins.

    Parameters
    ----------
    confidence : np.ndarray
        Predicted confidence scores in [0, 1]
    accuracy : np.ndarray
        Binary accuracy (1 if correct, 0 if incorrect)
    n_bins : int
        Number of bins for calibration

    Returns
    -------
    float
        Expected calibration error (lower is better)
    """
    confidence = np.asarray(confidence).ravel()
    accuracy = np.asarray(accuracy).ravel()

    if len(confidence) != len(accuracy):
        raise ValueError("confidence and accuracy must have same length")

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        in_bin = (confidence > bin_boundaries[i]) & (confidence <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            avg_confidence = np.mean(confidence[in_bin])
            avg_accuracy = np.mean(accuracy[in_bin])
            ece += np.abs(avg_confidence - avg_accuracy) * prop_in_bin

    return float(ece)


def temperature_scale(
    confidence: np.ndarray,
    accuracy: np.ndarray,
    *,
    n_bins: int = 15,
    init_temp: float = 1.0,
) -> CalibrationResult:
    """Fit temperature scaling to calibrate confidence scores.

    Learns optimal temperature T that minimizes ECE:
        calibrated = sigmoid(logit(confidence) / T)

    Parameters
    ----------
    confidence : np.ndarray
        Predicted confidence scores in (0, 1)
    accuracy : np.ndarray
        Binary accuracy (1 if correct, 0 if incorrect)
    n_bins : int
        Number of bins for ECE computation
    init_temp : float
        Initial temperature for optimization

    Returns
    -------
    CalibrationResult
        Calibration result with optimal temperature and calibrated scores
    """
    confidence = np.asarray(confidence).ravel().astype(np.float64)
    accuracy = np.asarray(accuracy).ravel().astype(np.float64)

    # Clip to avoid numerical issues with logit
    eps = 1e-7
    confidence = np.clip(confidence, eps, 1 - eps)

    # Convert to logits
    logits = logit(confidence)

    # Original ECE
    original_ece = expected_calibration_error(confidence, accuracy, n_bins=n_bins)

    def objective(temp: float) -> float:
        """Objective: minimize ECE."""
        if temp <= 0:
            return 1e10
        scaled = expit(logits / temp)
        return expected_calibration_error(scaled, accuracy, n_bins=n_bins)

    # Optimize temperature
    result = optimize.minimize_scalar(
        objective,
        bounds=(0.1, 10.0),
        method="bounded",
        options={"xatol": 1e-4},
    )

    optimal_temp = float(result.x)
    calibrated = expit(logits / optimal_temp).astype(np.float32)
    calibrated_ece = expected_calibration_error(calibrated, accuracy, n_bins=n_bins)

    log.info(
        "Temperature scaling: T=%.3f, ECE %.4f -> %.4f (%.1f%% reduction)",
        optimal_temp,
        original_ece,
        calibrated_ece,
        100 * (original_ece - calibrated_ece) / (original_ece + 1e-8),
    )

    return CalibrationResult(
        method="temperature_scaling",
        temperature=optimal_temp,
        calibrated_confidence=calibrated,
        original_ece=original_ece,
        calibrated_ece=calibrated_ece,
        n_samples=len(confidence),
        n_bins=n_bins,
    )


def apply_temperature(
    confidence: np.ndarray,
    temperature: float,
) -> np.ndarray:
    """Apply learned temperature to new confidence scores.

    Parameters
    ----------
    confidence : np.ndarray
        Confidence scores in (0, 1)
    temperature : float
        Temperature learned from calibration

    Returns
    -------
    np.ndarray
        Calibrated confidence scores
    """
    confidence = np.asarray(confidence).astype(np.float64)
    eps = 1e-7
    confidence = np.clip(confidence, eps, 1 - eps)
    logits = logit(confidence)
    return expit(logits / temperature).astype(np.float32)


def calibrate_reference_confidence(
    hlca_confidence: np.ndarray,
    luca_confidence: np.ndarray,
    hlca_accuracy: np.ndarray | None = None,
    luca_accuracy: np.ndarray | None = None,
    *,
    method: Literal["temperature", "isotonic", "none"] = "temperature",
    n_bins: int = 15,
) -> dict[str, CalibrationResult | np.ndarray]:
    """Calibrate dual-reference confidence scores.

    If accuracy labels are not provided, returns uncalibrated scores with
    a warning. For proper calibration, accuracy should indicate whether
    the reference mapping was "correct" (e.g., matched expected cell type).

    Parameters
    ----------
    hlca_confidence : np.ndarray
        HLCA mapping confidence scores
    luca_confidence : np.ndarray
        LuCA mapping confidence scores
    hlca_accuracy : np.ndarray, optional
        Binary accuracy for HLCA (for fitting calibration)
    luca_accuracy : np.ndarray, optional
        Binary accuracy for LuCA (for fitting calibration)
    method : str
        Calibration method: "temperature", "isotonic", or "none"
    n_bins : int
        Number of bins for ECE computation

    Returns
    -------
    dict
        Contains calibrated scores and calibration results
    """
    result: dict[str, CalibrationResult | np.ndarray] = {}

    if method == "none":
        result["hlca_calibrated"] = hlca_confidence
        result["luca_calibrated"] = luca_confidence
        log.info("Calibration disabled, returning original confidence scores")
        return result

    if method == "temperature":
        if hlca_accuracy is not None:
            hlca_calib = temperature_scale(hlca_confidence, hlca_accuracy, n_bins=n_bins)
            result["hlca_calibration"] = hlca_calib
            result["hlca_calibrated"] = hlca_calib.calibrated_confidence
        else:
            log.warning("HLCA accuracy not provided, skipping calibration")
            result["hlca_calibrated"] = hlca_confidence

        if luca_accuracy is not None:
            luca_calib = temperature_scale(luca_confidence, luca_accuracy, n_bins=n_bins)
            result["luca_calibration"] = luca_calib
            result["luca_calibrated"] = luca_calib.calibrated_confidence
        else:
            log.warning("LuCA accuracy not provided, skipping calibration")
            result["luca_calibrated"] = luca_confidence

    elif method == "isotonic":
        log.warning("Isotonic calibration not yet implemented, using temperature scaling")
        return calibrate_reference_confidence(
            hlca_confidence,
            luca_confidence,
            hlca_accuracy,
            luca_accuracy,
            method="temperature",
            n_bins=n_bins,
        )
    else:
        raise ValueError(f"Unknown calibration method: {method}")

    return result
