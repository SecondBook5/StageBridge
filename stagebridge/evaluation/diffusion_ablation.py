"""Diffusion ablation gate for fixed vs state-dependent diffusion."""
from __future__ import annotations

from typing import Any

from stagebridge.evaluation.reporting import GateEvaluationResult


def evaluate_diffusion_ablation(
    *,
    run_label: str,
    edge: str,
    config_signature: str,
    fixed_result: dict[str, Any] | None,
    state_dependent_result: dict[str, Any] | None,
) -> GateEvaluationResult:
    if fixed_result is None or state_dependent_result is None:
        return GateEvaluationResult(
            gate_name="diffusion_ablation",
            run_label=run_label,
            mode="diffusion_compare",
            edge=edge,
            config_signature=config_signature,
            primary_metric=None,
            secondary_metrics={},
            status="inconclusive",
            interpretation="Fixed and state-dependent diffusion are not both available yet for a fair comparison.",
            recommended_action="needs_more_data",
        )
    fixed_cal = float(fixed_result["calibration"]["mean_abs_shift_error"])
    state_cal = float(state_dependent_result["calibration"]["mean_abs_shift_error"])
    fixed_sink = float(fixed_result["heldout_metrics"]["sinkhorn"])
    state_sink = float(state_dependent_result["heldout_metrics"]["sinkhorn"])
    fixed_mean_diff = float(fixed_result["diffusion_diagnostics"]["mean_diffusion_scale"])
    state_mean_diff = float(state_dependent_result["diffusion_diagnostics"]["mean_diffusion_scale"])

    calibration_gain = fixed_cal - state_cal
    sinkhorn_gain = fixed_sink - state_sink
    if calibration_gain > 0.1 and sinkhorn_gain > 0.1:
        status = "pass"
        action = "keep"
        interpretation = "State-dependent diffusion improved both calibration and the primary transition metric, so it earns a place as an active candidate."
    elif calibration_gain > 0.05 or sinkhorn_gain > 0.1:
        status = "weak_pass"
        action = "keep_as_optional"
        interpretation = "State-dependent diffusion improved at least one meaningful criterion, but the gain should still be treated as provisional."
    else:
        status = "fail"
        action = "demote"
        interpretation = "State-dependent diffusion did not improve calibration or the primary transition metric enough to justify staying in the v1 story."
    return GateEvaluationResult(
        gate_name="diffusion_ablation",
        run_label=run_label,
        mode="diffusion_compare",
        edge=edge,
        config_signature=config_signature,
        primary_metric=state_cal,
        secondary_metrics={
            "fixed_calibration_error": fixed_cal,
            "state_dependent_calibration_error": state_cal,
            "fixed_sinkhorn": fixed_sink,
            "state_dependent_sinkhorn": state_sink,
            "fixed_mean_diffusion_scale": fixed_mean_diff,
            "state_dependent_mean_diffusion_scale": state_mean_diff,
        },
        status=status,
        interpretation=interpretation,
        recommended_action=action,
    )
