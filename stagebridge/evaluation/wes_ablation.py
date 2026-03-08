"""WES off/on gate for explicit regularization comparisons."""
from __future__ import annotations

from typing import Any

from stagebridge.evaluation.reporting import GateEvaluationResult


def evaluate_wes_ablation(
    *,
    run_label: str,
    edge: str,
    config_signature: str,
    wes_off_result: dict[str, Any] | None,
    wes_on_result: dict[str, Any] | None,
) -> GateEvaluationResult:
    if wes_off_result is None or wes_on_result is None:
        return GateEvaluationResult(
            gate_name="wes_ablation",
            run_label=run_label,
            mode="wes_compare",
            edge=edge,
            config_signature=config_signature,
            primary_metric=None,
            secondary_metrics={},
            status="inconclusive",
            interpretation="A matched WES off/on comparison is not yet available.",
            recommended_action="needs_more_data",
        )
    off_sink = float(wes_off_result["heldout_metrics"]["sinkhorn"])
    on_sink = float(wes_on_result["heldout_metrics"]["sinkhorn"])
    off_cal = float(wes_off_result["calibration"]["mean_abs_shift_error"])
    on_cal = float(wes_on_result["calibration"]["mean_abs_shift_error"])
    off_penalty = float(wes_off_result.get("wes_diagnostics", {}).get("regularizer_mean_penalty", 0.0))
    on_penalty = float(wes_on_result.get("wes_diagnostics", {}).get("regularizer_mean_penalty", 0.0))

    sinkhorn_gain = off_sink - on_sink
    calibration_gain = off_cal - on_cal
    if on_penalty <= 0.0:
        status = "inconclusive"
        action = "needs_more_data"
        interpretation = "The WES branch did not register a real regularization signal in this comparison, so it cannot be judged honestly yet."
    elif sinkhorn_gain > 0.1 and calibration_gain > 0.05:
        status = "pass"
        action = "keep"
        interpretation = "WES improved both the primary transition metric and calibration while entering the active computation, so it earns a place in the current story."
    elif sinkhorn_gain > 0.05 or calibration_gain > 0.05:
        status = "weak_pass"
        action = "keep_as_optional"
        interpretation = "WES changes the objective and helps at least one meaningful criterion, but the gain is still modest."
    else:
        status = "fail"
        action = "demote"
        interpretation = "WES is functionally connected, but it did not improve the matched comparison enough to remain a flagship pillar."
    return GateEvaluationResult(
        gate_name="wes_ablation",
        run_label=run_label,
        mode="wes_compare",
        edge=edge,
        config_signature=config_signature,
        primary_metric=on_sink,
        secondary_metrics={
            "wes_off_sinkhorn": off_sink,
            "wes_on_sinkhorn": on_sink,
            "wes_off_calibration_error": off_cal,
            "wes_on_calibration_error": on_cal,
            "wes_off_penalty": off_penalty,
            "wes_on_penalty": on_penalty,
        },
        status=status,
        interpretation=interpretation,
        recommended_action=action,
    )
