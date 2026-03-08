"""Context-ablation gate for pooled, deep-sets, set-only, and graph-of-sets modes."""
from __future__ import annotations

from typing import Any

from stagebridge.evaluation.reporting import GateEvaluationResult


def _score(payload: dict[str, Any]) -> tuple[float, float]:
    sinkhorn = float(payload["heldout_metrics"]["sinkhorn"])
    calibration = float(payload["calibration"]["mean_abs_shift_error"])
    return sinkhorn, calibration


def evaluate_context_ablation(
    *,
    run_label: str,
    edge: str,
    config_signature: str,
    mode_results: dict[str, dict[str, Any]],
) -> GateEvaluationResult:
    required = {"pooled", "deep_sets", "set_only", "graph_of_sets"}
    if any(mode not in mode_results for mode in required):
        return GateEvaluationResult(
            gate_name="context_ablation",
            run_label=run_label,
            mode="context_compare",
            edge=edge,
            config_signature=config_signature,
            primary_metric=None,
            secondary_metrics={"available_modes": sorted(mode_results)},
            status="inconclusive",
            interpretation="Not all required context modes are available for comparison.",
            recommended_action="needs_more_data",
        )

    pooled_sink, pooled_cal = _score(mode_results["pooled"])
    deep_sink, deep_cal = _score(mode_results["deep_sets"])
    set_sink, set_cal = _score(mode_results["set_only"])
    graph_sink, graph_cal = _score(mode_results["graph_of_sets"])

    set_beats_pooled = ((pooled_sink - set_sink) > 0.1) or ((pooled_cal - set_cal) > 0.05)
    set_beats_deep = ((deep_sink - set_sink) > 0.1) or ((deep_cal - set_cal) > 0.05)
    graph_beats_set = ((set_sink - graph_sink) > 0.1) or ((set_cal - graph_cal) > 0.05)

    if set_beats_pooled and set_beats_deep and graph_beats_set:
        status = "pass"
        action = "keep"
        interpretation = "Set-only beat both pooled and Deep Sets, and graph propagation also beat set-only on at least one meaningful criterion."
    elif set_beats_pooled and set_beats_deep and not graph_beats_set:
        status = "weak_pass"
        action = "keep_as_optional"
        interpretation = "Set-only beat pooled and Deep Sets, but graph-of-sets did not clearly beat set-only and should remain optional."
    elif not set_beats_pooled or not set_beats_deep:
        status = "fail"
        action = "demote"
        interpretation = "Set-only did not beat both simpler permutation-invariant baselines, so the typed set encoder has not yet earned flagship status."
    else:
        status = "inconclusive"
        action = "needs_more_data"
        interpretation = "The context comparison is mixed and not yet stable enough for a clean decision."

    return GateEvaluationResult(
        gate_name="context_ablation",
        run_label=run_label,
        mode="context_compare",
        edge=edge,
        config_signature=config_signature,
        primary_metric=min(pooled_sink, deep_sink, set_sink, graph_sink),
        secondary_metrics={
            "pooled_sinkhorn": pooled_sink,
            "deep_sets_sinkhorn": deep_sink,
            "set_only_sinkhorn": set_sink,
            "graph_of_sets_sinkhorn": graph_sink,
            "pooled_calibration": pooled_cal,
            "deep_sets_calibration": deep_cal,
            "set_only_calibration": set_cal,
            "graph_of_sets_calibration": graph_cal,
        },
        status=status,
        interpretation=interpretation,
        recommended_action=action,
    )
