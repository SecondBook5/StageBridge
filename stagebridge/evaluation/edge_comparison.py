"""Edge comparison gate for AAH->AIS vs AIS->MIA."""
from __future__ import annotations

from typing import Any

from stagebridge.evaluation.reporting import GateEvaluationResult


def evaluate_edge_comparison(
    *,
    run_label: str,
    mode: str,
    config_signature: str,
    edge_results: dict[str, dict[str, Any]],
) -> GateEvaluationResult:
    required = {"AAH->AIS", "AIS->MIA"}
    if any(edge not in edge_results for edge in required):
        return GateEvaluationResult(
            gate_name="edge_comparison",
            run_label=run_label,
            mode=mode,
            edge="AAH->AIS vs AIS->MIA",
            config_signature=config_signature,
            primary_metric=None,
            secondary_metrics={"available_edges": sorted(edge_results)},
            status="inconclusive",
            interpretation="Both biologically prioritized edges are not yet available for a fair comparison.",
            recommended_action="needs_more_data",
        )
    aah = edge_results["AAH->AIS"]
    ais = edge_results["AIS->MIA"]
    sinkhorn_delta = float(ais["heldout_metrics"]["sinkhorn"] - aah["heldout_metrics"]["sinkhorn"])
    calibration_delta = float(
        ais["calibration"]["mean_abs_shift_error"] - aah["calibration"]["mean_abs_shift_error"]
    )

    def _context_delta(result: dict[str, Any]) -> float | None:
        context = result.get("context_sensitivity")
        if not isinstance(context, dict):
            return None
        value = context.get("context_sensitivity_delta")
        return None if value is None else float(value)

    aah_context = _context_delta(aah)
    ais_context = _context_delta(ais)
    context_delta = None
    if aah_context is not None and ais_context is not None:
        context_delta = float(ais_context - aah_context)
    aah_biology = (aah.get("biology_summary") or {}).get("dominant_increase_group")
    ais_biology = (ais.get("biology_summary") or {}).get("dominant_increase_group")
    biology_difference = (
        aah_biology is not None
        and ais_biology is not None
        and str(aah_biology) != str(ais_biology)
    )

    hard_difference = abs(sinkhorn_delta) > 1.0
    calibration_difference = abs(calibration_delta) > 0.2
    context_difference = context_delta is not None and abs(context_delta) > 0.05

    if hard_difference and (calibration_difference or context_difference or biology_difference):
        status = "pass"
        action = "keep"
        interpretation = "The prioritized edges differ in transition difficulty and at least one supporting metric or biology summary, so an edge-specific comparison is justified."
    elif hard_difference or calibration_difference or context_difference or biology_difference:
        status = "weak_pass"
        action = "keep_as_optional"
        interpretation = "The two edges show a measurable metric or biology difference, but the claim still needs replication before stronger biological language is justified."
    else:
        status = "fail"
        action = "remove_from_v1_claims"
        interpretation = "The two prioritized edges do not yet separate clearly enough to justify an edge-specific claim."
    return GateEvaluationResult(
        gate_name="edge_comparison",
        run_label=run_label,
        mode=mode,
        edge="AAH->AIS vs AIS->MIA",
        config_signature=config_signature,
        primary_metric=sinkhorn_delta,
        secondary_metrics={
            "aah_to_ais_sinkhorn": float(aah["heldout_metrics"]["sinkhorn"]),
            "ais_to_mia_sinkhorn": float(ais["heldout_metrics"]["sinkhorn"]),
            "aah_to_ais_calibration": float(aah["calibration"]["mean_abs_shift_error"]),
            "ais_to_mia_calibration": float(ais["calibration"]["mean_abs_shift_error"]),
            "aah_to_ais_context_sensitivity": aah_context,
            "ais_to_mia_context_sensitivity": ais_context,
            "aah_to_ais_dominant_increase_group": aah_biology,
            "ais_to_mia_dominant_increase_group": ais_biology,
        },
        status=status,
        interpretation=interpretation,
        recommended_action=action,
    )
