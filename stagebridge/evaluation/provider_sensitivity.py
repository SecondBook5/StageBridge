"""Provider sensitivity gate for spatial mapping backends."""
from __future__ import annotations

from typing import Any

from stagebridge.evaluation.reporting import GateEvaluationResult


def evaluate_provider_sensitivity(
    *,
    run_label: str,
    edge: str,
    config_signature: str,
    provider_results: dict[str, dict[str, Any]],
) -> GateEvaluationResult:
    benchmark = provider_results.get("benchmark") if isinstance(provider_results, dict) else None
    if isinstance(benchmark, dict) and benchmark.get("provider_scores"):
        selected = benchmark.get("selected_provider")
        status = str(benchmark.get("selection_status", "inconclusive"))
        recommended_action = str(benchmark.get("recommended_action", "needs_more_data"))
        interpretation = str(benchmark.get("selection_reason", "No provider benchmark interpretation available."))
        return GateEvaluationResult(
            gate_name="provider_sensitivity",
            run_label=run_label,
            mode="provider_compare",
            edge=edge,
            config_signature=config_signature,
            primary_metric=None,
            secondary_metrics={"provider_scores": benchmark.get("provider_scores", [])},
            status=status if status in {"pass", "weak_pass", "fail", "inconclusive"} else "inconclusive",
            interpretation=interpretation,
            recommended_action=recommended_action
            if recommended_action in {"keep", "keep_as_optional", "demote", "remove_from_v1_claims", "needs_more_data"}
            else "needs_more_data",
        )

    runnable = {
        name: payload
        for name, payload in provider_results.items()
        if payload.get("status") in {"complete", "weak_pass", "pass"}
    }
    if len(runnable) < 2:
        return GateEvaluationResult(
            gate_name="provider_sensitivity",
            run_label=run_label,
            mode="provider_compare",
            edge=edge,
            config_signature=config_signature,
            primary_metric=None,
            secondary_metrics={"provider_status": provider_results},
            status="inconclusive",
            interpretation="Only one credible provider is runnable, so provider sensitivity cannot yet be judged honestly.",
            recommended_action="needs_more_data",
        )
    return GateEvaluationResult(
        gate_name="provider_sensitivity",
        run_label=run_label,
        mode="provider_compare",
        edge=edge,
        config_signature=config_signature,
        primary_metric=None,
        secondary_metrics=runnable,
        status="weak_pass",
        interpretation="Multiple providers are runnable, but provider-sensitivity conclusions still need direct downstream stability comparisons.",
        recommended_action="keep_as_optional",
    )
