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
