"""Latent sensitivity gate for StageBridge."""
from __future__ import annotations

from typing import Any

from stagebridge.evaluation.reporting import GateEvaluationResult


def evaluate_latent_sensitivity(
    *,
    run_label: str,
    edge: str,
    config_signature: str,
    backend_results: dict[str, dict[str, Any]],
) -> GateEvaluationResult:
    runnable = {name: payload for name, payload in backend_results.items() if payload.get("status") == "complete"}
    if len(runnable) < 2:
        return GateEvaluationResult(
            gate_name="latent_sensitivity",
            run_label=run_label,
            mode="latent_compare",
            edge=edge,
            config_signature=config_signature,
            primary_metric=None,
            secondary_metrics={"available_backends": list(runnable)},
            status="inconclusive",
            interpretation="Only one credible latent backend is currently available, so sensitivity is not yet testable.",
            recommended_action="needs_more_data",
        )
    return GateEvaluationResult(
        gate_name="latent_sensitivity",
        run_label=run_label,
        mode="latent_compare",
        edge=edge,
        config_signature=config_signature,
        primary_metric=None,
        secondary_metrics=runnable,
        status="weak_pass",
        interpretation="Multiple latent backends ran, but this gate still needs a stronger baseline comparison before claims can be hardened.",
        recommended_action="keep_as_optional",
    )
