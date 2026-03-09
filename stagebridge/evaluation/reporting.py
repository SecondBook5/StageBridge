"""Scientific gate reporting contracts for Mission 3."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

VALID_GATE_STATUS = {"pass", "weak_pass", "fail", "inconclusive"}
VALID_RECOMMENDED_ACTION = {
    "keep",
    "keep_as_optional",
    "demote",
    "remove_from_v1_claims",
    "needs_more_data",
}


@dataclass(slots=True, frozen=True)
class GateEvaluationResult:
    gate_name: str
    run_label: str
    mode: str
    edge: str
    config_signature: str
    primary_metric: float | None
    secondary_metrics: dict[str, Any]
    status: str
    interpretation: str
    recommended_action: str

    def to_dict(self) -> dict[str, Any]:
        if self.status not in VALID_GATE_STATUS:
            raise ValueError(f"Invalid gate status '{self.status}'.")
        if self.recommended_action not in VALID_RECOMMENDED_ACTION:
            raise ValueError(f"Invalid recommended_action '{self.recommended_action}'.")
        return asdict(self)


def scientific_gate_summary_payload(results: list[GateEvaluationResult]) -> dict[str, Any]:
    return {"gates": [result.to_dict() for result in results]}


def render_scientific_gate_summary_md(results: list[GateEvaluationResult]) -> str:
    lines = ["# Scientific Gate Summary", ""]
    for result in results:
        lines.extend(
            [
                f"## {result.gate_name}",
                f"- Run label: {result.run_label}",
                f"- Mode: {result.mode}",
                f"- Edge: {result.edge}",
                f"- Status: {result.status}",
                f"- Recommended action: {result.recommended_action}",
                f"- Primary metric: {result.primary_metric}",
                f"- Interpretation: {result.interpretation}",
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def _safe_context_delta(payload: dict[str, Any]) -> float | None:
    context = payload.get("context_sensitivity")
    if not isinstance(context, dict):
        return None
    value = context.get("context_sensitivity_delta")
    if value is None:
        return None
    return float(value)


def build_context_comparison_payload(
    *,
    edge: str,
    mode_results: dict[str, dict[str, Any]],
    gate_result: GateEvaluationResult,
) -> dict[str, Any]:
    ordered_modes = [
        "rna_only",
        "pooled",
        "deep_sets",
        "set_only",
        "typed_hierarchical_transformer",
        "deep_sets_transformer_hybrid",
        "graph_of_sets",
    ]
    comparison: dict[str, Any] = {
        "edge": edge,
        "gate": gate_result.to_dict(),
        "modes": {},
    }
    for mode in ordered_modes:
        if mode not in mode_results:
            continue
        payload = mode_results[mode]
        comparison["modes"][mode] = {
            "primary_metric": float(payload["heldout_metrics"]["sinkhorn"]),
            "calibration_error": float(payload["calibration"]["mean_abs_shift_error"]),
            "context_sensitivity_delta": _safe_context_delta(payload),
            "classifier_auc": float(payload["heldout_metrics"]["classifier_auc"]),
            "dominant_increase_group": (payload.get("biology_summary") or {}).get("dominant_increase_group"),
            "dominant_decrease_group": (payload.get("biology_summary") or {}).get("dominant_decrease_group"),
            "biology_interpretation": (payload.get("biology_summary") or {}).get("interpretation"),
            "status": payload.get("status", "complete"),
            "runtime_sanity": payload.get("runtime_sanity"),
        }
    return comparison


def render_context_comparison_md(
    *,
    edge: str,
    mode_results: dict[str, dict[str, Any]],
    gate_result: GateEvaluationResult,
) -> str:
    lines = [
        f"# Context Ablation: {edge}",
        "",
        f"- Gate status: {gate_result.status}",
        f"- Recommended action: {gate_result.recommended_action}",
        f"- Interpretation: {gate_result.interpretation}",
        "",
    ]
    for mode in [
        "rna_only",
        "pooled",
        "deep_sets",
        "set_only",
        "typed_hierarchical_transformer",
        "deep_sets_transformer_hybrid",
        "graph_of_sets",
    ]:
        if mode not in mode_results:
            continue
        payload = mode_results[mode]
        context_delta = _safe_context_delta(payload)
        lines.extend(
            [
                f"## {mode}",
                f"- Sinkhorn: {float(payload['heldout_metrics']['sinkhorn']):.6f}",
                f"- Calibration error: {float(payload['calibration']['mean_abs_shift_error']):.6f}",
                f"- Classifier AUC: {float(payload['heldout_metrics']['classifier_auc']):.6f}",
                f"- Context sensitivity delta: {context_delta if context_delta is not None else 'n/a'}",
                f"- Dominant increase group: {(payload.get('biology_summary') or {}).get('dominant_increase_group', 'n/a')}",
                f"- Dominant decrease group: {(payload.get('biology_summary') or {}).get('dominant_decrease_group', 'n/a')}",
                f"- Runtime sanity: {payload.get('runtime_sanity', 'n/a')}",
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def build_edge_comparison_payload(
    *,
    mode: str,
    edge_results: dict[str, dict[str, Any]],
    gate_result: GateEvaluationResult,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "mode": mode,
        "gate": gate_result.to_dict(),
        "edges": {},
    }
    for edge_name, result in edge_results.items():
        payload["edges"][edge_name] = {
            "primary_metric": float(result["heldout_metrics"]["sinkhorn"]),
            "calibration_error": float(result["calibration"]["mean_abs_shift_error"]),
            "context_sensitivity_delta": _safe_context_delta(result),
            "classifier_auc": float(result["heldout_metrics"]["classifier_auc"]),
            "dominant_increase_group": (result.get("biology_summary") or {}).get("dominant_increase_group"),
            "dominant_decrease_group": (result.get("biology_summary") or {}).get("dominant_decrease_group"),
            "biology_interpretation": (result.get("biology_summary") or {}).get("interpretation"),
            "runtime_sanity": result.get("runtime_sanity"),
        }
    return payload


def render_edge_comparison_md(
    *,
    mode: str,
    edge_results: dict[str, dict[str, Any]],
    gate_result: GateEvaluationResult,
) -> str:
    lines = [
        f"# Edge Comparison: {mode}",
        "",
        f"- Gate status: {gate_result.status}",
        f"- Recommended action: {gate_result.recommended_action}",
        f"- Interpretation: {gate_result.interpretation}",
        "",
    ]
    for edge_name, result in edge_results.items():
        context_delta = _safe_context_delta(result)
        lines.extend(
            [
                f"## {edge_name}",
                f"- Sinkhorn: {float(result['heldout_metrics']['sinkhorn']):.6f}",
                f"- Calibration error: {float(result['calibration']['mean_abs_shift_error']):.6f}",
                f"- Classifier AUC: {float(result['heldout_metrics']['classifier_auc']):.6f}",
                f"- Context sensitivity delta: {context_delta if context_delta is not None else 'n/a'}",
                f"- Dominant increase group: {(result.get('biology_summary') or {}).get('dominant_increase_group', 'n/a')}",
                f"- Dominant decrease group: {(result.get('biology_summary') or {}).get('dominant_decrease_group', 'n/a')}",
                f"- Runtime sanity: {result.get('runtime_sanity', 'n/a')}",
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"
