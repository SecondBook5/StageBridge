from __future__ import annotations

from stagebridge.evaluation.context_ablation import evaluate_context_ablation
from stagebridge.evaluation.diffusion_ablation import evaluate_diffusion_ablation
from stagebridge.evaluation.edge_comparison import evaluate_edge_comparison
from stagebridge.evaluation.provider_sensitivity import evaluate_provider_sensitivity
from stagebridge.evaluation.reporting import (
    GateEvaluationResult,
    build_context_comparison_payload,
    build_edge_comparison_payload,
    scientific_gate_summary_payload,
)
from stagebridge.evaluation.wes_ablation import evaluate_wes_ablation


def test_gate_evaluation_result_contract_is_machine_readable() -> None:
    result = GateEvaluationResult(
        gate_name="provider_sensitivity",
        run_label="gate_pass_1",
        mode="provider_compare",
        edge="AAH->AIS",
        config_signature="smoke",
        primary_metric=None,
        secondary_metrics={"provider_status": {"tangram": "complete"}},
        status="inconclusive",
        interpretation="Only one provider is runnable.",
        recommended_action="needs_more_data",
    )

    payload = scientific_gate_summary_payload([result])

    assert payload["gates"][0]["gate_name"] == "provider_sensitivity"
    assert payload["gates"][0]["status"] == "inconclusive"


def test_context_ablation_can_demote_set_only_when_deep_sets_is_better() -> None:
    result = evaluate_context_ablation(
        run_label="gate_pass_1",
        edge="AAH->AIS",
        config_signature="smoke",
        mode_results={
            "pooled": {
                "heldout_metrics": {"sinkhorn": 10.0},
                "calibration": {"mean_abs_shift_error": 1.0},
            },
            "set_only": {
                "heldout_metrics": {"sinkhorn": 12.0},
                "calibration": {"mean_abs_shift_error": 1.4},
            },
            "deep_sets": {
                "heldout_metrics": {"sinkhorn": 9.5},
                "calibration": {"mean_abs_shift_error": 0.9},
            },
            "graph_of_sets": {
                "heldout_metrics": {"sinkhorn": 13.0},
                "calibration": {"mean_abs_shift_error": 1.6},
            },
        },
    )

    assert result.status == "fail"
    assert result.recommended_action == "demote"


def test_provider_sensitivity_is_inconclusive_with_one_runnable_provider() -> None:
    result = evaluate_provider_sensitivity(
        run_label="gate_pass_1",
        edge="AAH->AIS",
        config_signature="smoke",
        provider_results={
            "tangram": {"status": "complete"},
            "tacco": {"status": "interface_ready"},
            "destvi": {"status": "interface_ready"},
        },
    )

    assert result.status == "inconclusive"
    assert result.recommended_action == "needs_more_data"


def test_provider_sensitivity_can_consume_benchmark_output() -> None:
    result = evaluate_provider_sensitivity(
        run_label="gate_pass_provider_benchmark",
        edge="AAH->AIS",
        config_signature="medium",
        provider_results={
            "benchmark": {
                "selected_provider": "tacco",
                "selection_status": "weak_pass",
                "selection_reason": "tacco ranked first but the margin was modest.",
                "recommended_action": "keep_as_optional",
                "provider_scores": [
                    {"method": "tacco", "hybrid_rank_score": 1.2},
                    {"method": "destvi", "hybrid_rank_score": 1.4},
                    {"method": "tangram", "hybrid_rank_score": 2.8},
                ],
            }
        },
    )

    assert result.status == "weak_pass"
    assert result.recommended_action == "keep_as_optional"
    assert result.secondary_metrics["provider_scores"][0]["method"] == "tacco"


def test_reporting_helpers_emit_machine_readable_context_and_edge_summaries() -> None:
    gate = GateEvaluationResult(
        gate_name="context_ablation",
        run_label="gate_pass_2",
        mode="context_compare",
        edge="AIS->MIA",
        config_signature="smoke",
        primary_metric=15.7,
        secondary_metrics={},
        status="weak_pass",
        interpretation="Set-only earned its place.",
        recommended_action="keep_as_optional",
    )
    mode_results = {
        "rna_only": {
            "heldout_metrics": {"sinkhorn": 16.2, "classifier_auc": 0.63},
            "calibration": {"mean_abs_shift_error": 4.78},
            "context_sensitivity": None,
        },
        "deep_sets": {
            "heldout_metrics": {"sinkhorn": 15.9, "classifier_auc": 0.61},
            "calibration": {"mean_abs_shift_error": 4.32},
            "context_sensitivity": {"context_sensitivity_delta": -0.05},
            "biology_summary": {"dominant_increase_group": "stromal", "dominant_decrease_group": "immune"},
        },
        "set_only": {
            "heldout_metrics": {"sinkhorn": 15.7, "classifier_auc": 0.60},
            "calibration": {"mean_abs_shift_error": 4.18},
            "context_sensitivity": {"context_sensitivity_delta": -0.10},
            "biology_summary": {"dominant_increase_group": "epithelial", "dominant_decrease_group": "vascular_program"},
        },
    }

    context_payload = build_context_comparison_payload(
        edge="AIS->MIA",
        mode_results=mode_results,
        gate_result=gate,
    )
    edge_payload = build_edge_comparison_payload(
        mode="set_only",
        edge_results={
            "AAH->AIS": mode_results["set_only"],
            "AIS->MIA": mode_results["set_only"],
        },
        gate_result=gate,
    )

    assert context_payload["edge"] == "AIS->MIA"
    assert context_payload["modes"]["rna_only"]["context_sensitivity_delta"] is None
    assert context_payload["modes"]["deep_sets"]["context_sensitivity_delta"] == -0.05
    assert context_payload["modes"]["deep_sets"]["dominant_increase_group"] == "stromal"
    assert context_payload["modes"]["set_only"]["context_sensitivity_delta"] == -0.10
    assert context_payload["modes"]["set_only"]["dominant_increase_group"] == "epithelial"
    assert edge_payload["mode"] == "set_only"
    assert "AAH->AIS" in edge_payload["edges"]


def test_edge_comparison_can_weak_pass_on_measurable_edge_difference() -> None:
    result = evaluate_edge_comparison(
        run_label="gate_pass_3",
        mode="set_only",
        config_signature="smoke",
        edge_results={
            "AAH->AIS": {
                "heldout_metrics": {"sinkhorn": 18.0, "classifier_auc": 0.7},
                "calibration": {"mean_abs_shift_error": 4.0},
                "context_sensitivity": {"context_sensitivity_delta": -0.20},
                "biology_summary": {"dominant_increase_group": "stromal"},
            },
            "AIS->MIA": {
                "heldout_metrics": {"sinkhorn": 15.5, "classifier_auc": 0.7},
                "calibration": {"mean_abs_shift_error": 4.3},
                "context_sensitivity": {"context_sensitivity_delta": -0.05},
                "biology_summary": {"dominant_increase_group": "epithelial"},
            },
        },
    )

    assert result.status in {"weak_pass", "pass"}
    assert result.recommended_action in {"keep", "keep_as_optional"}


def test_wes_ablation_can_demote_when_regularization_does_not_help() -> None:
    result = evaluate_wes_ablation(
        run_label="gate_pass_4",
        edge="AAH->AIS",
        config_signature="smoke",
        wes_off_result={
            "heldout_metrics": {"sinkhorn": 10.0},
            "calibration": {"mean_abs_shift_error": 1.0},
            "wes_diagnostics": {"regularizer_mean_penalty": 0.0},
        },
        wes_on_result={
            "heldout_metrics": {"sinkhorn": 10.5},
            "calibration": {"mean_abs_shift_error": 1.2},
            "wes_diagnostics": {"regularizer_mean_penalty": 0.2},
        },
    )

    assert result.status == "fail"
    assert result.recommended_action == "demote"


def test_diffusion_ablation_can_demote_when_state_dependent_does_not_help() -> None:
    result = evaluate_diffusion_ablation(
        run_label="gate_pass_5",
        edge="AAH->AIS",
        config_signature="smoke",
        fixed_result={
            "heldout_metrics": {"sinkhorn": 9.8},
            "calibration": {"mean_abs_shift_error": 1.0},
            "diffusion_diagnostics": {"mean_diffusion_scale": 0.4},
        },
        state_dependent_result={
            "heldout_metrics": {"sinkhorn": 10.0},
            "calibration": {"mean_abs_shift_error": 1.1},
            "diffusion_diagnostics": {"mean_diffusion_scale": 0.6},
        },
    )

    assert result.status == "fail"
    assert result.recommended_action == "demote"
