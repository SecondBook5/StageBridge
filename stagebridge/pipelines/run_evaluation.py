"""Evaluation pipeline entrypoint."""
from __future__ import annotations

from typing import Any

from omegaconf import DictConfig
import torch

from stagebridge.evaluation.ablations import summarize_ablation
from stagebridge.evaluation.biological_insight import summarize_edge_biology
from stagebridge.evaluation.calibration import summarize_transition_calibration
from stagebridge.evaluation.context_ablation import evaluate_context_ablation
from stagebridge.evaluation.context_sensitivity import compare_real_vs_shuffled_context
from stagebridge.evaluation.diffusion_ablation import evaluate_diffusion_ablation
from stagebridge.evaluation.edge_comparison import evaluate_edge_comparison
from stagebridge.evaluation.fixed_points import summarize_fixed_points
from stagebridge.evaluation.latent_sensitivity import evaluate_latent_sensitivity
from stagebridge.evaluation.metrics import heldout_transition_metrics, rollout_edge_transition
from stagebridge.evaluation.niche_regimes import summarize_niche_regimes
from stagebridge.evaluation.pseudotime_structure import summarize_pseudotime_structure
from stagebridge.evaluation.provider_sensitivity import evaluate_provider_sensitivity
from stagebridge.evaluation.reports import build_evaluation_report
from stagebridge.evaluation.reporting import render_scientific_gate_summary_md, scientific_gate_summary_payload
from stagebridge.evaluation.trajectory_analysis import summarize_edge_trajectory
from stagebridge.evaluation.wes_ablation import evaluate_wes_ablation
from stagebridge.pipelines.run_transition_model import run_transition_model


def run_evaluation(
    cfg: DictConfig,
    *,
    transition_output: dict[str, Any] | None = None,
    context_output: dict[str, Any] | None = None,
) -> dict[str, Any]:
    transition = transition_output or run_transition_model(cfg)
    if not transition.get("ok"):
        return {
            "ok": False,
            "pipeline": "evaluation",
            "status": "upstream_transition_failed",
        }

    model = transition["model"]
    x_src = transition["x_src_test"]
    x_tgt = transition["x_tgt_test"]
    context = transition["context"]
    context_tokens = transition.get("context_tokens")
    shuffled_context = transition["shuffled_context"]
    shuffled_context_tokens = transition.get("shuffled_context_tokens")
    edge_id = int(transition["edge_id"])
    mode = str(transition["mode"])

    x_pred = rollout_edge_transition(
        model,
        x_src,
        context=context,
        context_tokens=context_tokens,
        edge_id=edge_id,
        num_steps=8,
        stochastic=False,
    )
    heldout = heldout_transition_metrics(
        model,
        x_src,
        x_tgt,
        context=context,
        context_tokens=context_tokens,
        edge_id=edge_id,
        num_steps=8,
        stochastic=False,
        epsilon=float(cfg.get("transition_model", {}).get("schrodinger_bridge", {}).get("ot_epsilon", 0.05)),
        sinkhorn_iters=int(cfg.get("transition_model", {}).get("schrodinger_bridge", {}).get("sinkhorn_iters", 80)),
    )
    calibration = summarize_transition_calibration(x_src, x_pred, x_tgt)
    context_sensitivity = None
    if mode != "rna_only":
        context_sensitivity = compare_real_vs_shuffled_context(
            model,
            x_src,
            x_tgt,
            real_context=context,
            shuffled_context=shuffled_context,
            real_context_tokens=context_tokens,
            shuffled_context_tokens=shuffled_context_tokens,
            edge_id=edge_id,
            num_steps=8,
            stochastic=False,
            ot_epsilon=float(cfg.get("transition_model", {}).get("schrodinger_bridge", {}).get("ot_epsilon", 0.05)),
            sinkhorn_iters=int(cfg.get("transition_model", {}).get("schrodinger_bridge", {}).get("sinkhorn_iters", 80)),
        )
    trajectory = summarize_edge_trajectory(
        x_src.detach().cpu().numpy(),
        x_pred.detach().cpu().numpy(),
        x_tgt.detach().cpu().numpy(),
    )
    fixed_points = summarize_fixed_points(model, x_pred, context=context, edge_id=edge_id)
    niche_regimes = None
    if transition.get("typed_subset_tokens") is not None:
        niche_regimes = summarize_niche_regimes(
            transition["typed_subset_tokens"],
            transition["typed_feature_names"],
        )
    biology_summary = None
    typed_for_biology = None
    if context_output is not None:
        typed_for_biology = context_output.get("typed_tokens")
    if typed_for_biology is not None:
        biology_summary = summarize_edge_biology(
            typed_for_biology,
            edge_label=str(transition["edge"]),
            context_sensitivity=context_sensitivity,
            split_summary=transition["split_summary"],
            wes_diagnostics=transition.get("wes_diagnostics"),
        )
    pseudotime_structure = summarize_pseudotime_structure(x_src, x_pred, x_tgt)
    ablation = summarize_ablation(
        mode,
        heldout,
        wes_enabled=bool(transition.get("wes_diagnostics", {}).get("enabled", False)),
    )
    report = build_evaluation_report(
        edge_label=str(transition["edge"]),
        mode=mode,
        heldout_metrics=heldout,
        calibration=calibration,
        context_sensitivity=context_sensitivity,
        trajectory=trajectory,
        fixed_points=fixed_points,
        niche_regimes=niche_regimes,
        pseudotime_structure=pseudotime_structure,
        split_summary=transition["split_summary"],
        biology_summary=biology_summary,
    )
    diffusion = model.forward_diffusion(
        x_t=x_pred,
        t=x_pred.new_full((x_pred.shape[0],), 0.5),
        context=context,
        edge_ids=torch.full((x_pred.shape[0],), edge_id, dtype=torch.long, device=x_pred.device),
    )
    diffusion_diagnostics = {
        "mean_diffusion_scale": float(diffusion.mean().item()),
        "max_diffusion_scale": float(diffusion.max().item()),
        "min_diffusion_scale": float(diffusion.min().item()),
    }
    artifact_sources = {
        "evaluation_report.json": report,
        "heldout_metrics.json": heldout,
        "calibration.json": calibration,
        "diffusion_diagnostics.json": diffusion_diagnostics,
        "split_summary.json": transition["split_summary"],
        "dataset_transfer_diagnostics.json": transition.get("dataset_transfer_diagnostics", {}),
    }
    if transition.get("pretraining_summary") is not None:
        artifact_sources["pretraining_summary.json"] = transition.get("pretraining_summary", {})
    if transition.get("auxiliary_context_shuffle_metrics") is not None:
        artifact_sources["transformer_auxiliary_metrics.json"] = transition.get("auxiliary_context_shuffle_metrics", {})
    if context_sensitivity is not None:
        artifact_sources["context_sensitivity.json"] = context_sensitivity
    if niche_regimes is not None:
        artifact_sources["niche_regimes.json"] = niche_regimes
    if biology_summary is not None:
        artifact_sources["biology_summary.json"] = biology_summary

    return {
        "ok": True,
        "pipeline": "evaluation",
        "status": "complete",
        "heldout_metrics": heldout,
        "calibration": calibration,
        "context_sensitivity": context_sensitivity,
        "trajectory": trajectory,
        "fixed_points": fixed_points,
        "niche_regimes": niche_regimes,
        "biology_summary": biology_summary,
        "pseudotime_structure": pseudotime_structure,
        "diffusion_diagnostics": diffusion_diagnostics,
        "ablation": ablation,
        "report": report,
        "dataset_transfer_diagnostics": transition.get("dataset_transfer_diagnostics", {}),
        "artifact_sources": artifact_sources,
    }


def run_scientific_gate_pass(
    *,
    run_label: str,
    config_signature: str,
    latent_results: dict[str, dict[str, Any]] | None = None,
    provider_results: dict[str, dict[str, Any]] | None = None,
    context_mode_results: dict[str, dict[str, Any]] | None = None,
    fixed_diffusion_result: dict[str, Any] | None = None,
    state_dependent_result: dict[str, Any] | None = None,
    wes_off_result: dict[str, Any] | None = None,
    wes_on_result: dict[str, Any] | None = None,
    edge_results: dict[str, dict[str, Any]] | None = None,
    edge: str = "AAH->AIS",
    mode: str = "set_only",
) -> dict[str, Any]:
    gate_results = [
        evaluate_latent_sensitivity(
            run_label=run_label,
            edge=edge,
            config_signature=config_signature,
            backend_results=latent_results or {},
        ),
        evaluate_provider_sensitivity(
            run_label=run_label,
            edge=edge,
            config_signature=config_signature,
            provider_results=provider_results or {},
        ),
        evaluate_context_ablation(
            run_label=run_label,
            edge=edge,
            config_signature=config_signature,
            mode_results=context_mode_results or {},
        ),
        evaluate_diffusion_ablation(
            run_label=run_label,
            edge=edge,
            config_signature=config_signature,
            fixed_result=fixed_diffusion_result,
            state_dependent_result=state_dependent_result,
        ),
        evaluate_wes_ablation(
            run_label=run_label,
            edge=edge,
            config_signature=config_signature,
            wes_off_result=wes_off_result,
            wes_on_result=wes_on_result,
        ),
        evaluate_edge_comparison(
            run_label=run_label,
            mode=mode,
            config_signature=config_signature,
            edge_results=edge_results or {},
        ),
    ]
    return {
        "gate_results": [result.to_dict() for result in gate_results],
        "scientific_gate_summary": scientific_gate_summary_payload(gate_results),
        "scientific_gate_summary_md": render_scientific_gate_summary_md(gate_results),
    }
