from __future__ import annotations

from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

from stagebridge.pipelines.run_story_reporting import run_story_reporting


def test_run_story_reporting_writes_tables_and_figures(tmp_path: Path) -> None:
    transition = pd.DataFrame(
        {
            "edge": ["AIS->MIA", "AIS->MIA", "AIS->MIA"],
            "mode": ["rna_only", "pooled", "set_only"],
            "transformer_used": ["no", "no", "yes"],
            "local_context_used": ["no", "yes", "yes"],
            "graph_context_used": ["no", "no", "no"],
            "wes_enabled": ["no", "no", "no"],
            "diffusion_enabled": ["no", "no", "no"],
            "primary_metric": [16.30, 15.91, 15.75],
            "status": ["complete", "complete", "complete"],
            "course_interpretation": ["baseline", "baseline", "best transformer"],
        }
    )
    communication = pd.DataFrame(
        {
            "model_name": ["pooled", "stagebridge", "graph_transformer"],
            "fold": [0, 0, 0],
            "seed": [42, 42, 42],
            "auroc": [0.71, 0.43, 0.49],
            "auprc": [0.90, 0.79, 0.82],
            "balanced_accuracy": [0.51, 0.54, 0.50],
            "macro_f1": [0.60, 0.52, 0.55],
            "ece": [0.10, 0.14, 0.11],
            "context_shuffle_auroc_delta": [0.04, -0.02, 0.01],
            "context_shuffle_auprc_delta": [0.03, -0.01, 0.01],
            "artifact_dir": ["a", "b", "c"],
        }
    )
    manifest = pd.DataFrame(
        {
            "sample_id": ["S1", "S2", "S3"],
            "edge_label": ["AAH->AIS", "AIS->MIA", "AIS->MIA"],
            "progression_competent_label": [1, 0, 1],
        }
    )

    transition_path = tmp_path / "transition.csv"
    communication_path = tmp_path / "communication.csv"
    manifest_path = tmp_path / "manifest.csv"
    transition.to_csv(transition_path, index=False)
    communication.to_csv(communication_path, index=False)
    manifest.to_csv(manifest_path, index=False)

    cfg = OmegaConf.create(
        {
            "story_report": {
                "reports_root": str(tmp_path / "reports"),
                "transition_source": str(transition_path),
                "communication_ais_sources": [str(communication_path)],
                "communication_combined_sources": [],
                "curated_manifest_path": str(manifest_path),
            }
        }
    )

    result = run_story_reporting(cfg)

    assert result["ok"] is True
    reports_root = Path(result["reports_root"])
    assert (
        reports_root / "benchmarks" / "communication_relay" / "ais_model_family_summary.csv"
    ).exists()
    assert (
        reports_root / "benchmarks" / "story" / "transition_vs_communication_story.csv"
    ).exists()
    assert (
        reports_root
        / "poster"
        / "hca_general_meeting"
        / "figures"
        / "figure_transition_vs_communication_story.png"
    ).exists()
