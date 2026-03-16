from __future__ import annotations

from functools import lru_cache

import pandas as pd
import pytest

from stagebridge.data.luad_evo.metadata import resolve_luad_evo_paths
from stagebridge.notebook_api import compose_config
from stagebridge.pipelines.run_evaluation import run_evaluation
from stagebridge.pipelines.run_full import run_full
from stagebridge.pipelines.run_transition_model import run_transition_model
from stagebridge.results.run_writer import write_pipeline_scratch_run
from stagebridge.transition_model.train import build_stagewise_edge_split


@lru_cache(maxsize=1)
def _real_data_assets_available() -> bool:
    cfg = compose_config(
        "default",
        overrides=["data=local", "train=smoke", "evaluation=baseline"],
    )
    paths = resolve_luad_evo_paths(cfg)
    has_snrna = paths.snrna_latent_h5ad.exists() or paths.snrna_h5ad.exists()
    has_spatial = paths.spatial_h5ad.exists() or paths.spatial_tangram_h5ad.exists()
    return has_snrna and has_spatial


@pytest.fixture(autouse=True)
def _skip_real_data_smokes_when_assets_missing(request) -> None:
    if request.function.__name__ == "test_stagewise_edge_split_reports_missing_same_donor_overlap":
        return
    if not _real_data_assets_available():
        pytest.skip(
            "Real-data transition smoke tests require local LUAD assets (snRNA + spatial)."
        )


def test_stagewise_edge_split_reports_missing_same_donor_overlap() -> None:
    src_obs = pd.DataFrame({"donor_id": ["P1", "P2", "P3"], "stage": ["AIS"] * 3})
    tgt_obs = pd.DataFrame({"donor_id": ["P4", "P5"], "stage": ["MIA"] * 2})

    split = build_stagewise_edge_split(
        src_obs,
        tgt_obs,
        donor_col="donor_id",
        stage_src="AIS",
        stage_tgt="MIA",
    )

    assert split.split_strategy == "stagewise_donor_holdout"
    assert split.overlap_donors == []
    assert split.notes


def test_rna_only_transition_and_evaluation_smoke_runs_on_real_data() -> None:
    cfg = compose_config(
        "default",
        overrides=[
            "data=local",
            "train=smoke",
            "evaluation=baseline",
            "context_model.mode=rna_only",
            "transition_model.active_edge=[AAH,AIS]",
            "transition_model.max_cells_per_stage=32",
            "transition_model.schrodinger_bridge.sigma=0.0",
            "transition_model.wes_regularizer.enabled=false",
        ],
    )

    transition = run_transition_model(cfg)
    evaluation = run_evaluation(cfg, transition_output=transition)

    assert transition["ok"] is True
    assert transition["status"] == "complete"
    assert transition["edge"] == "AAH->AIS"
    assert transition["mode"] == "rna_only"
    assert transition["training_history"]
    assert evaluation["ok"] is True
    assert evaluation["status"] == "complete"
    assert "sinkhorn" in evaluation["heldout_metrics"]


def test_set_only_transition_smoke_runs_on_real_data() -> None:
    cfg = compose_config(
        "default",
        overrides=[
            "data=local",
            "train=smoke",
            "evaluation=baseline",
            "context_model.mode=set_only",
            "transition_model.active_edge=[AAH,AIS]",
            "transition_model.max_cells_per_stage=32",
            "transition_model.schrodinger_bridge.sigma=0.0",
            "transition_model.wes_regularizer.enabled=false",
        ],
    )

    transition = run_transition_model(cfg)

    assert transition["ok"] is True
    assert transition["status"] == "complete"
    assert transition["mode"] == "set_only"
    assert transition["context_diagnostics"]["context_norm"] > 0.0


def test_set_only_transition_jointly_trains_context_encoder_and_emits_attention() -> None:
    cfg = compose_config(
        "default",
        overrides=[
            "data=local",
            "train=smoke",
            "evaluation=baseline",
            "context_model.mode=set_only",
            "transition_model.active_edge=[AAH,AIS]",
            "transition_model.max_cells_per_stage=32",
            "transition_model.schrodinger_bridge.sigma=0.0",
            "transition_model.wes_regularizer.enabled=false",
        ],
    )

    transition = run_transition_model(cfg)

    assert transition["ok"] is True
    assert transition["trained_context_encoder"] is not None
    assert transition["encoder_parameter_delta"] > 0.0
    assert transition["attention_summary"] is not None
    assert transition["attention_summary"]["available_maps"]
    assert transition["attention_summary"]["top_token_attention"]
    assert transition["attention_summary"]["top_token_types"]
    assert transition["attention_summary"]["top_token_distance_bins"]
    assert transition["auxiliary_context_shuffle_metrics"]["accuracy"] >= 0.5
    assert transition["auxiliary_context_shuffle_metrics"]["loss"] >= 0.0
    assert transition["token_package"]["token_coords"] is not None
    assert transition["token_package"]["token_confidence"] is not None
    assert transition["token_package"]["token_type_ids"] is not None


def test_pooled_transition_smoke_runs_on_real_data() -> None:
    cfg = compose_config(
        "default",
        overrides=[
            "data=local",
            "train=smoke",
            "evaluation=baseline",
            "context_model.mode=pooled",
            "transition_model.active_edge=[AAH,AIS]",
            "transition_model.max_cells_per_stage=32",
            "transition_model.schrodinger_bridge.sigma=0.0",
            "transition_model.wes_regularizer.enabled=false",
        ],
    )

    transition = run_transition_model(cfg)

    assert transition["ok"] is True
    assert transition["status"] == "complete"
    assert transition["mode"] == "pooled"
    assert transition["context_diagnostics"]["context_norm"] > 0.0


def test_deep_sets_transition_smoke_runs_on_real_data() -> None:
    cfg = compose_config(
        "default",
        overrides=[
            "data=local",
            "train=smoke",
            "evaluation=baseline",
            "context_model.mode=deep_sets",
            "transition_model.active_edge=[AAH,AIS]",
            "transition_model.max_cells_per_stage=32",
            "transition_model.schrodinger_bridge.sigma=0.0",
            "transition_model.wes_regularizer.enabled=false",
        ],
    )

    transition = run_transition_model(cfg)

    assert transition["ok"] is True
    assert transition["status"] == "complete"
    assert transition["mode"] == "deep_sets"
    assert transition["context_diagnostics"]["context_norm"] > 0.0


def test_graph_of_sets_transition_smoke_runs_on_real_data() -> None:
    cfg = compose_config(
        "default",
        overrides=[
            "data=local",
            "train=smoke",
            "evaluation=baseline",
            "context_model=graph_of_sets",
            "transition_model.active_edge=[AAH,AIS]",
            "transition_model.max_cells_per_stage=32",
            "transition_model.schrodinger_bridge.sigma=0.0",
            "transition_model.wes_regularizer.enabled=false",
        ],
    )

    transition = run_transition_model(cfg)

    assert transition["ok"] is True
    assert transition["status"] == "complete"
    assert transition["mode"] == "graph_of_sets"
    assert transition["context_diagnostics"]["graph_num_edges"] > 0


def test_typed_hierarchical_transformer_transition_smoke_runs_on_real_data() -> None:
    cfg = compose_config(
        "default",
        overrides=[
            "data=local",
            "train=smoke",
            "evaluation=baseline",
            "context_model=typed_hierarchical_transformer",
            "transition_model.active_edge=[AAH,AIS]",
            "transition_model.max_cells_per_stage=24",
            "transition_model.schrodinger_bridge.sigma=0.0",
            "transition_model.wes_regularizer.enabled=false",
            "context_model.pretraining.max_epochs=1",
            "context_model.pretraining.steps_per_epoch=1",
            "context_model.pretraining.provider_consistency_enabled=false",
            "context_model.finetune.provider_consistency_weight=0.0",
        ],
    )

    transition = run_transition_model(cfg)
    evaluation = run_evaluation(cfg, transition_output=transition)

    assert transition["ok"] is True
    assert transition["mode"] == "typed_hierarchical_transformer"
    assert transition["context_tokens"] is not None
    assert transition["dataset_transfer_diagnostics"]["dataset_embedding_enabled"] is True
    assert transition["dataset_transfer_diagnostics"]["cross_dataset_negatives_used"] >= 1
    assert (
        transition["auxiliary_context_shuffle_metrics"]["task"]
        == "relational_pretraining_finetune"
    )
    assert (
        "dataset_id_mismatch"
        in transition["auxiliary_context_shuffle_metrics"]["negative_control_scores"]
    )
    assert transition["auxiliary_context_shuffle_metrics"]["drift_context_gate"] >= 0.0
    assert transition["pretraining_summary"] is not None
    assert transition["attention_summary"] is not None
    assert evaluation["ok"] is True


def test_deep_sets_transformer_hybrid_transition_smoke_runs_on_real_data() -> None:
    cfg = compose_config(
        "default",
        overrides=[
            "data=local",
            "train=smoke",
            "evaluation=baseline",
            "context_model=deep_sets_transformer_hybrid",
            "transition_model.active_edge=[AAH,AIS]",
            "transition_model.max_cells_per_stage=24",
            "transition_model.schrodinger_bridge.sigma=0.0",
            "transition_model.wes_regularizer.enabled=false",
        ],
    )

    transition = run_transition_model(cfg)
    evaluation = run_evaluation(cfg, transition_output=transition)

    assert transition["ok"] is True
    assert transition["mode"] == "deep_sets_transformer_hybrid"
    assert transition["trained_context_encoder"] is not None
    assert transition["context_tokens"] is not None
    assert transition["attention_summary"] is not None
    assert transition["attention_summary"]["hybrid_gate_mean"] >= 0.0
    assert transition["encoder_parameter_delta"] > 0.0
    assert evaluation["ok"] is True


def test_set_only_transition_smoke_runs_on_ais_to_mia_real_data() -> None:
    cfg = compose_config(
        "default",
        overrides=[
            "data=local",
            "train=smoke",
            "evaluation=baseline",
            "context_model.mode=set_only",
            "transition_model.active_edge=[AIS,MIA]",
            "transition_model.max_cells_per_stage=32",
            "transition_model.schrodinger_bridge.sigma=0.0",
            "transition_model.wes_regularizer.enabled=false",
        ],
    )

    transition = run_transition_model(cfg)

    assert transition["ok"] is True
    assert transition["status"] == "complete"
    assert transition["edge"] == "AIS->MIA"
    assert transition["mode"] == "set_only"
    assert transition["split_summary"]["split_strategy"] == "stagewise_donor_holdout"
    assert transition["split_summary"]["overlap_donors"] == []
    assert transition["context_diagnostics"]["context_norm"] > 0.0


def test_rna_only_transition_smoke_runs_on_ais_to_mia_real_data() -> None:
    cfg = compose_config(
        "default",
        overrides=[
            "data=local",
            "train=smoke",
            "evaluation=baseline",
            "context_model.mode=rna_only",
            "transition_model.active_edge=[AIS,MIA]",
            "transition_model.max_cells_per_stage=32",
            "transition_model.schrodinger_bridge.sigma=0.0",
            "transition_model.wes_regularizer.enabled=false",
        ],
    )

    transition = run_transition_model(cfg)
    evaluation = run_evaluation(cfg, transition_output=transition)

    assert transition["ok"] is True
    assert transition["status"] == "complete"
    assert transition["edge"] == "AIS->MIA"
    assert transition["mode"] == "rna_only"
    assert transition["split_summary"]["split_strategy"] == "stagewise_donor_holdout"
    assert transition["split_summary"]["overlap_donors"] == []
    assert evaluation["ok"] is True
    assert evaluation["status"] == "complete"
    assert "sinkhorn" in evaluation["heldout_metrics"]


def test_full_pipeline_threads_reference_and_spatial_outputs_into_transition() -> None:
    cfg = compose_config(
        "default",
        overrides=[
            "data=local",
            "train=smoke",
            "evaluation=baseline",
            "context_model.mode=set_only",
            "transition_model.active_edge=[AAH,AIS]",
            "transition_model.max_cells_per_stage=24",
            "transition_model.schrodinger_bridge.sigma=0.0",
            "transition_model.wes_regularizer.enabled=false",
        ],
    )

    full = run_full(cfg)
    reference = full["steps"]["reference"]
    spatial = full["steps"]["spatial_mapping"]
    context = full["steps"]["context_model"]
    transition = full["steps"]["transition_model"]

    assert full["ok"] is True
    assert transition["reference"]["source_path"] == reference["reference"]["source_path"]
    assert (
        transition["spatial_mapping"]["method"]
        == spatial["spatial_mapping"]["method"]
        == "tangram"
    )
    assert transition["context_model"]["mode"] == context["context_model"]["mode"] == "set_only"
    assert transition["context_diagnostics"]["spatial_mapping_method"] == "tangram"


def test_write_pipeline_scratch_run_records_edge_level_metadata(tmp_path) -> None:
    cfg = compose_config(
        "default",
        overrides=[
            "data=local",
            "train=smoke",
            "evaluation=baseline",
            "context_model.mode=rna_only",
            "transition_model.active_edge=[AAH,AIS]",
            "transition_model.max_cells_per_stage=24",
            "transition_model.schrodinger_bridge.sigma=0.0",
            "transition_model.wes_regularizer.enabled=false",
        ],
    )

    full = run_full(cfg)
    written = write_pipeline_scratch_run(
        cfg, full, notebook_source="StageBridge.ipynb", base_dir=tmp_path
    )

    assert written["ok"] is True
    assert written["run_metadata"]["mode"] == "rna_only"
    assert written["run_metadata"]["stage_edges"] == ["AAH->AIS"]
    assert written["metrics"]["primary_metric"] is not None
