"""Mission 2 tests for the lightweight scratch and milestone results system."""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from stagebridge.notebook_api import compose_config
from stagebridge.results import (
    find_results_registry_row,
    promote_current_scratch_run,
    read_milestone_index,
    read_promoted_results,
    read_results_registry,
    run_smoke_execution,
    write_scratch_run,
)


def _smoke_cfg():
    return compose_config(
        "default",
        overrides=["data=local", "train=smoke", "evaluation=baseline"],
    )


def test_write_scratch_run_creates_current_workspace(tmp_path: Path) -> None:
    cfg = _smoke_cfg()
    pipeline_output = {
        "steps": {
            "reference": {"ok": True, "status": "structural_stub"},
            "evaluation": {"ok": False, "status": "structural_stub"},
        }
    }

    result = write_scratch_run(
        cfg,
        pipeline_output,
        stdout_text="smoke stdout",
        artifact_sources={"notes/summary.txt": "artifact payload"},
        base_dir=tmp_path,
    )

    scratch_dir = tmp_path / "outputs" / "scratch" / "current"
    assert result["ok"] is True
    assert scratch_dir.is_dir()
    assert sorted(path.name for path in scratch_dir.iterdir()) == [
        "artifacts",
        "metrics.json",
        "resolved_config.yaml",
        "result_card.md",
        "run_metadata.json",
        "stdout.log",
    ]
    assert not (tmp_path / "outputs" / "scratch" / ".staging-current").exists()
    assert (scratch_dir / "artifacts" / "notes" / "summary.txt").read_text(encoding="utf-8") == "artifact payload"


def test_scratch_run_metadata_schema_and_result_card(tmp_path: Path) -> None:
    run_smoke_execution(_smoke_cfg(), base_dir=tmp_path)

    scratch_dir = tmp_path / "outputs" / "scratch" / "current"
    metadata = json.loads((scratch_dir / "run_metadata.json").read_text(encoding="utf-8"))
    metrics = json.loads((scratch_dir / "metrics.json").read_text(encoding="utf-8"))
    result_card = (scratch_dir / "result_card.md").read_text(encoding="utf-8")

    required_metadata_keys = {
        "timestamp",
        "git_commit",
        "git_short_hash",
        "git_branch",
        "experiment_name",
        "mode",
        "stage_edges",
        "seed",
        "split_name",
        "wes_regularizer_enabled",
        "spatial_mapping_method",
        "context_model_mode",
        "notebook_source",
        "status",
    }
    assert set(metadata) == required_metadata_keys
    assert metadata["status"] == "complete"
    assert metrics["ablation_label"] == "smoke_infrastructure"
    assert "## Run Attempt" in result_card
    assert "- Mode: smoke_infrastructure" in result_card
    assert "- Stage edge(s): Normal->AAH, AAH->AIS, AIS->MIA, MIA->LUAD" in result_card
    assert "- What worked:" in result_card
    assert "- What failed:" in result_card
    assert "- Milestone candidate: no" in result_card
    assert "- Next recommended step:" in result_card


def test_registry_updates_for_scratch_attempt(tmp_path: Path) -> None:
    scratch_result = run_smoke_execution(_smoke_cfg(), base_dir=tmp_path)

    registry_rows = read_results_registry(base_dir=tmp_path)
    registry_dir = tmp_path / "results" / "registry"
    assert (registry_dir / "results_registry.csv").exists()
    assert (registry_dir / "milestone_index.csv").exists()
    assert (registry_dir / "promoted_results.yaml").exists()
    assert len(registry_rows) == 1
    row = registry_rows[0]
    assert row["timestamp"] == scratch_result["run_metadata"]["timestamp"]
    assert row["status"] == "complete"
    assert row["promoted"] == "no"
    assert row["scratch_path"] == "outputs/scratch/current"


def test_milestone_promotion_from_scratch_updates_durable_registry(tmp_path: Path) -> None:
    scratch_result = run_smoke_execution(_smoke_cfg(), base_dir=tmp_path)

    promotion = promote_current_scratch_run(
        milestone_id="mission2_smoke_keep",
        summary="Mission 2 smoke promotion",
        importance_level="smoke",
        promotion_slots=["best_set_only"],
        interpretation_notes="Infrastructure proof only.",
        next_step_recommendation="Use this as the notebook/results-system baseline.",
        base_dir=tmp_path,
    )

    milestone_dir = Path(promotion.milestone_path)
    assert milestone_dir == tmp_path / "results" / "milestones" / "mission2_smoke_keep"
    assert sorted(path.name for path in milestone_dir.iterdir()) == [
        "artifacts",
        "metrics_snapshot.json",
        "milestone_summary.md",
        "resolved_config.yaml",
        "result_card.md",
        "run_metadata.json",
        "source_run.txt",
    ]

    milestone_rows = read_milestone_index(base_dir=tmp_path)
    assert len(milestone_rows) == 1
    assert milestone_rows[0]["milestone_id"] == "mission2_smoke_keep"
    assert milestone_rows[0]["importance_level"] == "smoke"

    registry_row = find_results_registry_row(
        scratch_result["run_metadata"]["timestamp"],
        base_dir=tmp_path,
    )
    assert registry_row is not None
    assert registry_row["promoted"] == "yes"
    assert registry_row["status"] == "promoted"
    assert registry_row["milestone_id"] == "mission2_smoke_keep"

    scratch_metadata = json.loads(
        (tmp_path / "outputs" / "scratch" / "current" / "run_metadata.json").read_text(encoding="utf-8")
    )
    assert scratch_metadata["status"] == "promoted"

    promoted_results = read_promoted_results(base_dir=tmp_path)
    assert promoted_results["latest_promoted"]["milestone_id"] == "mission2_smoke_keep"
    assert promoted_results["best_set_only"]["milestone_id"] == "mission2_smoke_keep"

    promoted_yaml = yaml.safe_load(
        (tmp_path / "results" / "registry" / "promoted_results.yaml").read_text(encoding="utf-8")
    )
    assert promoted_yaml["best_set_only"]["milestone_id"] == "mission2_smoke_keep"
