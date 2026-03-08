"""Notebook contract tests for the numbered StageBridge research frontend."""
from __future__ import annotations

import json
from pathlib import Path
import re


def test_stagebridge_notebook_is_only_active_top_level_notebook() -> None:
    notebooks = sorted(path.name for path in Path(".").glob("*.ipynb"))
    assert notebooks == ["StageBridge.ipynb"]


def test_stagebridge_notebook_is_thin_orchestration_surface() -> None:
    notebook = json.loads(Path("StageBridge.ipynb").read_text(encoding="utf-8"))
    markdown_cells = [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "markdown"
    ]
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )

    required_sections = [
        "## Step 1. Configure Run",
        "## Step 2. Validate Environment and Asset Paths",
        "## Step 3. Dataset Preprocessing and Cohort Preview",
        "## Step 4. HLCA Reference Latent Mapping",
        "## Step 5. HLCA Alignment and QC",
        "## Step 6. Spatial Provider Rebuild and Comparison",
        "## Step 7. Provider Winner Selection",
        "## Step 8. Typed Token Construction",
        "## Step 9. Context Model and Transformer Diagnostics",
        "## Step 10. Transition Fit",
        "## Step 11. Evaluation and Biology Readout",
        "## Step 12. Results Writeout and Registry",
    ]
    for section in required_sections:
        assert any(section in cell for cell in markdown_cells)

    assert "from stagebridge.pipelines.run_reference import run_reference" in code
    assert "from stagebridge.pipelines.run_spatial_mapping import run_spatial_mapping" in code
    assert "from stagebridge.pipelines.run_context_model import run_context_model" in code
    assert "from stagebridge.pipelines.run_transition_model import run_transition_model" in code
    assert "from stagebridge.pipelines.run_evaluation import run_evaluation" in code
    assert "from stagebridge.results import (" in code
    assert "write_pipeline_scratch_run(" in code or "write_scratch_run(" in code
    assert "run_data_preprocessing_overview(" in code
    assert "build_dataset_preprocessing_table(" in code
    assert "plot_snrna_preprocessing_frontend(" in code
    assert "plot_spatial_preprocessing_frontend(" in code
    assert "plot_wes_preprocessing_frontend(" in code
    assert "reference_output = run_reference(cfg)" in code
    assert "build_reference_evaluation_table(" in code
    assert "plot_reference_frontend(" in code
    assert "run_spatial_provider_ladder(" in code
    assert "run_provider_benchmark(" in code
    assert "apply_selected_provider(" in code
    assert "plot_provider_benchmark_frontend(" in code
    assert "plot_spatial_provider_abundance_frontend(" in code
    assert "spatial_output = run_spatial_mapping(" in code
    assert "plot_spatial_mapping_frontend(" in code
    assert "plot_context_frontend(" in code
    assert "context_output = run_context_model(" in code
    assert "transition_output = run_transition_model(" in code
    assert "evaluation_output = run_evaluation(" in code
    assert "plot_transition_frontend(" in code
    assert "build_step_status_table(" in code
    assert "read_results_registry()" in code
    assert "mission2_smoke" not in code

    assert not re.search(r"^class\\s+\\w+", code, flags=re.MULTILINE)
    assert "torch.nn.Module" not in code
    assert "optimizer.step(" not in code
    assert "for epoch in" not in code
    assert code.count("build_transition_summary_table(") == 1
