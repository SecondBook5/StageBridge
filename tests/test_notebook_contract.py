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

    # The notebook must have numbered steps covering the full pipeline
    required_keywords = [
        "Configure",
        "Validate",
        "Dataset",
        "Reference",
        "Spatial",
        "Context",
        "Transition",
        "Evaluation",
        "Results",
    ]
    combined_md = " ".join(markdown_cells)
    for keyword in required_keywords:
        assert keyword.lower() in combined_md.lower(), f"Missing section keyword: {keyword}"

    # Must use research frontend viz functions
    assert "plot_snrna_preprocessing_frontend(" in code
    assert "plot_spatial_preprocessing_frontend(" in code
    assert "plot_wes_preprocessing_frontend(" in code
    assert "plot_reference_frontend(" in code
    assert "plot_context_frontend(" in code
    assert "plot_transition_frontend(" in code
    assert "plot_transformer_attention_frontend(" in code

    # Must import from stagebridge (not define models inline)
    assert "from stagebridge" in code

    # Must NOT contain inline model definitions or training loops
    assert not re.search(r"^class\s+\w+", code, flags=re.MULTILINE)
    assert "torch.nn.Module" not in code
    assert "optimizer.step(" not in code
    assert "for epoch in" not in code
