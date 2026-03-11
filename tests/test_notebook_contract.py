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

    # The notebook must have sections covering the EA-MIST rescue pipeline
    required_keywords = [
        "Setup",
        "Preprocessing",
        "Reference",
        "Spatial",
        "Bags",
        "Ablation",
        "Results",
        "Transcriptom",
        "Figures",
    ]
    combined_md = " ".join(markdown_cells)
    for keyword in required_keywords:
        assert keyword.lower() in combined_md.lower(), f"Missing section keyword: {keyword}"

    # Must use stagebridge viz and API functions
    assert "configure_research_style" in code
    assert "plot_reference_frontend(" in code
    assert "compose_config(" in code

    # Must use dimensionality reduction methods
    assert "PCA" in code
    assert "UMAP" in code or "umap" in code

    # Must import from stagebridge (not define models inline)
    assert "from stagebridge" in code

    # Must NOT contain inline model definitions or training loops
    assert not re.search(r"^class\s+\w+", code, flags=re.MULTILINE)
    assert "torch.nn.Module" not in code
    assert "optimizer.step(" not in code
    assert "for epoch in" not in code
