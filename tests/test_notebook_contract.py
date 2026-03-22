"""Notebook contract tests for the numbered StageBridge research frontend."""

from __future__ import annotations

import json
from pathlib import Path


def test_stagebridge_notebook_is_only_active_top_level_notebook() -> None:
    notebooks = sorted(
        path.name for path in Path(".").glob("*.ipynb") if not path.name.startswith(".")
    )
    # Only canonical V1 comprehensive notebook should remain after cleanup
    assert notebooks == ["StageBridge_V1.ipynb"]


def test_stagebridge_notebook_is_thin_orchestration_surface() -> None:
    notebook_path = Path("StageBridge_V1.ipynb")
    if not notebook_path.exists():
        # Skip if notebook doesn't exist (fallback for legacy test)
        return
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
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

    # The notebook must have sections covering the V1 pipeline
    required_keywords = [
        "Ablation",  # Ablation studies
        "Comparison",  # Model comparisons
        "Training",  # Model training
        "Evaluation",  # Evaluation section
        "Data",  # Data loading/generation
    ]
    combined_md = " ".join(markdown_cells)
    for keyword in required_keywords:
        assert keyword.lower() in combined_md.lower(), f"Missing section keyword: {keyword}"

    # Must import from stagebridge (not define models inline)
    assert "from stagebridge" in code or "import stagebridge" in code

    # Must NOT contain inline model definitions
    assert "torch.nn.Module" not in code
