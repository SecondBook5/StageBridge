"""Mission 2 tests for the thin StageBridge notebook contract."""
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
        "## 1. Load Config",
        "## 2. Validate Environment Or Run Context",
        "## 3. Run Pipeline Entry Point(s)",
        "## 4. Display Outputs",
        "## 5. Write Scratch Run Record",
        "## 6. Inspect Registry",
        "## 7. Optionally Promote Milestone",
    ]
    for section in required_sections:
        assert any(section in cell for cell in markdown_cells)

    assert "from stagebridge.pipelines.run_full import run_full" in code
    assert "from stagebridge.results import (" in code
    assert "write_scratch_run(" in code
    assert "read_results_registry()" in code
    assert "promote_current_scratch_run(" in code

    assert not re.search(r"^class\\s+\\w+", code, flags=re.MULTILINE)
    assert "torch.nn.Module" not in code
    assert "optimizer.step(" not in code
    assert "for epoch in" not in code
