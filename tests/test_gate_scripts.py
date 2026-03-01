import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


def test_check_env_script_writes_machine_readable_output(tmp_path):
    out = tmp_path / "env_check.json"
    cmd = [sys.executable, "scripts/check_env.py", "--output", str(out)]
    subprocess.run(cmd, check=True)

    assert out.exists()
    payload = json.loads(out.read_text())
    assert "ok" in payload
    assert "checks" in payload
    assert "modules" in payload["checks"]
    assert "gpu" in payload["checks"]
    assert "configs" in payload["checks"]


def test_audit_data_script_reports_contracts(tmp_path):
    anndata = pytest.importorskip("anndata")
    n = 100
    d = 8
    rng = np.random.default_rng(2)

    def _make(path: Path) -> Path:
        ad = anndata.AnnData(X=np.zeros((n, 1), dtype=np.float32))
        ad.obs["stage"] = np.array(["Normal", "AAH", "AIS", "MIA", "LUAD"] * (n // 5), dtype=object)
        ad.obs["patient_id"] = np.array([f"D{i%5}" for i in range(n)], dtype=object)
        ad.obs["sample_id"] = np.array([f"S{i%10}" for i in range(n)], dtype=object)
        ad.obsm["X_pca"] = rng.normal(size=(n, d)).astype(np.float32)
        ad.write_h5ad(path)
        return path

    snrna = _make(tmp_path / "snrna.h5ad")
    spatial = _make(tmp_path / "spatial.h5ad")
    hlca = _make(tmp_path / "hlca.h5ad")

    out = tmp_path / "audit.json"
    cmd = [
        sys.executable,
        "scripts/audit_data.py",
        "--snrna",
        str(snrna),
        "--spatial",
        str(spatial),
        "--hlca",
        str(hlca),
        "--output",
        str(out),
    ]
    subprocess.run(cmd, check=True)

    payload = json.loads(out.read_text())
    assert payload["snrna_exists"] is True
    assert payload["spatial_exists"] is True
    assert payload["hlca_exists"] is True
    assert payload["required_obs_columns_ok"] is True
