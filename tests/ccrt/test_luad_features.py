"""Tests for LUAD feature ingestion and registration."""

from __future__ import annotations

import pytest
import torch

from stagebridge.ccrt.adapters.luad import (
    LUADFeatureBlockConfig,
    load_luad_feature_block,
    register_luad_feature_spaces,
)
from stagebridge.ccrt.contracts import CCRTValidationError


def _write(path, id_col, cols, rows):
    lines = [id_col + "," + ",".join(cols)]
    for oid, vals in rows:
        lines.append(oid + "," + ",".join(str(v) for v in vals))
    path.write_text("\n".join(lines), encoding="utf-8")


def block_cfg(path, feature_ids, role="reconstruction", metric=None, id_key="niche_id"):
    return LUADFeatureBlockConfig(
        feature_space_id="fs", role=role, source_path=path.name,
        observation_id_key=id_key, feature_ids=feature_ids, metric=metric,
    )


def test_load_preserves_feature_order(tmp_path):
    p = tmp_path / "f.csv"
    _write(p, "niche_id", ["A", "B"], [("n0", [1.0, 2.0]), ("n1", [3.0, 4.0])])
    block = load_luad_feature_block(block_cfg(p, ("B", "A")), source_root=tmp_path)
    assert torch.allclose(block.values[0], torch.tensor([2.0, 1.0], dtype=torch.float64))
    assert block.observation_ids == ("n0", "n1")


def test_missing_feature_fails(tmp_path):
    p = tmp_path / "f.csv"
    _write(p, "niche_id", ["A"], [("n0", [1.0])])
    with pytest.raises(CCRTValidationError):
        load_luad_feature_block(block_cfg(p, ("A", "MISSING")), source_root=tmp_path)


def test_duplicate_observation_ids_fail(tmp_path):
    p = tmp_path / "f.csv"
    _write(p, "niche_id", ["A"], [("n0", [1.0]), ("n0", [2.0])])
    with pytest.raises(CCRTValidationError):
        load_luad_feature_block(block_cfg(p, ("A",)), source_root=tmp_path)


def test_nonfinite_values_fail(tmp_path):
    p = tmp_path / "f.csv"
    _write(p, "niche_id", ["A"], [("n0", ["nan"]), ("n1", [2.0])])
    with pytest.raises(CCRTValidationError):
        load_luad_feature_block(block_cfg(p, ("A",)), source_root=tmp_path)


def test_metadata_feature_id_rejected(tmp_path):
    p = tmp_path / "f.csv"
    _write(p, "niche_id", ["stage", "A"], [("n0", [1.0, 2.0])])
    with pytest.raises(CCRTValidationError):
        load_luad_feature_block(block_cfg(p, ("stage", "A")), source_root=tmp_path)


def test_lesion_id_feature_id_rejected(tmp_path):
    p = tmp_path / "f.csv"
    _write(p, "niche_id", ["lesion_id", "A"], [("n0", [1.0, 2.0])])
    with pytest.raises(CCRTValidationError):
        load_luad_feature_block(block_cfg(p, ("lesion_id", "A")), source_root=tmp_path)


def test_backend_feature_id_rejected(tmp_path):
    p = tmp_path / "f.csv"
    _write(p, "niche_id", ["backend", "A"], [("n0", [1.0, 2.0])])
    with pytest.raises(CCRTValidationError):
        load_luad_feature_block(block_cfg(p, ("backend", "A")), source_root=tmp_path)


def test_coordinate_feature_id_rejected(tmp_path):
    p = tmp_path / "f.csv"
    _write(p, "niche_id", ["x_spatial_microns", "A"], [("n0", [1.0, 2.0])])
    with pytest.raises(CCRTValidationError):
        load_luad_feature_block(block_cfg(p, ("x_spatial_microns", "A")), source_root=tmp_path)


def test_unsupported_format_fails(tmp_path):
    p = tmp_path / "f.parquet"
    p.write_bytes(b"binary")
    with pytest.raises(CCRTValidationError):
        load_luad_feature_block(block_cfg(p, ("A",)), source_root=tmp_path)


def test_registration_and_semantic_spec(tmp_path):
    p = tmp_path / "sem.csv"
    _write(p, "niche_id", ["hlca_a", "hlca_b"], [("n0", [0.1, 0.2]), ("n1", [0.3, 0.4])])
    block = load_luad_feature_block(
        block_cfg(p, ("hlca_a", "hlca_b"), role="semantic", metric="squared_euclidean"),
        source_root=tmp_path,
    )
    reg = register_luad_feature_spaces([block])
    assert reg.contains("fs")
    reg.validate_tensor("fs", block.values.to(torch.float32), expected_role="semantic")


def test_regulatory_block_loads(tmp_path):
    p = tmp_path / "reg.csv"
    _write(p, "lesion_id", ["evo_tmb", "evo_purity"], [("l0", [0.5, 0.9])])
    block = load_luad_feature_block(
        block_cfg(p, ("evo_tmb", "evo_purity"), role="regulatory", id_key="lesion_id"),
        source_root=tmp_path,
    )
    assert block.spec.role == "regulatory"
    assert block.observation_ids == ("l0",)
