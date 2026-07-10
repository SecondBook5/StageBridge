"""Tests for PanIN feature ingestion and registration."""

from __future__ import annotations

import pytest
import torch

from stagebridge.ccrt.adapters.panin import (
    PanINFeatureBlockConfig,
    load_panin_feature_block,
    register_panin_feature_spaces,
)
from stagebridge.ccrt.contracts import CCRTValidationError


def _write(path, cols, rows):
    lines = ["cell_id," + ",".join(cols)]
    for oid, vals in rows:
        lines.append(oid + "," + ",".join(str(v) for v in vals))
    path.write_text("\n".join(lines), encoding="utf-8")


def block_cfg(path, feature_ids, role="reconstruction", metric=None):
    return PanINFeatureBlockConfig(
        feature_space_id="fs", role=role, source_path=path.name,
        observation_id_key="cell_id", feature_ids=feature_ids, metric=metric,
    )


def test_load_preserves_feature_order(tmp_path):
    p = tmp_path / "f.csv"
    _write(p, ["GENE_A", "GENE_B"], [("c0", [1.0, 2.0]), ("c1", [3.0, 4.0])])
    block = load_panin_feature_block(block_cfg(p, ("GENE_B", "GENE_A")), source_root=tmp_path)
    # order follows the config (B then A)
    assert torch.allclose(block.values[0], torch.tensor([2.0, 1.0], dtype=torch.float64))
    assert block.observation_ids == ("c0", "c1")


def test_missing_feature_fails(tmp_path):
    p = tmp_path / "f.csv"
    _write(p, ["GENE_A"], [("c0", [1.0])])
    with pytest.raises(CCRTValidationError):
        load_panin_feature_block(block_cfg(p, ("GENE_A", "GENE_MISSING")), source_root=tmp_path)


def test_duplicate_observation_ids_fail(tmp_path):
    p = tmp_path / "f.csv"
    _write(p, ["GENE_A"], [("c0", [1.0]), ("c0", [2.0])])
    with pytest.raises(CCRTValidationError):
        load_panin_feature_block(block_cfg(p, ("GENE_A",)), source_root=tmp_path)


def test_nonfinite_values_fail(tmp_path):
    p = tmp_path / "f.csv"
    _write(p, ["GENE_A"], [("c0", ["nan"]), ("c1", [2.0])])
    with pytest.raises(CCRTValidationError):
        load_panin_feature_block(block_cfg(p, ("GENE_A",)), source_root=tmp_path)


def test_metadata_feature_id_rejected(tmp_path):
    p = tmp_path / "f.csv"
    _write(p, ["stage", "GENE_A"], [("c0", [1.0, 2.0])])
    # "stage" is a metadata token -> rejected before load
    with pytest.raises(CCRTValidationError):
        load_panin_feature_block(block_cfg(p, ("stage", "GENE_A")), source_root=tmp_path)


def test_coordinate_feature_id_rejected(tmp_path):
    p = tmp_path / "f.csv"
    _write(p, ["x_centroid", "GENE_A"], [("c0", [1.0, 2.0])])
    with pytest.raises(CCRTValidationError):
        load_panin_feature_block(block_cfg(p, ("x_centroid", "GENE_A")), source_root=tmp_path)


def test_unsupported_format_fails(tmp_path):
    p = tmp_path / "f.rds"
    p.write_bytes(b"binary")
    with pytest.raises(CCRTValidationError):
        load_panin_feature_block(block_cfg(p, ("GENE_A",)), source_root=tmp_path)


def test_registration_and_semantic_spec(tmp_path):
    p = tmp_path / "sem.csv"
    _write(p, ["Pattern_1", "Pattern_2"], [("c0", [0.1, 0.2]), ("c1", [0.3, 0.4])])
    block = load_panin_feature_block(
        block_cfg(p, ("Pattern_1", "Pattern_2"), role="semantic", metric="squared_euclidean"),
        source_root=tmp_path,
    )
    reg = register_panin_feature_spaces([block])
    assert reg.contains("fs")
    reg.validate_tensor("fs", block.values.to(torch.float32), expected_role="semantic")
