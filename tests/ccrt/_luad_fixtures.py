"""Shared source-faithful LUAD fixtures for Milestone-9 tests.

Builds a tiny on-disk source tree (modality manifest JSON + CSV feature / context
/ niche tables) plus a spatial loader, mirroring the verified LUAD multimodal
layout (snRNA reference, Visium spatial spots, Tangram deconvolved context).
Deterministic; no external data, no giant h5ad objects, no biological claims.
"""

from __future__ import annotations

import dataclasses
import json
import random
from pathlib import Path

from stagebridge.ccrt.adapters.luad import (
    LUADModalityRecord,
    LUADSpatialLoader,
    LUADSpatialSpot,
    build_reference_luad_adapter_config,
)

# (spot_id, donor, sample, section, niche_id, lesion_id, stage_label, canonical_state, x, y)
RECEIVER_LAYOUT = [
    ("P3_s1", "P3", "GSM_P3", "GSM_P3_S1", "n_P3_1", "l_P3", "Normal", "normal", 0.0, 0.0),
    ("P3_s2", "P3", "GSM_P3", "GSM_P3_S1", "n_P3_2", "l_P3", "AAH", "aah", 10.0, 0.0),
    ("P4_s1", "P4", "GSM_P4", "GSM_P4_S1", "n_P4_1", "l_P4", "AAH", "aah", 0.0, 0.0),
    ("P4_s2", "P4", "GSM_P4", "GSM_P4_S1", "n_P4_2", "l_P4", "AIS", "ais", 12.0, 0.0),
    ("P10_s1", "P10", "GSM_P10", "GSM_P10_S1", "n_P10_1", "l_P10a", "AIS", "ais", 0.0, 0.0),
    ("P10_s2", "P10", "GSM_P10", "GSM_P10_S1", "n_P10_2", "l_P10b", "MIA", "mia", 8.0, 0.0),
    ("P10_s3", "P10", "GSM_P10", "GSM_P10_S1", "n_P10_3", "l_P10c", "LUAD", "invasive_luad", 16.0, 0.0),
]

# Deconvolved context components hosted at spots: (spot_id, source_sender_label).
# One backend x spot x type = one component (never abundance-replicated).
CONTEXT_LAYOUT = []
for _row in RECEIVER_LAYOUT:
    _spot = _row[0]
    CONTEXT_LAYOUT.append((_spot, "AT2"))
    CONTEXT_LAYOUT.append((_spot, "Macrophages"))

DONORS = ("P3", "P4", "P10")


def _niche_ids():
    return [row[4] for row in RECEIVER_LAYOUT]


def _lesion_ids():
    # unique, preserving order
    seen, out = set(), []
    for row in RECEIVER_LAYOUT:
        if row[5] not in seen:
            seen.add(row[5])
            out.append(row[5])
    return out


def _write_matrix(path: Path, id_col: str, ids, feature_cols, rng) -> None:
    lines = [id_col + "," + ",".join(feature_cols)]
    for oid in ids:
        lines.append(oid + "," + ",".join(f"{rng.random():.4f}" for _ in feature_cols))
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_context(path: Path, feature_cols, rng) -> None:
    header = ["spot_id", "sender_context_type", "abundance", *feature_cols]
    lines = [",".join(header)]
    for spot_id, label in CONTEXT_LAYOUT:
        abundance = f"{rng.random():.4f}"
        feats = ",".join(f"{rng.random():.4f}" for _ in feature_cols)
        lines.append(f"{spot_id},{label},{abundance},{feats}")
    path.write_text("\n".join(lines), encoding="utf-8")


def build_luad_source_fixture(root: Path, *, seed: int = 0):
    """Create a source-faithful fixture tree; return an adapter config for it."""
    rng = random.Random(seed)
    root.mkdir(parents=True, exist_ok=True)

    (root / "luad_source_manifest.json").write_text(
        json.dumps(
            {
                "dataset_commit": "fixturecommit",
                "layout_version": "luad-source-2025-snrna-visium-tangram-hlca-v1",
                "platforms": ["visium", "snrna"],
                "observation_units": {"visium": "spot", "snrna": "cell"},
                "donor_column": "donor_id",
                "patient_column": "patient_id",
                "sample_column": "sample_id",
                "section_column": "section_id",
                "stage_column": "stage",
                "annotation_columns": ["stage", "cell_type"],
                "coordinate_columns": ["x_spatial_microns", "y_spatial_microns"],
                "coordinate_units": {"visium": "microns"},
                "stage_labels": ["Normal", "AAH", "AIS", "MIA", "LUAD"],
                "annotation_labels": [
                    "AT2", "Basal", "Capillary", "Ciliated", "Fibroblast lineage",
                    "Macrophages", "Mast cells", "Secretory", "T cell lineage",
                ],
                "context_backends": ["tangram"],
                "modality_relationships": {
                    "snrna__visium": "same_donor",
                },
            }
        ),
        encoding="utf-8",
    )

    cfg = build_reference_luad_adapter_config(root)

    _write_matrix(
        root / "receiver.csv", "niche_id", _niche_ids(),
        cfg.receiver_feature_block.feature_ids, rng,
    )
    _write_matrix(
        root / "semantic.csv", "niche_id", _niche_ids(),
        cfg.semantic_feature_block.feature_ids, rng,
    )
    _write_matrix(
        root / "regulatory.csv", "lesion_id", _lesion_ids(),
        cfg.regulatory_feature_block.feature_ids, rng,
    )
    _write_context(root / "context.csv", cfg.context_backend.feature_ids, rng)

    return dataclasses.replace(
        cfg,
        receiver_feature_block=dataclasses.replace(
            cfg.receiver_feature_block, source_path="receiver.csv"
        ),
        semantic_feature_block=dataclasses.replace(
            cfg.semantic_feature_block, source_path="semantic.csv"
        ),
        regulatory_feature_block=dataclasses.replace(
            cfg.regulatory_feature_block, source_path="regulatory.csv"
        ),
        context_backend=dataclasses.replace(
            cfg.context_backend, source_path="context.csv"
        ),
    )


class FixtureLUADSpatialLoader(LUADSpatialLoader):
    """Deterministic spatial loader for the fixture."""

    def load_spots(self, config, audit):
        spots = []
        for (
            spot_id, donor, sample, section, niche_id, lesion_id,
            stage_label, state, x, y,
        ) in RECEIVER_LAYOUT:
            spots.append(
                LUADSpatialSpot(
                    spot_id=spot_id,
                    donor_id=donor,
                    sample_id=sample,
                    section_id=section,
                    platform="visium",
                    x_microns=x,
                    y_microns=y,
                    observation_unit="spot",
                    niche_id=niche_id,
                    lesion_id=lesion_id,
                    canonical_receiver_state_id=state,
                    source_stage_label=stage_label,
                )
            )
        return spots

    def load_modalities(self, config, audit):
        return [
            LUADModalityRecord(
                modality_id="snrna",
                accession="GSE308103",
                platform="snrna",
                observation_unit="cell",
                donor_ids=DONORS,
                sample_ids=("GSM_P3", "GSM_P4", "GSM_P10"),
            ),
            LUADModalityRecord(
                modality_id="visium",
                accession="GSE307534",
                platform="visium",
                observation_unit="spot",
                donor_ids=DONORS,
                sample_ids=("GSM_P3", "GSM_P4", "GSM_P10"),
            ),
        ]
