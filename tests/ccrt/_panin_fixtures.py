"""Shared source-faithful PanIN fixtures for Milestone-8 tests.

Builds a tiny on-disk source tree (manifest + CSV feature matrices) plus a
spatial loader, mirroring the verified Xenium cell-resolved layout. Deterministic;
no external source repository required.
"""

from __future__ import annotations

import dataclasses
import json
import random
from pathlib import Path

from stagebridge.ccrt.adapters.panin import (
    PanINSpatialLoader,
    PanINSpatialObservation,
    build_reference_panin_adapter_config,
)

# (obs_id, donor, receiver_state)
RECEIVER_LAYOUT = [
    ("r0", "1131", "normal_duct"),
    ("r1", "1131", "normal_duct"),
    ("r2", "1131", "low_grade_panin"),
    ("r3", "1132", "normal_duct"),
    ("r4", "1132", "low_grade_panin"),
    ("r5", "1132", "high_grade_panin"),
    ("r6", "1134", "low_grade_panin"),
    ("r7", "1134", "high_grade_panin"),
]
# (obs_id, donor, canonical_context_type)
SENDER_LAYOUT = [
    ("s0", "1131", "caf"),
    ("s1", "1131", "immune"),
    ("s2", "1132", "caf"),
    ("s3", "1132", "mycaf"),
    ("s4", "1134", "icaf"),
]


def _write_csv(path: Path, obs_ids, cols, rng) -> None:
    lines = ["cell_id," + ",".join(cols)]
    for oid in obs_ids:
        lines.append(oid + "," + ",".join(f"{rng.random():.4f}" for _ in cols))
    path.write_text("\n".join(lines), encoding="utf-8")


def build_panin_source_fixture(root: Path, *, seed: int = 0):
    """Create a source-faithful fixture tree; return an adapter config for it."""
    rng = random.Random(seed)
    root.mkdir(parents=True, exist_ok=True)

    (root / "panin_source_manifest.json").write_text(
        json.dumps(
            {
                "repository_commit": "fixturecommit",
                "layout_version": "panin-source-2024-xenium-cogaps-v1",
                "platforms": ["xenium"],
                "observation_units": {"xenium": "cell"},
                "donor_column": "donor_id",
                "sample_column": "sample_id",
                "section_column": "section_id",
                "stage_column": "cell_type_confirmed",
                "annotation_columns": ["cell_type_confirmed", "CAF_subtype"],
                "coordinate_columns": ["x_centroid", "y_centroid"],
                "coordinate_units": {"xenium": "microns"},
                "stage_labels": [
                    "normal epithelium",
                    "low_grade_PanIN",
                    "high_grade_PanIN",
                ],
                "annotation_labels": ["caf", "immune", "mycaf", "icaf", "apcaf"],
                "modality_relationships": {"xenium_visium": "unmatched"},
            }
        ),
        encoding="utf-8",
    )

    cfg = build_reference_panin_adapter_config(root)
    panel = cfg.receiver_feature_block.feature_ids
    all_ids = [o[0] for o in RECEIVER_LAYOUT] + [s[0] for s in SENDER_LAYOUT]
    _write_csv(root / "recv.csv", all_ids, panel, rng)
    _write_csv(root / "send.csv", all_ids, panel, rng)
    _write_csv(root / "sem.csv", all_ids, [f"Pattern_{i}" for i in range(1, 9)], rng)

    return dataclasses.replace(
        cfg,
        receiver_feature_block=dataclasses.replace(
            cfg.receiver_feature_block, source_path="recv.csv"
        ),
        sender_feature_block=dataclasses.replace(
            cfg.sender_feature_block, source_path="send.csv"
        ),
        semantic_feature_block=dataclasses.replace(
            cfg.semantic_feature_block, source_path="sem.csv"
        ),
    )


class FixtureSpatialLoader(PanINSpatialLoader):
    """Deterministic spatial loader for the fixture."""

    def load(self, config, audit):
        obs = []
        for i, (oid, donor, state) in enumerate(RECEIVER_LAYOUT):
            obs.append(
                PanINSpatialObservation(
                    observation_id=oid, donor_id=donor, sample_id=f"PanIN{donor}",
                    section_id=f"PanIN{donor}_S1", platform="xenium",
                    observation_unit="cell", x_microns=float(i), y_microns=float(i * 2),
                    source_annotation=state, canonical_receiver_state_id=state,
                )
            )
        for j, (oid, donor, ctx) in enumerate(SENDER_LAYOUT):
            obs.append(
                PanINSpatialObservation(
                    observation_id=oid, donor_id=donor, sample_id=f"PanIN{donor}",
                    section_id=f"PanIN{donor}_S1", platform="xenium",
                    observation_unit="cell", x_microns=float(j) + 0.5, y_microns=float(j),
                    source_annotation=ctx, canonical_context_type_id=ctx,
                )
            )
        return obs
