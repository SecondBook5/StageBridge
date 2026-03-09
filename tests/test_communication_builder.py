from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from stagebridge.context_model.communication_builder import build_communication_bags
from stagebridge.data.common.schema import LatentCohort, SpatialCohort, WESCohort


def _synthetic_latent() -> LatentCohort:
    latent = np.asarray(
        [
            [2.0, 0.0, 0.0],
            [1.8, 0.2, 0.0],
            [0.0, 1.8, 0.2],
            [0.1, 2.0, 0.0],
            [0.1, 1.9, 0.1],
            [0.0, 0.1, 2.0],
        ],
        dtype=np.float32,
    )
    obs = pd.DataFrame(
        {
            "cell_id": ["c1", "c2", "c3", "c4", "c5", "c6"],
            "sample_id": ["S_AAH", "S_AAH", "S_AIS", "S_AIS", "S_AIS", "S_MIA"],
            "donor_id": ["P1", "P1", "P1", "P1", "P1", "P1"],
            "stage": ["AAH", "AAH", "AIS", "AIS", "AIS", "MIA"],
            "hlca_label": ["AT2", "Basal", "AT2", "Secretory", "Basal", "AT2"],
        }
    )
    return LatentCohort(
        latent=latent,
        obs=obs,
        feature_names=("x0", "x1", "x2"),
        source_path=Path("/tmp/synthetic_latent.h5ad"),
        latent_key="X_hlca",
    )


def _synthetic_spatial() -> SpatialCohort:
    comps = np.asarray(
        [
            [0.7, 0.1, 0.0, 0.0, 0.1, 0.1],
            [0.6, 0.2, 0.0, 0.0, 0.1, 0.1],
            [0.2, 0.6, 0.0, 0.0, 0.1, 0.1],
            [0.3, 0.5, 0.0, 0.0, 0.1, 0.1],
            [0.1, 0.1, 0.0, 0.0, 0.7, 0.1],
            [0.1, 0.1, 0.0, 0.0, 0.6, 0.2],
        ],
        dtype=np.float32,
    )
    coords = np.asarray(
        [
            [0.0, 0.0],
            [0.2, 0.2],
            [1.0, 1.0],
            [1.2, 1.1],
            [2.0, 2.0],
            [2.2, 2.1],
        ],
        dtype=np.float32,
    )
    obs = pd.DataFrame(
        {
            "sample_id": ["S_AAH", "S_AAH", "S_AIS", "S_AIS", "S_MIA", "S_MIA"],
            "donor_id": ["P1", "P1", "P1", "P1", "P1", "P1"],
            "stage": ["AAH", "AAH", "AIS", "AIS", "MIA", "MIA"],
        }
    )
    feature_names = ("AT2", "Basal", "Capillary", "Ciliated", "Fibroblast lineage", "Macrophages")
    return SpatialCohort(compositions=comps, coords=coords, obs=obs, feature_names=feature_names, source_path=Path("/tmp/synthetic_spatial.h5ad"))


def _synthetic_wes() -> WESCohort:
    frame = pd.DataFrame(
        {
            "patient_id": ["P1", "P1", "P1"],
            "stage": ["AAH", "AIS", "MIA"],
            "tmb": [5.0, 6.0, 8.0],
            "kras_mut": [1.0, 1.0, 1.0],
            "egfr_mut": [0.0, 0.0, 0.0],
            "tp53_mut": [1.0, 1.0, 1.0],
            "stk11_mut": [0.0, 0.0, 0.0],
            "keap1_mut": [0.0, 0.0, 0.0],
            "smad4_mut": [0.0, 0.0, 0.0],
            "braf_mut": [0.0, 0.0, 0.0],
        }
    )
    return WESCohort(
        frame=frame,
        feature_columns=("tmb", "kras_mut", "egfr_mut", "tp53_mut", "stk11_mut", "keap1_mut", "smad4_mut", "braf_mut"),
        source_path=Path("/tmp/synthetic_wes.parquet"),
    )


def _synthetic_expression_frame() -> pd.DataFrame:
    genes = [
        "IL1B",
        "IL1R1",
        "IL6",
        "IL6ST",
        "TNF",
        "TNFRSF1A",
        "CXCL12",
        "CXCR4",
        "TGFB1",
        "TGFBR2",
        "AREG",
        "EGFR",
        "EPCAM",
        "KRT8",
        "CEACAM5",
        "MUC1",
        "KRT19",
        "KRT17",
        "ITGB1",
        "VIM",
        "MMP9",
        "OSMR",
        "ERBB2",
        "MET",
    ]
    base = np.linspace(0.1, 2.0, len(genes), dtype=np.float32)
    rows = []
    for idx, cell_id in enumerate(["c1", "c2", "c3", "c4", "c5", "c6"]):
        rows.append({"cell_id": cell_id, **{gene: float(base[g_idx] + 0.2 * idx) for g_idx, gene in enumerate(genes)}})
    return pd.DataFrame(rows)


def test_build_communication_bags_emits_named_token_families() -> None:
    bags, summary = build_communication_bags(
        _synthetic_latent(),
        _synthetic_spatial(),
        wes=_synthetic_wes(),
        expression_frame=_synthetic_expression_frame(),
        active_edges=("AAH->AIS", "AIS->MIA"),
        max_receiver_cells_per_sample=2,
        max_anchor_spots=2,
        max_sender_spots=2,
        max_lr_tokens=4,
        num_distance_rings=2,
        seed=7,
    )

    assert len(bags) == 2
    assert set(summary["edge_label"]) == {"AAH->AIS", "AIS->MIA"}
    first = bags[0].examples[0]
    assert first.sender_embeddings.shape[0] == 2
    assert first.lr_token_features.shape[0] == 4
    assert first.response_token_features.shape[0] >= 1
    assert first.relay_token_features.shape[0] >= 1
    assert first.lr_token_names
    assert first.response_token_names
    assert first.relay_token_names
