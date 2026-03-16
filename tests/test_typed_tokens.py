"""Mission 3 tests for typed niche token construction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from stagebridge.context_model.cell_to_spot_assignment import select_stage_donor_token_context
from stagebridge.context_model.token_builder import build_typed_spot_tokens
from stagebridge.context_model.token_schema import default_typed_token_schema


def test_typed_token_schema_and_builder() -> None:
    raw_feature_names = ["AT2", "Fibroblast lineage", "Macrophages", "Capillary"]
    schema = default_typed_token_schema(raw_feature_names)
    compositions = np.asarray(
        [
            [7.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 3.0, 3.0],
            [1.0, 1.0, 4.0, 4.0],
        ],
        dtype=np.float32,
    )
    coords = np.asarray([[0.0, 0.0], [1.0, 0.5], [2.0, 1.0]], dtype=np.float32)
    obs = pd.DataFrame(
        {
            "donor_id": ["P1", "P2", "P2"],
            "stage": ["AAH", "AIS", "AIS"],
        }
    )

    typed = build_typed_spot_tokens(compositions, coords, obs, raw_feature_names, schema=schema)
    assert typed.tokens.shape == (3, 4)
    assert typed.schema.typed_feature_names == (
        "epithelial",
        "stromal",
        "immune",
        "vascular_program",
    )
    assert np.allclose(typed.tokens.sum(axis=1), 1.0)
    assert typed.tokens[0, 0] > typed.tokens[0, 1]
    assert typed.tokens[2, 2] == typed.tokens[2, 3]

    subset_tokens, subset_coords = select_stage_donor_token_context(
        typed.tokens,
        typed.coords,
        typed.obs,
        donor_id="P2",
        stage="AIS",
        max_spots=8,
    )
    assert subset_tokens.shape == (2, 4)
    assert subset_coords.shape == (2, 2)
