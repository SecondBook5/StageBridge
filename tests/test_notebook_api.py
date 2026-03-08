from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from stagebridge.notebook_api import build_latent_comparison_table, build_seeded_mode_summary_table
from stagebridge.viz.research_frontend import plot_latent_comparison_frontend


def test_build_latent_comparison_table_and_plot() -> None:
    backend_results = {
        "hlca": {
            "reference": {
                "reference": {
                    "latent_shape": [96, 16],
                    "provenance": {"mode": "loaded"},
                    "source_path": "/tmp/hlca.h5ad",
                }
            },
            "evaluation": {
                "heldout_metrics": {"sinkhorn": 10.5},
                "calibration": {"mean_abs_shift_error": 1.2},
                "biology_summary": {
                    "dominant_increase_group": "stromal",
                    "dominant_decrease_group": "vascular_program",
                },
            },
        },
        "pca": {
            "reference": {
                "reference": {
                    "latent_shape": [96, 12],
                    "provenance": {"mode": "fit"},
                    "source_path": "/tmp/raw.h5ad",
                }
            },
            "evaluation": {
                "heldout_metrics": {"sinkhorn": 12.0},
                "calibration": {"mean_abs_shift_error": 2.4},
                "biology_summary": {
                    "dominant_increase_group": "stromal",
                    "dominant_decrease_group": "vascular_program",
                },
            },
        },
    }

    table = build_latent_comparison_table(backend_results)

    assert table.iloc[0]["backend"] == "hlca"
    assert table.iloc[0]["dominant_increase_group"] == "stromal"
    assert table.iloc[1]["provenance_mode"] == "fit"

    fig = plot_latent_comparison_frontend(table, edge="AAH->AIS", mode="set_only")
    assert fig is not None
    assert len(fig.axes) == 2


def test_build_seeded_mode_summary_table_aggregates_metrics_and_biology() -> None:
    seeded = {
        11: {
            "pooled": {
                "transition_model": {"split_summary": {"split_strategy": "paired_donor_holdout"}},
                "evaluation": {
                    "heldout_metrics": {"sinkhorn": 10.0},
                    "calibration": {"mean_abs_shift_error": 1.0},
                    "context_sensitivity": {"context_sensitivity_delta": 0.2},
                    "biology_summary": {
                        "dominant_increase_group": "stromal",
                        "dominant_decrease_group": "vascular_program",
                    },
                },
            },
            "set_only": {
                "transition_model": {"split_summary": {"split_strategy": "paired_donor_holdout"}},
                "evaluation": {
                    "heldout_metrics": {"sinkhorn": 9.0},
                    "calibration": {"mean_abs_shift_error": 0.8},
                    "context_sensitivity": {"context_sensitivity_delta": 0.3},
                    "biology_summary": {
                        "dominant_increase_group": "stromal",
                        "dominant_decrease_group": "vascular_program",
                    },
                },
            },
        },
        17: {
            "pooled": {
                "transition_model": {"split_summary": {"split_strategy": "paired_donor_holdout"}},
                "evaluation": {
                    "heldout_metrics": {"sinkhorn": 12.0},
                    "calibration": {"mean_abs_shift_error": 1.2},
                    "context_sensitivity": {"context_sensitivity_delta": 0.1},
                    "biology_summary": {
                        "dominant_increase_group": "stromal",
                        "dominant_decrease_group": "vascular_program",
                    },
                },
            },
            "set_only": {
                "transition_model": {"split_summary": {"split_strategy": "paired_donor_holdout"}},
                "evaluation": {
                    "heldout_metrics": {"sinkhorn": 8.0},
                    "calibration": {"mean_abs_shift_error": 0.7},
                    "context_sensitivity": {"context_sensitivity_delta": 0.4},
                    "biology_summary": {
                        "dominant_increase_group": "stromal",
                        "dominant_decrease_group": "vascular_program",
                    },
                },
            },
        },
    }

    table = build_seeded_mode_summary_table(seeded)

    assert table.iloc[0]["mode"] == "set_only"
    assert table.iloc[0]["n_seeds"] == 2
    assert table.iloc[0]["dominant_increase_group"] == "stromal"
    assert round(float(table.loc[table["mode"] == "pooled", "sinkhorn_mean"].iloc[0]), 3) == 11.0
