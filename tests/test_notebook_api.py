from __future__ import annotations

import matplotlib
from omegaconf import OmegaConf

matplotlib.use("Agg")

from stagebridge.notebook_api import (
    apply_selected_provider,
    build_latent_comparison_table,
    build_dataset_preprocessing_table,
    build_provider_benchmark_table,
    build_reference_evaluation_table,
    build_seeded_mode_summary_table,
    build_spatial_provider_agreement_table,
    build_spatial_provider_metric_table,
    build_spatial_provider_table,
    run_provider_benchmark,
    run_spatial_provider_ladder,
)
from stagebridge.viz.research_frontend import (
    plot_latent_comparison_frontend,
    plot_provider_benchmark_frontend,
    plot_snrna_preprocessing_frontend,
    plot_spatial_preprocessing_frontend,
    plot_spatial_provider_abundance_frontend,
    plot_spatial_provider_comparison_frontend,
    plot_spatial_provider_maps_frontend,
    plot_wes_preprocessing_frontend,
)


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


def test_build_reference_evaluation_table() -> None:
    reference_output = {
        "reference": {
            "diagnostics": {
                "stage_preservation": {
                    "probe": {
                        "logreg_accuracy": 0.8,
                        "balanced_accuracy": 0.75,
                        "chance_accuracy": 0.33,
                    },
                    "centroid_distances": {"AAH->AIS": 1.4, "AIS->MIA": 2.2},
                },
                "donor_leakage": {
                    "logreg_accuracy": 0.42,
                    "chance_accuracy": 0.25,
                },
                "gene_overlap": {
                    "reference_query_overlap_fraction": 0.62,
                    "missing_gene_fraction": 0.38,
                },
                "label_neighborhood": {
                    "mean_neighbor_label_agreement": 0.71,
                    "n_labeled_cells": 120,
                },
                "stage_label_alignment": {
                    "rows": ["AAH", "AIS"],
                    "cols": ["AT2", "Basal"],
                },
                "alignment_gate": {
                    "status": "weak_pass",
                    "recommended_action": "keep_as_optional",
                },
            },
            "label_transfer": {"coverage": 0.95},
        }
    }

    table = build_reference_evaluation_table(reference_output)

    assert "stage_probe_accuracy" in table["metric"].tolist()
    assert (
        round(
            float(table.loc[table["metric"] == "mean_stage_centroid_distance", "value"].iloc[0]), 3
        )
        == 1.8
    )


def test_dataset_preprocessing_tables_and_figures() -> None:
    data_output = {
        "snrna": {
            "obs": __import__("pandas").DataFrame(
                {
                    "stage": ["AAH", "AIS", "AIS"],
                    "donor_id": ["P1", "P2", "P3"],
                    "sample_id": ["S1", "S2", "S3"],
                }
            ),
            "latent": [[0.0, 0.0, 0.2], [1.0, 0.5, 0.1], [1.2, 0.6, 0.2]],
            "pca_embedding": [[0.0, 0.0], [1.0, 0.5], [1.2, 0.6]],
            "umap_embedding": [[0.1, 0.0], [0.9, 0.4], [1.3, 0.7]],
            "source_path": "/tmp/snrna.h5ad",
            "n_cells": 3,
            "n_genes": 100,
            "n_donors": 3,
            "n_samples": 3,
            "stage_counts": {"AAH": 1, "AIS": 2},
            "sample_stage_counts": __import__("pandas").DataFrame(
                {"AAH": [1, 0], "AIS": [0, 2]}, index=["S1", "S2"]
            ),
            "top_labels": [("AT2", 2), ("Basal", 1)],
        },
        "spatial": {
            "obs": __import__("pandas").DataFrame(
                {"stage": ["AAH", "AIS"], "donor_id": ["P1", "P2"], "sample_id": ["V1", "V2"]}
            ),
            "coords": [[0.0, 0.0], [1.0, 1.0]],
            "source_path": "/tmp/spatial.h5ad",
            "n_spots": 2,
            "n_genes": 50,
            "n_donors": 2,
            "n_samples": 2,
            "stage_counts": {"AAH": 1, "AIS": 1},
            "feature_panel": __import__("pandas").DataFrame(
                {"EPCAM": [1.0, 0.1], "COL1A1": [0.2, 0.9], "PTPRC": [0.5, 0.3], "VWF": [0.1, 0.6]}
            ),
            "feature_panel_genes": ["EPCAM", "COL1A1", "PTPRC", "VWF"],
            "feature_panel_roles": {
                "EPCAM": "epithelial",
                "COL1A1": "stromal",
                "PTPRC": "immune",
                "VWF": "vascular_program",
            },
            "feature_panel_uses_proxy_genes": False,
        },
        "wes": {
            "frame": __import__("pandas").DataFrame(
                {
                    "patient_id": ["P1", "P2"],
                    "stage": ["AAH", "AIS"],
                    "tmb": [10.0, 20.0],
                    "kras_mut": [0.0, 1.0],
                    "egfr_mut": [0.0, 0.0],
                    "tp53_mut": [1.0, 1.0],
                }
            ),
            "feature_columns": ["tmb", "kras_mut", "egfr_mut", "tp53_mut"],
            "source_path": "/tmp/wes.parquet",
            "n_rows": 2,
            "n_donors": 2,
            "n_stages": 2,
            "stage_counts": {"AAH": 1, "AIS": 1},
            "tmb_mean": 15.0,
        },
    }

    table = build_dataset_preprocessing_table(data_output)
    assert table["modality"].tolist() == ["snRNA-seq", "Visium", "WES"]

    snrna_fig = plot_snrna_preprocessing_frontend(data_output)
    spatial_fig = plot_spatial_preprocessing_frontend(data_output)
    wes_fig = plot_wes_preprocessing_frontend(data_output)
    assert snrna_fig is not None
    assert spatial_fig is not None
    assert wes_fig is not None


def test_provider_benchmark_and_selected_provider(monkeypatch) -> None:
    def _fake_run_reference(cfg):
        return {"reference": {"diagnostics": {"alignment_gate": {"status": "pass"}}}}

    def _fake_run_spatial_mapping(cfg, reference_output=None):
        method = str(cfg.spatial_mapping.method)
        score = {"tangram": 0.40, "tacco": 0.65, "destvi": 0.52}[method]
        entropy = {"tangram": 0.80, "tacco": 0.30, "destvi": 0.55}[method]
        matrix = {
            "tangram": [[0.40, 0.35, 0.25], [0.45, 0.30, 0.25]],
            "tacco": [[0.85, 0.10, 0.05], [0.80, 0.12, 0.08]],
            "destvi": [[0.50, 0.30, 0.20], [0.52, 0.28, 0.20]],
        }[method]
        return {
            "ok": True,
            "status": "complete",
            "mapping_result": type(
                "Mapping",
                (),
                {
                    "compositions": matrix,
                    "coords": [[0.0, 0.0], [1.0, 1.0]],
                    "obs": __import__("pandas").DataFrame(index=["spot1", "spot2"]),
                    "feature_names": ("AT2", "Fibroblast", "Immune"),
                },
            )(),
            "spatial_mapping": {
                "method": method,
                "status": "complete",
                "provider_version": "test",
                "execution_mode": str(cfg.spatial_mapping.execution_mode),
                "n_spots": 2,
                "n_features": 3,
                "source_path": f"/tmp/{method}.h5ad",
                "qc": {
                    "mean_row_sum": 1.0,
                    "mean_max_assignment": score,
                    "mean_entropy": entropy,
                },
            },
        }

    def _fake_run_context_model(cfg, spatial_output=None):
        return {
            "typed_tokens": {"placeholder": True},
            "context_model": {"mode": str(cfg.context_model.mode)},
        }

    def _fake_run_transition_model(
        cfg, reference_output=None, spatial_output=None, context_output=None
    ):
        method = str(cfg.spatial_mapping.method)
        mode = str(cfg.context_model.mode)
        edge = "->".join(cfg.transition_model.active_edge)
        base = {
            ("tangram", "pooled", "AAH->AIS"): 11.0,
            ("tangram", "deep_sets", "AAH->AIS"): 10.8,
            ("tangram", "pooled", "AIS->MIA"): 12.2,
            ("tangram", "deep_sets", "AIS->MIA"): 12.0,
            ("tacco", "pooled", "AAH->AIS"): 9.2,
            ("tacco", "deep_sets", "AAH->AIS"): 9.0,
            ("tacco", "pooled", "AIS->MIA"): 10.1,
            ("tacco", "deep_sets", "AIS->MIA"): 9.8,
            ("destvi", "pooled", "AAH->AIS"): 10.4,
            ("destvi", "deep_sets", "AAH->AIS"): 10.2,
            ("destvi", "pooled", "AIS->MIA"): 11.0,
            ("destvi", "deep_sets", "AIS->MIA"): 10.9,
        }[(method, mode, edge)]
        return {"edge": edge, "mode": mode, "status": "complete", "heldout_proxy": base}

    def _fake_run_evaluation(cfg, transition_output=None, context_output=None):
        method = str(cfg.spatial_mapping.method)
        edge = transition_output["edge"]
        base = float(transition_output["heldout_proxy"])
        dominant = {
            ("tangram", "AAH->AIS"): ("stromal", "vascular_program"),
            ("tangram", "AIS->MIA"): ("stromal", "epithelial"),
            ("tacco", "AAH->AIS"): ("immune", "stromal"),
            ("tacco", "AIS->MIA"): ("immune", "epithelial"),
            ("destvi", "AAH->AIS"): ("immune", "stromal"),
            ("destvi", "AIS->MIA"): ("immune", "epithelial"),
        }[(method, edge)]
        return {
            "status": "complete",
            "heldout_metrics": {"sinkhorn": base},
            "calibration": {"mean_abs_shift_error": base / 20.0},
            "biology_summary": {
                "dominant_increase_group": dominant[0],
                "dominant_decrease_group": dominant[1],
            },
        }

    monkeypatch.setattr("stagebridge.notebook_api.run_reference", _fake_run_reference)
    monkeypatch.setattr("stagebridge.notebook_api.run_spatial_mapping", _fake_run_spatial_mapping)
    monkeypatch.setattr("stagebridge.notebook_api.run_context_model", _fake_run_context_model)
    monkeypatch.setattr(
        "stagebridge.notebook_api.run_transition_model", _fake_run_transition_model
    )
    monkeypatch.setattr("stagebridge.notebook_api.run_evaluation", _fake_run_evaluation)

    cfg = OmegaConf.create(
        {
            "seed": 42,
            "train": {"profile": "medium", "seed": 42},
            "transition_model": {
                "wes_regularizer": {"enabled": False},
                "schrodinger_bridge": {"sigma": 0.0},
                "active_edge": ["AAH", "AIS"],
            },
            "profiles": {"spatial_mapping": "tangram", "train": "medium"},
            "spatial_mapping": {"method": "tangram", "execution_mode": "force_rebuild"},
            "context_model": {"mode": "set_only", "graph_enabled": False},
        }
    )

    benchmark_output = run_provider_benchmark(
        cfg,
        methods=["tangram", "tacco", "destvi"],
        modes=["pooled", "deep_sets"],
        edges=["AAH->AIS", "AIS->MIA"],
        seeds=[7, 13],
        use_tqdm=False,
    )
    table = build_provider_benchmark_table(benchmark_output)
    selected_cfg = apply_selected_provider(cfg, benchmark_output)

    assert table.iloc[0]["method"] == "tacco"
    assert benchmark_output["benchmark"]["selected_provider"] == "tacco"
    assert selected_cfg.spatial_mapping.method == "tacco"

    fig_benchmark = plot_provider_benchmark_frontend(benchmark_output)
    fig_abundance = plot_spatial_provider_abundance_frontend(
        benchmark_output["provider_outputs_by_seed"][7]
    )
    assert fig_benchmark is not None
    assert fig_abundance is not None


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


def test_run_spatial_provider_ladder_and_summary_table(monkeypatch) -> None:
    seen_calls: list[tuple[str, str]] = []

    def _fake_run_spatial_mapping(cfg, reference_output=None):
        method = str(cfg.spatial_mapping.method)
        execution_mode = str(cfg.spatial_mapping.execution_mode)
        seen_calls.append((method, execution_mode))
        return {
            "ok": True,
            "status": "complete",
            "mapping_result": {
                "unused": True,
            },
            "spatial_mapping": {
                "method": method,
                "status": "complete",
                "provider_version": "test",
                "execution_mode": execution_mode,
                "n_spots": 8 if method != "destvi" else 0,
                "n_features": 4 if method != "destvi" else 0,
                "source_path": f"/tmp/{method}.h5ad",
                "notes": f"{method} summary",
            },
        }

    monkeypatch.setattr("stagebridge.notebook_api.run_spatial_mapping", _fake_run_spatial_mapping)

    cfg = OmegaConf.create(
        {
            "profiles": {"spatial_mapping": "tangram"},
            "spatial_mapping": {"method": "tangram", "show_progress": False},
        }
    )
    results = run_spatial_provider_ladder(
        cfg, methods=["tangram", "tacco", "destvi"], use_tqdm=False
    )
    table = build_spatial_provider_table(results)

    assert seen_calls == [
        ("tangram", "force_rebuild"),
        ("tacco", "force_rebuild"),
        ("destvi", "force_rebuild"),
    ]
    assert table["method"].tolist() == ["tangram", "tacco", "destvi"]
    assert set(table["execution_mode"]) == {"force_rebuild"}

    fig = plot_spatial_provider_comparison_frontend(results)
    assert fig is not None
    assert len(fig.axes) == 3


def test_spatial_provider_metric_and_agreement_tables_and_maps() -> None:
    provider_outputs = {
        "tangram": {
            "status": "complete",
            "mapping_result": type(
                "Mapping",
                (),
                {
                    "compositions": [[0.7, 0.2, 0.1], [0.1, 0.7, 0.2]],
                    "coords": [[0.0, 0.0], [1.0, 1.0]],
                    "obs": __import__("pandas").DataFrame(index=["s1", "s2"]),
                    "feature_names": ("A", "B", "C"),
                },
            )(),
            "spatial_mapping": {
                "status": "complete",
                "execution_mode": "force_rebuild",
                "n_spots": 2,
                "n_features": 3,
                "provider_version": "x",
            },
        },
        "tacco": {
            "status": "complete",
            "mapping_result": type(
                "Mapping",
                (),
                {
                    "compositions": [[0.6, 0.3, 0.1], [0.2, 0.6, 0.2]],
                    "coords": [[0.0, 0.0], [1.0, 1.0]],
                    "obs": __import__("pandas").DataFrame(index=["s1", "s2"]),
                    "feature_names": ("A", "B", "C"),
                },
            )(),
            "spatial_mapping": {
                "status": "complete",
                "execution_mode": "force_rebuild",
                "n_spots": 2,
                "n_features": 3,
                "provider_version": "y",
            },
        },
        "destvi": {
            "status": "complete",
            "mapping_result": type(
                "Mapping",
                (),
                {
                    "compositions": [[0.4, 0.4, 0.2], [0.1, 0.2, 0.7]],
                    "coords": [[0.0, 0.0], [1.0, 1.0]],
                    "obs": __import__("pandas").DataFrame(index=["s1", "s2"]),
                    "feature_names": ("A", "B", "C"),
                },
            )(),
            "spatial_mapping": {
                "status": "complete",
                "execution_mode": "force_rebuild",
                "n_spots": 2,
                "n_features": 3,
                "provider_version": "z",
            },
        },
    }

    metric_table = build_spatial_provider_metric_table(provider_outputs)
    agreement_table = build_spatial_provider_agreement_table(provider_outputs)

    assert {"method", "qc_heuristic_score", "rows_close_to_one_frac", "dominant_feature"} <= set(
        metric_table.columns
    )
    assert len(metric_table) == 3
    assert {"left_method", "right_method", "winner_agreement", "mean_abs_diff"} <= set(
        agreement_table.columns
    )
    assert len(agreement_table) == 3

    fig = plot_spatial_provider_maps_frontend(provider_outputs)
    assert fig is not None
    assert len(fig.axes) == 3
