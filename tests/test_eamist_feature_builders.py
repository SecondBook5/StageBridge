from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from stagebridge.data.luad_evo.audit_luca_atlas import run as audit_luca_run
from stagebridge.data.luad_evo.build_eamist_bags import run as build_bags_run
from stagebridge.data.luad_evo.build_hlca_niche_features import run as build_hlca_run
from stagebridge.data.luad_evo.build_lesion_evo_features import run as build_evo_run
from stagebridge.data.luad_evo.build_luca_niche_features import run as build_luca_niche_run
from stagebridge.data.luad_evo.build_luca_reference import run as build_luca_reference_run
from stagebridge.data.luad_evo.feature_builder import required_panel_genes


def _write_luca_atlas(path: Path) -> Path:
    obs = pd.DataFrame(
        {
            "dataset": ["D1", "D1", "D2", "D2", "D3", "D3"],
            "patient_id": ["P1", "P1", "P2", "P2", "P3", "P3"],
            "sample_id": ["S1", "S1", "S2", "S2", "S3", "S3"],
            "cell_type_major": ["Epithelial", "Epithelial", "Immune", "Stromal", "Epithelial", "Immune"],
            "cell_state": [
                "AT2-like malignant invasive",
                "Basal-like malignant",
                "T cell activated",
                "Fibroblast stromal",
                "Secretory epithelial",
                "Macrophage inflammatory",
            ],
            "malignant_status": ["malignant", "malignant", "non_malignant", "non_malignant", "non_malignant", "non_malignant"],
            "epithelial_subtype": ["AT2", "Basal", None, None, "Secretory", None],
        },
        index=[f"luca_cell_{idx}" for idx in range(6)],
    )
    adata = ad.AnnData(X=np.asarray([[1.0, 0.1, 0.3], [1.2, 0.2, 0.4], [0.1, 1.0, 0.1], [0.2, 0.9, 0.2], [0.8, 0.3, 0.6], [0.1, 0.8, 0.3]], dtype=np.float32), obs=obs)
    adata.obsm["X_scVI"] = np.asarray(
        [
            [0.9, 0.1, 0.1, 0.2],
            [0.8, 0.2, 0.1, 0.1],
            [0.1, 0.9, 0.3, 0.2],
            [0.2, 0.3, 0.9, 0.8],
            [0.7, 0.4, 0.2, 0.3],
            [0.2, 0.8, 0.4, 0.2],
        ],
        dtype=np.float32,
    )
    adata.write_h5ad(path)
    return path


def _write_niche_parquet(path: Path) -> Path:
    rows = [
        {
            "spot_id": "spot_1",
            "donor_id": "P1",
            "patient_id": "P1",
            "stage": "AAH",
            "sample_id": "L1",
            "lesion_id": "L1",
            "x": 0.0,
            "y": 0.0,
            "tok_AT2": 0.55,
            "tok_Basal": 0.10,
            "tok_Capillary": 0.05,
            "tok_Ciliated": 0.05,
            "tok_Fibroblast lineage": 0.05,
            "tok_Macrophages": 0.05,
            "tok_Mast cells": 0.05,
            "tok_Secretory": 0.05,
            "tok_T cell lineage": 0.05,
        },
        {
            "spot_id": "spot_2",
            "donor_id": "P1",
            "patient_id": "P1",
            "stage": "AAH",
            "sample_id": "L1",
            "lesion_id": "L1",
            "x": 50.0,
            "y": 0.0,
            "tok_AT2": 0.40,
            "tok_Basal": 0.20,
            "tok_Capillary": 0.05,
            "tok_Ciliated": 0.05,
            "tok_Fibroblast lineage": 0.10,
            "tok_Macrophages": 0.05,
            "tok_Mast cells": 0.05,
            "tok_Secretory": 0.05,
            "tok_T cell lineage": 0.05,
        },
        {
            "spot_id": "spot_3",
            "donor_id": "P2",
            "patient_id": "P2",
            "stage": "AIS",
            "sample_id": "L2",
            "lesion_id": "L2",
            "x": 0.0,
            "y": 0.0,
            "tok_AT2": 0.15,
            "tok_Basal": 0.10,
            "tok_Capillary": 0.05,
            "tok_Ciliated": 0.05,
            "tok_Fibroblast lineage": 0.15,
            "tok_Macrophages": 0.20,
            "tok_Mast cells": 0.05,
            "tok_Secretory": 0.05,
            "tok_T cell lineage": 0.20,
        },
        {
            "spot_id": "spot_4",
            "donor_id": "P2",
            "patient_id": "P2",
            "stage": "AIS",
            "sample_id": "L2",
            "lesion_id": "L2",
            "x": 50.0,
            "y": 0.0,
            "tok_AT2": 0.10,
            "tok_Basal": 0.15,
            "tok_Capillary": 0.10,
            "tok_Ciliated": 0.05,
            "tok_Fibroblast lineage": 0.15,
            "tok_Macrophages": 0.20,
            "tok_Mast cells": 0.05,
            "tok_Secretory": 0.05,
            "tok_T cell lineage": 0.15,
        },
    ]
    df = pd.DataFrame(rows).set_index(pd.Index([f"niche_{idx}" for idx in range(4)], name="spot_obs_name"))
    for label in ("AT2", "Basal", "Capillary", "Ciliated", "Fibroblast lineage", "Macrophages", "Mast cells", "Secretory", "T cell lineage"):
        df[f"tok_smooth_{label}"] = df[f"tok_{label}"]
    df["entropy"] = 1.0
    df["confidence"] = 0.8
    df.to_parquet(path)
    return path


def _write_hlca_assets(latent_path: Path, labels_path: Path) -> tuple[Path, Path]:
    obs = pd.DataFrame(
        {
            "cell_id": [f"hlca_cell_{idx}" for idx in range(6)],
            "donor_id": ["P0", "P0", "P1", "P1", "P2", "P2"],
            "patient_id": ["P0", "P0", "P1", "P1", "P2", "P2"],
            "sample_id": ["H0", "H0", "H1", "H1", "H2", "H2"],
            "stage": ["Normal", "Normal", "AAH", "AAH", "AIS", "AIS"],
            "hlca_label": ["AT2", "Basal", "AT2", "Secretory", "Macrophages", "T cell lineage"],
        },
        index=[f"hlca_cell_{idx}" for idx in range(6)],
    )
    adata = ad.AnnData(X=np.asarray([[1.0, 0.1, 0.0], [0.8, 0.2, 0.0], [0.9, 0.1, 0.1], [0.6, 0.3, 0.2], [0.1, 0.9, 0.3], [0.2, 0.8, 0.4]], dtype=np.float32), obs=obs)
    adata.write_h5ad(latent_path)
    obs.loc[:, ["hlca_label"]].to_parquet(labels_path)
    return latent_path, labels_path


def _write_snrna_assets(raw_path: Path, latent_path: Path) -> tuple[Path, Path]:
    genes = required_panel_genes()
    obs = pd.DataFrame(
        {
            "cell_id": [f"sn_cell_{idx}" for idx in range(6)],
            "donor_id": ["P1", "P1", "P1", "P2", "P2", "P2"],
            "patient_id": ["P1", "P1", "P1", "P2", "P2", "P2"],
            "sample_id": ["L1", "L1", "L1", "L2", "L2", "L2"],
            "stage": ["AAH", "AAH", "AAH", "AIS", "AIS", "AIS"],
            "hlca_label": ["AT2", "Basal", "Secretory", "AT2", "Basal", "Secretory"],
        },
        index=[f"sn_cell_{idx}" for idx in range(6)],
    )
    raw = np.zeros((6, len(genes)), dtype=np.float32)
    for gene_name in ("EPCAM", "MUC1", "KRT19", "CEACAM5", "EGFR", "TGFBR2", "CXCR4", "ITGB1"):
        raw[:, genes.index(gene_name)] = np.linspace(0.5, 1.0, 6, dtype=np.float32)
    raw_adata = ad.AnnData(X=raw, obs=obs.copy(), var=pd.DataFrame(index=genes))
    raw_adata.write_h5ad(raw_path)

    latent = ad.AnnData(
        X=np.asarray(
            [
                [0.9, 0.1, 0.0, 0.0],
                [0.8, 0.2, 0.1, 0.0],
                [0.7, 0.3, 0.2, 0.1],
                [0.6, 0.2, 0.1, 0.1],
                [0.5, 0.3, 0.2, 0.1],
                [0.4, 0.4, 0.2, 0.1],
            ],
            dtype=np.float32,
        ),
        obs=obs.copy(),
    )
    latent.write_h5ad(latent_path)
    return raw_path, latent_path


def _write_evo_support(tmp_path: Path) -> dict[str, Path]:
    manifest = pd.DataFrame(
        [
            {"lesion_id": "L1", "sample_id": "L1", "patient_id": "P1", "donor_id": "P1", "stage": "AAH"},
            {"lesion_id": "L2", "sample_id": "L2", "patient_id": "P2", "donor_id": "P2", "stage": "AIS"},
        ]
    )
    wes = pd.DataFrame(
        [
            {"patient_id": "P1", "stage": "AAH", "tmb": 1.2, "kras_mut": 1.0, "egfr_mut": 0.0, "tp53_mut": 0.0, "stk11_mut": 0.0, "keap1_mut": 0.0, "smad4_mut": 0.0, "braf_mut": 0.0},
            {"patient_id": "P2", "stage": "AIS", "tmb": 0.8, "kras_mut": 0.0, "egfr_mut": 1.0, "tp53_mut": 1.0, "stk11_mut": 0.0, "keap1_mut": 0.0, "smad4_mut": 0.0, "braf_mut": 0.0},
        ]
    )
    refined = pd.DataFrame(
        [
            {"lesion_id": "L1", "sample_id": "L1", "patient_id": "P1", "donor_id": "P1", "stage": "AAH", "edge_label": "AAH->AIS", "refined_binary_label": "positive", "uncertainty_flag": False, "exclusion_flag": False, "progression_risk_score": 0.9, "confidence_tier": "high"},
            {"lesion_id": "L2", "sample_id": "L2", "patient_id": "P2", "donor_id": "P2", "stage": "AIS", "edge_label": "AIS->MIA", "refined_binary_label": "negative", "uncertainty_flag": False, "exclusion_flag": False, "progression_risk_score": 0.2, "confidence_tier": "high"},
        ]
    )
    cna = pd.DataFrame(
        [
            {"lesion_id": "L1", "purity": 0.6, "ploidy": 2.0, "fraction_genome_altered": 0.1, "cna_burden": 0.2, "num_focal_events": 1, "num_arm_level_events": 0, "allele_specific_imbalance": 0.0},
            {"lesion_id": "L2", "purity": 0.7, "ploidy": 2.2, "fraction_genome_altered": 0.2, "cna_burden": 0.3, "num_focal_events": 2, "num_arm_level_events": 1, "allele_specific_imbalance": 0.1},
        ]
    )
    clone = pd.DataFrame(
        [
            {"lesion_id": "L1", "num_clonal_clusters": 2, "dominant_clone_fraction": 0.7, "subclonal_entropy": 0.3, "shared_cluster_count_with_later_lesions": 1, "private_cluster_count": 1, "driver_cluster_count": 1},
            {"lesion_id": "L2", "num_clonal_clusters": 3, "dominant_clone_fraction": 0.6, "subclonal_entropy": 0.5, "shared_cluster_count_with_later_lesions": 0, "private_cluster_count": 2, "driver_cluster_count": 1},
        ]
    )
    phylogeny = pd.DataFrame(
        [
            {"lesion_id": "L1", "trunk_mutation_burden": 2.0, "branch_count": 1.0, "branch_length_mean": 0.4, "clone_sharing_score": 0.8, "descendant_sharing_score": 0.7, "trunk_membership_score": 0.9, "branch_specificity_score": 0.2, "evidence_of_progression_link": 1.0},
            {"lesion_id": "L2", "trunk_mutation_burden": 1.0, "branch_count": 2.0, "branch_length_mean": 0.6, "clone_sharing_score": 0.4, "descendant_sharing_score": 0.3, "trunk_membership_score": 0.5, "branch_specificity_score": 0.6, "evidence_of_progression_link": 0.0},
        ]
    )
    paths = {
        "manifest": tmp_path / "cleaned_manifest.csv",
        "wes": tmp_path / "wes.parquet",
        "refined": tmp_path / "refined.csv",
        "cna": tmp_path / "cna.csv",
        "clone": tmp_path / "clone.csv",
        "phylogeny": tmp_path / "phylogeny.csv",
    }
    manifest.to_csv(paths["manifest"], index=False)
    wes.to_parquet(paths["wes"], index=False)
    refined.to_csv(paths["refined"], index=False)
    cna.to_csv(paths["cna"], index=False)
    clone.to_csv(paths["clone"], index=False)
    phylogeny.to_csv(paths["phylogeny"], index=False)
    return paths


def test_luca_audit_and_reference_build(tmp_path: Path) -> None:
    atlas_path = _write_luca_atlas(tmp_path / "luca_atlas.h5ad")
    outdir = tmp_path / "luca_metadata"

    report = audit_luca_run(atlas_path, outdir)
    assert (outdir / "luca_audit_report.json").exists()
    assert (outdir / "luca_obs_schema.json").exists()
    assert (outdir / "luca_obsm_schema.json").exists()
    assert (outdir / "luca_obs.parquet").exists()
    assert report["selected_columns"]["state_column"] == "cell_state"

    manifest = build_luca_reference_run(atlas_path, outdir, chunk_size=2)
    assert manifest["embedding_key"] == "X_scVI"
    centroids = pd.read_parquet(outdir / "luca_state_centroids.parquet")
    summary = pd.read_parquet(outdir / "luca_state_summary.parquet")
    assert not centroids.empty
    assert not summary.empty
    assert "token_weight__AT2" in summary.columns


def test_hlca_luca_evo_and_bag_builders(tmp_path: Path) -> None:
    niche_path = _write_niche_parquet(tmp_path / "niche_tokens.parquet")
    atlas_path = _write_luca_atlas(tmp_path / "luca_atlas.h5ad")
    luca_meta = tmp_path / "luca_meta"
    build_luca_reference_run(atlas_path, luca_meta, chunk_size=2)
    luca_features_path = tmp_path / "niche_luca_features.parquet"
    build_luca_niche_run(
        niche_path,
        luca_meta / "luca_state_centroids.parquet",
        luca_meta / "luca_state_summary.parquet",
        luca_features_path,
        top_k=3,
    )
    luca_df = pd.read_parquet(luca_features_path)
    assert luca_df.shape[0] == 4
    assert "luca_tumor_adoption_score" in luca_df.columns

    hlca_latent, hlca_labels = _write_hlca_assets(tmp_path / "hlca_latent.h5ad", tmp_path / "hlca_labels.parquet")
    hlca_features_path = tmp_path / "niche_hlca_features.parquet"
    build_hlca_run(hlca_labels, hlca_latent, niche_path, hlca_features_path, top_k=3)
    hlca_df = pd.read_parquet(hlca_features_path)
    assert hlca_df.shape[0] == 4
    assert "hlca_normal_likeness_score" in hlca_df.columns

    support = _write_evo_support(tmp_path)
    evo_path = tmp_path / "lesion_evo_features.parquet"
    build_evo_run(
        support["wes"],
        evo_path,
        cleaned_manifest=support["manifest"],
        refined_labels=support["refined"],
        cna_summary=support["cna"],
        clone_summary=support["clone"],
        phylogeny_summary=support["phylogeny"],
    )
    evo_df = pd.read_parquet(evo_path)
    assert evo_df.shape[0] == 2
    assert "evo_driver_burden" in evo_df.columns

    raw_snrna, latent_snrna = _write_snrna_assets(tmp_path / "snrna_raw.h5ad", tmp_path / "snrna_latent.h5ad")
    viability_path = tmp_path / "split_viability_report.json"
    viability_path.write_text(
        json.dumps(
            {
                "edges": {
                    "AAH->AIS": {"binary_viable": True, "continuous_viable": True, "recommended_target": "binary_classification"},
                    "AIS->MIA": {"binary_viable": True, "continuous_viable": True, "recommended_target": "continuous_risk"},
                }
            }
        ),
        encoding="utf-8",
    )
    bags_path = tmp_path / "eamist_bags.parquet"
    build_bags_run(
        None,
        niche_path,
        hlca_features_path,
        luca_features_path,
        evo_path,
        bags_path,
        snrna_latent=latent_snrna,
        snrna_raw=raw_snrna,
        refined_labels=support["refined"],
        viability_report=viability_path,
    )
    bags = pd.read_parquet(bags_path)
    assert bags.shape[0] == 2
    assert set(["receiver_features", "ring_features", "hlca_features", "luca_features", "pathway_features", "niche_stats_features", "evo_features"]).issubset(bags.columns)
    assert len(bags.loc[0, "niche_ids"]) == 2
    assert len(bags.loc[0, "receiver_features"]) == 2
