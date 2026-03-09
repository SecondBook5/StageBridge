from __future__ import annotations

import numpy as np

from stagebridge.data.luad_evo.bag_dataset import LesionBagDataset, NeighborhoodPretrainDataset, collate_lesion_bags
from stagebridge.data.luad_evo.neighborhood_builder import _canonical_sample_key, _resolve_local_neighborhood_geometry
from stagebridge.data.luad_evo.splits import assert_no_split_leakage, build_lesion_folds
from stagebridge.utils.types import LesionBag, LocalNicheExample


def _make_bag(sample_id: str, donor_id: str, edge_label: str, label: float, shift: float) -> LesionBag:
    neighborhoods = []
    for idx in range(3):
        receiver = np.asarray([1.0 + shift, 0.5 + idx * 0.1, label], dtype=np.float32)
        rings = np.asarray(
            [
                [0.6 + 0.1 * label, 0.2, 0.2],
                [0.3, 0.5 + shift, 0.2],
                [0.2, 0.2, 0.6 + shift],
            ],
            dtype=np.float32,
        )
        lr = np.asarray([0.8 * label + 0.1, 0.4 + shift, 0.3, 0.2], dtype=np.float32)
        stats = np.asarray([3.0 + idx, 0.2, 0.3, 0.5, 0.9, 1.0], dtype=np.float32)
        neighborhoods.append(
            LocalNicheExample(
                lesion_id=sample_id,
                sample_id=sample_id,
                donor_id=donor_id,
                patient_id=donor_id,
                stage=edge_label.split("->", 1)[0],
                edge_label=edge_label,
                receiver_index=idx,
                receiver_embedding=receiver,
                receiver_state_id=idx % 2,
                ring_compositions=rings,
                lr_pathway_summary=lr,
                neighborhood_stats=stats,
                flat_features=np.concatenate([receiver, rings.reshape(-1), lr, stats]).astype(np.float32),
                center_coord=np.asarray([float(idx), float(idx + 1)], dtype=np.float32),
                receiver_confidence=0.9,
            )
        )
    edge_id = 0 if edge_label == "AAH->AIS" else 1
    return LesionBag(
        lesion_id=sample_id,
        sample_id=sample_id,
        donor_id=donor_id,
        patient_id=donor_id,
        stage=edge_label.split("->", 1)[0],
        edge_id=edge_id,
        edge_label=edge_label,
        label=float(label),
        label_weight=1.0,
        label_source="synthetic",
        neighborhoods=neighborhoods,
        evolution_features=np.asarray([0.1 + shift, label], dtype=np.float32),
    )


def test_collate_lesion_bags_and_pretrain_dataset() -> None:
    bags = [
        _make_bag("S1", "P1", "AIS->MIA", 1.0, 0.0),
        _make_bag("S2", "P2", "AIS->MIA", 0.0, 0.1),
    ]
    dataset = LesionBagDataset(bags)
    batch = collate_lesion_bags([dataset[0], dataset[1]])
    assert batch.receiver_embeddings.shape == (2, 3, 3)
    assert batch.ring_compositions.shape == (2, 3, 3, 3)
    assert batch.evolution_features is not None

    pretrain = NeighborhoodPretrainDataset(bags)
    assert len(pretrain) == 6


def test_build_lesion_folds_is_donor_held_out() -> None:
    bags = [
        _make_bag("S1", "P1", "AIS->MIA", 1.0, 0.0),
        _make_bag("S2", "P2", "AIS->MIA", 0.0, 0.1),
        _make_bag("S3", "P3", "AIS->MIA", 1.0, 0.2),
        _make_bag("S4", "P4", "AIS->MIA", 0.0, 0.3),
        _make_bag("S5", "P5", "AIS->MIA", 1.0, 0.4),
        _make_bag("S6", "P6", "AIS->MIA", 0.0, 0.5),
    ]
    folds = build_lesion_folds(bags, num_folds=3, seed=7, min_lesions_per_class=1)
    assert len(folds) == 3
    for fold in folds:
        assert_no_split_leakage(bags, fold)
        assert set(fold.train_donors).isdisjoint(set(fold.test_donors))


def test_build_lesion_folds_rejects_single_negative_donor_for_cv() -> None:
    bags = [
        _make_bag("S1", "P1", "AAH->AIS", 1.0, 0.0),
        _make_bag("S2", "P2", "AAH->AIS", 1.0, 0.1),
        _make_bag("S3", "P3", "AAH->AIS", 1.0, 0.2),
        _make_bag("S4", "P4", "AAH->AIS", 0.0, 0.3),
    ]
    try:
        build_lesion_folds(bags, num_folds=3, seed=7, min_lesions_per_class=1)
    except ValueError as exc:
        message = str(exc)
        assert "not possible" in message or "insufficient" in message or "missing label" in message
    else:
        raise AssertionError("Expected build_lesion_folds to reject unsupported donor-held-out CV.")


def test_canonical_sample_key_normalizes_curated_and_spatial_ids() -> None:
    assert _canonical_sample_key("GSM9237907_P4_AAH1") == "P4_AAH-1"
    assert _canonical_sample_key("GSM9226176_P4_AAH-1") == "P4_AAH-1"
    assert _canonical_sample_key("GSM9237960_P21_AIS1") == "P21_AIS-1"


def test_resolve_local_neighborhood_geometry_falls_back_to_adaptive_knn() -> None:
    coords = np.asarray(
        [
            [0.0, 0.0],
            [1000.0, 0.0],
            [2000.0, 0.0],
            [3000.0, 0.0],
            [4000.0, 0.0],
        ],
        dtype=np.float32,
    )
    ring_edges, density = _resolve_local_neighborhood_geometry(
        coords,
        center_index=0,
        configured_edges=None,
        neighborhood_radius=150.0,
        min_instances=3,
        adaptive_neighbor_k=3,
        num_rings=3,
    )
    assert len(ring_edges) == 4
    assert density >= 3.0
    assert ring_edges[-1] >= 2000.0
