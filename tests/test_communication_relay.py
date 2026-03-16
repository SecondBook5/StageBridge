from __future__ import annotations

import numpy as np

from stagebridge.context_model.communication_relay import (
    StageBridgeCommunicationModel,
    build_communication_model,
    collate_communication_bags,
)
from stagebridge.transition_model.disease_edges import edge_id_map
from stagebridge.utils.types import CommunicationBag, CommunicationNeighborhoodExample


def _make_bags() -> list[CommunicationBag]:
    edge_lookup = edge_id_map()
    example_a = CommunicationNeighborhoodExample(
        receiver_embedding=np.asarray([2.0, 0.0, 0.0], dtype=np.float32),
        receiver_programs=np.asarray([1.2, 0.9, 0.1], dtype=np.float32),
        sender_embeddings=np.asarray([[1.0, 0.1], [0.8, 0.2]], dtype=np.float32),
        sender_types=np.asarray([0, 1], dtype=np.int64),
        sender_offsets=np.asarray([[0.0, 0.0], [0.2, 0.1]], dtype=np.float32),
        ring_ids=np.asarray([0, 1], dtype=np.int64),
        lr_token_features=np.asarray(
            [[0.9, 0.7, 0.8, 1.0, 0.1, 0.0, 0.8, 0.0, 0.0, 0.6]], dtype=np.float32
        ),
        response_token_features=np.asarray([[0.8, 0.7, 0.0, 4.0, 1.0]], dtype=np.float32),
        relay_token_features=np.asarray([[0.7, 0.8, 0.56, 0.6, 0.0, 0.0]], dtype=np.float32),
        edge_id=edge_lookup["AAH->AIS"],
        sample_id="S1",
        donor_id="P1",
        weak_label=1.0,
        receiver_cell_id="c1",
        lr_token_names=["IL1B->IL1R1|inflammatory|sender_0"],
        response_token_names=["inflammatory_response|inflammatory"],
        relay_token_names=["inflammatory_relay|inflammatory_response"],
    )
    example_b = CommunicationNeighborhoodExample(
        receiver_embedding=np.asarray([0.0, 2.0, 0.0], dtype=np.float32),
        receiver_programs=np.asarray([0.2, 0.1, 1.1], dtype=np.float32),
        sender_embeddings=np.asarray([[0.1, 1.0], [0.2, 0.9]], dtype=np.float32),
        sender_types=np.asarray([1, 1], dtype=np.int64),
        sender_offsets=np.asarray([[0.0, 0.0], [0.3, 0.2]], dtype=np.float32),
        ring_ids=np.asarray([0, 1], dtype=np.int64),
        lr_token_features=np.asarray(
            [[0.2, 0.3, 0.1, 1.0, 0.2, 0.0, 0.7, 1.0, 1.0, 0.2]], dtype=np.float32
        ),
        response_token_features=np.asarray([[0.2, 0.1, 1.0, 4.0, 2.0]], dtype=np.float32),
        relay_token_features=np.asarray([[0.1, 0.2, 0.02, 0.7, 0.0, 1.0]], dtype=np.float32),
        edge_id=edge_lookup["AIS->MIA"],
        sample_id="S2",
        donor_id="P2",
        weak_label=0.0,
        receiver_cell_id="c2",
        lr_token_names=["CXCL12->CXCR4|chemokine|sender_0"],
        response_token_names=["migration_invasion|chemokine"],
        relay_token_names=["chemokine_relay|migration_invasion"],
    )
    return [
        CommunicationBag(
            sample_id="S1",
            donor_id="P1",
            edge_id=edge_lookup["AAH->AIS"],
            edge_label="AAH->AIS",
            weak_label=1.0,
            examples=[example_a],
            label_source="test",
        ),
        CommunicationBag(
            sample_id="S2",
            donor_id="P2",
            edge_id=edge_lookup["AIS->MIA"],
            edge_label="AIS->MIA",
            weak_label=0.0,
            examples=[example_b],
            label_source="test",
        ),
    ]


def test_stagebridge_communication_model_emits_bag_and_query_logits() -> None:
    batch = collate_communication_bags(_make_bags())
    model = StageBridgeCommunicationModel(
        receiver_dim=batch.receiver_embedding.shape[1],
        receiver_program_dim=batch.receiver_programs.shape[1],
        sender_dim=batch.sender_embeddings.shape[2],
        lr_dim=batch.lr_token_features.shape[2],
        response_dim=batch.response_token_features.shape[2],
        relay_dim=batch.relay_token_features.shape[2],
        hidden_dim=32,
        num_heads=4,
        dropout=0.0,
        num_sender_types=4,
        num_ring_ids=3,
        num_edges=max(edge_id_map().values()) + 1,
    )
    out = model(batch, return_attention=True)

    assert out.query_logits.shape[0] == 2
    assert out.bag_logits.shape[0] == 2
    assert out.context_tokens is not None
    assert "receiver_query_attention" in out.attention_maps


def test_build_communication_model_supports_baselines() -> None:
    batch = collate_communication_bags(_make_bags())
    for model_name in [
        "focal_only",
        "pooled",
        "deep_sets",
        "graphsage",
        "graph_transformer",
        "transformer_no_priors",
        "transformer_no_relay",
        "stagebridge",
    ]:
        model = build_communication_model(
            model_name,
            receiver_dim=batch.receiver_embedding.shape[1],
            receiver_program_dim=batch.receiver_programs.shape[1],
            sender_dim=batch.sender_embeddings.shape[2],
            lr_dim=batch.lr_token_features.shape[2],
            response_dim=batch.response_token_features.shape[2],
            relay_dim=batch.relay_token_features.shape[2],
            hidden_dim=32,
            num_heads=4,
            dropout=0.0,
            num_edges=max(edge_id_map().values()) + 1,
            num_sender_types=4,
            num_ring_ids=3,
        )
        out = model(batch, return_attention=False)
        assert out.bag_logits.shape[0] == 2
