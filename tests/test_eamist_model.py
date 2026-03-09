from __future__ import annotations

import torch

from stagebridge.context_model.lesion_set_transformer import EAMISTModel
from stagebridge.context_model.local_niche_encoder import LocalNicheTransformerEncoder
from stagebridge.context_model.prototype_bottleneck import PrototypeBottleneck
from stagebridge.utils.types import LesionBagBatch


def _make_batch() -> LesionBagBatch:
    return LesionBagBatch(
        receiver_embeddings=torch.randn(2, 4, 5),
        receiver_state_ids=torch.tensor([[0, 1, 0, 0], [1, 1, 0, 0]], dtype=torch.long),
        ring_compositions=torch.randn(2, 4, 3, 6),
        lr_pathway_summary=torch.randn(2, 4, 8),
        neighborhood_stats=torch.randn(2, 4, 6),
        flat_features=torch.randn(2, 4, 37),
        center_coords=torch.randn(2, 4, 2),
        neighborhood_mask=torch.tensor([[True, True, True, False], [True, True, False, False]]),
        edge_ids=torch.tensor([1, 1], dtype=torch.long),
        labels=torch.tensor([1.0, 0.0], dtype=torch.float32),
        label_weights=torch.tensor([1.0, 1.0], dtype=torch.float32),
        sample_ids=["S1", "S2"],
        lesion_ids=["S1", "S2"],
        donor_ids=["P1", "P2"],
        patient_ids=["P1", "P2"],
        stages=["AIS", "AIS"],
        label_sources=["synthetic", "synthetic"],
        evolution_features=torch.randn(2, 3),
    )


def test_local_niche_transformer_encoder_shapes() -> None:
    encoder = LocalNicheTransformerEncoder(
        receiver_dim=5,
        sender_feature_dim=6,
        lr_summary_dim=8,
        stats_dim=6,
        model_dim=16,
        num_heads=4,
        num_layers=2,
        num_receiver_states=4,
        num_rings=3,
        dropout=0.0,
    )
    output = encoder(
        receiver_embeddings=torch.randn(6, 5),
        receiver_state_ids=torch.tensor([0, 1, 2, 1, 0, 3], dtype=torch.long),
        ring_compositions=torch.randn(6, 3, 6),
        lr_pathway_summary=torch.randn(6, 8),
        neighborhood_stats=torch.randn(6, 6),
        return_attention=True,
    )
    assert output.neighborhood_embedding.shape == (6, 16)
    assert output.token_embeddings.shape[1] == 6


def test_prototype_bottleneck_and_eamist_forward() -> None:
    batch = _make_batch()
    bottleneck = PrototypeBottleneck(16, num_prototypes=8)
    assignments = bottleneck.get_assignment_weights(torch.randn(2, 4, 16))
    assert assignments.shape == (2, 4, 8)

    model = EAMISTModel(
        receiver_dim=5,
        sender_feature_dim=6,
        lr_summary_dim=8,
        stats_dim=6,
        flat_feature_dim=37,
        num_receiver_states=4,
        num_rings=3,
        hidden_dim=16,
        num_heads=4,
        num_layers=2,
        dropout=0.0,
        use_prototypes=True,
        num_prototypes=8,
        evolution_dim=3,
    )
    output = model(batch, return_attention=True)
    assert output.local_embeddings.shape == (2, 4, 16)
    assert output.logits.shape == (2, 2)
    assert output.selected_logits.shape == (2,)
    assert output.prototype_output is not None
