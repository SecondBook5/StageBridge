"""Model-readiness dry run: PanIN adapter output -> CCRT operator forward pass.

Composes the adapter with the model/training layers to prove the adapted PanIN
data are model-ready. No biological training, no HPO, no parameter optimization.
"""

from __future__ import annotations

import torch

from stagebridge.ccrt.adapters.panin import (
    adapt_reference_panin,
)
from stagebridge.ccrt.data import CCRTIndexRegistry
from stagebridge.ccrt.operators import (
    ContextResidualTransportConfig,
    ContextResidualTransportOperator,
)
from stagebridge.ccrt.representations import SemanticGeometryConfig
from stagebridge.ccrt.training import CCRTTrainingBatch, build_training_batch
from stagebridge.ccrt.transport import (
    SemanticTransportLoss,
    SemanticTransportLossConfig,
    SinkhornConfig,
)

from _panin_fixtures import FixtureSpatialLoader, build_panin_source_fixture


def test_panin_model_ready_forward(tmp_path):
    cfg = build_panin_source_fixture(tmp_path)
    out = adapt_reference_panin(cfg, spatial_loader=FixtureSpatialLoader())
    assert out.edge_partitions, "expected at least one supported edge partition"

    # index registry from the adapter's ontology spec
    registry = CCRTIndexRegistry.from_system_specs([out.ontology.biological_system_spec])

    partition = out.edge_partitions[0]
    training_batch = build_training_batch(
        source_batch=partition.source_batch,
        target_semantic_features=partition.target_semantic_features.to(torch.float32),
        index_registry=registry,
        dtype=torch.float32,
    )
    assert isinstance(training_batch, CCRTTrainingBatch)
    training_batch.validate()

    # sender-context types + distances + edge index preserved through tensorization
    assert training_batch.sender_context_type_ids.min().item() >= 0
    assert bool((training_batch.distance_to_receiver >= 0).all())
    assert training_batch.transition_edge_index is not None

    # small operator sized to the adapted feature dims
    d_r = training_batch.receiver_features.shape[1]
    d_s = training_batch.sender_features.shape[2]
    d_z = training_batch.source_semantic_features.shape[1]
    model = ContextResidualTransportOperator(
        ContextResidualTransportConfig(
            receiver_dim=d_r, sender_dim=d_s, hidden_dim=8, num_heads=2,
            num_sender_context_types=registry.num_sender_context_types,
            empty_sender_context_type_id=registry.empty_sender_context_type_index,
            regulatory_dim=2, drift_dim=d_z, growth_dim=1,
            num_transition_edges=registry.num_transition_edges,
        )
    )
    model.eval()
    with torch.no_grad():
        model_out = model(
            receiver_features=training_batch.receiver_features,
            sender_features=training_batch.sender_features,
            sender_mask=training_batch.sender_mask,
            distance_to_receiver=training_batch.distance_to_receiver,
            sender_context_type_ids=training_batch.sender_context_type_ids,
            transition_edge_index=training_batch.transition_edge_index,
        )
    b = training_batch.batch_size()
    assert model_out.full_drift.shape == (b, d_z)
    assert model_out.full_growth.shape == (b, 1)
    assert bool(torch.isfinite(model_out.full_drift).all())

    # native semantic transport loss on the adapted populations
    loss = SemanticTransportLoss(
        geometry=SemanticGeometryConfig(metric="squared_euclidean", normalization="none"),
        native_sinkhorn=SinkhornConfig(epsilon=0.2, max_iterations=50),
        loss=SemanticTransportLossConfig(),
    )
    with torch.no_grad():
        sem_out = loss(
            source_semantic_features=training_batch.source_semantic_features,
            target_semantic_features=training_batch.target_semantic_features,
            predicted_drift=model_out.full_drift,
        )
    assert bool(torch.isfinite(sem_out.total_loss))
