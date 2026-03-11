"""Shared type aliases and dataclasses used across StageBridge."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Union

import numpy as np
import scipy.sparse as sp

if TYPE_CHECKING:
    import torch


# A matrix that may be dense or sparse
ArrayLike = Union[np.ndarray, sp.spmatrix]
PathLike = Union[str, Path]


@dataclass(slots=True)
class StageBridgeConfig:
    """Configuration container for model, OT, and training defaults."""

    # Representation / model
    input_dim: int = 64
    hidden_dim: int = 128
    vector_field_hidden_dim: int = 256
    num_heads: int = 8
    num_inducing_points: int = 16
    num_seed_vectors: int = 1
    num_stages: int = 5
    time_embedding_dim: int = 32
    stage_embedding_dim: int = 32
    dropout: float = 0.1

    # OT + flow matching
    ot_epsilon: float = 0.05
    sinkhorn_iters: int = 80
    num_ot_pairs: int = 512
    context_consistency_weight: float = 0.1
    use_ot: bool = True
    use_stage_embedding: bool = True

    # Schrödinger Bridge / stochastic interpolant
    sigma: float = 0.0              # Brownian bridge noise level; 0.0 = deterministic OT-CFM
    use_stochastic_bridge: bool = False  # Enable SB interpolant during training

    # Cross-attention drift transformer
    # When True, the drift network attends over num_seed_vectors context tokens
    # rather than concatenating a single pooled vector.  This makes the
    # transformer functionally central: the model must attend to niche tokens
    # to produce the drift vector.
    use_cross_attn_drift: bool = False

    # Spatial Relative Position Encoding in ISAB
    # When True, ISAB1 receives (x,y) niche token coordinates and adds a
    # learned distance bias to the inducing→token attention logits.
    use_spatial_rpe: bool = False
    rpe_hidden_dim: int = 16

    # WES feature conditioning
    # When True, per-(patient, stage) somatic genomic features (TMB, driver
    # mutation flags) are projected and concatenated to the stage embedding.
    use_wes_features: bool = False
    wes_feature_dim: int = 8    # matches len(WES_FEATURE_COLS)
    wes_hidden_dim: int = 16    # projection bottleneck

    # ── Tier 3: Ligand-receptor signaling conditioning ──────────────────
    # When True, per-(patient, stage) LR interaction scores are projected
    # and concatenated to the stage embedding (like WES features).
    use_lr_features: bool = False
    lr_feature_dim: int = 24    # matches len(LUNG_LR_PAIRS)
    lr_hidden_dim: int = 32     # projection bottleneck

    # ── Tier 3: Spatial niche composition conditioning ──────────────────
    # When True, per-cell spatial niche composition vectors (from Tangram
    # KNN) are averaged over the source set and fused with pooled context.
    use_spatial_niche: bool = False
    spatial_niche_dim: int = 20     # number of cell types from Tangram
    spatial_niche_hidden: int = 32  # projection hidden dim

    # ── Tier 3: Multi-hop skip-stage consistency ────────────────────────
    # When True, a trajectory composition loss regularises direct skip
    # transitions to match chained adjacent transitions.
    use_multihop_consistency: bool = False
    multihop_consistency_weight: float = 0.1

    # ── Tier 3: Dirichlet stage assignment posterior ────────────────────
    # When True, a Dirichlet head predicts per-cell stage uncertainty.
    use_dirichlet_head: bool = False
    dirichlet_hidden_dim: int = 64
    dirichlet_loss_weight: float = 0.05

    # ── Graph-of-Sets Transformer (GoST) ────────────────────────────────
    # When True, PMA summaries are enriched via graph attention over
    # neighboring (patient, stage) nodes before conditioning the drift.
    use_graph_transformer: bool = False
    graph_num_layers: int = 2       # number of Graph Transformer blocks
    graph_num_heads: int = 4        # attention heads in graph attention

    # ── Unified genomic niche encoder (cross-dataset) ─────────────────
    # When True, uses a unified encoder that maps heterogeneous genomic
    # features (WES somatic variants + lpWGS copy-number) into a shared
    # niche embedding for cross-dataset Schrödinger bridge conditioning.
    use_genomic_niche: bool = False
    genomic_niche_dim: int = 32     # shared niche embedding dimension

    # Optimization
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    max_epochs: int = 150
    steps_per_epoch: int = 50
    val_steps: int = 10
    patience: int = 20
    gradient_accumulation_steps: int = 1

    # Runtime
    mixed_precision: bool = True
    device: str = "cuda"
    seed: int = 42
    # Number of early train steps to profile for runtime/memory diagnostics.
    profile_train_steps: int = 0

    def resolved_device(self) -> str:
        """Return a valid runtime device string."""
        if self.device.startswith("cuda"):
            try:
                import torch  # local import to avoid hard dependency at import time

                if torch.cuda.is_available():
                    return self.device
            except Exception:
                pass
            return "cpu"
        return self.device


@dataclass(slots=True)
class StageBatch:
    """Typed container for one stage-to-stage training batch."""

    x_src: "torch.Tensor"
    x_tgt: "torch.Tensor"
    stage_src: int
    stage_tgt: int
    donor_id: str
    x_set: "torch.Tensor | None" = None
    context_mask: "torch.Tensor | None" = None
    cell_type: "torch.Tensor | None" = None
    sample_id: str | None = None
    wes_features: "torch.Tensor | None" = None  # (wes_feature_dim,) per-patient WES vector
    niche_coords: "torch.Tensor | None" = None  # (m_niche, 2) spatial coords for niche tokens
    lr_features: "torch.Tensor | None" = None   # (lr_feature_dim,) per-patient LR scores
    spatial_niche: "torch.Tensor | None" = None  # (n_cells, spatial_niche_dim) per-cell niche
    stage_index: "torch.Tensor | None" = None    # (n_src,) integer stage labels for Dirichlet head
    genomic_niche: "torch.Tensor | None" = None  # (genomic_niche_dim,) unified niche embedding

    def to(self, device: str) -> "StageBatch":
        """Move tensor payloads to *device* and return a new StageBatch."""
        return StageBatch(
            x_src=self.x_src.to(device),
            x_tgt=self.x_tgt.to(device),
            stage_src=self.stage_src,
            stage_tgt=self.stage_tgt,
            donor_id=self.donor_id,
            x_set=self.x_set.to(device) if self.x_set is not None else None,
            context_mask=self.context_mask.to(device) if self.context_mask is not None else None,
            cell_type=self.cell_type.to(device) if self.cell_type is not None else None,
            sample_id=self.sample_id,
            wes_features=self.wes_features.to(device) if self.wes_features is not None else None,
            niche_coords=self.niche_coords.to(device) if self.niche_coords is not None else None,
            lr_features=self.lr_features.to(device) if self.lr_features is not None else None,
            spatial_niche=self.spatial_niche.to(device) if self.spatial_niche is not None else None,
            stage_index=self.stage_index.to(device) if self.stage_index is not None else None,
            genomic_niche=self.genomic_niche.to(device) if self.genomic_niche is not None else None,
        )


@dataclass(slots=True)
class CommunicationNeighborhoodExample:
    """One focal receiver example with its local communication neighborhood."""

    receiver_embedding: np.ndarray
    receiver_programs: np.ndarray
    sender_embeddings: np.ndarray
    sender_types: np.ndarray
    sender_offsets: np.ndarray
    ring_ids: np.ndarray
    lr_token_features: np.ndarray
    response_token_features: np.ndarray
    relay_token_features: np.ndarray
    edge_id: int
    sample_id: str
    donor_id: str
    weak_label: float
    receiver_cell_id: str | None = None
    stage: str | None = None
    lr_token_names: list[str] | None = None
    response_token_names: list[str] | None = None
    relay_token_names: list[str] | None = None
    wes_features: np.ndarray | None = None
    target_latent: np.ndarray | None = None


@dataclass(slots=True)
class CommunicationBag:
    """Sample-edge bag of focal receiver examples for MIL-style supervision."""

    sample_id: str
    donor_id: str
    edge_id: int
    edge_label: str
    weak_label: float
    examples: list[CommunicationNeighborhoodExample]
    label_source: str = "unknown"
    notes: str | None = None

    @property
    def num_examples(self) -> int:
        return len(self.examples)


@dataclass(slots=True)
class CommunicationBatch:
    """Padded bag batch for communication-relay classification."""

    receiver_embedding: "torch.Tensor"
    receiver_programs: "torch.Tensor"
    sender_embeddings: "torch.Tensor"
    sender_types: "torch.Tensor"
    sender_offsets: "torch.Tensor"
    ring_ids: "torch.Tensor"
    lr_token_features: "torch.Tensor"
    response_token_features: "torch.Tensor"
    relay_token_features: "torch.Tensor"
    query_mask: "torch.Tensor"
    sender_mask: "torch.Tensor"
    lr_mask: "torch.Tensor"
    response_mask: "torch.Tensor"
    relay_mask: "torch.Tensor"
    edge_ids: "torch.Tensor"
    weak_labels: "torch.Tensor"
    bag_index: "torch.Tensor"
    sample_ids: list[str]
    donor_ids: list[str]
    label_sources: list[str]
    receiver_cell_ids: list[str | None]
    wes_features: "torch.Tensor | None" = None
    target_latent: "torch.Tensor | None" = None

    def to(self, device: str) -> "CommunicationBatch":
        return CommunicationBatch(
            receiver_embedding=self.receiver_embedding.to(device),
            receiver_programs=self.receiver_programs.to(device),
            sender_embeddings=self.sender_embeddings.to(device),
            sender_types=self.sender_types.to(device),
            sender_offsets=self.sender_offsets.to(device),
            ring_ids=self.ring_ids.to(device),
            lr_token_features=self.lr_token_features.to(device),
            response_token_features=self.response_token_features.to(device),
            relay_token_features=self.relay_token_features.to(device),
            query_mask=self.query_mask.to(device),
            sender_mask=self.sender_mask.to(device),
            lr_mask=self.lr_mask.to(device),
            response_mask=self.response_mask.to(device),
            relay_mask=self.relay_mask.to(device),
            edge_ids=self.edge_ids.to(device),
            weak_labels=self.weak_labels.to(device),
            bag_index=self.bag_index.to(device),
            sample_ids=list(self.sample_ids),
            donor_ids=list(self.donor_ids),
            label_sources=list(self.label_sources),
            receiver_cell_ids=list(self.receiver_cell_ids),
            wes_features=None if self.wes_features is None else self.wes_features.to(device),
            target_latent=None if self.target_latent is None else self.target_latent.to(device),
        )


@dataclass(slots=True)
class LocalNicheExample:
    """Compact receiver-centered neighborhood used as one lesion-bag instance.

    This is the new EA-MIST local instance contract. Each object represents one
    compact local spatial niche centered on a pseudo-receiver derived from the
    lesion's epithelial-rich spatial support.

    Args:
        lesion_id: Stable lesion identifier. Defaults to ``sample_id`` in the
            current LUAD cohort because no separate lesion identifier is exposed.
        sample_id: Source spatial sample identifier.
        donor_id: Donor identifier used for held-out evaluation.
        patient_id: Patient identifier when distinct from donor_id.
        stage: Canonical stage label for the lesion.
        edge_label: Directed edge associated with the lesion's supervision task.
        receiver_index: Stable within-lesion index for the local niche.
        receiver_embedding: Receiver/pseudo-receiver latent embedding.
        receiver_state_id: Integer receiver state identifier.
        ring_compositions: Ring-wise sender composition tensor with shape
            ``(num_rings, num_sender_features)``.
        lr_pathway_summary: Compact ligand-receptor and pathway summary vector.
        neighborhood_stats: Compact density/diversity/uncertainty summary vector.
        flat_features: Flattened feature vector for MLP ablations and SSL tasks.
        center_coord: Spatial center of the neighborhood in tissue coordinates.
        receiver_confidence: Optional confidence score for the receiver estimate.
        notes: Optional provenance text.
    """

    lesion_id: str
    sample_id: str
    donor_id: str
    patient_id: str
    stage: str
    edge_label: str
    receiver_index: int
    receiver_embedding: np.ndarray
    receiver_state_id: int
    ring_compositions: np.ndarray
    lr_pathway_summary: np.ndarray
    neighborhood_stats: np.ndarray
    flat_features: np.ndarray
    center_coord: np.ndarray
    hlca_features: np.ndarray | None = None
    luca_features: np.ndarray | None = None
    receiver_state_label: str | None = None
    receiver_confidence: float | None = None
    notes: str | None = None

    def __setstate__(self, state: object) -> None:
        """Support unpickling caches built before optional fields were added."""
        if isinstance(state, tuple) and len(state) == 2:
            _, slot_dict = state
        elif isinstance(state, dict):
            slot_dict = state
        else:
            slot_dict = state if isinstance(state, dict) else {}
        defaults = {"hlca_features": None, "luca_features": None,
                     "receiver_state_label": None, "receiver_confidence": None, "notes": None}
        for slot in self.__slots__:
            value = slot_dict.get(slot, defaults.get(slot))
            object.__setattr__(self, slot, value)


@dataclass(slots=True)
class LesionBag:
    """Lesion-level bag used by EA-MIST weak supervision.

    Args:
        lesion_id: Stable lesion identifier.
        sample_id: Underlying spatial sample identifier.
        donor_id: Donor identifier.
        patient_id: Patient identifier.
        stage: Canonical lesion stage.
        edge_id: Integer edge identifier.
        edge_label: Edge label string like ``AIS->MIA``.
        label: Binary lesion label for the active edge.
        label_weight: Confidence or supervision weight for the lesion label.
        label_source: Provenance string describing curated vs heuristic source.
        neighborhoods: Local niche instances inside the lesion bag.
        evolution_features: Optional lesion-level evolution/WES feature vector.
        notes: Optional free-text notes.
    """

    lesion_id: str
    sample_id: str
    donor_id: str
    patient_id: str
    stage: str
    edge_id: int
    edge_label: str
    label: float
    label_weight: float
    label_source: str
    neighborhoods: list[LocalNicheExample]
    evolution_features: np.ndarray | None = None
    stage_index: int | None = None
    displacement_target: float | None = None
    edge_targets: np.ndarray | None = None
    edge_target_mask: np.ndarray | None = None
    edge_target_labels: tuple[str, ...] | None = None
    notes: str | None = None

    def __setstate__(self, state: object) -> None:
        """Support unpickling caches built before optional fields were added."""
        if isinstance(state, tuple) and len(state) == 2:
            _, slot_dict = state
        elif isinstance(state, dict):
            slot_dict = state
        else:
            slot_dict = state if isinstance(state, dict) else {}
        defaults = {"evolution_features": None, "stage_index": None,
                     "displacement_target": None, "edge_targets": None,
                     "edge_target_mask": None, "edge_target_labels": None, "notes": None}
        for slot in self.__slots__:
            value = slot_dict.get(slot, defaults.get(slot))
            object.__setattr__(self, slot, value)

    @property
    def num_neighborhoods(self) -> int:
        """Return the number of local niches in the lesion bag."""
        return len(self.neighborhoods)


@dataclass(slots=True)
class LesionBagBatch:
    """Padded lesion-bag batch for EA-MIST models.

    Args:
        receiver_embeddings: ``(B, N, D_r)`` receiver embedding tensor.
        receiver_state_ids: ``(B, N)`` receiver state ids.
        ring_compositions: ``(B, N, R, C)`` ring composition tensor.
        lr_pathway_summary: ``(B, N, L)`` compact LR/pathway summaries.
        neighborhood_stats: ``(B, N, S)`` compact neighborhood stats.
        flat_features: ``(B, N, F)`` flattened neighborhood feature tensor.
        center_coords: ``(B, N, 2)`` spatial centers.
        neighborhood_mask: ``(B, N)`` valid-instance mask.
        edge_ids: ``(B,)`` edge identifier for each lesion.
        labels: ``(B,)`` lesion-level binary labels.
        label_weights: ``(B,)`` lesion-level confidence weights.
        sample_ids: Sample identifiers aligned to batch rows.
        lesion_ids: Lesion identifiers aligned to batch rows.
        donor_ids: Donor identifiers aligned to batch rows.
        patient_ids: Patient identifiers aligned to batch rows.
        stages: Stage labels aligned to batch rows.
        label_sources: Label provenance aligned to batch rows.
        evolution_features: Optional ``(B, F_evo)`` lesion-level evolution tensor.
    """

    receiver_embeddings: "torch.Tensor"
    receiver_state_ids: "torch.Tensor"
    ring_compositions: "torch.Tensor"
    lr_pathway_summary: "torch.Tensor"
    neighborhood_stats: "torch.Tensor"
    flat_features: "torch.Tensor"
    center_coords: "torch.Tensor"
    hlca_features: "torch.Tensor | None"
    luca_features: "torch.Tensor | None"
    neighborhood_mask: "torch.Tensor"
    edge_ids: "torch.Tensor"
    labels: "torch.Tensor"
    label_weights: "torch.Tensor"
    stage_indices: "torch.Tensor | None"
    displacement_targets: "torch.Tensor | None"
    edge_targets: "torch.Tensor | None"
    edge_target_mask: "torch.Tensor | None"
    sample_ids: list[str]
    lesion_ids: list[str]
    donor_ids: list[str]
    patient_ids: list[str]
    stages: list[str]
    label_sources: list[str]
    edge_target_labels: tuple[str, ...] = ()
    evolution_features: "torch.Tensor | None" = None

    def to(self, device: str) -> "LesionBagBatch":
        """Move the tensor payload to ``device`` and return a new batch."""
        return LesionBagBatch(
            receiver_embeddings=self.receiver_embeddings.to(device),
            receiver_state_ids=self.receiver_state_ids.to(device),
            ring_compositions=self.ring_compositions.to(device),
            lr_pathway_summary=self.lr_pathway_summary.to(device),
            neighborhood_stats=self.neighborhood_stats.to(device),
            flat_features=self.flat_features.to(device),
            center_coords=self.center_coords.to(device),
            hlca_features=None if self.hlca_features is None else self.hlca_features.to(device),
            luca_features=None if self.luca_features is None else self.luca_features.to(device),
            neighborhood_mask=self.neighborhood_mask.to(device),
            edge_ids=self.edge_ids.to(device),
            labels=self.labels.to(device),
            label_weights=self.label_weights.to(device),
            stage_indices=None if self.stage_indices is None else self.stage_indices.to(device),
            displacement_targets=None if self.displacement_targets is None else self.displacement_targets.to(device),
            edge_targets=None if self.edge_targets is None else self.edge_targets.to(device),
            edge_target_mask=None if self.edge_target_mask is None else self.edge_target_mask.to(device),
            sample_ids=list(self.sample_ids),
            lesion_ids=list(self.lesion_ids),
            donor_ids=list(self.donor_ids),
            patient_ids=list(self.patient_ids),
            stages=list(self.stages),
            label_sources=list(self.label_sources),
            edge_target_labels=tuple(self.edge_target_labels),
            evolution_features=None if self.evolution_features is None else self.evolution_features.to(device),
        )


@dataclass(slots=True)
class DatasetAuditReport:
    """Structured report for data readiness checks."""

    snrna_path: str
    spatial_path: str
    hlca_path: str
    snrna_exists: bool
    spatial_exists: bool
    hlca_exists: bool
    snrna_shape: tuple[int, int] | None = None
    spatial_shape: tuple[int, int] | None = None
    hlca_shape: tuple[int, int] | None = None
    required_obs_columns_ok: bool = False
    canonical_stage_values_ok: bool = False
    issues: list[str] | None = None

    def to_dict(self) -> dict[str, object]:
        """Convert report to a JSON-serializable dictionary."""
        payload = asdict(self)
        payload["issues"] = payload.get("issues") or []
        return payload


@dataclass(slots=True)
class RunManifest:
    """Structured provenance record for one training/evaluation run."""

    run_name: str
    task: str
    model_name: str
    variant_label: str | None
    ablation: str | None
    seed: int
    device_requested: str
    device_resolved: str
    config_path: str | None
    config_hash: str | None
    timestamp_utc: str | None
    data_paths: dict[str, str]
    output_paths: dict[str, str]
    metrics_path: str | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Convert run manifest to dictionary."""
        return asdict(self)
