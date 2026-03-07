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
