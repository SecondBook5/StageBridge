"""High-level publication API for StageBridge.

This module provides a clean, user-friendly interface for using StageBridge,
designed to be intuitive for researchers familiar with scanpy/scvi-tools.

Example usage:
    import stagebridge as sb

    # Load pretrained model
    model = sb.StageBridge.from_pretrained("checkpoint.pt")

    # Run inference on new data
    predictions = model.predict(adata)

    # Get niche embeddings
    embeddings = model.embed_niches(neighborhoods)

    # Compute stage transitions
    transitions = model.compute_transitions(
        source_stage="Preinvasive",
        target_stage="Invasive"
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch

from stagebridge.contracts import (
    STAGE_TO_IDX,
    IDX_TO_STAGE,
    LATENT_DIM,
    HLCA_DIM,
    LUCA_DIM,
)
from stagebridge.models.stagebridge import StageBridge as _StageBridgeModel
from stagebridge.models.stagebridge import StageBridgeConfig

if TYPE_CHECKING:
    import anndata as ad


@dataclass
class PredictionOutput:
    """Output from model prediction.

    Attributes:
        predicted_embeddings: Predicted embeddings after transition [N, D]
        source_embeddings: Original embeddings [N, D]
        context_embeddings: Niche context embeddings [N, D]
        attention_weights: Attention to each token/ring [N, K]
        trajectories: Full trajectories if requested [N, T, D]
        cell_ids: Cell identifiers
        source_stage: Source stage name
        target_stage: Target stage name
    """

    predicted_embeddings: np.ndarray
    source_embeddings: np.ndarray
    context_embeddings: np.ndarray
    attention_weights: np.ndarray | None = None
    trajectories: np.ndarray | None = None
    cell_ids: list[str] | None = None
    source_stage: str | None = None
    target_stage: str | None = None

    def to_dataframe(self) -> pd.DataFrame:
        """Convert predictions to DataFrame."""
        records = []
        for i in range(len(self.predicted_embeddings)):
            record = {
                "cell_id": self.cell_ids[i] if self.cell_ids else f"cell_{i}",
                "source_stage": self.source_stage,
                "target_stage": self.target_stage,
                "predicted_embedding": self.predicted_embeddings[i].tolist(),
                "source_embedding": self.source_embeddings[i].tolist(),
                "context_embedding": self.context_embeddings[i].tolist(),
            }
            if self.attention_weights is not None:
                record["attention_weights"] = self.attention_weights[i].tolist()
            records.append(record)
        return pd.DataFrame(records)


@dataclass
class NicheEmbeddingOutput:
    """Output from niche embedding.

    Attributes:
        embeddings: Niche context embeddings [N, D]
        context_tokens: Individual token embeddings [N, K, D]
        attention_weights: Attention to each token [N, K]
        prototype_assignments: If using prototypes, soft assignments [N, P]
    """

    embeddings: np.ndarray
    context_tokens: np.ndarray | None = None
    attention_weights: np.ndarray | None = None
    prototype_assignments: np.ndarray | None = None


@dataclass
class TransitionOutput:
    """Output from transition computation.

    Attributes:
        trajectories: Full trajectories [N, T+1, D]
        velocities: Velocity at each step [N, T, D]
        transition_times: Time points [T+1]
        source_stage: Source stage name
        target_stage: Target stage name
    """

    trajectories: np.ndarray
    velocities: np.ndarray
    transition_times: np.ndarray
    source_stage: str
    target_stage: str


class StageBridgeAPI:
    """High-level API wrapper for StageBridge model.

    Provides a user-friendly interface similar to scvi-tools models.

    Example:
        model = StageBridgeAPI.from_pretrained("checkpoint.pt")
        embeddings = model.embed_niches(neighborhoods)
        predictions = model.predict(adata)
    """

    def __init__(
        self,
        model: _StageBridgeModel,
        config: StageBridgeConfig,
        device: str | torch.device = "cuda",
    ):
        """Initialize API wrapper.

        Args:
            model: Trained StageBridge model
            config: Model configuration
            device: Device to use for inference
        """
        self._model = model
        self._config = config
        self._device = torch.device(device) if isinstance(device, str) else device
        self._model.to(self._device)
        self._model.eval()

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str | Path,
        device: str = "auto",
        map_location: str | torch.device | None = None,
    ) -> "StageBridgeAPI":
        """Load a pretrained StageBridge model.

        Args:
            checkpoint_path: Path to checkpoint file (.pt)
            device: Device to load model on ("auto", "cuda", "cpu")
            map_location: Optional device to map checkpoint to

        Returns:
            StageBridgeAPI instance

        Example:
            model = StageBridgeAPI.from_pretrained("runs/exp1/checkpoints/best.pt")
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Determine device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if map_location is None:
            map_location = device

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)

        # Extract config
        config_dict = checkpoint.get("config", {}).get("model_config", {})
        config = StageBridgeConfig(**config_dict)

        # Create and load model
        model = _StageBridgeModel(config)
        model.load_state_dict(checkpoint["model_state_dict"])

        print(f"Loaded StageBridge model from {checkpoint_path}")
        print(f"  Hidden dim: {config.hidden_dim}")
        print(f"  Num stages: {config.num_stages}")
        print(f"  GW fusion: {config.use_gw_fusion}")

        return cls(model, config, device)

    @property
    def config(self) -> StageBridgeConfig:
        """Model configuration."""
        return self._config

    @property
    def device(self) -> torch.device:
        """Device model is on."""
        return self._device

    @property
    def num_stages(self) -> int:
        """Number of disease stages."""
        return self._config.num_stages

    @property
    def latent_dim(self) -> int:
        """Latent embedding dimension."""
        return LATENT_DIM

    @torch.no_grad()
    def predict(
        self,
        adata: "ad.AnnData | None" = None,
        neighborhoods: pd.DataFrame | None = None,
        source_stage: str | int = "Normal",
        target_stage: str | int = "Invasive",
        num_integration_steps: int = 8,
        return_trajectories: bool = False,
        batch_size: int = 256,
    ) -> PredictionOutput:
        """Run inference to predict cell state transitions.

        Args:
            adata: AnnData with obsm["X_stagebridge"] embeddings.
                   Must have neighborhoods prepared (see prepare_neighborhoods).
            neighborhoods: Alternatively, provide neighborhoods DataFrame directly
            source_stage: Source disease stage (name or index)
            target_stage: Target disease stage (name or index)
            num_integration_steps: Number of Euler steps for integration
            return_trajectories: If True, return full trajectories
            batch_size: Batch size for inference

        Returns:
            PredictionOutput with predicted embeddings

        Example:
            predictions = model.predict(
                adata,
                source_stage="Preinvasive",
                target_stage="Invasive"
            )
        """
        if adata is None and neighborhoods is None:
            raise ValueError("Must provide either adata or neighborhoods")

        # Convert stage names to indices
        if isinstance(source_stage, str):
            source_idx = STAGE_TO_IDX.get(source_stage)
            if source_idx is None:
                raise ValueError(f"Unknown stage: {source_stage}. Valid: {list(STAGE_TO_IDX.keys())}")
        else:
            source_idx = source_stage
            source_stage = IDX_TO_STAGE.get(source_idx, str(source_idx))

        if isinstance(target_stage, str):
            target_idx = STAGE_TO_IDX.get(target_stage)
            if target_idx is None:
                raise ValueError(f"Unknown stage: {target_stage}. Valid: {list(STAGE_TO_IDX.keys())}")
        else:
            target_idx = target_stage
            target_stage = IDX_TO_STAGE.get(target_idx, str(target_idx))

        # Prepare data
        if neighborhoods is not None:
            data = self._prepare_from_neighborhoods(neighborhoods)
        else:
            data = self._prepare_from_adata(adata)

        # Run in batches
        all_predicted = []
        all_context = []
        all_trajectories = []
        all_attention = []

        n_samples = len(data["receiver"])
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_data = {k: v[start:end] for k, v in data.items() if v is not None}

            # Encode niche
            niche_output = self._model.encode_niche(
                receiver=batch_data["receiver"].to(self._device),
                ring_cells=[rc.to(self._device) for rc in batch_data["ring_cells"]],
                ring_masks=[rm.to(self._device) for rm in batch_data["ring_masks"]],
                hlca=batch_data["hlca"].to(self._device),
                luca=batch_data["luca"].to(self._device),
                pathway=batch_data.get("pathway", torch.zeros(end - start, LATENT_DIM)).to(self._device),
                stats=batch_data.get("stats"),
            )

            context = niche_output.context
            context_tokens = niche_output.context_tokens

            # Compute stage transition
            stage_pair_id = self._model.encode_stage_pair_tensor(
                source_idx, target_idx, end - start, self._device
            )

            if return_trajectories:
                trajectory = self._model.sample_trajectory(
                    x0=batch_data["receiver"].to(self._device),
                    context=context,
                    stage_pair_id=stage_pair_id,
                    num_steps=num_integration_steps,
                    context_tokens=context_tokens,
                )
                all_trajectories.append(trajectory.cpu())
                predicted = trajectory[:, -1]
            else:
                predicted = self._model.integrate_euler(
                    x0=batch_data["receiver"].to(self._device),
                    context=context,
                    stage_pair_id=stage_pair_id,
                    num_steps=num_integration_steps,
                    context_tokens=context_tokens,
                )

            all_predicted.append(predicted.cpu())
            all_context.append(context.cpu())

            if niche_output.attention_weights is not None:
                all_attention.append(niche_output.attention_weights.cpu())

        # Concatenate results
        predicted_embeddings = torch.cat(all_predicted, dim=0).numpy()
        context_embeddings = torch.cat(all_context, dim=0).numpy()
        source_embeddings = data["receiver"].numpy()

        attention_weights = None
        if all_attention:
            attention_weights = torch.cat(all_attention, dim=0).numpy()

        trajectories = None
        if all_trajectories:
            trajectories = torch.cat(all_trajectories, dim=0).numpy()

        cell_ids = data.get("cell_ids")

        return PredictionOutput(
            predicted_embeddings=predicted_embeddings,
            source_embeddings=source_embeddings,
            context_embeddings=context_embeddings,
            attention_weights=attention_weights,
            trajectories=trajectories,
            cell_ids=cell_ids,
            source_stage=source_stage,
            target_stage=target_stage,
        )

    @torch.no_grad()
    def embed_niches(
        self,
        neighborhoods: pd.DataFrame,
        batch_size: int = 256,
        return_tokens: bool = False,
    ) -> NicheEmbeddingOutput:
        """Get niche context embeddings.

        Args:
            neighborhoods: DataFrame with neighborhood data
            batch_size: Batch size for processing
            return_tokens: If True, return individual token embeddings

        Returns:
            NicheEmbeddingOutput with embeddings

        Example:
            embeddings = model.embed_niches(neighborhoods)
            adata.obsm["X_niche"] = embeddings.embeddings
        """
        data = self._prepare_from_neighborhoods(neighborhoods)

        all_embeddings = []
        all_tokens = []
        all_attention = []
        all_prototypes = []

        n_samples = len(data["receiver"])
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_data = {k: v[start:end] for k, v in data.items() if v is not None}

            niche_output = self._model.encode_niche(
                receiver=batch_data["receiver"].to(self._device),
                ring_cells=[rc.to(self._device) for rc in batch_data["ring_cells"]],
                ring_masks=[rm.to(self._device) for rm in batch_data["ring_masks"]],
                hlca=batch_data["hlca"].to(self._device),
                luca=batch_data["luca"].to(self._device),
                pathway=batch_data.get("pathway", torch.zeros(end - start, LATENT_DIM)).to(self._device),
                stats=batch_data.get("stats"),
            )

            all_embeddings.append(niche_output.context.cpu())

            if return_tokens and niche_output.context_tokens is not None:
                all_tokens.append(niche_output.context_tokens.cpu())

            if niche_output.attention_weights is not None:
                all_attention.append(niche_output.attention_weights.cpu())

            if niche_output.niche_prototype_composition is not None:
                all_prototypes.append(niche_output.niche_prototype_composition.cpu())

        embeddings = torch.cat(all_embeddings, dim=0).numpy()

        context_tokens = None
        if all_tokens:
            context_tokens = torch.cat(all_tokens, dim=0).numpy()

        attention_weights = None
        if all_attention:
            attention_weights = torch.cat(all_attention, dim=0).numpy()

        prototype_assignments = None
        if all_prototypes:
            prototype_assignments = torch.cat(all_prototypes, dim=0).numpy()

        return NicheEmbeddingOutput(
            embeddings=embeddings,
            context_tokens=context_tokens,
            attention_weights=attention_weights,
            prototype_assignments=prototype_assignments,
        )

    @torch.no_grad()
    def compute_transitions(
        self,
        embeddings: np.ndarray,
        context: np.ndarray,
        source_stage: str | int = "Normal",
        target_stage: str | int = "Invasive",
        num_steps: int = 20,
    ) -> TransitionOutput:
        """Compute full transition trajectories.

        Args:
            embeddings: Source cell embeddings [N, D]
            context: Niche context embeddings [N, D]
            source_stage: Source stage name or index
            target_stage: Target stage name or index
            num_steps: Number of integration steps

        Returns:
            TransitionOutput with trajectories and velocities

        Example:
            transitions = model.compute_transitions(
                embeddings=adata.obsm["X_stagebridge"],
                context=adata.obsm["X_niche"],
                source_stage="Preinvasive",
                target_stage="Invasive"
            )
        """
        # Convert stage names
        if isinstance(source_stage, str):
            source_idx = STAGE_TO_IDX.get(source_stage)
            if source_idx is None:
                raise ValueError(f"Unknown stage: {source_stage}")
        else:
            source_idx = source_stage
            source_stage = IDX_TO_STAGE.get(source_idx, str(source_idx))

        if isinstance(target_stage, str):
            target_idx = STAGE_TO_IDX.get(target_stage)
            if target_idx is None:
                raise ValueError(f"Unknown stage: {target_stage}")
        else:
            target_idx = target_stage
            target_stage = IDX_TO_STAGE.get(target_idx, str(target_idx))

        # Convert to tensors
        x0 = torch.from_numpy(embeddings).float().to(self._device)
        ctx = torch.from_numpy(context).float().to(self._device)
        n = x0.shape[0]

        stage_pair_id = self._model.encode_stage_pair_tensor(
            source_idx, target_idx, n, self._device
        )

        # Sample trajectory
        trajectory = self._model.sample_trajectory(
            x0=x0,
            context=ctx,
            stage_pair_id=stage_pair_id,
            num_steps=num_steps,
            context_tokens=ctx.unsqueeze(1),
        )

        # Compute velocities
        velocities = trajectory[:, 1:] - trajectory[:, :-1]
        transition_times = np.linspace(0, 1, num_steps + 1)

        return TransitionOutput(
            trajectories=trajectory.cpu().numpy(),
            velocities=velocities.cpu().numpy(),
            transition_times=transition_times,
            source_stage=source_stage,
            target_stage=target_stage,
        )

    def _prepare_from_neighborhoods(self, neighborhoods: pd.DataFrame) -> dict[str, Any]:
        """Prepare batch data from neighborhoods DataFrame."""

        # Create temporary dataset to reuse parsing logic
        # This is a bit hacky but ensures consistency
        n = len(neighborhoods)

        # Check format
        has_ring_cols = "ring_1_cells" in neighborhoods.columns
        has_tokens = "tokens" in neighborhoods.columns

        if not has_ring_cols and not has_tokens:
            raise ValueError(
                "neighborhoods must have either ring columns (ring_1_cells, ...) "
                "or tokens column. Run prepare_neighborhoods() first."
            )

        receivers = []
        ring_cells_list = [[] for _ in range(4)]
        ring_masks_list = [[] for _ in range(4)]
        hlcas = []
        lucas = []
        pathways = []
        stats_list = []
        cell_ids = []

        max_cells = 50  # Standard max cells per ring

        for idx in range(n):
            row = neighborhoods.iloc[idx]

            if has_ring_cols:
                # Ring format
                receiver = np.array(row["receiver_z"], dtype=np.float32)
                hlca = np.array(row["hlca_z"], dtype=np.float32)
                luca = np.array(row["luca_z"], dtype=np.float32)

                for i in range(4):
                    cells_list = row[f"ring_{i+1}_cells"]
                    if cells_list is None or (isinstance(cells_list, np.ndarray) and cells_list.size == 0):
                        n_cells = 0
                    else:
                        n_cells = len(cells_list)

                    padded = np.zeros((max_cells, LATENT_DIM), dtype=np.float32)
                    mask = np.zeros(max_cells, dtype=bool)

                    if n_cells > 0:
                        n_use = min(n_cells, max_cells)
                        for j in range(n_use):
                            padded[j] = np.array(cells_list[j], dtype=np.float32)[:LATENT_DIM]
                            mask[j] = True

                    ring_cells_list[i].append(padded)
                    ring_masks_list[i].append(mask)

                pathway = None
                if "pathway_z" in row and row["pathway_z"] is not None:
                    pathway = np.array(row["pathway_z"], dtype=np.float32)

                stats = None
                if "stats_z" in row and row["stats_z"] is not None:
                    stats = np.array(row["stats_z"], dtype=np.float32)

            else:
                # Token format
                tokens = row["tokens"]
                receiver = np.array(tokens[0].get("z_fused", np.zeros(LATENT_DIM)), dtype=np.float32)
                hlca = np.array(tokens[5].get("z_hlca", np.zeros(HLCA_DIM)) if len(tokens) > 5 else np.zeros(HLCA_DIM), dtype=np.float32)
                luca = np.array(tokens[6].get("z_luca", np.zeros(LUCA_DIM)) if len(tokens) > 6 else np.zeros(LUCA_DIM), dtype=np.float32)

                for i in range(4):
                    ring_token = tokens[i + 1] if len(tokens) > i + 1 else {}
                    z_pooled = ring_token.get("z_pooled")

                    padded = np.zeros((max_cells, LATENT_DIM), dtype=np.float32)
                    mask = np.zeros(max_cells, dtype=bool)

                    if z_pooled is not None and len(z_pooled) > 0:
                        padded[0] = np.array(z_pooled, dtype=np.float32)[:LATENT_DIM]
                        mask[0] = True

                    ring_cells_list[i].append(padded)
                    ring_masks_list[i].append(mask)

                pathway = None
                stats = None

            receivers.append(receiver)
            hlcas.append(hlca)
            lucas.append(luca)
            pathways.append(pathway)
            stats_list.append(stats)
            cell_ids.append(row.get("cell_id", f"cell_{idx}"))

        # Stack into tensors
        result = {
            "receiver": torch.from_numpy(np.stack(receivers)),
            "ring_cells": [torch.from_numpy(np.stack(rc)) for rc in ring_cells_list],
            "ring_masks": [torch.from_numpy(np.stack(rm)) for rm in ring_masks_list],
            "hlca": torch.from_numpy(np.stack(hlcas)),
            "luca": torch.from_numpy(np.stack(lucas)),
            "cell_ids": cell_ids,
        }

        if any(p is not None for p in pathways):
            result["pathway"] = torch.from_numpy(np.stack([p if p is not None else np.zeros(LATENT_DIM) for p in pathways]))

        if any(s is not None for s in stats_list):
            # Get stats dim from first non-None
            stats_dim = next(s.shape[0] for s in stats_list if s is not None)
            result["stats"] = torch.from_numpy(np.stack([s if s is not None else np.zeros(stats_dim) for s in stats_list]))

        return result

    def _prepare_from_adata(self, adata: "ad.AnnData") -> dict[str, Any]:
        """Prepare batch data from AnnData object."""
        # Check for required data
        if "X_neighborhoods" not in adata.uns:
            raise ValueError(
                "AnnData must have neighborhoods prepared. "
                "Run prepare_neighborhoods(adata) first."
            )

        neighborhoods = adata.uns["X_neighborhoods"]
        return self._prepare_from_neighborhoods(neighborhoods)


# Convenience alias for backward compatibility
StageBridgeModel = StageBridgeAPI
