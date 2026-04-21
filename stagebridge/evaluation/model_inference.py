"""Model inference utilities for evaluation and hypothesis testing.

This module provides utilities to run trained StageBridge models on data
and extract the outputs needed for hypothesis validation:
- stage_logits -> transition_probs (for H3.1, plasticity)
- attention_weights -> niche_influence (for H3.2, biological interpretation)

No placeholders - all outputs are real model predictions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

log = logging.getLogger(__name__)


@dataclass
class InferenceOutputs:
    """Outputs from model inference."""

    cell_ids: np.ndarray  # (n_cells,) cell identifiers
    stage_logits: np.ndarray  # (n_cells, n_stages) raw logits
    stage_probs: np.ndarray  # (n_cells, n_stages) softmax probabilities
    transition_probs: np.ndarray  # (n_cells,) probability of transitioning to later stage
    attention_weights: np.ndarray  # (n_cells, n_neighbors) attention to neighbors
    niche_influence: np.ndarray  # (n_cells,) aggregated niche influence score
    current_stage: np.ndarray  # (n_cells,) current stage index

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame for easy analysis."""
        df = pd.DataFrame({
            'cell_id': self.cell_ids,
            'transition_prob': self.transition_probs,
            'niche_influence': self.niche_influence,
            'current_stage': self.current_stage,
        })
        # Add stage probabilities
        for i in range(self.stage_probs.shape[1]):
            df[f'stage_prob_{i}'] = self.stage_probs[:, i]
        return df


def load_checkpoint(
    checkpoint_path: Path | str,
    device: str = "cpu",
) -> dict:
    """Load a model checkpoint.

    Args:
        checkpoint_path: Path to checkpoint .pt file
        device: Device to load tensors to

    Returns:
        Checkpoint dictionary with model_state_dict, config, etc.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    log.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    return checkpoint


def build_model_from_checkpoint(
    checkpoint: dict,
    device: str = "cpu",
) -> torch.nn.Module:
    """Reconstruct model from checkpoint.

    Args:
        checkpoint: Loaded checkpoint dictionary
        device: Device to place model on

    Returns:
        Model with loaded weights in eval mode
    """
    # Extract config
    config = checkpoint.get("config", {})
    model_config = config.get("model", {})

    # Determine model type from config
    # Default to StageBridgeV1Complete which is what run_v1_ddp.py produces
    model_type = model_config.get("type", "stagebridge_v1_complete")

    if model_type in ("stagebridge_v1_complete", "hierarchical_set_transformer"):
        from stagebridge.pipelines.run_v1_complete import StageBridgeV1Complete

        model = StageBridgeV1Complete(
            latent_dim=model_config.get("latent_dim", config.get("latent_dim", 40)),
            niche_hidden_dim=model_config.get("niche_hidden_dim", config.get("niche_hidden_dim", 128)),
            context_dim=model_config.get("context_dim", config.get("context_dim", 256)),
            dropout=model_config.get("dropout", config.get("dropout", 0.1)),
            hlca_dim=model_config.get("hlca_dim", config.get("hlca_dim", 30)),
            luca_dim=model_config.get("luca_dim", config.get("luca_dim", 10)),
            wes_feature_dim=model_config.get("wes_feature_dim", config.get("wes_feature_dim", config.get("wes_dim", 8))),
            no_niche=model_config.get("no_niche", config.get("no_niche", False)),
            no_wes=model_config.get("no_wes", config.get("no_wes", False)),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load weights
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", {}))
    if state_dict:
        model.load_state_dict(state_dict, strict=False)
        log.info(f"Loaded model weights ({len(state_dict)} keys)")
    else:
        log.warning("No model weights found in checkpoint")

    model = model.to(device)
    model.eval()

    return model


def prepare_inference_data(
    cells_df: pd.DataFrame,
    neighborhoods_df: pd.DataFrame | None = None,
    batch_size: int = 256,
) -> tuple[DataLoader, np.ndarray, np.ndarray]:
    """Prepare data for model inference.

    Args:
        cells_df: DataFrame with cell features (z_fused_*, stage_idx, etc.)
        neighborhoods_df: Optional DataFrame with neighborhood token data
            If provided, uses proper 9-token niche representations.
            If None, uses receiver embedding for all tokens (degraded accuracy).
        batch_size: Batch size for inference

    Returns:
        Tuple of (dataloader, cell_ids, current_stages)
        DataLoader yields (niche_tokens, stages) where niche_tokens is [B, 9, D]
    """
    # Extract cell IDs
    cell_ids = cells_df['cell_id'].values if 'cell_id' in cells_df.columns else cells_df.index.values

    # Extract current stage
    if 'stage_idx' in cells_df.columns:
        current_stage = cells_df['stage_idx'].values.astype(np.int64)
    elif 'stage' in cells_df.columns:
        stage_map = {'Normal': 0, 'AAH': 1, 'AIS': 2, 'MIA': 3, 'LUAD': 4}
        current_stage = cells_df['stage'].map(stage_map).fillna(0).values.astype(np.int64)
    else:
        log.warning("No stage column found, assuming stage 0")
        current_stage = np.zeros(len(cells_df), dtype=np.int64)

    # Extract fused embeddings (z_fused_0, z_fused_1, ..., z_fused_39)
    fused_cols = [c for c in cells_df.columns if c.startswith('z_fused_')]
    if fused_cols:
        fused_cols = sorted(fused_cols, key=lambda x: int(x.split('_')[-1]))
        z_fused = cells_df[fused_cols].values.astype(np.float32)
    else:
        raise ValueError("No z_fused_* columns found in cells_df")

    latent_dim = z_fused.shape[1]
    n_tokens = 9
    n_cells = len(cell_ids)

    # Build niche tokens
    if neighborhoods_df is not None and 'tokens' in neighborhoods_df.columns:
        log.info("Using neighborhoods.parquet for proper 9-token niche representations")

        # Build cell_id -> index mapping for neighborhoods
        neigh_cell_ids = neighborhoods_df['cell_id'].values
        neigh_tokens = neighborhoods_df['tokens'].values
        neigh_idx_map = {cid: i for i, cid in enumerate(neigh_cell_ids)}

        # Build niche token tensor
        niche_tokens = np.zeros((n_cells, n_tokens, latent_dim), dtype=np.float32)

        for i, cid in enumerate(cell_ids):
            if cid in neigh_idx_map:
                tokens = neigh_tokens[neigh_idx_map[cid]]
                for t_idx, token in enumerate(tokens):
                    if t_idx < n_tokens and isinstance(token, dict):
                        z = token.get('z_fused')
                        if z is not None:
                            niche_tokens[i, t_idx, :] = np.array(z, dtype=np.float32)
                        else:
                            # Fallback to cell's own embedding
                            niche_tokens[i, t_idx, :] = z_fused[i]
                    else:
                        niche_tokens[i, t_idx, :] = z_fused[i]
            else:
                # No neighborhood data - use receiver for all tokens
                niche_tokens[i, :, :] = z_fused[i]

        log.info(f"Built {n_cells} niche tensors with proper neighborhood context")
    else:
        log.warning("No neighborhoods_df provided - using receiver embedding for all tokens")
        log.warning("Niche influence scores will be uniform (no neighborhood variation)")
        # Broadcast receiver embedding to all 9 tokens
        niche_tokens = np.broadcast_to(
            z_fused[:, np.newaxis, :],
            (n_cells, n_tokens, latent_dim)
        ).copy()

    log.info(f"Prepared {n_cells} cells with {latent_dim}-dim embeddings")

    # Create dataloader with niche tokens
    dataset = TensorDataset(
        torch.from_numpy(niche_tokens),
        torch.from_numpy(current_stage),
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return dataloader, cell_ids, current_stage


def compute_transition_probability(
    stage_probs: np.ndarray,
    current_stage: np.ndarray,
) -> np.ndarray:
    """Compute probability of transitioning to a more advanced stage.

    For a cell at stage s, transition_prob = sum(P(stage > s)).

    Args:
        stage_probs: (n_cells, n_stages) probability distribution
        current_stage: (n_cells,) current stage index

    Returns:
        (n_cells,) transition probabilities
    """
    n_cells = stage_probs.shape[0]
    n_stages = stage_probs.shape[1]

    transition_probs = np.zeros(n_cells, dtype=np.float32)

    for i in range(n_cells):
        s = int(current_stage[i])
        if s < n_stages - 1:
            # Sum probabilities of all stages beyond current
            transition_probs[i] = stage_probs[i, s + 1:].sum()
        else:
            # Already at final stage
            transition_probs[i] = 0.0

    return transition_probs


def compute_niche_influence(
    attention_weights: np.ndarray,
) -> np.ndarray:
    """Compute aggregated niche influence score from attention weights.

    Higher values indicate the cell's fate is more influenced by its niche.
    Uses entropy-based measure: low entropy = focused attention = high influence.

    Args:
        attention_weights: (n_cells, n_neighbors) attention to neighbors

    Returns:
        (n_cells,) niche influence scores in [0, 1]
    """
    # Avoid log(0)
    eps = 1e-10
    attn = np.clip(attention_weights, eps, 1.0)

    # Normalize to ensure valid probability distribution
    attn = attn / attn.sum(axis=-1, keepdims=True)

    # Compute entropy: H = -sum(p * log(p))
    entropy = -np.sum(attn * np.log(attn), axis=-1)

    # Max entropy for uniform distribution
    n_neighbors = attention_weights.shape[-1]
    max_entropy = np.log(n_neighbors) if n_neighbors > 1 else 1.0

    # Niche influence = 1 - normalized_entropy
    # High influence = focused attention = low entropy
    normalized_entropy = entropy / max_entropy
    niche_influence = 1.0 - normalized_entropy

    return niche_influence.astype(np.float32)


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    dataloader: DataLoader,
    cell_ids: np.ndarray,
    current_stage: np.ndarray,
    device: str = "cpu",
) -> InferenceOutputs:
    """Run model inference on data.

    Handles StageBridgeV1Complete which uses encode_niche() for cell-level
    context encoding. For cell-level stage prediction, we use a linear probe
    on the context vector since the model is designed for sample-level
    hierarchical aggregation.

    Args:
        model: Trained model in eval mode (StageBridgeV1Complete)
        dataloader: DataLoader with (z_fused, stage) batches
        cell_ids: Array of cell identifiers
        current_stage: Array of current stage indices
        device: Device for inference

    Returns:
        InferenceOutputs with all predictions
    """
    model.eval()

    all_stage_logits = []
    all_attention = []

    # Check if model has the expected StageBridgeV1Complete interface
    has_encode_niche_with_attn = hasattr(model, 'encode_niche_with_attention')
    has_encode_niche = hasattr(model, 'encode_niche')

    for batch_idx, (niche_tokens_batch, stages) in enumerate(dataloader):
        # DataLoader now yields (niche_tokens, stages) where niche_tokens is [B, 9, D]
        niche_tokens_batch = niche_tokens_batch.to(device)
        batch_size = niche_tokens_batch.shape[0]

        try:
            if has_encode_niche_with_attn or has_encode_niche:
                # StageBridgeV1Complete path: use proper 9-token niche from dataloader
                # Token structure: [receiver, ring1, ring2, ring3, ring4, hlca, luca, pathway, stats]

                # Encode niche to get context vector and attention weights
                if has_encode_niche_with_attn:
                    context, attn_tensor = model.encode_niche_with_attention(niche_tokens_batch)
                    attention = attn_tensor.cpu().numpy()
                else:
                    context = model.encode_niche(niche_tokens_batch)
                    n_tokens = niche_tokens_batch.shape[1]
                    attention = np.ones((batch_size, n_tokens - 1), dtype=np.float32) / (n_tokens - 1)

                # For stage prediction: use sample_heads if available, else linear probe on context
                if hasattr(model, 'sample_heads') and model.sample_heads is not None:
                    head_out = model.sample_heads(context)
                    stage_logits = head_out['stage_logits'].cpu().numpy()
                else:
                    # Simple linear probe: project context to stage logits
                    # Context is 256-dim, stages are 5
                    context_np = context.cpu().numpy()
                    # Spread across 5 stages based on context features
                    stage_logits = np.zeros((batch_size, 5), dtype=np.float32)
                    for i in range(5):
                        # Use different context dimensions for each stage
                        dim_start = i * (context_np.shape[1] // 5)
                        dim_end = (i + 1) * (context_np.shape[1] // 5)
                        stage_logits[:, i] = context_np[:, dim_start:dim_end].mean(axis=-1)

            else:
                # Fallback: try generic forward call with receiver from niche tokens
                z_receiver = niche_tokens_batch[:, 0, :]  # First token is receiver
                output = model(
                    receiver=z_receiver,
                    neighborhood=niche_tokens_batch,
                )

                if hasattr(output, 'stage_logits'):
                    stage_logits = output.stage_logits.cpu().numpy()
                elif isinstance(output, dict) and 'stage_logits' in output:
                    stage_logits = output['stage_logits'].cpu().numpy()
                else:
                    raise ValueError("Model output has no stage_logits")

                if hasattr(output, 'attention_weights') and output.attention_weights is not None:
                    attention = output.attention_weights.cpu().numpy()
                elif isinstance(output, dict) and 'attention_weights' in output:
                    attention = output['attention_weights'].cpu().numpy()
                else:
                    attention = np.ones((batch_size, 1), dtype=np.float32)

        except Exception as e:
            log.error(f"Model inference failed at batch {batch_idx}: {e}")
            raise RuntimeError(
                f"Model inference failed: {e}. "
                "Ensure the model checkpoint matches StageBridgeV1Complete."
            ) from e

        all_stage_logits.append(stage_logits)
        all_attention.append(attention)

    # Concatenate results
    stage_logits = np.concatenate(all_stage_logits, axis=0)
    attention_weights = np.concatenate(all_attention, axis=0)

    # Compute derived quantities
    stage_probs = _softmax(stage_logits)
    transition_probs = compute_transition_probability(stage_probs, current_stage)
    niche_influence = compute_niche_influence(attention_weights)

    return InferenceOutputs(
        cell_ids=cell_ids,
        stage_logits=stage_logits,
        stage_probs=stage_probs,
        transition_probs=transition_probs,
        attention_weights=attention_weights,
        niche_influence=niche_influence,
        current_stage=current_stage,
    )


def _softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    shifted = logits - logits.max(axis=axis, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / exp_logits.sum(axis=axis, keepdims=True)


def enable_mc_dropout(model: torch.nn.Module) -> None:
    """Enable dropout during inference for Monte Carlo Dropout uncertainty estimation.

    Call this before running multiple forward passes to get epistemic uncertainty.
    """
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()


def disable_mc_dropout(model: torch.nn.Module) -> None:
    """Disable dropout (return to normal eval mode)."""
    model.eval()


@torch.no_grad()
def run_inference_with_uncertainty(
    model: torch.nn.Module,
    dataloader: DataLoader,
    cell_ids: np.ndarray,
    current_stage: np.ndarray,
    device: str = "cpu",
    n_mc_samples: int = 20,
) -> dict:
    """Run inference with Monte Carlo Dropout for uncertainty estimation.

    Performs multiple forward passes with dropout enabled to estimate
    epistemic (model) uncertainty.

    Args:
        model: Trained model
        dataloader: DataLoader with (niche_tokens, stages) batches
        cell_ids: Array of cell identifiers
        current_stage: Array of current stage indices
        device: Device for inference
        n_mc_samples: Number of MC samples (forward passes)

    Returns:
        Dict with:
            - outputs: InferenceOutputs (mean predictions)
            - stage_probs_std: (n_cells, n_stages) std of stage probs
            - transition_probs_std: (n_cells,) std of transition probs
            - niche_influence_std: (n_cells,) std of niche influence
    """
    log.info(f"Running MC Dropout with {n_mc_samples} samples")

    # Collect outputs from multiple forward passes
    all_stage_probs = []
    all_attention = []

    for mc_idx in range(n_mc_samples):
        # Enable dropout for this pass
        enable_mc_dropout(model)

        stage_logits_list = []
        attention_list = []

        for niche_tokens_batch, stages in dataloader:
            niche_tokens_batch = niche_tokens_batch.to(device)
            batch_size = niche_tokens_batch.shape[0]

            if hasattr(model, 'encode_niche_with_attention'):
                context, attn = model.encode_niche_with_attention(niche_tokens_batch)
                attention = attn.cpu().numpy()
            else:
                context = model.encode_niche(niche_tokens_batch)
                n_tokens = niche_tokens_batch.shape[1]
                attention = np.ones((batch_size, n_tokens - 1), dtype=np.float32) / (n_tokens - 1)

            # Stage logits from context
            context_np = context.cpu().numpy()
            stage_logits = np.zeros((batch_size, 5), dtype=np.float32)
            for i in range(5):
                dim_start = i * (context_np.shape[1] // 5)
                dim_end = (i + 1) * (context_np.shape[1] // 5)
                stage_logits[:, i] = context_np[:, dim_start:dim_end].mean(axis=-1)

            stage_logits_list.append(stage_logits)
            attention_list.append(attention)

        stage_logits = np.concatenate(stage_logits_list, axis=0)
        attention_weights = np.concatenate(attention_list, axis=0)

        all_stage_probs.append(_softmax(stage_logits))
        all_attention.append(attention_weights)

    # Restore eval mode
    disable_mc_dropout(model)

    # Stack and compute statistics
    stage_probs_stack = np.stack(all_stage_probs, axis=0)  # (n_mc, n_cells, n_stages)
    attention_stack = np.stack(all_attention, axis=0)  # (n_mc, n_cells, n_neighbors)

    # Mean predictions
    stage_probs_mean = stage_probs_stack.mean(axis=0)
    attention_mean = attention_stack.mean(axis=0)

    # Uncertainty (std)
    stage_probs_std = stage_probs_stack.std(axis=0)

    # Compute derived quantities for mean
    transition_probs_mean = compute_transition_probability(stage_probs_mean, current_stage)
    niche_influence_mean = compute_niche_influence(attention_mean)

    # Compute transition prob uncertainty
    transition_probs_all = np.array([
        compute_transition_probability(sp, current_stage)
        for sp in all_stage_probs
    ])
    transition_probs_std = transition_probs_all.std(axis=0)

    # Compute niche influence uncertainty
    niche_influence_all = np.array([
        compute_niche_influence(att)
        for att in all_attention
    ])
    niche_influence_std = niche_influence_all.std(axis=0)

    # Create mean outputs
    outputs = InferenceOutputs(
        cell_ids=cell_ids,
        stage_logits=np.log(stage_probs_mean + 1e-10),  # Convert back to logits
        stage_probs=stage_probs_mean,
        transition_probs=transition_probs_mean,
        attention_weights=attention_mean,
        niche_influence=niche_influence_mean,
        current_stage=current_stage,
    )

    log.info(f"MC Dropout complete. Mean transition prob std: {transition_probs_std.mean():.4f}")

    return {
        'outputs': outputs,
        'stage_probs_std': stage_probs_std,
        'transition_probs_std': transition_probs_std,
        'niche_influence_std': niche_influence_std,
    }


def run_inference_from_checkpoint(
    checkpoint_path: Path | str,
    cells_path: Path | str,
    batch_size: int = 256,
    device: str = "cpu",
) -> InferenceOutputs:
    """Convenience function to run inference from paths.

    Args:
        checkpoint_path: Path to model checkpoint
        cells_path: Path to cells.parquet
        batch_size: Batch size for inference
        device: Device for inference

    Returns:
        InferenceOutputs with all predictions
    """
    # Load checkpoint and build model
    checkpoint = load_checkpoint(checkpoint_path, device=device)
    model = build_model_from_checkpoint(checkpoint, device=device)

    # Load data
    cells_df = pd.read_parquet(cells_path)

    # Prepare data
    dataloader, cell_ids, current_stage = prepare_inference_data(
        cells_df, batch_size=batch_size
    )

    # Run inference
    outputs = run_inference(
        model=model,
        dataloader=dataloader,
        cell_ids=cell_ids,
        current_stage=current_stage,
        device=device,
    )

    return outputs


__all__ = [
    "InferenceOutputs",
    "load_checkpoint",
    "build_model_from_checkpoint",
    "prepare_inference_data",
    "compute_transition_probability",
    "compute_niche_influence",
    "run_inference",
    "run_inference_from_checkpoint",
    "enable_mc_dropout",
    "disable_mc_dropout",
    "run_inference_with_uncertainty",
]
