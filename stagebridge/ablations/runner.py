"""
Ablation Runner - executes ablation experiments.
"""

import logging
from pathlib import Path
from typing import Any
from datetime import datetime

from .registry import AblationRegistry, AblationTier

log = logging.getLogger(__name__)


def run_ablation(
    ablation_name: str,
    base_config: dict[str, Any],
    output_dir: Path,
    seed: int = 42,
    smoke: bool = False,
) -> dict[str, Any]:
    """
    Run a single ablation experiment.

    Parameters
    ----------
    ablation_name : str
        Name of the registered ablation
    base_config : dict
        Base configuration to modify
    output_dir : Path
        Directory for outputs
    seed : int
        Random seed for reproducibility
    smoke : bool
        If True, run in smoke test mode (reduced data/epochs)

    Returns
    -------
    dict
        Results including metrics, config, and artifacts
    """
    ablation = AblationRegistry.get(ablation_name)
    log.info(f"Running ablation: {ablation.name}")
    log.info(f"  Description: {ablation.description}")
    log.info(f"  Hypothesis: {ablation.hypothesis}")

    # Apply config deltas
    config = _apply_deltas(base_config.copy(), ablation.config_deltas)

    # Create output directory
    ablation_dir = output_dir / ablation.name
    ablation_dir.mkdir(parents=True, exist_ok=True)

    # Run the model with modified config
    # This would call the actual training pipeline
    results = _execute_training(config, ablation_dir, seed, smoke)

    # Add ablation metadata
    results["ablation"] = ablation.to_dict()
    results["timestamp"] = datetime.now().isoformat()

    return results


def run_ablation_suite(
    tier: AblationTier,
    base_config: dict[str, Any],
    output_dir: Path,
    seed: int = 42,
    smoke: bool = False,
) -> dict[str, dict[str, Any]]:
    """
    Run all ablations in a tier.

    Parameters
    ----------
    tier : AblationTier
        Which tier of ablations to run
    base_config : dict
        Base configuration
    output_dir : Path
        Output directory
    seed : int
        Random seed
    smoke : bool
        Smoke test mode

    Returns
    -------
    dict
        Results for each ablation, keyed by name
    """
    ablations = AblationRegistry.get_tier(tier)
    log.info(f"Running {len(ablations)} ablations from Tier {tier.value}")

    results = {}
    for ablation in ablations:
        try:
            results[ablation.name] = run_ablation(
                ablation.name, base_config, output_dir, seed, smoke
            )
        except Exception as e:
            log.error(f"Ablation {ablation.name} failed: {e}")
            results[ablation.name] = {"error": str(e), "ablation": ablation.to_dict()}

    return results


def _apply_deltas(config: dict, deltas: dict) -> dict:
    """Apply config deltas using dot notation keys."""
    for key, value in deltas.items():
        parts = key.split(".")
        target = config
        for part in parts[:-1]:
            if part not in target:
                target[part] = {}
            target = target[part]
        target[parts[-1]] = value
    return config


def _execute_training(
    config: dict,
    output_dir: Path,
    seed: int,
    smoke: bool,
) -> dict[str, Any]:
    """Execute training with given config."""
    from stagebridge.validation.repro import set_all_seeds, save_repro_manifest

    # Set seeds for reproducibility
    set_all_seeds(seed)

    # Save reproducibility manifest
    save_repro_manifest(config, seed, output_dir)

    try:
        # Import training components
        from stagebridge.pipelines.run_v1_full import StageBridgeV1Full
        from stagebridge.data.loaders_optimized import get_dataloader_optimized
        import torch

        # Build model from config
        model_config = _extract_model_config(config)
        model = StageBridgeV1Full(**model_config)

        # Get data
        data_config = config.get("data", {})
        train_loader = get_dataloader_optimized(
            split="train",
            batch_size=config.get("batch_size", 32),
            **data_config
        )

        # Training parameters
        n_epochs = 5 if smoke else config.get("n_epochs", 100)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.get("lr", 1e-4),
            weight_decay=config.get("weight_decay", 1e-5),
        )

        # Training loop
        losses = []
        model.train()
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            n_batches = 0
            for batch in train_loader:
                optimizer.zero_grad()
                loss = model.training_step(batch, device=device)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            losses.append(avg_loss)
            log.info(f"  Epoch {epoch+1}/{n_epochs}: loss={avg_loss:.4f}")

        # Save model
        torch.save(model.state_dict(), output_dir / "model.pt")

        # Compute final metrics
        metrics = {
            "final_loss": losses[-1] if losses else float("nan"),
            "min_loss": min(losses) if losses else float("nan"),
            "n_epochs": n_epochs,
        }

        return {
            "metrics": metrics,
            "config": config,
            "output_dir": str(output_dir),
            "seed": seed,
            "smoke": smoke,
            "training_history": {"losses": losses},
        }

    except Exception as e:
        log.error(f"Training failed: {e}")
        return {
            "metrics": {},
            "config": config,
            "output_dir": str(output_dir),
            "seed": seed,
            "smoke": smoke,
            "error": str(e),
        }


def _extract_model_config(config: dict) -> dict:
    """Extract model-specific config from full config."""
    model_keys = [
        "reference_mode", "latent_dim", "hlca_dim", "luca_dim", "fusion_mode",
        "niche_encoder_type", "receiver_dim", "sender_dim", "niche_hidden_dim",
        "niche_heads", "niche_layers", "use_set_encoder", "set_hidden_dim",
        "set_heads", "use_ude", "use_cross_attention", "num_edges",
        "use_wes", "wes_dim", "wes_hidden_dim", "dropout",
    ]
    return {k: config[k] for k in model_keys if k in config}
