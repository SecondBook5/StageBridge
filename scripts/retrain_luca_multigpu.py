#!/usr/bin/env python
"""Retrain LuCA scANVI model with proper multi-GPU support.

For SLURM multi-GPU training, there are two approaches:

APPROACH 1 (Recommended): Single SLURM task, Lightning spawns GPU processes
    #SBATCH --ntasks-per-node=1
    #SBATCH --gres=gpu:4

    Then in code: use accelerator="auto", devices="auto", strategy="ddp_spawn"

APPROACH 2: SLURM manages GPU processes (one task per GPU)
    #SBATCH --ntasks-per-node=4
    #SBATCH --gres=gpu:4

    Then in code: devices=1 (each task sees one GPU)
    Launch with: srun python script.py

This script uses APPROACH 1 which is simpler for scvi-tools.

Usage:
    # Direct invocation (auto-detects GPUs)
    python scripts/retrain_luca_multigpu.py --atlas /path/to/luca_core_atlas.h5ad --output /path/to/output

    # With SLURM (create sbatch script as needed)
    srun python scripts/retrain_luca_multigpu.py --atlas /path/to/atlas.h5ad --output /path/to/output
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


def setup_gpu_environment():
    """Configure environment for multi-GPU training."""
    # Ensure CUDA devices are visible
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        # Try to detect available GPUs
        try:
            import torch
            n_gpus = torch.cuda.device_count()
            if n_gpus > 0:
                os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(n_gpus))
                print(f"Set CUDA_VISIBLE_DEVICES to {os.environ['CUDA_VISIBLE_DEVICES']}")
        except Exception:
            pass

    # For Lightning DDP in SLURM, these help
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")


def get_trainer_kwargs(n_gpus: int | None = None) -> dict:
    """Get PyTorch Lightning Trainer kwargs for multi-GPU training.

    Parameters
    ----------
    n_gpus : int, optional
        Number of GPUs to use. If None, auto-detect.

    Returns
    -------
    dict
        Kwargs to pass to scvi model.train()
    """
    import torch

    if n_gpus is None:
        n_gpus = torch.cuda.device_count()

    print(f"Detected {n_gpus} GPU(s)")

    if n_gpus == 0:
        # CPU training
        return {
            "accelerator": "cpu",
        }
    elif n_gpus == 1:
        # Single GPU - simple case
        return {
            "accelerator": "gpu",
            "devices": 1,
        }
    else:
        # Multi-GPU: use ddp_spawn strategy
        # ddp_spawn works with SLURM --ntasks-per-node=1
        # It spawns subprocesses internally rather than relying on SLURM
        return {
            "accelerator": "gpu",
            "devices": n_gpus,
            # ddp_spawn is more compatible with scvi-tools than plain ddp
            # It spawns new processes rather than forking
            "strategy": "ddp_spawn",
        }


def retrain_luca_scanvi(
    atlas_path: Path,
    output_dir: Path,
    *,
    n_latent: int = 10,
    n_hidden: int = 128,
    n_layers: int = 2,
    n_hvg: int = 6000,
    max_epochs: int = 400,
    batch_key: str = "dataset",
    labels_key: str = "cell_type",
    n_gpus: int | None = None,
):
    """Retrain LuCA scANVI model from scratch with multi-GPU support.

    Parameters
    ----------
    atlas_path : Path
        Path to LuCA core atlas h5ad file
    output_dir : Path
        Output directory for model and artifacts
    n_latent : int
        Latent dimension (default 10, matching original LuCA)
    n_hidden : int
        Hidden layer size (default 128)
    n_layers : int
        Number of layers (default 2)
    n_hvg : int
        Number of highly variable genes (default 6000)
    max_epochs : int
        Maximum training epochs (default 400)
    batch_key : str
        Batch key in adata.obs (default "dataset")
    labels_key : str
        Label key in adata.obs (default "cell_type")
    n_gpus : int, optional
        Number of GPUs (auto-detect if None)
    """
    import anndata
    import scanpy as sc
    import scvi

    print("=" * 60)
    print("Retrain LuCA scANVI Model (Multi-GPU)")
    print(f"Atlas: {atlas_path}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load atlas
    print("\nStep 1: Load atlas")
    adata = anndata.read_h5ad(atlas_path)
    print(f"  Shape: {adata.shape}")
    print(f"  Batch key '{batch_key}': {adata.obs[batch_key].nunique()} batches")
    print(f"  Labels key '{labels_key}': {adata.obs[labels_key].nunique()} cell types")

    # Use raw counts if available
    if adata.raw is not None:
        print("  Using adata.raw for counts")
        adata = adata.raw.to_adata()
    elif "counts" in adata.layers:
        print("  Using adata.layers['counts']")
        adata.X = adata.layers["counts"]

    # Compute HVGs
    print(f"\nStep 2: Compute {n_hvg} HVGs")
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_hvg,
        flavor="seurat_v3",
        batch_key=batch_key,
        subset=True,
    )
    print(f"  Shape after HVG: {adata.shape}")

    # Setup scVI first (required for scANVI)
    print("\nStep 3: Setup scVI model")
    scvi.model.SCVI.setup_anndata(
        adata,
        batch_key=batch_key,
    )

    # Train scVI base model
    print("\nStep 4: Train scVI base model")
    scvi_model = scvi.model.SCVI(
        adata,
        n_latent=n_latent,
        n_hidden=n_hidden,
        n_layers=n_layers,
        dropout_rate=0.2,
        dispersion="gene",
        gene_likelihood="nb",
    )

    # Get multi-GPU trainer kwargs
    trainer_kwargs = get_trainer_kwargs(n_gpus)
    print(f"  Trainer kwargs: {trainer_kwargs}")

    try:
        scvi_model.train(
            max_epochs=max_epochs,
            early_stopping=True,
            early_stopping_patience=20,
            check_val_every_n_epoch=1,
            train_size=0.9,
            **trainer_kwargs,
        )
        print("  scVI training complete")
    except Exception as e:
        print(f"  Multi-GPU failed: {e}")
        print("  Falling back to single GPU...")
        scvi_model.train(
            max_epochs=max_epochs,
            early_stopping=True,
            early_stopping_patience=20,
            check_val_every_n_epoch=1,
            train_size=0.9,
            accelerator="gpu",
            devices=1,
        )

    # Save scVI model
    scvi_path = output_dir / "scvi_model"
    scvi_model.save(str(scvi_path), overwrite=True)
    print(f"  Saved scVI model to {scvi_path}")

    # Train scANVI on top
    print("\nStep 5: Train scANVI model")
    scanvi_model = scvi.model.SCANVI.from_scvi_model(
        scvi_model,
        unlabeled_category="Unknown",
        labels_key=labels_key,
    )

    try:
        scanvi_model.train(
            max_epochs=max_epochs // 2,  # scANVI converges faster
            early_stopping=True,
            early_stopping_patience=15,
            check_val_every_n_epoch=1,
            train_size=0.9,
            **trainer_kwargs,
        )
        print("  scANVI training complete")
    except Exception as e:
        print(f"  Multi-GPU failed: {e}")
        print("  Falling back to single GPU...")
        scanvi_model.train(
            max_epochs=max_epochs // 2,
            early_stopping=True,
            early_stopping_patience=15,
            check_val_every_n_epoch=1,
            train_size=0.9,
            accelerator="gpu",
            devices=1,
        )

    # Save scANVI model
    scanvi_path = output_dir / "scanvi_model"
    scanvi_model.save(str(scanvi_path), overwrite=True)
    print(f"  Saved scANVI model to {scanvi_path}")

    # Extract and save latent representations
    print("\nStep 6: Extract latent representations")
    X_scvi = scvi_model.get_latent_representation()
    X_scanvi = scanvi_model.get_latent_representation()

    adata.obsm["X_scVI"] = X_scvi
    adata.obsm["X_scANVI"] = X_scanvi

    # Validate latents
    n_nan_scvi = np.isnan(X_scvi).any(axis=1).sum()
    n_nan_scanvi = np.isnan(X_scanvi).any(axis=1).sum()
    print(f"  X_scVI shape: {X_scvi.shape}, NaN rows: {n_nan_scvi}")
    print(f"  X_scANVI shape: {X_scanvi.shape}, NaN rows: {n_nan_scanvi}")

    if n_nan_scanvi > 0:
        print(f"  WARNING: {n_nan_scanvi} cells have NaN in scANVI latent!")

    # Save updated atlas with new embeddings
    output_atlas = output_dir / "luca_core_atlas_retrained.h5ad"
    adata.write_h5ad(output_atlas)
    print(f"  Saved atlas with embeddings to {output_atlas}")

    # Save training history
    history = {
        "scvi": scvi_model.history if hasattr(scvi_model, 'history') else None,
        "scanvi": scanvi_model.history if hasattr(scanvi_model, 'history') else None,
    }

    import json
    with open(output_dir / "training_history.json", "w") as f:
        json.dump({k: {kk: list(vv) for kk, vv in v.items()} if v else None
                   for k, v in history.items()}, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"  scVI model: {scvi_path}")
    print(f"  scANVI model: {scanvi_path}")
    print(f"  Atlas with embeddings: {output_atlas}")
    print("=" * 60)

    return scanvi_model


def main():
    parser = argparse.ArgumentParser(
        description="Retrain LuCA scANVI model with multi-GPU support"
    )
    parser.add_argument(
        "--atlas", type=Path, required=True,
        help="Path to LuCA core atlas h5ad"
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Output directory for models"
    )
    parser.add_argument(
        "--n-latent", type=int, default=10,
        help="Latent dimension (default: 10)"
    )
    parser.add_argument(
        "--n-hidden", type=int, default=128,
        help="Hidden layer size (default: 128)"
    )
    parser.add_argument(
        "--n-hvg", type=int, default=6000,
        help="Number of HVGs (default: 6000)"
    )
    parser.add_argument(
        "--max-epochs", type=int, default=400,
        help="Max training epochs (default: 400)"
    )
    parser.add_argument(
        "--batch-key", type=str, default="dataset",
        help="Batch key in adata.obs (default: dataset)"
    )
    parser.add_argument(
        "--labels-key", type=str, default="cell_type",
        help="Labels key in adata.obs (default: cell_type)"
    )
    parser.add_argument(
        "--n-gpus", type=int, default=None,
        help="Number of GPUs (default: auto-detect)"
    )

    args = parser.parse_args()

    # Setup environment
    setup_gpu_environment()

    # Run training
    retrain_luca_scanvi(
        atlas_path=args.atlas,
        output_dir=args.output,
        n_latent=args.n_latent,
        n_hidden=args.n_hidden,
        n_hvg=args.n_hvg,
        max_epochs=args.max_epochs,
        batch_key=args.batch_key,
        labels_key=args.labels_key,
        n_gpus=args.n_gpus,
    )


if __name__ == "__main__":
    main()
