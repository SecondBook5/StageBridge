#!/usr/bin/env python
"""Compute entropy for all completed checkpoints.

Outputs a JSON file with entropy stats for each fold/seed.
Parallelized across checkpoints using concurrent.futures.
"""

import json
import os
import torch
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from stagebridge.models import StageBridge, StageBridgeConfig
from stagebridge.loaders.dataset import create_dataloaders

N_WORKERS = int(os.environ.get('SLURM_CPUS_PER_TASK', 4))


def compute_entropy(ckpt_path: Path, data_dir: Path, fold_idx: int, device: str = 'cuda') -> dict:
    """Compute entropy stats for a single checkpoint."""
    print(f"\nProcessing: {ckpt_path.parent.parent.name}/{ckpt_path.parent.name}")

    # Load checkpoint (strict=False for backward compat with old evolution_branch)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = StageBridgeConfig(**ckpt['config']['model_config'])
    model = StageBridge(config).to(device)
    missing, unexpected = model.load_state_dict(ckpt['model_state_dict'], strict=False)
    if unexpected:
        print(f"  Ignoring {len(unexpected)} unexpected keys (old model version)")
    model.eval()

    # Get val loader
    _, val_loader, _ = create_dataloaders(data_dir, fold_idx=fold_idx, batch_size=64)

    # Collect entropy
    entropies = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Computing entropy"):
            batch = batch.to(device)
            output = model(batch, return_reconstruction=True)
            if output.entropy_loss is not None:
                entropies.append(output.entropy_loss.item())

    if not entropies:
        return {"error": "No entropy values collected"}

    entropies_t = torch.tensor(entropies)
    return {
        "mean": entropies_t.mean().item(),
        "std": entropies_t.std().item(),
        "min": entropies_t.min().item(),
        "max": entropies_t.max().item(),
        "n_batches": len(entropies),
    }


def process_checkpoint(args):
    """Worker function for parallel processing."""
    ckpt_path, data_dir, fold_idx, key = args
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        entropy_stats = compute_entropy(ckpt_path, data_dir, fold_idx, device)
        return key, entropy_stats
    except Exception as e:
        return key, {"error": str(e)}
    finally:
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    output_dir = Path("/data1/chaunzt1/stagebridge/outputs/v1")
    data_dir = Path("/data1/chaunzt1/stagebridge/processed/luad_evo/canonical")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Using {N_WORKERS} workers")

    # Collect all checkpoint tasks
    tasks = []
    for ckpt_path in sorted(output_dir.glob("full/fold_*/seed_*/checkpoints/best_checkpoint.pt")):
        parts = ckpt_path.parts
        fold_idx = int([p for p in parts if p.startswith("fold_")][0].split("_")[1])
        seed = int([p for p in parts if p.startswith("seed_")][0].split("_")[1])
        key = f"fold_{fold_idx}_seed_{seed}"
        tasks.append((ckpt_path, data_dir, fold_idx, key))

    print(f"Found {len(tasks)} checkpoints to process")

    results = {}

    # Process in parallel (but GPU work is sequential within each worker)
    with ProcessPoolExecutor(max_workers=min(N_WORKERS, len(tasks))) as executor:
        futures = {executor.submit(process_checkpoint, task): task[3] for task in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing checkpoints"):
            key = futures[future]
            try:
                result_key, entropy_stats = future.result()
                results[result_key] = entropy_stats
                if "mean" in entropy_stats:
                    print(f"  {result_key}: mean={entropy_stats['mean']:.6f}")
                else:
                    print(f"  {result_key}: ERROR - {entropy_stats.get('error', 'unknown')}")
            except Exception as e:
                print(f"  {key}: FAILED - {e}")
                results[key] = {"error": str(e)}

    # Save results
    output_file = output_dir / "entropy_stats.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to: {output_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("ENTROPY SUMMARY")
    print("=" * 60)
    valid_means = [r["mean"] for r in results.values() if "mean" in r]
    if valid_means:
        import numpy as np
        print(f"Overall mean: {np.mean(valid_means):.6f} +/- {np.std(valid_means):.6f}")
        print(f"N checkpoints: {len(valid_means)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
