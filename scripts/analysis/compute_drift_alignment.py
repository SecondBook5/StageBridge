#!/usr/bin/env python3
"""
Compute drift alignment between predicted velocities and OT transport directions.

This is the TRACEABLE SOURCE for drift alignment numbers in the paper.

Drift alignment = cosine similarity between:
  - Model's predicted velocity (drift_head output)
  - OT-derived transport direction (target_z - source_z from Sinkhorn coupling)

Usage:
    python scripts/compute_drift_alignment.py \
        --checkpoint /path/to/best_checkpoint.pt \
        --data-dir /path/to/canonical \
        --output drift_alignment.json
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from stagebridge.models import StageBridge, StageBridgeConfig
from stagebridge.loaders.dataset import create_dataloaders


def compute_cosine_similarity(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    """Compute cosine similarity between two vectors (batched)."""
    v1_norm = F.normalize(v1, p=2, dim=-1)
    v2_norm = F.normalize(v2, p=2, dim=-1)
    return (v1_norm * v2_norm).sum(dim=-1)


def compute_drift_alignment_for_checkpoint(
    checkpoint_path: Path,
    data_dir: Path,
    fold_idx: int,
    device: str = 'cuda',
    max_batches: int = None,
) -> dict:
    """Compute drift alignment for a single checkpoint."""

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = StageBridgeConfig(**ckpt['config']['model_config'])
    model = StageBridge(config).to(device)

    # Load with strict=False for backward compat
    missing, unexpected = model.load_state_dict(ckpt['model_state_dict'], strict=False)
    if unexpected:
        print(f"  Ignoring {len(unexpected)} unexpected keys")
    model.eval()

    # Get validation loader
    _, val_loader, _ = create_dataloaders(data_dir, fold_idx=fold_idx, batch_size=64)

    # Collect alignments
    all_alignments = []
    stage_alignments = {0: [], 1: [], 2: []}  # Normal->Pre, Pre->Inv, Normal->Inv

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="Computing alignment")):
            if max_batches and batch_idx >= max_batches:
                break

            batch = batch.to(device)

            # Get model output with velocity prediction
            output = model(batch, return_reconstruction=True)

            # The model predicts velocity in latent space
            # We need to compare with the "ground truth" direction from OT

            # For CFM training, the target is: z_target - z_source (transport direction)
            # The model's drift_head predicts this direction

            if hasattr(output, 'predicted_velocity') and output.predicted_velocity is not None:
                pred_v = output.predicted_velocity
            elif hasattr(output, 'drift') and output.drift is not None:
                pred_v = output.drift
            else:
                # Compute drift manually if not in output
                # The drift head takes receiver embedding + context
                if hasattr(model, 'drift_head'):
                    receiver_emb = output.receiver_embedding if hasattr(output, 'receiver_embedding') else output.z
                    context = output.context if hasattr(output, 'context') else None
                    if context is not None:
                        pred_v = model.drift_head(receiver_emb, context)
                    else:
                        continue
                else:
                    continue

            # Get target direction from batch (if available)
            # In CFM training, target_z is the destination embedding
            if hasattr(batch, 'target_z') and batch.target_z is not None:
                source_z = output.z if hasattr(output, 'z') else output.receiver_embedding
                target_z = batch.target_z
                transport_dir = target_z - source_z

                # Compute cosine similarity
                alignment = compute_cosine_similarity(pred_v, transport_dir)
                all_alignments.extend(alignment.cpu().numpy().tolist())

                # Track by stage transition
                if hasattr(batch, 'stage') and hasattr(batch, 'target_stage'):
                    for i, (s1, s2) in enumerate(zip(batch.stage, batch.target_stage)):
                        key = int(s1) * 3 + int(s2) if s1 < s2 else None
                        if key == 1:  # Normal (0) -> Preinvasive (1)
                            stage_alignments[0].append(alignment[i].item())
                        elif key == 4:  # Preinvasive (1) -> Invasive (2)
                            stage_alignments[1].append(alignment[i].item())
                        elif key == 2:  # Normal (0) -> Invasive (2)
                            stage_alignments[2].append(alignment[i].item())

    # Compute statistics
    results = {
        'checkpoint': str(checkpoint_path),
        'fold_idx': fold_idx,
        'n_samples': len(all_alignments),
    }

    if all_alignments:
        alignments = np.array(all_alignments)
        results['overall'] = {
            'mean': float(np.mean(alignments)),
            'std': float(np.std(alignments)),
            'median': float(np.median(alignments)),
            'min': float(np.min(alignments)),
            'max': float(np.max(alignments)),
        }

    # Per-transition statistics
    transition_names = {0: 'normal_to_preinvasive', 1: 'preinvasive_to_invasive', 2: 'normal_to_invasive'}
    for key, name in transition_names.items():
        if stage_alignments[key]:
            arr = np.array(stage_alignments[key])
            results[name] = {
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr)),
                'n_samples': len(arr),
            }

    return results


def main():
    parser = argparse.ArgumentParser(description='Compute drift alignment')
    parser.add_argument('--checkpoint', type=Path, default=None, help='Single checkpoint path')
    parser.add_argument('--results-dir', type=Path, default=Path('/home/booka/projects/StageBridge/results/v1'))
    parser.add_argument('--data-dir', type=Path, default=Path('/data1/chaunzt1/stagebridge/processed/luad_evo/canonical'))
    parser.add_argument('--output', type=Path, default=Path('drift_alignment.json'))
    parser.add_argument('--max-batches', type=int, default=None, help='Limit batches for testing')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    print("=" * 60)
    print("COMPUTING DRIFT ALIGNMENT")
    print("=" * 60)
    print(f"Date: {datetime.now().isoformat()}")
    print(f"Data dir: {args.data_dir}")
    print()

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    all_results = {
        'computed_at': datetime.now().isoformat(),
        'data_source': str(args.data_dir),
        'checkpoints': {},
    }

    if args.checkpoint:
        # Single checkpoint
        checkpoints = [args.checkpoint]
    else:
        # Find all checkpoints
        checkpoints = sorted(args.results_dir.glob("full/fold_*/seed_*/checkpoints/best_checkpoint.pt"))

    print(f"Found {len(checkpoints)} checkpoints")

    overall_alignments = []
    normal_preinvasive_alignments = []

    for ckpt_path in checkpoints:
        parts = ckpt_path.parts
        fold_idx = int([p for p in parts if p.startswith("fold_")][0].split("_")[1])
        seed = int([p for p in parts if p.startswith("seed_")][0].split("_")[1])
        key = f"fold_{fold_idx}_seed_{seed}"

        print(f"\nProcessing {key}...")

        try:
            results = compute_drift_alignment_for_checkpoint(
                ckpt_path, args.data_dir, fold_idx, device, args.max_batches
            )
            all_results['checkpoints'][key] = results

            if 'overall' in results:
                overall_alignments.append(results['overall']['mean'])
                print(f"  Overall alignment: {results['overall']['mean']:.4f}")

            if 'normal_to_preinvasive' in results:
                normal_preinvasive_alignments.append(results['normal_to_preinvasive']['mean'])
                print(f"  Normal->Preinvasive: {results['normal_to_preinvasive']['mean']:.4f}")

        except Exception as e:
            print(f"  ERROR: {e}")
            all_results['checkpoints'][key] = {'error': str(e)}

    # Aggregate statistics
    if overall_alignments:
        all_results['summary'] = {
            'overall': {
                'mean': float(np.mean(overall_alignments)),
                'std': float(np.std(overall_alignments)),
                'n_checkpoints': len(overall_alignments),
            }
        }

    if normal_preinvasive_alignments:
        all_results['summary']['normal_to_preinvasive'] = {
            'mean': float(np.mean(normal_preinvasive_alignments)),
            'std': float(np.std(normal_preinvasive_alignments)),
            'n_checkpoints': len(normal_preinvasive_alignments),
        }

    # Save results
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 60)
    print("DRIFT ALIGNMENT SUMMARY")
    print("=" * 60)

    if 'summary' in all_results:
        if 'overall' in all_results['summary']:
            s = all_results['summary']['overall']
            print(f"Overall alignment: {s['mean']:.4f} +/- {s['std']:.4f} (n={s['n_checkpoints']})")

        if 'normal_to_preinvasive' in all_results['summary']:
            s = all_results['summary']['normal_to_preinvasive']
            print(f"Normal->Preinvasive: {s['mean']:.4f} +/- {s['std']:.4f} (n={s['n_checkpoints']})")

    print(f"\nResults saved to: {args.output}")
    print("=" * 60)


if __name__ == '__main__':
    main()
