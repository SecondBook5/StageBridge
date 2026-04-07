#!/usr/bin/env python3
"""Generate semi-synthetic benchmark with ground truth.

Snakemake script - uses snakemake.output for paths.
"""

from pathlib import Path
import torch
import json

from stagebridge.benchmarks import generate_benchmark, SmokeTestConfig

# Get output paths from Snakemake
output_dir = Path(snakemake.output.benchmark).parent
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("Generating Semi-Synthetic Benchmark")
print("=" * 60)

# Generate fully synthetic benchmark with ground truth
config = SmokeTestConfig()
config.n_cells = 10000  # Larger for proper evaluation
report = generate_benchmark(config=config, mode='fully_synthetic')

# Save tensors
if 'tensors' in report:
    torch.save(report['tensors'], output_dir / 'semi_synthetic.pt')
    print(f"Saved tensors to {output_dir / 'semi_synthetic.pt'}")

# Save ground truth
ground_truth = {k: v for k, v in report.items() if k != 'tensors'}
with open(output_dir / 'ground_truth.json', 'w') as f:
    json.dump(ground_truth, f, indent=2, default=str)
print(f"Saved ground truth to {output_dir / 'ground_truth.json'}")

n_tensors = len(report.get('tensors', {}))
print(f"Generated benchmark with {n_tensors} tensors")
print("=" * 60)
