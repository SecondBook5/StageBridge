# Archived Pipeline Code

This directory contains old pipeline code that is NO LONGER USED.

## Why Archived

These files were superseded by the canonical training pipeline but kept for
reference in case we need to understand historical decisions.

## DO NOT USE

- `run_v1_full.py` - Used a DIFFERENT model class (StageBridgeV1Full) than production
  training (StageBridgeV1Complete). Ablations run on this would not be comparable
  to the full model results. Replaced by `run_v1_ablations.py` which uses the
  same model as `run_v1_ddp.py`.

## Current Pipeline

Use these instead:
- `run_v1_ddp.py` - Production training (StageBridgeV1Complete)
- `run_v1_ablations.py` - Ablation studies (same model as run_v1_ddp.py)
- `run_ablations.py` - Orchestrates ablation experiments
