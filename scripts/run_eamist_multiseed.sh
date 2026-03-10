#!/usr/bin/env bash
# Multi-seed EA-MIST benchmark: 3 folds × 3 seeds = 9 runs per model.
# Reuses a pretrained local encoder checkpoint.
#
# Usage:
#   bash scripts/run_eamist_multiseed.sh [RUN_NAME] [PRETRAIN_CHECKPOINT]
#
# Requires STAGEBRIDGE_DATA_ROOT to be set.

set -euo pipefail

RUN_NAME="${1:-eamist_multiseed_$(date +%Y%m%d_%H%M%S)}"
CHECKPOINT="${2:-outputs/scratch/eamist_v1_20260309/eamist_pretrain/best_local_encoder.pt}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="$ROOT/outputs/scratch/$RUN_NAME"
LOG_PATH="$OUTPUT_ROOT/workflow.log"

cd "$ROOT"
mkdir -p "$OUTPUT_ROOT"

{
  echo "[$(date --iso-8601=seconds)] ROOT=$ROOT CHECKPOINT=$CHECKPOINT"
  echo "[$(date --iso-8601=seconds)] START train_lesion (3 seeds)"

  python -u -m stagebridge.pipelines step train_lesion \
    -o context_model=eamist \
    -o run_name="$RUN_NAME" \
    -o context_model.eamist.device=cuda \
    -o context_model.eamist.require_cuda=true \
    -o context_model.eamist.batch_size_bags=2 \
    -o 'context_model.eamist.seeds=[42,123,456]' \
    -o context_model.eamist.pretrained_local_checkpoint="$CHECKPOINT"

  echo "[$(date --iso-8601=seconds)] TRAIN complete"

  python -u -m stagebridge.pipelines step eamist_report \
    -o context_model=eamist \
    -o run_name="$RUN_NAME"

  echo "[$(date --iso-8601=seconds)] REPORT complete"
} >>"$LOG_PATH" 2>&1
