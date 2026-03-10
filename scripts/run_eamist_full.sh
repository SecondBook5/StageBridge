#!/usr/bin/env bash
# End-to-end EA-MIST pipeline: pretrain local encoder, train, report.
#
# Usage:
#   bash scripts/run_eamist_full.sh [RUN_NAME]
#
# Requires STAGEBRIDGE_DATA_ROOT to be set.

set -euo pipefail

RUN_NAME="${1:-eamist_$(date +%Y%m%d_%H%M%S)}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="$ROOT/outputs/scratch/$RUN_NAME"
LOG_PATH="$OUTPUT_ROOT/workflow.log"

cd "$ROOT"
mkdir -p "$OUTPUT_ROOT"

{
  echo "[$(date --iso-8601=seconds)] ROOT=$ROOT PWD=$(pwd) PYTHON=$(command -v python)"
  echo "[$(date --iso-8601=seconds)] START pretrain_local"
  python -u -m stagebridge.pipelines step pretrain_local \
    -o context_model=eamist \
    -o run_name="$RUN_NAME" \
    -o context_model.eamist.device=cuda \
    -o context_model.eamist.require_cuda=true

  CHECKPOINT="$OUTPUT_ROOT/eamist_pretrain/best_local_encoder.pt"
  echo "[$(date --iso-8601=seconds)] PRETRAIN complete checkpoint=$CHECKPOINT"

  python -u -m stagebridge.pipelines step train_lesion \
    -o context_model=eamist \
    -o run_name="$RUN_NAME" \
    -o context_model.eamist.device=cuda \
    -o context_model.eamist.require_cuda=true \
    -o context_model.eamist.batch_size_bags=2 \
    -o context_model.eamist.pretrained_local_checkpoint="$CHECKPOINT"

  echo "[$(date --iso-8601=seconds)] TRAIN complete"

  python -u -m stagebridge.pipelines step eamist_report \
    -o context_model=eamist \
    -o run_name="$RUN_NAME"

  echo "[$(date --iso-8601=seconds)] REPORT complete"
} >>"$LOG_PATH" 2>&1
