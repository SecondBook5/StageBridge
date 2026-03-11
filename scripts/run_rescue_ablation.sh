#!/usr/bin/env bash
# Rescue ablation: grouped ordinal labels × atlas ablation grid.
# 3 model families × 5 atlas conditions × 3 folds × 3 seeds = 135 runs.
#
# Optional second pass adds negative controls:
#   atlas_label_shuffle, within_lesion_niche_shuffle
#
# Usage:
#   bash scripts/run_rescue_ablation.sh [RUN_NAME]
#   bash scripts/run_rescue_ablation.sh [RUN_NAME] --with-controls
#
# Requires STAGEBRIDGE_DATA_ROOT to be set.

set -euo pipefail

export STAGEBRIDGE_DATA_ROOT="${STAGEBRIDGE_DATA_ROOT:-/mnt/e/StageBridge_data}"

RUN_NAME="${1:-rescue_ablation_$(date +%Y%m%d_%H%M%S)}"
WITH_CONTROLS=false
if [[ "${2:-}" == "--with-controls" ]]; then
  WITH_CONTROLS=true
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="$ROOT/outputs/scratch/$RUN_NAME"
LOG_PATH="$OUTPUT_ROOT/workflow.log"

cd "$ROOT"
mkdir -p "$OUTPUT_ROOT"

REF_MODES='[no_atlas,hlca_only,luca_only,hlca_luca,hlca_luca_contrast]'
if [ "$WITH_CONTROLS" = true ]; then
  REF_MODES='[no_atlas,hlca_only,luca_only,hlca_luca,hlca_luca_contrast,atlas_label_shuffle,within_lesion_niche_shuffle]'
fi

{
  echo "[$(date --iso-8601=seconds)] ROOT=$ROOT RUN_NAME=$RUN_NAME WITH_CONTROLS=$WITH_CONTROLS"
  echo "[$(date --iso-8601=seconds)] START rescue ablation"

  python -u -m stagebridge.pipelines step train_lesion \
    -o context_model=eamist \
    -o run_name="$RUN_NAME" \
    -o context_model.eamist.device=cuda \
    -o context_model.eamist.require_cuda=true \
    -o context_model.eamist.batch_size_bags=2 \
    -o 'context_model.eamist.seeds=[42,123,456]' \
    -o context_model.eamist.use_grouped_labels=true \
    -o "context_model.eamist.reference_feature_modes=$REF_MODES" \
    -o 'context_model.eamist.model_families=[pooled,deep_sets,eamist]' \
    -o context_model.eamist.use_atlas_contrast_token=false \
    -o context_model.eamist.pretrained_local_checkpoint=null

  echo "[$(date --iso-8601=seconds)] TRAIN complete"

  python -u -m stagebridge.pipelines step eamist_report \
    -o context_model=eamist \
    -o run_name="$RUN_NAME"

  echo "[$(date --iso-8601=seconds)] REPORT complete"
} >>"$LOG_PATH" 2>&1
