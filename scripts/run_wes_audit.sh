#!/bin/bash
# Run WES panel audit for StageBridge v1
# Usage: bash scripts/run_wes_audit.sh

set -e

export DATA=/data1/chaunzt1/stagebridge
CELLS="$DATA/processed/luad_evo/canonical/cells.parquet"
OUTDIR="$DATA/processed/luad_evo/canonical/wes_audit"

echo "========================================"
echo "Step 1: Inspect cells.parquet structure"
echo "========================================"

python - <<'PY'
import pandas as pd
import os

path = os.environ["DATA"] + "/processed/luad_evo/canonical/cells.parquet"
df = pd.read_parquet(path)

print(f"Shape: {df.shape}")
print("\nCandidate metadata columns:")
for col in ["cell_id", "donor_id", "patient_id", "sample_id", "lesion_id", "stage", "data_type"]:
    print(f"  {col}: {'YES' if col in df.columns else 'NO'}")

print("\nMutation-like columns:")
for col in sorted([c for c in df.columns if c == "tmb" or c.endswith("_mut")]):
    print(f"  {col}")
PY

echo ""
echo "========================================"
echo "Step 2: Run WES panel audit"
echo "========================================"

python scripts/audit_wes_panel.py \
  --cells "$CELLS" \
  --outdir "$OUTDIR"

echo ""
echo "========================================"
echo "Step 3: Display results"
echo "========================================"

echo ""
echo "=== wes_feature_summary.csv ==="
cat "$OUTDIR/wes_feature_summary.csv"

echo ""
echo "=== wes_stage_summary.csv ==="
cat "$OUTDIR/wes_stage_summary.csv"

echo ""
echo "========================================"
echo "Audit complete. Review above outputs."
echo "========================================"
