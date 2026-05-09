#!/usr/bin/env python
"""Map snRNA data through HLCA/LuCA using scArches surgery.

Uses the SAME approach as spatial mapping so both modalities end up in the same space.

Usage:
    python scripts/map_snrna_to_reference.py \
        --snrna $DATA/processed/luad_evo/snrna_qc_normalized.h5ad \
        --output-dir $DATA/processed/luad_evo/reference_geometry \
        --hlca-model $DATA/references/hlca/... \
        --luca-model $DATA/references/luca/retrained_model_v3/scanvi_model_hlca_format
"""

from __future__ import annotations

import argparse
from pathlib import Path

from stagebridge.reference.scarches_mapper import map_spatial_to_reference


def main():
    parser = argparse.ArgumentParser(description="Map snRNA to reference spaces via scArches")
    parser.add_argument("--snrna", type=Path, required=True,
                        help="Path to snRNA h5ad")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output directory for embeddings")
    parser.add_argument("--hlca-model", type=Path, required=True,
                        help="Path to HLCA scANVI model directory")
    parser.add_argument("--luca-model", type=Path, required=True,
                        help="Path to LuCA scANVI model directory")
    parser.add_argument("--hlca-ref", type=Path, default=None,
                        help="Path to HLCA reference h5ad (optional)")
    parser.add_argument("--luca-ref", type=Path, default=None,
                        help="Path to LuCA reference h5ad (optional)")
    parser.add_argument("--batch-size", type=int, default=1024,
                        help="Inference batch size")
    parser.add_argument("--surgery-epochs", type=int, default=200,
                        help="Max epochs for scArches surgery")
    args = parser.parse_args()

    # Use a temp subdir to avoid overwriting spatial embeddings
    import tempfile
    import shutil

    temp_dir = Path(args.output_dir) / "_snrna_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    result = map_spatial_to_reference(
        spatial_path=args.snrna,  # Works with any h5ad
        output_dir=temp_dir,
        hlca_model_dir=args.hlca_model,
        luca_model_dir=args.luca_model,
        hlca_ref_path=args.hlca_ref,
        luca_ref_path=args.luca_ref,
        batch_size=args.batch_size,
        surgery_epochs=args.surgery_epochs,
    )

    # Move and rename outputs from spatial_* to snrna_*
    output_dir = Path(args.output_dir)
    renames = [
        ("spatial_hlca_embedding.parquet", "snrna_hlca_embedding.parquet"),
        ("spatial_luca_embedding.parquet", "snrna_luca_embedding.parquet"),
        ("spatial_fused_embedding.parquet", "snrna_fused_embedding.parquet"),
        ("hlca_training_history.json", "snrna_hlca_training_history.json"),
        ("luca_training_history.json", "snrna_luca_training_history.json"),
    ]
    for old, new in renames:
        old_path = temp_dir / old
        new_path = output_dir / new
        if old_path.exists():
            shutil.move(str(old_path), str(new_path))
            print(f"Moved {old} -> {new}")

    # Clean up temp dir
    shutil.rmtree(temp_dir, ignore_errors=True)
    print("Done!")


if __name__ == "__main__":
    main()
