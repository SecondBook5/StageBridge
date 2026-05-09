#!/usr/bin/env python3
"""Map spatial and/or snRNA to HLCA/LuCA reference spaces.

Two modes:
  --mode surgery  : scArches fine-tuning (single modality only)
  --mode direct   : Frozen model inference (required for multi-modality alignment)

For aligning spatial + snRNA in the same space, use --mode direct.

Usage:
    # Direct mode - align spatial + snRNA together
    python scripts/map_spatial_to_reference.py \
        --mode direct \
        --spatial /path/to/spatial.h5ad \
        --snrna /path/to/snrna.h5ad \
        --hlca-model /path/to/hlca/model \
        --luca-model /path/to/luca/model \
        --output-dir /path/to/output

    # Surgery mode - single modality (legacy)
    python scripts/map_spatial_to_reference.py \
        --mode surgery \
        --spatial /path/to/spatial.h5ad \
        --hlca-model /path/to/hlca/model \
        --luca-model /path/to/luca/model \
        --output-dir /path/to/output
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Map data to reference spaces")
    parser.add_argument("--mode", choices=["direct", "surgery"], default="direct",
                        help="Mapping mode: 'direct' for frozen model (multi-modality), 'surgery' for scArches fine-tuning")
    parser.add_argument("--spatial", type=Path, required=True,
                        help="Path to spatial h5ad")
    parser.add_argument("--snrna", type=Path, default=None,
                        help="Path to snRNA h5ad (required for direct mode)")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output directory for embeddings")
    parser.add_argument("--hlca-model", type=Path, required=True,
                        help="Path to HLCA scANVI model directory")
    parser.add_argument("--luca-model", type=Path, required=True,
                        help="Path to LuCA scANVI model directory")
    parser.add_argument("--luca-ref", type=Path, default=None,
                        help="Path to LuCA reference h5ad (for surgery mode)")
    parser.add_argument("--batch-size", type=int, default=1024,
                        help="Inference batch size")
    parser.add_argument("--surgery-epochs", type=int, default=200,
                        help="Max epochs for scArches surgery (surgery mode only)")
    args = parser.parse_args()

    if args.mode == "direct":
        if args.snrna is None:
            parser.error("--snrna required for direct mode")

        from stagebridge.reference.scarches_mapper import map_to_reference_direct

        map_to_reference_direct(
            spatial_path=args.spatial,
            snrna_path=args.snrna,
            luca_model_dir=args.luca_model,
            hlca_model_dir=args.hlca_model,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
        )

    else:  # surgery mode
        from stagebridge.reference.scarches_mapper import map_spatial_to_reference

        map_spatial_to_reference(
            spatial_path=args.spatial,
            output_dir=args.output_dir,
            hlca_model_dir=args.hlca_model,
            luca_model_dir=args.luca_model,
            luca_ref_path=args.luca_ref,
            batch_size=args.batch_size,
            surgery_epochs=args.surgery_epochs,
        )


if __name__ == "__main__":
    main()
