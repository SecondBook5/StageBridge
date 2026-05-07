#!/usr/bin/env python3
"""Map spatial spots directly to HLCA/LuCA reference spaces via scArches.

This produces embeddings in the SAME space as snRNA cells, solving the
modality separation problem. No deconvolution - direct expression mapping.

Usage:
    python scripts/map_spatial_to_reference.py \
        --spatial /path/to/spatial_merged.h5ad \
        --output-dir /path/to/output \
        --hlca-model /path/to/hlca/model \
        --luca-model /path/to/luca/model
"""

import argparse
from pathlib import Path

from stagebridge.reference.scarches_mapper import map_spatial_to_reference


def main():
    parser = argparse.ArgumentParser(description="Map spatial to reference spaces")
    parser.add_argument("--spatial", type=Path, required=True,
                        help="Path to spatial_merged.h5ad")
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

    map_spatial_to_reference(
        spatial_path=args.spatial,
        output_dir=args.output_dir,
        hlca_model_dir=args.hlca_model,
        luca_model_dir=args.luca_model,
        hlca_ref_path=args.hlca_ref,
        luca_ref_path=args.luca_ref,
        batch_size=args.batch_size,
        surgery_epochs=args.surgery_epochs,
    )


if __name__ == "__main__":
    main()
