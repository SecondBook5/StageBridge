#!/usr/bin/env python3
"""Rebuild neighborhoods.parquet with flattened neighbor arrays.

Fixes pyarrow overflow issue by storing neighbor_cells_flat as [K*40]
instead of nested [[40], [40], ...].
"""

import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", required=True, help="Path to cells.parquet")
    parser.add_argument("--output", required=True, help="Output path for neighborhoods_v2.parquet")
    parser.add_argument("--max-neighbors", type=int, default=100)
    parser.add_argument("--latent-dim", type=int, default=40)
    args = parser.parse_args()

    print("Loading cells...")
    cells = pd.read_parquet(args.cells)
    spatial = cells[cells["x"].notna()].copy()
    print(f"Spatial cells: {len(spatial)}")

    max_neighbors = args.max_neighbors
    latent_dim = args.latent_dim

    records = []
    donors = spatial["donor_id"].unique()
    print(f"Processing {len(donors)} donors...")

    for donor_id in tqdm(donors):
        donor_cells = spatial[spatial["donor_id"] == donor_id].reset_index(drop=True)
        if len(donor_cells) < max_neighbors + 1:
            print(f"  Skipping {donor_id}: only {len(donor_cells)} cells")
            continue

        coords = donor_cells[["x", "y"]].values
        tree = cKDTree(coords)

        for idx in range(len(donor_cells)):
            row = donor_cells.iloc[idx]
            dists, indices = tree.query(coords[idx], k=max_neighbors + 1)
            indices = indices[1:]  # exclude self
            dists = dists[1:]

            # Flatten neighbor embeddings
            neighbor_flat = []
            for ni in indices:
                neighbor_flat.extend(donor_cells.iloc[ni]["z_fused"])

            # Pad if needed
            if len(neighbor_flat) < max_neighbors * latent_dim:
                neighbor_flat.extend([0.0] * (max_neighbors * latent_dim - len(neighbor_flat)))

            # Safe float conversion
            def safe_float(val):
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    return 0.0
                return float(val)

            records.append({
                "cell_id": row["cell_id"],
                "donor_id": donor_id,
                "stage": row["stage"],
                "receiver_z": row["z_fused"],
                "hlca_z": row["z_hlca"],
                "luca_z": row["z_luca"],
                "neighbor_cells_flat": neighbor_flat,
                "neighbor_distances": dists.tolist(),
                "n_neighbors": len(indices),
                "proliferation_label": safe_float(row.get("proliferation_label", 0.0)),
                "pathway_targets": [
                    safe_float(row.get(f"pathway_{p}", 0.0))
                    for p in ["Androgen", "EGFR", "Estrogen", "Hypoxia", "JAK-STAT",
                              "MAPK", "NFkB", "PI3K", "TGFb", "TNFa", "Trail",
                              "VEGF", "WNT", "p53"]
                ],
                "stats_z": [
                    safe_float(row.get("caf_fraction", 0.0)),
                    safe_float(row.get("immune_fraction", 0.0)),
                    safe_float(row.get("diversity", 0.0)),
                    safe_float(row.get("S_score", 0.0)),
                    safe_float(row.get("G2M_score", 0.0)),
                ],
                "evolution_features": [
                    safe_float(row.get(c, 0.0))
                    for c in ["tmb", "kras_mut", "egfr_mut", "tp53_mut", "stk11_mut",
                              "keap1_mut", "smad4_mut", "braf_mut", "cnv_score",
                              "clone_size", "clone_rank", "is_major_clone", "clone_fraction",
                              "n_clones", "clonal_entropy", "clonal_diversity",
                              "clonal_pattern_idx", "aneuploidy_score", "clone_sharing_ratio",
                              "has_invasive_only_clones", "egfr_L858R", "egfr_exon19del",
                              "egfr_T790M", "kras_G12C", "kras_G12V", "has_level1_mutation",
                              "has_actionable_mutation", "cnv_score_z"]
                ],
            })

    print(f"Saving {len(records)} neighborhoods...")
    df = pd.DataFrame(records)
    df.to_parquet(args.output, index=False)
    print(f"Done! Saved to {args.output}")

    # Verify
    print("\nVerifying...")
    test = pd.read_parquet(args.output).head(1)
    nc = test["neighbor_cells_flat"].iloc[0]
    print(f"neighbor_cells_flat: type={type(nc)}, len={len(nc)}")
    print(f"First 5 values: {nc[:5]}")


if __name__ == "__main__":
    main()
