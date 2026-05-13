#!/usr/bin/env python3
"""Rebuild neighborhoods saving large arrays as numpy, not parquet."""

import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-neighbors", type=int, default=100)
    parser.add_argument("--latent-dim", type=int, default=40)
    args = parser.parse_args()

    print("Loading cells...")
    cells = pd.read_parquet(args.cells)
    spatial = cells[cells["x"].notna()].copy()
    n_cells = len(spatial)
    print(f"Spatial cells: {n_cells}")

    max_neighbors = args.max_neighbors
    latent_dim = args.latent_dim

    # Pre-allocate numpy arrays
    neighbor_embeddings = np.zeros((n_cells, max_neighbors, latent_dim), dtype=np.float32)
    neighbor_distances = np.zeros((n_cells, max_neighbors), dtype=np.float32)
    n_neighbors_arr = np.zeros(n_cells, dtype=np.int32)

    meta_records = []

    donors = spatial["donor_id"].unique()
    print(f"Processing {len(donors)} donors...")

    cell_idx = 0
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
            indices = indices[1:]
            dists = dists[1:]

            # Fill numpy arrays directly
            for j, ni in enumerate(indices):
                neighbor_embeddings[cell_idx, j, :] = donor_cells.iloc[ni]["z_fused"]
            neighbor_distances[cell_idx, :len(dists)] = dists
            n_neighbors_arr[cell_idx] = len(indices)

            def safe_float(val):
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    return 0.0
                return float(val)

            meta_records.append({
                "cell_id": row["cell_id"],
                "donor_id": donor_id,
                "stage": row["stage"],
                "receiver_z": row["z_fused"],
                "hlca_z": row["z_hlca"],
                "luca_z": row["z_luca"],
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
            cell_idx += 1

    # Trim to actual size
    neighbor_embeddings = neighbor_embeddings[:cell_idx]
    neighbor_distances = neighbor_distances[:cell_idx]
    n_neighbors_arr = n_neighbors_arr[:cell_idx]

    print(f"\nSaving {cell_idx} neighborhoods...")
    print(f"  neighbor_embeddings: {neighbor_embeddings.shape}")
    print(f"  neighbor_distances: {neighbor_distances.shape}")

    np.save(f"{args.output_dir}/neighbor_embeddings.npy", neighbor_embeddings)
    np.save(f"{args.output_dir}/neighbor_distances.npy", neighbor_distances)

    meta_df = pd.DataFrame(meta_records)
    meta_df.to_parquet(f"{args.output_dir}/neighborhoods_meta.parquet", index=False)
    print(f"  neighborhoods_meta: {len(meta_df)} rows")

    print("\nVerifying...")
    emb = np.load(f"{args.output_dir}/neighbor_embeddings.npy")
    print(f"  Loaded embeddings: {emb.shape}")
    print(f"  First neighbor[0,0,:5]: {emb[0, 0, :5]}")
    print(f"  Sum (should be non-zero): {emb.sum()}")

    print("\nDone!")


if __name__ == "__main__":
    main()
