#!/bin/bash
#SBATCH --job-name=cell_type_transfer
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=2:00:00
#SBATCH --output=/home/booka/StageBridge/logs/cell_type_transfer_%j.out
#SBATCH --error=/home/booka/StageBridge/logs/cell_type_transfer_%j.err

set -euo pipefail
cd /home/booka/StageBridge
source ~/.bashrc
conda activate /scratch/chaunzt1/stagebridge_env

DATA="/scratch/chaunzt1/stagebridge"
mkdir -p /home/booka/StageBridge/logs

echo "=== Cell Type Label Transfer ==="
echo "Start: $(date)"

python << 'PYTHON_SCRIPT'
import numpy as np
import pandas as pd
import anndata as ad
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

DATA = "/scratch/chaunzt1/stagebridge"

print("=== Cell Type Label Transfer ===")
print()

# Load HLCA reference
print("Loading HLCA reference...")
hlca = ad.read_h5ad(f"{DATA}/references/hlca/hlca_reference.h5ad")
ref_latent = hlca.obsm['X_scANVI']
ref_labels_coarse = hlca.obs['ann_level_1'].values
ref_labels_fine = hlca.obs['cell_type'].values
print(f"  Reference: {ref_latent.shape[0]} cells, {ref_latent.shape[1]} dims")
print(f"  Coarse types: {np.unique(ref_labels_coarse)}")

# Load query embeddings
print("Loading query embeddings...")
query_emb = pd.read_parquet(f"{DATA}/processed/luad_evo/reference_geometry/hlca_embedding.parquet")
latent_cols = [c for c in query_emb.columns if c.startswith('hlca_latent_')]
query_latent = query_emb[latent_cols].values
print(f"  Query: {query_latent.shape[0]} cells, {query_latent.shape[1]} dims")

# k-NN label transfer
print()
print("Building k-NN index (k=15)...")
knn = NearestNeighbors(n_neighbors=15, metric='euclidean', n_jobs=-1)
knn.fit(ref_latent)

print("Finding nearest neighbors...")
distances, indices = knn.kneighbors(query_latent)

print("Transferring labels (majority vote)...")
transferred_coarse = []
transferred_fine = []
for i in range(len(query_latent)):
    # Coarse
    neighbor_labels = ref_labels_coarse[indices[i]]
    unique, counts = np.unique(neighbor_labels, return_counts=True)
    transferred_coarse.append(unique[counts.argmax()])
    # Fine
    neighbor_labels = ref_labels_fine[indices[i]]
    unique, counts = np.unique(neighbor_labels, return_counts=True)
    transferred_fine.append(unique[counts.argmax()])

print()
print("=== Results ===")
print("Coarse cell types:")
print(pd.Series(transferred_coarse).value_counts())
print()
print("Fine cell types (top 15):")
print(pd.Series(transferred_fine).value_counts().head(15))

# Save updated embeddings
query_emb['cell_type'] = transferred_fine
query_emb['cell_type_coarse'] = transferred_coarse
query_emb['knn_distance_mean'] = distances.mean(axis=1)
output_path = f"{DATA}/processed/luad_evo/reference_geometry/hlca_embedding_with_celltypes.parquet"
query_emb.to_parquet(output_path)
print(f"\nSaved: {output_path}")

# Create cell type lookup
cell_type_map = dict(zip(query_emb['cell_id'], transferred_fine))
cell_type_coarse_map = dict(zip(query_emb['cell_id'], transferred_coarse))

# Update snRNA h5ad files
print("\nUpdating snRNA h5ad files...")
for fname in ['snrna_merged.h5ad', 'snrna_qc_normalized.h5ad']:
    fpath = f"{DATA}/processed/luad_evo/{fname}"
    print(f"  Loading {fname}...")
    adata = ad.read_h5ad(fpath)

    # Map cell types using index
    adata.obs['cell_type'] = adata.obs.index.map(cell_type_map)
    adata.obs['cell_type_coarse'] = adata.obs.index.map(cell_type_coarse_map)

    # Check for unmapped cells
    unmapped = adata.obs['cell_type'].isna().sum()
    if unmapped > 0:
        print(f"    WARNING: {unmapped} cells without cell type mapping")

    print(f"  Saving {fname}...")
    adata.write_h5ad(fpath)
    print(f"    Done: {adata.shape[0]} cells with cell_type column")

print("\n=== Label transfer complete! ===")
PYTHON_SCRIPT

echo "End: $(date)"
