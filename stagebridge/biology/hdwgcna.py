"""hdWGCNA (high-dimensional Weighted Gene Co-expression Network Analysis).

Identifies gene co-expression modules from single-cell data using the
metacell approach for noise reduction.

Reference: Morabito et al. "hdWGCNA identifies co-expression networks
in high-dimensional transcriptomics data" Cell Reports Methods (2023)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import anndata as ad


def setup_metacells(
    adata: "ad.AnnData",
    group_col: str = "cell_type",
    target_metacells: int = 50,
    min_cells_per_metacell: int = 25,
) -> "ad.AnnData":
    """Create metacells by aggregating similar cells.

    Metacells reduce noise and computational burden while preserving
    biological signal for WGCNA.

    Args:
        adata: AnnData with expression data
        group_col: Column to group cells by (e.g., cell_type, leiden)
        target_metacells: Target number of metacells per group
        min_cells_per_metacell: Minimum cells to form a metacell

    Returns:
        AnnData with metacell expression profiles
    """
    import scanpy as sc
    from sklearn.cluster import KMeans

    metacell_data = []
    metacell_meta = []

    for group in adata.obs[group_col].unique():
        group_mask = adata.obs[group_col] == group
        group_adata = adata[group_mask]
        n_cells = group_adata.n_obs

        if n_cells < min_cells_per_metacell:
            continue

        # Determine number of metacells for this group
        n_metacells = min(target_metacells, n_cells // min_cells_per_metacell)
        n_metacells = max(1, n_metacells)

        # Use KMeans on PCA to cluster cells into metacells
        if "X_pca" not in group_adata.obsm:
            sc.pp.pca(group_adata, n_comps=min(50, n_cells - 1))

        pca = group_adata.obsm["X_pca"]

        if n_metacells == 1:
            labels = np.zeros(n_cells, dtype=int)
        else:
            kmeans = KMeans(n_clusters=n_metacells, random_state=42, n_init=10)
            labels = kmeans.fit_predict(pca)

        # Aggregate cells into metacells
        for mc_id in range(n_metacells):
            mc_mask = labels == mc_id
            n_mc_cells = mc_mask.sum()

            if n_mc_cells < min_cells_per_metacell // 2:
                continue

            # Sum expression (for count data) or mean (for normalized)
            mc_expr = group_adata[mc_mask].X
            if hasattr(mc_expr, "toarray"):
                mc_expr = mc_expr.toarray()
            mc_expr = mc_expr.mean(axis=0).flatten()

            metacell_data.append(mc_expr)
            metacell_meta.append({
                "metacell_id": f"{group}_{mc_id}",
                group_col: group,
                "n_cells": n_mc_cells,
            })

    # Create metacell AnnData
    import anndata
    metacell_adata = anndata.AnnData(
        X=np.array(metacell_data),
        obs=pd.DataFrame(metacell_meta),
        var=adata.var.copy(),
    )
    metacell_adata.obs.index = metacell_adata.obs["metacell_id"]

    print(f"Created {metacell_adata.n_obs} metacells from {adata.n_obs} cells")

    return metacell_adata


def run_wgcna(
    adata: "ad.AnnData",
    soft_power: int | None = None,
    min_module_size: int = 30,
    deep_split: int = 2,
    merge_cut_height: float = 0.25,
    n_top_genes: int = 5000,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run WGCNA to identify co-expression modules.

    Args:
        adata: AnnData (metacells recommended)
        soft_power: Soft-thresholding power (None = auto-detect)
        min_module_size: Minimum genes per module
        deep_split: Controls sensitivity of module detection (0-4, higher = more modules)
        merge_cut_height: Height for merging similar modules (0-1)
        n_top_genes: Number of highly variable genes to use

    Returns:
        Tuple of:
        - module_genes: DataFrame mapping genes to modules
        - module_eigengenes: DataFrame of module eigengenes per sample
        - hub_genes: Dict mapping module -> top hub genes
    """
    import scanpy as sc
    from scipy.cluster.hierarchy import linkage, fcluster, cut_tree
    from scipy.spatial.distance import pdist
    from sklearn.decomposition import PCA

    # Select highly variable genes
    if adata.n_vars > n_top_genes:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor="seurat_v3")
        expr = adata[:, adata.var["highly_variable"]].X
        gene_names = adata.var_names[adata.var["highly_variable"]].tolist()
    else:
        expr = adata.X
        gene_names = adata.var_names.tolist()

    if hasattr(expr, "toarray"):
        expr = expr.toarray()

    print(f"Running WGCNA on {expr.shape[1]} genes x {expr.shape[0]} samples")

    # Calculate correlation matrix
    print("Calculating correlation matrix...")
    cor_matrix = np.corrcoef(expr.T)
    cor_matrix = np.nan_to_num(cor_matrix, nan=0)

    # Auto-detect soft power if not provided
    if soft_power is None:
        soft_power = _pick_soft_threshold(cor_matrix)
        print(f"Auto-selected soft power: {soft_power}")

    # Create adjacency matrix
    print("Creating adjacency matrix...")
    adj_matrix = np.abs(cor_matrix) ** soft_power

    # Calculate topological overlap matrix (TOM)
    print("Calculating TOM...")
    tom = _calculate_tom(adj_matrix)

    # Cluster genes using TOM dissimilarity
    print("Clustering genes...")
    tom_dist = 1 - tom
    np.fill_diagonal(tom_dist, 0)

    # Convert to condensed distance matrix
    tom_dist_condensed = pdist(tom_dist)

    # Hierarchical clustering
    Z = linkage(tom_dist_condensed, method="average")

    # Dynamic tree cutting - use maxclust to get reasonable number of modules
    # Estimate number of clusters based on deep_split parameter
    # deep_split 0-4 maps to roughly 5-50 target modules
    n_genes = len(gene_names)
    target_modules = max(5, min(50, n_genes // (200 - deep_split * 30)))
    print(f"  Target modules: ~{target_modules} (deep_split={deep_split})")

    # Cut tree using maxclust criterion for initial clustering
    clusters = fcluster(Z, t=target_modules, criterion="maxclust")

    # Filter small modules
    module_counts = pd.Series(clusters).value_counts()
    valid_modules = module_counts[module_counts >= min_module_size].index
    print(f"  Initial clusters: {len(module_counts)}, valid (>={min_module_size} genes): {len(valid_modules)}")

    # Merge similar modules based on eigengene correlation
    # First assign initial module labels
    initial_labels = {c: f"M{i}" for i, c in enumerate(sorted(valid_modules))}

    # Assign module colors/names
    module_genes = pd.DataFrame({
        "gene": gene_names,
        "module": [initial_labels.get(c, "grey") for c in clusters],
    })

    n_modules = len(valid_modules)
    print(f"Found {n_modules} modules (+ grey/unassigned)")

    # Calculate module eigengenes
    print("Calculating module eigengenes...")
    module_eigengenes = {}
    hub_genes = {}

    for module in module_genes["module"].unique():
        if module == "grey":
            continue

        mod_genes = module_genes[module_genes["module"] == module]["gene"].tolist()
        mod_idx = [gene_names.index(g) for g in mod_genes]
        mod_expr = expr[:, mod_idx]

        # Module eigengene = first PC
        pca = PCA(n_components=1)
        me = pca.fit_transform(mod_expr).flatten()
        module_eigengenes[module] = me

        # Hub genes = highest connectivity within module
        mod_adj = adj_matrix[np.ix_(mod_idx, mod_idx)]
        connectivity = mod_adj.sum(axis=1) - 1  # exclude self
        top_hub_idx = np.argsort(connectivity)[-10:][::-1]
        hub_genes[module] = [mod_genes[i] for i in top_hub_idx]

    me_df = pd.DataFrame(module_eigengenes, index=adata.obs.index)

    return module_genes, me_df, hub_genes


def _pick_soft_threshold(cor_matrix: np.ndarray, powers: list[int] | None = None) -> int:
    """Auto-select soft-thresholding power based on scale-free topology."""
    if powers is None:
        powers = list(range(1, 21))

    best_power = 6  # default
    best_r2 = 0

    for power in powers:
        adj = np.abs(cor_matrix) ** power
        k = adj.sum(axis=1) - 1

        # Fit scale-free topology
        log_k = np.log10(k + 1)
        log_pk = np.log10(np.histogram(k, bins=20)[0] + 1)

        # R^2 of fit
        if len(log_pk) > 2:
            r2 = np.corrcoef(log_k[:len(log_pk)], log_pk)[0, 1] ** 2
            if r2 > best_r2 and r2 > 0.8:
                best_r2 = r2
                best_power = power

    return best_power


def _calculate_tom(adj_matrix: np.ndarray) -> np.ndarray:
    """Calculate Topological Overlap Matrix."""
    n = adj_matrix.shape[0]

    # k = connectivity
    k = adj_matrix.sum(axis=1) - np.diag(adj_matrix)

    # TOM calculation
    tom = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            # l_ij = sum of shared neighbors
            l_ij = np.sum(adj_matrix[i, :] * adj_matrix[:, j])

            # TOM formula
            denom = min(k[i], k[j]) + 1 - adj_matrix[i, j]
            if denom > 0:
                tom[i, j] = (l_ij + adj_matrix[i, j]) / denom
                tom[j, i] = tom[i, j]

    np.fill_diagonal(tom, 1)

    return tom


def correlate_modules_with_traits(
    module_eigengenes: pd.DataFrame,
    traits: pd.DataFrame,
) -> pd.DataFrame:
    """Correlate module eigengenes with sample traits.

    Args:
        module_eigengenes: Module eigengenes per sample
        traits: Trait values per sample (numeric)

    Returns:
        DataFrame with correlations and p-values
    """
    from scipy.stats import pearsonr

    results = []

    for module in module_eigengenes.columns:
        me = module_eigengenes[module].values

        for trait in traits.columns:
            trait_vals = traits[trait].values

            # Remove NaN
            valid = ~(np.isnan(me) | np.isnan(trait_vals))
            if valid.sum() < 5:
                continue

            r, p = pearsonr(me[valid], trait_vals[valid])

            results.append({
                "module": module,
                "trait": trait,
                "correlation": r,
                "pvalue": p,
            })

    return pd.DataFrame(results)


def run_hdwgcna_pipeline(
    h5ad_path: str | Path,
    output_dir: str | Path,
    group_col: str = "cell_type",
    stage_col: str = "stage",
    n_top_genes: int = 5000,
) -> dict[str, Path]:
    """Run complete hdWGCNA pipeline.

    Args:
        h5ad_path: Path to h5ad file
        output_dir: Output directory
        group_col: Column for metacell grouping
        stage_col: Column for trait correlation
        n_top_genes: Number of HVGs to use

    Returns:
        Dict of output paths
    """
    import scanpy as sc

    h5ad_path = Path(h5ad_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {h5ad_path}...")
    adata = sc.read_h5ad(h5ad_path)
    print(f"  {adata.n_obs} cells x {adata.n_vars} genes")

    # Create metacells
    print("\nCreating metacells...")
    metacells = setup_metacells(adata, group_col=group_col)

    # Run WGCNA
    print("\nRunning WGCNA...")
    module_genes, module_eigengenes, hub_genes = run_wgcna(
        metacells, n_top_genes=n_top_genes
    )

    # Save module genes
    module_genes.to_parquet(output_dir / "wgcna_module_genes.parquet")
    print(f"Saved module genes: {len(module_genes)} genes")

    # Save module eigengenes
    module_eigengenes.to_parquet(output_dir / "wgcna_module_eigengenes.parquet")
    print(f"Saved module eigengenes: {module_eigengenes.shape}")

    # Save hub genes
    hub_df = pd.DataFrame([
        {"module": m, "rank": i, "gene": g}
        for m, genes in hub_genes.items()
        for i, g in enumerate(genes)
    ])
    hub_df.to_parquet(output_dir / "wgcna_hub_genes.parquet")
    print(f"Saved hub genes: {len(hub_df)} entries")

    # Correlate with stage if available
    if stage_col in metacells.obs.columns:
        print("\nCorrelating modules with stage...")

        # Encode stage as numeric
        stage_order = {"Normal": 0, "AAH": 1, "AIS": 2, "MIA": 3, "LUAD": 4}
        stage_numeric = metacells.obs[stage_col].map(stage_order)

        if stage_numeric.notna().sum() > 10:
            traits = pd.DataFrame({
                "stage_numeric": stage_numeric.values
            }, index=metacells.obs.index)

            correlations = correlate_modules_with_traits(module_eigengenes, traits)
            correlations.to_parquet(output_dir / "wgcna_stage_correlations.parquet")
            print(f"Saved stage correlations: {len(correlations)} module-trait pairs")

    print("\nhdWGCNA complete!")

    return {
        "module_genes": output_dir / "wgcna_module_genes.parquet",
        "module_eigengenes": output_dir / "wgcna_module_eigengenes.parquet",
        "hub_genes": output_dir / "wgcna_hub_genes.parquet",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run hdWGCNA analysis")
    parser.add_argument("--h5ad", required=True, help="Path to h5ad file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--group-col", default="cell_type", help="Metacell grouping column")
    parser.add_argument("--stage-col", default="stage", help="Stage column for correlation")
    parser.add_argument("--n-genes", type=int, default=5000, help="Number of HVGs")
    args = parser.parse_args()

    run_hdwgcna_pipeline(
        args.h5ad,
        args.output,
        group_col=args.group_col,
        stage_col=args.stage_col,
        n_top_genes=args.n_genes,
    )
