"""Spatial-specific GSEA (Gene Set Enrichment Analysis).

Performs GSEA that incorporates spatial context by:
1. Defining spatial neighborhoods and computing neighborhood-level signatures
2. Running GSEA on spatially-aggregated expression profiles
3. Identifying pathways enriched in specific spatial contexts (tumor-adjacent, immune niches, etc.)

Reference: Subramanian et al. "Gene set enrichment analysis" PNAS (2005)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import anndata as ad


# Common gene set collections
HALLMARK_SETS = [
    "HALLMARK_INFLAMMATORY_RESPONSE",
    "HALLMARK_IL6_JAK_STAT3_SIGNALING",
    "HALLMARK_TNFA_SIGNALING_VIA_NFKB",
    "HALLMARK_INTERFERON_GAMMA_RESPONSE",
    "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION",
    "HALLMARK_HYPOXIA",
    "HALLMARK_ANGIOGENESIS",
    "HALLMARK_APOPTOSIS",
    "HALLMARK_P53_PATHWAY",
    "HALLMARK_MYC_TARGETS_V1",
    "HALLMARK_E2F_TARGETS",
    "HALLMARK_G2M_CHECKPOINT",
    "HALLMARK_OXIDATIVE_PHOSPHORYLATION",
    "HALLMARK_GLYCOLYSIS",
]


def define_spatial_contexts(
    adata: "ad.AnnData",
    cell_type_col: str = "cell_type",
    stage_col: str = "stage",
    context_radius: float = 50.0,
) -> pd.DataFrame:
    """Define spatial contexts based on neighborhood composition.

    Args:
        adata: AnnData with spatial coordinates in obsm['spatial']
        cell_type_col: Column for cell type annotations
        stage_col: Column for stage labels
        context_radius: Radius for defining neighborhoods (microns)

    Returns:
        DataFrame with spatial context labels per cell
    """
    from scipy.spatial import cKDTree

    if "spatial" not in adata.obsm:
        raise ValueError("No spatial coordinates found in obsm['spatial']")

    coords = adata.obsm["spatial"]
    cell_types = adata.obs[cell_type_col].values

    # Build spatial index
    tree = cKDTree(coords)

    # Define context categories based on neighborhood composition
    contexts = []

    for i in range(adata.n_obs):
        # Find neighbors within radius
        neighbor_idx = tree.query_ball_point(coords[i], context_radius)
        neighbor_types = cell_types[neighbor_idx]

        # Compute neighborhood composition
        type_counts = pd.Series(neighbor_types).value_counts(normalize=True)

        # Classify context based on composition
        if _is_tumor_adjacent(type_counts):
            context = "tumor_adjacent"
        elif _is_immune_niche(type_counts):
            context = "immune_niche"
        elif _is_stromal_niche(type_counts):
            context = "stromal_niche"
        elif _is_epithelial_niche(type_counts):
            context = "epithelial_niche"
        else:
            context = "mixed"

        contexts.append({
            "cell_id": adata.obs.index[i],
            "spatial_context": context,
            "n_neighbors": len(neighbor_idx),
        })

    return pd.DataFrame(contexts).set_index("cell_id")


def _is_tumor_adjacent(type_counts: pd.Series) -> bool:
    """Check if neighborhood is tumor-adjacent."""
    tumor_types = ["tumor", "cancer", "malignant", "epithelial_tumor"]
    immune_types = ["macrophage", "t_cell", "b_cell", "nk_cell", "dendritic"]

    tumor_frac = sum(type_counts.get(t, 0) for t in tumor_types if t.lower() in str(type_counts.index).lower())
    immune_frac = sum(type_counts.get(t, 0) for t in immune_types if t.lower() in str(type_counts.index).lower())

    return tumor_frac > 0.2 and immune_frac > 0.1


def _is_immune_niche(type_counts: pd.Series) -> bool:
    """Check if neighborhood is immune-dominated."""
    immune_keywords = ["macrophage", "t_cell", "b_cell", "nk", "dendritic", "monocyte", "immune"]
    immune_frac = sum(
        v for k, v in type_counts.items()
        if any(kw in str(k).lower() for kw in immune_keywords)
    )
    return immune_frac > 0.5


def _is_stromal_niche(type_counts: pd.Series) -> bool:
    """Check if neighborhood is stromal-dominated."""
    stromal_keywords = ["fibroblast", "caf", "stromal", "pericyte", "endothelial"]
    stromal_frac = sum(
        v for k, v in type_counts.items()
        if any(kw in str(k).lower() for kw in stromal_keywords)
    )
    return stromal_frac > 0.4


def _is_epithelial_niche(type_counts: pd.Series) -> bool:
    """Check if neighborhood is epithelial-dominated."""
    epi_keywords = ["epithelial", "alveolar", "at1", "at2", "club", "basal", "ciliated"]
    epi_frac = sum(
        v for k, v in type_counts.items()
        if any(kw in str(k).lower() for kw in epi_keywords)
    )
    return epi_frac > 0.5


def compute_spatial_signatures(
    adata: "ad.AnnData",
    spatial_contexts: pd.DataFrame,
    n_top_genes: int = 2000,
) -> pd.DataFrame:
    """Compute gene signatures for each spatial context.

    Args:
        adata: AnnData with expression data
        spatial_contexts: DataFrame with spatial_context column
        n_top_genes: Number of highly variable genes to use

    Returns:
        DataFrame with mean expression per context
    """
    import scanpy as sc

    # Select HVGs
    if adata.n_vars > n_top_genes:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor="seurat_v3")
        gene_mask = adata.var["highly_variable"]
    else:
        gene_mask = np.ones(adata.n_vars, dtype=bool)

    # Add context labels to adata
    adata.obs["_spatial_context"] = spatial_contexts.loc[adata.obs.index, "spatial_context"]

    # Compute mean expression per context
    signatures = {}
    for context in adata.obs["_spatial_context"].unique():
        mask = adata.obs["_spatial_context"] == context
        expr = adata[mask, gene_mask].X

        if hasattr(expr, "toarray"):
            expr = expr.toarray()

        signatures[context] = expr.mean(axis=0)

    sig_df = pd.DataFrame(
        signatures,
        index=adata.var_names[gene_mask],
    )

    return sig_df


def run_gsea_per_context(
    signatures: pd.DataFrame,
    reference_context: str = "epithelial_niche",
    gene_sets: str = "MSigDB_Hallmark_2020",
    organism: str = "human",
) -> dict[str, pd.DataFrame]:
    """Run GSEA comparing each context to reference.

    Args:
        signatures: Mean expression per context
        reference_context: Context to use as baseline
        gene_sets: Gene set library name
        organism: Organism for gene sets

    Returns:
        Dict mapping context -> GSEA results DataFrame
    """
    import gseapy as gp

    results = {}

    ref_expr = signatures[reference_context]

    for context in signatures.columns:
        if context == reference_context:
            continue

        # Compute log2 fold change vs reference
        test_expr = signatures[context]

        # Add pseudocount to avoid log(0)
        log2fc = np.log2((test_expr + 1) / (ref_expr + 1))

        # Create ranked gene list
        ranked = pd.Series(log2fc.values, index=signatures.index)
        ranked = ranked.sort_values(ascending=False)

        # Run GSEA prerank
        try:
            gsea_res = gp.prerank(
                rnk=ranked,
                gene_sets=gene_sets,
                organism=organism,
                min_size=15,
                max_size=500,
                permutation_num=1000,
                threads=4,
                seed=42,
                verbose=False,
            )
            results[context] = gsea_res.res2d
        except Exception as e:
            print(f"  GSEA failed for {context}: {e}")
            continue

    return results


def run_gsea_by_stage_and_context(
    adata: "ad.AnnData",
    spatial_contexts: pd.DataFrame,
    stage_col: str = "stage",
    gene_sets: str = "MSigDB_Hallmark_2020",
) -> pd.DataFrame:
    """Run GSEA stratified by both stage and spatial context.

    Args:
        adata: AnnData with expression data
        spatial_contexts: DataFrame with spatial_context column
        stage_col: Column for stage labels
        gene_sets: Gene set library name

    Returns:
        DataFrame with GSEA results per stage-context combination
    """
    import gseapy as gp
    import scanpy as sc

    adata.obs["_spatial_context"] = spatial_contexts.loc[adata.obs.index, "spatial_context"]

    all_results = []

    stages = adata.obs[stage_col].unique()
    contexts = adata.obs["_spatial_context"].unique()

    for stage in stages:
        stage_adata = adata[adata.obs[stage_col] == stage]

        for context in contexts:
            ctx_mask = stage_adata.obs["_spatial_context"] == context
            n_cells = ctx_mask.sum()

            if n_cells < 50:
                print(f"  Skipping {stage}/{context}: only {n_cells} cells")
                continue

            print(f"  Processing {stage}/{context}: {n_cells} cells")

            # Run DE vs rest of stage
            stage_adata.obs["_target"] = ctx_mask.map({True: "target", False: "rest"})

            try:
                sc.tl.rank_genes_groups(
                    stage_adata,
                    groupby="_target",
                    groups=["target"],
                    reference="rest",
                    method="wilcoxon",
                )

                de_df = sc.get.rank_genes_groups_df(stage_adata, group="target")

                # Create ranked list from DE results
                ranked = de_df.set_index("names")["scores"]
                ranked = ranked.sort_values(ascending=False)

                # Run GSEA prerank
                gsea_res = gp.prerank(
                    rnk=ranked,
                    gene_sets=gene_sets,
                    min_size=15,
                    max_size=500,
                    permutation_num=1000,
                    threads=4,
                    seed=42,
                    verbose=False,
                )

                res_df = gsea_res.res2d.copy()
                res_df["stage"] = stage
                res_df["spatial_context"] = context
                res_df["n_cells"] = n_cells

                all_results.append(res_df)

            except Exception as e:
                print(f"  Failed for {stage}/{context}: {e}")
                continue

    if all_results:
        return pd.concat(all_results, ignore_index=True)
    else:
        return pd.DataFrame()


def run_spatial_gsea_pipeline(
    h5ad_path: str | Path,
    output_dir: str | Path,
    cell_type_col: str = "cell_type",
    stage_col: str = "stage",
    context_radius: float = 50.0,
    gene_sets: str = "MSigDB_Hallmark_2020",
) -> dict[str, Path]:
    """Run complete spatial GSEA pipeline.

    Args:
        h5ad_path: Path to h5ad file with spatial coordinates
        output_dir: Output directory
        cell_type_col: Cell type column
        stage_col: Stage column
        context_radius: Neighborhood radius in microns
        gene_sets: Gene set library

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

    outputs = {}

    # Check for spatial coordinates
    has_spatial = "spatial" in adata.obsm

    if has_spatial:
        print("\nDefining spatial contexts...")
        spatial_contexts = define_spatial_contexts(
            adata,
            cell_type_col=cell_type_col,
            stage_col=stage_col,
            context_radius=context_radius,
        )
        spatial_contexts.to_parquet(output_dir / "spatial_contexts.parquet")
        outputs["spatial_contexts"] = output_dir / "spatial_contexts.parquet"

        print("\nContext distribution:")
        print(spatial_contexts["spatial_context"].value_counts())

        print("\nComputing spatial signatures...")
        signatures = compute_spatial_signatures(adata, spatial_contexts)
        signatures.to_parquet(output_dir / "spatial_signatures.parquet")
        outputs["signatures"] = output_dir / "spatial_signatures.parquet"

        print("\nRunning GSEA per context...")
        context_gsea = run_gsea_per_context(signatures, gene_sets=gene_sets)
        for context, res_df in context_gsea.items():
            out_path = output_dir / f"gsea_context_{context}.parquet"
            res_df.to_parquet(out_path)
            outputs[f"gsea_{context}"] = out_path
            print(f"  Saved {context}: {len(res_df)} pathways")

        print("\nRunning GSEA by stage and context...")
        stage_context_gsea = run_gsea_by_stage_and_context(
            adata, spatial_contexts, stage_col=stage_col, gene_sets=gene_sets
        )
        if len(stage_context_gsea) > 0:
            stage_context_gsea.to_parquet(output_dir / "gsea_stage_context.parquet")
            outputs["gsea_stage_context"] = output_dir / "gsea_stage_context.parquet"
            print(f"  Saved stage-context GSEA: {len(stage_context_gsea)} results")

    else:
        print("\nNo spatial coordinates found. Running cell-type-based GSEA...")
        # Fall back to cell-type-based context
        adata.obs["_spatial_context"] = adata.obs[cell_type_col]

        # Create pseudo-spatial contexts based on cell type neighborhoods
        spatial_contexts = pd.DataFrame({
            "spatial_context": adata.obs[cell_type_col].values,
            "n_neighbors": 0,  # not applicable
        }, index=adata.obs.index)

        print("\nRunning GSEA by stage and cell type...")
        stage_context_gsea = run_gsea_by_stage_and_context(
            adata, spatial_contexts, stage_col=stage_col, gene_sets=gene_sets
        )
        if len(stage_context_gsea) > 0:
            stage_context_gsea.to_parquet(output_dir / "gsea_stage_celltype.parquet")
            outputs["gsea_stage_celltype"] = output_dir / "gsea_stage_celltype.parquet"
            print(f"  Saved stage-celltype GSEA: {len(stage_context_gsea)} results")

    print("\nSpatial GSEA complete!")
    return outputs


def run_snrna_gsea_pipeline(
    h5ad_path: str | Path,
    output_dir: str | Path,
    cell_type_col: str = "cell_type",
    stage_col: str = "stage",
    gene_sets: str = "MSigDB_Hallmark_2020",
) -> dict[str, Path]:
    """Run GSEA pipeline for snRNA-seq data (no spatial).

    This is optimized for single-nucleus RNA-seq without spatial coordinates.
    Runs GSEA per cell type and per stage.

    Args:
        h5ad_path: Path to h5ad file
        output_dir: Output directory
        cell_type_col: Cell type column
        stage_col: Stage column
        gene_sets: Gene set library

    Returns:
        Dict of output paths
    """
    import gseapy as gp
    import scanpy as sc

    h5ad_path = Path(h5ad_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {h5ad_path}...")
    adata = sc.read_h5ad(h5ad_path)
    print(f"  {adata.n_obs} cells x {adata.n_vars} genes")

    outputs = {}

    # GSEA per cell type (comparing each to all others)
    print("\nRunning GSEA per cell type...")
    cell_types = adata.obs[cell_type_col].unique()

    celltype_results = []
    for ct in cell_types:
        ct_mask = adata.obs[cell_type_col] == ct
        n_cells = ct_mask.sum()

        if n_cells < 100:
            print(f"  Skipping {ct}: only {n_cells} cells")
            continue

        print(f"  Processing {ct}: {n_cells} cells")

        adata.obs["_target"] = ct_mask.map({True: "target", False: "rest"})

        try:
            sc.tl.rank_genes_groups(
                adata,
                groupby="_target",
                groups=["target"],
                reference="rest",
                method="wilcoxon",
            )

            de_df = sc.get.rank_genes_groups_df(adata, group="target")
            ranked = de_df.set_index("names")["scores"].sort_values(ascending=False)

            gsea_res = gp.prerank(
                rnk=ranked,
                gene_sets=gene_sets,
                min_size=15,
                max_size=500,
                permutation_num=1000,
                threads=4,
                seed=42,
                verbose=False,
            )

            res_df = gsea_res.res2d.copy()
            res_df["cell_type"] = ct
            res_df["n_cells"] = n_cells
            celltype_results.append(res_df)

        except Exception as e:
            print(f"  Failed for {ct}: {e}")

    if celltype_results:
        ct_df = pd.concat(celltype_results, ignore_index=True)
        ct_df.to_parquet(output_dir / "gsea_by_celltype.parquet")
        outputs["gsea_celltype"] = output_dir / "gsea_by_celltype.parquet"
        print(f"  Saved cell type GSEA: {len(ct_df)} results")

    # GSEA per stage (comparing each to all others)
    print("\nRunning GSEA per stage...")
    stages = adata.obs[stage_col].unique()

    stage_results = []
    for stage in stages:
        stage_mask = adata.obs[stage_col] == stage
        n_cells = stage_mask.sum()

        if n_cells < 100:
            print(f"  Skipping {stage}: only {n_cells} cells")
            continue

        print(f"  Processing {stage}: {n_cells} cells")

        adata.obs["_target"] = stage_mask.map({True: "target", False: "rest"})

        try:
            sc.tl.rank_genes_groups(
                adata,
                groupby="_target",
                groups=["target"],
                reference="rest",
                method="wilcoxon",
            )

            de_df = sc.get.rank_genes_groups_df(adata, group="target")
            ranked = de_df.set_index("names")["scores"].sort_values(ascending=False)

            gsea_res = gp.prerank(
                rnk=ranked,
                gene_sets=gene_sets,
                min_size=15,
                max_size=500,
                permutation_num=1000,
                threads=4,
                seed=42,
                verbose=False,
            )

            res_df = gsea_res.res2d.copy()
            res_df["stage"] = stage
            res_df["n_cells"] = n_cells
            stage_results.append(res_df)

        except Exception as e:
            print(f"  Failed for {stage}: {e}")

    if stage_results:
        stage_df = pd.concat(stage_results, ignore_index=True)
        stage_df.to_parquet(output_dir / "gsea_by_stage.parquet")
        outputs["gsea_stage"] = output_dir / "gsea_by_stage.parquet"
        print(f"  Saved stage GSEA: {len(stage_df)} results")

    # GSEA per stage-celltype combination
    print("\nRunning GSEA per stage-celltype...")
    combo_results = []

    for stage in stages:
        stage_adata = adata[adata.obs[stage_col] == stage]

        for ct in cell_types:
            ct_mask = stage_adata.obs[cell_type_col] == ct
            n_cells = ct_mask.sum()

            if n_cells < 50:
                continue

            print(f"  Processing {stage}/{ct}: {n_cells} cells")

            stage_adata.obs["_target"] = ct_mask.map({True: "target", False: "rest"})

            try:
                sc.tl.rank_genes_groups(
                    stage_adata,
                    groupby="_target",
                    groups=["target"],
                    reference="rest",
                    method="wilcoxon",
                )

                de_df = sc.get.rank_genes_groups_df(stage_adata, group="target")
                ranked = de_df.set_index("names")["scores"].sort_values(ascending=False)

                gsea_res = gp.prerank(
                    rnk=ranked,
                    gene_sets=gene_sets,
                    min_size=15,
                    max_size=500,
                    permutation_num=500,  # fewer permutations for speed
                    threads=4,
                    seed=42,
                    verbose=False,
                )

                res_df = gsea_res.res2d.copy()
                res_df["stage"] = stage
                res_df["cell_type"] = ct
                res_df["n_cells"] = n_cells
                combo_results.append(res_df)

            except Exception as e:
                print(f"  Failed for {stage}/{ct}: {e}")

    if combo_results:
        combo_df = pd.concat(combo_results, ignore_index=True)
        combo_df.to_parquet(output_dir / "gsea_by_stage_celltype.parquet")
        outputs["gsea_stage_celltype"] = output_dir / "gsea_by_stage_celltype.parquet"
        print(f"  Saved stage-celltype GSEA: {len(combo_df)} results")

    print("\nsnRNA GSEA complete!")
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run spatial or snRNA GSEA analysis")
    parser.add_argument("--h5ad", required=True, help="Path to h5ad file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--mode", choices=["spatial", "snrna"], default="snrna",
                        help="Analysis mode: spatial (with coordinates) or snrna (without)")
    parser.add_argument("--cell-type-col", default="cell_type", help="Cell type column")
    parser.add_argument("--stage-col", default="stage", help="Stage column")
    parser.add_argument("--context-radius", type=float, default=50.0,
                        help="Neighborhood radius in microns (spatial mode)")
    parser.add_argument("--gene-sets", default="MSigDB_Hallmark_2020",
                        help="Gene set library for GSEA")
    args = parser.parse_args()

    if args.mode == "spatial":
        run_spatial_gsea_pipeline(
            args.h5ad,
            args.output,
            cell_type_col=args.cell_type_col,
            stage_col=args.stage_col,
            context_radius=args.context_radius,
            gene_sets=args.gene_sets,
        )
    else:
        run_snrna_gsea_pipeline(
            args.h5ad,
            args.output,
            cell_type_col=args.cell_type_col,
            stage_col=args.stage_col,
            gene_sets=args.gene_sets,
        )
