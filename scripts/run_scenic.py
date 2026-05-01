#!/usr/bin/env python
"""Full pySCENIC regulon analysis.

Usage:
    python scripts/run_scenic.py [--figures]
"""

import argparse
import sys
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse

# Paths
DATA = Path("/data1/chaunzt1/stagebridge/processed/luad_evo")
CANONICAL = DATA / "canonical"
SNRNA = DATA / "snrna_with_celltypes.h5ad"
DB_DIR = DATA / "scenic_dbs"
MOTIF_DB = DB_DIR / "hg38_10kbp_up_10kbp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather"
ANNOTATIONS = DB_DIR / "motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl"
TF_LIST = DB_DIR / "allTFs_hg38.txt"
OUTPUT_DIR = CANONICAL / "scenic"
N_JOBS = 8


def load_h5ad_minimal(path):
    """Load h5ad without problematic uns fields."""
    print(f"Loading {path}...")

    with h5py.File(path, "r") as f:
        # Read var names
        var_names = None
        for key in ["gene", "gene_ids", "gene_names", "_index", "index"]:
            if "var" in f and key in f["var"]:
                data = f["var"][key]
                if isinstance(data, h5py.Dataset):
                    var_names = data[:].astype(str)
                    print(f"  Using var key: {key}")
                    break

        # Read obs names and stage
        obs_names = None
        stages = None
        cell_types = None
        for key in ["cell_id", "barcode", "cell_ids", "_index", "index"]:
            if "obs" in f and key in f["obs"]:
                data = f["obs"][key]
                if isinstance(data, h5py.Dataset):
                    obs_names = data[:].astype(str)
                    print(f"  Using obs key: {key}")
                    break

        # Try to get stage
        if "obs" in f and "stage" in f["obs"]:
            data = f["obs"]["stage"]
            if isinstance(data, h5py.Dataset):
                stages = data[:].astype(str)

        # Try to get cell type
        for key in ["cell_type", "luca_cell_type", "cell_type_luca"]:
            if "obs" in f and key in f["obs"]:
                data = f["obs"][key]
                if isinstance(data, h5py.Dataset):
                    cell_types = data[:].astype(str)
                    break

        # Read X (handle sparse or dense)
        if "X" in f:
            X_grp = f["X"]
            if isinstance(X_grp, h5py.Dataset):
                X = X_grp[:]
            elif "data" in X_grp:  # sparse format
                data = X_grp["data"][:]
                indices = X_grp["indices"][:]
                indptr = X_grp["indptr"][:]
                shape = tuple(X_grp.attrs["shape"]) if "shape" in X_grp.attrs else (len(obs_names), len(var_names))
                X = sparse.csr_matrix((data, indices, indptr), shape=shape)
            else:
                raise ValueError(f"Unknown X format: {list(X_grp.keys())}")

    if not sparse.issparse(X):
        X = sparse.csr_matrix(X)

    adata = ad.AnnData(X=X)
    if obs_names is not None:
        adata.obs_names = pd.Index(obs_names)
    if var_names is not None:
        adata.var_names = pd.Index(var_names)
    if stages is not None:
        adata.obs["stage"] = stages
    if cell_types is not None:
        adata.obs["cell_type"] = cell_types

    print(f"  {adata.n_obs:,} cells x {adata.n_vars:,} genes")
    return adata


def run_scenic():
    """Run full pySCENIC pipeline."""
    from arboreto.algo import grnboost2
    from arboreto.utils import load_tf_names
    from ctxcore.rnkdb import FeatherRankingDatabase
    from pyscenic.aucell import aucell
    from pyscenic.prune import df2regulons, prune2df

    print("Running full pySCENIC pipeline...")
    print("  This takes 2-4 hours on 800k cells")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    adata = load_h5ad_minimal(SNRNA)

    # Get expression matrix
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    expr_df = pd.DataFrame(X, index=adata.obs_names, columns=adata.var_names)

    # Step 1: GRN inference
    adj_path = OUTPUT_DIR / "adjacencies.parquet"
    if adj_path.exists():
        print(f"Loading existing adjacencies from {adj_path}")
        adjacencies = pd.read_parquet(adj_path)
    else:
        print("Step 1: Running GRNBoost2...")
        tf_list = load_tf_names(str(TF_LIST))
        tf_list = [tf for tf in tf_list if tf in adata.var_names]
        print(f"  Using {len(tf_list)} TFs present in data")

        adjacencies = grnboost2(
            expression_data=expr_df,
            tf_names=tf_list,
            verbose=True,
            client_or_address="local",
            seed=42,
        )
        adjacencies.to_parquet(adj_path)
        print(f"  Found {len(adjacencies):,} TF-target pairs")

    # Step 2: Motif pruning
    print("Step 2: Pruning with cistarget motif enrichment...")
    dbs = [FeatherRankingDatabase(MOTIF_DB)]
    df_motifs = prune2df(dbs, adjacencies, str(ANNOTATIONS), num_workers=N_JOBS)
    regulons = df2regulons(df_motifs)
    print(f"  Found {len(regulons)} regulons")

    # Step 3: AUCell scoring
    print("Step 3: Computing AUCell scores...")
    auc_mtx = aucell(expr_df, regulons, num_workers=N_JOBS)
    auc_path = OUTPUT_DIR / "aucell_scores.parquet"
    auc_mtx.to_parquet(auc_path)
    print(f"  Computed activity for {auc_mtx.shape[0]:,} cells x {auc_mtx.shape[1]} regulons")

    # Save regulon summary
    print("Saving regulon summary...")
    regulon_dict = {r.name: list(r.genes) for r in regulons}
    regulon_sizes = {name: len(genes) for name, genes in regulon_dict.items()}
    summary = pd.DataFrame({
        "regulon": list(regulon_sizes.keys()),
        "n_genes": list(regulon_sizes.values()),
        "mean_activity": [auc_mtx[r].mean() for r in regulon_sizes.keys()],
        "std_activity": [auc_mtx[r].std() for r in regulon_sizes.keys()],
    })
    summary.to_parquet(OUTPUT_DIR / "regulon_scores.parquet")

    print("pySCENIC complete!")
    print(f"  adjacencies: {adj_path}")
    print(f"  aucell: {auc_path}")
    print(f"  summary: {OUTPUT_DIR / 'regulon_scores.parquet'}")

    return adata, auc_mtx, adjacencies


def generate_figures(adata, auc_mtx):
    """Generate publication-quality SCENIC figures."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["font.size"] = 10

    FIGURE_DIR = OUTPUT_DIR / "figures"
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    stage_order = ["Normal", "AAH", "AIS", "MIA", "LUAD"]
    stages = adata.obs.get("stage")
    cell_types = adata.obs.get("cell_type")

    # Align indices
    common = auc_mtx.index.intersection(adata.obs.index)
    auc = auc_mtx.loc[common]
    if stages is not None:
        stages = adata.obs.loc[common, "stage"]
    if cell_types is not None:
        cell_types = adata.obs.loc[common, "cell_type"]

    print(f"  {len(auc)} cells, {auc.shape[1]} regulons")

    # Figure 1: Top regulons heatmap by stage
    print("Figure 1: Regulon activity heatmap by stage...")
    if stages is not None:
        stage_means = auc.groupby(stages).mean()
        stage_means = stage_means.reindex([s for s in stage_order if s in stage_means.index])

        regulon_var = auc.var().sort_values(ascending=False)
        top_regulons = regulon_var.head(40).index.tolist()

        fig, ax = plt.subplots(figsize=(14, 8))
        sns.heatmap(stage_means[top_regulons].T, cmap="RdBu_r", center=0,
                    xticklabels=True, yticklabels=True, ax=ax,
                    cbar_kws={"label": "Mean AUCell Score"})
        ax.set_xlabel("Stage")
        ax.set_ylabel("Regulon (TF)")
        ax.set_title("Top 40 Variable Regulons Across Disease Stages")
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "regulon_heatmap_by_stage.png", bbox_inches="tight")
        plt.close()
        print("  Saved regulon_heatmap_by_stage.png")

    # Figure 2: Regulon specificity dotplot
    print("Figure 2: Stage-specific regulon dotplot...")
    if stages is not None:
        stage_means = auc.groupby(stages).mean()
        global_mean = auc.mean()
        global_std = auc.std()
        zscore = (stage_means - global_mean) / (global_std + 1e-8)
        zscore = zscore.reindex([s for s in stage_order if s in zscore.index])

        top_per_stage = []
        for stage in zscore.index:
            top = zscore.loc[stage].nlargest(8).index.tolist()
            top_per_stage.extend(top)
        top_specific = list(dict.fromkeys(top_per_stage))[:35]

        plot_data = []
        for stage in zscore.index:
            for reg in top_specific:
                plot_data.append({
                    "Stage": stage,
                    "Regulon": reg,
                    "Z-score": zscore.loc[stage, reg],
                    "Mean Activity": stage_means.loc[stage, reg],
                })
        plot_df = pd.DataFrame(plot_data)

        fig, ax = plt.subplots(figsize=(12, 10))
        scatter = ax.scatter(
            plot_df["Stage"].map({s: i for i, s in enumerate(stage_order)}),
            plot_df["Regulon"],
            c=plot_df["Z-score"],
            s=np.abs(plot_df["Z-score"]) * 50 + 20,
            cmap="RdBu_r",
            vmin=-3, vmax=3,
            alpha=0.8,
            edgecolors="black",
            linewidths=0.5,
        )
        ax.set_xticks(range(len([s for s in stage_order if s in zscore.index])))
        ax.set_xticklabels([s for s in stage_order if s in zscore.index])
        ax.set_xlabel("Stage")
        ax.set_ylabel("Regulon (TF)")
        ax.set_title("Stage-Specific Regulon Activity")
        plt.colorbar(scatter, ax=ax, label="Z-score")
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "regulon_specificity_dotplot.png", bbox_inches="tight")
        plt.close()
        print("  Saved regulon_specificity_dotplot.png")

    # Figure 3: Regulon activity by cell type
    print("Figure 3: Regulon activity by cell type...")
    if cell_types is not None:
        ct_means = auc.groupby(cell_types).mean()

        ct_counts = cell_types.value_counts()
        major_cts = ct_counts[ct_counts >= 100].index.tolist()[:20]
        ct_means = ct_means.loc[ct_means.index.isin(major_cts)]

        ct_var = ct_means.var().sort_values(ascending=False)
        top_ct_regs = ct_var.head(30).index.tolist()

        fig, ax = plt.subplots(figsize=(16, 10))
        sns.heatmap(ct_means[top_ct_regs].T, cmap="YlOrRd",
                    xticklabels=True, yticklabels=True, ax=ax,
                    cbar_kws={"label": "Mean AUCell Score"})
        ax.set_xlabel("Cell Type")
        ax.set_ylabel("Regulon (TF)")
        ax.set_title("Top Regulons by Cell Type")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "regulon_heatmap_by_celltype.png", bbox_inches="tight")
        plt.close()
        print("  Saved regulon_heatmap_by_celltype.png")

    # Figure 4: TF-target network
    print("Figure 4: TF-target network...")
    adj_path = OUTPUT_DIR / "adjacencies.parquet"
    if adj_path.exists():
        try:
            import networkx as nx
            adj = pd.read_parquet(adj_path)

            tf_importance = adj.groupby("TF")["importance"].sum().nlargest(15)
            top_tfs = tf_importance.index.tolist()

            G = nx.DiGraph()
            for tf in top_tfs:
                G.add_node(tf, node_type="TF")
                targets = adj[adj["TF"] == tf].nlargest(5, "importance")
                for _, row in targets.iterrows():
                    G.add_node(row["target"], node_type="target")
                    G.add_edge(tf, row["target"], weight=row["importance"])

            fig, ax = plt.subplots(figsize=(14, 14))
            pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

            tf_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "TF"]
            target_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "target"]

            nx.draw_networkx_nodes(G, pos, nodelist=tf_nodes, node_color="#E74C3C",
                                  node_size=800, alpha=0.9, ax=ax)
            nx.draw_networkx_nodes(G, pos, nodelist=target_nodes, node_color="#3498DB",
                                  node_size=300, alpha=0.7, ax=ax)

            edges = G.edges(data=True)
            weights = [d["weight"] for _, _, d in edges]
            max_w = max(weights) if weights else 1
            edge_widths = [2 * w / max_w + 0.5 for w in weights]
            nx.draw_networkx_edges(G, pos, alpha=0.5, width=edge_widths,
                                  edge_color="gray", arrows=True,
                                  arrowsize=15, ax=ax)

            nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold", ax=ax)

            ax.set_title("Top TF-Target Regulatory Network\n(Red=TF, Blue=Target)", fontsize=14)
            ax.axis("off")
            plt.tight_layout()
            plt.savefig(FIGURE_DIR / "tf_target_network.png", bbox_inches="tight")
            plt.close()
            print("  Saved tf_target_network.png")
        except ImportError:
            print("  networkx not installed, skipping network figure")

    # Figure 5: Binarized regulon activity
    print("Figure 5: Binarized regulon states...")
    if stages is not None:
        binary = pd.DataFrame(index=auc.index)
        for col in auc.columns[:30]:
            threshold = auc[col].quantile(0.75)
            binary[col] = (auc[col] > threshold).astype(int)

        binary["stage"] = stages.values
        prop_on = binary.groupby("stage").mean()
        prop_on = prop_on.reindex([s for s in stage_order if s in prop_on.index])

        fig, ax = plt.subplots(figsize=(14, 8))
        sns.heatmap(prop_on.T, cmap="YlGnBu", vmin=0, vmax=1,
                    xticklabels=True, yticklabels=True, ax=ax,
                    cbar_kws={"label": "Proportion Active"})
        ax.set_xlabel("Stage")
        ax.set_ylabel("Regulon (TF)")
        ax.set_title("Proportion of Cells with Active Regulon (>75th percentile)")
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "regulon_binary_heatmap.png", bbox_inches="tight")
        plt.close()
        print("  Saved regulon_binary_heatmap.png")

    # Figure 6: Progression-associated regulons
    print("Figure 6: Progression trajectory regulons...")
    if stages is not None:
        stage_num = stages.map({"Normal": 0, "AAH": 1, "AIS": 2, "MIA": 3, "LUAD": 4})
        valid = ~stage_num.isna()

        correlations = {}
        for col in auc.columns:
            if valid.sum() > 100:
                correlations[col] = np.corrcoef(stage_num[valid], auc.loc[valid.values, col])[0, 1]

        corr_df = pd.Series(correlations).sort_values()

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        top_increasing = corr_df.tail(15)
        axes[0].barh(range(len(top_increasing)), top_increasing.values, color="#E74C3C")
        axes[0].set_yticks(range(len(top_increasing)))
        axes[0].set_yticklabels(top_increasing.index)
        axes[0].set_xlabel("Correlation with Progression")
        axes[0].set_title("Regulons INCREASING with Progression")
        axes[0].axvline(0, color="black", linestyle="-", linewidth=0.5)

        top_decreasing = corr_df.head(15)
        axes[1].barh(range(len(top_decreasing)), top_decreasing.values, color="#3498DB")
        axes[1].set_yticks(range(len(top_decreasing)))
        axes[1].set_yticklabels(top_decreasing.index)
        axes[1].set_xlabel("Correlation with Progression")
        axes[1].set_title("Regulons DECREASING with Progression")
        axes[1].axvline(0, color="black", linestyle="-", linewidth=0.5)

        plt.suptitle("Regulons Associated with Disease Progression", fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "regulon_progression_correlation.png", bbox_inches="tight")
        plt.close()
        print("  Saved regulon_progression_correlation.png")

    print(f"\nAll figures saved to: {FIGURE_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pySCENIC analysis")
    parser.add_argument("--figures", action="store_true", help="Run SCENIC then generate figures")
    parser.add_argument("--figures-only", action="store_true", help="Generate figures from existing results")
    args = parser.parse_args()

    # Figures-only mode (called by shell script after CLI completes)
    if args.figures_only:
        if not (OUTPUT_DIR / "aucell_scores.parquet").exists():
            print("ERROR: No aucell_scores.parquet found. Run SCENIC first.")
            sys.exit(1)
        print("Generating figures from existing results...")
        adata = load_h5ad_minimal(SNRNA)
        auc_mtx = pd.read_parquet(OUTPUT_DIR / "aucell_scores.parquet")
        generate_figures(adata, auc_mtx)
        sys.exit(0)

    # Check if results exist
    if (OUTPUT_DIR / "aucell_scores.parquet").exists():
        print("SCENIC results already exist")
        if args.figures:
            print("Generating figures from existing results...")
            adata = load_h5ad_minimal(SNRNA)
            auc_mtx = pd.read_parquet(OUTPUT_DIR / "aucell_scores.parquet")
            generate_figures(adata, auc_mtx)
        sys.exit(0)

    # Run SCENIC
    adata, auc_mtx, adjacencies = run_scenic()

    # Generate figures if requested
    if args.figures:
        print("\n" + "=" * 46)
        print("Generating SCENIC figures...")
        print("=" * 46)
        generate_figures(adata, auc_mtx)
