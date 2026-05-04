"""External baseline methods for trajectory inference comparison.

These methods use external packages (moscot, cellrank, commot) and have
different training/inference paradigms than StageBridge.

Usage:
    python -m stagebridge.baselines.external \
        --method moscot \
        --data-dir /path/to/data \
        --output-dir /path/to/output \
        --fold-idx 0
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


def run_moscot(
    neighborhoods_path: Path,
    output_dir: Path,
    fold_idx: int = 0,
    epsilon: float = 0.05,
    tau_a: float = 0.95,
    tau_b: float = 0.95,
) -> dict:
    """Run moscot temporal OT for trajectory inference.

    moscot learns transport plans between timepoints using optimal transport.
    Unlike StageBridge, it does NOT learn a velocity field or condition on niche.

    Args:
        neighborhoods_path: Path to neighborhoods.parquet
        output_dir: Output directory
        fold_idx: Cross-validation fold (for train/test split)
        epsilon: Entropic regularization
        tau_a, tau_b: Unbalanced OT parameters

    Returns:
        Dictionary with metrics and predictions
    """
    try:
        import moscot
        from moscot.problems.time import TemporalProblem
    except ImportError:
        raise ImportError("moscot not installed. Run: pip install moscot")

    try:
        import anndata as ad
        import scanpy as sc
    except ImportError:
        raise ImportError("scanpy/anndata required for moscot")

    print(f"Running moscot (epsilon={epsilon}, tau_a={tau_a}, tau_b={tau_b})")

    # Load data
    df = pd.read_parquet(neighborhoods_path)

    # Build AnnData from receiver embeddings
    # Handle both formats: receiver_z (list/array column) or receiver_z_* (multiple columns)
    if "receiver_z" in df.columns:
        # receiver_z is a list/array per row
        X = np.stack(df["receiver_z"].values).astype(np.float32)
    else:
        receiver_cols = [c for c in df.columns if c.startswith("receiver_z_")]
        X = df[receiver_cols].values.astype(np.float32)

    # Stage mapping
    stage_map = {"Normal": 0, "Preinvasive": 1, "Invasive": 2}
    stages = df["stage"].map(stage_map).values

    adata = ad.AnnData(X=X.astype(np.float32))
    adata.obs["stage"] = stages
    adata.obs["stage_name"] = df["stage"].values
    adata.obs["cell_id"] = df["cell_id"].values if "cell_id" in df.columns else range(len(df))

    # Compute neighbors for moscot
    sc.pp.neighbors(adata, n_neighbors=30, use_rep="X")

    # Setup temporal problem
    tp = TemporalProblem(adata)
    tp = tp.prepare(time_key="stage")
    tp = tp.solve(epsilon=epsilon, tau_a=tau_a, tau_b=tau_b)

    # Extract transport matrices
    transport_01 = tp.solutions[(0, 1)].transport_matrix if (0, 1) in tp.solutions else None
    transport_12 = tp.solutions[(1, 2)].transport_matrix if (1, 2) in tp.solutions else None

    # Compute interpolated trajectories (moscot's "prediction")
    predictions = {}
    for source, target in [(0, 1), (1, 2), (0, 2)]:
        if (source, target) in tp.solutions:
            # Get source cells
            source_mask = adata.obs["stage"] == source
            source_X = adata.X[source_mask]

            # Transport to target
            T = tp.solutions[(source, target)].transport_matrix
            # Normalize rows to get conditional transport
            T_norm = T / (T.sum(axis=1, keepdims=True) + 1e-10)

            # Get target cells
            target_mask = adata.obs["stage"] == target
            target_X = adata.X[target_mask]

            # Predicted target = transport-weighted average of target cells
            if T_norm.shape[1] == target_X.shape[0]:
                predicted_X = T_norm @ target_X
                predictions[f"{source}_to_{target}"] = {
                    "source_embeddings": source_X,
                    "predicted_embeddings": predicted_X,
                    "transport_matrix": T,
                }

    # Compute metrics
    metrics = {
        "method": "moscot",
        "epsilon": epsilon,
        "tau_a": tau_a,
        "tau_b": tau_b,
        "n_cells": len(df),
        "n_stages": len(stage_map),
    }

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save predictions
    for key, pred in predictions.items():
        np.savez(
            output_dir / f"predictions_{key}.npz",
            source_embeddings=pred["source_embeddings"],
            predicted_embeddings=pred["predicted_embeddings"],
        )

    # Save metrics
    result = {
        "method": "moscot",
        "fold_idx": fold_idx,
        "metrics": metrics,
        "completed_at": datetime.now().isoformat(),
    }

    with open(output_dir / "moscot_results.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"moscot complete. Results saved to {output_dir}")
    return result


def run_cellrank(
    neighborhoods_path: Path,
    output_dir: Path,
    fold_idx: int = 0,
    n_states: int = 3,
) -> dict:
    """Run CellRank for trajectory inference.

    CellRank uses RNA velocity or pseudotime to infer transition probabilities.
    Since we don't have velocity, we use CytoTRACE-based kernel.

    Args:
        neighborhoods_path: Path to neighborhoods.parquet
        output_dir: Output directory
        fold_idx: Cross-validation fold
        n_states: Number of macrostates to identify

    Returns:
        Dictionary with metrics and predictions
    """
    try:
        import cellrank as cr
        from cellrank.kernels import CytoTRACEKernel
    except ImportError:
        raise ImportError("cellrank not installed. Run: pip install cellrank")

    try:
        import anndata as ad
        import scanpy as sc
    except ImportError:
        raise ImportError("scanpy/anndata required for cellrank")

    print(f"Running CellRank (n_states={n_states})")

    # Load h5ad with gene expression (required for CytoTRACE)
    snrna_h5ad = neighborhoods_path.parent.parent / "snrna_with_celltypes.h5ad"
    if not snrna_h5ad.exists():
        # Try config path
        snrna_h5ad = Path("/data1/chaunzt1/stagebridge/processed/luad_evo/snrna_with_celltypes.h5ad")

    if not snrna_h5ad.exists():
        raise FileNotFoundError(f"snRNA h5ad not found. CellRank CytoTRACE requires gene expression.")

    print(f"Loading snRNA data from {snrna_h5ad}")
    adata = ad.read_h5ad(snrna_h5ad)

    # Ensure stage column exists
    if "stage" not in adata.obs.columns:
        # Try to get from neighborhoods
        df = pd.read_parquet(neighborhoods_path)
        if "cell_id" in df.columns:
            stage_map = df.set_index("cell_id")["stage"].to_dict()
            adata.obs["stage"] = adata.obs.index.map(stage_map)

    # Preprocess if needed
    if "Ms" not in adata.layers:
        # CytoTRACE needs moments - compute them
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=2000)
        sc.pp.pca(adata)
        sc.pp.neighbors(adata)
        # Compute moments for CytoTRACE
        sc.tl.diffmap(adata)
        adata.layers["Ms"] = adata.X.copy()  # Use normalized counts as proxy

    # CytoTRACE kernel (uses gene expression entropy as proxy for differentiation)
    ctk = CytoTRACEKernel(adata)
    ctk.compute_cytotrace()
    ctk.compute_transition_matrix()

    # Compute terminal states
    g = cr.estimators.GPCCA(ctk)
    g.compute_schur(n_components=20)
    g.compute_macrostates(n_states=n_states, cluster_key="stage")

    # Predict terminal states (required before fate probabilities in CellRank 2.x)
    g.predict_terminal_states()

    # Get transition matrix
    T = ctk.transition_matrix

    # Compute fate probabilities
    g.compute_fate_probabilities()
    fate_probs = g.fate_probabilities

    # Metrics
    metrics = {
        "method": "cellrank",
        "n_states": n_states,
        "n_cells": len(df),
        "macrostates": list(g.macrostates.cat.categories) if hasattr(g.macrostates, 'cat') else [],
    }

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save fate probabilities
    if fate_probs is not None:
        np.save(output_dir / "fate_probabilities.npy", fate_probs.values)

    # Save transition matrix (sparse)
    from scipy import sparse
    sparse.save_npz(output_dir / "transition_matrix.npz", sparse.csr_matrix(T))

    result = {
        "method": "cellrank",
        "fold_idx": fold_idx,
        "metrics": metrics,
        "completed_at": datetime.now().isoformat(),
    }

    with open(output_dir / "cellrank_results.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"CellRank complete. Results saved to {output_dir}")
    return result


def run_commot(
    neighborhoods_path: Path,
    output_dir: Path,
    fold_idx: int = 0,
    database: str = "CellChat",
    spatial_h5ad_path: Path | None = None,
) -> dict:
    """Run COMMOT for cell-cell communication analysis.

    COMMOT uses optimal transport to infer spatial communication patterns.
    This serves as a baseline for StageBridge's attention-based niche modeling.

    Args:
        neighborhoods_path: Path to neighborhoods.parquet (for cell IDs and stage info)
        output_dir: Output directory
        fold_idx: Cross-validation fold
        database: Ligand-receptor database (CellChat, CellPhoneDB, etc.)
        spatial_h5ad_path: Path to spatial h5ad with gene expression (required)

    Returns:
        Dictionary with metrics and predictions
    """
    try:
        import commot as ct
    except ImportError:
        raise ImportError("commot not installed. Run: pip install commot")

    try:
        import anndata as ad
        import scanpy as sc
    except ImportError:
        raise ImportError("scanpy/anndata required for commot")

    print(f"Running COMMOT (database={database})")

    # Load neighborhoods for cell IDs and stage mapping
    df = pd.read_parquet(neighborhoods_path)

    # Load spatial h5ad with gene expression
    if spatial_h5ad_path is None:
        # Try default path
        spatial_h5ad_path = neighborhoods_path.parent.parent / "spatial_merged.h5ad"

    if not spatial_h5ad_path.exists():
        print(f"Warning: spatial h5ad not found at {spatial_h5ad_path}")
        print("COMMOT requires gene expression data. Skipping.")

        result = {
            "method": "commot",
            "fold_idx": fold_idx,
            "status": "skipped",
            "reason": f"spatial h5ad not found at {spatial_h5ad_path}",
            "completed_at": datetime.now().isoformat(),
        }

        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "commot_results.json", "w") as f:
            json.dump(result, f, indent=2)

        return result

    print(f"Loading spatial data from {spatial_h5ad_path}")
    adata = ad.read_h5ad(spatial_h5ad_path)

    # Add stage information from neighborhoods if not present
    if "stage" not in adata.obs.columns:
        # Match by cell_id if available
        if "cell_id" in df.columns and "cell_id" in adata.obs.columns:
            stage_map = df.set_index("cell_id")["stage"].to_dict()
            adata.obs["stage"] = adata.obs["cell_id"].map(stage_map)
        elif "cell_id" in df.columns:
            stage_map = df.set_index("cell_id")["stage"].to_dict()
            adata.obs["stage"] = adata.obs.index.map(stage_map)

    # Ensure spatial coordinates exist
    if "spatial" not in adata.obsm:
        if "x" in adata.obs.columns and "y" in adata.obs.columns:
            adata.obsm["spatial"] = adata.obs[["x", "y"]].values
        else:
            print("Warning: No spatial coordinates found in h5ad. Skipping COMMOT.")
            result = {
                "method": "commot",
                "fold_idx": fold_idx,
                "status": "skipped",
                "reason": "No spatial coordinates in h5ad",
                "completed_at": datetime.now().isoformat(),
            }
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_dir / "commot_results.json", "w") as f:
                json.dump(result, f, indent=2)
            return result

    # Preprocess if not already done
    if adata.X.max() > 50:  # Likely raw counts
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)

    # Get ligand-receptor database (COMMOT requires explicit df_ligrec)
    df_ligrec = ct.pp.ligand_receptor_database(
        database=database,
        species="human",
        signaling_type="Secreted Signaling",
    )

    # Filter to expressed genes
    df_ligrec = ct.pp.filter_lr_database(
        df_ligrec=df_ligrec,
        adata=adata,
        heteromeric=True,
        min_cell_pct=0.05,
    )

    # Run COMMOT
    ct.tl.spatial_communication(
        adata,
        database_name=database.lower(),  # lowercase for output keys
        df_ligrec=df_ligrec,
        dis_thr=500,  # Distance threshold in spatial units
        heteromeric=True,
        pathway_sum=True,
    )

    # Extract communication scores (key format: commot-{database_name}-sum-sender)
    sender_key = f"commot-{database.lower()}-sum-sender"
    receiver_key = f"commot-{database.lower()}-sum-receiver"
    if sender_key in adata.obsm:
        sender_scores = adata.obsm[sender_key]
        receiver_scores = adata.obsm[receiver_key]
    else:
        sender_scores = None
        receiver_scores = None

    metrics = {
        "method": "commot",
        "database": database,
        "n_cells": len(df),
        "n_genes": len(gene_cols),
        "has_spatial": "x" in df.columns,
    }

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)

    if sender_scores is not None:
        np.save(output_dir / "sender_scores.npy", sender_scores)
        np.save(output_dir / "receiver_scores.npy", receiver_scores)

    result = {
        "method": "commot",
        "fold_idx": fold_idx,
        "metrics": metrics,
        "completed_at": datetime.now().isoformat(),
    }

    with open(output_dir / "commot_results.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"COMMOT complete. Results saved to {output_dir}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Run external baseline methods")
    parser.add_argument(
        "--method",
        required=True,
        choices=["moscot", "cellrank", "commot"],
        help="External method to run",
    )
    parser.add_argument("--data-dir", required=True, type=Path, help="Data directory")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output directory")
    parser.add_argument("--fold-idx", type=int, default=0, help="CV fold index")
    parser.add_argument("--spatial-h5ad", type=Path, default=None,
                        help="Path to spatial h5ad with gene expression (for commot)")
    args = parser.parse_args()

    neighborhoods_path = args.data_dir / "neighborhoods.parquet"

    if args.method == "moscot":
        run_moscot(neighborhoods_path, args.output_dir, args.fold_idx)
    elif args.method == "cellrank":
        run_cellrank(neighborhoods_path, args.output_dir, args.fold_idx)
    elif args.method == "commot":
        spatial_path = args.spatial_h5ad or Path("/data1/chaunzt1/stagebridge/processed/luad_evo/spatial_merged.h5ad")
        run_commot(neighborhoods_path, args.output_dir, args.fold_idx, spatial_h5ad_path=spatial_path)


if __name__ == "__main__":
    main()
