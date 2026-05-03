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
    receiver_cols = [c for c in df.columns if c.startswith("receiver_z_")]
    X = df[receiver_cols].values

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

    # Load data
    df = pd.read_parquet(neighborhoods_path)

    # Build AnnData
    receiver_cols = [c for c in df.columns if c.startswith("receiver_z_")]
    X = df[receiver_cols].values

    adata = ad.AnnData(X=X.astype(np.float32))
    adata.obs["stage"] = df["stage"].values
    adata.obs["cell_id"] = df["cell_id"].values if "cell_id" in df.columns else range(len(df))

    # Compute neighbors
    sc.pp.neighbors(adata, n_neighbors=30, use_rep="X")

    # CytoTRACE kernel (uses gene expression entropy as proxy for differentiation)
    ctk = CytoTRACEKernel(adata)
    ctk.compute_cytotrace()
    ctk.compute_transition_matrix()

    # Compute terminal states
    g = cr.estimators.GPCCA(ctk)
    g.compute_schur(n_components=20)
    g.compute_macrostates(n_states=n_states, cluster_key="stage")

    # Get transition matrix
    T = ctk.transition_matrix

    # Compute absorption probabilities to terminal states
    g.compute_absorption_probabilities()
    abs_probs = g.absorption_probabilities

    # Metrics
    metrics = {
        "method": "cellrank",
        "n_states": n_states,
        "n_cells": len(df),
        "macrostates": list(g.macrostates.cat.categories) if hasattr(g.macrostates, 'cat') else [],
    }

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save absorption probabilities
    if abs_probs is not None:
        np.save(output_dir / "absorption_probabilities.npy", abs_probs.values)

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
) -> dict:
    """Run COMMOT for cell-cell communication analysis.

    COMMOT uses optimal transport to infer spatial communication patterns.
    This serves as a baseline for StageBridge's attention-based niche modeling.

    Note: COMMOT requires gene expression, not just embeddings. This implementation
    assumes the neighborhoods file has been augmented with gene expression or
    we use the embeddings as a proxy.

    Args:
        neighborhoods_path: Path to neighborhoods.parquet
        output_dir: Output directory
        fold_idx: Cross-validation fold
        database: Ligand-receptor database (CellChat, CellPhoneDB, etc.)

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

    # Load data
    df = pd.read_parquet(neighborhoods_path)

    # Check for gene expression columns
    # COMMOT needs actual gene expression, not just embeddings
    # For now, we'll check if we have it, otherwise skip
    gene_cols = [c for c in df.columns if not c.startswith(("receiver_", "ring_", "hlca_", "luca_"))]
    gene_cols = [c for c in gene_cols if c not in ["cell_id", "donor_id", "stage", "x", "y", "sample_id"]]

    if len(gene_cols) < 100:
        print("Warning: COMMOT requires gene expression data, not embeddings.")
        print("Skipping COMMOT analysis. To run, augment neighborhoods with gene expression.")

        result = {
            "method": "commot",
            "fold_idx": fold_idx,
            "status": "skipped",
            "reason": "Gene expression data required but not found",
            "completed_at": datetime.now().isoformat(),
        }

        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "commot_results.json", "w") as f:
            json.dump(result, f, indent=2)

        return result

    # Build AnnData with gene expression
    X = df[gene_cols].values
    adata = ad.AnnData(X=X.astype(np.float32))
    adata.var_names = gene_cols
    adata.obs["stage"] = df["stage"].values

    # Add spatial coordinates if available
    if "x" in df.columns and "y" in df.columns:
        adata.obsm["spatial"] = df[["x", "y"]].values

    # Preprocess
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)

    # Run COMMOT
    ct.tl.spatial_communication(
        adata,
        database_name=database,
        dis_thr=500,  # Distance threshold in spatial units
        heteromeric=True,
    )

    # Extract communication scores
    if "commot-cellchat-sum-sender" in adata.obsm:
        sender_scores = adata.obsm["commot-cellchat-sum-sender"]
        receiver_scores = adata.obsm["commot-cellchat-sum-receiver"]
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
    args = parser.parse_args()

    neighborhoods_path = args.data_dir / "neighborhoods.parquet"

    if args.method == "moscot":
        run_moscot(neighborhoods_path, args.output_dir, args.fold_idx)
    elif args.method == "cellrank":
        run_cellrank(neighborhoods_path, args.output_dir, args.fold_idx)
    elif args.method == "commot":
        run_commot(neighborhoods_path, args.output_dir, args.fold_idx)


if __name__ == "__main__":
    main()
