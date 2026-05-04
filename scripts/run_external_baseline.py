#!/usr/bin/env python
"""Standalone script for running external baselines.

This script is separate from stagebridge package to avoid import conflicts
when running in isolated conda environments.

Usage:
    python run_external_baseline.py \
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

import numpy as np
import pandas as pd


def run_moscot(
    neighborhoods_path: Path,
    output_dir: Path,
    fold_idx: int = 0,
    epsilon: float = 0.05,
    tau_a: float = 0.95,
    tau_b: float = 0.95,
    max_cells_per_stage: int = 30000,  # Subsample to avoid OOM
) -> dict:
    """Run moscot temporal OT for trajectory inference.

    Note: moscot builds full NxM transport matrices which can't fit in memory
    for large datasets. We subsample to max_cells_per_stage per stage.
    With 30K cells/stage, transport matrix is ~900M elements = ~7GB (feasible).
    """
    try:
        from moscot.problems.time import TemporalProblem
    except ImportError:
        raise ImportError("moscot not installed. Run: pip install moscot")

    import anndata as ad
    import scanpy as sc

    print(f"Running moscot (epsilon={epsilon}, tau_a={tau_a}, tau_b={tau_b})")

    df = pd.read_parquet(neighborhoods_path)
    print(f"Loaded {len(df)} cells")

    # Build AnnData from receiver embeddings
    if "receiver_z" in df.columns:
        X = np.stack(df["receiver_z"].values).astype(np.float32)
    else:
        receiver_cols = [c for c in df.columns if c.startswith("receiver_z_")]
        X = df[receiver_cols].values.astype(np.float32)

    stage_map = {"Normal": 0, "Preinvasive": 1, "Invasive": 2}
    stages = df["stage"].map(stage_map).values

    adata = ad.AnnData(X=X)
    adata.obs["stage"] = stages
    adata.obs["stage_name"] = df["stage"].values
    adata.obs["cell_id"] = df["cell_id"].values if "cell_id" in df.columns else range(len(df))

    # Subsample to avoid OOM - moscot builds full transport matrices
    # 331K x 296K = ~1.5TB which is impossible
    stage_counts = adata.obs["stage"].value_counts()
    print(f"Stage counts before subsampling: {dict(stage_counts)}")

    needs_subsample = any(count > max_cells_per_stage for count in stage_counts.values)
    if needs_subsample:
        print(f"Subsampling to max {max_cells_per_stage} cells per stage to avoid OOM")

        # Stratified sampling: preserve cell type proportions if available
        cell_type_col = None
        for col in ["cell_type", "celltype", "cell_type_fine"]:
            if col in df.columns:
                cell_type_col = col
                adata.obs["cell_type"] = df[col].values
                break

        keep_idx = []
        rng = np.random.default_rng(42 + fold_idx)

        for stage in stage_counts.index:
            stage_mask = adata.obs["stage"] == stage
            stage_idx = np.where(stage_mask)[0]

            if len(stage_idx) <= max_cells_per_stage:
                keep_idx.extend(stage_idx)
                continue

            if cell_type_col is not None:
                # Stratified by cell type
                stage_adata = adata[stage_mask]
                ct_counts = stage_adata.obs["cell_type"].value_counts()
                ct_fracs = ct_counts / ct_counts.sum()

                sampled_idx = []
                for ct, frac in ct_fracs.items():
                    ct_stage_idx = np.where(
                        (adata.obs["stage"] == stage) & (adata.obs["cell_type"] == ct)
                    )[0]
                    n_sample = max(1, int(frac * max_cells_per_stage))
                    n_sample = min(n_sample, len(ct_stage_idx))
                    sampled = rng.choice(ct_stage_idx, n_sample, replace=False)
                    sampled_idx.extend(sampled)

                # Trim to exact max if over
                if len(sampled_idx) > max_cells_per_stage:
                    sampled_idx = rng.choice(sampled_idx, max_cells_per_stage, replace=False).tolist()
                keep_idx.extend(sampled_idx)
            else:
                # Random subsample
                sampled = rng.choice(stage_idx, max_cells_per_stage, replace=False)
                keep_idx.extend(sampled)

        keep_idx = sorted(set(keep_idx))
        adata = adata[keep_idx].copy()
        print(f"After stratified subsampling: {len(adata)} cells")
        stage_counts = adata.obs["stage"].value_counts()
        print(f"Stage counts after subsampling: {dict(stage_counts)}")

    sc.pp.neighbors(adata, n_neighbors=30, use_rep="X")

    tp = TemporalProblem(adata)
    tp = tp.prepare(time_key="stage")
    tp = tp.solve(epsilon=epsilon, tau_a=tau_a, tau_b=tau_b)

    predictions = {}
    for source, target in [(0, 1), (1, 2), (0, 2)]:
        if (source, target) in tp.solutions:
            source_mask = adata.obs["stage"] == source
            source_X = adata.X[source_mask]
            T = tp.solutions[(source, target)].transport_matrix
            T_norm = T / (T.sum(axis=1, keepdims=True) + 1e-10)
            target_mask = adata.obs["stage"] == target
            target_X = adata.X[target_mask]
            if T_norm.shape[1] == target_X.shape[0]:
                predicted_X = T_norm @ target_X
                predictions[f"{source}_to_{target}"] = {
                    "source_embeddings": source_X,
                    "predicted_embeddings": predicted_X,
                }

    metrics = {
        "method": "moscot",
        "epsilon": epsilon,
        "tau_a": tau_a,
        "tau_b": tau_b,
        "n_cells_original": len(df),
        "n_cells_used": len(adata),
        "max_cells_per_stage": max_cells_per_stage,
        "subsampled": needs_subsample,
        "n_stages": len(stage_map),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    for key, pred in predictions.items():
        np.savez(
            output_dir / f"predictions_{key}.npz",
            source_embeddings=pred["source_embeddings"],
            predicted_embeddings=pred["predicted_embeddings"],
        )

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
    snrna_h5ad_path: Path | None = None,
) -> dict:
    """Run CellRank for trajectory inference."""
    try:
        import cellrank as cr
        from cellrank.kernels import CytoTRACEKernel
    except ImportError:
        raise ImportError("cellrank not installed. Run: pip install cellrank")

    import anndata as ad
    import scanpy as sc

    print(f"Running CellRank (n_states={n_states})")

    # Load h5ad with gene expression
    if snrna_h5ad_path is None:
        snrna_h5ad_path = neighborhoods_path.parent.parent / "snrna_with_celltypes.h5ad"
    if not snrna_h5ad_path.exists():
        snrna_h5ad_path = Path("/data1/chaunzt1/stagebridge/processed/luad_evo/snrna_with_celltypes.h5ad")

    if not snrna_h5ad_path.exists():
        raise FileNotFoundError(f"snRNA h5ad not found at {snrna_h5ad_path}")

    print(f"Loading snRNA data from {snrna_h5ad_path}")
    adata = ad.read_h5ad(snrna_h5ad_path)

    # Get stage from neighborhoods if needed
    if "stage" not in adata.obs.columns:
        df = pd.read_parquet(neighborhoods_path)
        if "cell_id" in df.columns:
            stage_map = df.set_index("cell_id")["stage"].to_dict()
            adata.obs["stage"] = adata.obs.index.map(stage_map)

    # Preprocess if needed
    if "Ms" not in adata.layers:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=2000)
        sc.pp.pca(adata)
        sc.pp.neighbors(adata)
        sc.tl.diffmap(adata)
        adata.layers["Ms"] = adata.X.copy()

    ctk = CytoTRACEKernel(adata)
    ctk.compute_cytotrace()
    ctk.compute_transition_matrix()

    g = cr.estimators.GPCCA(ctk)
    g.compute_schur(n_components=20)
    g.compute_macrostates(n_states=n_states, cluster_key="stage")
    g.predict_terminal_states()

    T = ctk.transition_matrix
    g.compute_fate_probabilities()
    fate_probs = g.fate_probabilities

    metrics = {
        "method": "cellrank",
        "n_states": n_states,
        "n_cells": adata.n_obs,
        "macrostates": list(g.macrostates.cat.categories) if hasattr(g.macrostates, 'cat') else [],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    if fate_probs is not None:
        np.save(output_dir / "fate_probabilities.npy", fate_probs.values)

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
    """Run COMMOT for cell-cell communication analysis."""
    try:
        import commot as ct
    except ImportError:
        raise ImportError("commot not installed. Run: pip install commot")

    import anndata as ad
    import scanpy as sc

    print(f"Running COMMOT (database={database})")

    if spatial_h5ad_path is None:
        spatial_h5ad_path = neighborhoods_path.parent.parent / "spatial_merged.h5ad"

    if not spatial_h5ad_path.exists():
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

    df = pd.read_parquet(neighborhoods_path)
    if "stage" not in adata.obs.columns and "cell_id" in df.columns:
        stage_map = df.set_index("cell_id")["stage"].to_dict()
        adata.obs["stage"] = adata.obs.index.map(stage_map)

    if "spatial" not in adata.obsm:
        if "x" in adata.obs.columns and "y" in adata.obs.columns:
            adata.obsm["spatial"] = adata.obs[["x", "y"]].values
        else:
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

    if adata.X.max() > 50:
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)

    df_ligrec = ct.pp.ligand_receptor_database(
        database=database,
        species="human",
        signaling_type="Secreted Signaling",
    )

    df_ligrec = ct.pp.filter_lr_database(
        df_ligrec=df_ligrec,
        adata=adata,
        heteromeric=True,
        min_cell_pct=0.05,
    )

    ct.tl.spatial_communication(
        adata,
        database_name=database.lower(),
        df_ligrec=df_ligrec,
        dis_thr=500,
        heteromeric=True,
        pathway_sum=True,
    )

    sender_key = f"commot-{database.lower()}-sum-sender"
    receiver_key = f"commot-{database.lower()}-sum-receiver"
    sender_scores = adata.obsm.get(sender_key)
    receiver_scores = adata.obsm.get(receiver_key)

    metrics = {
        "method": "commot",
        "database": database,
        "n_cells": adata.n_obs,
        "n_genes": adata.n_vars,
    }

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
    parser.add_argument("--method", required=True, choices=["moscot", "cellrank", "commot"])
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fold-idx", type=int, default=0)
    parser.add_argument("--spatial-h5ad", type=Path, default=None)
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
