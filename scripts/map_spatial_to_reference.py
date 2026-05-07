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

import numpy as np
import pandas as pd
import anndata as ad


def map_spatial_to_reference(
    spatial_path: Path,
    output_dir: Path,
    hlca_model_dir: Path,
    luca_model_dir: Path,
    hlca_ref_path: Path | None = None,
    luca_ref_path: Path | None = None,
    batch_size: int = 1024,
    surgery_epochs: int = 200,
):
    """Map spatial spots to HLCA and LuCA reference spaces.

    Args:
        spatial_path: Path to spatial_merged.h5ad
        output_dir: Output directory for embeddings
        hlca_model_dir: Path to HLCA scANVI model
        luca_model_dir: Path to LuCA scANVI model
        hlca_ref_path: Path to HLCA reference h5ad (optional, for gene alignment)
        luca_ref_path: Path to LuCA reference h5ad (optional, for gene alignment)
        batch_size: Inference batch size
        surgery_epochs: Max epochs for scArches surgery
    """
    try:
        from scvi.model import SCANVI
    except ImportError:
        raise ImportError("scvi-tools required. Install with: pip install scvi-tools")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Mapping spatial spots to reference spaces")
    print("=" * 60)

    # Load spatial data
    print(f"\nLoading spatial data from {spatial_path}...")
    spatial = ad.read_h5ad(spatial_path)
    print(f"  {spatial.n_obs:,} spots, {spatial.n_vars:,} genes")

    # Map to HLCA
    print(f"\n{'=' * 60}")
    print("Mapping to HLCA reference (30d)")
    print("=" * 60)
    hlca_result = _map_to_reference(
        spatial,
        hlca_model_dir,
        hlca_ref_path,
        reference_name="HLCA",
        batch_size=batch_size,
        surgery_epochs=surgery_epochs,
    )

    # Map to LuCA
    print(f"\n{'=' * 60}")
    print("Mapping to LuCA reference (10d)")
    print("=" * 60)
    luca_result = _map_to_reference(
        spatial,
        luca_model_dir,
        luca_ref_path,
        reference_name="LuCA",
        batch_size=batch_size,
        surgery_epochs=surgery_epochs,
        # LuCA architecture (from retrain_luca_multigpu.py) - used if attr.pkl missing
        n_latent=10,
        n_hidden=128,
        n_layers=2,
        batch_key="dataset",
        labels_key="cell_type",
    )

    # Save embeddings
    print(f"\n{'=' * 60}")
    print("Saving embeddings")
    print("=" * 60)

    # HLCA embeddings
    hlca_df = pd.DataFrame(
        hlca_result["latent"],
        index=spatial.obs_names,
        columns=[f"hlca_latent_{i}" for i in range(hlca_result["latent"].shape[1])]
    )
    hlca_df.index.name = "cell_id"
    if hlca_result["labels"] is not None:
        hlca_df["cell_type_hlca"] = hlca_result["labels"]
    if hlca_result["confidence"] is not None:
        hlca_df["cell_type_hlca_confidence"] = hlca_result["confidence"]

    hlca_path = output_dir / "spatial_hlca_embedding.parquet"
    hlca_df.to_parquet(hlca_path)
    print(f"  Saved HLCA embeddings: {hlca_path}")
    print(f"    Shape: {hlca_result['latent'].shape}")

    # LuCA embeddings
    luca_df = pd.DataFrame(
        luca_result["latent"],
        index=spatial.obs_names,
        columns=[f"luca_latent_{i}" for i in range(luca_result["latent"].shape[1])]
    )
    luca_df.index.name = "cell_id"
    if luca_result["labels"] is not None:
        luca_df["cell_type_luca"] = luca_result["labels"]
    if luca_result["confidence"] is not None:
        luca_df["cell_type_luca_confidence"] = luca_result["confidence"]

    luca_path = output_dir / "spatial_luca_embedding.parquet"
    luca_df.to_parquet(luca_path)
    print(f"  Saved LuCA embeddings: {luca_path}")
    print(f"    Shape: {luca_result['latent'].shape}")

    # Fused embeddings (concatenate HLCA + LuCA)
    fused_latent = np.concatenate([hlca_result["latent"], luca_result["latent"]], axis=1)
    fused_df = pd.DataFrame(
        fused_latent,
        index=spatial.obs_names,
        columns=[f"hlca_latent_{i}" for i in range(hlca_result["latent"].shape[1])] +
                [f"luca_latent_{i}" for i in range(luca_result["latent"].shape[1])]
    )
    fused_df.index.name = "cell_id"

    # Add metadata
    for col in ["donor_id", "patient_id", "stage", "sample_id"]:
        if col in spatial.obs.columns:
            fused_df[col] = spatial.obs[col].values

    fused_path = output_dir / "spatial_fused_embedding.parquet"
    fused_df.to_parquet(fused_path)
    print(f"  Saved fused embeddings: {fused_path}")
    print(f"    Shape: {fused_latent.shape}")

    print(f"\n{'=' * 60}")
    print("Done!")
    print("=" * 60)

    return {
        "hlca": hlca_result,
        "luca": luca_result,
        "fused_path": fused_path,
    }


def _map_to_reference(
    adata: ad.AnnData,
    model_dir: Path,
    ref_path: Path | None,
    reference_name: str,
    batch_size: int = 1024,
    surgery_epochs: int = 200,
    # Manual architecture override for models without attr.pkl
    n_latent: int | None = None,
    n_hidden: int | None = None,
    n_layers: int | None = None,
    batch_key: str = "dataset",
    labels_key: str = "cell_type",
) -> dict:
    """Map query cells to reference space via scArches surgery."""
    from scvi.model import SCANVI
    import torch

    model_dir = Path(model_dir)

    print(f"  Loading {reference_name} model from {model_dir}...")

    # Load reference if provided (for gene alignment)
    ref_adata = None
    if ref_path and Path(ref_path).exists():
        print(f"  Loading reference from {ref_path}...")
        ref_adata = ad.read_h5ad(ref_path)
        print(f"    Reference: {ref_adata.n_obs:,} cells, {ref_adata.n_vars:,} genes")

    # Check if model has attr.pkl or metadata json (standard scvi-tools format)
    has_metadata = (
        (model_dir / "attr.pkl").exists() or
        (model_dir / "_scvi_required_metadata.json").exists()
    )

    # Load model
    ref_model = None
    if has_metadata:
        # Standard load
        try:
            ref_model = SCANVI.load(str(model_dir), adata=ref_adata)
            print(f"  Model loaded successfully")
            print(f"    Latent dim: {ref_model.module.n_latent}")
        except Exception as e:
            print(f"  Standard load failed: {e}")

    if ref_model is None and n_latent is not None and ref_adata is not None:
        # Manual reconstruction from weights + architecture
        print(f"  Attempting manual model reconstruction...")
        print(f"    Architecture: n_latent={n_latent}, n_hidden={n_hidden}, n_layers={n_layers}")

        # Use raw counts if available
        if ref_adata.raw is not None:
            print(f"    Using adata.raw for counts")
            ref_adata_counts = ref_adata.raw.to_adata()
        elif "counts" in ref_adata.layers:
            print(f"    Using adata.layers['counts']")
            ref_adata_counts = ref_adata.copy()
            ref_adata_counts.X = ref_adata_counts.layers["counts"]
        else:
            ref_adata_counts = ref_adata

        # Setup anndata for scANVI
        SCANVI.setup_anndata(
            ref_adata_counts,
            batch_key=batch_key,
            labels_key=labels_key,
        )

        # Create model with same architecture
        ref_model = SCANVI(
            ref_adata_counts,
            n_latent=n_latent,
            n_hidden=n_hidden or 128,
            n_layers=n_layers or 2,
            unlabeled_category="Unknown",
        )

        # Load weights
        model_pt = model_dir / "model.pt"
        if model_pt.exists():
            state_dict = torch.load(model_pt, map_location="cpu")
            ref_model.module.load_state_dict(state_dict)
            print(f"  Loaded weights from {model_pt}")
        else:
            raise FileNotFoundError(f"No model.pt found at {model_dir}")

    if ref_model is None:
        raise RuntimeError(
            f"Could not load model from {model_dir}. "
            f"Either provide attr.pkl or specify architecture (n_latent, n_hidden, n_layers) + ref_path."
        )

    # Prepare query data
    print(f"  Preparing query data...")
    query = adata.copy()

    # Add required columns
    query.obs["scanvi_label"] = "unlabeled"
    if "batch" not in query.obs.columns:
        query.obs["batch"] = "query"
    if "dataset" not in query.obs.columns:
        query.obs["dataset"] = "query"

    # Check gene name format and convert if needed
    ref_var_names = ref_model.adata.var_names if ref_model.adata is not None else None
    if ref_var_names is not None:
        query_var_names = set(query.var_names)
        ref_var_set = set(ref_var_names)
        overlap = query_var_names & ref_var_set
        print(f"    Query genes: {len(query_var_names):,}")
        print(f"    Reference genes: {len(ref_var_set):,}")
        print(f"    Direct overlap: {len(overlap):,}")

        # If no overlap, check if reference uses Ensembl IDs with feature_name column
        if len(overlap) < 100 and "feature_name" in ref_model.adata.var.columns:
            print(f"    Low overlap - attempting symbol to Ensembl conversion...")
            # Build symbol -> ensembl mapping from reference
            ref_var = ref_model.adata.var
            symbol_to_ensembl = dict(zip(ref_var["feature_name"], ref_var.index))

            # Convert query var_names from symbols to Ensembl
            new_var_names = []
            for gene in query.var_names:
                if gene in symbol_to_ensembl:
                    new_var_names.append(symbol_to_ensembl[gene])
                else:
                    new_var_names.append(gene)  # Keep original if no match

            query.var_names = new_var_names
            query.var_names_make_unique()

            # Recompute overlap
            new_overlap = set(query.var_names) & ref_var_set
            print(f"    After conversion overlap: {len(new_overlap):,}")

    # Align genes with reference
    try:
        SCANVI.prepare_query_anndata(query, ref_model)
        print(f"    Aligned to {query.n_vars:,} genes")
    except Exception as e:
        print(f"  ERROR preparing query: {e}")
        raise

    # Surgery (fine-tune on query)
    print(f"  Running scArches surgery (max {surgery_epochs} epochs)...")
    try:
        query_model = SCANVI.load_query_data(query, ref_model)
        query_model.train(
            max_epochs=surgery_epochs,
            early_stopping=True,
            early_stopping_monitor="elbo_validation",
            early_stopping_patience=20,
            train_size=0.9,
            batch_size=batch_size,
        )
        print(f"    Surgery complete")
    except Exception as e:
        print(f"  ERROR during surgery: {e}")
        raise

    # Get latent representation
    print(f"  Extracting latent representation...")
    latent = query_model.get_latent_representation(query, batch_size=batch_size)
    latent = np.asarray(latent, dtype=np.float32)
    print(f"    Latent shape: {latent.shape}")

    # Get predictions
    print(f"  Predicting cell types...")
    try:
        predictions = query_model.predict(query, batch_size=batch_size)
        if isinstance(predictions, pd.DataFrame):
            labels = predictions.iloc[:, 0].values
        else:
            labels = np.asarray(predictions)
        labels = labels.astype(str)

        # Get confidence
        probs = query_model.predict(query, soft=True, batch_size=batch_size)
        if isinstance(probs, pd.DataFrame):
            probs = probs.values
        probs = np.asarray(probs, dtype=np.float32)
        confidence = probs.max(axis=1)

        n_types = len(np.unique(labels))
        print(f"    Predicted {n_types} cell types")
        print(f"    Mean confidence: {confidence.mean():.3f}")
    except Exception as e:
        print(f"  WARNING: Could not get predictions: {e}")
        labels = None
        confidence = None

    return {
        "latent": latent,
        "labels": labels,
        "confidence": confidence,
    }


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
