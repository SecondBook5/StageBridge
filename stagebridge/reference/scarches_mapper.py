"""scArches mapping for spatial-to-reference alignment.

Maps spatial expression directly through HLCA/LuCA reference models via scArches surgery,
putting spatial spots in the SAME latent space as snRNA cells.

Usage:
    from stagebridge.reference.scarches_mapper import map_spatial_to_reference

    result = map_spatial_to_reference(
        spatial_path="/path/to/spatial_merged.h5ad",
        output_dir="/path/to/output",
        hlca_model_dir="/path/to/hlca/model",
        luca_model_dir="/path/to/luca/model",
    )
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import anndata as ad


def map_spatial_to_reference(
    spatial_path: Path | str,
    output_dir: Path | str,
    hlca_model_dir: Path | str,
    luca_model_dir: Path | str,
    hlca_ref_path: Path | str | None = None,
    luca_ref_path: Path | str | None = None,
    batch_size: int = 1024,
    surgery_epochs: int = 200,
) -> dict[str, Any]:
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

    Returns:
        dict with keys: hlca, luca, fused_path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Mapping spatial spots to reference spaces")
    print("=" * 60)

    # Load spatial data
    print(f"\nLoading spatial data from {spatial_path}...")
    spatial = ad.read_h5ad(spatial_path)
    print(f"  {spatial.n_obs:,} spots, {spatial.n_vars:,} genes")

    # Map to HLCA (or load if already exists)
    hlca_path = output_dir / "spatial_hlca_embedding.parquet"
    if hlca_path.exists():
        print(f"\n{'=' * 60}")
        print("Loading existing HLCA embeddings (30d)")
        print("=" * 60)
        hlca_df = pd.read_parquet(hlca_path)
        hlca_cols = [c for c in hlca_df.columns if c.startswith("hlca_latent_")]
        hlca_result = {
            "latent": hlca_df[hlca_cols].values,
            "labels": hlca_df["cell_type_hlca"].values if "cell_type_hlca" in hlca_df.columns else None,
            "confidence": hlca_df["cell_type_hlca_confidence"].values if "cell_type_hlca_confidence" in hlca_df.columns else None,
        }
        print(f"  Loaded from {hlca_path}")
        print(f"    Shape: {hlca_result['latent'].shape}")
    else:
        print(f"\n{'=' * 60}")
        print("Mapping to HLCA reference (30d)")
        print("=" * 60)
        hlca_result = _map_to_reference(
            spatial,
            Path(hlca_model_dir),
            Path(hlca_ref_path) if hlca_ref_path else None,
            reference_name="HLCA",
            batch_size=batch_size,
            surgery_epochs=surgery_epochs,
        )
        # Save immediately after completion
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
        hlca_df.to_parquet(hlca_path)
        print(f"  Saved HLCA embeddings: {hlca_path}")
        print(f"    Shape: {hlca_result['latent'].shape}")
        # Save training history
        if hlca_result.get("history"):
            with open(output_dir / "hlca_training_history.json", "w") as f:
                json.dump(hlca_result["history"], f, indent=2, default=str)
            print(f"  Saved HLCA training history")

    # Map to LuCA (or load if already exists)
    luca_path = output_dir / "spatial_luca_embedding.parquet"
    if luca_path.exists():
        print(f"\n{'=' * 60}")
        print("Loading existing LuCA embeddings (10d)")
        print("=" * 60)
        luca_df = pd.read_parquet(luca_path)
        luca_cols = [c for c in luca_df.columns if c.startswith("luca_latent_")]
        luca_result = {
            "latent": luca_df[luca_cols].values,
            "labels": luca_df["cell_type_luca"].values if "cell_type_luca" in luca_df.columns else None,
            "confidence": luca_df["cell_type_luca_confidence"].values if "cell_type_luca_confidence" in luca_df.columns else None,
        }
        print(f"  Loaded from {luca_path}")
        print(f"    Shape: {luca_result['latent'].shape}")
    else:
        print(f"\n{'=' * 60}")
        print("Mapping to LuCA reference (10d)")
        print("=" * 60)
        luca_result = _map_to_reference(
            spatial,
            Path(luca_model_dir),
            Path(luca_ref_path) if luca_ref_path else None,
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
        # Save immediately after completion
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
        luca_df.to_parquet(luca_path)
        print(f"  Saved LuCA embeddings: {luca_path}")
        print(f"    Shape: {luca_result['latent'].shape}")
        # Save training history
        if luca_result.get("history"):
            with open(output_dir / "luca_training_history.json", "w") as f:
                json.dump(luca_result["history"], f, indent=2, default=str)
            print(f"  Saved LuCA training history")

    # Fused embeddings (concatenate HLCA + LuCA)
    print(f"\n{'=' * 60}")
    print("Saving fused embeddings")
    print("=" * 60)
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
) -> dict[str, Any]:
    """Map query cells to reference space via scArches surgery."""
    try:
        from scvi.model import SCANVI
    except ImportError:
        raise ImportError("scvi-tools required. Install with: pip install scvi-tools")

    import torch

    model_dir = Path(model_dir)

    print(f"  Loading {reference_name} model from {model_dir}...")

    # Load reference if provided (for gene alignment)
    # Use backed mode to avoid loading full matrix into memory
    ref_adata = None
    if ref_path and Path(ref_path).exists():
        print(f"  Loading reference from {ref_path} (backed mode)...")
        ref_adata = ad.read_h5ad(ref_path, backed='r')
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
        # NOTE: This handles LuCA model saved from retrain_luca_multigpu.py which
        # saves {"model_state_dict": ..., "var_names": [...], "attr_dict": {...}}
        # but NOT the standard attr.pkl file that SCANVI.load() expects.
        # The attr_dict format may be incompatible with newer scvi-tools versions,
        # so we reconstruct the model from scratch and load weights directly.
        print(f"  Attempting manual model reconstruction...")

        # Check for full checkpoint or backup
        model_pt = model_dir / "model.pt"
        model_pt_backup = model_dir / "model_full_checkpoint.pt"

        # Prefer backup if it exists (contains full checkpoint with attr_dict)
        if model_pt_backup.exists():
            checkpoint_path = model_pt_backup
        elif model_pt.exists():
            checkpoint_path = model_pt
        else:
            raise FileNotFoundError(f"No model.pt found at {model_dir}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Handle both formats: dict with model_state_dict or raw state_dict
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            print(f"  Loaded checkpoint with keys: {list(checkpoint.keys())}")
            state_dict = checkpoint["model_state_dict"]
            var_names = checkpoint.get("var_names")
            attr_dict = checkpoint.get("attr_dict")
        else:
            # Raw state_dict - try to load var_names from csv
            state_dict = checkpoint
            var_names = None
            attr_dict = None
            var_names_csv = model_dir / "var_names.csv"
            if var_names_csv.exists():
                var_df = pd.read_csv(var_names_csv, index_col=0)
                # Handle both formats: column named "var_names" or genes as index
                if "var_names" in var_df.columns:
                    var_names = var_df["var_names"].tolist()
                else:
                    var_names = var_df.index.tolist()

        if var_names is None:
            raise ValueError("Could not determine model genes - no var_names in checkpoint or var_names.csv")

        print(f"    Model has {len(var_names)} genes, architecture: n_latent={n_latent}, n_hidden={n_hidden}, n_layers={n_layers}")

        # Remove pyro keys if present
        keys_to_remove = [k for k in state_dict.keys() if k.startswith("pyro_")]
        for k in keys_to_remove:
            del state_dict[k]

        # Prepare reference data with correct genes
        # Only need a small subset for model setup - use 100 cells to save memory
        n_setup_cells = 100
        # Handle backed mode - need to load subset into memory
        if hasattr(ref_adata, 'isbacked') and ref_adata.isbacked:
            print(f"    Loading {n_setup_cells} cells from backed reference...")
            ref_adata_counts = ref_adata[:n_setup_cells].to_memory()
            if ref_adata_counts.raw is not None:
                ref_adata_counts = ref_adata_counts.raw.to_adata()
            elif "counts" in ref_adata_counts.layers:
                ref_adata_counts.X = ref_adata_counts.layers["counts"]
        elif ref_adata.raw is not None:
            ref_subset = ref_adata[:n_setup_cells].copy()
            ref_adata_counts = ref_subset.raw.to_adata()
        elif "counts" in ref_adata.layers:
            ref_adata_counts = ref_adata[:n_setup_cells].copy()
            ref_adata_counts.X = ref_adata_counts.layers["counts"]
        else:
            ref_adata_counts = ref_adata[:n_setup_cells].copy()
        print(f"    Using {n_setup_cells} cells for model setup")

        # Subset to model genes
        model_genes_set = set(var_names)
        ref_genes_set = set(ref_adata_counts.var_names)
        common_genes = [g for g in var_names if g in ref_genes_set]
        print(f"    Model genes: {len(var_names)}, Reference genes: {len(ref_genes_set)}, Overlap: {len(common_genes)}")

        if len(common_genes) < len(var_names) * 0.9:
            raise ValueError(f"Insufficient gene overlap: {len(common_genes)}/{len(var_names)}")

        # Subset and reorder to match model's gene order
        ref_adata_counts = ref_adata_counts[:, common_genes].copy()
        print(f"    Reference subset to {ref_adata_counts.n_vars} genes")

        # Get batch categories from attr_dict - REQUIRED for correct architecture
        batch_categories = None
        label_categories = None
        if attr_dict and "registry_" in attr_dict:
            registry = attr_dict["registry_"]
            if "field_registries" in registry:
                field_reg = registry["field_registries"]
                if "batch" in field_reg and "state_registry" in field_reg["batch"]:
                    batch_state = field_reg["batch"]["state_registry"]
                    if "categorical_mapping" in batch_state:
                        batch_categories = list(batch_state["categorical_mapping"])
                        print(f"    Found {len(batch_categories)} batch categories in attr_dict")
                if "labels" in field_reg and "state_registry" in field_reg["labels"]:
                    label_state = field_reg["labels"]["state_registry"]
                    if "categorical_mapping" in label_state:
                        label_categories = list(label_state["categorical_mapping"])
                        print(f"    Found {len(label_categories)} label categories in attr_dict")

        # Trust the weights, not attr_dict - get actual sizes from state_dict
        # y_prior shape is [1, n_labels], classifier output is [n_labels, hidden]
        # NOTE: n_labels includes the unlabeled category, so known labels = n_labels - 1
        if "y_prior" in state_dict:
            n_labels_from_weights = state_dict["y_prior"].shape[1]
            n_known_labels = n_labels_from_weights - 1  # Subtract 1 for unlabeled category
            print(f"    Actual n_labels from weights: {n_labels_from_weights} (= {n_known_labels} known + 1 unlabeled)")
            if label_categories:
                # Remove "Unknown" or unlabeled if present, then trim to n_known_labels
                label_categories = [l for l in label_categories if l.lower() not in ("unknown", "unlabeled")]
                if len(label_categories) != n_known_labels:
                    print(f"    WARNING: attr_dict has {len(label_categories)} known labels but need {n_known_labels}")
                    if len(label_categories) > n_known_labels:
                        label_categories = label_categories[:n_known_labels]
                    else:
                        for i in range(n_known_labels - len(label_categories)):
                            label_categories.append(f"label_{len(label_categories) + i}")
                print(f"    Using {len(label_categories)} known label categories (+ Unknown = {len(label_categories) + 1})")

        if batch_categories is None:
            raise ValueError("Could not extract batch categories from checkpoint - needed for architecture match")

        # Set up batch column with ALL original categories to match architecture
        # Cycle through all categories so scvi-tools sees them all as "observed"
        batch_values = [batch_categories[i % len(batch_categories)] for i in range(ref_adata_counts.n_obs)]
        ref_adata_counts.obs[batch_key] = pd.Categorical(batch_values, categories=batch_categories)
        print(f"    Set batch column with {len(batch_categories)} categories (cycling)")

        # Set up labels column - use first known label for all cells
        # SCANVI.from_scvi_model will add "Unknown" as the unlabeled category
        if label_categories is None:
            raise ValueError("Could not extract label categories from checkpoint - needed for architecture match")
        ref_adata_counts.obs[labels_key] = pd.Categorical(
            [label_categories[0]] * ref_adata_counts.n_obs,
            categories=label_categories
        )
        print(f"    Set labels column with {len(label_categories)} known categories")

        # Setup SCVI first (base for SCANVI)
        from scvi.model import SCVI
        print(f"    Setting up fresh SCVI model with {len(batch_categories)} batches...")
        SCVI.setup_anndata(ref_adata_counts, batch_key=batch_key)

        # Architecture from checkpoint analysis:
        # - Input size 6021 = 6000 genes + 21 batches -> encode_covariates=True
        # - Layer 0.0 only (no 0.1 BatchNorm) -> use_batch_norm=False
        # - Hidden size 149 = 128 + 21 batches -> deeply_inject_covariates=True
        encode_covariates = True
        use_batch_norm = False  # Checkpoint has no BatchNorm keys
        deeply_inject_covariates = True  # Hidden layers also get batch concat

        print(f"    Architecture: encode_covariates={encode_covariates}, use_batch_norm={use_batch_norm}, deeply_inject={deeply_inject_covariates}")

        scvi_model = SCVI(
            ref_adata_counts,
            n_latent=n_latent,
            n_hidden=n_hidden,
            n_layers=n_layers,
            encode_covariates=encode_covariates,
            use_batch_norm="none" if not use_batch_norm else "both",
            deeply_inject_covariates=deeply_inject_covariates,
        )

        # Convert SCVI to SCANVI
        print(f"    Converting to SCANVI...")
        ref_model = SCANVI.from_scvi_model(
            scvi_model,
            unlabeled_category="Unknown",
            labels_key=labels_key,
        )

        # Load weights
        print(f"    Loading weights from checkpoint...")
        try:
            # Try strict load first
            ref_model.module.load_state_dict(state_dict, strict=True)
            print(f"    Weights loaded successfully (strict)")
        except RuntimeError as e:
            print(f"    Strict load failed: {e}")
            print(f"    Trying non-strict load...")
            # Non-strict load - may have missing/extra keys
            missing, unexpected = ref_model.module.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"    Missing keys: {missing[:5]}..." if len(missing) > 5 else f"    Missing keys: {missing}")
            if unexpected:
                print(f"    Unexpected keys: {unexpected[:5]}..." if len(unexpected) > 5 else f"    Unexpected keys: {unexpected}")

        print(f"  Model reconstructed successfully")
        print(f"    Latent dim: {ref_model.module.n_latent}")

    if ref_model is None:
        raise RuntimeError(
            f"Could not load model from {model_dir}. "
            f"Either provide attr.pkl or specify architecture (n_latent, n_hidden, n_layers) + ref_path."
        )

    # Prepare query data
    print(f"  Preparing query data...")
    query = adata.copy()

    # Add required columns - must use same batch_key as reference model
    query.obs["scanvi_label"] = "unlabeled"

    # Get batch categories from reference model's registry
    ref_batch_key = None
    ref_batch_cats = None
    if hasattr(ref_model, 'adata_manager') and ref_model.adata_manager is not None:
        try:
            registry = ref_model.adata_manager.registry
            if "field_registries" in registry:
                field_reg = registry["field_registries"]
                if "batch" in field_reg:
                    ref_batch_key = field_reg["batch"].get("attr_key", batch_key)
                    if "state_registry" in field_reg["batch"]:
                        cats = field_reg["batch"]["state_registry"].get("categorical_mapping")
                        if cats is not None:
                            ref_batch_cats = list(cats)
        except Exception:
            pass

    # Fall back to passed batch_key
    if ref_batch_key is None:
        ref_batch_key = batch_key

    # Set query batch column - scArches expects "query" as a new batch category
    # but all original categories must be present in the Categorical
    if ref_batch_cats is not None:
        all_batch_cats = ref_batch_cats + ["query"]
        query.obs[ref_batch_key] = pd.Categorical(
            ["query"] * query.n_obs,
            categories=all_batch_cats
        )
        print(f"    Set {ref_batch_key} with {len(all_batch_cats)} categories (including 'query')")
    else:
        query.obs[ref_batch_key] = "query"
        print(f"    Set {ref_batch_key} = 'query'")

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

    # Surgery (fine-tune on query) - or direct inference if surgery fails
    print(f"  Running scArches surgery (max {surgery_epochs} epochs)...")
    query_model = None
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
        print(f"  WARNING: Surgery failed: {e}")
        print(f"  Falling back to direct inference (no fine-tuning)...")
        query_model = None

    # Get latent representation
    print(f"  Extracting latent representation...")
    model_for_inference = query_model if query_model is not None else ref_model
    latent = model_for_inference.get_latent_representation(query, batch_size=batch_size)
    latent = np.asarray(latent, dtype=np.float32)
    print(f"    Latent shape: {latent.shape}")

    # Get predictions
    print(f"  Predicting cell types...")
    try:
        predictions = model_for_inference.predict(query, batch_size=batch_size)
        if isinstance(predictions, pd.DataFrame):
            labels = predictions.iloc[:, 0].values
        else:
            labels = np.asarray(predictions)
        labels = labels.astype(str)

        # Get confidence
        probs = model_for_inference.predict(query, soft=True, batch_size=batch_size)
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

    # Extract training history for plotting
    history = None
    if hasattr(query_model, 'history') and query_model.history is not None:
        history = {k: list(v) for k, v in query_model.history.items()}

    return {
        "latent": latent,
        "labels": labels,
        "confidence": confidence,
        "history": history,
    }
