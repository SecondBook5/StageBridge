"""Reference mapping via scArches surgery.

Maps query cells into HLCA (30d) and LuCA (10d) latent spaces using
scArches query-to-reference alignment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from stagebridge.contracts import HLCA_DIM, LUCA_DIM


@dataclass(slots=True, frozen=True)
class MappingResult:
    """Result from reference mapping.

    Attributes:
        latent: [N, D] latent embeddings
        labels: [N] predicted cell type labels
        confidence: [N] prediction confidence (max probability)
        entropy: [N] prediction entropy
        cell_ids: Cell identifiers
        reference: Which reference was used
    """

    latent: np.ndarray
    labels: np.ndarray
    confidence: np.ndarray | None
    entropy: np.ndarray | None
    cell_ids: pd.Index
    reference: Literal["hlca", "luca"]

    @property
    def n_cells(self) -> int:
        return len(self.latent)

    @property
    def latent_dim(self) -> int:
        return self.latent.shape[1]


class ReferenceMapper:
    """Maps query cells to HLCA and LuCA reference spaces.

    Uses scArches surgery to align query cells with reference atlases.
    Supports k-NN fallback when surgery fails or for speed.

    Args:
        hlca_model_dir: Path to HLCA scANVI model
        luca_model_dir: Path to LuCA scANVI model
        hlca_ref_path: Path to HLCA reference h5ad (for model loading)
        luca_ref_path: Path to LuCA reference h5ad (for model loading)
        surgery_epochs: Max epochs for scArches surgery
        batch_size: Inference batch size
        use_knn_fallback: Use k-NN if surgery fails
    """

    def __init__(
        self,
        hlca_model_dir: str | Path | None = None,
        luca_model_dir: str | Path | None = None,
        hlca_ref_path: str | Path | None = None,
        luca_ref_path: str | Path | None = None,
        surgery_epochs: int = 200,
        batch_size: int = 1024,
        use_knn_fallback: bool = True,
    ):
        self.hlca_model_dir = Path(hlca_model_dir) if hlca_model_dir else None
        self.luca_model_dir = Path(luca_model_dir) if luca_model_dir else None
        self.hlca_ref_path = Path(hlca_ref_path) if hlca_ref_path else None
        self.luca_ref_path = Path(luca_ref_path) if luca_ref_path else None
        self.surgery_epochs = surgery_epochs
        self.batch_size = batch_size
        self.use_knn_fallback = use_knn_fallback

        self._hlca_model = None
        self._luca_model = None

    def map_to_hlca(
        self,
        adata: Any,
        *,
        return_labels: bool = True,
        return_probs: bool = True,
    ) -> MappingResult:
        """Map query cells to HLCA latent space.

        Args:
            adata: AnnData with query cells
            return_labels: Return predicted cell type labels
            return_probs: Return prediction probabilities

        Returns:
            MappingResult with HLCA embeddings
        """
        return self._map_to_reference(
            adata,
            reference="hlca",
            return_labels=return_labels,
            return_probs=return_probs,
        )

    def map_to_luca(
        self,
        adata: Any,
        *,
        return_labels: bool = True,
        return_probs: bool = True,
    ) -> MappingResult:
        """Map query cells to LuCA latent space.

        Args:
            adata: AnnData with query cells
            return_labels: Return predicted cell type labels
            return_probs: Return prediction probabilities

        Returns:
            MappingResult with LuCA embeddings
        """
        return self._map_to_reference(
            adata,
            reference="luca",
            return_labels=return_labels,
            return_probs=return_probs,
        )

    def _map_to_reference(
        self,
        adata: Any,
        reference: Literal["hlca", "luca"],
        return_labels: bool = True,
        return_probs: bool = True,
    ) -> MappingResult:
        """Internal mapping implementation."""
        try:
            from scvi.model import SCANVI
        except ImportError as e:
            raise ImportError(
                "scvi-tools required for reference mapping. "
                "Install with: pip install scvi-tools"
            ) from e

        import anndata

        if reference == "hlca":
            model_dir = self.hlca_model_dir
            ref_path = self.hlca_ref_path
            expected_dim = HLCA_DIM
        else:
            model_dir = self.luca_model_dir
            ref_path = self.luca_ref_path
            expected_dim = LUCA_DIM

        if model_dir is None or not model_dir.exists():
            return self._knn_fallback(adata, reference, expected_dim)

        ref_adata = anndata.read_h5ad(ref_path) if ref_path else None

        try:
            ref_model = SCANVI.load(str(model_dir), adata=ref_adata)
        except Exception:
            if self.use_knn_fallback:
                return self._knn_fallback(adata, reference, expected_dim)
            raise

        query_copy = adata.copy()
        query_copy.obs["scanvi_label"] = "unlabeled"
        if "dataset" not in query_copy.obs.columns:
            query_copy.obs["dataset"] = "query"

        try:
            SCANVI.prepare_query_anndata(query_copy, ref_model)
            query_model = SCANVI.load_query_data(query_copy, ref_model)

            query_model.train(
                max_epochs=self.surgery_epochs,
                early_stopping=True,
                early_stopping_monitor="elbo_validation",
                early_stopping_patience=15,
                train_size=0.9,
                enable_progress_bar=False,
            )

            latent = query_model.get_latent_representation(
                query_copy, batch_size=self.batch_size
            )
            latent = np.asarray(latent, dtype=np.float32)

            labels = None
            confidence = None
            entropy = None

            if return_labels:
                pred = query_model.predict(query_copy, batch_size=self.batch_size)
                if isinstance(pred, tuple):
                    pred = pred[0]
                if isinstance(pred, pd.DataFrame):
                    pred = pred.iloc[:, 0].to_numpy()
                labels = np.asarray(pred, dtype=object).astype(str)

            if return_probs:
                probs = query_model.predict(
                    query_copy, soft=True, batch_size=self.batch_size
                )
                if isinstance(probs, tuple):
                    probs = probs[0]
                if isinstance(probs, pd.DataFrame):
                    probs = probs.to_numpy(dtype=np.float32)
                probs = np.asarray(probs, dtype=np.float32)
                confidence = probs.max(axis=1)
                entropy = -(probs * np.log(probs + 1e-12)).sum(axis=1)

        except Exception:
            if self.use_knn_fallback:
                return self._knn_fallback(adata, reference, expected_dim)
            raise

        return MappingResult(
            latent=latent,
            labels=labels,
            confidence=confidence,
            entropy=entropy,
            cell_ids=adata.obs.index.copy(),
            reference=reference,
        )

    def _knn_fallback(
        self,
        adata: Any,
        reference: Literal["hlca", "luca"],
        expected_dim: int,
    ) -> MappingResult:
        """k-NN fallback when surgery is unavailable.

        Projects query cells using PCA then aligns to reference scale.
        """
        from sklearn.decomposition import PCA

        if hasattr(adata, "X"):
            X = adata.X
            if hasattr(X, "toarray"):
                X = X.toarray()
            X = np.asarray(X, dtype=np.float32)
        else:
            raise ValueError("adata must have X attribute")

        X_log = np.log1p(X)

        pca = PCA(n_components=min(expected_dim, X_log.shape[1], X_log.shape[0]))
        latent = pca.fit_transform(X_log).astype(np.float32)

        if latent.shape[1] < expected_dim:
            pad = np.zeros((latent.shape[0], expected_dim - latent.shape[1]), dtype=np.float32)
            latent = np.hstack([latent, pad])

        latent = (latent - latent.mean(axis=0)) / (latent.std(axis=0) + 1e-6)

        return MappingResult(
            latent=latent,
            labels=None,
            confidence=None,
            entropy=None,
            cell_ids=adata.obs.index.copy(),
            reference=reference,
        )


def map_to_hlca(
    adata: Any,
    model_dir: str | Path,
    ref_path: str | Path | None = None,
    **kwargs: Any,
) -> MappingResult:
    """Convenience function for HLCA mapping."""
    mapper = ReferenceMapper(
        hlca_model_dir=model_dir,
        hlca_ref_path=ref_path,
        **kwargs,
    )
    return mapper.map_to_hlca(adata)


def map_to_luca(
    adata: Any,
    model_dir: str | Path,
    ref_path: str | Path | None = None,
    **kwargs: Any,
) -> MappingResult:
    """Convenience function for LuCA mapping."""
    mapper = ReferenceMapper(
        luca_model_dir=model_dir,
        luca_ref_path=ref_path,
        **kwargs,
    )
    return mapper.map_to_luca(adata)


# =============================================================================
# Default paths (can be overridden via environment or config)
# =============================================================================
import os

DEFAULT_PATHS = {
    "hlca_model": os.environ.get(
        "STAGEBRIDGE_HLCA_MODEL",
        "/data1/chaunzt1/stagebridge/references/hlca/model"
    ),
    "hlca_ref": os.environ.get(
        "STAGEBRIDGE_HLCA_REF",
        "/data1/chaunzt1/stagebridge/references/hlca/hlca_reference.h5ad"
    ),
    "luca_model": os.environ.get(
        "STAGEBRIDGE_LUCA_MODEL",
        "/scratch/chaunzt1/stagebridge/references/luca/retrained_model/scanvi_model"
    ),
    "luca_ref": os.environ.get(
        "STAGEBRIDGE_LUCA_REF",
        "/scratch/chaunzt1/stagebridge/references/luca/luca_core_atlas.h5ad"
    ),
}


def transfer_labels(
    adata: Any,
    reference: Literal["hlca", "luca"] = "luca",
    *,
    model_dir: str | Path | None = None,
    ref_path: str | Path | None = None,
    label_col: str | None = None,
    confidence_col: str | None = None,
    inplace: bool = True,
    use_knn: bool = False,
    knn_k: int = 15,
    **kwargs: Any,
) -> Any:
    """Transfer cell type labels from reference atlas to query data.

    High-level API for label transfer. Adds cell type labels to adata.obs.

    Args:
        adata: AnnData with query cells
        reference: Which reference to use ("hlca" or "luca")
        model_dir: Path to scANVI model (uses default if None)
        ref_path: Path to reference h5ad (uses default if None)
        label_col: Column name for transferred labels (default: "cell_type_{reference}")
        confidence_col: Column name for confidence (default: "cell_type_{reference}_confidence")
        inplace: Modify adata in place (if False, returns copy)
        use_knn: Use k-NN label transfer instead of scArches surgery (faster)
        knn_k: Number of neighbors for k-NN transfer
        **kwargs: Additional arguments to ReferenceMapper

    Returns:
        AnnData with transferred labels in .obs

    Example:
        >>> import stagebridge as sb
        >>> adata = sb.reference.transfer_labels(adata, reference="luca")
        >>> adata.obs["cell_type_luca"].value_counts()
    """
    if not inplace:
        adata = adata.copy()

    # Set default column names
    if label_col is None:
        label_col = f"cell_type_{reference}"
    if confidence_col is None:
        confidence_col = f"cell_type_{reference}_confidence"

    # Use default paths if not provided
    if model_dir is None:
        model_dir = DEFAULT_PATHS[f"{reference}_model"]
    if ref_path is None:
        ref_path = DEFAULT_PATHS[f"{reference}_ref"]

    model_dir = Path(model_dir)
    ref_path = Path(ref_path) if ref_path else None

    # Check paths exist
    if not model_dir.exists():
        raise FileNotFoundError(
            f"{reference.upper()} model not found at {model_dir}. "
            f"Set STAGEBRIDGE_{reference.upper()}_MODEL environment variable "
            f"or pass model_dir explicitly."
        )

    print(f"Transferring labels from {reference.upper()} reference...")
    print(f"  Model: {model_dir}")
    print(f"  Reference: {ref_path}")
    print(f"  Query: {adata.n_obs:,} cells")

    if use_knn:
        # Fast k-NN label transfer
        result = _knn_label_transfer(
            adata, reference, model_dir, ref_path, knn_k
        )
    else:
        # Full scArches surgery
        mapper = ReferenceMapper(
            hlca_model_dir=model_dir if reference == "hlca" else None,
            luca_model_dir=model_dir if reference == "luca" else None,
            hlca_ref_path=ref_path if reference == "hlca" else None,
            luca_ref_path=ref_path if reference == "luca" else None,
            **kwargs,
        )

        if reference == "hlca":
            result = mapper.map_to_hlca(adata, return_labels=True, return_probs=True)
        else:
            result = mapper.map_to_luca(adata, return_labels=True, return_probs=True)

    # Add labels to adata
    adata.obs[label_col] = result.labels
    if result.confidence is not None:
        adata.obs[confidence_col] = result.confidence
    if result.entropy is not None:
        adata.obs[f"{label_col}_entropy"] = result.entropy

    # Add latent embedding
    latent_key = f"X_{reference.upper()}"
    adata.obsm[latent_key] = result.latent

    # Summary
    n_types = pd.Series(result.labels).nunique()
    print(f"  Transferred {n_types} cell types")
    print(f"  Labels stored in: adata.obs['{label_col}']")
    print(f"  Embeddings stored in: adata.obsm['{latent_key}']")

    return adata


def _knn_label_transfer(
    adata: Any,
    reference: Literal["hlca", "luca"],
    model_dir: Path,
    ref_path: Path | None,
    k: int = 15,
) -> MappingResult:
    """Fast k-NN label transfer from reference atlas.

    Loads reference latent + labels, finds k nearest neighbors for each
    query cell, transfers labels by majority vote.
    """
    import anndata as ad
    from sklearn.neighbors import NearestNeighbors

    # Load reference
    if ref_path is None or not ref_path.exists():
        raise FileNotFoundError(f"Reference h5ad required for k-NN: {ref_path}")

    print(f"  Loading reference atlas...")
    ref_adata = ad.read_h5ad(ref_path)

    # Get reference latent and labels
    latent_key = "X_scANVI" if "X_scANVI" in ref_adata.obsm else "X_scVI"
    if latent_key not in ref_adata.obsm:
        raise ValueError(f"Reference has no {latent_key} in obsm")

    ref_latent = ref_adata.obsm[latent_key]
    ref_labels = ref_adata.obs["cell_type"].values

    print(f"  Reference: {ref_latent.shape[0]:,} cells, {ref_latent.shape[1]}d latent")

    # Get query latent (need to compute via PCA if not available)
    query_latent_key = f"X_{reference.upper()}"
    if query_latent_key in adata.obsm:
        query_latent = adata.obsm[query_latent_key]
    else:
        # Project query into reference space via PCA
        from sklearn.decomposition import PCA
        print(f"  Computing query embedding via PCA...")

        X = adata.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.log1p(X)

        pca = PCA(n_components=ref_latent.shape[1])
        pca.fit(ref_latent)  # Fit on reference
        query_latent = pca.transform(X)

    # k-NN
    print(f"  Finding {k} nearest neighbors...")
    knn = NearestNeighbors(n_neighbors=k, metric="euclidean", n_jobs=-1)
    knn.fit(ref_latent)
    distances, indices = knn.kneighbors(query_latent)

    # Majority vote
    print(f"  Transferring labels by majority vote...")
    transferred_labels = []
    for i in range(len(query_latent)):
        neighbor_labels = ref_labels[indices[i]]
        unique, counts = np.unique(neighbor_labels, return_counts=True)
        transferred_labels.append(unique[counts.argmax()])

    # Confidence = fraction of neighbors with majority label
    confidence = np.array([
        np.sum(ref_labels[indices[i]] == transferred_labels[i]) / k
        for i in range(len(query_latent))
    ])

    return MappingResult(
        latent=query_latent.astype(np.float32),
        labels=np.array(transferred_labels),
        confidence=confidence.astype(np.float32),
        entropy=None,
        cell_ids=adata.obs_names.copy(),
        reference=reference,
    )
