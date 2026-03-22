"""Reference preparation and harmonization for query-to-reference mapping.

This module handles feature space alignment between query and reference data,
including gene symbol harmonization and expression matrix subsetting.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

from stagebridge.logging_utils import get_logger

log = get_logger(__name__)


# Regex for ENSEMBL gene IDs with version suffix
_ENSG_RE = re.compile(r"^(ENSG\d+)(?:\.\d+)?$")


def _strip_ensembl_version(gene_id: str) -> str | None:
    """Strip version suffix from ENSEMBL ID (ENSG00001234.5 -> ENSG00001234)."""
    match = _ENSG_RE.match(str(gene_id).strip())
    if match:
        return match.group(1)
    return None


def _normalize_gene_symbol(symbol: str) -> str:
    """Normalize gene symbol for matching (uppercase, strip whitespace)."""
    return str(symbol).strip().upper()


def align_gene_symbols(
    query_genes: pd.Index | np.ndarray,
    target_genes: pd.Index | np.ndarray,
    *,
    target_symbol_col: pd.Series | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Align query gene names to target gene namespace.

    Supports three matching strategies in order:
    1. Direct match (exact string match)
    2. ENSEMBL ID match (strip version suffixes)
    3. Symbol-based match (if target_symbol_col provided)

    Parameters
    ----------
    query_genes : pd.Index or np.ndarray
        Query gene names/IDs
    target_genes : pd.Index or np.ndarray
        Target (reference) gene names/IDs
    target_symbol_col : pd.Series, optional
        Gene symbols for target genes (e.g., reference.var['feature_name'])

    Returns
    -------
    tuple[np.ndarray, dict]
        - mapping array: for each target gene, index into query (-1 if missing)
        - alignment report dictionary
    """
    query_arr = np.asarray(query_genes).astype(str)
    target_arr = np.asarray(target_genes).astype(str)

    n_query = len(query_arr)
    n_target = len(target_arr)

    # Build query lookup tables
    query_direct = {g: i for i, g in enumerate(query_arr)}

    # ENSEMBL lookup (strip versions)
    query_ensg = {}
    for i, g in enumerate(query_arr):
        ensg = _strip_ensembl_version(g)
        if ensg and ensg not in query_ensg:
            query_ensg[ensg] = i

    # Symbol lookup (normalized)
    query_symbol = {}
    for i, g in enumerate(query_arr):
        norm = _normalize_gene_symbol(g)
        if norm and norm not in query_symbol:
            query_symbol[norm] = i

    # Build target ENSEMBL lookup
    target_ensg = np.array([_strip_ensembl_version(g) for g in target_arr], dtype=object)

    # Build target symbol lookup if provided
    target_symbols = None
    if target_symbol_col is not None:
        target_symbols = np.array(
            [_normalize_gene_symbol(s) for s in target_symbol_col], dtype=object
        )

    # Map each target gene to query index
    mapping = np.full(n_target, -1, dtype=np.int64)
    match_method = np.full(n_target, "", dtype=object)

    for i, tgt in enumerate(target_arr):
        # Strategy 1: Direct match
        if tgt in query_direct:
            mapping[i] = query_direct[tgt]
            match_method[i] = "direct"
            continue

        # Strategy 2: ENSEMBL match
        if target_ensg[i] is not None and target_ensg[i] in query_ensg:
            mapping[i] = query_ensg[target_ensg[i]]
            match_method[i] = "ensembl"
            continue

        # Strategy 3: Symbol match
        if target_symbols is not None and target_symbols[i]:
            sym = target_symbols[i]
            if sym in query_symbol:
                mapping[i] = query_symbol[sym]
                match_method[i] = "symbol"
                continue

    # Compute statistics
    n_matched = int((mapping >= 0).sum())
    n_direct = int((match_method == "direct").sum())
    n_ensg = int((match_method == "ensembl").sum())
    n_symbol = int((match_method == "symbol").sum())

    report = {
        "query_gene_count": n_query,
        "target_gene_count": n_target,
        "matched_count": n_matched,
        "match_fraction": n_matched / max(n_target, 1),
        "direct_matches": n_direct,
        "ensembl_matches": n_ensg,
        "symbol_matches": n_symbol,
        "unmatched_count": n_target - n_matched,
    }

    log.info(
        "Gene alignment: %d/%d matched (%.1f%%) - direct=%d, ensembl=%d, symbol=%d",
        n_matched,
        n_target,
        report["match_fraction"] * 100,
        n_direct,
        n_ensg,
        n_symbol,
    )

    return mapping, report


def prepare_reference_for_mapping(
    reference: Any,
    query: Any,
    *,
    reference_symbol_col: str | None = "feature_name",
) -> tuple[Any, dict[str, Any]]:
    """Prepare reference data for query mapping by aligning feature spaces.

    This creates a reference AnnData subset that is compatible with the query
    feature space. Missing genes are filled with zeros.

    Parameters
    ----------
    reference : AnnData or LoadedReference
        Reference atlas data
    query : AnnData
        Query data
    reference_symbol_col : str, optional
        Column in reference.var containing gene symbols

    Returns
    -------
    tuple[AnnData, dict]
        - Harmonized reference AnnData (same gene order as reference)
        - Preparation report
    """

    # Handle LoadedReference wrapper
    if hasattr(reference, "adata"):
        reference = reference.adata

    # Get symbol column if available
    symbol_col = None
    if reference_symbol_col and reference_symbol_col in reference.var.columns:
        symbol_col = reference.var[reference_symbol_col]

    # Align genes
    mapping, align_report = align_gene_symbols(
        query.var_names,
        reference.var_names,
        target_symbol_col=symbol_col,
    )

    report = {
        "alignment": align_report,
        "reference_genes": reference.n_vars,
        "query_genes": query.n_vars,
        "genes_with_data": int((mapping >= 0).sum()),
    }

    return reference, report


def subset_query_to_reference_genes(
    query: Any,
    reference: Any,
    *,
    reference_symbol_col: str | None = "feature_name",
    fill_missing: bool = True,
) -> tuple[Any, np.ndarray, dict[str, Any]]:
    """Subset and reorder query data to match reference gene space.

    Parameters
    ----------
    query : AnnData
        Query data to subset
    reference : AnnData or LoadedReference
        Reference providing target gene space
    reference_symbol_col : str, optional
        Column in reference.var containing gene symbols
    fill_missing : bool
        If True, fill missing genes with zeros. If False, only include matched genes.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, dict]
        - Expression matrix aligned to reference genes (n_cells x n_ref_genes)
        - Mask of which reference genes have data
        - Preparation report
    """
    # Handle LoadedReference wrapper
    if hasattr(reference, "adata"):
        reference = reference.adata

    # Get symbol column if available
    symbol_col = None
    if reference_symbol_col and reference_symbol_col in reference.var.columns:
        symbol_col = reference.var[reference_symbol_col]

    # Align genes: for each ref gene, get index in query
    mapping, align_report = align_gene_symbols(
        query.var_names,
        reference.var_names,
        target_symbol_col=symbol_col,
    )

    n_cells = query.n_obs
    n_ref_genes = reference.n_vars

    # Get query expression matrix
    X_query = query.X
    if sp.issparse(X_query):
        X_query = X_query.toarray()
    X_query = np.asarray(X_query, dtype=np.float32)

    if fill_missing:
        # Create full matrix with zeros for missing genes
        X_aligned = np.zeros((n_cells, n_ref_genes), dtype=np.float32)
        mask = mapping >= 0
        X_aligned[:, mask] = X_query[:, mapping[mask]]
    else:
        # Only include matched genes
        mask = mapping >= 0
        X_aligned = X_query[:, mapping[mask]]

    report = {
        "alignment": align_report,
        "output_shape": list(X_aligned.shape),
        "genes_with_data": int(mask.sum()),
        "genes_missing": int((~mask).sum()),
        "fill_missing": fill_missing,
    }

    return X_aligned, mask, report


def harmonize_metadata(
    query: Any,
    *,
    cell_id_col: str | None = None,
    donor_col: str = "donor_id",
    sample_col: str = "sample_id",
    stage_col: str = "stage",
) -> pd.DataFrame:
    """Harmonize query metadata to standard schema.

    Parameters
    ----------
    query : AnnData
        Query data
    cell_id_col : str, optional
        Column containing cell IDs. If None, uses index.
    donor_col : str
        Column containing donor IDs
    sample_col : str
        Column containing sample IDs
    stage_col : str
        Column containing stage labels

    Returns
    -------
    pd.DataFrame
        Harmonized metadata with standard columns
    """
    obs = query.obs.copy()

    # Cell ID
    if cell_id_col and cell_id_col in obs.columns:
        cell_ids = obs[cell_id_col].astype(str)
    else:
        cell_ids = obs.index.astype(str)

    result = pd.DataFrame({"cell_id": cell_ids})
    result.index = obs.index

    # Donor ID
    if donor_col in obs.columns:
        result["donor_id"] = obs[donor_col].astype(str)
    else:
        result["donor_id"] = "unknown_donor"

    # Sample ID
    if sample_col in obs.columns:
        result["sample_id"] = obs[sample_col].astype(str)
    else:
        result["sample_id"] = "unknown_sample"

    # Stage
    if stage_col in obs.columns:
        result["stage_id"] = obs[stage_col].astype(str)
    else:
        result["stage_id"] = "unknown_stage"

    return result
