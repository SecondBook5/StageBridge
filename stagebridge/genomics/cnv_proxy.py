"""CNV proxy estimation from expression data.

Provides clone-state proxy estimates from expression-derived CNV signals.
Prefers existing inferCNV/CopyKAT outputs when available.

IMPORTANT: Expression-derived CNV is approximate and should be labeled
as "clone_state_proxy", not definitive CNV calls.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_infercnv_output(path: str | Path) -> pd.DataFrame:
    """Load inferCNV results.

    Args:
        path: Path to inferCNV output directory or file

    Returns:
        DataFrame with cell-level CNV scores
    """
    path = Path(path)

    if path.is_dir():
        expr_path = path / "infercnv.observations.txt"
        if not expr_path.exists():
            expr_path = path / "infercnv.14_HMM_predHMMi6.leiden.hmm_mode-subclusters.repr_intensities.observations.txt"
    else:
        expr_path = path

    if not expr_path.exists():
        raise FileNotFoundError(f"inferCNV output not found: {expr_path}")

    df = pd.read_csv(expr_path, sep="\t", index_col=0)

    cnv_scores = pd.DataFrame({
        "barcode": df.index,
        "cnv_score": df.var(axis=1).values,
        "cnv_mean_deviation": (df - 1).abs().mean(axis=1).values,
        "method": "inferCNV",
    })

    return cnv_scores


def load_copykat_output(path: str | Path) -> pd.DataFrame:
    """Load CopyKAT results.

    Args:
        path: Path to CopyKAT prediction file

    Returns:
        DataFrame with cell-level CNV predictions
    """
    path = Path(path)

    if path.is_dir():
        pred_path = path / "copykat_prediction.txt"
    else:
        pred_path = path

    if not pred_path.exists():
        raise FileNotFoundError(f"CopyKAT output not found: {pred_path}")

    df = pd.read_csv(pred_path, sep="\t")

    barcode_col = None
    for col in ["cell.names", "cell_names", "barcode", "Cell"]:
        if col in df.columns:
            barcode_col = col
            break

    if barcode_col is None:
        raise ValueError("Could not find barcode column in CopyKAT output")

    pred_col = None
    for col in ["copykat.pred", "copykat_pred", "prediction"]:
        if col in df.columns:
            pred_col = col
            break

    result = pd.DataFrame({
        "barcode": df[barcode_col],
        "copykat_prediction": df[pred_col] if pred_col else "unknown",
        "method": "CopyKAT",
    })

    if pred_col:
        result["is_aneuploid"] = result["copykat_prediction"].str.lower() == "aneuploid"
    else:
        result["is_aneuploid"] = False

    return result


def infer_cnv_proxy_from_expression(
    adata,
    reference_cell_types: list[str] | None = None,
    window_size: int = 100,
    step_size: int = 10,
) -> pd.DataFrame:
    """Infer CNV proxy scores from expression data.

    This is a simplified approach that computes expression deviation
    in genomic windows as a proxy for CNV. For rigorous analysis,
    use inferCNV or CopyKAT.

    Args:
        adata: AnnData object with expression data
        reference_cell_types: Cell types to use as diploid reference
        window_size: Number of genes per window
        step_size: Step size for sliding window

    Returns:
        DataFrame with CNV proxy scores per cell
    """
    import anndata

    if not isinstance(adata, anndata.AnnData):
        raise TypeError("adata must be an AnnData object")

    if reference_cell_types is None:
        reference_cell_types = ["Endothelial", "Fibroblast", "T_cell", "B_cell"]

    if "cell_type" in adata.obs.columns:
        ref_mask = adata.obs["cell_type"].isin(reference_cell_types)
        if ref_mask.sum() < 10:
            logger.warning(
                f"Few reference cells found ({ref_mask.sum()}). "
                "CNV proxy may be unreliable."
            )
            ref_mask = pd.Series(True, index=adata.obs.index)
    else:
        ref_mask = pd.Series(True, index=adata.obs.index)

    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()

    ref_mean = X[ref_mask.values].mean(axis=0)
    ref_mean = np.where(ref_mean > 0, ref_mean, 1)

    ratio = X / ref_mean

    n_genes = X.shape[1]
    n_windows = (n_genes - window_size) // step_size + 1

    if n_windows <= 0:
        window_size = n_genes
        n_windows = 1
        step_size = n_genes

    window_scores = np.zeros((X.shape[0], n_windows))
    for i in range(n_windows):
        start = i * step_size
        end = start + window_size
        window_scores[:, i] = np.mean(ratio[:, start:end], axis=1)

    cnv_score = np.var(window_scores, axis=1)
    cnv_mean_dev = np.mean(np.abs(window_scores - 1), axis=1)

    result = pd.DataFrame({
        "barcode": adata.obs.index,
        "cnv_proxy_score": cnv_score,
        "cnv_mean_deviation": cnv_mean_dev,
        "n_windows": n_windows,
        "method": "expression_proxy",
        "caution": "Expression-derived CNV proxy. Use inferCNV/CopyKAT for rigorous analysis.",
    })

    return result


def merge_cnv_proxy_with_spots(
    cnv_df: pd.DataFrame,
    spatial_metadata_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge CNV proxy scores with spatial metadata.

    Args:
        cnv_df: DataFrame with CNV proxy scores
        spatial_metadata_df: DataFrame with spatial coordinates

    Returns:
        Merged DataFrame
    """
    meta_cols = ["barcode"]
    for col in ["x", "y", "sample_id", "donor_id", "stage", "cell_type"]:
        if col in spatial_metadata_df.columns:
            meta_cols.append(col)

    result = cnv_df.merge(
        spatial_metadata_df[meta_cols].drop_duplicates(subset=["barcode"]),
        on="barcode",
        how="left",
    )

    return result


def summarize_cnv_proxy_by_transition_score(
    cnv_df: pd.DataFrame,
    transition_df: pd.DataFrame,
    high_quantile: float = 0.90,
) -> pd.DataFrame:
    """Summarize CNV proxy by transition score groups.

    Args:
        cnv_df: DataFrame with CNV proxy scores
        transition_df: DataFrame with transition scores
        high_quantile: Quantile threshold for high-transition

    Returns:
        Summary DataFrame
    """
    df = cnv_df.merge(
        transition_df[["barcode", "transition_score"]],
        on="barcode",
        how="left",
    )

    threshold = df["transition_score"].quantile(high_quantile)
    df["transition_group"] = np.where(
        df["transition_score"] >= threshold,
        "high_transition",
        "low_transition",
    )

    score_col = "cnv_proxy_score" if "cnv_proxy_score" in df.columns else "cnv_score"

    summary = df.groupby("transition_group").agg({
        score_col: ["mean", "std", "count"],
        "cnv_mean_deviation": ["mean", "std"],
    }).reset_index()

    summary.columns = [
        "transition_group",
        "cnv_score_mean", "cnv_score_std", "n_cells",
        "cnv_deviation_mean", "cnv_deviation_std",
    ]

    return summary


def classify_clone_state(
    cnv_score: float,
    threshold_low: float = 0.05,
    threshold_high: float = 0.15,
) -> str:
    """Classify clone state based on CNV proxy score.

    Args:
        cnv_score: CNV proxy score
        threshold_low: Threshold for near-diploid
        threshold_high: Threshold for high aneuploidy

    Returns:
        Clone state label
    """
    if cnv_score < threshold_low:
        return "near_diploid"
    elif cnv_score < threshold_high:
        return "intermediate_aneuploidy"
    else:
        return "high_aneuploidy"
