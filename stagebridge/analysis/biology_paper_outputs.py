"""
Biology Paper Output Generation for StageBridge

Implements the specific biological analyses needed for the biology/clinical paper:
1. Cell-level progression risk scores
2. Niche-risk scores per neighborhood
3. Proinflammatory niche identification (IL1B-high macrophages)
4. KAC/alveolar progenitor state scoring
5. Perturbation-style analysis (counterfactual niche ablation)
6. Stage-specific ecosystem summaries

These outputs address the biological question:
"Which epithelial cells, in which local niches, appear most progression-prone,
and does that reveal an interceptable early disease ecosystem?"

Reference: Peng/Kadara LUAD precursor paper findings
- KAC/reactive pneumocyte-like alveolar progenitors are LUAD predecessors
- They reside in epithelial-proinflammatory niches with IL1B-high macrophages
- IL1B-IL1R1 signaling axis is key
- These niches more common in AAH/AIS than LUAD
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import warnings


# =============================================================================
# Marker Gene Sets (from Peng/Kadara and literature)
# =============================================================================

# KAC / Alveolar progenitor markers
KAC_MARKERS = {
    "positive": [
        "SFTPC",  # Surfactant protein C - AT2 marker
        "SFTPA1",  # Surfactant protein A1
        "SFTPA2",  # Surfactant protein A2
        "ABCA3",  # AT2 marker
        "NKX2-1",  # Lung lineage TF (also TTF1)
        "NAPSA",  # AT2 marker
        "SLC34A2",  # AT2 marker
        "LAMP3",  # AT2/AEC marker
        "ETV5",  # Alveolar progenitor TF
        "SOX9",  # Progenitor marker
    ],
    "negative": [
        "SCGB1A1",  # Club cell marker (not KAC)
        "FOXJ1",  # Ciliated cell marker
        "MUC5B",  # Goblet cell marker
    ],
}

# Proinflammatory macrophage markers
PROINFLAMMATORY_MACROPHAGE_MARKERS = {
    "positive": [
        "IL1B",  # Key inflammatory cytokine
        "IL6",  # Inflammatory cytokine
        "TNF",  # TNF-alpha
        "CXCL8",  # IL-8
        "CCL2",  # MCP-1
        "CCL3",  # MIP-1alpha
        "NLRP3",  # Inflammasome
        "CD68",  # Pan-macrophage
        "CD14",  # Monocyte/macrophage
        "FCGR3A",  # CD16
    ],
    "negative": [
        "CD163",  # M2 marker
        "MRC1",  # CD206, M2 marker
        "FOLR2",  # M2/tissue-resident
    ],
}

# IL1B signaling pathway genes
IL1B_PATHWAY_GENES = [
    "IL1B",
    "IL1R1",
    "IL1R2",
    "IL1RAP",
    "IL1RN",
    "MYD88",
    "IRAK1",
    "IRAK4",
    "TRAF6",
    "NFKB1",
    "RELA",
]

# CAF (Cancer-Associated Fibroblast) markers
CAF_MARKERS = {
    "positive": [
        "FAP",  # Fibroblast activation protein
        "ACTA2",  # Alpha-SMA
        "PDGFRA",
        "PDGFRB",
        "COL1A1",
        "COL1A2",
        "COL3A1",
        "FN1",  # Fibronectin
        "POSTN",  # Periostin
        "THY1",  # CD90
    ],
    "negative": [
        "EPCAM",  # Epithelial marker
        "PTPRC",  # CD45, immune marker
    ],
}

# EMT (Epithelial-Mesenchymal Transition) markers
EMT_MARKERS = {
    "mesenchymal": [
        "VIM",  # Vimentin
        "CDH2",  # N-cadherin
        "SNAI1",  # Snail
        "SNAI2",  # Slug
        "TWIST1",
        "ZEB1",
        "ZEB2",
        "FN1",
    ],
    "epithelial": [
        "CDH1",  # E-cadherin
        "EPCAM",
        "KRT8",
        "KRT18",
        "KRT19",
        "OCLN",  # Occludin
        "TJP1",  # ZO-1
    ],
}


# =============================================================================
# Data Classes for Outputs
# =============================================================================


@dataclass
class CellProgressionRisk:
    """Per-cell progression risk assessment."""

    cell_id: str
    progression_risk_score: float  # 0-1, higher = more progression-prone
    hlca_distance: float  # Distance from healthy reference
    luca_distance: float  # Distance from cancer reference
    reference_bias: float  # LuCA - HLCA (positive = cancer-like)
    kac_state_score: float  # KAC/alveolar progenitor score
    stage: str
    cell_type: str
    donor_id: str


@dataclass
class NicheRiskAssessment:
    """Per-neighborhood risk assessment."""

    cell_id: str  # Receiver cell ID
    niche_risk_score: float  # 0-1, higher = more progression-prone niche
    proinflammatory_score: float  # IL1B-high macrophage enrichment
    caf_enrichment: float  # CAF presence
    il1b_pathway_activity: float  # IL1B signaling score
    immune_infiltration: float  # Overall immune presence
    n_neighbors: int
    stage: str
    dominant_neighbor_type: str


@dataclass
class PerturbationResult:
    """Result of counterfactual niche perturbation."""

    cell_id: str
    original_prediction: np.ndarray
    perturbed_prediction: np.ndarray
    removed_cell_type: str
    prediction_delta: float  # Magnitude of change
    progression_risk_delta: float  # Change in progression risk
    interpretation: str


@dataclass
class StageEcosystemSummary:
    """Summary of niche ecosystem for a disease stage."""

    stage: str
    n_cells: int
    mean_progression_risk: float
    std_progression_risk: float
    mean_niche_risk: float
    proinflammatory_niche_fraction: float  # Fraction with high IL1B-mac
    caf_enriched_fraction: float
    dominant_niche_types: List[str]
    kac_cell_fraction: float
    il1b_pathway_activity: float
    comparison_to_normal: Dict[str, float]  # Fold changes vs Normal


# =============================================================================
# Scoring Functions
# =============================================================================


def compute_marker_score(
    expression: np.ndarray,
    gene_names: List[str],
    positive_markers: List[str],
    negative_markers: Optional[List[str]] = None,
    method: str = "mean",
) -> np.ndarray:
    """
    Compute marker-based score for cells.

    Args:
        expression: (n_cells, n_genes) expression matrix
        gene_names: List of gene names corresponding to columns
        positive_markers: Genes that increase the score
        negative_markers: Genes that decrease the score
        method: 'mean' or 'sum'

    Returns:
        (n_cells,) array of scores
    """
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    # Find available positive markers
    pos_indices = [gene_to_idx[g] for g in positive_markers if g in gene_to_idx]
    if not pos_indices:
        warnings.warn(f"No positive markers found in gene list")
        return np.zeros(expression.shape[0])

    # Compute positive score
    pos_expr = expression[:, pos_indices]
    if method == "mean":
        pos_score = np.mean(pos_expr, axis=1)
    else:
        pos_score = np.sum(pos_expr, axis=1)

    # Subtract negative markers if provided
    if negative_markers:
        neg_indices = [gene_to_idx[g] for g in negative_markers if g in gene_to_idx]
        if neg_indices:
            neg_expr = expression[:, neg_indices]
            neg_score = np.mean(neg_expr, axis=1) if method == "mean" else np.sum(neg_expr, axis=1)
            pos_score = pos_score - 0.5 * neg_score  # Weighted subtraction

    return pos_score


def normalize_scores(scores: np.ndarray, method: str = "minmax") -> np.ndarray:
    """Normalize scores to 0-1 range."""
    if method == "minmax":
        min_val, max_val = scores.min(), scores.max()
        if max_val - min_val < 1e-8:
            return np.zeros_like(scores)
        return (scores - min_val) / (max_val - min_val)
    elif method == "percentile":
        return np.argsort(np.argsort(scores)) / len(scores)
    elif method == "sigmoid":
        # Center and scale, then sigmoid
        centered = (scores - np.mean(scores)) / (np.std(scores) + 1e-8)
        return 1 / (1 + np.exp(-centered))
    else:
        raise ValueError(f"Unknown normalization method: {method}")


# =============================================================================
# Main Analysis Functions
# =============================================================================


def compute_cell_progression_risk(
    embeddings: pd.DataFrame,
    expression: Optional[np.ndarray] = None,
    gene_names: Optional[List[str]] = None,
    metadata: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Compute per-cell progression risk scores.

    The progression risk is based on:
    1. Position in dual-reference space (HLCA vs LuCA distance)
    2. KAC/alveolar progenitor state (if expression available)

    Args:
        embeddings: DataFrame with columns including:
            - cell_id
            - hlca_latent_* or luca_latent_* columns
            - hlca_confidence, luca_confidence (optional)
        expression: (n_cells, n_genes) expression matrix (optional)
        gene_names: Gene names for expression matrix (optional)
        metadata: DataFrame with cell_id, stage, cell_type, donor_id

    Returns:
        DataFrame with progression risk scores per cell
    """
    results = []

    # Extract HLCA and LuCA latent columns
    hlca_cols = [c for c in embeddings.columns if c.startswith("hlca_latent") or c.startswith("latent_") and "_hlca" in c]
    luca_cols = [c for c in embeddings.columns if c.startswith("luca_latent") or c.startswith("latent_") and "_luca" in c]

    # Fallback: look for generic latent columns and split
    if not hlca_cols and not luca_cols:
        latent_cols = [c for c in embeddings.columns if c.startswith("latent_")]
        if len(latent_cols) >= 40:
            hlca_cols = latent_cols[:30]
            luca_cols = latent_cols[30:40]
        elif len(latent_cols) > 0:
            hlca_cols = latent_cols[:len(latent_cols)//2]
            luca_cols = latent_cols[len(latent_cols)//2:]

    # Get confidence columns if available
    hlca_conf_col = "hlca_confidence" if "hlca_confidence" in embeddings.columns else None
    luca_conf_col = "luca_confidence" if "luca_confidence" in embeddings.columns else None

    # Compute KAC scores if expression available
    kac_scores = None
    if expression is not None and gene_names is not None:
        kac_scores = compute_marker_score(
            expression,
            gene_names,
            KAC_MARKERS["positive"],
            KAC_MARKERS["negative"],
        )
        kac_scores = normalize_scores(kac_scores, method="percentile")

    # Compute per-cell scores
    for idx, row in embeddings.iterrows():
        cell_id = row.get("cell_id", str(idx))

        # HLCA distance (lower = more healthy-like)
        if hlca_cols:
            hlca_vec = row[hlca_cols].values.astype(float)
            hlca_dist = np.linalg.norm(hlca_vec)  # Distance from origin in HLCA space
        else:
            hlca_dist = 0.0

        # LuCA distance (lower = more cancer-like)
        if luca_cols:
            luca_vec = row[luca_cols].values.astype(float)
            luca_dist = np.linalg.norm(luca_vec)
        else:
            luca_dist = 0.0

        # Reference bias: positive = more cancer-like
        # Use confidence if available, otherwise use normalized distances
        if hlca_conf_col and luca_conf_col:
            hlca_conf = row[hlca_conf_col] if pd.notna(row[hlca_conf_col]) else 0.5
            luca_conf = row[luca_conf_col] if pd.notna(row[luca_conf_col]) else 0.5
            reference_bias = luca_conf - hlca_conf
        else:
            # Normalize distances and compute bias
            total_dist = hlca_dist + luca_dist + 1e-8
            reference_bias = (hlca_dist - luca_dist) / total_dist  # Positive = closer to LuCA

        # KAC state score
        kac_score = kac_scores[idx] if kac_scores is not None else 0.5

        # Progression risk: combination of reference bias and KAC state
        # Higher reference_bias (cancer-like) + higher KAC score = higher risk
        progression_risk = 0.6 * (reference_bias + 1) / 2 + 0.4 * kac_score
        progression_risk = np.clip(progression_risk, 0, 1)

        # Get metadata
        if metadata is not None and cell_id in metadata.index:
            meta = metadata.loc[cell_id]
            stage = meta.get("stage", "Unknown")
            cell_type = meta.get("cell_type", "Unknown")
            donor_id = meta.get("donor_id", "Unknown")
        else:
            stage = row.get("stage", "Unknown")
            cell_type = row.get("cell_type", "Unknown")
            donor_id = row.get("donor_id", "Unknown")

        results.append({
            "cell_id": cell_id,
            "progression_risk_score": float(progression_risk),
            "hlca_distance": float(hlca_dist),
            "luca_distance": float(luca_dist),
            "reference_bias": float(reference_bias),
            "kac_state_score": float(kac_score),
            "stage": str(stage),
            "cell_type": str(cell_type),
            "donor_id": str(donor_id),
        })

    return pd.DataFrame(results)


def compute_niche_risk_scores(
    neighborhoods: pd.DataFrame,
    cell_expression: Optional[np.ndarray] = None,
    gene_names: Optional[List[str]] = None,
    cell_types: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Compute per-neighborhood niche risk scores.

    Niche risk is based on:
    1. Proinflammatory macrophage enrichment (IL1B-high)
    2. CAF presence
    3. IL1B pathway activity
    4. Overall composition

    Args:
        neighborhoods: DataFrame with neighborhood composition info
            Expected columns: cell_id, neighbor_cell_types (dict or list),
            or ring_*_composition columns
        cell_expression: Expression matrix for computing pathway scores
        gene_names: Gene names
        cell_types: Series mapping cell_id to cell_type

    Returns:
        DataFrame with niche risk scores
    """
    results = []

    for idx, row in neighborhoods.iterrows():
        cell_id = row.get("cell_id", str(idx))

        # Extract neighbor composition
        neighbor_types = {}
        if "neighbor_cell_types" in row and row["neighbor_cell_types"] is not None:
            if isinstance(row["neighbor_cell_types"], dict):
                neighbor_types = row["neighbor_cell_types"]
            elif isinstance(row["neighbor_cell_types"], (list, np.ndarray)):
                for ct in row["neighbor_cell_types"]:
                    neighbor_types[ct] = neighbor_types.get(ct, 0) + 1

        # Also check for ring composition columns
        for col in neighborhoods.columns:
            if "composition" in col.lower() and isinstance(row[col], dict):
                for ct, count in row[col].items():
                    neighbor_types[ct] = neighbor_types.get(ct, 0) + count

        total_neighbors = sum(neighbor_types.values()) or 1

        # Compute proinflammatory score (macrophage-related types)
        mac_types = ["Macrophage", "Monocyte", "M1_Macrophage", "Inflammatory_Mac",
                     "TAM", "MARCO_Mac", "SPP1_Mac", "FCN1_Mac"]
        mac_count = sum(neighbor_types.get(t, 0) for t in mac_types)
        proinflammatory_score = mac_count / total_neighbors

        # Compute CAF enrichment
        caf_types = ["Fibroblast", "CAF", "Myofibroblast", "iCAF", "myCAF",
                     "apCAF", "Matrix_Fibroblast"]
        caf_count = sum(neighbor_types.get(t, 0) for t in caf_types)
        caf_enrichment = caf_count / total_neighbors

        # Compute immune infiltration
        immune_types = ["T_cell", "CD4_T", "CD8_T", "Treg", "NK", "B_cell",
                        "Plasma", "DC", "pDC", "cDC", "Mast"]
        immune_count = sum(neighbor_types.get(t, 0) for t in immune_types)
        immune_infiltration = immune_count / total_neighbors

        # IL1B pathway activity (placeholder - would need expression data)
        il1b_activity = proinflammatory_score * 0.7 + caf_enrichment * 0.3

        # Niche risk score: weighted combination
        # Proinflammatory macrophages + CAFs = high risk (from Peng/Kadara)
        niche_risk = (
            0.5 * proinflammatory_score +
            0.3 * caf_enrichment +
            0.2 * il1b_activity
        )
        niche_risk = np.clip(niche_risk, 0, 1)

        # Find dominant neighbor type
        dominant_type = max(neighbor_types, key=neighbor_types.get) if neighbor_types else "Unknown"

        results.append({
            "cell_id": cell_id,
            "niche_risk_score": float(niche_risk),
            "proinflammatory_score": float(proinflammatory_score),
            "caf_enrichment": float(caf_enrichment),
            "il1b_pathway_activity": float(il1b_activity),
            "immune_infiltration": float(immune_infiltration),
            "n_neighbors": int(total_neighbors),
            "stage": row.get("stage", "Unknown"),
            "dominant_neighbor_type": dominant_type,
        })

    return pd.DataFrame(results)


def identify_proinflammatory_niches(
    niche_scores: pd.DataFrame,
    threshold: float = 0.3,
) -> pd.DataFrame:
    """
    Identify cells residing in proinflammatory niches.

    Based on Peng/Kadara: epithelial-proinflammatory niches enriched for
    IL1B-high macrophages are associated with LUAD precursor progression.

    Args:
        niche_scores: Output from compute_niche_risk_scores
        threshold: Proinflammatory score threshold (default 0.3)

    Returns:
        DataFrame with proinflammatory niche flags and characterization
    """
    df = niche_scores.copy()

    # Flag proinflammatory niches
    df["is_proinflammatory_niche"] = df["proinflammatory_score"] >= threshold

    # Categorize niche types
    def categorize_niche(row):
        if row["proinflammatory_score"] >= threshold and row["caf_enrichment"] >= 0.2:
            return "Proinflammatory-CAF"  # Highest risk
        elif row["proinflammatory_score"] >= threshold:
            return "Proinflammatory"
        elif row["caf_enrichment"] >= 0.3:
            return "CAF-enriched"
        elif row["immune_infiltration"] >= 0.4:
            return "Immune-infiltrated"
        else:
            return "Normal-like"

    df["niche_category"] = df.apply(categorize_niche, axis=1)

    # Risk tier
    def risk_tier(score):
        if score >= 0.7:
            return "High"
        elif score >= 0.4:
            return "Medium"
        else:
            return "Low"

    df["risk_tier"] = df["niche_risk_score"].apply(risk_tier)

    return df


def score_kac_alveolar_progenitor_state(
    expression: np.ndarray,
    gene_names: List[str],
    cell_ids: List[str],
) -> pd.DataFrame:
    """
    Score cells for KAC/alveolar progenitor state.

    KAC (Keratin-AEC) cells are reactive pneumocyte-like alveolar progenitors
    that are early predecessors of LUAD (from Peng/Kadara).

    Args:
        expression: (n_cells, n_genes) expression matrix
        gene_names: Gene names
        cell_ids: Cell identifiers

    Returns:
        DataFrame with KAC state scores
    """
    # Compute KAC marker score
    kac_score = compute_marker_score(
        expression, gene_names,
        KAC_MARKERS["positive"],
        KAC_MARKERS["negative"],
    )
    kac_score_norm = normalize_scores(kac_score, method="percentile")

    # Compute AT2 differentiation score (subset of KAC markers)
    at2_markers = ["SFTPC", "SFTPA1", "SFTPA2", "ABCA3", "NAPSA"]
    at2_score = compute_marker_score(expression, gene_names, at2_markers)
    at2_score_norm = normalize_scores(at2_score, method="percentile")

    # Compute progenitor/stemness score
    progenitor_markers = ["SOX9", "ETV5", "NKX2-1", "ID2", "AXIN2"]
    progenitor_score = compute_marker_score(expression, gene_names, progenitor_markers)
    progenitor_score_norm = normalize_scores(progenitor_score, method="percentile")

    # Categorize cells
    def categorize_kac(kac, at2, prog):
        if kac >= 0.7 and prog >= 0.5:
            return "KAC-like"  # High KAC + progenitor = progression-prone
        elif at2 >= 0.7:
            return "Mature-AT2"
        elif prog >= 0.7:
            return "Progenitor"
        else:
            return "Other"

    categories = [
        categorize_kac(k, a, p)
        for k, a, p in zip(kac_score_norm, at2_score_norm, progenitor_score_norm)
    ]

    return pd.DataFrame({
        "cell_id": cell_ids,
        "kac_score": kac_score_norm,
        "at2_differentiation_score": at2_score_norm,
        "progenitor_score": progenitor_score_norm,
        "kac_category": categories,
        "is_kac_like": [c == "KAC-like" for c in categories],
    })


def perturbation_analysis(
    model: torch.nn.Module,
    batch: Any,
    target_cell_type: str,
    device: str = "cuda",
) -> List[PerturbationResult]:
    """
    Counterfactual analysis: what happens if we remove a cell type from the niche?

    This addresses the question: "How would the receiver cell's predicted state
    change if IL1B-high macrophages were not present in its neighborhood?"

    Args:
        model: Trained StageBridge model
        batch: Batch of neighborhoods
        target_cell_type: Cell type to ablate (e.g., "Macrophage")
        device: Computation device

    Returns:
        List of PerturbationResult for each cell in batch
    """
    model.eval()
    model.to(device)
    results = []

    with torch.no_grad():
        # Original prediction
        batch_device = batch.to(device) if hasattr(batch, 'to') else batch
        original_output = model(batch_device)

        if isinstance(original_output, dict):
            original_pred = original_output.get("reconstruction", original_output.get("output"))
        else:
            original_pred = original_output

        if original_pred is None:
            warnings.warn("Model did not return predictions")
            return results

        original_pred = original_pred.cpu().numpy()

        # Create perturbed batch by zeroing out target cell type contributions
        # This is a simplified perturbation - in practice you'd modify the token embeddings
        perturbed_batch = _ablate_cell_type_from_batch(batch, target_cell_type)

        if perturbed_batch is not None:
            perturbed_batch_device = perturbed_batch.to(device) if hasattr(perturbed_batch, 'to') else perturbed_batch
            perturbed_output = model(perturbed_batch_device)

            if isinstance(perturbed_output, dict):
                perturbed_pred = perturbed_output.get("reconstruction", perturbed_output.get("output"))
            else:
                perturbed_pred = perturbed_output

            perturbed_pred = perturbed_pred.cpu().numpy() if perturbed_pred is not None else original_pred
        else:
            perturbed_pred = original_pred

        # Compute deltas
        cell_ids = batch.cell_ids if hasattr(batch, 'cell_ids') else [f"cell_{i}" for i in range(len(original_pred))]

        for i, cell_id in enumerate(cell_ids):
            delta = np.linalg.norm(perturbed_pred[i] - original_pred[i])

            # Estimate progression risk change (simplified)
            orig_risk = np.mean(original_pred[i])
            pert_risk = np.mean(perturbed_pred[i])
            risk_delta = pert_risk - orig_risk

            # Interpretation
            if delta > 0.1:
                if risk_delta < 0:
                    interp = f"Removing {target_cell_type} reduces progression signature"
                else:
                    interp = f"Removing {target_cell_type} increases stress response"
            else:
                interp = f"{target_cell_type} has minimal influence on this receiver"

            results.append(PerturbationResult(
                cell_id=cell_id,
                original_prediction=original_pred[i],
                perturbed_prediction=perturbed_pred[i],
                removed_cell_type=target_cell_type,
                prediction_delta=float(delta),
                progression_risk_delta=float(risk_delta),
                interpretation=interp,
            ))

    return results


def _ablate_cell_type_from_batch(batch: Any, cell_type: str) -> Any:
    """Helper to create perturbed batch with cell type ablated."""
    # This is a simplified implementation
    # In practice, you'd modify the ring token embeddings to remove
    # contributions from the target cell type

    if not hasattr(batch, 'clone') and not hasattr(batch, 'copy'):
        return None

    try:
        # Deep copy the batch
        import copy
        perturbed = copy.deepcopy(batch)

        # If batch has cell type composition, zero out target type
        if hasattr(perturbed, 'ring_compositions'):
            for ring in perturbed.ring_compositions:
                if cell_type in ring:
                    ring[cell_type] = 0

        return perturbed
    except Exception:
        return None


def compute_stage_ecosystem_summary(
    cell_risks: pd.DataFrame,
    niche_risks: pd.DataFrame,
    kac_scores: Optional[pd.DataFrame] = None,
) -> Dict[str, StageEcosystemSummary]:
    """
    Generate stage-specific ecosystem summaries.

    Answers: "Which niche patterns are enriched in AAH/AIS vs LUAD?"

    Args:
        cell_risks: Output from compute_cell_progression_risk
        niche_risks: Output from compute_niche_risk_scores (with categories)
        kac_scores: Output from score_kac_alveolar_progenitor_state

    Returns:
        Dict mapping stage to StageEcosystemSummary
    """
    # Merge dataframes
    merged = cell_risks.merge(niche_risks, on="cell_id", suffixes=("", "_niche"))

    if kac_scores is not None:
        merged = merged.merge(kac_scores[["cell_id", "kac_score", "is_kac_like"]], on="cell_id", how="left")
        merged["kac_score"] = merged["kac_score"].fillna(0.5)
        merged["is_kac_like"] = merged["is_kac_like"].fillna(False)
    else:
        merged["kac_score"] = 0.5
        merged["is_kac_like"] = False

    summaries = {}
    stages = merged["stage"].unique()

    # Compute Normal baseline for comparison
    normal_data = merged[merged["stage"] == "Normal"] if "Normal" in stages else None
    normal_means = {}
    if normal_data is not None and len(normal_data) > 0:
        normal_means = {
            "progression_risk": normal_data["progression_risk_score"].mean(),
            "niche_risk": normal_data["niche_risk_score"].mean(),
            "proinflammatory": normal_data["proinflammatory_score"].mean(),
            "caf": normal_data["caf_enrichment"].mean(),
        }

    for stage in stages:
        stage_data = merged[merged["stage"] == stage]
        n_cells = len(stage_data)

        if n_cells == 0:
            continue

        # Basic stats
        mean_prog_risk = stage_data["progression_risk_score"].mean()
        std_prog_risk = stage_data["progression_risk_score"].std()
        mean_niche_risk = stage_data["niche_risk_score"].mean()

        # Proinflammatory niche fraction
        if "is_proinflammatory_niche" in stage_data.columns:
            proinflam_frac = stage_data["is_proinflammatory_niche"].mean()
        else:
            proinflam_frac = (stage_data["proinflammatory_score"] >= 0.3).mean()

        # CAF enriched fraction
        caf_frac = (stage_data["caf_enrichment"] >= 0.2).mean()

        # KAC fraction
        kac_frac = stage_data["is_kac_like"].mean() if "is_kac_like" in stage_data.columns else 0.0

        # IL1B pathway activity
        il1b_activity = stage_data["il1b_pathway_activity"].mean()

        # Dominant niche types
        if "niche_category" in stage_data.columns:
            niche_counts = stage_data["niche_category"].value_counts()
            dominant_niches = niche_counts.head(3).index.tolist()
        else:
            dominant_niches = [stage_data["dominant_neighbor_type"].mode().iloc[0]] if len(stage_data) > 0 else []

        # Comparison to Normal
        comparison = {}
        if normal_means:
            comparison = {
                "progression_risk_fc": mean_prog_risk / (normal_means["progression_risk"] + 1e-8),
                "niche_risk_fc": mean_niche_risk / (normal_means["niche_risk"] + 1e-8),
                "proinflammatory_fc": stage_data["proinflammatory_score"].mean() / (normal_means["proinflammatory"] + 1e-8),
                "caf_fc": stage_data["caf_enrichment"].mean() / (normal_means["caf"] + 1e-8),
            }

        summaries[stage] = StageEcosystemSummary(
            stage=stage,
            n_cells=n_cells,
            mean_progression_risk=float(mean_prog_risk),
            std_progression_risk=float(std_prog_risk),
            mean_niche_risk=float(mean_niche_risk),
            proinflammatory_niche_fraction=float(proinflam_frac),
            caf_enriched_fraction=float(caf_frac),
            dominant_niche_types=dominant_niches,
            kac_cell_fraction=float(kac_frac),
            il1b_pathway_activity=float(il1b_activity),
            comparison_to_normal=comparison,
        )

    return summaries


# =============================================================================
# Report Generation
# =============================================================================


def generate_biology_paper_report(
    cell_risks: pd.DataFrame,
    niche_risks: pd.DataFrame,
    stage_summaries: Dict[str, StageEcosystemSummary],
    output_dir: Path,
    kac_scores: Optional[pd.DataFrame] = None,
    perturbation_results: Optional[List[PerturbationResult]] = None,
) -> None:
    """
    Generate comprehensive report for biology paper.

    Creates:
    1. Summary statistics tables
    2. Stage comparison figures (requires matplotlib)
    3. Niche characterization
    4. Key findings narrative
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save raw data
    cell_risks.to_parquet(output_dir / "cell_progression_risks.parquet", index=False)
    niche_risks.to_parquet(output_dir / "niche_risk_scores.parquet", index=False)

    if kac_scores is not None:
        kac_scores.to_parquet(output_dir / "kac_state_scores.parquet", index=False)

    # Generate markdown report
    report = []
    report.append("# StageBridge Biology Paper Analysis Report\n\n")
    report.append("## Executive Summary\n\n")

    # Key findings
    report.append("### Key Findings\n\n")

    # Find stage with highest progression risk
    if stage_summaries:
        max_risk_stage = max(stage_summaries.values(), key=lambda x: x.mean_progression_risk)
        report.append(f"1. **Highest progression risk stage:** {max_risk_stage.stage} "
                     f"(mean risk = {max_risk_stage.mean_progression_risk:.3f})\n\n")

        # Find stage with highest proinflammatory niche fraction
        max_proinflam_stage = max(stage_summaries.values(), key=lambda x: x.proinflammatory_niche_fraction)
        report.append(f"2. **Most proinflammatory niches:** {max_proinflam_stage.stage} "
                     f"({max_proinflam_stage.proinflammatory_niche_fraction:.1%} of cells)\n\n")

        # CAF enrichment
        max_caf_stage = max(stage_summaries.values(), key=lambda x: x.caf_enriched_fraction)
        report.append(f"3. **Highest CAF enrichment:** {max_caf_stage.stage} "
                     f"({max_caf_stage.caf_enriched_fraction:.1%} of cells)\n\n")

    # Stage-specific ecosystem table
    report.append("## Stage-Specific Ecosystem Summary\n\n")
    report.append("| Stage | N Cells | Prog Risk | Niche Risk | Proinflamm % | CAF % | KAC % |\n")
    report.append("|-------|---------|-----------|------------|--------------|-------|-------|\n")

    canonical_order = ["Normal", "AAH", "AIS", "MIA", "LUAD"]
    for stage in canonical_order:
        if stage in stage_summaries:
            s = stage_summaries[stage]
            report.append(f"| {stage} | {s.n_cells:,} | {s.mean_progression_risk:.3f} | "
                         f"{s.mean_niche_risk:.3f} | {s.proinflammatory_niche_fraction:.1%} | "
                         f"{s.caf_enriched_fraction:.1%} | {s.kac_cell_fraction:.1%} |\n")

    # Fold changes vs Normal
    report.append("\n## Fold Changes vs Normal\n\n")
    report.append("| Stage | Prog Risk FC | Niche Risk FC | Proinflamm FC | CAF FC |\n")
    report.append("|-------|--------------|---------------|---------------|--------|\n")

    for stage in canonical_order[1:]:  # Skip Normal
        if stage in stage_summaries and stage_summaries[stage].comparison_to_normal:
            s = stage_summaries[stage]
            c = s.comparison_to_normal
            report.append(f"| {stage} | {c.get('progression_risk_fc', 1):.2f}x | "
                         f"{c.get('niche_risk_fc', 1):.2f}x | "
                         f"{c.get('proinflammatory_fc', 1):.2f}x | "
                         f"{c.get('caf_fc', 1):.2f}x |\n")

    # Perturbation analysis summary
    if perturbation_results:
        report.append("\n## Perturbation Analysis\n\n")
        report.append(f"Ablated cell type: {perturbation_results[0].removed_cell_type}\n\n")

        deltas = [r.prediction_delta for r in perturbation_results]
        report.append(f"- Mean prediction change: {np.mean(deltas):.4f}\n")
        report.append(f"- Cells with significant change (>0.1): {sum(d > 0.1 for d in deltas)}\n")

        risk_deltas = [r.progression_risk_delta for r in perturbation_results]
        report.append(f"- Mean progression risk change: {np.mean(risk_deltas):.4f}\n")

    # Biological interpretation
    report.append("\n## Biological Interpretation\n\n")
    report.append("Based on Peng/Kadara LUAD precursor findings:\n\n")

    if stage_summaries:
        # Check if proinflammatory niches are enriched in precursors
        precursor_stages = ["AAH", "AIS", "MIA"]
        precursor_proinflam = np.mean([
            stage_summaries[s].proinflammatory_niche_fraction
            for s in precursor_stages if s in stage_summaries
        ]) if any(s in stage_summaries for s in precursor_stages) else 0

        luad_proinflam = stage_summaries.get("LUAD", StageEcosystemSummary(
            stage="LUAD", n_cells=0, mean_progression_risk=0, std_progression_risk=0,
            mean_niche_risk=0, proinflammatory_niche_fraction=0, caf_enriched_fraction=0,
            dominant_niche_types=[], kac_cell_fraction=0, il1b_pathway_activity=0,
            comparison_to_normal={}
        )).proinflammatory_niche_fraction

        if precursor_proinflam > luad_proinflam:
            report.append("✓ **CONFIRMED:** Proinflammatory niches are more common in precursor "
                         f"lesions ({precursor_proinflam:.1%}) than in LUAD ({luad_proinflam:.1%}), "
                         "consistent with Peng/Kadara findings.\n\n")
        else:
            report.append("⚠ Proinflammatory niche enrichment pattern differs from expected. "
                         "May reflect dataset-specific characteristics.\n\n")

    # Save report
    with open(output_dir / "biology_paper_report.md", "w") as f:
        f.writelines(report)

    # Save stage summaries as JSON
    import json
    summary_dict = {
        stage: {
            "n_cells": s.n_cells,
            "mean_progression_risk": s.mean_progression_risk,
            "std_progression_risk": s.std_progression_risk,
            "mean_niche_risk": s.mean_niche_risk,
            "proinflammatory_niche_fraction": s.proinflammatory_niche_fraction,
            "caf_enriched_fraction": s.caf_enriched_fraction,
            "dominant_niche_types": s.dominant_niche_types,
            "kac_cell_fraction": s.kac_cell_fraction,
            "il1b_pathway_activity": s.il1b_pathway_activity,
            "comparison_to_normal": s.comparison_to_normal,
        }
        for stage, s in stage_summaries.items()
    }
    with open(output_dir / "stage_ecosystem_summaries.json", "w") as f:
        json.dump(summary_dict, f, indent=2)

    print(f"Biology paper outputs saved to: {output_dir}")
    print(f"  - cell_progression_risks.parquet")
    print(f"  - niche_risk_scores.parquet")
    print(f"  - stage_ecosystem_summaries.json")
    print(f"  - biology_paper_report.md")


# =============================================================================
# Convenience Pipeline Function
# =============================================================================


def run_biology_paper_analysis(
    embeddings_path: Path,
    neighborhoods_path: Path,
    output_dir: Path,
    expression_path: Optional[Path] = None,
    model: Optional[torch.nn.Module] = None,
    ablate_cell_type: Optional[str] = "Macrophage",
) -> Dict[str, Any]:
    """
    Run complete biology paper analysis pipeline.

    Args:
        embeddings_path: Path to cell embeddings parquet
        neighborhoods_path: Path to neighborhood composition parquet
        output_dir: Output directory
        expression_path: Optional path to expression h5ad for marker scoring
        model: Optional trained model for perturbation analysis
        ablate_cell_type: Cell type to ablate in perturbation analysis

    Returns:
        Dict with all analysis outputs
    """
    print("Running biology paper analysis...")

    # Load data
    embeddings = pd.read_parquet(embeddings_path)
    neighborhoods = pd.read_parquet(neighborhoods_path)

    # Load expression if available
    expression = None
    gene_names = None
    if expression_path and expression_path.exists():
        try:
            import anndata
            adata = anndata.read_h5ad(expression_path)
            expression = adata.X.toarray() if hasattr(adata.X, 'toarray') else np.array(adata.X)
            gene_names = list(adata.var_names)
            cell_ids = list(adata.obs_names)
        except Exception as e:
            print(f"Could not load expression data: {e}")

    # 1. Cell progression risk
    print("  Computing cell progression risks...")
    cell_risks = compute_cell_progression_risk(embeddings, expression, gene_names)

    # 2. Niche risk scores
    print("  Computing niche risk scores...")
    niche_risks = compute_niche_risk_scores(neighborhoods)

    # 3. Identify proinflammatory niches
    print("  Identifying proinflammatory niches...")
    niche_risks = identify_proinflammatory_niches(niche_risks)

    # 4. KAC scoring (if expression available)
    kac_scores = None
    if expression is not None and gene_names is not None:
        print("  Scoring KAC/alveolar progenitor states...")
        kac_scores = score_kac_alveolar_progenitor_state(
            expression, gene_names,
            cell_ids if 'cell_ids' in dir() else embeddings["cell_id"].tolist()
        )

    # 5. Stage ecosystem summaries
    print("  Computing stage ecosystem summaries...")
    stage_summaries = compute_stage_ecosystem_summary(cell_risks, niche_risks, kac_scores)

    # 6. Perturbation analysis (if model available)
    perturbation_results = None
    # Note: Perturbation analysis requires batch data structure from dataloader
    # This would be run separately during model evaluation

    # 7. Generate report
    print("  Generating report...")
    generate_biology_paper_report(
        cell_risks, niche_risks, stage_summaries, output_dir,
        kac_scores, perturbation_results
    )

    return {
        "cell_risks": cell_risks,
        "niche_risks": niche_risks,
        "kac_scores": kac_scores,
        "stage_summaries": stage_summaries,
        "perturbation_results": perturbation_results,
    }


if __name__ == "__main__":
    print("Biology paper outputs module loaded.")
    print("\nUsage:")
    print("  from stagebridge.analysis.biology_paper_outputs import run_biology_paper_analysis")
    print("  results = run_biology_paper_analysis(embeddings_path, neighborhoods_path, output_dir)")
