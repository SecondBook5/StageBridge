"""Pathway and proliferation target computation for auxiliary supervision.

Paper-inspired auxiliary losses:
- OSDR (Nature 2026): Ki67 proliferation from niche context
- SpatialFusion: PROGENy pathway regression

These targets are computed from expression data and used to train
auxiliary heads that force the latent space to encode biologically
meaningful features.
"""

from __future__ import annotations

import torch

# PROGENy pathway gene sets (top markers per pathway)
# Reference: Schubert et al., Nature Communications 2018
PROGENY_PATHWAYS = {
    "Androgen": ["KLK3", "KLK2", "TMPRSS2", "NKX3-1", "FKBP5"],
    "EGFR": ["AREG", "EREG", "HBEGF", "DUSP6", "SPRY2"],
    "Estrogen": ["GREB1", "PGR", "TFF1", "XBP1", "CXCL12"],
    "Hypoxia": ["VEGFA", "SLC2A1", "LDHA", "PGK1", "BNIP3", "CA9"],
    "JAK-STAT": ["SOCS1", "SOCS3", "IRF1", "IRF9", "STAT1"],
    "MAPK": ["DUSP6", "SPRY2", "ETV4", "FOS", "JUN", "EGR1"],
    "NFkB": ["NFKBIA", "TNFAIP3", "CCL2", "CXCL8", "IL6"],
    "p53": ["CDKN1A", "BAX", "MDM2", "GADD45A", "PMAIP1"],
    "PI3K": ["PTEN", "INPP5D", "PIK3R1", "AKT1", "FOXO3"],
    "TGFb": ["SMAD7", "SERPINE1", "COL1A1", "TGFBI", "CTGF"],
    "TNFa": ["TNFAIP3", "NFKBIA", "CCL2", "CXCL1", "IL1B"],
    "Trail": ["TNFRSF10A", "TNFRSF10B", "CASP8", "CASP3"],
    "VEGF": ["VEGFA", "KDR", "FLT1", "PGF", "NRP1"],
    "WNT": ["AXIN2", "NKD1", "DKK1", "LEF1", "MYC", "CCND1"],
}

# Proliferation markers (from stagebridge/biology/signatures.py)
PROLIFERATION_MARKERS = ["MKI67", "TOP2A", "PCNA", "MCM2", "MCM6"]

# KAC (KRT8+ Alveolar Intermediate Cell) markers
# These cells are the key intermediate in the AT2 -> LUAD progression trajectory
# From Nature 2024 Kadara lab + Peng et al. 2020 Cancer Cell
KAC_MARKERS = [
    "KRT8",  # Primary marker
    "CLDN4",  # Tight junction
    "CDKN1A",  # p21 senescence (Nature 2024)
    "CDKN2A",  # p16 senescence (Nature 2024)
    "PLAUR",  # uPAR invasion (Nature 2024)
    "CEACAM5",
    "CEACAM6",
    "MUC1",
    "MSLN",
    "CD24",
]


def compute_pathway_targets(
    expression: torch.Tensor,
    gene_names: list[str],
    device: torch.device,
    reference_stats: dict | None = None,
) -> torch.Tensor | None:
    """Compute PROGENy-style pathway scores from expression.

    From SpatialFusion: pathway regression loss improves latent organization
    by forcing it to encode biologically meaningful pathway activity.

    IMPORTANT: To avoid train/val leakage, pass reference_stats computed from
    training set only. If None, computes stats from current batch (use only
    when computing training set targets or for inference).

    Args:
        expression: [B, n_genes] expression matrix
        gene_names: List of gene names corresponding to expression columns
        device: Torch device
        reference_stats: Optional dict with pre-computed mean/std per pathway
                        from training set. Keys: pathway names, values: (mean, std) tuples

    Returns:
        [B, n_pathways] pathway activity scores, or None if no genes found
    """
    if gene_names is None or len(gene_names) == 0:
        return None

    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    n_pathways = len(PROGENY_PATHWAYS)
    batch_size = expression.shape[0]

    scores = torch.zeros(batch_size, n_pathways, device=device)
    any_found = False

    for p_idx, (pathway, genes) in enumerate(PROGENY_PATHWAYS.items()):
        gene_idx = [gene_to_idx[g] for g in genes if g in gene_to_idx]
        if len(gene_idx) > 0:
            any_found = True
            pathway_expr = expression[:, gene_idx]

            # Use reference stats if provided (for val/test), else compute from batch
            if reference_stats is not None and pathway in reference_stats:
                mean, std = reference_stats[pathway]
                mean = mean.to(device)
                std = std.to(device)
            else:
                mean = pathway_expr.mean(dim=0, keepdim=True)
                std = pathway_expr.std(dim=0, keepdim=True, unbiased=False) + 1e-8

            z_scores = (pathway_expr - mean) / std
            scores[:, p_idx] = z_scores.mean(dim=1)

    return scores if any_found else None


def compute_pathway_reference_stats(
    expression: torch.Tensor,
    gene_names: list[str],
) -> dict:
    """Compute reference statistics for pathway targets from training set.

    Use this to compute mean/std from training data only, then pass to
    compute_pathway_targets() for validation/test sets to avoid leakage.

    Args:
        expression: [B, n_genes] training expression matrix
        gene_names: List of gene names

    Returns:
        Dict mapping pathway names to (mean, std) tuples
    """
    if gene_names is None or len(gene_names) == 0:
        return {}

    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    stats = {}

    for pathway, genes in PROGENY_PATHWAYS.items():
        gene_idx = [gene_to_idx[g] for g in genes if g in gene_to_idx]
        if len(gene_idx) > 0:
            pathway_expr = expression[:, gene_idx]
            mean = pathway_expr.mean(dim=0, keepdim=True)
            std = pathway_expr.std(dim=0, keepdim=True, unbiased=False) + 1e-8
            stats[pathway] = (mean.cpu(), std.cpu())

    return stats


def compute_proliferation_targets(
    expression: torch.Tensor,
    gene_names: list[str],
    device: torch.device,
    threshold_pct: float = 75,
) -> torch.Tensor | None:
    """Compute Ki67/MKI67 proliferation targets.

    From OSDR (Nature 2026): Ki67 as direct readout of division rate.
    If niche encoder predicts Ki67 well, it's learning dynamically-relevant features.

    Args:
        expression: [B, n_genes] expression matrix
        gene_names: List of gene names
        device: Torch device
        threshold_pct: Percentile threshold for binary classification

    Returns:
        [B, 1] binary proliferation targets, or None if MKI67 not found
    """
    if gene_names is None or len(gene_names) == 0:
        return None

    # Find Ki67/MKI67
    ki67_names = ["MKI67", "KI67", "Ki67", "mki67"]
    ki67_idx = None

    for name in ki67_names:
        if name in gene_names:
            ki67_idx = gene_names.index(name)
            break
        # Case-insensitive search
        for i, g in enumerate(gene_names):
            if g.upper() == name.upper():
                ki67_idx = i
                break
        if ki67_idx is not None:
            break

    if ki67_idx is None:
        return None

    ki67_expr = expression[:, ki67_idx]
    threshold = torch.quantile(ki67_expr, threshold_pct / 100)
    return (ki67_expr > threshold).float().unsqueeze(1)


def compute_kac_targets(
    expression: torch.Tensor,
    gene_names: list[str],
    device: torch.device,
    reference_stats: tuple | None = None,
) -> torch.Tensor | None:
    """Compute KAC (KRT8+ Alveolar Intermediate Cell) signature score.

    From Nature 2024 Kadara lab: KACs are the key intermediate state in
    AT2 -> LUAD progression. Trajectory: Normal -> AT2 -> AIC -> KAC -> Tumor.
    KACs are found in tumor-adjacent normal tissue BEFORE tumors form,
    making this the critical cell state our transition model should capture.

    Markers: KRT8, CLDN4, CDKN1A, CDKN2A, PLAUR (senescence + invasion)

    IMPORTANT: To avoid train/val leakage, pass reference_stats computed from
    training set only.

    Args:
        expression: [B, n_genes] expression matrix
        gene_names: List of gene names corresponding to expression columns
        device: Torch device
        reference_stats: Optional (mean, std) tuple from training set

    Returns:
        [B, 1] KAC signature scores, or None if insufficient genes found
    """
    if gene_names is None or len(gene_names) == 0:
        return None

    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    gene_idx = [gene_to_idx[g] for g in KAC_MARKERS if g in gene_to_idx]

    if len(gene_idx) < 3:  # Need at least 3 markers
        return None

    kac_expr = expression[:, gene_idx]

    # Use reference stats if provided (for val/test), else compute from batch
    if reference_stats is not None:
        mean, std = reference_stats
        mean = mean.to(device)
        std = std.to(device)
    else:
        mean = kac_expr.mean(dim=0, keepdim=True)
        std = kac_expr.std(dim=0, keepdim=True, unbiased=False) + 1e-8

    z_scores = (kac_expr - mean) / std
    return z_scores.mean(dim=1, keepdim=True)


def compute_kac_reference_stats(
    expression: torch.Tensor,
    gene_names: list[str],
) -> tuple | None:
    """Compute reference statistics for KAC targets from training set."""
    if gene_names is None or len(gene_names) == 0:
        return None

    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    gene_idx = [gene_to_idx[g] for g in KAC_MARKERS if g in gene_to_idx]

    if len(gene_idx) < 3:
        return None

    kac_expr = expression[:, gene_idx]
    mean = kac_expr.mean(dim=0, keepdim=True)
    std = kac_expr.std(dim=0, keepdim=True, unbiased=False) + 1e-8
    return (mean.cpu(), std.cpu())


def compute_il1b_targets(
    expression: torch.Tensor,
    gene_names: list[str],
    device: torch.device,
    reference_stats: tuple | None = None,
) -> torch.Tensor | None:
    """Compute IL1B pathway activity from direct gene expression.

    This is the KEY biological target for the Peng/Kadara hypothesis:
    IL1B+ macrophages drive IL1B-IL1R1 signaling in epithelial cells.

    We measure both IL1B (the ligand) and IL1R1 (the receptor) expression
    to capture the full signaling axis. High IL1B in macrophage-rich niches
    and high IL1R1 in epithelial cells indicates active signaling.

    IMPORTANT: To avoid train/val leakage, pass reference_stats computed from
    training set only.

    Args:
        expression: [B, n_genes] expression matrix
        gene_names: List of gene names corresponding to expression columns
        device: Torch device
        reference_stats: Optional (mean, std) tuple from training set

    Returns:
        [B, 1] IL1B pathway activity scores, or None if genes not found
    """
    if gene_names is None or len(gene_names) == 0:
        return None

    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    gene_idx = [gene_to_idx[g] for g in IL1B_SIGNALING_GENES if g in gene_to_idx]

    if len(gene_idx) < 1:  # Need at least IL1B itself
        return None

    il1b_expr = expression[:, gene_idx]

    # Use reference stats if provided (for val/test), else compute from batch
    if reference_stats is not None:
        mean, std = reference_stats
        mean = mean.to(device)
        std = std.to(device)
    else:
        mean = il1b_expr.mean(dim=0, keepdim=True)
        std = il1b_expr.std(dim=0, keepdim=True, unbiased=False) + 1e-8

    z_scores = (il1b_expr - mean) / std
    return z_scores.mean(dim=1, keepdim=True)


def compute_il1b_reference_stats(
    expression: torch.Tensor,
    gene_names: list[str],
) -> tuple | None:
    """Compute reference statistics for IL1B targets from training set."""
    if gene_names is None or len(gene_names) == 0:
        return None

    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    gene_idx = [gene_to_idx[g] for g in IL1B_SIGNALING_GENES if g in gene_to_idx]

    if len(gene_idx) < 1:
        return None

    il1b_expr = expression[:, gene_idx]
    mean = il1b_expr.mean(dim=0, keepdim=True)
    std = il1b_expr.std(dim=0, keepdim=True, unbiased=False) + 1e-8
    return (mean.cpu(), std.cpu())


# IL1B signaling axis genes for reference
IL1B_SIGNALING_GENES = ["IL1B", "IL1R1", "IL1R2", "IL1RAP", "IL1RN"]


# =============================================================================
# RAW (UN-NORMALIZED) TARGET COMPUTATION
# These functions compute raw mean expression without z-scoring.
# Use these in complete_data_prep.py to store raw values, then z-score
# at training time using train-only statistics to prevent leakage.
# =============================================================================


def compute_pathway_raw(
    expression: torch.Tensor,
    gene_names: list[str],
) -> torch.Tensor | None:
    """Compute raw pathway mean expression (NO z-scoring).

    Store these values in cells.parquet. Z-score at training time using
    train-only statistics to prevent leakage.

    Args:
        expression: [B, n_genes] expression matrix
        gene_names: List of gene names

    Returns:
        [B, n_pathways] raw pathway mean expression, or None if no genes found
    """
    if gene_names is None or len(gene_names) == 0:
        return None

    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    n_pathways = len(PROGENY_PATHWAYS)
    batch_size = expression.shape[0]

    raw_means = torch.zeros(batch_size, n_pathways)
    any_found = False

    for p_idx, (pathway, genes) in enumerate(PROGENY_PATHWAYS.items()):
        gene_idx = [gene_to_idx[g] for g in genes if g in gene_to_idx]
        if len(gene_idx) > 0:
            any_found = True
            pathway_expr = expression[:, gene_idx]
            raw_means[:, p_idx] = pathway_expr.mean(dim=1)

    return raw_means if any_found else None


def compute_il1b_raw(
    expression: torch.Tensor,
    gene_names: list[str],
) -> torch.Tensor | None:
    """Compute raw IL1B signaling mean expression (NO z-scoring).

    Store in cells.parquet, z-score at training time.

    Args:
        expression: [B, n_genes] expression matrix
        gene_names: List of gene names

    Returns:
        [B, 1] raw IL1B mean expression, or None if genes not found
    """
    if gene_names is None or len(gene_names) == 0:
        return None

    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    gene_idx = [gene_to_idx[g] for g in IL1B_SIGNALING_GENES if g in gene_to_idx]

    if len(gene_idx) < 1:
        return None

    il1b_expr = expression[:, gene_idx]
    return il1b_expr.mean(dim=1, keepdim=True)


def compute_kac_raw(
    expression: torch.Tensor,
    gene_names: list[str],
) -> torch.Tensor | None:
    """Compute raw KAC signature mean expression (NO z-scoring).

    Store in cells.parquet, z-score at training time.

    Args:
        expression: [B, n_genes] expression matrix
        gene_names: List of gene names

    Returns:
        [B, 1] raw KAC mean expression, or None if insufficient genes
    """
    if gene_names is None or len(gene_names) == 0:
        return None

    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    gene_idx = [gene_to_idx[g] for g in KAC_MARKERS if g in gene_to_idx]

    if len(gene_idx) < 3:
        return None

    kac_expr = expression[:, gene_idx]
    return kac_expr.mean(dim=1, keepdim=True)


def zscore_from_train_stats(
    raw_values: torch.Tensor,
    train_idx: torch.Tensor,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Z-score values using train-only statistics.

    Args:
        raw_values: [N, D] raw values for all cells
        train_idx: indices of training cells

    Returns:
        (z_scored_values, (train_mean, train_std)) for reproducibility
    """
    train_values = raw_values[train_idx]
    train_mean = train_values.mean(dim=0, keepdim=True)
    train_std = train_values.std(dim=0, keepdim=True, unbiased=False) + 1e-8

    z_scored = (raw_values - train_mean) / train_std
    return z_scored, (train_mean.squeeze(), train_std.squeeze())


def get_pathway_gene_list() -> list[str]:
    """Get all unique genes used in pathway scoring.

    Useful for ensuring these genes are included in processed data.
    """
    genes = set()
    for pathway_genes in PROGENY_PATHWAYS.values():
        genes.update(pathway_genes)
    genes.update(PROLIFERATION_MARKERS)
    genes.update(KAC_MARKERS)
    genes.update(IL1B_SIGNALING_GENES)
    return sorted(genes)
