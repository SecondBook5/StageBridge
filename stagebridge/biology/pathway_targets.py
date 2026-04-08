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
) -> torch.Tensor | None:
    """Compute PROGENy-style pathway scores from expression.

    From SpatialFusion: pathway regression loss improves latent organization
    by forcing it to encode biologically meaningful pathway activity.

    Args:
        expression: [B, n_genes] expression matrix
        gene_names: List of gene names corresponding to expression columns
        device: Torch device

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
            # Z-score normalize and mean (unbiased=False handles single samples)
            mean = pathway_expr.mean(dim=0, keepdim=True)
            std = pathway_expr.std(dim=0, keepdim=True, unbiased=False) + 1e-8
            z_scores = (pathway_expr - mean) / std
            scores[:, p_idx] = z_scores.mean(dim=1)

    return scores if any_found else None


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
) -> torch.Tensor | None:
    """Compute KAC (KRT8+ Alveolar Intermediate Cell) signature score.

    From Nature 2024 Kadara lab: KACs are the key intermediate state in
    AT2 -> LUAD progression. Trajectory: Normal -> AT2 -> AIC -> KAC -> Tumor.
    KACs are found in tumor-adjacent normal tissue BEFORE tumors form,
    making this the critical cell state our transition model should capture.

    Markers: KRT8, CLDN4, CDKN1A, CDKN2A, PLAUR (senescence + invasion)

    Args:
        expression: [B, n_genes] expression matrix
        gene_names: List of gene names corresponding to expression columns
        device: Torch device

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
    # Z-score normalize and mean
    mean = kac_expr.mean(dim=0, keepdim=True)
    std = kac_expr.std(dim=0, keepdim=True, unbiased=False) + 1e-8
    z_scores = (kac_expr - mean) / std
    return z_scores.mean(dim=1, keepdim=True)


def get_pathway_gene_list() -> list[str]:
    """Get all unique genes used in pathway scoring.

    Useful for ensuring these genes are included in processed data.
    """
    genes = set()
    for pathway_genes in PROGENY_PATHWAYS.values():
        genes.update(pathway_genes)
    genes.update(PROLIFERATION_MARKERS)
    genes.update(KAC_MARKERS)
    return sorted(genes)
