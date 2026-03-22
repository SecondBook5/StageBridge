"""
Curated gene signatures for biological interpretation.

Signatures are derived from:
- MSigDB Hallmark gene sets
- Published lung cancer studies (Peng et al., Laughney et al.)
- HLCA cell type markers
- Cancer biology literature

Each signature is a dict with:
- genes: list of gene symbols
- description: biological meaning
- source: publication/database reference
"""

from typing import Any
import numpy as np
import pandas as pd

# =============================================================================
# EMT (Epithelial-Mesenchymal Transition) Signatures
# =============================================================================

EMT_SIGNATURES = {
    "emt_hallmark": {
        "genes": [
            "VIM", "CDH2", "SNAI1", "SNAI2", "ZEB1", "ZEB2", "TWIST1", "TWIST2",
            "FN1", "MMP2", "MMP9", "SPARC", "COL1A1", "COL3A1", "ACTA2",
            "TAGLN", "CNN1", "SERPINE1", "ITGB1", "ITGA5",
        ],
        "description": "Hallmark EMT genes - mesenchymal markers",
        "source": "MSigDB Hallmark EMT",
    },
    "epithelial_markers": {
        "genes": [
            "CDH1", "EPCAM", "KRT8", "KRT18", "KRT19", "CLDN1", "CLDN3",
            "CLDN4", "CLDN7", "OCLN", "TJP1", "DSP", "PKP1", "JUP",
        ],
        "description": "Epithelial markers - lost during EMT",
        "source": "Literature consensus",
    },
    "partial_emt": {
        "genes": [
            "VIM", "CDH1", "ZEB1", "SNAI2", "KRT8", "KRT18", "EPCAM",
            "OVOL1", "OVOL2", "GRHL2", "ELF3", "ELF5",
        ],
        "description": "Partial/hybrid EMT state markers",
        "source": "Pastushenko & Blanpain 2019",
    },
}

# =============================================================================
# CAF (Cancer-Associated Fibroblast) Signatures
# =============================================================================

CAF_SIGNATURES = {
    "caf_general": {
        "genes": [
            "FAP", "PDPN", "ACTA2", "PDGFRA", "PDGFRB", "COL1A1", "COL1A2",
            "COL3A1", "FN1", "VIM", "THY1", "DCN", "LUM", "POSTN",
        ],
        "description": "General CAF markers",
        "source": "Sahai et al. 2020 Nature Reviews Cancer",
    },
    "caf_inflammatory": {
        "genes": [
            "IL6", "IL11", "CXCL1", "CXCL2", "CXCL12", "CCL2", "LIF",
            "PDGFA", "HAS1", "HAS2", "CFD", "LMNA",
        ],
        "description": "Inflammatory CAF (iCAF) markers",
        "source": "Ohlund et al. 2017",
    },
    "caf_myofibroblastic": {
        "genes": [
            "ACTA2", "TAGLN", "MYL9", "TPM1", "TPM2", "CNN1", "MYLK",
            "ACTG2", "MYH11", "LMOD1", "SYNPO2",
        ],
        "description": "Myofibroblastic CAF (myCAF) markers",
        "source": "Ohlund et al. 2017",
    },
    "caf_antigen_presenting": {
        "genes": [
            "CD74", "HLA-DRA", "HLA-DRB1", "HLA-DPA1", "HLA-DPB1",
            "HLA-DQA1", "HLA-DQB1", "CIITA", "SLPI",
        ],
        "description": "Antigen-presenting CAF (apCAF) markers",
        "source": "Elyada et al. 2019",
    },
}

# =============================================================================
# Immune Signatures
# =============================================================================

IMMUNE_SIGNATURES = {
    "t_cell_exhaustion": {
        "genes": [
            "PDCD1", "LAG3", "HAVCR2", "TIGIT", "CTLA4", "TOX", "TOX2",
            "EOMES", "PRDM1", "BATF", "IRF4", "NR4A1", "NR4A2", "NR4A3",
        ],
        "description": "T cell exhaustion markers",
        "source": "Wherry & Kurachi 2015",
    },
    "t_cell_cytotoxic": {
        "genes": [
            "GZMA", "GZMB", "GZMK", "PRF1", "IFNG", "TNF", "FASLG",
            "NKG7", "GNLY", "CTSW", "CST7", "CCL5",
        ],
        "description": "Cytotoxic T cell effector genes",
        "source": "Literature consensus",
    },
    "treg": {
        "genes": [
            "FOXP3", "IL2RA", "CTLA4", "TIGIT", "IKZF2", "TNFRSF18",
            "TNFRSF4", "CCR8", "BATF", "IL10", "TGFB1",
        ],
        "description": "Regulatory T cell markers",
        "source": "Zheng et al. 2017",
    },
    "macrophage_m1": {
        "genes": [
            "CD80", "CD86", "IL1B", "IL6", "TNF", "NOS2", "CXCL9",
            "CXCL10", "CXCL11", "IDO1", "SOCS1", "IRF5",
        ],
        "description": "M1 (pro-inflammatory) macrophage markers",
        "source": "Murray et al. 2014",
    },
    "macrophage_m2": {
        "genes": [
            "CD163", "MRC1", "CD206", "ARG1", "IL10", "TGFB1", "CCL18",
            "CCL22", "VEGFA", "MMP9", "MARCO", "MSR1",
        ],
        "description": "M2 (anti-inflammatory/pro-tumor) macrophage markers",
        "source": "Murray et al. 2014",
    },
    "il1b_macrophage": {
        "genes": [
            "IL1B", "CCL2", "CCL3", "CCL4", "CXCL2", "CXCL3", "CXCL8",
            "NLRP3", "CASP1", "PYCARD", "S100A8", "S100A9",
        ],
        "description": "IL1B+ inflammatory macrophage signature (Peng et al.)",
        "source": "Peng et al. 2020 Cancer Cell",
    },
    "neutrophil": {
        "genes": [
            "S100A8", "S100A9", "S100A12", "CXCR2", "FCGR3B", "CSF3R",
            "SELL", "CXCL8", "MMP8", "MMP9", "ELANE", "LCN2",
        ],
        "description": "Neutrophil markers",
        "source": "Literature consensus",
    },
}

# =============================================================================
# Lung Cancer-Specific Signatures
# =============================================================================

LUNG_CANCER_SIGNATURES = {
    "at2_markers": {
        "genes": [
            "SFTPC", "SFTPB", "SFTPA1", "SFTPA2", "SFTPD", "ABCA3",
            "LAMP3", "NAPSA", "NKX2-1", "ETV5", "CLDN18", "HOPX",
        ],
        "description": "Alveolar type 2 (AT2) cell markers - LUAD cell of origin",
        "source": "HLCA + Travaglini et al. 2020",
    },
    "at1_markers": {
        "genes": [
            "AGER", "PDPN", "CAV1", "AQP5", "HOPX", "CLIC5", "RTKN2",
        ],
        "description": "Alveolar type 1 (AT1) cell markers",
        "source": "HLCA + Travaglini et al. 2020",
    },
    "kac_intermediate": {
        "genes": [
            "KRT8", "CEACAM5", "CEACAM6", "MUC1", "MSLN", "CD24",
            "CLDN4", "TACSTD2", "EPCAM", "KRT7",
        ],
        "description": "KRT8+/CEACAM5+ alveolar intermediate cells (Peng et al.)",
        "source": "Peng et al. 2020 Cancer Cell",
    },
    "club_cell": {
        "genes": [
            "SCGB1A1", "SCGB3A1", "SCGB3A2", "CYP2F1", "BPIFB1",
        ],
        "description": "Club cell markers",
        "source": "HLCA",
    },
    "nfkb_pathway": {
        "genes": [
            "RELA", "RELB", "NFKB1", "NFKB2", "REL", "NFKBIA", "NFKBIB",
            "IKBKB", "IKBKG", "CHUK", "BCL2", "BCL2L1", "BIRC3",
        ],
        "description": "NF-kB pathway genes (dominant in precursor stages)",
        "source": "Peng et al. 2020 Cancer Cell",
    },
    "wnt_pathway": {
        "genes": [
            "WNT3A", "WNT5A", "WNT7A", "CTNNB1", "APC", "AXIN1", "AXIN2",
            "GSK3B", "TCF7L2", "LEF1", "MYC", "CCND1",
        ],
        "description": "WNT pathway genes",
        "source": "KEGG + Literature",
    },
    "notch_pathway": {
        "genes": [
            "NOTCH1", "NOTCH2", "NOTCH3", "JAG1", "JAG2", "DLL1", "DLL4",
            "HES1", "HEY1", "HEY2", "RBPJ", "MAML1",
        ],
        "description": "NOTCH pathway genes",
        "source": "KEGG + Literature",
    },
}

# =============================================================================
# Proliferation Signature
# =============================================================================

PROLIFERATION_SIGNATURE = {
    "proliferation": {
        "genes": [
            "MKI67", "TOP2A", "PCNA", "MCM2", "MCM3", "MCM4", "MCM5",
            "MCM6", "MCM7", "CCNA2", "CCNB1", "CCNB2", "CCNE1", "CCNE2",
            "CDK1", "CDK2", "AURKA", "AURKB", "PLK1", "BUB1", "BUB1B",
        ],
        "description": "Cell proliferation / cell cycle markers",
        "source": "Literature consensus",
    },
}

# =============================================================================
# Combined Signature Dictionary
# =============================================================================

GENE_SIGNATURES = {
    **EMT_SIGNATURES,
    **CAF_SIGNATURES,
    **IMMUNE_SIGNATURES,
    **LUNG_CANCER_SIGNATURES,
    **PROLIFERATION_SIGNATURE,
}


# =============================================================================
# Scoring Functions
# =============================================================================

def get_signature_genes(signature_name: str) -> list[str]:
    """Get genes for a named signature."""
    if signature_name not in GENE_SIGNATURES:
        raise ValueError(f"Unknown signature: {signature_name}. Available: {list(GENE_SIGNATURES.keys())}")
    return GENE_SIGNATURES[signature_name]["genes"]


def score_signature(
    adata: Any,
    signature_name: str,
    layer: str | None = None,
    use_raw: bool = False,
) -> np.ndarray:
    """
    Score cells for a gene signature using mean z-score method.

    Parameters
    ----------
    adata : AnnData
        Gene expression data
    signature_name : str
        Name of signature to score
    layer : str, optional
        Layer to use for expression
    use_raw : bool
        Use raw counts

    Returns
    -------
    ndarray
        Signature scores per cell (n_cells,)
    """
    import scipy.sparse as sp

    genes = get_signature_genes(signature_name)

    # Get expression matrix
    if use_raw and adata.raw is not None:
        X = adata.raw.X
        var_names = list(adata.raw.var_names)
    elif layer and layer in adata.layers:
        X = adata.layers[layer]
        var_names = list(adata.var_names)
    else:
        X = adata.X
        var_names = list(adata.var_names)

    # Find overlapping genes
    gene_idx = [var_names.index(g) for g in genes if g in var_names]
    [genes[i] for i, g in enumerate(genes) if g in var_names]

    if len(gene_idx) == 0:
        return np.zeros(adata.n_obs)

    if len(gene_idx) < len(genes) * 0.3:
        import warnings
        warnings.warn(
            f"Only {len(gene_idx)}/{len(genes)} genes found for {signature_name}. "
            f"Missing: {[g for g in genes if g not in var_names][:5]}...",
            stacklevel=2,
        )

    # Extract expression
    if sp.issparse(X):
        expr = np.asarray(X[:, gene_idx].todense())
    else:
        expr = X[:, gene_idx]

    # Z-score normalization per gene
    expr_mean = expr.mean(axis=0, keepdims=True)
    expr_std = expr.std(axis=0, keepdims=True) + 1e-10
    z_scores = (expr - expr_mean) / expr_std

    # Mean z-score across genes
    scores = z_scores.mean(axis=1)

    return np.asarray(scores).ravel()


def score_all_signatures(
    adata: Any,
    signatures: dict[str, dict] | None = None,
    layer: str | None = None,
    use_raw: bool = False,
    add_to_obs: bool = True,
    prefix: str = "sig_",
) -> pd.DataFrame:
    """
    Score cells for all signatures.

    Parameters
    ----------
    adata : AnnData
        Gene expression data
    signatures : dict, optional
        Signature dict (default: GENE_SIGNATURES)
    layer : str, optional
        Layer to use
    use_raw : bool
        Use raw counts
    add_to_obs : bool
        Add scores to adata.obs
    prefix : str
        Prefix for column names

    Returns
    -------
    DataFrame
        Signature scores (n_cells, n_signatures)
    """
    if signatures is None:
        signatures = GENE_SIGNATURES

    scores = {}
    for sig_name in signatures:
        scores[sig_name] = score_signature(adata, sig_name, layer=layer, use_raw=use_raw)

    df = pd.DataFrame(scores, index=adata.obs_names)

    if add_to_obs:
        for col in df.columns:
            adata.obs[f"{prefix}{col}"] = df[col].values

    return df
