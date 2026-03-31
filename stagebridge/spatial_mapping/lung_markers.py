"""
Curated marker genes for human lung cell types.

Sources:
- Travaglini et al. 2020 (Nature) - Human Lung Cell Atlas
- Sikkema et al. 2023 (Nature Medicine) - HLCA v2
- Salcher et al. 2022 (Cancer Cell) - Lung cancer atlas
- LungMAP (lungmap.net) - Lung molecular atlas
- PanglaoDB - Validated markers
"""

# Epithelial cells
EPITHELIAL_MARKERS = {
    "AT1": [
        "AGER", "PDPN", "CAV1", "HOPX", "RTKN2", "SPOCK2",
        "EMP2", "CLDN18", "AQP5", "GPRC5A"
    ],
    "AT2": [
        "SFTPC", "SFTPB", "SFTPA1", "SFTPA2", "LAMP3", "ABCA3",
        "LPCAT1", "PGC", "NAPSA", "SFTA2"
    ],
    "Club": [
        "SCGB1A1", "SCGB3A2", "CYP2F1", "LYPD2", "BPIFB1",
        "CCKAR", "CBR2", "RARRES1"
    ],
    "Ciliated": [
        "FOXJ1", "TUBA1A", "DNAH5", "CAPS", "PIFO", "SNTN",
        "RSPH1", "TPPP3", "CCDC153", "LRRC6"
    ],
    "Basal": [
        "KRT5", "KRT14", "KRT17", "TP63", "NGFR", "DLK2",
        "ITGA6", "SOX2", "MYC"
    ],
    "Goblet": [
        "MUC5AC", "MUC5B", "TFF3", "SPDEF", "FCGBP", "AGR2"
    ],
    # Transitional/precursor states
    "Transitional_AT2": [
        "SFTPC", "KRT8", "KRT18", "CLDN4", "AGER",  # Mixed AT2/AT1
        "NAPSA", "HOPX"
    ],
    "KRT5_KRT17": [  # Injury-associated basal-like
        "KRT5", "KRT17", "TP63", "CXCL17", "S100A2"
    ],
}

# Immune cells
IMMUNE_MARKERS = {
    "Macrophages": [
        "CD68", "CD163", "C1QA", "C1QB", "C1QC", "APOE",
        "MARCO", "MSR1", "FCGR3A"
    ],
    "Macrophages_alveolar": [
        "FABP4", "MARCO", "PPARG", "MCEMP1", "CD68"
    ],
    "Macrophages_IL1B_high": [  # Your target population
        "IL1B", "CCL3", "CCL4", "TNF", "CXCL8", "CD68",
        "NFKB1", "S100A8", "S100A9"
    ],
    "Monocytes": [
        "CD14", "FCGR3A", "S100A8", "S100A9", "LYZ", "VCAN"
    ],
    "T_CD4": [
        "CD3D", "CD3E", "CD4", "IL7R", "MAL", "LTB"
    ],
    "T_CD8": [
        "CD3D", "CD3E", "CD8A", "CD8B", "GZMK", "CCL5"
    ],
    "T_reg": [  # Rare but important
        "FOXP3", "IL2RA", "CTLA4", "IKZF2", "CD4", "IL10"
    ],
    "NK": [
        "GNLY", "NKG7", "KLRD1", "KLRF1", "NCAM1", "GZMB"
    ],
    "B_cells": [
        "CD79A", "CD79B", "MS4A1", "CD19", "IGHM", "IGKC"
    ],
    "Plasma_cells": [
        "IGHG1", "IGHG2", "IGHG3", "IGHG4", "JCHAIN", "MZB1", "SDC1"
    ],
    "Dendritic_cDC1": [  # Rare
        "CLEC9A", "XCR1", "BATF3", "IRF8", "CADM1", "BTLA"
    ],
    "Dendritic_cDC2": [  # Rare
        "CD1C", "FCER1A", "CLEC10A", "CD1E"
    ],
    "Dendritic_pDC": [  # Rare
        "LILRA4", "IL3RA", "CLEC4C", "ITM2C", "IRF7", "TCF4"
    ],
    "Mast": [  # Rare
        "TPSAB1", "TPSB2", "CPA3", "KIT", "HDC", "MS4A2"
    ],
}

# Stromal cells
STROMAL_MARKERS = {
    "Fibroblasts": [
        "COL1A1", "COL1A2", "DCN", "LUM", "PDGFRA", "C1R", "C1S"
    ],
    "Myofibroblasts": [
        "ACTA2", "TAGLN", "MYH11", "MYLK", "TPM2", "COL1A1"
    ],
    "CAF_inflammatory": [  # Cancer-associated fibroblasts
        "COL1A1", "IL6", "CXCL12", "CCL2", "PDGFRA",
        "FAP", "S100A4"
    ],
    "Pericytes": [
        "RGS5", "NOTCH3", "PDGFRB", "ACTA2", "NDUFA4L2", "KCNJ8"
    ],
    "Smooth_muscle": [
        "ACTA2", "MYH11", "MYLK", "CNN1", "TAGLN", "TPM2"
    ],
}

# Endothelial cells
ENDOTHELIAL_MARKERS = {
    "Endothelial": [
        "PECAM1", "VWF", "CDH5", "CLDN5", "FLT1", "KDR"
    ],
    "Endothelial_capillary": [
        "PECAM1", "CA4", "GPIHBP1", "PRSS23", "SOSTDC1"
    ],
    "Endothelial_arterial": [
        "PECAM1", "GJA5", "SEMA3G", "BMX", "DLL4"
    ],
    "Endothelial_venous": [
        "PECAM1", "ACKR1", "SELP", "VWF", "NRP2"
    ],
    "Lymphatic": [
        "PROX1", "LYVE1", "FLT4", "PDPN", "CCL21", "TFF3"
    ],
}

# Cancer/malignant markers
CANCER_MARKERS = {
    "Cancer_epithelial": [
        "EPCAM", "KRT8", "KRT18", "KRT19", "CDH1",
        "MUC1", "CEACAM5", "CEACAM6"
    ],
    "Cancer_EMT": [
        "VIM", "SNAI1", "SNAI2", "TWIST1", "ZEB1", "ZEB2",
        "CDH2", "FN1"
    ],
    "Cancer_proliferating": [
        "MKI67", "TOP2A", "PCNA", "CCNB1", "CCNB2", "CDK1"
    ],
}

# Combine all
ALL_LUNG_MARKERS = {
    **EPITHELIAL_MARKERS,
    **IMMUNE_MARKERS,
    **STROMAL_MARKERS,
    **ENDOTHELIAL_MARKERS,
    **CANCER_MARKERS,
}


def get_markers_for_reference(
    reference_cell_types: list[str],
    marker_dict: dict[str, list[str]] | None = None,
) -> dict[str, list[str]]:
    """
    Get markers for specific cell types in your reference.

    Args:
        reference_cell_types: Cell type names in your reference
        marker_dict: Optional custom marker dict (defaults to ALL_LUNG_MARKERS)

    Returns:
        Dict mapping cell type to marker genes

    Example:
        >>> ref_markers = get_markers_for_reference(hlca.obs['cell_type'].unique())
    """
    if marker_dict is None:
        marker_dict = ALL_LUNG_MARKERS

    # Try to match reference cell types to marker dict
    matched_markers = {}

    for ref_ct in reference_cell_types:
        ref_ct_clean = str(ref_ct).strip()

        # Exact match
        if ref_ct_clean in marker_dict:
            matched_markers[ref_ct_clean] = marker_dict[ref_ct_clean]
            continue

        # Fuzzy match (handle spaces, underscores, case)
        ref_ct_norm = ref_ct_clean.lower().replace(" ", "_").replace("-", "_")

        for marker_ct, markers in marker_dict.items():
            marker_ct_norm = marker_ct.lower().replace(" ", "_").replace("-", "_")

            if ref_ct_norm == marker_ct_norm or ref_ct_norm in marker_ct_norm or marker_ct_norm in ref_ct_norm:
                matched_markers[ref_ct_clean] = markers
                break

    # Warn about unmatched
    unmatched = set(reference_cell_types) - set(matched_markers.keys())
    if unmatched:
        print(f"Warning: No markers found for {len(unmatched)} cell types: {list(unmatched)[:5]}...")

    return matched_markers


def get_progression_relevant_markers() -> dict[str, list[str]]:
    """
    Get markers specifically relevant to LUAD progression.

    Based on Peng et al. and Kadara lab work on IL1B-IL1R1 axis.

    Returns:
        Dict with progression-relevant marker sets
    """
    return {
        # Core progression markers
        "IL1B_high_macrophages": IMMUNE_MARKERS["Macrophages_IL1B_high"],
        "Tregs": IMMUNE_MARKERS["T_reg"],
        "AT2_cells": EPITHELIAL_MARKERS["AT2"],
        "Transitional_AT2": EPITHELIAL_MARKERS["Transitional_AT2"],
        "Inflammatory_CAF": STROMAL_MARKERS["CAF_inflammatory"],

        # EMT markers (progression)
        "EMT_signature": CANCER_MARKERS["Cancer_EMT"],
        "Proliferation_signature": CANCER_MARKERS["Cancer_proliferating"],
    }


# Cell cycle genes (from Tirosh et al. 2016 Science)
S_PHASE_GENES = [
    "MCM5", "PCNA", "TYMS", "FEN1", "MCM2", "MCM4", "RRM1", "UNG",
    "GINS2", "MCM6", "CDCA7", "DTL", "PRIM1", "UHRF1", "MLF1IP",
    "HELLS", "RFC2", "RPA2", "NASP", "RAD51AP1", "GMNN", "WDR76",
    "SLBP", "CCNE2", "UBR7", "POLD3", "MSH2", "ATAD2", "RAD51",
    "RRM2", "CDC45", "CDC6", "EXO1", "TIPIN", "DSCC1", "BLM",
    "CASP8AP2", "USP1", "CLSPN", "POLA1", "CHAF1B", "BRIP1", "E2F8"
]

G2M_PHASE_GENES = [
    "HMGB2", "CDK1", "NUSAP1", "UBE2C", "BIRC5", "TPX2", "TOP2A",
    "NDC80", "CKS2", "NUF2", "CKS1B", "MKI67", "TMPO", "CENPF",
    "TACC3", "FAM64A", "SMC4", "CCNB2", "CKAP2L", "CKAP2", "AURKB",
    "BUB1", "KIF11", "ANP32E", "TUBB4B", "GTSE1", "KIF20B", "HJURP",
    "CDCA3", "HN1", "CDC20", "TTK", "CDC25C", "KIF2C", "RANGAP1",
    "NCAPD2", "DLGAP5", "CDCA2", "CDCA8", "ECT2", "KIF23", "HMMR",
    "AURKA", "PSRC1", "ANLN", "LBR", "CKAP5", "CENPE", "CTCF",
    "NEK2", "G2E3", "GAS2L3", "CBX5", "CENPA"
]


def add_cycling_cell_states(
    adata,
    s_genes: list[str] | None = None,
    g2m_genes: list[str] | None = None,
    score_threshold: float = 0.2,
    cell_type_key: str = "cell_type",
) -> None:
    """
    Add cycling cell states to AnnData.

    Modifies adata in place:
    - Adds 'S_score' and 'G2M_score' to .obs
    - Adds 'cell_state' to .obs with cycling variants

    Args:
        adata: AnnData object
        s_genes: S phase genes (defaults to Tirosh et al.)
        g2m_genes: G2M phase genes (defaults to Tirosh et al.)
        score_threshold: Threshold for calling cycling (default: 0.2)
        cell_type_key: Column with cell type annotations

    Example:
        >>> import scanpy as sc
        >>> adata = sc.read_h5ad("reference.h5ad")
        >>> add_cycling_cell_states(adata)
        >>> # Now adata.obs has 'cell_state' with e.g. "T_CD8_cycling"
    """
    if s_genes is None:
        s_genes = S_PHASE_GENES
    if g2m_genes is None:
        g2m_genes = G2M_PHASE_GENES

    # Score cell cycle
    import scanpy as sc
    sc.tl.score_genes_cell_cycle(
        adata,
        s_genes=s_genes,
        g2m_genes=g2m_genes,
    )

    # Create cell_state column
    adata.obs['cell_state'] = adata.obs[cell_type_key].astype(str)

    # Mark cycling cells
    cycling_mask = (adata.obs['S_score'] > score_threshold) | (adata.obs['G2M_score'] > score_threshold)
    adata.obs.loc[cycling_mask, 'cell_state'] = adata.obs.loc[cycling_mask, cell_type_key].astype(str) + '_cycling'

    n_cycling = cycling_mask.sum()
    print(f"Identified {n_cycling} cycling cells ({100*n_cycling/len(adata):.1f}%)")
    print("Added 'cell_state' column with cycling variants")
