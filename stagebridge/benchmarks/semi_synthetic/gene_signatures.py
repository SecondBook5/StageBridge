"""
Gene signatures for semi-synthetic benchmark expression perturbation.

Defines biologically-motivated gene programs that are activated when
specific interaction rules are triggered. These create recoverable
ground-truth signals in the expression data.

References:
- EMT signatures from MSigDB Hallmark
- IL1B response genes from KEGG/Reactome
- CAF-induced signatures from Peng et al., Lambrechts et al.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class GeneSignature:
    """A gene signature with positive and negative markers."""

    name: str
    description: str
    positive_genes: list[str]  # Genes upregulated by this effect
    negative_genes: list[str]  # Genes downregulated by this effect
    effect_scale: float = 1.0  # Magnitude multiplier

    def get_all_genes(self) -> set[str]:
        """Get all genes in the signature."""
        return set(self.positive_genes) | set(self.negative_genes)


# =============================================================================
# BIOLOGICALLY-MOTIVATED GENE SIGNATURES
# =============================================================================

# EMT (Epithelial-Mesenchymal Transition) signature
# Key genes from MSigDB HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION
EMT_SIGNATURE = GeneSignature(
    name="EMT",
    description="Epithelial-mesenchymal transition program",
    positive_genes=[
        # Mesenchymal markers (upregulated in EMT)
        "VIM", "FN1", "CDH2", "SNAI1", "SNAI2", "ZEB1", "ZEB2",
        "TWIST1", "TWIST2", "MMP2", "MMP9", "SPARC", "COL1A1",
        "COL3A1", "ACTA2", "TAGLN", "S100A4", "LOX", "LOXL2",
        "SERPINE1", "TGFB1", "TGFBI", "ITGA5", "ITGB1", "THY1",
        "PDGFRB", "FAP", "POSTN", "COMP", "BGN", "DCN",
    ],
    negative_genes=[
        # Epithelial markers (downregulated in EMT)
        "CDH1", "EPCAM", "KRT8", "KRT18", "KRT19", "OCLN", "TJP1",
        "CLDN1", "CLDN3", "CLDN4", "CLDN7", "MUC1", "DSP", "PKP3",
        "ESRP1", "ESRP2", "GRHL2",
    ],
    effect_scale=1.0,
)

# IL1B inflammatory response signature
# Key genes from IL1 signaling pathway
IL1B_RESPONSE_SIGNATURE = GeneSignature(
    name="IL1B_response",
    description="IL1B-mediated inflammatory response in receivers",
    positive_genes=[
        # IL1 response genes
        "IL1R1", "IL1RAP", "MYD88", "IRAK1", "IRAK4", "TRAF6",
        "NFKB1", "NFKB2", "RELA", "RELB", "CXCL1", "CXCL2",
        "CXCL3", "CXCL8", "CCL2", "CCL3", "CCL4", "CCL20",
        "IL6", "IL8", "PTGS2", "ICAM1", "VCAM1", "SELE",
        "MMP1", "MMP3", "SOD2", "NOS2", "TNFAIP3", "BIRC3",
        "BCL2A1", "IRF1", "STAT1", "STAT3", "JUN", "FOS",
    ],
    negative_genes=[
        # Anti-inflammatory / homeostatic genes
        "IL10", "TGFB1", "IL1RN", "SOCS1", "SOCS3", "DUSP1",
    ],
    effect_scale=0.8,
)

# CAF-induced signature (Cancer-Associated Fibroblast effects)
CAF_INDUCED_SIGNATURE = GeneSignature(
    name="CAF_induced",
    description="CAF-secreted factor response in epithelial cells",
    positive_genes=[
        # Growth factors and their targets
        "EGFR", "ERBB2", "ERBB3", "MET", "AXL", "IGF1R",
        "FGFR1", "FGFR2", "PDGFRA", "KIT",
        # Downstream proliferation
        "MYC", "CCND1", "CCND2", "CDK4", "CDK6", "E2F1",
        "MCM2", "MCM7", "PCNA", "MKI67", "TOP2A",
        # Survival
        "BCL2", "BCL2L1", "MCL1", "BIRC5", "XIAP",
        # Migration/invasion
        "ROCK1", "ROCK2", "RAC1", "CDC42", "RHOA",
    ],
    negative_genes=[
        # Tumor suppressors / differentiation
        "CDKN1A", "CDKN1B", "CDKN2A", "RB1", "TP53",
        "PTEN", "APC", "SMAD4",
    ],
    effect_scale=0.9,
)

# Hypoxia response signature
HYPOXIA_SIGNATURE = GeneSignature(
    name="hypoxia",
    description="HIF1A-mediated hypoxia response",
    positive_genes=[
        # Core hypoxia response
        "HIF1A", "EPAS1", "ARNT", "VEGFA", "VEGFB", "VEGFC",
        "SLC2A1", "SLC2A3", "HK1", "HK2", "PFKFB3", "LDHA",
        "PDK1", "BNIP3", "BNIP3L", "CA9", "ADM", "ANGPTL4",
        "LOX", "P4HA1", "P4HA2", "PLOD2", "ENO1", "ENO2",
        "ALDOA", "PGK1", "GAPDH", "TPI1",
    ],
    negative_genes=[
        # Oxidative phosphorylation
        "COX4I1", "COX5A", "COX6A1", "NDUFA1", "NDUFB1",
        "ATP5F1A", "ATP5F1B",
    ],
    effect_scale=0.7,
)

# Proliferation signature
PROLIFERATION_SIGNATURE = GeneSignature(
    name="proliferation",
    description="Cell cycle and proliferation program",
    positive_genes=[
        # G1/S
        "CCND1", "CCND2", "CCNE1", "CCNE2", "CDK2", "CDK4", "CDK6",
        "E2F1", "E2F2", "RB1", "MCM2", "MCM3", "MCM4", "MCM5",
        "MCM6", "MCM7", "CDC6", "CDC45", "ORC1",
        # S phase
        "PCNA", "RFC1", "RFC2", "RFC3", "RFC4", "RFC5",
        "POLA1", "POLD1", "POLE", "RPA1", "RPA2",
        # G2/M
        "CCNA2", "CCNB1", "CCNB2", "CDK1", "CDC25A", "CDC25B",
        "CDC25C", "PLK1", "AURKA", "AURKB", "BUB1", "BUB1B",
        "MAD2L1", "CDC20", "PTTG1",
        # General proliferation markers
        "MKI67", "TOP2A", "BIRC5", "UBE2C", "NUSAP1", "TPX2",
    ],
    negative_genes=[
        # Cell cycle inhibitors
        "CDKN1A", "CDKN1B", "CDKN2A", "CDKN2B", "GADD45A",
    ],
    effect_scale=0.8,
)

# Apoptosis resistance signature
APOPTOSIS_RESISTANCE_SIGNATURE = GeneSignature(
    name="apoptosis_resistance",
    description="Anti-apoptotic survival program",
    positive_genes=[
        # Anti-apoptotic
        "BCL2", "BCL2L1", "BCL2L2", "MCL1", "BIRC2", "BIRC3",
        "BIRC5", "XIAP", "CFLAR", "TNFAIP3", "NFKB1", "RELA",
        # Survival signaling
        "AKT1", "AKT2", "PIK3CA", "PIK3CB", "MTOR", "RPS6KB1",
        "SGK1", "IRS1", "IRS2",
    ],
    negative_genes=[
        # Pro-apoptotic
        "BAX", "BAK1", "BID", "BIM", "PUMA", "NOXA", "BAD",
        "CASP3", "CASP7", "CASP8", "CASP9", "APAF1", "CYCS",
        "DIABLO", "FADD", "FAS", "TNFRSF10A", "TNFRSF10B",
    ],
    effect_scale=0.6,
)


# =============================================================================
# EFFECT NAME -> SIGNATURE MAPPING
# =============================================================================

EFFECT_SIGNATURES: dict[str, list[GeneSignature]] = {
    # CAF-induced EMT uses both EMT and CAF signatures
    "CAF_induced_EMT": [EMT_SIGNATURE, CAF_INDUCED_SIGNATURE],

    # Immune modulation uses IL1B response
    "immune_modulation": [IL1B_RESPONSE_SIGNATURE],

    # Distant CAF effect is weaker, just CAF signature
    "CAF_distant_effect": [CAF_INDUCED_SIGNATURE],

    # Additional effects that can be used
    "hypoxia_response": [HYPOXIA_SIGNATURE],
    "proliferation_induction": [PROLIFERATION_SIGNATURE],
    "survival_signaling": [APOPTOSIS_RESISTANCE_SIGNATURE],
    "inflammatory_response": [IL1B_RESPONSE_SIGNATURE],
    "emt_induction": [EMT_SIGNATURE],
}


def get_signatures_for_effect(effect_name: str) -> list[GeneSignature]:
    """Get gene signatures associated with an effect name."""
    # Try exact match first
    if effect_name in EFFECT_SIGNATURES:
        return EFFECT_SIGNATURES[effect_name]

    # Try case-insensitive match
    effect_lower = effect_name.lower()
    for key, sigs in EFFECT_SIGNATURES.items():
        if key.lower() == effect_lower:
            return sigs

    # Try partial match
    for key, sigs in EFFECT_SIGNATURES.items():
        if effect_lower in key.lower() or key.lower() in effect_lower:
            return sigs

    # Default to EMT if no match (most common effect)
    return [EMT_SIGNATURE]


def get_all_signature_genes() -> set[str]:
    """Get all genes used in any signature."""
    all_genes = set()
    for sigs in EFFECT_SIGNATURES.values():
        for sig in sigs:
            all_genes |= sig.get_all_genes()
    return all_genes


@dataclass
class ExpressionPerturbation:
    """Represents a perturbation to be applied to expression."""

    cell_idx: int
    gene_perturbations: dict[str, float]  # gene_name -> delta
    total_magnitude: float
    triggered_by: str  # rule_id
    effect_name: str
    distance: float
    stage: str | None


def compute_perturbation(
    effect_name: str,
    effect_strength: float,
    distance: float,
    interaction_radius: float,
    stage: str | None = None,
    stage_modulation: dict[str, float] | None = None,
    available_genes: set[str] | None = None,
    noise_scale: float = 0.1,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """
    Compute gene-level perturbations for an interaction effect.

    Args:
        effect_name: Name of the effect (maps to signatures)
        effect_strength: Base strength of the effect (0-1)
        distance: Distance from sender to receiver
        interaction_radius: Maximum interaction radius
        stage: Disease stage for stage-dependent modulation
        stage_modulation: Stage -> multiplier mapping
        available_genes: Set of genes available in the expression matrix
        noise_scale: Scale of random noise added to perturbations
        rng: Random number generator

    Returns:
        Dictionary mapping gene names to perturbation values
    """
    if rng is None:
        rng = np.random.default_rng()

    # Get signatures for this effect
    signatures = get_signatures_for_effect(effect_name)

    # Compute distance decay (exponential)
    # At distance=0, decay=1. At distance=radius, decay~0.37
    distance_decay = np.exp(-distance / interaction_radius)

    # Apply stage modulation
    stage_mult = 1.0
    if stage and stage_modulation:
        stage_mult = stage_modulation.get(stage, 1.0)

    # Final magnitude
    magnitude = effect_strength * distance_decay * stage_mult

    # Compute gene perturbations
    perturbations: dict[str, float] = {}

    for sig in signatures:
        sig_magnitude = magnitude * sig.effect_scale

        # Positive genes (upregulated)
        for gene in sig.positive_genes:
            if available_genes is None or gene in available_genes:
                # Add some noise for realism
                noise = rng.normal(0, noise_scale * sig_magnitude)
                delta = sig_magnitude + noise
                perturbations[gene] = perturbations.get(gene, 0) + delta

        # Negative genes (downregulated)
        for gene in sig.negative_genes:
            if available_genes is None or gene in available_genes:
                noise = rng.normal(0, noise_scale * sig_magnitude)
                delta = -(sig_magnitude + noise)
                perturbations[gene] = perturbations.get(gene, 0) + delta

    return perturbations
