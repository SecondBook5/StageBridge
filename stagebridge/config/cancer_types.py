"""Cancer type configurations for StageBridge.

Defines stage progressions, cell type markers, reference atlases, and
known biology for different cancer types.

Currently supported:
- LUAD: Lung adenocarcinoma (default, original StageBridge target)
- PDAC: Pancreatic ductal adenocarcinoma (via PanIN progression)

To add a new cancer type:
1. Create a CancerConfig instance with stage definitions
2. Register it with register_cancer_config()
3. Optionally add reference atlas configuration
4. Add known biological mechanisms for validation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Literal

import yaml


class MechanismType(StrEnum):
    """Categories of biological mechanisms."""
    LIGAND_RECEPTOR = "ligand_receptor"
    CELL_STATE = "cell_state"
    NICHE_COMPOSITION = "niche_composition"
    GENE_PROGRAM = "gene_program"


@dataclass(frozen=True)
class BiologicalMechanism:
    """A known biological mechanism for validation.

    Used to verify model recovers expected biology.
    """
    name: str
    mechanism_type: MechanismType
    expected_stage: str
    markers: tuple[str, ...]
    literature_source: str
    description: str
    priority: int = 1

    def __hash__(self) -> int:
        return hash(self.name)


@dataclass(frozen=True)
class CellTypeMarkers:
    """Cell type marker genes for a cancer type.

    Markers are used for:
    - Cell type annotation validation
    - Known biology validation
    - Niche composition analysis
    """
    # Key cell types for this cancer
    progenitor_markers: tuple[str, ...]  # Cells of origin / progenitors
    malignant_markers: tuple[str, ...]   # Cancer cell markers
    stromal_markers: tuple[str, ...]     # CAF/fibroblast markers
    immune_markers: tuple[str, ...]      # Immune cell markers

    # Optional: specific cell states
    stemness_markers: tuple[str, ...] = ()
    senescence_markers: tuple[str, ...] = ()
    emt_mesenchymal: tuple[str, ...] = ()
    emt_epithelial: tuple[str, ...] = ()


@dataclass
class StageConfig:
    """Stage progression configuration for a cancer type.

    Defines the full stage vocabulary and mappings to coarser systems.
    """
    # Full clinical staging (finest granularity)
    stages_full: tuple[str, ...]

    # 3-stage mapping: (Normal, Preinvasive, Invasive)
    # Maps each full stage to one of these categories
    stages_3_mapping: dict[str, str]

    # Optional 4-stage mapping (for some cancer types)
    stages_4: tuple[str, ...] | None = None
    stages_4_mapping: dict[str, str] | None = None

    # Stage display colors (for visualization)
    stage_colors: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        # Validate mappings
        for stage in self.stages_full:
            if stage not in self.stages_3_mapping:
                raise ValueError(f"Stage '{stage}' not in stages_3_mapping")
            if self.stages_3_mapping[stage] not in ("Normal", "Preinvasive", "Invasive"):
                raise ValueError(
                    f"Stage '{stage}' mapped to invalid 3-stage: {self.stages_3_mapping[stage]}"
                )

    @property
    def stages_3(self) -> tuple[str, ...]:
        """Standard 3-stage system."""
        return ("Normal", "Preinvasive", "Invasive")

    def get_stages(self, system: Literal["3", "4", "5", "full"]) -> tuple[str, ...]:
        """Get stage names for a staging system."""
        if system == "3":
            return self.stages_3
        elif system == "4":
            if self.stages_4 is None:
                raise ValueError("4-stage system not defined for this cancer type")
            return self.stages_4
        elif system in ("5", "full"):
            return self.stages_full
        else:
            raise ValueError(f"Unknown stage system: {system}")

    def convert_to_3(self, stage: str) -> str:
        """Convert any stage to 3-stage system."""
        if stage in self.stages_3:
            return stage
        if stage not in self.stages_3_mapping:
            raise ValueError(f"Unknown stage: {stage}")
        return self.stages_3_mapping[stage]

    def convert_to_4(self, stage: str) -> str:
        """Convert any stage to 4-stage system (if available)."""
        if self.stages_4_mapping is None:
            raise ValueError("4-stage system not defined for this cancer type")
        if stage in self.stages_4:
            return stage
        if stage not in self.stages_4_mapping:
            raise ValueError(f"Unknown stage: {stage}")
        return self.stages_4_mapping[stage]


@dataclass
class ReferenceAtlasConfig:
    """Reference atlas configuration.

    Supports single-reference, dual-reference, or no-reference modes.
    """
    name: str
    description: str

    # Latent space dimensions (fixed by pretrained models)
    latent_dim: int

    # Key for retrieving embeddings from h5ad
    latent_key: str = "X_scanvi_emb"

    # Optional paths (can be set via environment or config)
    model_dir: Path | None = None
    reference_h5ad: Path | None = None

    # Cell type annotation key in reference
    cell_type_key: str = "cell_type"

    def validate_paths(self) -> list[str]:
        """Check if required paths exist."""
        errors = []
        if self.model_dir and not self.model_dir.exists():
            errors.append(f"Model dir not found: {self.model_dir}")
        if self.reference_h5ad and not self.reference_h5ad.exists():
            errors.append(f"Reference h5ad not found: {self.reference_h5ad}")
        return errors


@dataclass
class CancerConfig:
    """Complete configuration for a cancer type.

    This is the main configuration object that bundles:
    - Stage definitions
    - Reference atlases
    - Cell type markers
    - Known biological mechanisms
    """
    name: str
    description: str

    # Stage configuration
    stages: StageConfig

    # Reference atlases (can be empty for no-reference mode)
    references: dict[str, ReferenceAtlasConfig] = field(default_factory=dict)

    # Cell type markers
    cell_markers: CellTypeMarkers | None = None

    # Known biological mechanisms for validation
    known_mechanisms: tuple[BiologicalMechanism, ...] = ()

    # Gene signatures (EMT, senescence, etc. - cancer-specific if needed)
    gene_signatures: dict[str, tuple[str, ...]] = field(default_factory=dict)

    # Default reference mode
    reference_mode: Literal["single", "dual", "none"] = "dual"

    # Primary and secondary references (for dual mode)
    primary_reference: str | None = None
    secondary_reference: str | None = None

    def get_fused_dim(self) -> int:
        """Get total dimension of fused reference embeddings."""
        if not self.references:
            return 0
        if self.reference_mode == "single" and self.primary_reference:
            return self.references[self.primary_reference].latent_dim
        elif self.reference_mode == "dual":
            total = 0
            if self.primary_reference:
                total += self.references[self.primary_reference].latent_dim
            if self.secondary_reference:
                total += self.references[self.secondary_reference].latent_dim
            return total
        return 0

    def get_reference(self, name: str) -> ReferenceAtlasConfig | None:
        """Get a reference atlas configuration by name."""
        return self.references.get(name)

    def validate(self) -> list[str]:
        """Validate configuration completeness."""
        errors = []

        if not self.stages.stages_full:
            errors.append("No stages defined")

        if self.reference_mode == "dual":
            if not self.primary_reference:
                errors.append("Dual reference mode requires primary_reference")
            if not self.secondary_reference:
                errors.append("Dual reference mode requires secondary_reference")
            if self.primary_reference and self.primary_reference not in self.references:
                errors.append(f"Primary reference '{self.primary_reference}' not in references")
            if self.secondary_reference and self.secondary_reference not in self.references:
                errors.append(f"Secondary reference '{self.secondary_reference}' not in references")

        elif self.reference_mode == "single":
            if not self.primary_reference:
                errors.append("Single reference mode requires primary_reference")
            if self.primary_reference and self.primary_reference not in self.references:
                errors.append(f"Primary reference '{self.primary_reference}' not in references")

        return errors


# =============================================================================
# LUAD Configuration (Default)
# =============================================================================

_LUAD_STAGES = StageConfig(
    stages_full=("Normal", "AAH", "AIS", "MIA", "LUAD"),
    stages_3_mapping={
        "Normal": "Normal",
        "AAH": "Preinvasive",
        "AIS": "Preinvasive",
        "MIA": "Invasive",
        "LUAD": "Invasive",
    },
    stages_4=("Normal", "AAH", "AIS", "Invasive"),
    stages_4_mapping={
        "Normal": "Normal",
        "AAH": "AAH",
        "AIS": "AIS",
        "MIA": "Invasive",
        "LUAD": "Invasive",
    },
    stage_colors={
        "Normal": "#228B22",      # Forest green
        "AAH": "#50C878",         # Emerald
        "AIS": "#DAA520",         # Gold
        "MIA": "#CD5C5C",         # Indian red
        "LUAD": "#8B0000",        # Dark red
        "Preinvasive": "#4169E1", # Royal blue
        "Invasive": "#8B0000",    # Dark red
    },
)

_LUAD_MARKERS = CellTypeMarkers(
    progenitor_markers=(
        "KRT5", "KRT17", "SOX9", "TP63",  # KAC/reactive pneumocyte
        "SFTPC", "SFTPA1", "SFTPA2",       # AT2 markers
        "NKX2-1",                           # Lung lineage TF
    ),
    malignant_markers=(
        "EPCAM", "KRT7", "KRT8", "KRT18",   # Epithelial
        "MUC1", "NAPSA",                     # Lung adenocarcinoma
    ),
    stromal_markers=(
        "ACTA2", "COL1A1", "COL3A1", "FAP",  # myCAF
        "IL6", "CXCL12", "PDGFRA",           # iCAF
        "CD74", "HLA-DRA",                    # apCAF
    ),
    immune_markers=(
        "CD3D", "CD3E", "CD3G",   # T cells
        "CD68", "CD163",          # Macrophages
        "CD19", "MS4A1",          # B cells
        "NKG7", "GNLY",           # NK cells
    ),
    stemness_markers=("SOX2", "NANOG", "POU5F1", "KLF4"),
    senescence_markers=("CDKN1A", "CDKN2A", "TP53", "RB1", "SERPINE1", "GLB1"),
    emt_mesenchymal=(
        "VIM", "CDH2", "SNAI1", "SNAI2", "ZEB1", "ZEB2", "TWIST1",
        "FN1", "MMP2", "MMP9", "COL1A1",
    ),
    emt_epithelial=(
        "CDH1", "EPCAM", "KRT8", "KRT18", "KRT19", "CLDN1", "CLDN4",
    ),
)

_LUAD_MECHANISMS = (
    BiologicalMechanism(
        name="IL1B_IL1R1_preinvasive",
        mechanism_type=MechanismType.LIGAND_RECEPTOR,
        expected_stage="Preinvasive",
        markers=("IL1B", "IL1R1"),
        literature_source="Peng et al. 2023, Kadara et al.",
        description="IL1B+ macrophage-epithelial axis enriched in AAH/AIS vs LUAD",
        priority=1,
    ),
    BiologicalMechanism(
        name="KAC_progenitor_expansion",
        mechanism_type=MechanismType.CELL_STATE,
        expected_stage="Preinvasive",
        markers=("KRT5", "KRT17", "SOX9", "TP63"),
        literature_source="Peng et al. 2023",
        description="KAC/reactive pneumocyte progenitors as LUAD predecessors",
        priority=1,
    ),
    BiologicalMechanism(
        name="iCAF_proinflammatory",
        mechanism_type=MechanismType.NICHE_COMPOSITION,
        expected_stage="Preinvasive",
        markers=("IL6", "CXCL12", "PDGFRA"),
        literature_source="CAF literature, Peng et al.",
        description="Inflammatory CAF enrichment in early progression",
        priority=2,
    ),
    BiologicalMechanism(
        name="myCAF_invasive",
        mechanism_type=MechanismType.NICHE_COMPOSITION,
        expected_stage="Invasive",
        markers=("ACTA2", "COL1A1", "COL3A1", "FAP"),
        literature_source="CAF literature",
        description="Myofibroblastic CAF enrichment in invasive disease",
        priority=2,
    ),
    BiologicalMechanism(
        name="EMT_progression",
        mechanism_type=MechanismType.GENE_PROGRAM,
        expected_stage="Invasive",
        markers=("VIM", "CDH2", "SNAI1", "TWIST1", "ZEB1"),
        literature_source="EMT literature",
        description="EMT marker upregulation with progression",
        priority=2,
    ),
    BiologicalMechanism(
        name="EGFR_signaling",
        mechanism_type=MechanismType.LIGAND_RECEPTOR,
        expected_stage="Preinvasive",
        markers=("EGF", "AREG", "EGFR"),
        literature_source="Alcolea et al. 2026",
        description="EGF-SOX9 axis in precancerous niche remodeling",
        priority=2,
    ),
)

LUAD_CONFIG = CancerConfig(
    name="luad",
    description="Lung adenocarcinoma (Normal -> AAH -> AIS -> MIA -> LUAD)",
    stages=_LUAD_STAGES,
    references={
        "hlca": ReferenceAtlasConfig(
            name="hlca",
            description="Human Lung Cell Atlas - healthy reference",
            latent_dim=30,
            latent_key="X_scanvi_emb",
        ),
        "luca": ReferenceAtlasConfig(
            name="luca",
            description="Lung Cancer Atlas - disease reference",
            latent_dim=10,
            latent_key="X_scANVI",
        ),
    },
    cell_markers=_LUAD_MARKERS,
    known_mechanisms=_LUAD_MECHANISMS,
    reference_mode="dual",
    primary_reference="hlca",
    secondary_reference="luca",
)


# =============================================================================
# PDAC Configuration
# =============================================================================

_PDAC_STAGES = StageConfig(
    stages_full=("Normal", "PanIN1", "PanIN2", "PanIN3", "PDAC"),
    stages_3_mapping={
        "Normal": "Normal",
        "PanIN1": "Preinvasive",
        "PanIN2": "Preinvasive",
        "PanIN3": "Preinvasive",
        "PDAC": "Invasive",
    },
    stages_4=("Normal", "LowGrade", "HighGrade", "Invasive"),
    stages_4_mapping={
        "Normal": "Normal",
        "PanIN1": "LowGrade",
        "PanIN2": "LowGrade",
        "PanIN3": "HighGrade",
        "PDAC": "Invasive",
    },
    stage_colors={
        "Normal": "#228B22",       # Forest green
        "PanIN1": "#90EE90",       # Light green
        "PanIN2": "#FFD700",       # Gold
        "PanIN3": "#FFA500",       # Orange
        "PDAC": "#8B0000",         # Dark red
        "Preinvasive": "#4169E1",  # Royal blue
        "Invasive": "#8B0000",     # Dark red
        "LowGrade": "#90EE90",
        "HighGrade": "#FFA500",
    },
)

_PDAC_MARKERS = CellTypeMarkers(
    progenitor_markers=(
        "SOX9", "PDX1", "NKX6-1",    # Pancreatic progenitors
        "KRT19", "MUC1",              # Ductal markers
        "PTF1A", "RBPJL",             # Acinar markers
    ),
    malignant_markers=(
        "EPCAM", "KRT7", "KRT19",     # Epithelial
        "MUC5AC", "MUC1",             # PDAC markers
        "CEACAM5", "CEACAM6",         # CEA family
        "S100A4", "S100P",            # S100 family
    ),
    stromal_markers=(
        "ACTA2", "COL1A1", "COL3A1", "FAP",   # myCAF
        "IL6", "CXCL12", "PDGFRA",            # iCAF
        "HAS1", "HAS2",                        # Hyaluronan synthases
    ),
    immune_markers=(
        "CD3D", "CD3E", "CD3G",   # T cells
        "CD68", "CD163",          # Macrophages (TAMs)
        "CD19", "MS4A1",          # B cells
        "ARG1", "CD206",          # M2 macrophages
    ),
    stemness_markers=("SOX9", "NANOG", "POU5F1", "ALDH1A1", "CD44", "CD133"),
    senescence_markers=("CDKN1A", "CDKN2A", "TP53", "RB1", "SERPINE1"),
    emt_mesenchymal=(
        "VIM", "CDH2", "SNAI1", "ZEB1", "ZEB2", "TWIST1",
        "FN1", "MMP2", "MMP9",
    ),
    emt_epithelial=(
        "CDH1", "EPCAM", "KRT8", "KRT19", "CLDN1", "CLDN4",
    ),
)

_PDAC_MECHANISMS = (
    BiologicalMechanism(
        name="KRAS_activation",
        mechanism_type=MechanismType.GENE_PROGRAM,
        expected_stage="Preinvasive",
        markers=("KRAS", "DUSP6", "ETV4", "ETV5"),
        literature_source="PDAC driver mutation literature",
        description="KRAS-driven MAPK pathway activation in PanIN",
        priority=1,
    ),
    BiologicalMechanism(
        name="acinar_ductal_metaplasia",
        mechanism_type=MechanismType.CELL_STATE,
        expected_stage="Preinvasive",
        markers=("SOX9", "KRT19", "MUC1", "PTF1A"),
        literature_source="ADM literature",
        description="Acinar-to-ductal metaplasia in early PanIN",
        priority=1,
    ),
    BiologicalMechanism(
        name="desmoplastic_stroma",
        mechanism_type=MechanismType.NICHE_COMPOSITION,
        expected_stage="Invasive",
        markers=("ACTA2", "COL1A1", "FAP", "HAS2"),
        literature_source="PDAC stroma literature",
        description="Dense desmoplastic stroma in PDAC",
        priority=1,
    ),
    BiologicalMechanism(
        name="immunosuppressive_TAM",
        mechanism_type=MechanismType.NICHE_COMPOSITION,
        expected_stage="Invasive",
        markers=("CD163", "ARG1", "CD206", "IL10"),
        literature_source="PDAC TME literature",
        description="M2-polarized TAM accumulation",
        priority=2,
    ),
    BiologicalMechanism(
        name="TGFb_signaling",
        mechanism_type=MechanismType.LIGAND_RECEPTOR,
        expected_stage="Invasive",
        markers=("TGFB1", "TGFBR1", "TGFBR2", "SMAD4"),
        literature_source="PDAC signaling literature",
        description="TGFb-driven EMT and stromal activation",
        priority=2,
    ),
    BiologicalMechanism(
        name="neural_invasion",
        mechanism_type=MechanismType.GENE_PROGRAM,
        expected_stage="Invasive",
        markers=("NTRK1", "NGF", "GDNF", "NCAM1"),
        literature_source="PDAC neural invasion literature",
        description="Perineural invasion signature",
        priority=3,
    ),
)

PDAC_CONFIG = CancerConfig(
    name="pdac",
    description="Pancreatic ductal adenocarcinoma (Normal -> PanIN1-3 -> PDAC)",
    stages=_PDAC_STAGES,
    references={
        # PDAC doesn't have canonical HLCA/LuCA equivalents yet
        # Users can add custom references or use single-reference mode
    },
    cell_markers=_PDAC_MARKERS,
    known_mechanisms=_PDAC_MECHANISMS,
    reference_mode="none",  # Can be overridden when reference atlases become available
    primary_reference=None,
    secondary_reference=None,
)


# =============================================================================
# Configuration Registry
# =============================================================================

_CANCER_REGISTRY: dict[str, CancerConfig] = {
    "luad": LUAD_CONFIG,
    "pdac": PDAC_CONFIG,
}


def get_cancer_config(cancer_type: str) -> CancerConfig:
    """Get configuration for a cancer type.

    Args:
        cancer_type: Cancer type name (e.g., "luad", "pdac")

    Returns:
        CancerConfig for the specified cancer type

    Raises:
        ValueError: If cancer type is not registered
    """
    cancer_type = cancer_type.lower()
    if cancer_type not in _CANCER_REGISTRY:
        available = ", ".join(_CANCER_REGISTRY.keys())
        raise ValueError(f"Unknown cancer type: {cancer_type}. Available: {available}")
    return _CANCER_REGISTRY[cancer_type]


def get_available_cancer_types() -> list[str]:
    """Get list of available cancer type configurations."""
    return list(_CANCER_REGISTRY.keys())


def register_cancer_config(config: CancerConfig) -> None:
    """Register a new cancer type configuration.

    Args:
        config: CancerConfig to register

    Raises:
        ValueError: If cancer type name is already registered
    """
    name = config.name.lower()
    if name in _CANCER_REGISTRY:
        raise ValueError(f"Cancer type '{name}' already registered. Use a different name.")
    errors = config.validate()
    if errors:
        raise ValueError(f"Invalid config for {name}: {errors}")
    _CANCER_REGISTRY[name] = config


def load_cancer_config_from_yaml(yaml_path: Path) -> CancerConfig:
    """Load cancer configuration from YAML file.

    Args:
        yaml_path: Path to YAML configuration file

    Returns:
        CancerConfig loaded from file
    """
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    # Parse stages
    stages = StageConfig(
        stages_full=tuple(data["stages"]["full"]),
        stages_3_mapping=data["stages"]["mapping_to_3"],
        stages_4=tuple(data["stages"].get("stages_4", [])) or None,
        stages_4_mapping=data["stages"].get("mapping_to_4"),
        stage_colors=data["stages"].get("colors", {}),
    )

    # Parse references
    references = {}
    for ref_name, ref_data in data.get("references", {}).items():
        references[ref_name] = ReferenceAtlasConfig(
            name=ref_name,
            description=ref_data.get("description", ""),
            latent_dim=ref_data["latent_dim"],
            latent_key=ref_data.get("latent_key", "X_scanvi_emb"),
            model_dir=Path(ref_data["model_dir"]) if ref_data.get("model_dir") else None,
            reference_h5ad=Path(ref_data["reference_h5ad"]) if ref_data.get("reference_h5ad") else None,
        )

    # Parse markers
    markers_data = data.get("cell_markers", {})
    cell_markers = CellTypeMarkers(
        progenitor_markers=tuple(markers_data.get("progenitor", [])),
        malignant_markers=tuple(markers_data.get("malignant", [])),
        stromal_markers=tuple(markers_data.get("stromal", [])),
        immune_markers=tuple(markers_data.get("immune", [])),
        stemness_markers=tuple(markers_data.get("stemness", [])),
        senescence_markers=tuple(markers_data.get("senescence", [])),
        emt_mesenchymal=tuple(markers_data.get("emt_mesenchymal", [])),
        emt_epithelial=tuple(markers_data.get("emt_epithelial", [])),
    ) if markers_data else None

    # Parse mechanisms
    mechanisms = []
    for mech_data in data.get("known_mechanisms", []):
        mechanisms.append(BiologicalMechanism(
            name=mech_data["name"],
            mechanism_type=MechanismType(mech_data["type"]),
            expected_stage=mech_data["expected_stage"],
            markers=tuple(mech_data["markers"]),
            literature_source=mech_data.get("literature_source", ""),
            description=mech_data.get("description", ""),
            priority=mech_data.get("priority", 1),
        ))

    return CancerConfig(
        name=data["name"],
        description=data.get("description", ""),
        stages=stages,
        references=references,
        cell_markers=cell_markers,
        known_mechanisms=tuple(mechanisms),
        gene_signatures=data.get("gene_signatures", {}),
        reference_mode=data.get("reference_mode", "dual"),
        primary_reference=data.get("primary_reference"),
        secondary_reference=data.get("secondary_reference"),
    )
