"""Canonical data contract for StageBridge v1.

THE SINGLE SOURCE OF TRUTH for all constants, schemas, and validators.
All other code MUST import from here. No duplication allowed.

Canonical artifacts:
- cells.parquet: Cell-level features with fused embeddings
- neighborhoods.parquet: 9-token niche structure per cell
- split_manifest.json: Donor-held-out CV splits
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

# =============================================================================
# STAGE DEFINITIONS
# =============================================================================

# Full clinical staging
STAGES_5 = ("Normal", "AAH", "AIS", "MIA", "LUAD")

# 4-stage: keeps AAH separate (intervention window), merges invasive
STAGES_4 = ("Normal", "AAH", "AIS", "Invasive")

# 3-stage: preinvasive vs invasive
STAGES_3 = ("Normal", "Preinvasive", "Invasive")

# Mappings from 5-stage to coarser systems
STAGE_5_TO_4 = {
    "Normal": "Normal",
    "AAH": "AAH",
    "AIS": "AIS",
    "MIA": "Invasive",
    "LUAD": "Invasive",
}

STAGE_5_TO_3 = {
    "Normal": "Normal",
    "AAH": "Preinvasive",
    "AIS": "Preinvasive",
    "MIA": "Invasive",
    "LUAD": "Invasive",
}

_STAGE_SYSTEMS = {
    "3": STAGES_3,
    "4": STAGES_4,
    "5": STAGES_5,
}


def get_stage_system(
    system: Literal["3", "4", "5"] = "3",
) -> tuple[tuple[str, ...], dict[str, int], dict[int, str]]:
    """Get stage names and mappings for a stage system.

    Args:
        system: "3", "4", or "5" stage system

    Returns:
        (stage_names, stage_to_index, index_to_stage)
    """
    stages = _STAGE_SYSTEMS[system]
    s2i = {s: i for i, s in enumerate(stages)}
    i2s = {i: s for i, s in enumerate(stages)}
    return stages, s2i, i2s


def convert_stage(stage: str, to_system: Literal["3", "4"]) -> str:
    """Convert a 5-stage name to a coarser system."""
    mapping = STAGE_5_TO_3 if to_system == "3" else STAGE_5_TO_4
    if stage not in mapping:
        raise ValueError(f"Unknown stage '{stage}'. Expected one of {STAGES_5}")
    return mapping[stage]


# Default 3-stage (better donor coverage for cross-stage learning)
STAGES, STAGE_TO_IDX, IDX_TO_STAGE = get_stage_system("3")
N_STAGES = len(STAGES)


# =============================================================================
# LATENT DIMENSIONS & FUSION
# =============================================================================

HLCA_DIM = 30
LUCA_DIM = 10

# Fusion methods for dual-reference embeddings
# Valid options for use_gw_fusion / gw_fusion_type:
#   - False / "concat": Simple concatenation [HLCA; LuCA] -> 40d
#   - "learned_projection": Learned weighted projection (fallback)
#   - "pretrained": Precomputed GW alignment with neural transport map (recommended)
GW_FUSION_TYPES = ("concat", "learned_projection", "pretrained")

# Legacy fusion methods (deprecated - use GW_FUSION_TYPES)
FUSION_METHODS = ("concat", "weighted", "gated", "film")


def get_fused_dim(method: str = "concat") -> int:
    """Get output dimension for a fusion method.

    Args:
        method:
            - "concat": Simple concatenation [HLCA; LuCA] -> 40d
            - "learned_projection": Learned weighted projection -> configurable
            - "pretrained": Precomputed GW with neural map -> configurable
            - "weighted"/"gated"/"film": Legacy methods -> 30d

    Returns:
        Output dimension of fused embedding
    """
    if method in ("concat", "learned_projection", "pretrained"):
        return HLCA_DIM + LUCA_DIM  # 40 (default, can be overridden by gw_output_dim)
    elif method in ("weighted", "gated", "film"):
        return HLCA_DIM  # 30 (project to larger space)
    else:
        raise ValueError(f"Unknown fusion method '{method}'. Expected one of {GW_FUSION_TYPES} or {FUSION_METHODS}")


# Default: concat (simplest, proven to work)
LATENT_DIM = get_fused_dim("concat")  # 40d


# =============================================================================
# NICHE TOKEN STRUCTURE (9 tokens)
# =============================================================================

N_TOKENS = 9
TOKEN_NAMES = (
    "receiver",  # 0: Central cell
    "ring1",     # 1: Innermost spatial ring
    "ring2",     # 2
    "ring3",     # 3
    "ring4",     # 4: Outermost spatial ring
    "hlca",      # 5: HLCA reference embedding
    "luca",      # 6: LuCA reference embedding
    "pathway",   # 7: Pathway activity features
    "stats",     # 8: Summary statistics
)

# Token type IDs for type embeddings
TOKEN_TYPE_IDS = {
    "receiver": 0,
    "spatial": 1,   # ring1-4
    "hlca": 2,
    "luca": 3,
    "pathway": 4,
    "stats": 5,
}


# =============================================================================
# WES (WHOLE EXOME SEQUENCING) FEATURES
# =============================================================================

WES_COLS = (
    # Core binary mutation flags
    "tmb",
    "kras_mut",
    "egfr_mut",
    "tp53_mut",
    "stk11_mut",
    "keap1_mut",
    "smad4_mut",
    "braf_mut",
    # Specific hotspot variants (for OncoKB actionability)
    "egfr_L858R",
    "egfr_exon19del",
    "egfr_T790M",
    "kras_G12C",
    "kras_G12V",
    # OncoKB actionability
    "has_level1_mutation",
    "has_actionable_mutation",
)
WES_DIM = len(WES_COLS)  # 17 features


# =============================================================================
# CLONAL FEATURES (from inferCNV)
# =============================================================================

# Cell-level clonal features
CLONAL_CELL_COLS = (
    "cnv_score",           # CNV burden per cell
    "cnv_score_z",         # Z-scored within patient
    "clone_size",          # Number of cells in this clone
    "clone_rank",          # Rank within patient (0=largest)
    "is_major_clone",      # Binary: is largest clone
    "clone_fraction",      # Fraction of patient cells in this clone
)

# Patient-level clonal features (broadcasted to cells)
CLONAL_PATIENT_COLS = (
    "n_clones",                  # Total clones in patient
    "clonal_entropy",            # Shannon entropy of clone distribution
    "clonal_diversity",          # Gini-Simpson diversity
    "clonal_pattern_idx",        # 0=1a (new invasive), 1=1b (shared), 2=uncategorized
    "aneuploidy_score",          # Overall aneuploidy burden
    "clone_sharing_ratio",       # Fraction of clones shared precursor<->invasive
    "has_invasive_only_clones",  # Binary: new clones in invasive
)

CLONAL_COLS = CLONAL_CELL_COLS + CLONAL_PATIENT_COLS
CLONAL_DIM = len(CLONAL_COLS)  # 13 features


# =============================================================================
# EVOLUTION BRANCH (combined WES + clonal)
# =============================================================================

EVOLUTION_COLS = WES_COLS + CLONAL_COLS
EVOLUTION_DIM = len(EVOLUTION_COLS)  # 30 features


# =============================================================================
# PATHWAY FEATURES (in token 7)
# =============================================================================

PATHWAY_FEATURES = ("emt_score", "caf_fraction", "immune_fraction", "il1b_score")
N_PROGENY_PATHWAYS = 14  # PROGENy canonical pathways

PROGENY_PATHWAY_NAMES = (
    "Androgen", "EGFR", "Estrogen", "Hypoxia", "JAK-STAT",
    "MAPK", "NFkB", "PI3K", "TGFb", "TNFa",
    "Trail", "VEGF", "WNT", "p53",
)
N_PROGENY_PATHWAYS = 14

EXTENDED_PATHWAY_NAMES = PROGENY_PATHWAY_NAMES + ("cGAS_STING",)
N_EXTENDED_PATHWAYS = 15

KI67_COLUMN = "ki67_positive"

# =============================================================================
# BIOLOGICAL SIGNAL DEFINITIONS
# =============================================================================

CELL_CYCLE_PHASES = ("G1", "S", "G2M")
CELL_CYCLE_COLUMNS = ("cell_cycle_phase", "S_score", "G2M_score")

SENESCENCE_MARKERS = (
    "CDKN1A",  # p21
    "CDKN2A",  # p16
    "TP53",
    "RB1",
    "SERPINE1",  # PAI-1
    "IL6",
    "IL8",
    "MMP3",
)
SENESCENCE_SCORE_COLUMN = "senescence_score"

CAF_SUBTYPES = (
    "myCAF",      # Myofibroblastic CAF (ACTA2+, COL1A1+)
    "iCAF",       # Inflammatory CAF (IL6+, CXCL12+)
    "apCAF",      # Antigen-presenting CAF (CD74+, HLA-DR+)
)
CAF_SCORE_COLUMNS = ("myCAF_score", "iCAF_score", "apCAF_score")

PLASTICITY_MARKERS = (
    "epithelial_score",    # E-cadherin, EpCAM
    "mesenchymal_score",   # Vimentin, N-cadherin
    "emt_score",           # Composite EMT
    "stemness_score",      # SOX2, NANOG, OCT4
)

CLONAL_COLUMNS = (
    "clone_id",
    "subclone_id",
    "clone_size",
    "driver_mutations",
)

LIANA_LR_COLUMNS = (
    "liana_score",         # Aggregate L-R interaction score
    "lr_pairs_detected",   # Number of significant L-R pairs
    "top_lr_pair",         # Most significant L-R pair name
)


# =============================================================================
# STATS TOKEN (CONDITIONING INPUTS)
# =============================================================================
# These biological signals are INPUTS to the model (conditioning), NOT targets.
# This avoids circular validation - we control for these effects rather than
# predicting them. See memory/design_conditioning_vs_encoding.md.
#
# NOTE: Do NOT include il1b_raw, kac_raw, or other validation targets here.
# Those are what we're trying to discover, not control for.

STATS_TOKEN_COLUMNS = (
    # Niche composition (from DestVI deconvolution)
    "caf_fraction",
    "immune_fraction",
    "diversity",
    # Cell cycle (needed to identify rare cycling KAC progenitors)
    "S_score",
    "G2M_score",
)

STATS_TOKEN_DIM = len(STATS_TOKEN_COLUMNS)  # 5 features


# =============================================================================
# DATA TYPES
# =============================================================================

DATA_TYPES = ("snrna", "spatial")


# =============================================================================
# REQUIRED COLUMNS
# =============================================================================

# Core required columns (always needed)
CELLS_REQUIRED_COLS = ("cell_id", "donor_id", "stage")

# Alternative formats for neighborhoods:
# Format A: Pre-assembled tokens column (list/array of 9 token embeddings)
# Format B: Separate columns for each component (ring_1_cells, receiver_z, etc.)
NEIGHBORHOODS_REQUIRED_COLS = ("cell_id", "donor_id")  # Minimal required
NEIGHBORHOODS_TOKENS_COL = "tokens"  # Format A
NEIGHBORHOODS_RING_COLS = ("ring_1_cells", "ring_2_cells", "ring_3_cells", "ring_4_cells")  # Format B


# =============================================================================
# CONTRACT VALIDATION
# =============================================================================

@dataclass
class ContractViolation:
    """A specific contract violation."""
    severity: Literal["error", "warning"]
    category: str
    message: str
    details: dict | None = None


class ContractValidator:
    """Validates data artifacts against the canonical contract.

    Usage:
        validator = ContractValidator(data_dir)
        validator.validate_all()  # Raises on errors

        # Or specific artifacts
        validator.validate_cells()
        validator.validate_neighborhoods()
        validator.validate_splits()
    """

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.violations: list[ContractViolation] = []

    def _error(self, category: str, message: str, details: dict | None = None):
        self.violations.append(ContractViolation("error", category, message, details))

    def _warning(self, category: str, message: str, details: dict | None = None):
        self.violations.append(ContractViolation("warning", category, message, details))

    def validate_all(self, raise_on_error: bool = True) -> list[ContractViolation]:
        """Validate all canonical artifacts."""
        self.violations = []

        cells_path = self.data_dir / "cells.parquet"
        neighborhoods_path = self.data_dir / "neighborhoods.parquet"
        splits_path = self.data_dir / "split_manifest.json"

        if not cells_path.exists():
            self._error("files", f"cells.parquet not found at {cells_path}")
        if not neighborhoods_path.exists():
            self._error("files", f"neighborhoods.parquet not found at {neighborhoods_path}")
        if not splits_path.exists():
            self._error("files", f"split_manifest.json not found at {splits_path}")

        errors = [v for v in self.violations if v.severity == "error"]
        if errors and raise_on_error:
            raise ValueError(f"Contract violations: {[e.message for e in errors]}")

        if cells_path.exists():
            self.validate_cells()
        if neighborhoods_path.exists():
            self.validate_neighborhoods()
        if splits_path.exists():
            self.validate_splits()

        if cells_path.exists() and neighborhoods_path.exists():
            self._validate_alignment()

        errors = [v for v in self.violations if v.severity == "error"]
        if errors and raise_on_error:
            msgs = "\n".join([f"  - [{e.category}] {e.message}" for e in errors])
            raise ValueError(f"Contract violations:\n{msgs}")

        return self.violations

    def validate_cells(self) -> pd.DataFrame:
        """Validate cells.parquet."""
        cells = pd.read_parquet(self.data_dir / "cells.parquet")

        for col in CELLS_REQUIRED_COLS:
            if col not in cells.columns:
                self._error("cells", f"Missing required column: {col}")

        if "stage" in cells.columns:
            unique = set(cells["stage"].unique())
            valid = set(STAGES_3) | set(STAGES_5)
            invalid = unique - valid
            if invalid:
                self._error("cells", f"Invalid stages: {invalid}")

        if "data_type" in cells.columns:
            invalid = set(cells["data_type"].unique()) - set(DATA_TYPES)
            if invalid:
                self._error("cells", f"Invalid data_type: {invalid}")

        # Check fused embeddings - accept either format:
        # Format A: z_fused_0, z_fused_1, ..., z_fused_39 (individual columns)
        # Format B: z_fused (single column with array/list)
        fused_cols = [f"z_fused_{i}" for i in range(LATENT_DIM)]
        has_individual = all(c in cells.columns for c in fused_cols)
        has_single = "z_fused" in cells.columns
        if not has_individual and not has_single:
            self._error("cells", f"Missing fused embedding: need either z_fused or z_fused_0..z_fused_{LATENT_DIM-1}")

        # WES columns are optional - partial is fine (missing will be zero-filled at runtime)

        for col in ["cell_id", "donor_id", "stage"]:
            if col in cells.columns and cells[col].isna().any():
                self._error("cells", f"NaN values in {col}")

        return cells

    def validate_neighborhoods(self) -> pd.DataFrame:
        """Validate neighborhoods.parquet."""
        neighborhoods = pd.read_parquet(self.data_dir / "neighborhoods.parquet")

        for col in NEIGHBORHOODS_REQUIRED_COLS:
            if col not in neighborhoods.columns:
                self._error("neighborhoods", f"Missing required column: {col}")

        # Check token structure - accept either format:
        # Format A: tokens column (pre-assembled array of 9 token embeddings)
        # Format B: separate columns (ring_1_cells, ring_2_cells, ..., receiver_z, hlca_z, luca_z)
        has_tokens_col = NEIGHBORHOODS_TOKENS_COL in neighborhoods.columns
        has_ring_cols = all(c in neighborhoods.columns for c in NEIGHBORHOODS_RING_COLS)

        if not has_tokens_col and not has_ring_cols:
            self._error("neighborhoods", f"Missing token structure: need either 'tokens' column or ring columns {NEIGHBORHOODS_RING_COLS}")

        if has_tokens_col:
            n_check = min(100, len(neighborhoods))
            indices = np.random.choice(len(neighborhoods), n_check, replace=False)

            for idx in indices:
                tokens = neighborhoods.iloc[idx][NEIGHBORHOODS_TOKENS_COL]
                if not isinstance(tokens, (list, np.ndarray)):
                    self._error("neighborhoods", f"Row {idx}: tokens not list/array")
                    continue
                if len(tokens) != N_TOKENS:
                    self._warning("neighborhoods", f"Row {idx}: {len(tokens)} tokens, expected {N_TOKENS}")

        return neighborhoods

    def validate_splits(self) -> dict:
        """Validate split_manifest.json."""
        with open(self.data_dir / "split_manifest.json") as f:
            splits = json.load(f)

        if "folds" not in splits:
            self._error("splits", "Missing 'folds' key")
            return splits

        for i, fold in enumerate(splits["folds"]):
            for key in ["train_donors", "val_donors", "test_donors"]:
                if key not in fold:
                    self._error("splits", f"Fold {i} missing '{key}'")

            if all(k in fold for k in ["train_donors", "val_donors", "test_donors"]):
                train = set(fold["train_donors"])
                val = set(fold["val_donors"])
                test = set(fold["test_donors"])

                if train & val:
                    self._error("splits", f"Fold {i}: train/val overlap (LEAKAGE)")
                if train & test:
                    self._error("splits", f"Fold {i}: train/test overlap (LEAKAGE)")
                if val & test:
                    self._error("splits", f"Fold {i}: val/test overlap (LEAKAGE)")

        return splits

    def _validate_alignment(self):
        """Validate cells and neighborhoods alignment."""
        cells = pd.read_parquet(self.data_dir / "cells.parquet")
        neighborhoods = pd.read_parquet(self.data_dir / "neighborhoods.parquet")

        cell_ids = set(cells["cell_id"])
        neighborhood_ids = set(neighborhoods["cell_id"])

        orphans = neighborhood_ids - cell_ids
        if orphans:
            self._error("alignment", f"{len(orphans)} neighborhoods without cells")

        # Check that spatial cells have neighborhoods
        # Determine spatial cells based on available data
        if "data_type" in cells.columns:
            spatial = set(cells[cells["data_type"] == "spatial"]["cell_id"])
        elif "x_spatial" in cells.columns:
            # If x_spatial exists, cells with non-null coordinates are spatial
            spatial = set(cells[cells["x_spatial"].notna()]["cell_id"])
        else:
            # Can't determine spatial cells - skip this check
            spatial = set()

        if spatial:
            spatial_without = spatial - neighborhood_ids
            if spatial_without:
                # Only warn if it's a small number (could be edge cases)
                if len(spatial_without) <= 10:
                    self._warning("alignment", f"{len(spatial_without)} spatial cells without neighborhoods")
                else:
                    self._error("alignment", f"{len(spatial_without)} spatial cells without neighborhoods")


def validate_contract(data_dir: str | Path) -> list[ContractViolation]:
    """Validate canonical contract. Raises on errors."""
    return ContractValidator(data_dir).validate_all(raise_on_error=True)


# =============================================================================
# STAGE UTILITIES
# =============================================================================

def stage_to_idx(stage: str, system: Literal["3", "5"] = "3") -> int:
    """Get numeric index for a stage name."""
    _, s2i, _ = get_stage_system(system)
    if stage not in s2i:
        raise ValueError(f"Invalid stage '{stage}' for {system}-stage system")
    return s2i[stage]


def idx_to_stage(idx: int, system: Literal["3", "5"] = "3") -> str:
    """Get stage name from numeric index."""
    _, _, i2s = get_stage_system(system)
    if idx not in i2s:
        raise ValueError(f"Invalid index {idx} for {system}-stage system")
    return i2s[idx]


def convert_5_to_3(stage: str) -> str:
    """Convert 5-stage to 3-stage."""
    if stage not in STAGE_5_TO_3:
        raise ValueError(f"Invalid 5-stage: {stage}")
    return STAGE_5_TO_3[stage]


# =============================================================================
# PIPELINE OUTPUT CONTRACTS
# =============================================================================

@dataclass
class TrainingOutputContract:
    """Contract for training run outputs.

    Required directory structure:
        {run_dir}/
        ├── config.yaml           # Full Hydra config
        ├── training_summary.json # Metrics summary
        ├── checkpoints/
        │   ├── best.pt          # Best validation checkpoint
        │   └── final.pt         # Final epoch checkpoint
        └── logs/
            └── metrics.json     # Per-epoch metrics
    """

    REQUIRED_FILES: tuple[str, ...] = (
        "config.yaml",
        "training_summary.json",
        "checkpoints/best.pt",
    )

    SUMMARY_REQUIRED_KEYS: tuple[str, ...] = (
        "best_epoch",
        "best_val_loss",
        "total_epochs",
        "model_config",
        "trainer_config",
    )

    CHECKPOINT_REQUIRED_KEYS: tuple[str, ...] = (
        "epoch",
        "model_state_dict",
        "config",
    )

    @classmethod
    def validate(cls, run_dir: Path) -> list[str]:
        """Return list of contract violations."""
        errors = []
        run_dir = Path(run_dir)

        for f in cls.REQUIRED_FILES:
            if not (run_dir / f).exists():
                errors.append(f"Missing: {f}")

        summary_path = run_dir / "training_summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)
            for key in cls.SUMMARY_REQUIRED_KEYS:
                if key not in summary:
                    errors.append(f"Summary missing key: {key}")

        return errors


@dataclass
class InferenceOutputContract:
    """Contract for inference/prediction outputs.

    Required file: predictions.parquet
    """

    REQUIRED_COLS: tuple[str, ...] = (
        "cell_id",
        "donor_id",
        "source_stage",
        "target_stage",
        "predicted_embedding",  # List[float] of LATENT_DIM
        "context_gate",         # float in [0, 1], interpretability
    )

    OPTIONAL_COLS: tuple[str, ...] = (
        "source_embedding",
        "attention_weights",
        "neighbor_importance",
    )

    @classmethod
    def validate(cls, df: pd.DataFrame) -> list[str]:
        """Return list of contract violations."""
        errors = []

        for col in cls.REQUIRED_COLS:
            if col not in df.columns:
                errors.append(f"Missing column: {col}")

        if "predicted_embedding" in df.columns and len(df) > 0:
            sample = df["predicted_embedding"].iloc[0]
            if hasattr(sample, "__len__") and len(sample) != LATENT_DIM:
                errors.append(f"predicted_embedding dim {len(sample)} != {LATENT_DIM}")

        if "context_gate" in df.columns:
            gates = df["context_gate"]
            if gates.min() < -0.01 or gates.max() > 1.01:
                errors.append(f"context_gate outside [0,1]: [{gates.min():.3f}, {gates.max():.3f}]")

        return errors


@dataclass
class EvaluationOutputContract:
    """Contract for evaluation metrics output.

    Required file: evaluation.json
    """

    REQUIRED_METRICS: tuple[str, ...] = (
        "wasserstein_distance",
        "mean_displacement",
        "stage_accuracy",
    )

    REQUIRED_METADATA: tuple[str, ...] = (
        "model_name",
        "checkpoint_path",
        "fold_idx",
        "n_samples",
        "evaluated_at",
    )

    @classmethod
    def validate(cls, metrics: dict) -> list[str]:
        """Return list of contract violations."""
        errors = []

        for key in cls.REQUIRED_METRICS:
            if key not in metrics:
                errors.append(f"Missing metric: {key}")

        for key in cls.REQUIRED_METADATA:
            if key not in metrics.get("metadata", {}):
                errors.append(f"Missing metadata: {key}")

        return errors


@dataclass
class AblationOutputContract:
    """Contract for ablation study outputs."""

    ABLATION_TYPES: tuple[str, ...] = (
        "no_niche",        # Zero out niche context
        "no_distance",     # Remove distance encoding
        "no_gate",         # Fix gate=1 (always use context)
        "random_niche",    # Randomize neighbor assignment
        "hlca_only",       # Single reference: HLCA
        "luca_only",       # Single reference: LuCA
        "frozen_encoder",  # Freeze SSL encoder, train only transition head
        "no_ring_pooling", # Mean pooling instead of ISAB+PMA
        "no_context_refiner",  # Remove HierarchicalSetTransformer
        "no_gw_fusion",    # Concat only (no GW)
        "gw_learned",      # Learned projection (no precomputed GW)
        "no_evolution",    # Remove WES/clonal evolution features
    )

    REQUIRED_KEYS: tuple[str, ...] = (
        "ablation_type",
        "metrics",
        "delta_vs_full",
    )

    @classmethod
    def validate(cls, result: dict) -> list[str]:
        """Return list of contract violations."""
        errors = []

        for key in cls.REQUIRED_KEYS:
            if key not in result:
                errors.append(f"Missing: {key}")

        if "ablation_type" in result:
            if result["ablation_type"] not in cls.ABLATION_TYPES:
                errors.append(f"Unknown ablation: {result['ablation_type']}")

        return errors


@dataclass
class BaselineOutputContract:
    """Contract for baseline comparison outputs."""

    BASELINE_NAMES: tuple[str, ...] = (
        "pooling_mlp",
        "deepsets",
        "set_transformer",
        "graphsage",
    )

    REQUIRED_KEYS: tuple[str, ...] = (
        "baseline_name",
        "metrics",
        "n_parameters",
        "training_time_seconds",
    )

    @classmethod
    def validate(cls, result: dict) -> list[str]:
        """Return list of contract violations."""
        errors = []

        for key in cls.REQUIRED_KEYS:
            if key not in result:
                errors.append(f"Missing: {key}")

        if "baseline_name" in result:
            if result["baseline_name"] not in cls.BASELINE_NAMES:
                errors.append(f"Unknown baseline: {result['baseline_name']}")

        return errors


def assert_contract(errors: list[str], stage_name: str) -> None:
    """Raise AssertionError if any contract violations."""
    if errors:
        msg = f"CONTRACT VIOLATION in {stage_name}:\n" + "\n".join(f"  - {e}" for e in errors)
        raise AssertionError(msg)


# =============================================================================
# EXPLICIT PARQUET COLUMN SCHEMAS
# =============================================================================

@dataclass
class ColumnSchema:
    """Schema for a single column."""
    name: str
    dtype: Literal["str", "int", "float", "bool", "list", "object"]
    required: bool = True
    nullable: bool = False
    description: str = ""


@dataclass
class ParquetSchema:
    """Schema for a parquet file with typed columns."""
    name: str
    description: str
    columns: list[ColumnSchema]

    def required_columns(self) -> list[str]:
        return [c.name for c in self.columns if c.required]

    def validate(self, df: pd.DataFrame) -> list[str]:
        """Validate DataFrame, return list of errors."""
        errors = []
        for col in self.columns:
            if col.required and col.name not in df.columns:
                errors.append(f"Missing required column: {col.name}")
            if col.name in df.columns and not col.nullable:
                if df[col.name].isna().any():
                    errors.append(f"Column '{col.name}' has nulls but nullable=False")
        return errors


# -----------------------------------------------------------------------------
# cells.parquet schema
# -----------------------------------------------------------------------------

CELLS_SCHEMA = ParquetSchema(
    name="cells.parquet",
    description="Cell-level features with fused dual-reference embeddings",
    columns=[
        # Identifiers (required)
        ColumnSchema("cell_id", "str", required=True, nullable=False, description="Unique cell identifier"),
        ColumnSchema("donor_id", "str", required=True, nullable=False, description="Donor/patient identifier"),
        ColumnSchema("stage", "str", required=True, nullable=False, description="Disease stage (Normal/Preinvasive/Invasive or 5-stage)"),
        ColumnSchema("data_type", "str", required=True, nullable=False, description="snrna or spatial"),
        # Fused embeddings (required - 40 columns z_fused_0..z_fused_39)
        *[ColumnSchema(f"z_fused_{i}", "float", required=True, nullable=False, description=f"Fused embedding dim {i}") for i in range(LATENT_DIM)],
        # HLCA embeddings (optional - 30 columns)
        *[ColumnSchema(f"z_hlca_{i}", "float", required=False, nullable=True, description=f"HLCA embedding dim {i}") for i in range(HLCA_DIM)],
        # LuCA embeddings (optional - 10 columns)
        *[ColumnSchema(f"z_luca_{i}", "float", required=False, nullable=True, description=f"LuCA embedding dim {i}") for i in range(LUCA_DIM)],
        # Spatial coordinates (required for spatial data_type)
        ColumnSchema("x", "float", required=False, nullable=True, description="Spatial X coordinate"),
        ColumnSchema("y", "float", required=False, nullable=True, description="Spatial Y coordinate"),
        # Cell type annotations
        ColumnSchema("cell_type_hlca", "str", required=False, nullable=True, description="Cell type from HLCA reference"),
        ColumnSchema("cell_type_luca", "str", required=False, nullable=True, description="Cell type from LuCA reference"),
        # WES features (optional)
        *[ColumnSchema(col, "float", required=False, nullable=True, description=f"WES feature: {col}") for col in WES_COLS],
        # Pathway features
        *[ColumnSchema(col, "float", required=False, nullable=True, description=f"Pathway feature: {col}") for col in PATHWAY_FEATURES],
        # Stats token features (conditioning inputs)
        *[ColumnSchema(col, "float", required=False, nullable=True, description=f"Stats token feature: {col}") for col in STATS_TOKEN_COLUMNS],
        # Sample metadata
        ColumnSchema("sample_id", "str", required=False, nullable=True, description="Sample identifier"),
        ColumnSchema("lesion_id", "str", required=False, nullable=True, description="Lesion identifier"),
    ],
)


# -----------------------------------------------------------------------------
# neighborhoods.parquet schema
# Supports two formats:
#   1. AMICI format (PREFERRED): neighbor_cells + neighbor_distances for continuous attention
#   2. Ring format (LEGACY): ring_N_cells for discrete ring binning
# The dataloader auto-detects format based on column presence.
# -----------------------------------------------------------------------------

NEIGHBORHOODS_SCHEMA = ParquetSchema(
    name="neighborhoods.parquet",
    description="Niche data for receiver-centered modeling. Supports AMICI (continuous) or ring (discrete) format.",
    columns=[
        # Common required columns
        ColumnSchema("cell_id", "str", required=True, nullable=False, description="Cell identifier (matches cells.parquet)"),
        ColumnSchema("donor_id", "str", required=True, nullable=False, description="Donor identifier"),
        ColumnSchema("stage", "str", required=True, nullable=False, description="Disease stage"),
        ColumnSchema("receiver_z", "list", required=True, nullable=False, description="Receiver fused embedding [LATENT_DIM]"),
        ColumnSchema("hlca_z", "list", required=True, nullable=False, description="HLCA reference embedding [HLCA_DIM=30]"),
        ColumnSchema("luca_z", "list", required=True, nullable=False, description="LuCA reference embedding [LUCA_DIM=10]"),

        # AMICI format (PREFERRED): continuous distance attention
        ColumnSchema("neighbor_cells", "list", required=False, nullable=True, description="AMICI: List of neighbor embeddings sorted by distance"),
        ColumnSchema("neighbor_distances", "list", required=False, nullable=True, description="AMICI: Distances to neighbors in microns"),

        # Ring format (LEGACY): discrete spatial bins
        ColumnSchema("ring_1_cells", "list", required=False, nullable=True, description="Ring: Cell embeddings in ring 1 (0-50um)"),
        ColumnSchema("ring_2_cells", "list", required=False, nullable=True, description="Ring: Cell embeddings in ring 2 (50-100um)"),
        ColumnSchema("ring_3_cells", "list", required=False, nullable=True, description="Ring: Cell embeddings in ring 3 (100-150um)"),
        ColumnSchema("ring_4_cells", "list", required=False, nullable=True, description="Ring: Cell embeddings in ring 4 (150-200um)"),
        ColumnSchema("ring_1_distances", "list", required=False, nullable=True, description="Ring: Distances to ring 1 cells"),
        ColumnSchema("ring_2_distances", "list", required=False, nullable=True, description="Ring: Distances to ring 2 cells"),
        ColumnSchema("ring_3_distances", "list", required=False, nullable=True, description="Ring: Distances to ring 3 cells"),
        ColumnSchema("ring_4_distances", "list", required=False, nullable=True, description="Ring: Distances to ring 4 cells"),

        # Optional features
        ColumnSchema("pathway_z", "list", required=False, nullable=True, description="Pathway features [LATENT_DIM=40]"),
        ColumnSchema("stats_z", "list", required=False, nullable=True, description="Stats features [STATS_TOKEN_DIM=5]"),
    ],
)

# Maximum cells per ring (ring format)
MAX_CELLS_PER_RING = 50

# Maximum neighbors (AMICI format)
MAX_NEIGHBORS = 100

# Ring distance boundaries in microns (ring format)
RING_BOUNDARIES = (0, 50, 100, 150, 200)  # ring 1: 0-50um, ring 2: 50-100um, etc.

# Legacy schema alias (deprecated - use NEIGHBORHOODS_SCHEMA)
NEIGHBORHOODS_LEGACY_SCHEMA = ParquetSchema(
    name="neighborhoods_legacy.parquet",
    description="DEPRECATED: Pre-pooled 9-token structure. Use NEIGHBORHOODS_SCHEMA instead.",
    columns=[
        ColumnSchema("cell_id", "str", required=True, nullable=False, description="Cell identifier"),
        ColumnSchema("donor_id", "str", required=True, nullable=False, description="Donor identifier"),
        ColumnSchema("tokens", "list", required=True, nullable=False, description="List of 9 pre-pooled token embeddings"),
    ],
)


# -----------------------------------------------------------------------------
# transition_scores.parquet schema (StageBridge output)
# -----------------------------------------------------------------------------

TRANSITION_SCORES_SCHEMA = ParquetSchema(
    name="transition_scores.parquet",
    description="StageBridge model output: transition probabilities per cell/spot",
    columns=[
        ColumnSchema("cell_id", "str", required=True, nullable=False, description="Cell identifier"),
        ColumnSchema("donor_id", "str", required=True, nullable=False, description="Donor identifier"),
        ColumnSchema("sample_id", "str", required=False, nullable=True, description="Sample identifier"),
        ColumnSchema("transition_score", "float", required=True, nullable=False, description="Transition probability [0,1]"),
        ColumnSchema("source_stage", "str", required=True, nullable=False, description="Current/source stage"),
        ColumnSchema("target_stage", "str", required=True, nullable=False, description="Predicted target stage"),
        # Model outputs
        ColumnSchema("context_gate", "float", required=False, nullable=True, description="Niche context gate value [0,1]"),
        ColumnSchema("uncertainty", "float", required=False, nullable=True, description="Prediction uncertainty"),
        # Optional embeddings (stored as lists)
        ColumnSchema("predicted_embedding", "list", required=False, nullable=True, description="Predicted embedding (LATENT_DIM)"),
        ColumnSchema("source_embedding", "list", required=False, nullable=True, description="Source cell embedding"),
        # Spatial info
        ColumnSchema("x", "float", required=False, nullable=True, description="Spatial X coordinate"),
        ColumnSchema("y", "float", required=False, nullable=True, description="Spatial Y coordinate"),
    ],
)


# -----------------------------------------------------------------------------
# Schema registry
# -----------------------------------------------------------------------------

PARQUET_SCHEMAS = {
    "cells": CELLS_SCHEMA,
    "neighborhoods": NEIGHBORHOODS_SCHEMA,
    "transition_scores": TRANSITION_SCORES_SCHEMA,
}


def validate_parquet(df: pd.DataFrame, schema_name: str) -> list[str]:
    """Validate a DataFrame against a named schema."""
    if schema_name not in PARQUET_SCHEMAS:
        raise ValueError(f"Unknown schema: {schema_name}. Available: {list(PARQUET_SCHEMAS.keys())}")
    return PARQUET_SCHEMAS[schema_name].validate(df)


def print_schema_markdown(schema: ParquetSchema) -> str:
    """Generate markdown documentation for a schema."""
    lines = [f"## {schema.name}", "", schema.description, ""]
    lines.append("| Column | Type | Required | Nullable | Description |")
    lines.append("|--------|------|----------|----------|-------------|")
    for col in schema.columns:
        req = "Yes" if col.required else "No"
        null = "Yes" if col.nullable else "No"
        lines.append(f"| {col.name} | {col.dtype} | {req} | {null} | {col.description} |")
    return "\n".join(lines)
