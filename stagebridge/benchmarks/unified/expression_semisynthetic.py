"""
Expression-aware semi-synthetic benchmark generator.

Proper semi-synthetic protocol with expression-level ground truth:
1. Load real scRNA-seq expression profiles
2. Subcluster each cell type into "interacting" vs "non-interacting" pools
3. Generate synthetic spatial layout
4. Assign cells from interacting pool if sender within radius, else non-interacting
5. Ground truth = DE genes between subclusters + interaction labels

This creates expression-level ground truth that models must recover,
not just binary labels. The key insight is that interaction status must
be reflected in actual expression differences, not just labels.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy import stats

from stagebridge.logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class InteractionSpec:
    """Specification for a sender->receiver interaction."""

    sender_celltype: str
    receiver_celltype: str
    interaction_radius: float  # in spatial units (e.g., micrometers)
    interaction_name: str

    # Stage modulation (optional) - affects both interaction probability AND effect size
    stage_weights: dict[str, float] | None = None

    # Associated pathways that should be activated in interacting cells
    associated_pathways: list[str] = field(default_factory=list)

    # Stage-specific DE effect sizes (multiplier on base effect)
    # e.g., {"AAH": 1.5, "AIS": 1.2, "LUAD": 0.6} means stronger effect in early stages
    stage_effect_sizes: dict[str, float] | None = None

    def get_stage_weight(self, stage: str) -> float:
        if self.stage_weights is None:
            return 1.0
        return self.stage_weights.get(stage, 1.0)

    def get_stage_effect_size(self, stage: str) -> float:
        """Get DE effect size multiplier for a stage."""
        if self.stage_effect_sizes is None:
            return 1.0
        return self.stage_effect_sizes.get(stage, 1.0)


@dataclass
class SubclusterPool:
    """Pool of cells from a subcluster with expression profiles."""

    celltype: str
    subcluster: str  # "interacting" or "noninteracting"
    cell_ids: np.ndarray
    expression: np.ndarray  # (n_cells, n_genes)
    metadata: pd.DataFrame  # cell-level metadata

    @property
    def n_cells(self) -> int:
        return len(self.cell_ids)


@dataclass
class DEGeneSet:
    """Differentially expressed genes between interacting/non-interacting subclusters."""

    celltype: str
    interaction_name: str
    upregulated_genes: list[str]  # Genes UP in interacting
    downregulated_genes: list[str]  # Genes DOWN in interacting
    effect_sizes: dict[str, float]  # gene -> log2FC
    pvalues: dict[str, float]  # gene -> adjusted p-value

    def to_dict(self) -> dict[str, Any]:
        return {
            "celltype": self.celltype,
            "interaction_name": self.interaction_name,
            "n_upregulated": len(self.upregulated_genes),
            "n_downregulated": len(self.downregulated_genes),
            "upregulated_genes": self.upregulated_genes[:50],  # Top 50
            "downregulated_genes": self.downregulated_genes[:50],
            "effect_sizes": {k: float(v) for k, v in list(self.effect_sizes.items())[:100]},
        }


# Cancer-relevant pathways (from PROGENy)
PROGENY_PATHWAYS = [
    "EGFR", "MAPK", "PI3K", "TGFb", "NFkB",
    "TNFa", "JAK-STAT", "VEGF", "Hypoxia", "WNT",
]

# Pathway gene signatures (simplified - top markers per pathway)
# In real usage, these come from PROGENy database
PATHWAY_SIGNATURES = {
    "EGFR": ["EGFR", "ERBB2", "ERBB3", "EGF", "AREG", "HBEGF"],
    "MAPK": ["MAPK1", "MAPK3", "MAP2K1", "RAF1", "BRAF", "KRAS"],
    "PI3K": ["PIK3CA", "PIK3CB", "AKT1", "PTEN", "MTOR", "TSC1"],
    "TGFb": ["TGFB1", "TGFB2", "SMAD2", "SMAD3", "SMAD4", "SERPINE1"],
    "NFkB": ["NFKB1", "RELA", "IKBKB", "IL1B", "IL6", "TNF"],
    "TNFa": ["TNF", "TNFRSF1A", "TRADD", "RIPK1", "CASP8", "BIRC3"],
    "JAK-STAT": ["JAK1", "JAK2", "STAT1", "STAT3", "SOCS1", "SOCS3"],
    "VEGF": ["VEGFA", "VEGFB", "KDR", "FLT1", "HIF1A", "EPAS1"],
    "Hypoxia": ["HIF1A", "EPAS1", "LDHA", "PGK1", "ENO1", "GLUT1"],
    "WNT": ["WNT3A", "CTNNB1", "APC", "AXIN1", "TCF7", "LEF1"],
}


@dataclass
class ExpressionSemisyntheticConfig:
    """Configuration for expression-aware semi-synthetic benchmark."""

    # Data sources
    expression_source: Path | None = None  # Path to h5ad with real expression

    # Subclustering
    n_subclusters: int = 2  # Split each cell type into this many subclusters
    min_cells_per_subcluster: int = 100
    leiden_resolution: float = 0.5

    # Spatial layout
    n_worlds: int = 10
    cells_per_world: int = 2000
    world_width: float = 2000.0  # micrometers
    world_height: float = 2000.0

    # Interactions
    interactions: list[InteractionSpec] = field(default_factory=list)

    # Gene selection
    n_hvg: int = 500
    de_pval_threshold: float = 0.05
    de_logfc_threshold: float = 0.5

    # Pathway scoring
    include_pathways: bool = True
    pathways: list[str] = field(default_factory=lambda: PROGENY_PATHWAYS.copy())

    # Output
    output_dir: Path = field(default_factory=lambda: Path("benchmarks/expression_semisynthetic"))
    benchmark_name: str = "expression_v1"
    seed: int = 42

    # Stages
    stages: list[str] = field(default_factory=lambda: ["Normal", "AAH", "AIS", "MIA", "ADC"])


@dataclass
class ExpressionSemisyntheticReport:
    """Report from benchmark generation."""

    n_cells_total: int = 0
    n_worlds: int = 0
    de_gene_sets: list[dict] = field(default_factory=list)
    interaction_rates: dict[str, float] = field(default_factory=dict)
    subcluster_sizes: dict[str, dict[str, int]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_cells_total": self.n_cells_total,
            "n_worlds": self.n_worlds,
            "de_gene_sets": self.de_gene_sets,
            "interaction_rates": self.interaction_rates,
            "subcluster_sizes": self.subcluster_sizes,
        }


class ExpressionSemisyntheticGenerator:
    """Generate expression-aware semi-synthetic benchmarks with DE gene ground truth."""

    def __init__(self, config: ExpressionSemisyntheticConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)

        # Data containers
        self.adata = None
        self.gene_names: np.ndarray | None = None
        self.subcluster_pools: dict[str, dict[str, SubclusterPool]] = {}
        self.de_gene_sets: dict[str, DEGeneSet] = {}
        self.worlds: list[pd.DataFrame] = []

    def generate(self, use_fallback: bool = True) -> ExpressionSemisyntheticReport:
        """Generate the complete expression-aware benchmark.

        Args:
            use_fallback: Use synthetic fallback if real data unavailable

        Returns:
            Report with generation statistics
        """
        report = ExpressionSemisyntheticReport()

        # Ensure we have interactions configured
        if not self.config.interactions:
            log.warning("No interactions configured, using defaults")
            self.config.interactions = [
                InteractionSpec(
                    sender_celltype="immune",
                    receiver_celltype="epithelial",
                    interaction_radius=50.0,
                    interaction_name="immune_epithelial",
                ),
                InteractionSpec(
                    sender_celltype="fibroblast",
                    receiver_celltype="epithelial",
                    interaction_radius=30.0,
                    interaction_name="caf_epithelial",
                ),
            ]

        # Step 1: Load expression data
        log.info("Step 1: Loading expression data...")
        if not self._load_expression_data(use_fallback):
            raise RuntimeError("Failed to load expression data")

        # Step 2: Subcluster each receiver cell type
        log.info("Step 2: Subclustering receiver cell types...")
        self._subcluster_celltypes()
        for ct, pools in self.subcluster_pools.items():
            report.subcluster_sizes[ct] = {k: v.n_cells for k, v in pools.items()}

        # Step 3: Compute DE genes between subclusters
        log.info("Step 3: Computing DE genes between subclusters...")
        self._compute_de_genes()
        for de_set in self.de_gene_sets.values():
            report.de_gene_sets.append(de_set.to_dict())

        # Step 4: Generate spatial worlds with expression assignment
        log.info("Step 4: Generating spatial worlds...")
        self._generate_worlds()
        report.n_worlds = len(self.worlds)
        report.n_cells_total = sum(len(w) for w in self.worlds)

        # Step 5: Compute interaction rates
        for world in self.worlds:
            for interaction in self.config.interactions:
                col = f"is_interacting_{interaction.interaction_name}"
                if col in world.columns:
                    rate = world[col].mean()
                    key = interaction.interaction_name
                    if key not in report.interaction_rates:
                        report.interaction_rates[key] = []
                    report.interaction_rates[key].append(rate)

        # Average rates
        report.interaction_rates = {
            k: float(np.mean(v)) for k, v in report.interaction_rates.items()
        }

        # Step 6: Export
        log.info("Step 5: Exporting benchmark...")
        self._export_benchmark(report)

        log.info(
            "Expression semi-synthetic benchmark complete: %d cells across %d worlds",
            report.n_cells_total,
            report.n_worlds,
        )

        return report

    def _load_expression_data(self, use_fallback: bool) -> bool:
        """Load real expression data or create synthetic fallback."""
        try:
            import anndata
            import scanpy as sc
        except ImportError:
            log.warning("scanpy/anndata not available, using fallback")
            return self._create_fallback_data()

        if self.config.expression_source and self.config.expression_source.exists():
            log.info("Loading expression from %s", self.config.expression_source)
            self.adata = anndata.read_h5ad(self.config.expression_source)

            # Select HVGs
            if self.adata.n_vars > self.config.n_hvg:
                sc.pp.highly_variable_genes(self.adata, n_top_genes=self.config.n_hvg)
                self.adata = self.adata[:, self.adata.var.highly_variable].copy()

            self.gene_names = np.array(self.adata.var_names)
            log.info("Loaded %d cells x %d genes", self.adata.n_obs, self.adata.n_vars)
            return True
        elif use_fallback:
            return self._create_fallback_data()
        else:
            return False

    def _create_fallback_data(self) -> bool:
        """Create synthetic expression data for testing."""
        log.info("Creating synthetic fallback expression data")

        n_cells = 10000
        n_genes = self.config.n_hvg

        # Get cell types from interactions
        celltypes = set()
        for interaction in self.config.interactions:
            celltypes.add(interaction.sender_celltype)
            celltypes.add(interaction.receiver_celltype)

        if not celltypes:
            celltypes = {"epithelial", "immune", "fibroblast"}

        celltypes = list(celltypes)

        # Generate synthetic expression
        # Each cell type has a base signature
        celltype_signatures = {}
        for ct in celltypes:
            celltype_signatures[ct] = self.rng.standard_normal(n_genes) * 0.5

        # Assign cells to cell types
        cells_per_type = n_cells // len(celltypes)

        expressions = []
        cell_types = []
        stages = []

        for ct in celltypes:
            for _ in range(cells_per_type):
                # Base expression
                expr = celltype_signatures[ct] + self.rng.normal(0, 0.3, n_genes)
                expr = np.maximum(expr, 0)
                expressions.append(expr)
                cell_types.append(ct)
                stages.append(self.rng.choice(self.config.stages))

        # Create AnnData-like structure
        self.gene_names = np.array([f"gene_{i}" for i in range(n_genes)])

        # Store as simple arrays (no AnnData dependency for fallback)
        self._fallback_expression = np.array(expressions)
        self._fallback_celltypes = np.array(cell_types)
        self._fallback_stages = np.array(stages)
        self.adata = None  # Flag that we're using fallback

        log.info("Created fallback data: %d cells x %d genes", len(expressions), n_genes)
        return True

    def _subcluster_celltypes(self) -> None:
        """Split each receiver cell type into interacting/non-interacting subclusters."""
        receiver_types = {i.receiver_celltype for i in self.config.interactions}

        for ct in receiver_types:
            log.info("Subclustering %s...", ct)

            if self.adata is not None:
                # Real data: use Leiden clustering
                pools = self._subcluster_real_data(ct)
            else:
                # Fallback: create synthetic subclusters
                pools = self._subcluster_fallback(ct)

            if pools:
                self.subcluster_pools[ct] = pools
                log.info(
                    "  %s: interacting=%d, noninteracting=%d",
                    ct,
                    pools["interacting"].n_cells,
                    pools["noninteracting"].n_cells,
                )

    def _subcluster_real_data(self, celltype: str) -> dict[str, SubclusterPool] | None:
        """Subcluster using Leiden on real data."""
        import scanpy as sc

        # Get cells of this type
        celltype_col = self._find_celltype_column()
        if celltype_col is None:
            return None

        mask = self.adata.obs[celltype_col] == celltype
        if mask.sum() < self.config.min_cells_per_subcluster * 2:
            log.warning("Not enough cells for %s: %d", celltype, mask.sum())
            return None

        subset = self.adata[mask].copy()

        # Run Leiden
        sc.pp.neighbors(subset, n_neighbors=15)
        sc.tl.leiden(subset, resolution=self.config.leiden_resolution, key_added="subcluster")

        # Take largest two clusters as interacting/noninteracting
        cluster_sizes = subset.obs["subcluster"].value_counts()
        if len(cluster_sizes) < 2:
            # Force split if only one cluster
            n = len(subset)
            subset.obs["subcluster"] = ["0"] * (n // 2) + ["1"] * (n - n // 2)
            cluster_sizes = subset.obs["subcluster"].value_counts()

        top_clusters = cluster_sizes.index[:2].tolist()

        pools = {}
        for i, (cluster_id, label) in enumerate(zip(top_clusters, ["interacting", "noninteracting"])):
            cluster_mask = subset.obs["subcluster"] == cluster_id
            cluster_cells = subset[cluster_mask]

            pools[label] = SubclusterPool(
                celltype=celltype,
                subcluster=label,
                cell_ids=np.array(cluster_cells.obs_names),
                expression=cluster_cells.X.toarray() if hasattr(cluster_cells.X, "toarray") else cluster_cells.X,
                metadata=cluster_cells.obs.copy(),
            )

        return pools

    def _subcluster_fallback(self, celltype: str) -> dict[str, SubclusterPool] | None:
        """Create synthetic subclusters with expression differences."""
        mask = self._fallback_celltypes == celltype
        n_cells = mask.sum()

        if n_cells < self.config.min_cells_per_subcluster * 2:
            log.warning("Not enough cells for %s: %d", celltype, n_cells)
            return None

        indices = np.where(mask)[0]
        self.rng.shuffle(indices)

        # Split in half
        split = n_cells // 2
        interacting_idx = indices[:split]
        noninteracting_idx = indices[split:]

        # Add expression shift to interacting cells (simulating DE genes)
        # This creates ~50 DE genes with strong effect
        n_de_genes = 50
        de_gene_indices = self.rng.choice(len(self.gene_names), n_de_genes, replace=False)

        # Copy expression and add shift
        interacting_expr = self._fallback_expression[interacting_idx].copy()
        noninteracting_expr = self._fallback_expression[noninteracting_idx].copy()

        # Shift expression in interacting cells
        shift = np.zeros(len(self.gene_names))
        shift[de_gene_indices[:25]] = 1.0  # 25 genes upregulated
        shift[de_gene_indices[25:]] = -0.5  # 25 genes downregulated
        interacting_expr = interacting_expr + shift
        interacting_expr = np.maximum(interacting_expr, 0)

        pools = {
            "interacting": SubclusterPool(
                celltype=celltype,
                subcluster="interacting",
                cell_ids=np.array([f"{celltype}_int_{i}" for i in range(len(interacting_idx))]),
                expression=interacting_expr,
                metadata=pd.DataFrame({
                    "stage": self._fallback_stages[interacting_idx],
                    "celltype": celltype,
                }),
            ),
            "noninteracting": SubclusterPool(
                celltype=celltype,
                subcluster="noninteracting",
                cell_ids=np.array([f"{celltype}_nonint_{i}" for i in range(len(noninteracting_idx))]),
                expression=noninteracting_expr,
                metadata=pd.DataFrame({
                    "stage": self._fallback_stages[noninteracting_idx],
                    "celltype": celltype,
                }),
            ),
        }

        return pools

    def _find_celltype_column(self) -> str | None:
        """Find the cell type column in adata.obs."""
        if self.adata is None:
            return None

        candidates = ["cell_type", "celltype", "cell_type_fine", "ann_level_3", "ann_level_2"]
        for col in candidates:
            if col in self.adata.obs.columns:
                return col
        return None

    def _compute_pathway_scores(self, expression: np.ndarray) -> dict[str, np.ndarray]:
        """Compute pathway activity scores for cells.

        Uses a simplified scoring: mean expression of pathway genes.
        In real usage, would use PROGENy weights.

        Args:
            expression: (n_cells, n_genes) array

        Returns:
            Dict of pathway -> (n_cells,) scores
        """
        pathway_scores = {}

        for pathway in self.config.pathways:
            if pathway not in PATHWAY_SIGNATURES:
                continue

            # Find genes in this pathway that exist in our gene set
            pathway_genes = PATHWAY_SIGNATURES[pathway]
            gene_indices = []
            for gene in pathway_genes:
                # Check if gene exists (exact or partial match)
                for i, g in enumerate(self.gene_names):
                    if gene.lower() in g.lower() or g.lower() in gene.lower():
                        gene_indices.append(i)
                        break

            if gene_indices:
                # Mean expression of pathway genes (z-scored per gene first)
                pathway_expr = expression[:, gene_indices]
                # Z-score per gene
                means = pathway_expr.mean(axis=0, keepdims=True)
                stds = pathway_expr.std(axis=0, keepdims=True) + 1e-6
                pathway_expr_z = (pathway_expr - means) / stds
                # Mean across genes = pathway score
                pathway_scores[pathway] = pathway_expr_z.mean(axis=1)
            else:
                # No genes found - use random noise as placeholder
                pathway_scores[pathway] = self.rng.standard_normal(expression.shape[0]) * 0.1

        return pathway_scores

    def _apply_stage_modulated_effect(
        self,
        expression: np.ndarray,
        stage: str,
        interaction: InteractionSpec,
        is_interacting: bool,
    ) -> np.ndarray:
        """Apply stage-modulated DE effect to expression.

        The effect size of the DE genes varies by stage according to
        the interaction's stage_effect_sizes.

        Args:
            expression: Base expression vector
            stage: Cell's disease stage
            interaction: The interaction spec
            is_interacting: Whether this cell is interacting

        Returns:
            Modified expression with stage-appropriate effect
        """
        if not is_interacting:
            return expression

        # Get the base effect size and stage multiplier
        base_effect = 1.0
        stage_multiplier = interaction.get_stage_effect_size(stage)

        # Get DE genes for this interaction's receiver type
        celltype = interaction.receiver_celltype
        if celltype not in self.de_gene_sets:
            return expression

        de_set = self.de_gene_sets[celltype]
        expr = expression.copy()

        # Apply modulated effect to DE genes
        for gene in de_set.upregulated_genes:
            if gene in self.gene_names:
                idx = np.where(self.gene_names == gene)[0]
                if len(idx) > 0:
                    # Scale the existing effect by stage multiplier
                    # Interacting cells already have higher expression from subcluster
                    # This adds additional stage modulation
                    expr[idx] *= (1 + 0.2 * (stage_multiplier - 1))

        for gene in de_set.downregulated_genes:
            if gene in self.gene_names:
                idx = np.where(self.gene_names == gene)[0]
                if len(idx) > 0:
                    expr[idx] *= (1 - 0.2 * (stage_multiplier - 1))

        # Also boost associated pathways if specified
        if interaction.associated_pathways:
            for pathway in interaction.associated_pathways:
                if pathway in PATHWAY_SIGNATURES:
                    for gene in PATHWAY_SIGNATURES[pathway]:
                        for i, g in enumerate(self.gene_names):
                            if gene.lower() in g.lower():
                                expr[i] *= (1 + 0.3 * stage_multiplier)
                                break

        return np.maximum(expr, 0)  # Keep non-negative

    def _compute_de_genes(self) -> None:
        """Compute DE genes between interacting and non-interacting subclusters."""
        for celltype, pools in self.subcluster_pools.items():
            interacting = pools["interacting"]
            noninteracting = pools["noninteracting"]

            # Compute per-gene statistics
            upregulated = []
            downregulated = []
            effect_sizes = {}
            pvalues = {}

            for i, gene in enumerate(self.gene_names):
                expr_int = interacting.expression[:, i]
                expr_nonint = noninteracting.expression[:, i]

                # Log2 fold change
                mean_int = np.mean(expr_int) + 1e-6
                mean_nonint = np.mean(expr_nonint) + 1e-6
                log2fc = np.log2(mean_int / mean_nonint)

                # T-test
                if np.std(expr_int) > 0 and np.std(expr_nonint) > 0:
                    _, pval = stats.ttest_ind(expr_int, expr_nonint)
                else:
                    pval = 1.0

                effect_sizes[gene] = log2fc
                pvalues[gene] = pval

                if pval < self.config.de_pval_threshold:
                    if log2fc > self.config.de_logfc_threshold:
                        upregulated.append(gene)
                    elif log2fc < -self.config.de_logfc_threshold:
                        downregulated.append(gene)

            # Find relevant interaction
            interaction_name = "unknown"
            for interaction in self.config.interactions:
                if interaction.receiver_celltype == celltype:
                    interaction_name = interaction.interaction_name
                    break

            self.de_gene_sets[celltype] = DEGeneSet(
                celltype=celltype,
                interaction_name=interaction_name,
                upregulated_genes=sorted(upregulated, key=lambda g: -effect_sizes[g]),
                downregulated_genes=sorted(downregulated, key=lambda g: effect_sizes[g]),
                effect_sizes=effect_sizes,
                pvalues=pvalues,
            )

            log.info(
                "  %s: %d upregulated, %d downregulated genes",
                celltype,
                len(upregulated),
                len(downregulated),
            )

    def _generate_worlds(self) -> None:
        """Generate synthetic spatial worlds with proper expression assignment."""
        for world_idx in range(self.config.n_worlds):
            world = self._generate_single_world(world_idx)
            self.worlds.append(world)

    def _generate_single_world(self, world_idx: int) -> pd.DataFrame:
        """Generate a single world with proximity-based expression assignment."""
        world_seed = self.config.seed + world_idx * 1000
        world_rng = np.random.default_rng(world_seed)

        n_cells = self.config.cells_per_world

        # Get all cell types from interactions OR subcluster pools OR fallback
        all_celltypes = set()
        for interaction in self.config.interactions:
            all_celltypes.add(interaction.sender_celltype)
            all_celltypes.add(interaction.receiver_celltype)

        if not all_celltypes:
            all_celltypes = set(self.subcluster_pools.keys())

        if not all_celltypes and hasattr(self, "_fallback_celltypes"):
            all_celltypes = set(np.unique(self._fallback_celltypes))

        if not all_celltypes:
            # Ultimate fallback
            all_celltypes = {"epithelial", "immune", "fibroblast"}

        all_celltypes = list(all_celltypes)
        cells_per_type = n_cells // max(1, len(all_celltypes))

        # Generate spatial positions with gradient pattern
        # Each cell type occupies a quadrant with gradient transitions
        all_cells = []

        for ct_idx, celltype in enumerate(all_celltypes):
            # Assign base quadrant
            quadrant_x = ct_idx % 2
            quadrant_y = ct_idx // 2 % 2

            for i in range(cells_per_type):
                # Position with gradient at boundaries
                base_x = quadrant_x * (self.config.world_width / 2)
                base_y = quadrant_y * (self.config.world_height / 2)

                x = base_x + world_rng.uniform(0, self.config.world_width / 2)
                y = base_y + world_rng.uniform(0, self.config.world_height / 2)

                # Add some noise to create mixed regions at boundaries
                x += world_rng.normal(0, self.config.world_width * 0.05)
                y += world_rng.normal(0, self.config.world_height * 0.05)

                # Clip to world bounds
                x = np.clip(x, 0, self.config.world_width)
                y = np.clip(y, 0, self.config.world_height)

                stage = world_rng.choice(self.config.stages)

                all_cells.append({
                    "x": x,
                    "y": y,
                    "celltype": celltype,
                    "stage": stage,
                    "cell_idx": len(all_cells),
                })

        cells_df = pd.DataFrame(all_cells)

        # Now assign expression based on spatial proximity
        coords = cells_df[["x", "y"]].values
        distances = cdist(coords, coords)

        # For each interaction, determine which receivers are "interacting"
        for interaction in self.config.interactions:
            sender_mask = cells_df["celltype"] == interaction.sender_celltype
            receiver_mask = cells_df["celltype"] == interaction.receiver_celltype

            is_interacting = np.zeros(len(cells_df), dtype=bool)

            for idx in np.where(receiver_mask)[0]:
                # Check if any sender is within radius
                dists = distances[idx]
                nearby_senders = sender_mask & (dists <= interaction.interaction_radius) & (dists > 0)

                if nearby_senders.any():
                    is_interacting[idx] = True

            col_name = f"is_interacting_{interaction.interaction_name}"
            cells_df[col_name] = is_interacting

        # Assign expression from appropriate subcluster with stage modulation
        expressions = []
        for idx, row in cells_df.iterrows():
            celltype = row["celltype"]
            stage = row["stage"]

            if celltype in self.subcluster_pools:
                # Receiver cell type - check interaction status
                pools = self.subcluster_pools[celltype]

                # Check if interacting for any interaction and find which one
                is_int = False
                active_interaction = None
                for interaction in self.config.interactions:
                    if interaction.receiver_celltype == celltype:
                        col = f"is_interacting_{interaction.interaction_name}"
                        if cells_df.loc[idx, col]:
                            is_int = True
                            active_interaction = interaction
                            break

                pool = pools["interacting"] if is_int else pools["noninteracting"]

                # Sample random cell from pool
                cell_idx = world_rng.integers(0, pool.n_cells)
                expr = pool.expression[cell_idx].copy()

                # Apply stage-modulated effect if interacting
                if is_int and active_interaction is not None:
                    expr = self._apply_stage_modulated_effect(
                        expr, stage, active_interaction, is_int
                    )
            else:
                # Sender cell type - just sample from fallback
                if self.adata is not None:
                    # Real data
                    ct_mask = self.adata.obs[self._find_celltype_column()] == celltype
                    ct_indices = np.where(ct_mask)[0]
                    if len(ct_indices) > 0:
                        cell_idx = world_rng.choice(ct_indices)
                        X = self.adata.X[cell_idx]
                        expr = X.toarray().flatten() if hasattr(X, "toarray") else X.flatten()
                    else:
                        expr = world_rng.standard_normal(len(self.gene_names))
                else:
                    # Fallback
                    mask = self._fallback_celltypes == celltype
                    indices = np.where(mask)[0]
                    if len(indices) > 0:
                        cell_idx = world_rng.choice(indices)
                        expr = self._fallback_expression[cell_idx].copy()
                    else:
                        expr = world_rng.standard_normal(len(self.gene_names))

            expressions.append(expr)

        cells_df["expression"] = expressions
        cells_df["world_id"] = f"world_{world_idx:04d}"
        cells_df["synthetic_cell_id"] = [f"world{world_idx}_cell{i}" for i in range(len(cells_df))]

        # Compute overall is_interacting (any interaction)
        interaction_cols = [c for c in cells_df.columns if c.startswith("is_interacting_")]
        if interaction_cols:
            cells_df["is_interacting"] = cells_df[interaction_cols].any(axis=1)
        else:
            cells_df["is_interacting"] = False

        # Compute pathway scores if enabled
        if self.config.include_pathways:
            expr_array = np.array(expressions)
            pathway_scores = self._compute_pathway_scores(expr_array)
            for pathway, scores in pathway_scores.items():
                cells_df[f"pathway_{pathway}"] = scores

        return cells_df

    def _export_benchmark(self, report: ExpressionSemisyntheticReport) -> None:
        """Export benchmark to disk."""
        output_dir = self.config.output_dir / self.config.benchmark_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Export ground truth DE genes
        de_ground_truth = {}
        for celltype, de_set in self.de_gene_sets.items():
            de_ground_truth[celltype] = {
                "upregulated": de_set.upregulated_genes,
                "downregulated": de_set.downregulated_genes,
                "effect_sizes": {k: float(v) for k, v in de_set.effect_sizes.items()},
            }

        with open(output_dir / "de_ground_truth.json", "w") as f:
            json.dump(de_ground_truth, f, indent=2)

        # Export world data
        for i, world in enumerate(self.worlds):
            world_path = output_dir / f"world_{i:04d}.parquet"

            # Convert expression to separate columns or save separately
            world_meta = world.drop(columns=["expression"])
            world_meta.to_parquet(world_path)

            # Save expression as numpy
            expr_array = np.array(world["expression"].tolist())
            np.save(output_dir / f"world_{i:04d}_expression.npy", expr_array)

        # Export gene names
        np.save(output_dir / "gene_names.npy", self.gene_names)

        # Export config
        config_dict = {
            "n_worlds": self.config.n_worlds,
            "cells_per_world": self.config.cells_per_world,
            "n_hvg": self.config.n_hvg,
            "interactions": [
                {
                    "sender": i.sender_celltype,
                    "receiver": i.receiver_celltype,
                    "radius": i.interaction_radius,
                    "name": i.interaction_name,
                    "associated_pathways": i.associated_pathways,
                    "stage_effect_sizes": i.stage_effect_sizes,
                }
                for i in self.config.interactions
            ],
            "stages": self.config.stages,
            "pathways": self.config.pathways if self.config.include_pathways else [],
            "seed": self.config.seed,
        }

        with open(output_dir / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)

        # Export pathway ground truth (which pathways should be active in which interactions)
        if self.config.include_pathways:
            pathway_ground_truth = {
                "pathways": self.config.pathways,
                "pathway_signatures": {
                    p: PATHWAY_SIGNATURES.get(p, [])
                    for p in self.config.pathways
                },
                "interaction_pathways": {
                    i.interaction_name: i.associated_pathways
                    for i in self.config.interactions
                },
                "stage_effect_sizes": {
                    i.interaction_name: i.stage_effect_sizes or {}
                    for i in self.config.interactions
                },
            }
            with open(output_dir / "pathway_ground_truth.json", "w") as f:
                json.dump(pathway_ground_truth, f, indent=2)

        # Export report
        with open(output_dir / "generation_report.json", "w") as f:
            json.dump(report.to_dict(), f, indent=2)

        log.info("Exported benchmark to %s", output_dir)


def create_default_config() -> ExpressionSemisyntheticConfig:
    """Create default expression-aware config for StageBridge."""
    return ExpressionSemisyntheticConfig(
        interactions=[
            InteractionSpec(
                sender_celltype="immune",
                receiver_celltype="epithelial",
                interaction_radius=50.0,
                interaction_name="immune_epithelial",
                stage_weights={"Normal": 0.3, "AAH": 0.5, "AIS": 0.7, "MIA": 0.9, "ADC": 1.0},
            ),
            InteractionSpec(
                sender_celltype="fibroblast",
                receiver_celltype="epithelial",
                interaction_radius=30.0,
                interaction_name="caf_epithelial",
                stage_weights={"Normal": 0.2, "AAH": 0.4, "AIS": 0.6, "MIA": 0.8, "ADC": 1.0},
            ),
        ],
        n_worlds=10,
        cells_per_world=2000,
        n_hvg=500,
        stages=["Normal", "AAH", "AIS", "MIA", "ADC"],
    )


def generate_expression_benchmark(
    config: ExpressionSemisyntheticConfig | None = None,
    use_fallback: bool = True,
) -> ExpressionSemisyntheticReport:
    """Generate an expression-aware semi-synthetic benchmark.

    Args:
        config: Configuration (uses default if None)
        use_fallback: Use synthetic fallback if real data unavailable

    Returns:
        Generation report with statistics
    """
    if config is None:
        config = create_default_config()

    generator = ExpressionSemisyntheticGenerator(config)
    return generator.generate(use_fallback=use_fallback)
