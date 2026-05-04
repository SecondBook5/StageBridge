"""Ground truth labeler for real single-cell and spatial data.

This module applies AMICI-style ground truth labeling to REAL data:
- Takes your actual snRNA-seq and spatial data (h5ad files)
- Applies explicit interaction rules based on known biology (IL1B-IL1R1, etc.)
- Labels cells as "interacting" vs "non-interacting" based on spatial proximity
- Records which cells are within interaction range of relevant senders
- Provides ground truth for evaluating StageBridge attention patterns

Unlike the synthetic generator, this uses YOUR actual expression profiles
and spatial coordinates - we just add ground truth labels on top.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from tqdm import tqdm


@dataclass
class InteractionRule:
    """Defines a sender-receiver interaction with distance threshold.

    Based on known biology from Peng/Kadara and LUAD literature:
    - IL1B-IL1R1: Macrophage (IL1B+) → Epithelial (IL1R1+), short-range
    - CXCL12-CXCR4: Fibroblast → Epithelial, chemokine gradient
    - TGFB1-TGFBR1: CAF → Epithelial, EMT induction
    """
    sender_type: str
    receiver_type: str
    max_distance: float  # microns
    ligand_gene: str  # gene expressed by sender
    receptor_gene: str  # gene expressed by receiver
    downstream_genes: list[str] = field(default_factory=list)
    interaction_name: str = ""

    def __post_init__(self):
        if not self.interaction_name:
            self.interaction_name = f"{self.ligand_gene}-{self.receptor_gene}"


@dataclass
class GroundTruthLabels:
    """Ground truth labels for real data benchmark.

    Enables evaluation of StageBridge attention:
    - Does attention to ring correlate with sender presence?
    - Are "interacting" cells reconstructed differently?
    """
    # Per-cell labels (aligned with adata.obs index)
    cell_ids: np.ndarray  # cell barcodes
    is_interacting: np.ndarray  # bool - within range of relevant sender
    interaction_type: np.ndarray  # string - which interaction rule applies
    nearest_sender_distance: np.ndarray  # float - distance to nearest sender
    nearest_sender_id: np.ndarray  # string - ID of nearest sender
    expected_attention_ring: np.ndarray  # int - which ring contains sender

    # Interaction metadata
    receiver_sender_pairs: pd.DataFrame  # detailed pair info

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame for easy analysis."""
        return pd.DataFrame({
            "cell_id": self.cell_ids,
            "is_interacting": self.is_interacting,
            "interaction_type": self.interaction_type,
            "nearest_sender_distance": self.nearest_sender_distance,
            "nearest_sender_id": self.nearest_sender_id,
            "expected_attention_ring": self.expected_attention_ring,
        })

    def save(self, output_dir: str | Path):
        """Save ground truth to files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save main labels
        self.to_dataframe().to_parquet(output_dir / "ground_truth_labels.parquet")

        # Save receiver-sender pairs
        self.receiver_sender_pairs.to_parquet(output_dir / "receiver_sender_pairs.parquet")

        # Save summary
        summary = {
            "n_cells": len(self.cell_ids),
            "n_interacting": int(self.is_interacting.sum()),
            "pct_interacting": float(100 * self.is_interacting.mean()),
            "interaction_types": list(np.unique(self.interaction_type[self.is_interacting])),
            "n_pairs": len(self.receiver_sender_pairs),
        }
        with open(output_dir / "ground_truth_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"Saved ground truth to {output_dir}")


# Default interaction rules based on LUAD biology
# Sources: Peng/Kadara 2024, CellChat, CellPhoneDB, LUAD literature
DEFAULT_LUAD_RULES = [
    # =========================================================================
    # PRO-INFLAMMATORY / IL1 SIGNALING (from Peng/Kadara - key for progression)
    # =========================================================================
    InteractionRule(
        sender_type="Macrophage",
        receiver_type="Epithelial",
        max_distance=50.0,
        ligand_gene="IL1B",
        receptor_gene="IL1R1",
        downstream_genes=["CXCL8", "IL6", "NFKBIA", "PTGS2"],
        interaction_name="IL1B-IL1R1",
    ),
    InteractionRule(
        sender_type="Macrophage",
        receiver_type="Epithelial",
        max_distance=50.0,
        ligand_gene="IL1A",
        receptor_gene="IL1R1",
        downstream_genes=["CXCL8", "IL6"],
        interaction_name="IL1A-IL1R1",
    ),
    InteractionRule(
        sender_type="Macrophage",
        receiver_type="Epithelial",
        max_distance=60.0,
        ligand_gene="TNF",
        receptor_gene="TNFRSF1A",
        downstream_genes=["NFKBIA", "BIRC3", "CXCL1"],
        interaction_name="TNF-TNFR1",
    ),
    InteractionRule(
        sender_type="Macrophage",
        receiver_type="Fibroblast",
        max_distance=60.0,
        ligand_gene="TNF",
        receptor_gene="TNFRSF1A",
        downstream_genes=["IL6", "CXCL12"],
        interaction_name="TNF-TNFR1_fibro",
    ),

    # =========================================================================
    # CHEMOKINE SIGNALING
    # =========================================================================
    InteractionRule(
        sender_type="Fibroblast",
        receiver_type="Epithelial",
        max_distance=100.0,
        ligand_gene="CXCL12",
        receptor_gene="CXCR4",
        downstream_genes=["MMP9", "VEGFA", "MYC"],
        interaction_name="CXCL12-CXCR4",
    ),
    InteractionRule(
        sender_type="Macrophage",
        receiver_type="T cell",
        max_distance=80.0,
        ligand_gene="CXCL9",
        receptor_gene="CXCR3",
        downstream_genes=["IFNG", "GZMB"],
        interaction_name="CXCL9-CXCR3",
    ),
    InteractionRule(
        sender_type="Macrophage",
        receiver_type="T cell",
        max_distance=80.0,
        ligand_gene="CXCL10",
        receptor_gene="CXCR3",
        downstream_genes=["IFNG", "GZMB"],
        interaction_name="CXCL10-CXCR3",
    ),
    InteractionRule(
        sender_type="Epithelial",
        receiver_type="Macrophage",
        max_distance=75.0,
        ligand_gene="CCL2",
        receptor_gene="CCR2",
        downstream_genes=["IL1B", "TNF"],
        interaction_name="CCL2-CCR2",
    ),

    # =========================================================================
    # TGF-BETA / EMT SIGNALING (CAF-mediated)
    # =========================================================================
    InteractionRule(
        sender_type="CAF",
        receiver_type="Epithelial",
        max_distance=75.0,
        ligand_gene="TGFB1",
        receptor_gene="TGFBR1",
        downstream_genes=["SNAI1", "VIM", "CDH2", "ZEB1"],
        interaction_name="TGFB1-TGFBR1",
    ),
    InteractionRule(
        sender_type="CAF",
        receiver_type="Epithelial",
        max_distance=75.0,
        ligand_gene="TGFB1",
        receptor_gene="TGFBR2",
        downstream_genes=["SNAI1", "VIM", "SMAD3"],
        interaction_name="TGFB1-TGFBR2",
    ),
    InteractionRule(
        sender_type="Fibroblast",
        receiver_type="Epithelial",
        max_distance=75.0,
        ligand_gene="TGFB1",
        receptor_gene="TGFBR1",
        downstream_genes=["SNAI1", "VIM"],
        interaction_name="TGFB1-TGFBR1_fibro",
    ),

    # =========================================================================
    # GROWTH FACTOR SIGNALING (HGF, EGF, FGF)
    # =========================================================================
    InteractionRule(
        sender_type="Fibroblast",
        receiver_type="Epithelial",
        max_distance=100.0,
        ligand_gene="HGF",
        receptor_gene="MET",
        downstream_genes=["MYC", "CCND1", "MMP9"],
        interaction_name="HGF-MET",
    ),
    InteractionRule(
        sender_type="CAF",
        receiver_type="Epithelial",
        max_distance=100.0,
        ligand_gene="HGF",
        receptor_gene="MET",
        downstream_genes=["MYC", "CCND1", "MMP9"],
        interaction_name="HGF-MET_caf",
    ),
    InteractionRule(
        sender_type="Macrophage",
        receiver_type="Epithelial",
        max_distance=60.0,
        ligand_gene="EGF",
        receptor_gene="EGFR",
        downstream_genes=["MYC", "CCND1", "FOS"],
        interaction_name="EGF-EGFR",
    ),
    InteractionRule(
        sender_type="Fibroblast",
        receiver_type="Epithelial",
        max_distance=80.0,
        ligand_gene="FGF2",
        receptor_gene="FGFR1",
        downstream_genes=["MYC", "ETV4"],
        interaction_name="FGF2-FGFR1",
    ),
    InteractionRule(
        sender_type="Fibroblast",
        receiver_type="Epithelial",
        max_distance=80.0,
        ligand_gene="FGF7",
        receptor_gene="FGFR2",
        downstream_genes=["MYC", "FOXA1"],
        interaction_name="FGF7-FGFR2",
    ),

    # =========================================================================
    # SPP1/OSTEOPONTIN (tumor-promoting macrophages)
    # =========================================================================
    InteractionRule(
        sender_type="Macrophage",
        receiver_type="Epithelial",
        max_distance=60.0,
        ligand_gene="SPP1",
        receptor_gene="CD44",
        downstream_genes=["MYC", "CCND1", "MMP9"],
        interaction_name="SPP1-CD44",
    ),
    InteractionRule(
        sender_type="Macrophage",
        receiver_type="Epithelial",
        max_distance=60.0,
        ligand_gene="SPP1",
        receptor_gene="ITGAV",
        downstream_genes=["MYC", "VIM"],
        interaction_name="SPP1-ITGAV",
    ),

    # =========================================================================
    # WNT SIGNALING
    # =========================================================================
    InteractionRule(
        sender_type="Fibroblast",
        receiver_type="Epithelial",
        max_distance=75.0,
        ligand_gene="WNT5A",
        receptor_gene="FZD5",
        downstream_genes=["MYC", "CCND1", "AXIN2"],
        interaction_name="WNT5A-FZD5",
    ),
    InteractionRule(
        sender_type="CAF",
        receiver_type="Epithelial",
        max_distance=75.0,
        ligand_gene="WNT2",
        receptor_gene="FZD7",
        downstream_genes=["MYC", "AXIN2", "LGR5"],
        interaction_name="WNT2-FZD7",
    ),

    # =========================================================================
    # NOTCH SIGNALING (juxtacrine - very short range)
    # =========================================================================
    InteractionRule(
        sender_type="Epithelial",
        receiver_type="Epithelial",
        max_distance=30.0,  # Contact-dependent
        ligand_gene="JAG1",
        receptor_gene="NOTCH1",
        downstream_genes=["HES1", "HEY1"],
        interaction_name="JAG1-NOTCH1",
    ),
    InteractionRule(
        sender_type="Epithelial",
        receiver_type="Epithelial",
        max_distance=30.0,
        ligand_gene="DLL1",
        receptor_gene="NOTCH1",
        downstream_genes=["HES1", "HEY1"],
        interaction_name="DLL1-NOTCH1",
    ),

    # =========================================================================
    # IMMUNE CHECKPOINT / T CELL EXHAUSTION
    # =========================================================================
    InteractionRule(
        sender_type="Epithelial",
        receiver_type="T cell",
        max_distance=40.0,
        ligand_gene="CD274",  # PD-L1
        receptor_gene="PDCD1",  # PD-1
        downstream_genes=["TOX", "LAG3"],
        interaction_name="PDL1-PD1",
    ),
    InteractionRule(
        sender_type="Macrophage",
        receiver_type="T cell",
        max_distance=40.0,
        ligand_gene="CD274",
        receptor_gene="PDCD1",
        downstream_genes=["TOX", "LAG3"],
        interaction_name="PDL1-PD1_mac",
    ),
    InteractionRule(
        sender_type="Epithelial",
        receiver_type="T cell",
        max_distance=40.0,
        ligand_gene="CD80",
        receptor_gene="CTLA4",
        downstream_genes=["TOX"],
        interaction_name="CD80-CTLA4",
    ),

    # =========================================================================
    # VEGF / ANGIOGENESIS
    # =========================================================================
    InteractionRule(
        sender_type="Epithelial",
        receiver_type="Endothelial",
        max_distance=100.0,
        ligand_gene="VEGFA",
        receptor_gene="KDR",  # VEGFR2
        downstream_genes=["PECAM1", "CDH5"],
        interaction_name="VEGFA-VEGFR2",
    ),
    InteractionRule(
        sender_type="Macrophage",
        receiver_type="Endothelial",
        max_distance=100.0,
        ligand_gene="VEGFA",
        receptor_gene="KDR",
        downstream_genes=["PECAM1"],
        interaction_name="VEGFA-VEGFR2_mac",
    ),

    # =========================================================================
    # ECM / INTEGRIN SIGNALING
    # =========================================================================
    InteractionRule(
        sender_type="CAF",
        receiver_type="Epithelial",
        max_distance=50.0,
        ligand_gene="FN1",
        receptor_gene="ITGB1",
        downstream_genes=["FAK", "SRC"],
        interaction_name="FN1-ITGB1",
    ),
    InteractionRule(
        sender_type="Fibroblast",
        receiver_type="Epithelial",
        max_distance=50.0,
        ligand_gene="COL1A1",
        receptor_gene="ITGA2",
        downstream_genes=["FAK"],
        interaction_name="COL1A1-ITGA2",
    ),

    # =========================================================================
    # PDGF SIGNALING (fibroblast activation)
    # =========================================================================
    InteractionRule(
        sender_type="Epithelial",
        receiver_type="Fibroblast",
        max_distance=80.0,
        ligand_gene="PDGFA",
        receptor_gene="PDGFRA",
        downstream_genes=["ACTA2", "COL1A1"],
        interaction_name="PDGFA-PDGFRA",
    ),
    InteractionRule(
        sender_type="Epithelial",
        receiver_type="Fibroblast",
        max_distance=80.0,
        ligand_gene="PDGFB",
        receptor_gene="PDGFRB",
        downstream_genes=["ACTA2", "FAP"],
        interaction_name="PDGFB-PDGFRB",
    ),

    # =========================================================================
    # IL6 SIGNALING (inflammation/STAT3)
    # =========================================================================
    InteractionRule(
        sender_type="Macrophage",
        receiver_type="Epithelial",
        max_distance=70.0,
        ligand_gene="IL6",
        receptor_gene="IL6R",
        downstream_genes=["STAT3", "SOCS3", "BCL2"],
        interaction_name="IL6-IL6R",
    ),
    InteractionRule(
        sender_type="CAF",
        receiver_type="Epithelial",
        max_distance=70.0,
        ligand_gene="IL6",
        receptor_gene="IL6R",
        downstream_genes=["STAT3", "SOCS3"],
        interaction_name="IL6-IL6R_caf",
    ),

    # =========================================================================
    # COMPLEMENT (innate immunity)
    # =========================================================================
    InteractionRule(
        sender_type="Macrophage",
        receiver_type="Epithelial",
        max_distance=60.0,
        ligand_gene="C3",
        receptor_gene="C3AR1",
        downstream_genes=["NFE2L2"],
        interaction_name="C3-C3AR1",
    ),

    # =========================================================================
    # SEMAPHORIN (axon guidance, repurposed in cancer)
    # =========================================================================
    InteractionRule(
        sender_type="Fibroblast",
        receiver_type="Epithelial",
        max_distance=80.0,
        ligand_gene="SEMA3A",
        receptor_gene="NRP1",
        downstream_genes=["VEGFA"],
        interaction_name="SEMA3A-NRP1",
    ),
]


class GroundTruthLabeler:
    """Labels real data with ground truth interaction status.

    Usage:
        labeler = GroundTruthLabeler()
        labeler.add_rule(InteractionRule(...))  # or use defaults

        # From AnnData
        labels = labeler.label_anndata(adata, cell_type_key="cell_type")

        # From neighborhoods.parquet
        labels = labeler.label_neighborhoods(neighborhoods_df)
    """

    def __init__(
        self,
        ring_radii: list[float] = [50, 100, 150, 200],
        use_default_rules: bool = True,
    ):
        self.ring_radii = ring_radii
        self.rules: list[InteractionRule] = []
        if use_default_rules:
            self.rules.extend(DEFAULT_LUAD_RULES)

    def add_rule(self, rule: InteractionRule) -> "GroundTruthLabeler":
        """Add an interaction rule. Chainable."""
        self.rules.append(rule)
        return self

    def clear_rules(self) -> "GroundTruthLabeler":
        """Clear all rules."""
        self.rules = []
        return self

    def label_anndata(
        self,
        adata,
        cell_type_key: str = "cell_type",
        spatial_key: str = "spatial",
        ligand_threshold: float = 0.5,  # log-normalized expression threshold
        receptor_threshold: float = 0.5,
    ) -> GroundTruthLabels:
        """Label cells in AnnData with ground truth interaction status.

        Args:
            adata: AnnData with spatial coordinates and expression
            cell_type_key: Column in obs with cell type labels
            spatial_key: Key in obsm with spatial coordinates
            ligand_threshold: Expression threshold to consider ligand "expressed"
            receptor_threshold: Expression threshold to consider receptor "expressed"

        Returns:
            GroundTruthLabels with interaction status per cell
        """
        import anndata as ad

        n_cells = adata.n_obs
        coords = adata.obsm[spatial_key]
        cell_types = adata.obs[cell_type_key].values
        cell_ids = adata.obs_names.values

        # Build spatial index
        tree = cKDTree(coords)

        # Initialize labels
        is_interacting = np.zeros(n_cells, dtype=bool)
        interaction_type = np.array(["none"] * n_cells, dtype=object)
        nearest_sender_dist = np.full(n_cells, np.inf)
        nearest_sender_id = np.array([""] * n_cells, dtype=object)
        expected_ring = np.full(n_cells, -1, dtype=int)
        pairs = []

        # Get expression matrix (dense for gene lookup)
        if hasattr(adata.X, "toarray"):
            X = adata.X.toarray()
        else:
            X = np.array(adata.X)
        gene_names = list(adata.var_names)

        for rule in self.rules:
            print(f"Processing rule: {rule.interaction_name}")

            # Find potential senders: correct cell type AND expressing ligand
            sender_type_mask = np.array([
                rule.sender_type.lower() in str(ct).lower()
                for ct in cell_types
            ])

            if rule.ligand_gene in gene_names:
                ligand_idx = gene_names.index(rule.ligand_gene)
                ligand_expr = X[:, ligand_idx]
                sender_expr_mask = ligand_expr > ligand_threshold
                sender_mask = sender_type_mask & sender_expr_mask
            else:
                print(f"  Warning: {rule.ligand_gene} not in gene list, using cell type only")
                sender_mask = sender_type_mask

            # Find potential receivers: correct cell type AND expressing receptor
            receiver_type_mask = np.array([
                rule.receiver_type.lower() in str(ct).lower()
                for ct in cell_types
            ])

            if rule.receptor_gene in gene_names:
                receptor_idx = gene_names.index(rule.receptor_gene)
                receptor_expr = X[:, receptor_idx]
                receiver_expr_mask = receptor_expr > receptor_threshold
                receiver_mask = receiver_type_mask & receiver_expr_mask
            else:
                print(f"  Warning: {rule.receptor_gene} not in gene list, using cell type only")
                receiver_mask = receiver_type_mask

            sender_idx = np.where(sender_mask)[0]
            receiver_idx = np.where(receiver_mask)[0]

            print(f"  Senders ({rule.sender_type}+{rule.ligand_gene}): {len(sender_idx)}")
            print(f"  Receivers ({rule.receiver_type}+{rule.receptor_gene}): {len(receiver_idx)}")
            sys.stdout.flush()

            if len(sender_idx) == 0 or len(receiver_idx) == 0:
                continue

            # For each receiver, find nearby senders
            sender_set = set(sender_idx)
            for recv_i in tqdm(receiver_idx, desc=f"  {rule.interaction_name}", file=sys.stdout):
                nearby = tree.query_ball_point(coords[recv_i], rule.max_distance)
                nearby_senders = [i for i in nearby if i in sender_set and i != recv_i]

                if nearby_senders:
                    # Find nearest sender
                    distances = np.linalg.norm(
                        coords[nearby_senders] - coords[recv_i], axis=1
                    )
                    nearest_idx = np.argmin(distances)
                    nearest_sender = nearby_senders[nearest_idx]
                    dist = distances[nearest_idx]

                    # Update if this is closer than previous interactions
                    if dist < nearest_sender_dist[recv_i]:
                        is_interacting[recv_i] = True
                        interaction_type[recv_i] = rule.interaction_name
                        nearest_sender_dist[recv_i] = dist
                        nearest_sender_id[recv_i] = cell_ids[nearest_sender]

                        # Determine which ring
                        radii = [0] + self.ring_radii
                        for ring_idx in range(len(self.ring_radii)):
                            if radii[ring_idx] <= dist < radii[ring_idx + 1]:
                                expected_ring[recv_i] = ring_idx
                                break

                    # Record all pairs
                    for sender_i in nearby_senders:
                        d = np.linalg.norm(coords[sender_i] - coords[recv_i])
                        pairs.append({
                            "receiver_id": cell_ids[recv_i],
                            "sender_id": cell_ids[sender_i],
                            "receiver_idx": recv_i,
                            "sender_idx": sender_i,
                            "distance": d,
                            "rule_name": rule.interaction_name,
                            "receiver_type": cell_types[recv_i],
                            "sender_type": cell_types[sender_i],
                        })

        pairs_df = pd.DataFrame(pairs) if pairs else pd.DataFrame(
            columns=["receiver_id", "sender_id", "receiver_idx", "sender_idx",
                     "distance", "rule_name", "receiver_type", "sender_type"]
        )

        n_interacting = is_interacting.sum()
        print(f"\nTotal interacting cells: {n_interacting} ({100*n_interacting/n_cells:.1f}%)")
        print(f"Total pairs: {len(pairs_df)}")

        return GroundTruthLabels(
            cell_ids=cell_ids,
            is_interacting=is_interacting,
            interaction_type=interaction_type,
            nearest_sender_distance=nearest_sender_dist,
            nearest_sender_id=nearest_sender_id,
            expected_attention_ring=expected_ring,
            receiver_sender_pairs=pairs_df,
        )

    def label_neighborhoods(
        self,
        neighborhoods: pd.DataFrame,
        cell_type_key: str = "cell_type",
    ) -> GroundTruthLabels:
        """Label cells in neighborhoods.parquet with ground truth.

        This version works with pre-computed neighborhoods that have
        spatial coordinates (x, y) and cell type labels.

        Args:
            neighborhoods: DataFrame with x, y, cell_type columns
            cell_type_key: Column name for cell types

        Returns:
            GroundTruthLabels with interaction status per cell
        """
        n_cells = len(neighborhoods)
        coords = neighborhoods[["x", "y"]].values
        cell_types = neighborhoods[cell_type_key].values
        cell_ids = neighborhoods["cell_id"].values if "cell_id" in neighborhoods.columns else np.arange(n_cells).astype(str)

        # Build spatial index
        tree = cKDTree(coords)

        # Initialize labels
        is_interacting = np.zeros(n_cells, dtype=bool)
        interaction_type = np.array(["none"] * n_cells, dtype=object)
        nearest_sender_dist = np.full(n_cells, np.inf)
        nearest_sender_id = np.array([""] * n_cells, dtype=object)
        expected_ring = np.full(n_cells, -1, dtype=int)
        pairs = []

        for rule in self.rules:
            print(f"Processing rule: {rule.interaction_name}")

            # Find by cell type (no expression data in neighborhoods.parquet)
            sender_mask = np.array([
                rule.sender_type.lower() in str(ct).lower()
                for ct in cell_types
            ])
            receiver_mask = np.array([
                rule.receiver_type.lower() in str(ct).lower()
                for ct in cell_types
            ])

            sender_idx = np.where(sender_mask)[0]
            receiver_idx = np.where(receiver_mask)[0]

            print(f"  Senders ({rule.sender_type}): {len(sender_idx)}")
            print(f"  Receivers ({rule.receiver_type}): {len(receiver_idx)}")
            sys.stdout.flush()

            if len(sender_idx) == 0 or len(receiver_idx) == 0:
                continue

            # For each receiver, find nearby senders
            sender_set = set(sender_idx)
            for recv_i in tqdm(receiver_idx, desc=f"  {rule.interaction_name}", file=sys.stdout):
                nearby = tree.query_ball_point(coords[recv_i], rule.max_distance)
                nearby_senders = [i for i in nearby if i in sender_set and i != recv_i]

                if nearby_senders:
                    distances = np.linalg.norm(
                        coords[nearby_senders] - coords[recv_i], axis=1
                    )
                    nearest_idx = np.argmin(distances)
                    nearest_sender = nearby_senders[nearest_idx]
                    dist = distances[nearest_idx]

                    if dist < nearest_sender_dist[recv_i]:
                        is_interacting[recv_i] = True
                        interaction_type[recv_i] = rule.interaction_name
                        nearest_sender_dist[recv_i] = dist
                        nearest_sender_id[recv_i] = str(cell_ids[nearest_sender])

                        radii = [0] + self.ring_radii
                        for ring_idx in range(len(self.ring_radii)):
                            if radii[ring_idx] <= dist < radii[ring_idx + 1]:
                                expected_ring[recv_i] = ring_idx
                                break

                    for sender_i in nearby_senders:
                        d = np.linalg.norm(coords[sender_i] - coords[recv_i])
                        pairs.append({
                            "receiver_id": str(cell_ids[recv_i]),
                            "sender_id": str(cell_ids[sender_i]),
                            "receiver_idx": recv_i,
                            "sender_idx": sender_i,
                            "distance": d,
                            "rule_name": rule.interaction_name,
                            "receiver_type": cell_types[recv_i],
                            "sender_type": cell_types[sender_i],
                        })

        pairs_df = pd.DataFrame(pairs) if pairs else pd.DataFrame(
            columns=["receiver_id", "sender_id", "receiver_idx", "sender_idx",
                     "distance", "rule_name", "receiver_type", "sender_type"]
        )

        n_interacting = is_interacting.sum()
        print(f"\nTotal interacting cells: {n_interacting} ({100*n_interacting/n_cells:.1f}%)")
        print(f"Total pairs: {len(pairs_df)}")

        return GroundTruthLabels(
            cell_ids=cell_ids,
            is_interacting=is_interacting,
            interaction_type=interaction_type,
            nearest_sender_distance=nearest_sender_dist,
            nearest_sender_id=nearest_sender_id,
            expected_attention_ring=expected_ring,
            receiver_sender_pairs=pairs_df,
        )


def main():
    """CLI for labeling data with ground truth."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Label real data with ground truth interaction status"
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Input file (h5ad or parquet)"
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Output directory for ground truth files"
    )
    parser.add_argument(
        "--cell-type-key", default="cell_type",
        help="Column name for cell types"
    )
    parser.add_argument(
        "--spatial-key", default="spatial",
        help="Key in obsm for spatial coordinates (h5ad only)"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    labeler = GroundTruthLabeler(use_default_rules=True)

    print(f"Loading: {input_path}")
    print(f"Using {len(labeler.rules)} interaction rules:")
    for rule in labeler.rules:
        print(f"  {rule.interaction_name}: {rule.sender_type} -> {rule.receiver_type} (<{rule.max_distance}um)")

    if input_path.suffix == ".h5ad":
        import scanpy as sc
        adata = sc.read_h5ad(input_path)
        print(f"Loaded AnnData: {adata.n_obs} cells, {adata.n_vars} genes")
        labels = labeler.label_anndata(
            adata,
            cell_type_key=args.cell_type_key,
            spatial_key=args.spatial_key,
        )
    elif input_path.suffix == ".parquet":
        df = pd.read_parquet(input_path)
        print(f"Loaded parquet: {len(df)} rows")
        labels = labeler.label_neighborhoods(df, cell_type_key=args.cell_type_key)
    else:
        raise ValueError(f"Unknown file type: {input_path.suffix}")

    labels.save(args.output)


if __name__ == "__main__":
    main()
