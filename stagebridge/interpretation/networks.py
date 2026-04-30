"""Cell-cell interaction network inference for StageBridge.

Adapted from AMICI's interaction network analysis for:
- Visium spot-level data (deconvolved cell type proportions)
- Ring-based spatial structure
- Stage progression context

Key outputs:
- Interaction weight matrices between cell types
- Directed graphs showing information flow
- Stage-specific interaction changes
- Progression-associated interaction rewiring
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal
import warnings

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, pearsonr
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

from stagebridge.contracts import IDX_TO_STAGE

if TYPE_CHECKING:
    from stagebridge.models import StageBridge
    import networkx as nx


@dataclass
class InteractionEdge:
    """Directed edge in interaction network.

    Represents information flow: sender -> receiver.
    In StageBridge context: neighbor cell type influences receiver reconstruction.
    """
    sender: str
    receiver: str
    weight: float
    p_value: float
    stage: str | None = None
    n_samples: int = 0


@dataclass
class InteractionNetwork:
    """Cell-cell interaction network from attention patterns.

    Built by correlating:
    - Attention to neighbors of cell type X
    - Reconstruction quality improvement

    Higher attention to cell type X AND better reconstruction
    suggests X is providing useful information to the receiver.

    Attributes:
        edges: List of interaction edges
        weight_matrix: Cell type x cell type weight matrix
        cell_types: List of cell types
        stage_networks: Optional per-stage breakdown
    """
    edges: list[InteractionEdge] = field(default_factory=list)
    weight_matrix: pd.DataFrame | None = None
    cell_types: list[str] = field(default_factory=list)
    stage_networks: dict[str, "InteractionNetwork"] = field(default_factory=dict)

    @classmethod
    def from_attention_and_composition(
        cls,
        attention_df: pd.DataFrame,
        composition_df: pd.DataFrame,
        cell_type_col: str = "cell_type",
        min_correlation: float = 0.1,
        p_threshold: float = 0.05,
        method: Literal["spearman", "pearson"] = "spearman",
    ) -> "InteractionNetwork":
        """Build network from attention patterns and niche composition.

        For each receiver cell type R and potential sender S:
        1. Get cells of type R
        2. Correlate attention to ring tokens with proportion of S in rings
        3. High positive correlation = S influences R

        Args:
            attention_df: DataFrame with attention columns (attn_ring1, etc.)
                         and cell_type column
            composition_df: DataFrame with cell type proportions per ring
                           (ring1_S, ring2_S, etc. for each sender type S)
            cell_type_col: Column name for receiver cell type
            min_correlation: Minimum correlation to include edge
            p_threshold: P-value threshold after FDR correction
            method: Correlation method

        Returns:
            InteractionNetwork
        """
        if cell_type_col not in attention_df.columns:
            raise ValueError(f"Column {cell_type_col} not in attention_df")

        corr_func = spearmanr if method == "spearman" else pearsonr

        ring_attn_cols = sorted([c for c in attention_df.columns if c.startswith("attn_ring")])
        if not ring_attn_cols:
            warnings.warn("No ring attention columns found")
            return cls()

        sender_types = set()
        for col in composition_df.columns:
            if col.startswith("ring") and "_" in col:
                parts = col.split("_", 1)
                if len(parts) == 2:
                    sender_types.add(parts[1])

        receiver_types = attention_df[cell_type_col].unique()
        all_cell_types = sorted(set(sender_types) | set(receiver_types))

        edges = []
        weight_data = []

        for receiver in receiver_types:
            receiver_mask = attention_df[cell_type_col] == receiver
            receiver_attn = attention_df.loc[receiver_mask, ring_attn_cols]

            if len(receiver_attn) < 10:
                continue

            receiver_comp = composition_df.loc[receiver_mask]

            for sender in sender_types:
                if sender == receiver:
                    continue

                sender_cols = [c for c in composition_df.columns if c.endswith(f"_{sender}")]
                if not sender_cols:
                    continue

                sender_proportion = receiver_comp[sender_cols].sum(axis=1)

                if sender_proportion.std() < 1e-6:
                    continue

                total_attn = receiver_attn.sum(axis=1)

                valid_mask = ~(total_attn.isna() | sender_proportion.isna())
                if valid_mask.sum() < 10:
                    continue

                corr, p_val = corr_func(
                    total_attn[valid_mask],
                    sender_proportion[valid_mask],
                )

                if np.isnan(corr):
                    continue

                edges.append(InteractionEdge(
                    sender=sender,
                    receiver=receiver,
                    weight=float(corr),
                    p_value=float(p_val),
                    n_samples=int(valid_mask.sum()),
                ))

                weight_data.append({
                    "sender": sender,
                    "receiver": receiver,
                    "weight": corr,
                    "p_value": p_val,
                })

        if edges:
            p_values = [e.p_value for e in edges]
            _, p_adj, _, _ = multipletests(p_values, method="fdr_bh")
            for edge, p_a in zip(edges, p_adj):
                edge.p_value = p_a

        significant_edges = [
            e for e in edges
            if abs(e.weight) >= min_correlation and e.p_value < p_threshold
        ]

        weight_matrix = pd.DataFrame(
            0.0,
            index=all_cell_types,
            columns=all_cell_types,
        )
        for e in significant_edges:
            if e.sender in weight_matrix.index and e.receiver in weight_matrix.columns:
                weight_matrix.loc[e.sender, e.receiver] = e.weight

        return cls(
            edges=significant_edges,
            weight_matrix=weight_matrix,
            cell_types=all_cell_types,
        )

    @classmethod
    def from_ablation_results(
        cls,
        ablation_df: pd.DataFrame,
        receiver_cell_types: pd.Series,
        sender_mapping: dict[str, str],
        significance_threshold: float = 0.05,
    ) -> "InteractionNetwork":
        """Build network from token ablation analysis.

        If ablating a token (e.g., ring containing cell type X) increases
        reconstruction error for receiver type Y, then X->Y interaction exists.

        Args:
            ablation_df: Per-sample ablation results with delta_loss columns
            receiver_cell_types: Cell type labels for each sample
            sender_mapping: Maps token names to dominant cell types
            significance_threshold: P-value threshold

        Returns:
            InteractionNetwork
        """
        delta_cols = [c for c in ablation_df.columns if c.endswith("_delta")]
        if not delta_cols:
            return cls()

        edges = []
        receiver_types = receiver_cell_types.unique()

        for receiver in receiver_types:
            receiver_mask = receiver_cell_types == receiver
            receiver_data = ablation_df.loc[receiver_mask]

            if len(receiver_data) < 10:
                continue

            for delta_col in delta_cols:
                token_name = delta_col.replace("_delta", "")
                sender = sender_mapping.get(token_name)

                if sender is None or sender == receiver:
                    continue

                delta_values = receiver_data[delta_col].dropna()
                if len(delta_values) < 10:
                    continue

                mean_delta = delta_values.mean()
                from scipy.stats import ttest_1samp
                _, p_val = ttest_1samp(delta_values, 0, alternative="greater")

                edges.append(InteractionEdge(
                    sender=sender,
                    receiver=receiver,
                    weight=float(mean_delta),
                    p_value=float(p_val),
                    n_samples=len(delta_values),
                ))

        if edges:
            p_values = [e.p_value for e in edges]
            _, p_adj, _, _ = multipletests(p_values, method="fdr_bh")
            for edge, p_a in zip(edges, p_adj):
                edge.p_value = p_a

        significant_edges = [
            e for e in edges if e.p_value < significance_threshold and e.weight > 0
        ]

        cell_types = sorted(set([e.sender for e in significant_edges] + [e.receiver for e in significant_edges]))

        weight_matrix = pd.DataFrame(0.0, index=cell_types, columns=cell_types)
        for e in significant_edges:
            weight_matrix.loc[e.sender, e.receiver] = e.weight

        return cls(
            edges=significant_edges,
            weight_matrix=weight_matrix,
            cell_types=cell_types,
        )

    def to_networkx(self, weight_threshold: float = 0.0) -> "nx.DiGraph":
        """Convert to NetworkX directed graph.

        Args:
            weight_threshold: Minimum edge weight to include

        Returns:
            nx.DiGraph with nodes=cell_types, edges=interactions
        """
        import networkx as nx

        G = nx.DiGraph()
        G.add_nodes_from(self.cell_types)

        for edge in self.edges:
            if abs(edge.weight) >= weight_threshold:
                G.add_edge(
                    edge.sender,
                    edge.receiver,
                    weight=edge.weight,
                    p_value=edge.p_value,
                    n_samples=edge.n_samples,
                )

        return G

    def to_dataframe(self) -> pd.DataFrame:
        """Convert edges to DataFrame."""
        return pd.DataFrame([
            {
                "sender": e.sender,
                "receiver": e.receiver,
                "weight": e.weight,
                "p_value": e.p_value,
                "stage": e.stage,
                "n_samples": e.n_samples,
            }
            for e in self.edges
        ])

    def get_hub_scores(self) -> pd.DataFrame:
        """Compute hub scores for each cell type.

        Returns DataFrame with:
        - out_degree: Number of outgoing edges (sender activity)
        - in_degree: Number of incoming edges (receiver activity)
        - out_strength: Sum of outgoing weights
        - in_strength: Sum of incoming weights
        - hub_score: out_strength + in_strength
        """
        G = self.to_networkx()

        scores = []
        for node in G.nodes():
            out_edges = list(G.out_edges(node, data=True))
            in_edges = list(G.in_edges(node, data=True))

            out_strength = sum(d.get("weight", 0) for _, _, d in out_edges)
            in_strength = sum(d.get("weight", 0) for _, _, d in in_edges)

            scores.append({
                "cell_type": node,
                "out_degree": len(out_edges),
                "in_degree": len(in_edges),
                "out_strength": out_strength,
                "in_strength": in_strength,
                "hub_score": out_strength + in_strength,
            })

        return pd.DataFrame(scores).sort_values("hub_score", ascending=False)

    def compare_stages(self, other: "InteractionNetwork", other_stage: str) -> pd.DataFrame:
        """Compare interaction weights between two stages.

        Returns DataFrame showing edges that changed significantly.
        """
        self_df = self.to_dataframe()
        other_df = other.to_dataframe()

        if self_df.empty or other_df.empty:
            return pd.DataFrame()

        merged = pd.merge(
            self_df[["sender", "receiver", "weight"]],
            other_df[["sender", "receiver", "weight"]],
            on=["sender", "receiver"],
            how="outer",
            suffixes=("_self", f"_{other_stage}"),
        ).fillna(0)

        merged["delta"] = merged[f"weight_{other_stage}"] - merged["weight_self"]
        merged["abs_delta"] = merged["delta"].abs()

        return merged.sort_values("abs_delta", ascending=False)


def build_interaction_network(
    attention_df: pd.DataFrame,
    composition_df: pd.DataFrame,
    cell_type_col: str = "cell_type",
    by_stage: bool = True,
    stage_col: str = "stage_idx",
) -> InteractionNetwork | dict[str, InteractionNetwork]:
    """Convenience function to build interaction networks.

    Args:
        attention_df: Attention patterns from AttentionModule
        composition_df: Cell type proportions per ring (from spatial mapping)
        cell_type_col: Receiver cell type column
        by_stage: Build separate networks per stage
        stage_col: Stage column name

    Returns:
        Single InteractionNetwork or dict of stage->network
    """
    if not by_stage:
        return InteractionNetwork.from_attention_and_composition(
            attention_df=attention_df,
            composition_df=composition_df,
            cell_type_col=cell_type_col,
        )

    networks = {}
    for stage_idx in sorted(attention_df[stage_col].unique()):
        stage_name = IDX_TO_STAGE.get(stage_idx, f"stage_{stage_idx}")

        stage_attn = attention_df[attention_df[stage_col] == stage_idx]
        stage_comp = composition_df.loc[stage_attn.index]

        network = InteractionNetwork.from_attention_and_composition(
            attention_df=stage_attn,
            composition_df=stage_comp,
            cell_type_col=cell_type_col,
        )

        for edge in network.edges:
            edge.stage = stage_name

        networks[stage_name] = network

    return networks
