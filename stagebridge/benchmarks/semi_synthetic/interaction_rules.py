"""
Interaction rule application for semi-synthetic benchmark.

Applies explicit sender->receiver interaction rules to assign
interacting vs non-interacting states based on neighborhood composition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from stagebridge.benchmarks.semi_synthetic.configs import InteractionRule


@dataclass
class InteractionResult:
    """Result of applying interaction rules to a cell."""

    cell_id: str
    is_interacting: bool
    triggered_rules: list[str]
    sender_counts: dict[str, int]
    dominant_interaction: str | None
    interaction_strength: float
    stage_context: str | None


@dataclass
class InteractionApplicationReport:
    """Report on interaction rule application."""

    n_cells_processed: int = 0
    n_interacting: int = 0
    n_non_interacting: int = 0
    rule_trigger_counts: dict[str, int] = field(default_factory=dict)
    stage_interaction_rates: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_cells_processed": self.n_cells_processed,
            "n_interacting": self.n_interacting,
            "n_non_interacting": self.n_non_interacting,
            "interaction_rate": self.n_interacting / max(1, self.n_cells_processed),
            "rule_trigger_counts": self.rule_trigger_counts,
            "stage_interaction_rates": self.stage_interaction_rates,
        }


class InteractionRuleEngine:
    """Engine for applying interaction rules to cells in a world."""

    def __init__(
        self,
        rules: list[InteractionRule],
        seed: int = 42,
    ):
        self.rules = rules
        self.rng = np.random.default_rng(seed)

        # Index rules by receiver group for efficient lookup
        self._rules_by_receiver: dict[str, list[InteractionRule]] = {}
        for rule in rules:
            if rule.receiver_group not in self._rules_by_receiver:
                self._rules_by_receiver[rule.receiver_group] = []
            self._rules_by_receiver[rule.receiver_group].append(rule)

    def apply_to_world(
        self,
        cell_positions: pd.DataFrame,
        cell_group_column: str = "cell_group",
        stage_column: str | None = "stage",
    ) -> tuple[pd.DataFrame, InteractionApplicationReport]:
        """Apply interaction rules to all cells in a world.

        Args:
            cell_positions: DataFrame with cell positions and groups
            cell_group_column: Column name for cell group
            stage_column: Optional column name for stage

        Returns:
            Tuple of (updated DataFrame with interaction labels, report)
        """
        report = InteractionApplicationReport()

        # Initialize interaction columns
        cell_positions = cell_positions.copy()
        cell_positions["is_interacting"] = False
        cell_positions["triggered_rules"] = ""
        cell_positions["dominant_interaction"] = None
        cell_positions["interaction_strength"] = 0.0
        cell_positions["n_effective_senders"] = 0

        coords = cell_positions[["x", "y"]].values
        groups = cell_positions[cell_group_column].values
        stages = cell_positions[stage_column].values if stage_column and stage_column in cell_positions.columns else None

        # Process each receiver cell
        receiver_groups = set(self._rules_by_receiver.keys())

        for idx in range(len(cell_positions)):
            cell_group = groups[idx]

            # Skip non-receivers
            if cell_group not in receiver_groups:
                continue

            report.n_cells_processed += 1
            cell_stage = stages[idx] if stages is not None else None

            # Get applicable rules
            applicable_rules = self._rules_by_receiver.get(cell_group, [])
            if not applicable_rules:
                report.n_non_interacting += 1
                continue

            # Check each rule
            result = self._evaluate_rules_for_cell(
                idx, coords, groups, applicable_rules, cell_stage
            )

            # Update cell data
            cell_positions.at[cell_positions.index[idx], "is_interacting"] = result.is_interacting
            cell_positions.at[cell_positions.index[idx], "triggered_rules"] = ",".join(result.triggered_rules)
            cell_positions.at[cell_positions.index[idx], "dominant_interaction"] = result.dominant_interaction
            cell_positions.at[cell_positions.index[idx], "interaction_strength"] = result.interaction_strength

            # Count senders
            total_senders = sum(result.sender_counts.values())
            cell_positions.at[cell_positions.index[idx], "n_effective_senders"] = total_senders

            # Update report
            if result.is_interacting:
                report.n_interacting += 1
                for rule_id in result.triggered_rules:
                    report.rule_trigger_counts[rule_id] = report.rule_trigger_counts.get(rule_id, 0) + 1
            else:
                report.n_non_interacting += 1

        # Compute stage-specific interaction rates
        if stages is not None:
            for stage in np.unique(stages):
                stage_mask = (stages == stage) & (cell_positions[cell_group_column].isin(receiver_groups))
                if stage_mask.sum() > 0:
                    rate = cell_positions.loc[stage_mask, "is_interacting"].mean()
                    report.stage_interaction_rates[str(stage)] = float(rate)

        return cell_positions, report

    def _evaluate_rules_for_cell(
        self,
        cell_idx: int,
        coords: np.ndarray,
        groups: np.ndarray,
        rules: list[InteractionRule],
        cell_stage: str | None,
    ) -> InteractionResult:
        """Evaluate all applicable rules for a single cell."""
        cell_coord = coords[cell_idx]
        triggered_rules = []
        sender_counts: dict[str, int] = {}
        max_strength = 0.0
        dominant_rule = None

        for rule in rules:
            # Count senders within radius
            distances = np.sqrt(((coords - cell_coord) ** 2).sum(axis=1))
            sender_mask = (groups == rule.sender_group) & (distances <= rule.interaction_radius) & (distances > 0)
            n_senders = sender_mask.sum()

            if n_senders > 0:
                sender_counts[rule.sender_group] = sender_counts.get(rule.sender_group, 0) + n_senders

                # Get effective strength (may be stage-modulated)
                effect_strength = rule.get_stage_effect(cell_stage) if cell_stage else rule.effect_strength

                # Probability of interaction increases with sender count
                # Using saturating function: p = strength * (1 - exp(-n_senders / 2))
                interaction_prob = effect_strength * (1 - np.exp(-n_senders / 2))

                # Stochastic determination
                if self.rng.random() < interaction_prob:
                    triggered_rules.append(rule.rule_id)
                    if interaction_prob > max_strength:
                        max_strength = interaction_prob
                        dominant_rule = rule.effect_name

        is_interacting = len(triggered_rules) > 0

        return InteractionResult(
            cell_id=str(cell_idx),
            is_interacting=is_interacting,
            triggered_rules=triggered_rules,
            sender_counts=sender_counts,
            dominant_interaction=dominant_rule,
            interaction_strength=max_strength,
            stage_context=cell_stage,
        )


def compute_ground_truth_labels(
    cell_positions: pd.DataFrame,
    rules: list[InteractionRule],
) -> pd.DataFrame:
    """Compute deterministic ground truth labels (no stochasticity).

    This provides a deterministic version for evaluation where we
    compute what the expected interaction state should be based purely
    on neighborhood composition.
    """
    cell_positions = cell_positions.copy()

    coords = cell_positions[["x", "y"]].values
    groups = cell_positions["cell_group"].values

    # For each rule, compute deterministic influence
    for rule in rules:
        col_name = f"gt_{rule.rule_id}_strength"
        strengths = np.zeros(len(cell_positions))

        for idx in range(len(cell_positions)):
            if groups[idx] != rule.receiver_group:
                continue

            cell_coord = coords[idx]
            distances = np.sqrt(((coords - cell_coord) ** 2).sum(axis=1))
            sender_mask = (groups == rule.sender_group) & (distances <= rule.interaction_radius) & (distances > 0)
            n_senders = sender_mask.sum()

            if n_senders > 0:
                # Deterministic strength based on sender count
                strengths[idx] = rule.effect_strength * (1 - np.exp(-n_senders / 2))

        cell_positions[col_name] = strengths

    # Compute aggregate ground truth
    gt_columns = [col for col in cell_positions.columns if col.startswith("gt_") and col.endswith("_strength")]
    if gt_columns:
        cell_positions["gt_max_interaction_strength"] = cell_positions[gt_columns].max(axis=1)
        cell_positions["gt_should_interact"] = cell_positions["gt_max_interaction_strength"] > 0.3

    return cell_positions


def create_distance_decay_benchmark(
    cell_positions: pd.DataFrame,
    sender_group: str,
    receiver_group: str,
    radii: list[float],
) -> pd.DataFrame:
    """Create benchmark data for testing distance sensitivity.

    For each radius, compute the sender count and expected effect.
    This allows evaluation of whether the model captures distance decay.
    """
    coords = cell_positions[["x", "y"]].values
    groups = cell_positions["cell_group"].values

    receiver_mask = groups == receiver_group

    results = []
    for idx in np.where(receiver_mask)[0]:
        cell_coord = coords[idx]
        distances = np.sqrt(((coords - cell_coord) ** 2).sum(axis=1))
        sender_mask = (groups == sender_group) & (distances > 0)

        record = {
            "cell_idx": idx,
            "cell_id": cell_positions.index[idx] if hasattr(cell_positions, 'index') else idx,
        }

        for radius in radii:
            n_senders = ((sender_mask) & (distances <= radius)).sum()
            record[f"n_senders_r{int(radius)}"] = n_senders

        results.append(record)

    return pd.DataFrame(results)
