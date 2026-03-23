"""
Ground truth management for unified benchmark.

Combines recoverable parameters from:
- Suite A: Flow field dynamics (drift, diffusion, stage centroids)
- Suite B: Niche influence vectors (causal sender->receiver effects)
- Suite C: Clone structure (evolutionary compatibility)
- Suite D: Spatial interaction rules
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stagebridge.benchmarks.unified.config import (
    UnifiedBenchmarkConfig,
)


@dataclass
class GroundTruth:
    """Complete ground truth for benchmark evaluation.

    Contains all recoverable parameters that models should learn.
    """

    # Suite A: Flow field dynamics
    stage_centroids: dict[str, list[float]] = field(default_factory=dict)
    drift_strength: float = 1.0
    diffusion_strength: float = 0.2
    flow_field_type: str = "linear"

    # Suite B: Niche influence vectors
    influence_vectors: dict[str, list[float]] = field(default_factory=dict)
    influence_strengths: dict[str, float] = field(default_factory=dict)
    influential_celltypes: list[str] = field(default_factory=list)

    # Suite C: Clone structure
    clone_signatures: dict[str, list[float]] = field(default_factory=dict)
    clone_divergence: float = 0.3
    donor_clone_mapping: dict[str, list[str]] = field(default_factory=dict)

    # Suite D: Interaction rules
    interaction_rules: list[dict[str, Any]] = field(default_factory=list)
    interaction_radii: dict[str, float] = field(default_factory=dict)

    # Reference geometry
    luca_rotation: list[list[float]] | None = None
    luca_shift: list[float] | None = None

    # Generation metadata
    config_summary: dict[str, Any] = field(default_factory=dict)
    generation_seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "flow_field": {
                "stage_centroids": self.stage_centroids,
                "drift_strength": self.drift_strength,
                "diffusion_strength": self.diffusion_strength,
                "flow_field_type": self.flow_field_type,
            },
            "niche_influence": {
                "influence_vectors": self.influence_vectors,
                "influence_strengths": self.influence_strengths,
                "influential_celltypes": self.influential_celltypes,
            },
            "clone_structure": {
                "clone_signatures": self.clone_signatures,
                "clone_divergence": self.clone_divergence,
                "donor_clone_mapping": self.donor_clone_mapping,
            },
            "interaction_rules": {
                "rules": self.interaction_rules,
                "radii": self.interaction_radii,
            },
            "reference_geometry": {
                "luca_rotation": self.luca_rotation,
                "luca_shift": self.luca_shift,
            },
            "metadata": {
                "config_summary": self.config_summary,
                "generation_seed": self.generation_seed,
            },
        }

    def save(self, path: Path) -> None:
        """Save ground truth to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "GroundTruth":
        """Load ground truth from JSON file."""
        with open(path) as f:
            data = json.load(f)

        return cls(
            stage_centroids=data.get("flow_field", {}).get("stage_centroids", {}),
            drift_strength=data.get("flow_field", {}).get("drift_strength", 1.0),
            diffusion_strength=data.get("flow_field", {}).get("diffusion_strength", 0.2),
            flow_field_type=data.get("flow_field", {}).get("flow_field_type", "linear"),
            influence_vectors=data.get("niche_influence", {}).get("influence_vectors", {}),
            influence_strengths=data.get("niche_influence", {}).get("influence_strengths", {}),
            influential_celltypes=data.get("niche_influence", {}).get("influential_celltypes", []),
            clone_signatures=data.get("clone_structure", {}).get("clone_signatures", {}),
            clone_divergence=data.get("clone_structure", {}).get("clone_divergence", 0.3),
            donor_clone_mapping=data.get("clone_structure", {}).get("donor_clone_mapping", {}),
            interaction_rules=data.get("interaction_rules", {}).get("rules", []),
            interaction_radii=data.get("interaction_rules", {}).get("radii", {}),
            luca_rotation=data.get("reference_geometry", {}).get("luca_rotation"),
            luca_shift=data.get("reference_geometry", {}).get("luca_shift"),
            config_summary=data.get("metadata", {}).get("config_summary", {}),
            generation_seed=data.get("metadata", {}).get("generation_seed", 42),
        )


@dataclass
class RecoveryMetrics:
    """Metrics for evaluating ground truth recovery."""

    # Flow field recovery
    centroid_correlation: float = 0.0
    drift_direction_cosine: float = 0.0
    flow_rmse: float = 0.0

    # Niche influence recovery
    influence_direction_cosines: dict[str, float] = field(default_factory=dict)
    influence_strength_correlation: float = 0.0
    influential_celltype_precision: float = 0.0
    influential_celltype_recall: float = 0.0

    # Clone recovery
    clone_compatibility_auc: float = 0.0
    matched_vs_shuffled_gap: float = 0.0

    # Interaction rule recovery
    radius_sensitivity_correlation: float = 0.0
    stage_modulation_correlation: float = 0.0

    # Overall
    composite_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "flow_field": {
                "centroid_correlation": self.centroid_correlation,
                "drift_direction_cosine": self.drift_direction_cosine,
                "flow_rmse": self.flow_rmse,
            },
            "niche_influence": {
                "influence_direction_cosines": self.influence_direction_cosines,
                "influence_strength_correlation": self.influence_strength_correlation,
                "influential_celltype_precision": self.influential_celltype_precision,
                "influential_celltype_recall": self.influential_celltype_recall,
            },
            "clone_structure": {
                "clone_compatibility_auc": self.clone_compatibility_auc,
                "matched_vs_shuffled_gap": self.matched_vs_shuffled_gap,
            },
            "interaction_rules": {
                "radius_sensitivity_correlation": self.radius_sensitivity_correlation,
                "stage_modulation_correlation": self.stage_modulation_correlation,
            },
            "composite_score": self.composite_score,
        }


class GroundTruthRecovery:
    """Evaluate model recovery of ground truth parameters."""

    def __init__(self, ground_truth: GroundTruth):
        self.gt = ground_truth

    def evaluate_flow_field_recovery(
        self,
        predicted_centroids: dict[str, np.ndarray] | None = None,
        predicted_transitions: pd.DataFrame | None = None,
    ) -> dict[str, float]:
        """Evaluate recovery of flow field dynamics.

        Args:
            predicted_centroids: Model's predicted stage centroids
            predicted_transitions: DataFrame with (z_source, z_predicted_target)

        Returns:
            Dictionary of flow field metrics
        """
        metrics = {}

        if predicted_centroids is not None:
            # Compare centroids
            gt_centroids = {
                stage: np.array(coords) for stage, coords in self.gt.stage_centroids.items()
            }

            correlations = []
            for stage in gt_centroids:
                if stage in predicted_centroids:
                    gt = gt_centroids[stage]
                    pred = predicted_centroids[stage]
                    min_dim = min(len(gt), len(pred))
                    corr = np.corrcoef(gt[:min_dim], pred[:min_dim])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)

            metrics["centroid_correlation"] = float(np.mean(correlations)) if correlations else 0.0

        if predicted_transitions is not None and len(predicted_transitions) > 0:
            # Evaluate predicted transitions against ground truth drift
            if (
                "z_source" in predicted_transitions.columns
                and "z_predicted_target" in predicted_transitions.columns
            ):
                z_src = np.stack(predicted_transitions["z_source"].values)
                z_pred = np.stack(predicted_transitions["z_predicted_target"].values)
                z_gt_target = (
                    np.stack(predicted_transitions["z_target"].values)
                    if "z_target" in predicted_transitions.columns
                    else None
                )

                if z_gt_target is not None:
                    # RMSE between predicted and actual targets
                    rmse = np.sqrt(np.mean((z_pred - z_gt_target) ** 2))
                    metrics["flow_rmse"] = float(rmse)

                    # Direction cosine (average cosine similarity of drift vectors)
                    pred_drift = z_pred - z_src
                    gt_drift = z_gt_target - z_src

                    pred_norm = pred_drift / (
                        np.linalg.norm(pred_drift, axis=1, keepdims=True) + 1e-8
                    )
                    gt_norm = gt_drift / (np.linalg.norm(gt_drift, axis=1, keepdims=True) + 1e-8)

                    cosines = (pred_norm * gt_norm).sum(axis=1)
                    metrics["drift_direction_cosine"] = float(np.mean(cosines))

        return metrics

    def evaluate_niche_influence_recovery(
        self,
        predicted_influence_vectors: dict[str, np.ndarray] | None = None,
        predicted_influential_celltypes: list[str] | None = None,
        receiver_predictions: pd.DataFrame | None = None,
    ) -> dict[str, float]:
        """Evaluate recovery of niche influence parameters (Suite B).

        Args:
            predicted_influence_vectors: Model's learned influence directions
            predicted_influential_celltypes: Model's identified influential cell types
            receiver_predictions: DataFrame with receiver state predictions

        Returns:
            Dictionary of niche influence metrics
        """
        metrics = {}

        if predicted_influence_vectors is not None:
            # Compare influence vectors (direction cosines)
            gt_vectors = {ct: np.array(vec) for ct, vec in self.gt.influence_vectors.items()}

            direction_cosines = {}
            for ct in gt_vectors:
                if ct in predicted_influence_vectors:
                    gt = gt_vectors[ct]
                    pred = predicted_influence_vectors[ct]
                    min_dim = min(len(gt), len(pred))

                    gt_norm = gt[:min_dim] / (np.linalg.norm(gt[:min_dim]) + 1e-8)
                    pred_norm = pred[:min_dim] / (np.linalg.norm(pred[:min_dim]) + 1e-8)

                    cosine = float(np.dot(gt_norm, pred_norm))
                    direction_cosines[ct] = cosine

            metrics["influence_direction_cosines"] = direction_cosines
            if direction_cosines:
                metrics["mean_direction_cosine"] = float(np.mean(list(direction_cosines.values())))

        if predicted_influential_celltypes is not None:
            gt_set = set(self.gt.influential_celltypes)
            pred_set = set(predicted_influential_celltypes)

            if len(pred_set) > 0:
                precision = len(gt_set & pred_set) / len(pred_set)
                metrics["influential_celltype_precision"] = precision
            else:
                metrics["influential_celltype_precision"] = 0.0

            if len(gt_set) > 0:
                recall = len(gt_set & pred_set) / len(gt_set)
                metrics["influential_celltype_recall"] = recall
            else:
                metrics["influential_celltype_recall"] = 0.0

        return metrics

    def evaluate_clone_recovery(
        self,
        compatibility_scores: pd.DataFrame | None = None,
    ) -> dict[str, float]:
        """Evaluate recovery of clone compatibility (Suite C).

        Args:
            compatibility_scores: DataFrame with (donor_a, donor_b, is_matched, score)

        Returns:
            Dictionary of clone recovery metrics
        """
        metrics = {}

        if compatibility_scores is not None and "is_matched" in compatibility_scores.columns:
            matched = compatibility_scores[compatibility_scores["is_matched"]]
            shuffled = compatibility_scores[~compatibility_scores["is_matched"]]

            if len(matched) > 0 and len(shuffled) > 0:
                # Matched should have higher compatibility scores
                matched_mean = matched["score"].mean() if "score" in matched.columns else 0.0
                shuffled_mean = shuffled["score"].mean() if "score" in shuffled.columns else 0.0

                metrics["matched_vs_shuffled_gap"] = float(matched_mean - shuffled_mean)

                # AUC for distinguishing matched vs shuffled
                try:
                    from sklearn.metrics import roc_auc_score

                    if len(matched) > 0 and len(shuffled) > 0:
                        labels = np.concatenate(
                            [
                                np.ones(len(matched)),
                                np.zeros(len(shuffled)),
                            ]
                        )
                        scores = np.concatenate(
                            [
                                matched["score"].values,
                                shuffled["score"].values,
                            ]
                        )
                        auc = roc_auc_score(labels, scores)
                        metrics["clone_compatibility_auc"] = float(auc)
                except Exception:
                    pass

        return metrics

    def evaluate_interaction_rules(
        self,
        predicted_interactions: pd.DataFrame | None = None,
        ground_truth_interactions: pd.DataFrame | None = None,
    ) -> dict[str, float]:
        """Evaluate recovery of interaction rule parameters.

        Args:
            predicted_interactions: Model predictions for receiver interactions
            ground_truth_interactions: Ground truth interaction labels

        Returns:
            Dictionary of interaction rule metrics
        """
        from stagebridge.benchmarks.semi_synthetic.metrics import (
            evaluate_receiver_state_recovery,
        )

        metrics = {}

        if predicted_interactions is not None and ground_truth_interactions is not None:
            if "predicted_interacting" in predicted_interactions.columns:
                pred = predicted_interactions["predicted_interacting"]
                gt = ground_truth_interactions.get(
                    "is_interacting", ground_truth_interactions.get("gt_should_interact")
                )

                if gt is not None:
                    state_metrics = evaluate_receiver_state_recovery(
                        pred,
                        gt,
                        predicted_interactions.get("predicted_prob"),
                    )
                    metrics["receiver_accuracy"] = state_metrics.accuracy
                    metrics["receiver_f1"] = state_metrics.f1
                    metrics["receiver_auroc"] = state_metrics.auroc

        return metrics

    def compute_composite_score(
        self,
        flow_metrics: dict[str, float] | None = None,
        niche_metrics: dict[str, float] | None = None,
        clone_metrics: dict[str, float] | None = None,
        interaction_metrics: dict[str, float] | None = None,
        weights: dict[str, float] | None = None,
    ) -> RecoveryMetrics:
        """Compute composite recovery score.

        Args:
            flow_metrics: Flow field recovery metrics
            niche_metrics: Niche influence recovery metrics
            clone_metrics: Clone recovery metrics
            interaction_metrics: Interaction rule metrics
            weights: Optional custom weights

        Returns:
            RecoveryMetrics with all scores
        """
        if weights is None:
            weights = {
                "flow": 0.2,
                "niche": 0.3,
                "clone": 0.2,
                "interaction": 0.3,
            }

        result = RecoveryMetrics()

        score_components = []

        if flow_metrics:
            result.centroid_correlation = flow_metrics.get("centroid_correlation", 0.0)
            result.drift_direction_cosine = flow_metrics.get("drift_direction_cosine", 0.0)
            result.flow_rmse = flow_metrics.get("flow_rmse", 0.0)

            flow_score = (result.centroid_correlation + result.drift_direction_cosine) / 2
            score_components.append(("flow", flow_score, weights["flow"]))

        if niche_metrics:
            result.influence_direction_cosines = niche_metrics.get(
                "influence_direction_cosines", {}
            )
            result.influence_strength_correlation = niche_metrics.get("mean_direction_cosine", 0.0)
            result.influential_celltype_precision = niche_metrics.get(
                "influential_celltype_precision", 0.0
            )
            result.influential_celltype_recall = niche_metrics.get(
                "influential_celltype_recall", 0.0
            )

            niche_score = (
                result.influence_strength_correlation
                + result.influential_celltype_precision
                + result.influential_celltype_recall
            ) / 3
            score_components.append(("niche", niche_score, weights["niche"]))

        if clone_metrics:
            result.clone_compatibility_auc = clone_metrics.get("clone_compatibility_auc", 0.0)
            result.matched_vs_shuffled_gap = clone_metrics.get("matched_vs_shuffled_gap", 0.0)

            clone_score = result.clone_compatibility_auc
            score_components.append(("clone", clone_score, weights["clone"]))

        if interaction_metrics:
            result.radius_sensitivity_correlation = interaction_metrics.get(
                "radius_sensitivity_correlation", 0.0
            )
            result.stage_modulation_correlation = interaction_metrics.get(
                "stage_modulation_correlation", 0.0
            )

            interaction_score = interaction_metrics.get("receiver_auroc", 0.0)
            score_components.append(("interaction", interaction_score, weights["interaction"]))

        # Weighted composite
        if score_components:
            total_weight = sum(w for _, _, w in score_components)
            result.composite_score = sum(s * w for _, s, w in score_components) / total_weight

        return result


def build_ground_truth_from_config(
    config: UnifiedBenchmarkConfig,
    rng: np.random.Generator,
) -> GroundTruth:
    """Build ground truth structure from configuration.

    This initializes the ground truth with known parameters that
    will be used during data generation.
    """
    gt = GroundTruth(generation_seed=config.seed)

    # Flow field (Suite A)
    gt.drift_strength = config.dynamics.drift_strength
    gt.diffusion_strength = config.dynamics.diffusion_strength
    gt.flow_field_type = config.dynamics.flow_field_type

    # Initialize stage centroids
    base_trajectory = np.linspace(0, 3, len(config.stages))
    for i, stage in enumerate(config.stages):
        centroid = np.zeros(config.latent_dim)
        centroid[0] = base_trajectory[i]
        if i >= 2:
            centroid[1] = 0.5 * (i - 1)
        gt.stage_centroids[stage] = centroid.tolist()

    # Niche influence (Suite B)
    sender_groups = [g.name for g in config.cell_groups if g.role == "sender"]
    gt.influential_celltypes = sender_groups

    for rule in config.interaction_rules:
        if rule.niche_influence is not None:
            # Generate random influence direction
            direction = rng.standard_normal(config.latent_dim)
            direction = direction / np.linalg.norm(direction)
            direction = direction * rule.niche_influence.strength

            gt.influence_vectors[rule.niche_influence.influence_name] = direction.tolist()
            gt.influence_strengths[rule.niche_influence.influence_name] = (
                rule.niche_influence.strength
            )

    # Clone structure (Suite C)
    gt.clone_divergence = config.dynamics.clone_divergence

    # Interaction rules (Suite D)
    for rule in config.interaction_rules:
        gt.interaction_rules.append(
            {
                "rule_id": rule.rule_id,
                "sender_group": rule.sender_group,
                "receiver_group": rule.receiver_group,
                "interaction_radius": rule.interaction_radius,
                "effect_strength": rule.effect_strength,
                "effect_name": rule.effect_name,
            }
        )
        gt.interaction_radii[rule.rule_id] = rule.interaction_radius

    # Reference geometry
    rotation_angle = config.dynamics.hlca_luca_rotation
    rotation_2d = [
        [float(np.cos(rotation_angle)), float(-np.sin(rotation_angle))],
        [float(np.sin(rotation_angle)), float(np.cos(rotation_angle))],
    ]
    gt.luca_rotation = rotation_2d

    shift = np.zeros(min(4, config.latent_dim))
    shift[0] = config.dynamics.hlca_luca_shift
    gt.luca_shift = shift.tolist()

    # Config summary
    gt.config_summary = {
        "benchmark_name": config.benchmark_name,
        "mode": config.mode,
        "difficulty": config.difficulty,
        "n_cells": config.n_cells,
        "n_donors": config.n_donors,
        "n_stages": len(config.stages),
        "latent_dim": config.latent_dim,
    }

    return gt
