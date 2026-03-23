"""
Explainability module for StageBridge.

Provides interpretability tools to support the core claim:
"Cross-sectional progression becomes more identifiable when conditioned on
receiver-centered local niche context."

Two main analysis types:
1. Baseline SHAP Analysis: Show what features drive stage predictions across
   the baseline ladder (PoolingMLP → StageBridge)
2. Sender Influence Analysis: Quantify how different sender cell types
   influence receiver state reconstruction

Usage:
    from stagebridge.analysis.explainability import (
        BaselineSHAPAnalyzer,
        SenderInfluenceAnalyzer,
        plot_sender_influence_heatmap,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
import numpy as np
import pandas as pd


@dataclass
class SHAPResult:
    """Container for SHAP analysis results."""

    model_name: str
    shap_values: np.ndarray  # (n_samples, n_features) or (n_samples, n_features, n_classes)
    feature_names: list[str]
    base_value: float | np.ndarray
    expected_value: float | np.ndarray

    # Aggregated importance
    mean_abs_shap: pd.Series = field(default=None)

    def __post_init__(self):
        if self.mean_abs_shap is None:
            # Compute mean absolute SHAP values per feature
            if self.shap_values.ndim == 3:
                # Multi-class: average across classes
                vals = np.abs(self.shap_values).mean(axis=(0, 2))
            else:
                vals = np.abs(self.shap_values).mean(axis=0)
            self.mean_abs_shap = pd.Series(vals, index=self.feature_names).sort_values(
                ascending=False
            )

    def top_features(self, n: int = 20) -> pd.Series:
        """Get top N most important features."""
        return self.mean_abs_shap.head(n)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert SHAP values to DataFrame."""
        if self.shap_values.ndim == 3:
            # Multi-class: use first class or average
            vals = self.shap_values[:, :, 0]
        else:
            vals = self.shap_values
        return pd.DataFrame(vals, columns=self.feature_names)


@dataclass
class SenderInfluenceResult:
    """Container for sender influence analysis results."""

    receiver_cell_type: str
    sender_influences: pd.DataFrame  # (n_receivers, n_sender_types)
    mean_influence: pd.Series  # Mean influence per sender type
    stage_specific: dict[str, pd.Series] = field(default_factory=dict)  # Per-stage breakdown

    def top_influencers(self, n: int = 5) -> pd.Series:
        """Get top N most influential sender types."""
        return self.mean_influence.sort_values(ascending=False).head(n)


class BaselineSHAPAnalyzer:
    """
    SHAP analysis for baseline ladder models.

    Compares feature importance across:
    - PoolingMLP (no structure)
    - DeepSets (permutation invariance)
    - SetTransformer (flat attention)
    - GraphSAGE (spatial structure)
    - StageBridge (receiver-centered niche)

    Example:
        analyzer = BaselineSHAPAnalyzer()
        results = analyzer.analyze_all_baselines(
            models={'pooling': model1, 'stagebridge': model2},
            X_test=test_features,
            feature_names=gene_names,
        )
        analyzer.plot_comparison(results)
    """

    def __init__(self, background_samples: int = 100, n_shap_samples: int = 500):
        """
        Args:
            background_samples: Number of background samples for SHAP explainer
            n_shap_samples: Number of samples to explain
        """
        self.background_samples = background_samples
        self.n_shap_samples = n_shap_samples
        self._shap = None

    def _get_shap(self):
        """Lazy import of SHAP."""
        if self._shap is None:
            try:
                import shap

                self._shap = shap
            except ImportError as e:
                raise ImportError("SHAP not installed. Install with: pip install shap") from e
        return self._shap

    def analyze_model(
        self,
        model: Any,
        X: np.ndarray,
        feature_names: list[str],
        model_name: str = "model",
        predict_fn: Callable | None = None,
    ) -> SHAPResult:
        """
        Run SHAP analysis on a single model.

        Args:
            model: Trained model with predict or predict_proba method
            X: Feature matrix (n_samples, n_features)
            feature_names: List of feature names
            model_name: Name for this model
            predict_fn: Custom prediction function. If None, uses model.predict

        Returns:
            SHAPResult with SHAP values and feature importance
        """
        shap = self._get_shap()

        # Select background and explanation samples
        n_samples = min(len(X), self.n_shap_samples)
        n_background = min(len(X), self.background_samples)

        bg_idx = np.random.choice(len(X), n_background, replace=False)
        X_background = X[bg_idx]

        explain_idx = np.random.choice(len(X), n_samples, replace=False)
        X_explain = X[explain_idx]

        # Create explainer
        if predict_fn is not None:
            explainer = shap.KernelExplainer(predict_fn, X_background)
        elif hasattr(model, "predict_proba"):
            explainer = shap.KernelExplainer(model.predict_proba, X_background)
        elif hasattr(model, "predict"):
            explainer = shap.KernelExplainer(model.predict, X_background)
        else:
            raise ValueError("Model must have predict or predict_proba method")

        # Compute SHAP values
        shap_values = explainer.shap_values(X_explain)

        # Handle multi-output
        if isinstance(shap_values, list):
            shap_values = np.stack(shap_values, axis=-1)

        return SHAPResult(
            model_name=model_name,
            shap_values=shap_values,
            feature_names=feature_names,
            base_value=explainer.expected_value,
            expected_value=explainer.expected_value,
        )

    def analyze_pytorch_model(
        self,
        model: Any,
        X: np.ndarray,
        feature_names: list[str],
        model_name: str = "model",
        device: str = "cpu",
    ) -> SHAPResult:
        """
        Run SHAP analysis on a PyTorch model.

        Args:
            model: PyTorch model
            X: Feature matrix (n_samples, n_features)
            feature_names: List of feature names
            model_name: Name for this model
            device: Device to run on

        Returns:
            SHAPResult with SHAP values
        """
        self._get_shap()

        try:
            import torch
        except ImportError as e:
            raise ImportError("PyTorch required for PyTorch model analysis") from e

        model.eval()
        model.to(device)

        def predict_fn(x):
            with torch.no_grad():
                x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
                output = model(x_tensor)
                if hasattr(output, "cpu"):
                    output = output.cpu().numpy()
                return output

        return self.analyze_model(
            model=model,
            X=X,
            feature_names=feature_names,
            model_name=model_name,
            predict_fn=predict_fn,
        )

    def analyze_all_baselines(
        self,
        models: dict[str, Any],
        X: np.ndarray,
        feature_names: list[str],
        predict_fns: dict[str, Callable] | None = None,
    ) -> dict[str, SHAPResult]:
        """
        Analyze all baseline models.

        Args:
            models: Dict mapping model names to model objects
            X: Feature matrix
            feature_names: List of feature names
            predict_fns: Optional dict of custom predict functions per model

        Returns:
            Dict mapping model names to SHAPResult
        """
        predict_fns = predict_fns or {}
        results = {}

        for name, model in models.items():
            print(f"Analyzing {name}...")
            results[name] = self.analyze_model(
                model=model,
                X=X,
                feature_names=feature_names,
                model_name=name,
                predict_fn=predict_fns.get(name),
            )

        return results

    def compare_top_features(
        self,
        results: dict[str, SHAPResult],
        n_top: int = 20,
    ) -> pd.DataFrame:
        """
        Compare top features across models.

        Returns DataFrame with feature importance per model.
        """
        comparison = {}
        for name, result in results.items():
            comparison[name] = result.mean_abs_shap

        df = pd.DataFrame(comparison)

        # Get union of top features across all models
        top_features = set()
        for result in results.values():
            top_features.update(result.top_features(n_top).index.tolist())

        return df.loc[list(top_features)].fillna(0)

    def plot_comparison(
        self,
        results: dict[str, SHAPResult],
        n_top: int = 15,
        save_path: Path | None = None,
    ):
        """Plot feature importance comparison across models."""
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError("matplotlib required for plotting") from e

        n_models = len(results)
        fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 6))

        if n_models == 1:
            axes = [axes]

        for ax, (name, result) in zip(axes, results.items()):
            top = result.top_features(n_top)
            ax.barh(range(len(top)), top.values)
            ax.set_yticks(range(len(top)))
            ax.set_yticklabels(top.index)
            ax.invert_yaxis()
            ax.set_xlabel("Mean |SHAP value|")
            ax.set_title(f"{name}")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.show()

        plt.close()


class SenderInfluenceAnalyzer:
    """
    Analyze how different sender cell types influence receiver state.

    This directly supports the core StageBridge claim by quantifying
    niche influence on receiver states.

    Example:
        analyzer = SenderInfluenceAnalyzer(model)
        influence = analyzer.compute_influence(
            receivers=receiver_cells,
            niche_data=niche_features,
            sender_types=['Macrophage', 'Fibroblast', 'T_cell'],
        )
        analyzer.plot_influence_heatmap(influence)
    """

    def __init__(self, model: Any, device: str = "cpu"):
        """
        Args:
            model: Trained StageBridge model with forward method
            device: Device to run on
        """
        self.model = model
        self.device = device

    def compute_influence(
        self,
        receiver_features: np.ndarray,
        niche_features: np.ndarray,
        sender_mask: np.ndarray,
        sender_type_labels: np.ndarray,
        receiver_cell_type: str = "all",
    ) -> SenderInfluenceResult:
        """
        Compute sender influence by masking out each sender type.

        Args:
            receiver_features: Receiver cell features (n_receivers, n_features)
            niche_features: Niche/neighbor features (n_receivers, n_neighbors, n_features)
            sender_mask: Boolean mask for valid neighbors (n_receivers, n_neighbors)
            sender_type_labels: Cell type labels for neighbors (n_receivers, n_neighbors)
            receiver_cell_type: Name of receiver cell type being analyzed

        Returns:
            SenderInfluenceResult with influence scores per sender type
        """
        try:
            import torch
        except ImportError as e:
            raise ImportError("PyTorch required") from e

        self.model.eval()
        self.model.to(self.device)

        # Get unique sender types
        sender_types = np.unique(sender_type_labels[sender_mask])
        sender_types = [st for st in sender_types if st != ""]  # Remove empty

        n_receivers = len(receiver_features)
        influences = np.zeros((n_receivers, len(sender_types)))

        # Baseline prediction (full niche)
        with torch.no_grad():
            receiver_t = torch.tensor(receiver_features, dtype=torch.float32, device=self.device)
            niche_t = torch.tensor(niche_features, dtype=torch.float32, device=self.device)
            mask_t = torch.tensor(sender_mask, dtype=torch.bool, device=self.device)

            baseline_pred = self.model(receiver_t, niche_t, mask_t)
            if hasattr(baseline_pred, "cpu"):
                baseline_pred = baseline_pred.cpu().numpy()

        # Compute influence by masking each sender type
        for i, sender_type in enumerate(sender_types):
            # Create mask that excludes this sender type
            exclude_mask = sender_mask.copy()
            exclude_mask[sender_type_labels == sender_type] = False

            with torch.no_grad():
                mask_excluded = torch.tensor(exclude_mask, dtype=torch.bool, device=self.device)
                pred_excluded = self.model(receiver_t, niche_t, mask_excluded)
                if hasattr(pred_excluded, "cpu"):
                    pred_excluded = pred_excluded.cpu().numpy()

            # Influence = change in prediction when sender type is removed
            # Higher influence = larger change when removed
            if baseline_pred.ndim == 1:
                influence = np.abs(baseline_pred - pred_excluded)
            else:
                influence = np.linalg.norm(baseline_pred - pred_excluded, axis=-1)

            influences[:, i] = influence

        # Create result
        influence_df = pd.DataFrame(influences, columns=sender_types)
        mean_influence = influence_df.mean(axis=0).sort_values(ascending=False)

        return SenderInfluenceResult(
            receiver_cell_type=receiver_cell_type,
            sender_influences=influence_df,
            mean_influence=mean_influence,
        )

    def compute_influence_by_stage(
        self,
        receiver_features: np.ndarray,
        niche_features: np.ndarray,
        sender_mask: np.ndarray,
        sender_type_labels: np.ndarray,
        stage_labels: np.ndarray,
        receiver_cell_type: str = "all",
    ) -> SenderInfluenceResult:
        """
        Compute sender influence broken down by progression stage.

        This reveals how niche influence changes across AAH → AIS → MIA → ADC.

        Args:
            stage_labels: Stage label per receiver (n_receivers,)
            [other args same as compute_influence]

        Returns:
            SenderInfluenceResult with stage_specific breakdown
        """
        # First compute overall influence
        result = self.compute_influence(
            receiver_features=receiver_features,
            niche_features=niche_features,
            sender_mask=sender_mask,
            sender_type_labels=sender_type_labels,
            receiver_cell_type=receiver_cell_type,
        )

        # Break down by stage
        stages = np.unique(stage_labels)
        stage_specific = {}

        for stage in stages:
            stage_mask = stage_labels == stage
            stage_influences = result.sender_influences.loc[stage_mask]
            stage_specific[stage] = stage_influences.mean(axis=0)

        result.stage_specific = stage_specific
        return result

    def plot_influence_heatmap(
        self,
        result: SenderInfluenceResult,
        save_path: Path | None = None,
    ):
        """Plot sender influence as heatmap (stages × sender types)."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError as e:
            raise ImportError("matplotlib and seaborn required for plotting") from e

        if not result.stage_specific:
            # Just plot mean influence as bar chart
            fig, ax = plt.subplots(figsize=(10, 5))
            result.mean_influence.plot(kind="barh", ax=ax)
            ax.set_xlabel("Mean Influence Score")
            ax.set_title(f"Sender Influence on {result.receiver_cell_type}")
            ax.invert_yaxis()
        else:
            # Plot heatmap with stages
            stage_df = pd.DataFrame(result.stage_specific).T

            # Order stages by progression
            stage_order = ["AAH", "AIS", "MIA", "ADC"]
            stage_df = stage_df.reindex([s for s in stage_order if s in stage_df.index])

            fig, ax = plt.subplots(figsize=(12, 5))
            sns.heatmap(
                stage_df,
                annot=True,
                fmt=".2f",
                cmap="YlOrRd",
                ax=ax,
            )
            ax.set_xlabel("Sender Cell Type")
            ax.set_ylabel("Progression Stage")
            ax.set_title(f"Sender Influence on {result.receiver_cell_type} by Stage")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.show()

        plt.close()


class AttentionAnalyzer:
    """
    Analyze attention patterns from transformer-based models.

    Extracts and visualizes attention weights to show what the model
    "looks at" when making predictions.
    """

    def __init__(self, model: Any):
        """
        Args:
            model: Model with attention weights accessible
        """
        self.model = model

    def extract_attention(
        self,
        receiver_features: np.ndarray,
        niche_features: np.ndarray,
        sender_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Extract attention weights from model.

        Returns:
            Attention weights (n_receivers, n_heads, n_neighbors) or
            (n_receivers, n_neighbors) if single head
        """
        try:
            import torch
        except ImportError as e:
            raise ImportError("PyTorch required") from e

        self.model.eval()

        # Hook to capture attention weights
        attention_weights = []

        def attention_hook(module, input, output):
            # Assuming output contains attention weights
            if isinstance(output, tuple) and len(output) > 1:
                attention_weights.append(output[1])  # Usually (batch, heads, seq, seq)

        # Register hooks on attention modules
        hooks = []
        for name, module in self.model.named_modules():
            if "attention" in name.lower() or "attn" in name.lower():
                hooks.append(module.register_forward_hook(attention_hook))

        # Forward pass
        with torch.no_grad():
            receiver_t = torch.tensor(receiver_features, dtype=torch.float32)
            niche_t = torch.tensor(niche_features, dtype=torch.float32)
            mask_t = torch.tensor(sender_mask, dtype=torch.bool)
            _ = self.model(receiver_t, niche_t, mask_t)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        if attention_weights:
            return attention_weights[-1].cpu().numpy()  # Return last layer attention
        else:
            raise ValueError("No attention weights captured. Model may not have attention layers.")

    def plot_attention_by_sender_type(
        self,
        attention: np.ndarray,
        sender_type_labels: np.ndarray,
        save_path: Path | None = None,
    ):
        """
        Aggregate and plot attention by sender cell type.

        Args:
            attention: Attention weights (n_receivers, n_neighbors)
            sender_type_labels: Cell type per neighbor (n_receivers, n_neighbors)
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError("matplotlib required") from e

        # Aggregate attention by sender type
        sender_types = np.unique(sender_type_labels)
        sender_types = [st for st in sender_types if st != ""]

        mean_attention = {}
        for st in sender_types:
            mask = sender_type_labels == st
            if mask.any():
                mean_attention[st] = attention[mask].mean()

        # Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        types = list(mean_attention.keys())
        values = [mean_attention[t] for t in types]

        ax.barh(types, values)
        ax.set_xlabel("Mean Attention Weight")
        ax.set_title("Attention by Sender Cell Type")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.show()

        plt.close()


# Convenience functions


def plot_sender_influence_heatmap(
    influence_result: SenderInfluenceResult,
    save_path: Path | None = None,
):
    """Convenience function to plot sender influence heatmap."""
    analyzer = SenderInfluenceAnalyzer(model=None)  # Model not needed for plotting
    analyzer.plot_influence_heatmap(influence_result, save_path)


def compare_baseline_importance(
    results: dict[str, SHAPResult],
    save_path: Path | None = None,
) -> pd.DataFrame:
    """
    Compare feature importance across baseline models.

    Returns summary table and optionally saves comparison plot.
    """
    analyzer = BaselineSHAPAnalyzer()
    comparison = analyzer.compare_top_features(results)

    if save_path:
        analyzer.plot_comparison(results, save_path=save_path)

    return comparison
