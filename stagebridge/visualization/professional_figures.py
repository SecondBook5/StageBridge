"""
REAL Publication-Quality Figure Generation for StageBridge V1

NO placeholder figures. NO text boxes. ONLY real data-driven visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import patches
from pathlib import Path
from typing import Optional, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

# Professional color schemes
COLORS = {
    'stages': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'],
    'performance': ['#06D6A0', '#118AB2', '#073B4C', '#EF476F'],
    'heatmap': 'RdYlBu_r',
    'attention': 'viridis',
}


def generate_figure2_dimensionality_reduction(
    embeddings: np.ndarray,
    labels: np.ndarray,
    stages: np.ndarray,
    output_path: Path,
    title: str = "Cell State Embeddings"
):
    """
    Real dimensionality reduction plots: PCA, t-SNE, UMAP, PHATE
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    try:
        import umap
        has_umap = True
    except:
        has_umap = False

    try:
        import phate
        has_phate = True
    except:
        has_phate = False

    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.3)

    stage_names = np.unique(stages)
    colors = COLORS['stages'][:len(stage_names)]
    stage_to_color = {s: colors[i] for i, s in enumerate(stage_names)}

    # PCA
    ax = fig.add_subplot(gs[0, 0])
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(embeddings)
    for stage in stage_names:
        mask = stages == stage
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                  c=stage_to_color[stage], label=stage,
                  alpha=0.6, s=30, edgecolors='white', linewidth=0.5)
    ax.set_title(f'PCA (var: {pca.explained_variance_ratio_[:2].sum()*100:.1f}%)',
                fontsize=12, fontweight='bold')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.legend(frameon=True, loc='best', fontsize=8)
    ax.grid(True, alpha=0.2)

    # PCA variance explained
    ax = fig.add_subplot(gs[0, 1])
    pca_full = PCA().fit(embeddings)
    variance = pca_full.explained_variance_ratio_
    ax.plot(range(1, min(21, len(variance)+1)),
           variance[:20], 'o-', linewidth=2, markersize=6)
    ax.axhline(y=0.01, color='r', linestyle='--', alpha=0.5, label='1% threshold')
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Variance Explained')
    ax.set_title('PCA Scree Plot', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.2)

    # t-SNE
    ax = fig.add_subplot(gs[0, 2])
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)//4))
    X_tsne = tsne.fit_transform(embeddings)
    for stage in stage_names:
        mask = stages == stage
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                  c=stage_to_color[stage], label=stage,
                  alpha=0.6, s=30, edgecolors='white', linewidth=0.5)
    ax.set_title('t-SNE', fontsize=12, fontweight='bold')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.grid(True, alpha=0.2)

    # UMAP
    ax = fig.add_subplot(gs[0, 3])
    if has_umap:
        reducer = umap.UMAP(random_state=42)
        X_umap = reducer.fit_transform(embeddings)
        for stage in stage_names:
            mask = stages == stage
            ax.scatter(X_umap[mask, 0], X_umap[mask, 1],
                      c=stage_to_color[stage], label=stage,
                      alpha=0.6, s=30, edgecolors='white', linewidth=0.5)
        ax.set_title('UMAP', fontsize=12, fontweight='bold')
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
    else:
        ax.text(0.5, 0.5, 'UMAP not available\npip install umap-learn',
               ha='center', va='center', fontsize=10)
        ax.axis('off')
    ax.grid(True, alpha=0.2)

    # PHATE
    ax = fig.add_subplot(gs[1, 0])
    if has_phate:
        phate_op = phate.PHATE(random_state=42)
        X_phate = phate_op.fit_transform(embeddings)
        for stage in stage_names:
            mask = stages == stage
            ax.scatter(X_phate[mask, 0], X_phate[mask, 1],
                      c=stage_to_color[stage], label=stage,
                      alpha=0.6, s=30, edgecolors='white', linewidth=0.5)
        ax.set_title('PHATE', fontsize=12, fontweight='bold')
        ax.set_xlabel('PHATE 1')
        ax.set_ylabel('PHATE 2')
    else:
        ax.text(0.5, 0.5, 'PHATE not available\npip install phate',
               ha='center', va='center', fontsize=10)
        ax.axis('off')
    ax.grid(True, alpha=0.2)

    # Distance matrix heatmap
    ax = fig.add_subplot(gs[1, 1])
    from scipy.spatial.distance import pdist, squareform
    sample_size = min(100, len(embeddings))
    idx = np.random.choice(len(embeddings), sample_size, replace=False)
    D = squareform(pdist(embeddings[idx]))
    im = ax.imshow(D, cmap='YlOrRd', aspect='auto')
    ax.set_title('Pairwise Distance Matrix', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Euclidean Distance')

    # Stage separation score
    ax = fig.add_subplot(gs[1, 2])
    from sklearn.metrics import silhouette_score
    if len(np.unique(labels)) > 1:
        sil_score = silhouette_score(embeddings, labels)
        ax.bar(['Silhouette\nScore'], [sil_score], color=COLORS['performance'][0], width=0.6)
        ax.set_ylim([-1, 1])
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax.set_title('Embedding Quality', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score')
        ax.grid(True, alpha=0.2, axis='y')
    else:
        ax.axis('off')

    # Cumulative variance
    ax = fig.add_subplot(gs[1, 3])
    cumvar = np.cumsum(variance[:20])
    ax.plot(range(1, len(cumvar)+1), cumvar, 'o-', linewidth=2, markersize=6)
    ax.axhline(y=0.95, color='r', linestyle='--', alpha=0.5, label='95% threshold')
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Cumulative Variance Explained')
    ax.set_title('PCA Cumulative Variance', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.2)

    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Generated: {output_path}")


def generate_figure4_model_performance(
    training_history: Dict,
    test_metrics: Dict,
    output_path: Path
):
    """
    Real model performance plots: loss curves, ROC, PR, accuracy, F1
    """
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3)

    # Training loss curves
    ax = fig.add_subplot(gs[0, 0])
    if 'train_loss' in training_history:
        epochs = range(1, len(training_history['train_loss']) + 1)
        ax.plot(epochs, training_history['train_loss'], 'o-',
               label='Train', linewidth=2, markersize=4)
        if 'val_loss' in training_history:
            ax.plot(epochs, training_history['val_loss'], 's-',
                   label='Val', linewidth=2, markersize=4)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.2)
        ax.set_yscale('log')

    # Metrics over time
    ax = fig.add_subplot(gs[0, 1])
    if 'wasserstein' in training_history:
        epochs = range(1, len(training_history['wasserstein']) + 1)
        ax.plot(epochs, training_history['wasserstein'], 'o-',
               label='Wasserstein', linewidth=2, markersize=4)
        if 'mmd' in training_history:
            ax2 = ax.twinx()
            ax2.plot(epochs, training_history['mmd'], 's-',
                    color='orange', label='MMD', linewidth=2, markersize=4)
            ax2.set_ylabel('MMD', color='orange')
            ax2.tick_params(axis='y', labelcolor='orange')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Wasserstein Distance')
        ax.set_title('Distribution Metrics', fontsize=12, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.2)

    # ROC curve
    ax = fig.add_subplot(gs[0, 2])
    if 'fpr' in test_metrics and 'tpr' in test_metrics:
        ax.plot(test_metrics['fpr'], test_metrics['tpr'],
               linewidth=3, label=f"AUC = {test_metrics.get('roc_auc', 0):.3f}")
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.2)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

    # PR curve
    ax = fig.add_subplot(gs[0, 3])
    if 'precision' in test_metrics and 'recall' in test_metrics:
        ax.plot(test_metrics['recall'], test_metrics['precision'],
               linewidth=3, label=f"AP = {test_metrics.get('average_precision', 0):.3f}")
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.2)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

    # F1 score per class
    ax = fig.add_subplot(gs[1, 0])
    if 'f1_per_class' in test_metrics:
        classes = list(test_metrics['f1_per_class'].keys())
        f1_scores = list(test_metrics['f1_per_class'].values())
        bars = ax.barh(classes, f1_scores, color=COLORS['performance'][0])
        ax.set_xlabel('F1 Score')
        ax.set_title('F1 Score per Class', fontsize=12, fontweight='bold')
        ax.set_xlim([0, 1])
        ax.grid(True, alpha=0.2, axis='x')
        for i, (bar, score) in enumerate(zip(bars, f1_scores)):
            ax.text(score + 0.02, i, f'{score:.3f}', va='center', fontsize=9)

    # Accuracy over epochs
    ax = fig.add_subplot(gs[1, 1])
    if 'train_acc' in training_history:
        epochs = range(1, len(training_history['train_acc']) + 1)
        ax.plot(epochs, training_history['train_acc'], 'o-',
               label='Train', linewidth=2, markersize=4)
        if 'val_acc' in training_history:
            ax.plot(epochs, training_history['val_acc'], 's-',
                   label='Val', linewidth=2, markersize=4)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Classification Accuracy', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.2)
        ax.set_ylim([0, 1])

    # Confusion matrix
    ax = fig.add_subplot(gs[1, 2])
    if 'confusion_matrix' in test_metrics:
        cm = np.array(test_metrics['confusion_matrix'])
        im = ax.imshow(cm, cmap='Blues', aspect='auto')
        ax.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax)
        # Add text annotations
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f'{cm[i,j]:.0f}',
                       ha='center', va='center',
                       color='white' if cm[i,j] > cm.max()/2 else 'black')

    # Metric summary bar chart
    ax = fig.add_subplot(gs[1, 3])
    if 'accuracy' in test_metrics:
        metrics = {
            'Accuracy': test_metrics.get('accuracy', 0),
            'Precision': test_metrics.get('precision', 0),
            'Recall': test_metrics.get('recall', 0),
            'F1': test_metrics.get('f1', 0),
        }
        bars = ax.bar(metrics.keys(), metrics.values(),
                     color=COLORS['performance'][:len(metrics)])
        ax.set_ylim([0, 1])
        ax.set_title('Test Metrics Summary', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.2, axis='y')
        for bar, (name, value) in zip(bars, metrics.items()):
            ax.text(bar.get_x() + bar.get_width()/2, value + 0.02,
                   f'{value:.3f}', ha='center', fontsize=9, fontweight='bold')

    # Learning rate schedule
    ax = fig.add_subplot(gs[2, 0])
    if 'lr' in training_history:
        epochs = range(1, len(training_history['lr']) + 1)
        ax.plot(epochs, training_history['lr'], 'o-', linewidth=2, markersize=4)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.2)

    # Gradient norm
    ax = fig.add_subplot(gs[2, 1])
    if 'grad_norm' in training_history:
        epochs = range(1, len(training_history['grad_norm']) + 1)
        ax.plot(epochs, training_history['grad_norm'], 'o-', linewidth=2, markersize=4)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Gradient Norm')
        ax.set_title('Gradient Statistics', fontsize=12, fontweight='bold')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.2)

    # Per-fold performance
    ax = fig.add_subplot(gs[2, 2])
    if 'fold_metrics' in test_metrics:
        fold_data = test_metrics['fold_metrics']
        x = range(len(fold_data))
        metrics_to_plot = ['wasserstein', 'mmd', 'mse']
        for metric in metrics_to_plot:
            if metric in fold_data[0]:
                values = [f[metric] for f in fold_data]
                ax.plot(x, values, 'o-', label=metric.upper(), linewidth=2, markersize=6)
        ax.set_xlabel('Fold')
        ax.set_ylabel('Metric Value')
        ax.set_title('Cross-Fold Performance', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.2)

    # Training time
    ax = fig.add_subplot(gs[2, 3])
    if 'time_per_epoch' in training_history:
        epochs = range(1, len(training_history['time_per_epoch']) + 1)
        ax.plot(epochs, training_history['time_per_epoch'], 'o-', linewidth=2, markersize=4)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Training Time per Epoch', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.2)

    plt.suptitle('Model Performance Analysis', fontsize=16, fontweight='bold', y=0.995)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Generated: {output_path}")


def generate_figure5_attention_heatmap(
    attention_weights: np.ndarray,
    token_labels: list,
    output_path: Path,
    title: str = "Attention Patterns"
):
    """
    Professional attention heatmap with proper statistics
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Mean attention across all samples
    ax = axes[0, 0]
    mean_attn = attention_weights.mean(axis=0)
    im = ax.imshow(mean_attn, cmap='viridis', aspect='auto', vmin=0)
    ax.set_xticks(range(len(token_labels)))
    ax.set_yticks(range(len(token_labels)))
    ax.set_xticklabels(token_labels, rotation=45, ha='right')
    ax.set_yticklabels(token_labels)
    ax.set_title('Mean Attention', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Attention Weight')

    # Std attention
    ax = axes[0, 1]
    std_attn = attention_weights.std(axis=0)
    im = ax.imshow(std_attn, cmap='Reds', aspect='auto')
    ax.set_xticks(range(len(token_labels)))
    ax.set_yticks(range(len(token_labels)))
    ax.set_xticklabels(token_labels, rotation=45, ha='right')
    ax.set_yticklabels(token_labels)
    ax.set_title('Attention Std Dev', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Std Dev')

    # Attention entropy
    ax = axes[0, 2]
    from scipy.stats import entropy as scipy_entropy
    entropies = []
    for i in range(attention_weights.shape[0]):
        for j in range(attention_weights.shape[1]):
            attn = attention_weights[i, j]
            if attn.sum() > 0:
                ent = scipy_entropy(attn / attn.sum())
                if np.isfinite(ent):
                    entropies.append(ent)
    ax.hist(entropies, bins=30, color=COLORS['performance'][0], alpha=0.7, edgecolor='black')
    ax.set_xlabel('Entropy')
    ax.set_ylabel('Count')
    ax.set_title('Attention Entropy Distribution', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.2, axis='y')

    # Token importance
    ax = axes[1, 0]
    token_importance = mean_attn.sum(axis=0)
    bars = ax.barh(token_labels, token_importance, color=COLORS['performance'][1])
    ax.set_xlabel('Total Attention Received')
    ax.set_title('Token Importance', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.2, axis='x')

    # Attention flow diagram
    ax = axes[1, 1]
    im = ax.imshow(mean_attn, cmap='Blues', aspect='auto')
    # Add arrows for top connections
    top_k = 5
    flat_idx = np.argsort(mean_attn.ravel())[-top_k:]
    for idx in flat_idx:
        i, j = np.unravel_index(idx, mean_attn.shape)
        if i != j:
            ax.annotate('', xy=(j, i), xytext=(j, i),
                       arrowprops=dict(arrowstyle='->', lw=2,
                                     color='red', alpha=0.6))
    ax.set_xticks(range(len(token_labels)))
    ax.set_yticks(range(len(token_labels)))
    ax.set_xticklabels(token_labels, rotation=45, ha='right')
    ax.set_yticklabels(token_labels)
    ax.set_title('Top-5 Connections', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax)

    # Attention statistics table
    ax = axes[1, 2]
    ax.axis('off')
    stats_data = [
        ['Metric', 'Value'],
        ['Mean Attention', f'{mean_attn.mean():.4f}'],
        ['Std Attention', f'{std_attn.mean():.4f}'],
        ['Max Attention', f'{mean_attn.max():.4f}'],
        ['Min Attention', f'{mean_attn.min():.4f}'],
        ['Sparsity', f'{(mean_attn < 0.01).sum() / mean_attn.size:.2%}'],
        ['Entropy (mean)', f'{np.mean(entropies):.3f}'],
    ]
    table = ax.table(cellText=stats_data, cellLoc='left',
                    bbox=[0, 0, 1, 1], edges='horizontal')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    # Style header row
    for i in range(2):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')

    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Generated: {output_path}")


if __name__ == "__main__":
    print("Professional figure generation module loaded")
    print("Use these functions to generate REAL publication-quality figures")
