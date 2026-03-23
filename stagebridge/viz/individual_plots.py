"""
Individual publication-quality plots - NO GRIDS

Each function creates ONE standalone, high-quality plot.
User can assemble them into figures as needed.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")


def plot_pca_with_variance(
    embeddings: np.ndarray, labels: np.ndarray, output_path: Path, dpi: int = 300
):
    """Individual PCA plot with variance explained"""
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            c=[colors[i]],
            label=f"Stage {label}",
            alpha=0.6,
            s=50,
            edgecolors="white",
            linewidth=0.5,
        )

    plt.xlabel(
        f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)", fontsize=12, fontweight="bold"
    )
    plt.ylabel(
        f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)", fontsize=12, fontweight="bold"
    )
    plt.title(
        f"PCA (Total variance: {pca.explained_variance_ratio_[:2].sum() * 100:.1f}%)",
        fontsize=14,
        fontweight="bold",
    )
    plt.legend(frameon=True, loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_tsne(embeddings: np.ndarray, labels: np.ndarray, output_path: Path, dpi: int = 300):
    """Individual t-SNE plot"""
    from sklearn.manifold import TSNE

    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) // 4))
    X_tsne = tsne.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(
            X_tsne[mask, 0],
            X_tsne[mask, 1],
            c=[colors[i]],
            label=f"Stage {label}",
            alpha=0.6,
            s=50,
            edgecolors="white",
            linewidth=0.5,
        )

    plt.xlabel("t-SNE 1", fontsize=12, fontweight="bold")
    plt.ylabel("t-SNE 2", fontsize=12, fontweight="bold")
    plt.title("t-SNE Projection", fontsize=14, fontweight="bold")
    plt.legend(frameon=True, loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_umap(embeddings: np.ndarray, labels: np.ndarray, output_path: Path, dpi: int = 300):
    """Individual UMAP plot"""
    try:
        import umap
    except ImportError:
        print("UMAP not available - pip install umap-learn")
        return

    reducer = umap.UMAP(random_state=42)
    X_umap = reducer.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(
            X_umap[mask, 0],
            X_umap[mask, 1],
            c=[colors[i]],
            label=f"Stage {label}",
            alpha=0.6,
            s=50,
            edgecolors="white",
            linewidth=0.5,
        )

    plt.xlabel("UMAP 1", fontsize=12, fontweight="bold")
    plt.ylabel("UMAP 2", fontsize=12, fontweight="bold")
    plt.title("UMAP Projection", fontsize=14, fontweight="bold")
    plt.legend(frameon=True, loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_phate(embeddings: np.ndarray, labels: np.ndarray, output_path: Path, dpi: int = 300):
    """Individual PHATE plot"""
    try:
        import phate
    except ImportError:
        print("PHATE not available - pip install phate")
        return

    phate_op = phate.PHATE(random_state=42)
    X_phate = phate_op.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(
            X_phate[mask, 0],
            X_phate[mask, 1],
            c=[colors[i]],
            label=f"Stage {label}",
            alpha=0.6,
            s=50,
            edgecolors="white",
            linewidth=0.5,
        )

    plt.xlabel("PHATE 1", fontsize=12, fontweight="bold")
    plt.ylabel("PHATE 2", fontsize=12, fontweight="bold")
    plt.title("PHATE Projection", fontsize=14, fontweight="bold")
    plt.legend(frameon=True, loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_loss_curve(train_loss: list, val_loss: list | None, output_path: Path, dpi: int = 300):
    """Individual loss curve plot"""
    plt.figure(figsize=(10, 6))

    epochs = range(1, len(train_loss) + 1)
    plt.plot(
        epochs, train_loss, "o-", label="Train Loss", linewidth=2, markersize=6, color="#3498db"
    )

    if val_loss is not None:
        plt.plot(
            epochs,
            val_loss,
            "s-",
            label="Validation Loss",
            linewidth=2,
            markersize=6,
            color="#e74c3c",
        )

    plt.xlabel("Epoch", fontsize=12, fontweight="bold")
    plt.ylabel("Loss", fontsize=12, fontweight="bold")
    plt.title("Training Loss Curve", fontsize=14, fontweight="bold")
    plt.legend(frameon=True, loc="best", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_roc_curve(
    fpr: np.ndarray, tpr: np.ndarray, auc_score: float, output_path: Path, dpi: int = 300
):
    """Individual ROC curve plot"""
    plt.figure(figsize=(8, 8))

    plt.plot(fpr, tpr, linewidth=3, label=f"ROC (AUC = {auc_score:.3f})", color="#2ecc71")
    plt.plot([0, 1], [0, 1], "k--", linewidth=2, alpha=0.5, label="Random")

    plt.xlabel("False Positive Rate", fontsize=12, fontweight="bold")
    plt.ylabel("True Positive Rate", fontsize=12, fontweight="bold")
    plt.title(f"ROC Curve (AUC = {auc_score:.3f})", fontsize=14, fontweight="bold")
    plt.legend(frameon=True, loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_pr_curve(
    precision: np.ndarray, recall: np.ndarray, ap_score: float, output_path: Path, dpi: int = 300
):
    """Individual Precision-Recall curve plot"""
    plt.figure(figsize=(8, 8))

    plt.plot(recall, precision, linewidth=3, label=f"PR (AP = {ap_score:.3f})", color="#9b59b6")

    plt.xlabel("Recall", fontsize=12, fontweight="bold")
    plt.ylabel("Precision", fontsize=12, fontweight="bold")
    plt.title(f"Precision-Recall Curve (AP = {ap_score:.3f})", fontsize=14, fontweight="bold")
    plt.legend(frameon=True, loc="lower left", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_accuracy_curve(train_acc: list, val_acc: list | None, output_path: Path, dpi: int = 300):
    """Individual accuracy curve plot"""
    plt.figure(figsize=(10, 6))

    epochs = range(1, len(train_acc) + 1)
    plt.plot(
        epochs, train_acc, "o-", label="Train Accuracy", linewidth=2, markersize=6, color="#3498db"
    )

    if val_acc is not None:
        plt.plot(
            epochs,
            val_acc,
            "s-",
            label="Validation Accuracy",
            linewidth=2,
            markersize=6,
            color="#e74c3c",
        )

    plt.xlabel("Epoch", fontsize=12, fontweight="bold")
    plt.ylabel("Accuracy", fontsize=12, fontweight="bold")
    plt.title("Classification Accuracy", fontsize=14, fontweight="bold")
    plt.legend(frameon=True, loc="best", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_f1_scores(f1_per_class: dict, output_path: Path, dpi: int = 300):
    """Individual F1 scores plot"""
    plt.figure(figsize=(10, 6))

    classes = list(f1_per_class.keys())
    scores = list(f1_per_class.values())

    bars = plt.barh(classes, scores, color="#f39c12", edgecolor="black", linewidth=1.5)
    plt.xlabel("F1 Score", fontsize=12, fontweight="bold")
    plt.title("F1 Score per Class", fontsize=14, fontweight="bold")
    plt.xlim([0, 1])
    plt.grid(True, alpha=0.3, axis="x")

    # Add value labels
    for i, (bar, score) in enumerate(zip(bars, scores)):
        plt.text(score + 0.02, i, f"{score:.3f}", va="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, class_names: list, output_path: Path, dpi: int = 300):
    """Individual confusion matrix plot"""
    plt.figure(figsize=(10, 8))

    im = plt.imshow(cm, cmap="Blues", aspect="auto")
    plt.colorbar(im, label="Count")

    plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    plt.yticks(range(len(class_names)), class_names)
    plt.xlabel("Predicted", fontsize=12, fontweight="bold")
    plt.ylabel("True", fontsize=12, fontweight="bold")
    plt.title("Confusion Matrix", fontsize=14, fontweight="bold")

    # Add text annotations
    threshold = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text_color = "white" if cm[i, j] > threshold else "black"
            plt.text(
                j,
                i,
                f"{cm[i, j]:.0f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=11,
                fontweight="bold",
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_attention_heatmap(
    attention: np.ndarray, token_labels: list, output_path: Path, dpi: int = 300
):
    """Individual attention heatmap"""
    plt.figure(figsize=(10, 9))

    mean_attn = attention.mean(axis=0)
    im = plt.imshow(mean_attn, cmap="viridis", aspect="auto")
    plt.colorbar(im, label="Attention Weight")

    plt.xticks(range(len(token_labels)), token_labels, rotation=45, ha="right")
    plt.yticks(range(len(token_labels)), token_labels)
    plt.xlabel("Key Token", fontsize=12, fontweight="bold")
    plt.ylabel("Query Token", fontsize=12, fontweight="bold")
    plt.title("Mean Attention Pattern", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()


if __name__ == "__main__":
    print("Individual plot generation module loaded")
    print("Each function creates ONE standalone, high-quality plot")
