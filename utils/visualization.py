"""
Plotting and visualization helpers for Machine Learning workflows.
"""

from pathlib import Path
from typing import Optional, List
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(
    y_true,
    y_pred,
    labels: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
) -> None:
    """Plot and optionally save a styled confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels if labels else "auto",
        yticklabels=labels if labels else "auto",
    )
    plt.title(title, fontsize=14, pad=12)
    plt.ylabel("Actual Label", fontsize=11)
    plt.xlabel("Predicted Label", fontsize=11)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"[Visual Saved] -> {save_path}")
    plt.close()


def plot_learning_curve(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    title: str = "Training & Validation Loss",
    save_path: Optional[str] = None,
) -> None:
    """Plot training and validation loss curves over epochs."""
    plt.figure(figsize=(7, 4.5))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label="Training Loss", color="#1f77b4", linewidth=2)

    if val_losses:
        plt.plot(epochs, val_losses, label="Validation Loss", color="#ff7f0e", linewidth=2, linestyle="--")

    plt.title(title, fontsize=13, pad=10)
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Loss", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"[Visual Saved] -> {save_path}")
    plt.close()
