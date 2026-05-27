"""
evaluation/visualiser.py
────────────────────────
Offline evaluation plots only — no training curves.

Generates:
  • ROC curve
  • Confusion matrix heatmap
  • Fake-score distribution by class
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from evaluation.metrics import LABEL_FAKE, LABEL_REAL, confusion_matrix, roc_curve

logger = logging.getLogger(__name__)


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for evaluation plots. "
            "Install with: pip install matplotlib"
        ) from exc


def plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    title: str = "ROC Curve",
    output_path: Path | None = None,
    dpi: int = 150,
) -> Path | None:
    """
    Plot ROC curve (TPR vs FPR) and optionally save to disk.

    Under class imbalance, inspect this curve rather than accuracy alone:
    a model can have high accuracy but poor TPR at low FPR operating points.
    """
    plt = _require_matplotlib()

    fpr, tpr, _ = roc_curve(y_true, y_score)
    from evaluation.metrics import roc_auc_score

    auc = roc_auc_score(y_true, y_score)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color="#7B61FF", linewidth=2, label=f"ROC (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#888888", linewidth=1, label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Fake detection rate)")
    ax.set_title(title)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi)
        logger.info("[visualiser] Saved ROC curve → %s", output_path)
        plt.close(fig)
        return output_path

    plt.close(fig)
    return None


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    title: str = "Confusion Matrix",
    output_path: Path | None = None,
    dpi: int = 150,
) -> Path | None:
    """Plot confusion matrix heatmap (rows=true, cols=predicted)."""
    plt = _require_matplotlib()

    cm = confusion_matrix(y_true, y_pred)
    class_names = ["Real", "Fake"]

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title(title)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(int(cm[i, j]), "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi)
        logger.info("[visualiser] Saved confusion matrix → %s", output_path)
        plt.close(fig)
        return output_path

    plt.close(fig)
    return None


def plot_score_distribution(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    title: str = "Predicted P(Fake) Distribution",
    output_path: Path | None = None,
    dpi: int = 150,
) -> Path | None:
    """
    Histogram of predicted fake probabilities, separated by true class.

    Threshold tuning: choose an operating point where fake/real histograms
    separate well for your target false-positive rate.
    """
    plt = _require_matplotlib()

    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    y_score = np.asarray(y_score, dtype=np.float64).ravel()

    scores_real = y_score[y_true == LABEL_REAL]
    scores_fake = y_score[y_true == LABEL_FAKE]

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0.0, 1.0, 41)

    if len(scores_real) > 0:
        ax.hist(
            scores_real,
            bins=bins,
            alpha=0.6,
            label=f"Real (n={len(scores_real)})",
            color="#00E6FF",
            edgecolor="white",
        )
    if len(scores_fake) > 0:
        ax.hist(
            scores_fake,
            bins=bins,
            alpha=0.6,
            label=f"Fake (n={len(scores_fake)})",
            color="#7B61FF",
            edgecolor="white",
        )

    ax.axvline(0.5, color="#FF4444", linestyle="--", linewidth=1, label="Threshold 0.5")
    ax.set_xlabel("Predicted P(Fake)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi)
        logger.info("[visualiser] Saved score distribution → %s", output_path)
        plt.close(fig)
        return output_path

    plt.close(fig)
    return None


def save_evaluation_plots(
    y_true: np.ndarray,
    y_score: np.ndarray,
    y_pred: np.ndarray,
    output_dir: Path,
    *,
    split: str = "eval",
) -> dict[str, Path]:
    """
    Save all evaluation plots for one split into ``output_dir``.

    Returns:
        Mapping plot name → saved file path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved: dict[str, Path] = {}

    roc_path = plot_roc_curve(
        y_true,
        y_score,
        title=f"ROC — {split}",
        output_path=output_dir / f"{split}_roc_curve.png",
    )
    if roc_path:
        saved["roc_curve"] = roc_path

    cm_path = plot_confusion_matrix(
        y_true,
        y_pred,
        title=f"Confusion Matrix — {split}",
        output_path=output_dir / f"{split}_confusion_matrix.png",
    )
    if cm_path:
        saved["confusion_matrix"] = cm_path

    dist_path = plot_score_distribution(
        y_true,
        y_score,
        title=f"P(Fake) Distribution — {split}",
        output_path=output_dir / f"{split}_score_distribution.png",
    )
    if dist_path:
        saved["score_distribution"] = dist_path

    return saved


__all__ = [
    "plot_roc_curve",
    "plot_confusion_matrix",
    "plot_score_distribution",
    "save_evaluation_plots",
]
