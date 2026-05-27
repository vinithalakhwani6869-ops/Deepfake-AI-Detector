"""
evaluation/metrics.py
─────────────────────
Offline classification metrics computed strictly from (y_true, y_pred, y_score).

WHY ACCURACY ALONE IS INSUFFICIENT FOR DEEPFAKE DETECTION
──────────────────────────────────────────────────────────
Deepfake datasets are often heavily imbalanced (many more real than fake frames,
or vice versa). A model that always predicts "Real" can achieve high accuracy
while detecting zero fakes. Accuracy answers "how often am I right?" but not
"how well do I catch fakes without flooding users with false alarms?"

WHY ROC-AUC IS MORE RELIABLE UNDER CLASS IMBALANCE
──────────────────────────────────────────────────
ROC-AUC integrates true-positive rate vs false-positive rate across all
classification thresholds. It measures ranking quality: "do fake samples tend
to receive higher fake-probability than real samples?" regardless of the
operating point. This is why ROC-AUC is standard on DFDC-style benchmarks.

WHY THRESHOLD TUNING IS REQUIRED
────────────────────────────────
The API reports a binary verdict using argmax (equivalent to threshold 0.5 on
calibrated probabilities). Production systems often tune the decision threshold
on a validation split to hit a target false-positive rate (e.g. minimise false
accusations) or maximise F1. Metrics at threshold 0.5 are reported by default;
``find_best_threshold()`` supports data-driven tuning on validation data.

All functions operate on numpy arrays — no placeholder or synthetic values.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

import numpy as np

# Label convention — must match data/dataset.py and core/detector.py
LABEL_REAL: int = 0
LABEL_FAKE: int = 1


@dataclass(frozen=True)
class MetricResult:
    """Container for a full metric report."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    confusion_matrix: list[list[int]]
    threshold: float
    num_samples: int
    num_real: int
    num_fake: int
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int

    def to_dict(self) -> dict:
        return asdict(self)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Compute 2×2 confusion matrix for binary labels.

    Layout:
        [[TN, FP],
         [FN, TP]]

    Rows = true class (0=real, 1=fake), cols = predicted class.
    """
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.int64).ravel()

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred must have the same length, "
            f"got {y_true.shape[0]} vs {y_pred.shape[0]}"
        )

    cm = np.zeros((2, 2), dtype=np.int64)
    for t, p in zip(y_true, y_pred, strict=True):
        if t not in (LABEL_REAL, LABEL_FAKE) or p not in (LABEL_REAL, LABEL_FAKE):
            raise ValueError(
                f"Labels must be {LABEL_REAL} (real) or {LABEL_FAKE} (fake), "
                f"got true={t}, pred={p}"
            )
        cm[int(t), int(p)] += 1
    return cm


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def metrics_from_confusion_matrix(
    cm: np.ndarray,
    *,
    threshold: float = 0.5,
    num_samples: int | None = None,
) -> dict[str, float]:
    """
    Derive scalar metrics from a 2×2 confusion matrix.

    Definitions (fake = positive class):
        precision = TP / (TP + FP)
        recall    = TP / (TP + FN)   (a.k.a. true positive rate / detection rate)
        f1        = harmonic mean of precision and recall
        accuracy  = (TP + TN) / total
    """
    cm = np.asarray(cm, dtype=np.int64)
    if cm.shape != (2, 2):
        raise ValueError(f"Expected 2×2 confusion matrix, got shape {cm.shape}")

    tn, fp = int(cm[LABEL_REAL, LABEL_REAL]), int(cm[LABEL_REAL, LABEL_FAKE])
    fn, tp = int(cm[LABEL_FAKE, LABEL_REAL]), int(cm[LABEL_FAKE, LABEL_FAKE])

    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    accuracy = _safe_divide(tp + tn, tp + tn + fp + fn)

    total = num_samples if num_samples is not None else tp + tn + fp + fn

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "num_samples": total,
        "num_real": tn + fp,
        "num_fake": fn + tp,
        "threshold": threshold,
    }


def roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Compute ROC-AUC using the Mann–Whitney U / rank statistic.

    ``y_score`` should be the predicted probability (or score) for the **fake**
    class (label=1). Returns ``nan`` if only one class is present.
    """
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    y_score = np.asarray(y_score, dtype=np.float64).ravel()

    if y_true.shape != y_score.shape:
        raise ValueError("y_true and y_score must have the same length")

    n_pos = int(np.sum(y_true == LABEL_FAKE))
    n_neg = int(np.sum(y_true == LABEL_REAL))

    if n_pos == 0 or n_neg == 0:
        return float("nan")

    # Rank all scores (average ranks for ties)
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=np.float64)

    sorted_scores = y_score[order]
    sorted_labels = y_true[order]

    # Handle ties: assign mean rank within each tied group
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            mean_rank = ranks[order[i : j + 1]].mean()
            ranks[order[i : j + 1]] = mean_rank
        i = j + 1

    rank_sum_pos = ranks[y_true == LABEL_FAKE].sum()
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    num_thresholds: int = 101,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ROC curve (FPR, TPR, thresholds).

    Returns:
        fpr, tpr, thresholds — arrays suitable for plotting.
    """
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    y_score = np.asarray(y_score, dtype=np.float64).ravel()

    thresholds = np.linspace(0.0, 1.0, num_thresholds)
    tpr_list: list[float] = []
    fpr_list: list[float] = []

    n_pos = max(int(np.sum(y_true == LABEL_FAKE)), 1)
    n_neg = max(int(np.sum(y_true == LABEL_REAL)), 1)

    for thr in thresholds:
        y_pred = (y_score >= thr).astype(np.int64)
        cm = confusion_matrix(y_true, y_pred)
        tp = int(cm[LABEL_FAKE, LABEL_FAKE])
        fn = int(cm[LABEL_FAKE, LABEL_REAL])
        fp = int(cm[LABEL_REAL, LABEL_FAKE])
        tn = int(cm[LABEL_REAL, LABEL_REAL])
        tpr_list.append(tp / n_pos)
        fpr_list.append(fp / n_neg)

    return (
        np.asarray(fpr_list, dtype=np.float64),
        np.asarray(tpr_list, dtype=np.float64),
        thresholds,
    )


def predictions_at_threshold(y_score: np.ndarray, threshold: float) -> np.ndarray:
    """Convert fake-class probabilities to binary predictions."""
    y_score = np.asarray(y_score, dtype=np.float64).ravel()
    return (y_score >= threshold).astype(np.int64)


def find_best_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    metric: Literal["f1", "youden"] = "f1",
    num_steps: int = 101,
) -> tuple[float, float]:
    """
    Search thresholds on validation data.

    Args:
        metric: ``f1`` maximises F1; ``youden`` maximises TPR − FPR (Youden's J).

    Returns:
        (best_threshold, best_metric_value)
    """
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    y_score = np.asarray(y_score, dtype=np.float64).ravel()

    thresholds = np.linspace(0.0, 1.0, num_steps)
    best_thr = 0.5
    best_val = -1.0

    for thr in thresholds:
        y_pred = predictions_at_threshold(y_score, float(thr))
        cm = confusion_matrix(y_true, y_pred)
        scalars = metrics_from_confusion_matrix(cm, threshold=float(thr))

        if metric == "f1":
            value = scalars["f1"]
        else:
            tp = scalars["true_positives"]
            fn = scalars["false_negatives"]
            fp = scalars["false_positives"]
            tn = scalars["true_negatives"]
            tpr = _safe_divide(tp, tp + fn)
            fpr = _safe_divide(fp, fp + tn)
            value = tpr - fpr

        if value > best_val:
            best_val = value
            best_thr = float(thr)

    return best_thr, best_val


def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    threshold: float = 0.5,
) -> MetricResult:
    """
    Compute the full metric report from ground-truth labels and fake-class scores.

    Args:
        y_true:    Integer labels (0=real, 1=fake).
        y_score:   Predicted P(fake) in [0, 1] for each sample.
        threshold: Decision threshold applied to y_score for hard predictions.
    """
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    y_score = np.asarray(y_score, dtype=np.float64).ravel()

    if len(y_true) == 0:
        raise ValueError("Cannot compute metrics on an empty evaluation set")

    y_pred = predictions_at_threshold(y_score, threshold)
    cm = confusion_matrix(y_true, y_pred)
    scalars = metrics_from_confusion_matrix(cm, threshold=threshold, num_samples=len(y_true))
    auc = roc_auc_score(y_true, y_score)

    return MetricResult(
        accuracy=scalars["accuracy"],
        precision=scalars["precision"],
        recall=scalars["recall"],
        f1=scalars["f1"],
        roc_auc=auc,
        confusion_matrix=cm.tolist(),
        threshold=threshold,
        num_samples=scalars["num_samples"],
        num_real=scalars["num_real"],
        num_fake=scalars["num_fake"],
        true_positives=scalars["true_positives"],
        true_negatives=scalars["true_negatives"],
        false_positives=scalars["false_positives"],
        false_negatives=scalars["false_negatives"],
    )


__all__ = [
    "MetricResult",
    "LABEL_REAL",
    "LABEL_FAKE",
    "confusion_matrix",
    "metrics_from_confusion_matrix",
    "roc_auc_score",
    "roc_curve",
    "predictions_at_threshold",
    "find_best_threshold",
    "compute_metrics",
]
