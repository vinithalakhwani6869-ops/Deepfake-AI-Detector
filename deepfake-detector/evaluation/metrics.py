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

For example, if 95% of your test set is real images, a naive classifier that
always outputs "Real" achieves 95% accuracy but 0% recall on fakes — useless for
production deepfake detection.

WHY ROC-AUC IS MORE RELIABLE UNDER CLASS IMBALANCE
──────────────────────────────────────────────────
ROC-AUC integrates true-positive rate vs false-positive rate across all
classification thresholds. It measures ranking quality: "do fake samples tend
to receive higher fake-probability than real samples?" regardless of the
operating point. This is why ROC-AUC is standard on DFDC-style benchmarks.

ROC-AUC is invariant to class imbalance — it evaluates how well the model
discriminates between classes regardless of their relative frequency.
An imbalanced dataset doesn't hurt ROC-AUC, whereas precision or recall alone
can be misleading.

WHY THRESHOLD TUNING IS REQUIRED
────────────────────────────────
The API reports a binary verdict using argmax (equivalent to threshold 0.5 on
calibrated probabilities). Production systems often tune the decision threshold
on a validation split to hit a target false-positive rate (e.g. minimise false
accusations) or maximise F1. Metrics at threshold 0.5 are reported by default;
``find_best_threshold()`` supports data-driven tuning on validation data.

Threshold tuning is particularly important for deepfake detection because:
  • False positives (accusing real people of being fake) are often more costly
    than false negatives (missing some fakes).
  • A high-confidence threshold on P(fake) reduces false positives at the cost
    of missed detections.
  • Different deployment scenarios (social media moderation vs law enforcement)
    may have different optimal thresholds.

CLASS IMBALANCE CONSIDERATIONS
──────────────────────────────
Deepfake detection datasets are inherently imbalanced:
  • DFDC (Kaggle): ~70% real, ~30% fake
  • FaceForensics++: ~80% real, ~20% fake
  • Many real-world streams: >>90% real, <<10% fake

Under imbalance:
  • Accuracy is misleading. A classifier that always guesses "real" can achieve
    70–80% accuracy on test sets.
  • F1-score becomes critical. It penalises both false positives and false
    negatives, making it robust to imbalance.
  • Recall (sensitivity) is important for safety-critical applications: you must
    catch most fakes even if you flag some real images too.
  • Precision matters for user experience: false accusations hurt trust.

This module provides all metrics needed to navigate these trade-offs.

DEEPFAKE DETECTION METRIC LIMITATIONS
──────────────────────────────────────
No single metric is sufficient for evaluating deepfake detectors:

1. Temporal consistency: Offline metrics operate on single frames, but real
   videos are temporal. A model good at frame-level detection may still fail
   on temporal patterns (e.g. flickering eyes, lipsyncing artifacts).

2. Generalisation across codecs: A model trained on high-quality video may
   fail on H.264/H.265 compressed or low-resolution streams. Metrics computed
   on a single codec don't predict cross-codec performance.

3. Adversarial robustness: Deepfake detection metrics ignore adversarial
   attacks (e.g. adding imperceptible perturbations to fool the detector).
   Production systems need adversarial evaluation.

4. Face parsing: Metrics assume faces are already detected and extracted.
   Face detection failures (e.g. extreme angles, occlusion) are not captured
   in image-level metrics.

5. Demographic bias: Metrics don't reveal if the model is more accurate on
   certain ethnicities, genders, or age groups. This is critical for fairness.

All functions operate on numpy arrays — no placeholder or synthetic values.
All computations are deterministic and reproducible.
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
    """
    Container for a complete metric report from a single evaluation pass.
    
    All fields are immutable (frozen=True) to prevent accidental modification
    of saved results.
    """

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
        """Serialise to a plain dict for JSON export."""
        return asdict(self)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Compute 2×2 confusion matrix for binary labels.

    Layout:
        [[TN, FP],
         [FN, TP]]

    Rows = true class (0=real, 1=fake), cols = predicted class.
    
    Args:
        y_true: True binary labels (0 or 1), shape (n,).
        y_pred: Predicted binary labels (0 or 1), shape (n,).
    
    Returns:
        2×2 confusion matrix as int64 numpy array.
    
    Raises:
        ValueError: if shapes don't match or labels contain invalid values.
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
    """Divide safely, returning 0 if denominator is 0."""
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
        precision = TP / (TP + FP)     — of samples predicted fake, how many are actually fake?
        recall    = TP / (TP + FN)     — of actual fakes, how many did we catch?
        f1        = 2 * (precision * recall) / (precision + recall)  — harmonic mean
        accuracy  = (TP + TN) / total  — overall correctness (misleading under imbalance)
    
    Args:
        cm: 2×2 confusion matrix from confusion_matrix().
        threshold: Decision threshold (used only for metadata; the matrix is already thresholded).
        num_samples: Total number of samples (defaults to sum of confusion matrix).
    
    Returns:
        Dict with keys: accuracy, precision, recall, f1, true_positives, true_negatives,
        false_positives, false_negatives, num_samples, num_real, num_fake, threshold.
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

    Interpretation: ROC-AUC measures the probability that a randomly chosen
    fake sample receives a higher fake-probability than a randomly chosen
    real sample.

    ``y_score`` should be the predicted probability (or score) for the **fake**
    class (label=1). Returns ``nan`` if only one class is present.
    
    Args:
        y_true: True binary labels (0 or 1), shape (n,).
        y_score: Predicted probability of fake class [0, 1], shape (n,).
    
    Returns:
        ROC-AUC score in [0, 1]. Returns nan if only one class is present.
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
    Compute ROC curve (FPR, TPR, thresholds) across all decision thresholds.

    Useful for:
    • Plotting the ROC curve (FPR on x-axis, TPR on y-axis)
    • Finding an optimal threshold for a specific operating point
    • Understanding the trade-off between true positive rate and false positive rate
    
    Args:
        y_true: True binary labels (0 or 1), shape (n,).
        y_score: Predicted probability of fake class [0, 1], shape (n,).
        num_thresholds: Number of threshold points to evaluate (default 101).
    
    Returns:
        Tuple of (fpr, tpr, thresholds) — all numpy arrays of shape (num_thresholds,).
        fpr: False positive rate (1 - specificity) at each threshold.
        tpr: True positive rate (sensitivity/recall) at each threshold.
        thresholds: Decision thresholds tested.
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
    """
    Convert fake-class probabilities to binary predictions at a given threshold.
    
    Args:
        y_score: Predicted probability of fake class [0, 1], shape (n,).
        threshold: Decision threshold; samples with score >= threshold are predicted fake.
    
    Returns:
        Binary predictions (0 or 1), shape (n,).
    """
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
    Search for the optimal decision threshold on validation data.

    Two metrics are supported:
    • ``f1``: Maximises F1-score (harmonic mean of precision and recall).
             Good default for balanced accuracy/precision/recall trade-off.
    • ``youden``: Maximises Youden's J = TPR − FPR (sensitivity − (1 − specificity)).
                 Emphasises balanced sensitivity and specificity.

    Args:
        y_true: True binary labels (0 or 1), shape (n,).
        y_score: Predicted probability of fake class [0, 1], shape (n,).
        metric: Metric to maximise ("f1" or "youden").
        num_steps: Number of thresholds to evaluate (default 101).

    Returns:
        Tuple of (best_threshold, best_metric_value).
        best_threshold: Optimal decision threshold.
        best_metric_value: Metric value at the best threshold.
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
        else:  # youden
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
    Compute the complete metric report from ground-truth labels and fake-class scores.

    This is the primary entry point for offline evaluation. It computes all metrics
    needed for model assessment: accuracy, precision, recall, F1, ROC-AUC, and
    confusion matrix breakdown.

    Args:
        y_true:    Integer labels (0=real, 1=fake), shape (n,).
        y_score:   Predicted P(fake) in [0, 1] for each sample, shape (n,).
        threshold: Decision threshold applied to y_score for hard predictions (default 0.5).

    Returns:
        MetricResult dataclass with all metrics.

    Raises:
        ValueError: if y_true or y_score is empty.
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
