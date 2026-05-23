"""
training/losses.py
──────────────────
Loss functions for binary deepfake classification.

WHEN TO USE EACH LOSS — DEEPFAKE DETECTION CONTEXT
────────────────────────────────────────────────────

BCEWithLogitsLoss (Binary Cross-Entropy with Logits)
─────────────────────────────────────────────────────
Use when:
  • Dataset is approximately balanced (imbalance ratio < 3:1)
  • You are starting a new experiment and want a stable baseline
  • Combined with BalancedSampler (sampler handles imbalance, loss stays simple)

How it works:
  Combines sigmoid activation and binary cross-entropy into one numerically
  stable operation. More numerically stable than computing sigmoid then BCELoss
  separately (avoids log(0) underflow in float32).

  loss = -[y * log(σ(x)) + (1-y) * log(1 - σ(x))]

For deepfake detection specifically:
  BCEWithLogitsLoss with pos_weight is effective when real/fake are balanced.
  It penalises all errors equally — appropriate when both false positives
  (flagging real as fake) and false negatives (missing fakes) are equally costly.


FocalLoss
──────────
Use when:
  • Dataset is severely imbalanced (ratio > 5:1) AND you are not using a sampler
  • The model is confident but wrong on hard examples
  • You want to downweight easy examples and focus training on hard ones

How it works:
  Focal Loss modifies BCE by adding a factor (1 - p_t)^γ that reduces the
  loss contribution of well-classified (easy) examples:

  FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

  where p_t is the probability of the correct class.

  • When γ = 0: Focal Loss reduces exactly to BCE (with alpha weighting)
  • When γ = 2 (default): easy examples (p_t > 0.9) contribute ~100× less
    loss than hard examples (p_t ≈ 0.5)
  • α_t: per-class weight, typically set to inverse class frequency

For deepfake detection specifically:
  Hard examples for deepfake detection are images where the forgery is
  subtle — high-quality GAN outputs, well-compressed images, or faces
  photographed at unusual angles. These hard examples should drive training.
  Easy examples (obviously fake low-quality deepfakes or clearly real photos)
  already get low loss — Focal Loss stops them from dominating the gradient.

  Recommended γ = 2.0, α = 0.25 (real class) when real is the minority.


LabelSmoothingBCELoss
──────────────────────
Use when:
  • Labels in the training set may be noisy or uncertain
  • You observe overconfident predictions at inference time (probability
    clustering near 0.0 and 1.0, poor calibration)
  • Dataset contains borderline examples that experts disagree on

How it works:
  Instead of hard labels {0, 1}, smooth to {ε, 1-ε} where ε is the
  smoothing factor (typically 0.05 to 0.15):

  smoothed_label = label * (1 - ε) + ε / num_classes

  This prevents the model from becoming maximally confident and improves
  calibration — the confidence score returned by the API will be more
  reliable as a probability estimate.

For deepfake detection specifically:
  Some deepfakes are genuinely ambiguous — partially processed, low-quality,
  or only partially manipulated (e.g. only hair swapped, face is real).
  For these, hard labels of 0 or 1 are misleading. Label smoothing ε=0.1
  effectively tells the model: "be 90% confident, not 100%".

  Also critical for the API: the confidence score we return to users
  should be a calibrated probability, not just a ranking score.
  Label smoothing significantly improves post-training calibration.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# BCE WITH LOGITS LOSS
# ══════════════════════════════════════════════════════════════════════════════

class BCEFocalAdapter(nn.Module):
    """
    Thin wrapper around nn.BCEWithLogitsLoss.

    Accepts raw logits of shape (batch_size, 2) and one-hot labels,
    matching the output shape of EfficientNetB0Classifier.

    Extracts the "fake" class logit (index 1) and converts integer labels
    to float, which is what BCEWithLogitsLoss expects.

    Args:
        pos_weight: Weight for the positive (Fake) class.
                    Set to (n_real / n_fake) to compensate for imbalance.
                    If None, both classes are weighted equally.
        reduction:  'mean' (default) or 'sum'.
    """

    def __init__(
        self,
        pos_weight: Optional[float] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()

        pw_tensor = torch.tensor([pos_weight]) if pos_weight is not None else None
        self._loss = nn.BCEWithLogitsLoss(
            pos_weight=pw_tensor,
            reduction=reduction,
        )
        logger.info(
            "[losses] BCEWithLogitsLoss  pos_weight=%s  reduction=%s",
            pos_weight, reduction,
        )

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: Raw model output of shape (batch_size, 2).
            labels: Integer labels of shape (batch_size,), values in {0, 1}.

        Returns:
            Scalar loss tensor.
        """
        # Extract fake class logit — BCEWithLogitsLoss expects (batch_size,)
        fake_logits = logits[:, 1]
        float_labels = labels.float()

        # Move pos_weight to same device as logits (needed for multi-GPU)
        if self._loss.pos_weight is not None:
            self._loss.pos_weight = self._loss.pos_weight.to(logits.device)

        return self._loss(fake_logits, float_labels)


# ══════════════════════════════════════════════════════════════════════════════
# FOCAL LOSS
# ══════════════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification.

    Lin et al. "Focal Loss for Dense Object Detection" (RetinaNet, ICCV 2017).
    Adapted here for binary deepfake classification.

    Focal Loss is particularly effective for deepfake detection because:
      1. It down-weights easy examples (obvious fakes / obvious real images)
      2. Concentrates gradient updates on hard examples (high-quality deepfakes,
         ambiguous faces, compressed images where artifacts are subtle)
      3. Handles severe class imbalance through the alpha weighting term

    Mathematical formulation:
      FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

      where:
        p_t = sigmoid(logit)  if label = 1 (Fake)
        p_t = 1 - sigmoid(logit)  if label = 0 (Real)
        γ = focusing parameter (≥ 0)
        α_t = class weight (0 < α < 1)

    Args:
        gamma:      Focusing parameter. 0 reduces to BCE. 2.0 is recommended.
                    Higher values focus more on hard examples.
                    Values > 5 can cause training instability.
        alpha:      Weight for the positive (Fake) class.
                    Set to n_real / (n_real + n_fake) when Fake is the majority.
                    Set to n_fake / (n_real + n_fake) when Real is the majority.
                    None = no class weighting (α = 0.5 effectively).
        reduction:  'mean' (default) or 'sum'.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[float] = 0.25,
        reduction: str = "mean",
    ) -> None:
        super().__init__()

        if gamma < 0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")
        if alpha is not None and not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")

        self.gamma     = gamma
        self.alpha     = alpha
        self.reduction = reduction

        logger.info(
            "[losses] FocalLoss  gamma=%.1f  alpha=%s  reduction=%s",
            gamma, alpha, reduction,
        )

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: Raw model output of shape (batch_size, 2).
            labels: Integer labels of shape (batch_size,), values in {0, 1}.

        Returns:
            Scalar focal loss tensor.
        """
        # Extract fake class logit → probability via sigmoid
        fake_logits  = logits[:, 1]
        float_labels = labels.float()

        # Compute binary cross-entropy without reduction
        # F.binary_cross_entropy_with_logits is numerically stable
        bce_loss = F.binary_cross_entropy_with_logits(
            fake_logits,
            float_labels,
            reduction="none",
        )

        # p_t: probability of the CORRECT class
        # For label=1 (Fake): p_t = sigmoid(logit)
        # For label=0 (Real): p_t = 1 - sigmoid(logit)
        probs = torch.sigmoid(fake_logits)
        p_t   = torch.where(labels == 1, probs, 1.0 - probs)

        # Modulating factor: (1 - p_t)^γ
        # Approaches 0 for easy examples (p_t → 1), weight stays at 1 for hard
        focal_weight = (1.0 - p_t).pow(self.gamma)

        # Alpha weighting: α for positive (Fake), (1-α) for negative (Real)
        if self.alpha is not None:
            alpha_t = torch.where(
                labels == 1,
                torch.full_like(p_t, self.alpha),
                torch.full_like(p_t, 1.0 - self.alpha),
            )
            focal_weight = alpha_t * focal_weight

        focal_loss = focal_weight * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


# ══════════════════════════════════════════════════════════════════════════════
# LABEL SMOOTHING BCE LOSS
# ══════════════════════════════════════════════════════════════════════════════

class LabelSmoothingBCELoss(nn.Module):
    """
    Binary Cross-Entropy with label smoothing.

    Converts hard labels {0, 1} to soft labels {ε/2, 1 - ε/2}:
      label = 0  →  ε / 2
      label = 1  →  1 - ε / 2

    Benefits for deepfake detection:
      1. Calibration: confidence scores reflect true probability rather than
         over-confident class separation from hard label training.
      2. Robustness: prevents overfitting to potentially mislabelled examples
         in large-scale datasets scraped from the web.
      3. Generalisation: soft targets distribute probability mass slightly
         toward the alternative class, acting as a regulariser.

    Args:
        smoothing:  Label smoothing factor ε, typically 0.05–0.15.
                    0.0 = no smoothing (reduces to standard BCE).
                    0.1 is a good default for deepfake datasets.
        reduction:  'mean' (default) or 'sum'.
    """

    def __init__(
        self,
        smoothing: float = 0.1,
        reduction: str = "mean",
    ) -> None:
        super().__init__()

        if not (0.0 <= smoothing < 0.5):
            raise ValueError(
                f"smoothing must be in [0, 0.5), got {smoothing}. "
                f"Values >= 0.5 invert the labels."
            )

        self.smoothing = smoothing
        self.reduction = reduction

        logger.info(
            "[losses] LabelSmoothingBCELoss  smoothing=%.2f  reduction=%s",
            smoothing, reduction,
        )

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: Raw model output of shape (batch_size, 2).
            labels: Integer labels of shape (batch_size,), values in {0, 1}.

        Returns:
            Scalar loss tensor.
        """
        # Extract fake class logit
        fake_logits  = logits[:, 1]
        float_labels = labels.float()

        # Smooth: 0 → ε/2,  1 → 1 - ε/2
        smoothed = float_labels * (1.0 - self.smoothing) + 0.5 * self.smoothing

        loss = F.binary_cross_entropy_with_logits(
            fake_logits,
            smoothed,
            reduction=self.reduction,
        )
        return loss


# ══════════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def build_loss(
    name: str,
    gamma: float = 2.0,
    alpha: Optional[float] = 0.25,
    pos_weight: Optional[float] = None,
    smoothing: float = 0.1,
    reduction: str = "mean",
) -> nn.Module:
    """
    Build and return a loss function by name.

    Args:
        name:        One of "bce", "focal", "label_smoothing".
        gamma:       FocalLoss focusing parameter (ignored for other losses).
        alpha:       FocalLoss class weight (ignored for other losses).
        pos_weight:  BCEWithLogitsLoss positive class weight (ignored for others).
        smoothing:   LabelSmoothingBCELoss smoothing factor (ignored for others).
        reduction:   'mean' or 'sum'.

    Returns:
        An nn.Module implementing the requested loss.

    Raises:
        ValueError: if name is not recognised.
    """
    name = name.lower().strip()

    if name == "bce":
        return BCEFocalAdapter(pos_weight=pos_weight, reduction=reduction)

    elif name == "focal":
        return FocalLoss(gamma=gamma, alpha=alpha, reduction=reduction)

    elif name in ("label_smoothing", "smoothing"):
        return LabelSmoothingBCELoss(smoothing=smoothing, reduction=reduction)

    else:
        raise ValueError(
            f"Unknown loss: {name!r}. "
            f"Available: 'bce', 'focal', 'label_smoothing'."
        )


# ── Public API ─────────────────────────────────────────────────────────────────
__all__ = [
    "BCEFocalAdapter",
    "FocalLoss",
    "LabelSmoothingBCELoss",
    "build_loss",
]