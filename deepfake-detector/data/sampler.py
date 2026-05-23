"""
data/sampler.py
───────────────
Class-balanced sampling for imbalanced deepfake detection datasets.

The problem
───────────
Real-world deepfake datasets are significantly imbalanced:
  • FaceForensics++ (c23): ~50/50 but only 1 000 videos each — moderate scale
  • DFDC:  ~100k fake : ~20k real — 5:1 imbalance
  • Wild-scraped data: often 10:1 or worse

An unbalanced DataLoader leads to:
  1. The model sees mostly fake examples per epoch, biasing it toward predicting Fake.
  2. Loss is dominated by the majority class, slowing learning on the minority class.
  3. The final classifier threshold shifts, degrading recall on the minority class.

The solution
────────────
WeightedRandomSampler assigns each sample a weight inversely proportional to
its class frequency. After one full pass (epoch), each class has been sampled
approximately equally, regardless of its natural frequency in the dataset.

This is mathematically equivalent to oversampling the minority class or
undersampling the majority class, but without discarding data.

Two sampling strategies
───────────────────────
BALANCED (default):
  Each class is sampled at equal frequency regardless of dataset size.
  Best when class imbalance is severe (>3:1 ratio).

SQRT (square-root):
  Sample weight is proportional to 1 / sqrt(class_count).
  A softer form of balancing. Better when classes are mildly imbalanced (2:1 to 3:1)
  because it preserves some of the natural distribution while reducing dominance.
"""

from __future__ import annotations

import logging
import math
from enum import Enum
from typing import Iterator, Sequence

import torch
from torch.utils.data import Sampler, WeightedRandomSampler

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY ENUM
# ══════════════════════════════════════════════════════════════════════════════

class SamplingStrategy(str, Enum):
    """
    Controls how sample weights are computed from class frequencies.

    BALANCED:
        weight_i = 1 / count(class_i)
        All classes contribute equally per epoch.
        Use when imbalance ratio > 3:1.

    SQRT:
        weight_i = 1 / sqrt(count(class_i))
        Moderate balancing — reduces majority class dominance
        without completely equalising contributions.
        Use when imbalance ratio is 2:1 to 3:1.
    """
    BALANCED = "balanced"
    SQRT     = "sqrt"


# ══════════════════════════════════════════════════════════════════════════════
# BALANCED SAMPLER
# ══════════════════════════════════════════════════════════════════════════════

class BalancedSampler(Sampler[int]):
    """
    Class-balanced sampler that wraps PyTorch's WeightedRandomSampler.

    Each sample is assigned a weight based on its class frequency so that
    all classes appear at approximately equal rates during training.

    Compatible with torch.utils.data.DataLoader as the `sampler` argument.
    Because this sampler uses replacement, it cannot be used together with
    `shuffle=True` in DataLoader — the DataModule handles this correctly.

    Args:
        labels:     Sequence of integer class labels (0 = Real, 1 = Fake).
                    Obtained from DeepfakeDataset.labels.
        strategy:   SamplingStrategy.BALANCED (default) or SamplingStrategy.SQRT.
        num_samples: Number of samples to draw per epoch.
                     Defaults to len(labels) — one full "synthetic balanced epoch".
                     Set higher to see each sample more times per epoch.
        replacement: Whether to sample with replacement. Must be True for
                     oversampling the minority class. Default: True.
        generator:   Optional torch.Generator for reproducible sampling.

    Example:
        sampler = BalancedSampler(labels=dataset.labels)
        loader  = DataLoader(dataset, batch_size=32, sampler=sampler)
    """

    def __init__(
        self,
        labels: Sequence[int],
        strategy: SamplingStrategy = SamplingStrategy.BALANCED,
        num_samples: Optional[int] = None,
        replacement: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        if len(labels) == 0:
            raise ValueError("labels must not be empty.")

        self.labels      = list(labels)
        self.strategy    = strategy
        self.replacement = replacement
        self.generator   = generator

        # Compute per-class weights
        self._class_counts = self._count_classes(self.labels)
        self._class_weights = self._compute_class_weights(
            self._class_counts, strategy
        )

        # Assign weight to every sample based on its class
        self._sample_weights: torch.Tensor = self._assign_sample_weights(
            self.labels, self._class_weights
        )

        # Number of samples to draw per epoch
        self._num_samples = num_samples if num_samples is not None else len(labels)

        # Build the underlying PyTorch sampler
        self._sampler = WeightedRandomSampler(
            weights=self._sample_weights,
            num_samples=self._num_samples,
            replacement=self.replacement,
            generator=self.generator,
        )

        # Log balancing details
        self._log_balancing_info()

    # ── Weight computation ────────────────────────────────────────────────────

    @staticmethod
    def _count_classes(labels: list[int]) -> dict[int, int]:
        """Return a dict mapping class index → sample count."""
        counts: dict[int, int] = {}
        for label in labels:
            counts[label] = counts.get(label, 0) + 1
        return counts

    @staticmethod
    def _compute_class_weights(
        class_counts: dict[int, int],
        strategy: SamplingStrategy,
    ) -> dict[int, float]:
        """
        Compute the weight for each class based on strategy.

        BALANCED: weight = 1 / count
        SQRT:     weight = 1 / sqrt(count)

        Returns a dict mapping class index → weight.
        """
        weights: dict[int, float] = {}
        for class_idx, count in class_counts.items():
            if count == 0:
                raise ValueError(
                    f"Class {class_idx} has 0 samples. "
                    f"Cannot compute weight for empty class."
                )
            if strategy == SamplingStrategy.BALANCED:
                weights[class_idx] = 1.0 / count
            elif strategy == SamplingStrategy.SQRT:
                weights[class_idx] = 1.0 / math.sqrt(count)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
        return weights

    @staticmethod
    def _assign_sample_weights(
        labels: list[int],
        class_weights: dict[int, float],
    ) -> torch.Tensor:
        """
        Build a 1D float tensor of per-sample weights.

        PyTorch's WeightedRandomSampler expects a weight for every sample,
        not just every class. We map each sample to its class weight here.
        """
        # Verify all label values have a corresponding weight
        for label in set(labels):
            if label not in class_weights:
                raise ValueError(
                    f"Label {label} found in dataset but has no class weight. "
                    f"Available classes: {list(class_weights.keys())}"
                )

        return torch.tensor(
            [class_weights[label] for label in labels],
            dtype=torch.double,
        )

    # ── Logging ───────────────────────────────────────────────────────────────

    def _log_balancing_info(self) -> None:
        """Log class distribution before and after weighting."""
        total = len(self.labels)
        logger.info(
            "[sampler] Strategy: %s  |  num_samples/epoch: %d",
            self.strategy.value,
            self._num_samples,
        )
        for class_idx, count in sorted(self._class_counts.items()):
            natural_pct  = 100.0 * count / total
            weight       = self._class_weights[class_idx]
            expected_pct = 100.0 * weight / sum(self._class_weights.values())
            logger.info(
                "[sampler]   class=%d  count=%d (%.1f%%)  weight=%.6f  "
                "expected_sample_pct=%.1f%%",
                class_idx, count, natural_pct, weight, expected_pct,
            )

    # ── Sampler interface ─────────────────────────────────────────────────────

    def __iter__(self) -> Iterator[int]:
        """Delegate iteration to the underlying WeightedRandomSampler."""
        return iter(self._sampler)

    def __len__(self) -> int:
        return self._num_samples

    # ── Inspection ────────────────────────────────────────────────────────────

    @property
    def class_counts(self) -> dict[int, int]:
        """Return the raw class counts from the original labels."""
        return dict(self._class_counts)

    @property
    def class_weights(self) -> dict[int, float]:
        """Return the computed weight per class."""
        return dict(self._class_weights)

    @property
    def sample_weights(self) -> torch.Tensor:
        """Return the per-sample weight tensor (read-only view)."""
        return self._sample_weights.clone()

    @property
    def imbalance_ratio(self) -> float:
        """
        Return max_class_count / min_class_count.
        Values > 1.0 indicate imbalance. Values > 3.0 warrant BALANCED strategy.
        """
        counts = list(self._class_counts.values())
        return max(counts) / max(min(counts), 1)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"n_samples={len(self.labels)}, "
            f"strategy={self.strategy.value!r}, "
            f"imbalance_ratio={self.imbalance_ratio:.2f}, "
            f"num_samples_per_epoch={self._num_samples})"
        )


# ══════════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def build_sampler(
    labels: Sequence[int],
    imbalance_threshold: float = 3.0,
    num_samples: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
) -> BalancedSampler:
    """
    Auto-select sampling strategy based on the imbalance ratio.

    Computes the natural imbalance of the dataset and picks:
      • BALANCED if max/min class ratio > imbalance_threshold
      • SQRT     if max/min class ratio <= imbalance_threshold

    Args:
        labels:               List of integer labels from DeepfakeDataset.labels.
        imbalance_threshold:  Ratio above which BALANCED strategy is used. Default: 3.0.
        num_samples:          Samples per epoch. Defaults to len(labels).
        generator:            Optional torch.Generator for reproducibility.

    Returns:
        A configured BalancedSampler instance.
    """
    from collections import Counter
    counts = Counter(labels)

    if len(counts) == 0:
        raise ValueError("Cannot build sampler from empty labels.")

    count_values = list(counts.values())
    ratio        = max(count_values) / max(min(count_values), 1)

    strategy = (
        SamplingStrategy.BALANCED if ratio > imbalance_threshold
        else SamplingStrategy.SQRT
    )

    logger.info(
        "[sampler] Auto-selected strategy='%s' (imbalance_ratio=%.2f, threshold=%.1f)",
        strategy.value, ratio, imbalance_threshold,
    )

    return BalancedSampler(
        labels=list(labels),
        strategy=strategy,
        num_samples=num_samples,
        generator=generator,
    )


# ── Type alias for Optional ───────────────────────────────────────────────────
# Avoids importing Optional in every file that uses sampler types
from typing import Optional


# ── Public API ─────────────────────────────────────────────────────────────────
__all__ = [
    "BalancedSampler",
    "SamplingStrategy",
    "build_sampler",
]