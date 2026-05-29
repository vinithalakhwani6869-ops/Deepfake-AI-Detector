"""
Logit / probability tensor → human-readable label + confidence percentage.

Extracted from the original DeepfakeDetector.predict() logic.
Index 0 = Real, index 1 = Fake.

Production enhancements:
  - Explicit confidence clamping (safety vs. numerical issues)
  - Logging for auditing predictions
  - Type hints and docstrings
"""

from __future__ import annotations

import logging
import torch

logger = logging.getLogger(__name__)


def logits_to_prediction(logits: torch.Tensor) -> tuple[str, float]:
    """
    Convert model logits to (label, confidence_percent).
    
    Deterministic softmax-based classification:
      - Whichever class has higher probability wins
      - Confidence clamped to [0.0, 100.0] for safety

    Args:
        logits: Tensor of shape (1, 2) — raw class logits.

    Returns:
        label — "Real" or "Fake"
        confidence — probability of predicted class × 100 (0.0–100.0)
    
    Raises:
        ValueError: If logits shape is incorrect.
    """
    if logits.shape != (1, 2):
        msg = f"Expected logits shape (1, 2), got {logits.shape}"
        logger.error("[postprocess] %s", msg)
        raise ValueError(msg)
    
    # Apply softmax to convert logits to probabilities
    probs = torch.softmax(logits, dim=1)
    real_prob = probs[0, 0].item()
    fake_prob = probs[0, 1].item()
    
    # Determine label
    if fake_prob >= real_prob:
        label = "Fake"
        confidence = fake_prob * 100.0
    else:
        label = "Real"
        confidence = real_prob * 100.0
    
    # Safety clamp (handle numerical issues)
    confidence = max(0.0, min(100.0, confidence))
    
    logger.debug(
        "[postprocess] Probabilities: Real=%.4f Fake=%.4f → Label=%s Confidence=%.2f%%",
        real_prob,
        fake_prob,
        label,
        confidence,
    )
    
    return label, confidence
