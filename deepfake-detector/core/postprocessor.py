"""
Logit / probability tensor → human-readable label + confidence percentage.

Extracted from the original DeepfakeDetector.predict() logic.
Index 0 = Real, index 1 = Fake.
"""

from __future__ import annotations

import torch


def logits_to_prediction(logits: torch.Tensor) -> tuple[str, float]:
    """
    Convert model logits to (label, confidence_percent).

    Args:
        logits: Tensor of shape (1, 2) — raw class logits.

    Returns:
        label — "Real" or "Fake"
        confidence — probability of predicted class × 100 (0.0–100.0)
    """
    probs = torch.softmax(logits, dim=1)
    real_prob = probs[0, 0].item()
    fake_prob = probs[0, 1].item()

    if fake_prob >= real_prob:
        return "Fake", fake_prob * 100.0
    return "Real", real_prob * 100.0
