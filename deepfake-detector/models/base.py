"""
Abstract detector interface — contract for all backbone implementations.

Future models (Xception, ViT) should subclass BaseDetector and register
with core.model_registry.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch.nn as nn


class BaseDetector(ABC):
    """Interface contract for binary deepfake classifiers."""

    @abstractmethod
    def build(self, num_classes: int = 2) -> nn.Module:
        """Return an untrained model with the correct classification head."""
        ...
