"""
models/efficientnet.py
──────────────────────
EfficientNet binary classifier for deepfake detection.

WHY EFFICIENTNET-B0 INITIALLY
──────────────────────────────
EfficientNet-B0 was chosen as the initial architecture for three concrete reasons:

1. Compound scaling efficiency.
   EfficientNet scales width, depth, and resolution jointly using a fixed
   coefficient. B0 sits at the Pareto frontier: it achieves higher accuracy
   per FLOP than ResNet-50 (comparable accuracy at ~5× fewer parameters).
   For a forensic classification task where the discriminative signals are
   subtle (frequency artifacts, blending seams), a parameter-efficient backbone
   avoids wasting capacity on low-level texture reconstruction.

2. Proven deepfake detection track record.
   B4/B7 variants dominated the DFDC Kaggle leaderboard (2020). B0 inherits
   the same architectural pattern at lower cost — a practical starting point
   before scaling to B4 once the training pipeline is validated.

3. Deployment compatibility.
   B0's 5.3M parameters and 224×224 input fit comfortably in 4GB GPU RAM with
   batch_size=64+. B4 at 19M parameters and 380×380 requires >8GB for the same
   batch size. B0 allows local development on consumer hardware while B4 runs
   on a production GPU server — same codebase, different config.

TRANSFER LEARNING DECISIONS
────────────────────────────
All EfficientNet weights available from torchvision were pretrained on ImageNet-1K.
The backbone (features extractor) learns generic visual representations:
  • Early layers: edges, gradients, textures
  • Middle layers: object parts, frequency patterns
  • Late layers: semantic concepts

For deepfake detection, the frequency pattern detectors in the middle layers
are directly useful — GAN-generated images leave systematic frequency artifacts
that align with what these layers are already sensitive to.

We therefore use a two-phase training strategy:
  Phase 1 (frozen backbone, free head):
    Trains only the new binary classifier head for 1–3 epochs.
    Reason: the random init on the new head would produce large gradients that
    would damage the pretrained backbone weights if everything is unfrozen from
    the start. Warming up the head first stabilises the gradient flow.

  Phase 2 (full fine-tuning, all layers):
    Unfreezes the backbone and trains everything with a lower learning rate.
    Reason: once the head is warm, the backbone can be carefully adapted to
    the deepfake distribution. Use LR ≈ 1e-4 (10× lower than head LR).

OVERFITTING PREVENTION
───────────────────────
Four mechanisms are active simultaneously:

1. Dropout (p=0.2 by default, configurable):
   Randomly zeros activations before the final linear layer.
   Prevents the head from memorising specific backbone feature combinations.

2. Stochastic depth (drop_connect_rate, handled internally by torchvision):
   EfficientNet uses stochastic depth — randomly skips entire MBConv blocks
   during training. Acts as a regulariser at the architecture level.

3. Data augmentation (JPEG simulation, ColorJitter — see data/transforms.py):
   The primary defence against overfitting on clean training data.

4. Weight decay via AdamW (handled in training/optimizers.py):
   L2 regularisation on all weight parameters. Prevents individual weights
   from growing large.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights, EfficientNet_B4_Weights

logger = logging.getLogger(__name__)

# Number of output classes — always 2 for binary deepfake detection
# (index 0 = Real, index 1 = Fake, consistent with dataset.py and detector.py)
NUM_CLASSES: int = 2


# ══════════════════════════════════════════════════════════════════════════════
# GENERIC EFFICIENTNET CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════

class EfficientNetClassifier(nn.Module):
    """
    Generic EfficientNet with a binary classification head for deepfake detection.

    Architecture:
        backbone: EfficientNet feature extractor (pretrained on ImageNet-1K)
        classifier: Sequential(Dropout(p), Linear(in_features → 2))

    Output:
        Raw logits of shape (batch_size, 2).
    """

    def __init__(
        self,
        variant: str,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
        dropout_rate: float = 0.2,
        freeze_backbone: bool = False,
        drop_connect_rate: float = 0.2,
    ) -> None:
        super().__init__()

        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}")

        self.variant = variant.lower()
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.freeze_backbone = freeze_backbone
        self.drop_connect_rate = drop_connect_rate

        # ── Load backbone ─────────────────────────────────────────────────────
        if self.variant == "b0":
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.efficientnet_b0(
                weights=weights,
                stochastic_depth_prob=drop_connect_rate,
            )
        elif self.variant == "b4":
            weights = EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.efficientnet_b4(
                weights=weights,
                stochastic_depth_prob=drop_connect_rate,
            )
        else:
            raise ValueError(f"Unsupported EfficientNet variant: {self.variant}")

        # ── Replace classifier head ───────────────────────────────────────────
        in_features: int = self.backbone.classifier[1].in_features

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(in_features, num_classes),
        )

        logger.info(
            "[efficientnet] Built EfficientNet-%s  pretrained=%s  "
            "num_classes=%d  dropout=%.2f  freeze_backbone=%s  in_features=%d",
            self.variant.upper(), pretrained, num_classes, dropout_rate, freeze_backbone, in_features,
        )

        # ── Apply backbone freeze if requested ───────────────────────────────
        if freeze_backbone:
            self.freeze()

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through backbone + classifier head."""
        return self.backbone(x)

    # ── Freeze / unfreeze ─────────────────────────────────────────────────────

    def freeze(self) -> None:
        """Freeze all backbone (feature extractor) parameters."""
        for param in self.backbone.features.parameters():
            param.requires_grad = False

        frozen_params = sum(p.numel() for p in self.backbone.features.parameters())
        logger.info(
            "[efficientnet] EfficientNet-%s Backbone FROZEN — %d parameters frozen",
            self.variant.upper(), frozen_params,
        )

    def unfreeze(self) -> None:
        """Unfreeze all backbone parameters for full fine-tuning."""
        for param in self.backbone.features.parameters():
            param.requires_grad = True

        unfrozen_params = sum(p.numel() for p in self.backbone.features.parameters())
        logger.info(
            "[efficientnet] EfficientNet-%s Backbone UNFROZEN — %d parameters trainable",
            self.variant.upper(), unfrozen_params,
        )

    def unfreeze_last_n_blocks(self, n: int) -> None:
        """Unfreeze only the last n MBConv blocks of the backbone."""
        blocks = list(self.backbone.features.children())
        total = len(blocks)
        start = max(0, total - n)

        for param in self.backbone.features.parameters():
            param.requires_grad = False

        unfrozen_params = 0
        for block in blocks[start:]:
            for param in block.parameters():
                param.requires_grad = True
                unfrozen_params += param.numel()

        logger.info(
            "[efficientnet] EfficientNet-%s Unfroze last %d / %d blocks — %d parameters trainable",
            self.variant.upper(), n, total, unfrozen_params,
        )

    # ── Inspection ────────────────────────────────────────────────────────────

    def count_parameters(self) -> dict[str, int]:
        """Return counts of total, trainable, and frozen parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
        }

    def __repr__(self) -> str:
        counts = self.count_parameters()
        return (
            f"EfficientNet{self.variant.upper()}Classifier("
            f"dropout={self.dropout_rate}, "
            f"params_total={counts['total']:,}, "
            f"params_trainable={counts['trainable']:,})"
        )


# ══════════════════════════════════════════════════════════════════════════════
# CONCRETE CLASSIFIERS
# ══════════════════════════════════════════════════════════════════════════════

class EfficientNetB0Classifier(EfficientNetClassifier):
    """EfficientNet-B0 with a binary classification head for deepfake detection."""

    def __init__(self, **kwargs) -> None:
        super().__init__(variant="b0", **kwargs)


class EfficientNetB4Classifier(EfficientNetClassifier):
    """EfficientNet-B4 with a binary classification head for deepfake detection."""

    def __init__(self, **kwargs) -> None:
        super().__init__(variant="b4", **kwargs)


# ── Public API ─────────────────────────────────────────────────────────────────
__all__ = [
    "EfficientNetB0Classifier",
    "EfficientNetB4Classifier",
    "NUM_CLASSES",
]